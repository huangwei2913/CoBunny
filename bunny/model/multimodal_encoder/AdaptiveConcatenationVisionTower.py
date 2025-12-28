# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from copy import deepcopy
import random
import math
from .dino_encoder import DinoVisionTower
from  bunny.util.utils import CrossAttentionBlock
from .oryx_vit import OryxViTWrapper
from bunny.util.merge import bipartite_soft_matching_merge,random_bipartite_soft_matching
from PIL import Image
from torchvision import transforms
from typing import Dict

class WeightedPseudoCLSHead(nn.Module):
    """
    用于从 Tokens 序列中生成一个伪 CLS Token 的可学习加权池化头 (O(L) 操作)。
    它通过一个 MLP 预测每个 Token 对全局表示的贡献度。
    """
    def __init__(self, dim, hidden_dim_ratio=2):
        super().__init__()
        
        hidden_dim = int(dim * hidden_dim_ratio) # 例如 1024 * 2 = 2048
        
        self.score_predictor = nn.Sequential(
            # 1. 扩大维度，学习更复杂的特征
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            # 2. 压缩到单个分数
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tokens):
        """
        Args:
            tokens (torch.Tensor): 输入 Tokens 序列 [B, L, D] (例如 [B, 576, 1024])

        Returns:
            torch.Tensor: 伪 CLS Token [B, 1, D]
        """
        
        # 1. 预测每个 Token 的权重 (Softmax Score)
        # scores 形状: [B, L, 1]
        scores = self.score_predictor(tokens)
        
        # 2. 对权重进行 Softmax 归一化
        # 确保所有 Token 的权重和为 1，且是正数
        weights = F.softmax(scores, dim=1) 
        
        # 3. 加权求和得到伪 CLS Token
        # tokens: [B, L, D]
        # weights: [B, L, 1] (广播到 D)
        # B_pseudo_cls: [B, 1, D]
        
        # torch.sum(..., dim=1) 沿着序列维度 L 求和
        B_pseudo_cls = torch.sum(tokens * weights, dim=1, keepdim=True)
        
        return B_pseudo_cls


def linear_to_2d_pooling(tokens: torch.Tensor, layers: int, token_per_layer: int) -> torch.Tensor:
    """
    实现 O(L) 的二维结构化池化（Tokens 减半）。
    这里采用 2x1 步长池化（在宽度上每隔一个取平均），实现 50% 压缩。
    
    tokens: [B, L_group, D] (例如 [B, 512, 1024])
    layers: 组内包含的层数 (例如 2)
    token_per_layer: 每层 Token 数量 (例如 256)
    """
    B, L_group, D = tokens.shape
    
    # 1. 还原到 (B * layers, H, W, D) 结构
    H = W = int(math.sqrt(token_per_layer))
    
    # 检查是否可还原
    if H * W * layers != L_group:
        raise ValueError(f"Token Group Length {L_group} does not match layers*H*W ({layers}*{H}*{W})")

    # [B, L_group, D] -> [B * layers, H * W, D] -> [B * layers, H, W, D]
    grid_tokens = tokens.view(B * layers, H, W, D)

    # 2. 执行 2x1 步长池化（在 W 维度上进行平均，压缩 50%）
    # [B * layers, H, W, D] -> [B * layers, H, W//2, D]
    
    # 分割奇偶列 (W 维度)
    W_pooled = W // 2 * 2 # 确保偶数 W
    tokens_odd = grid_tokens[:, :, :W_pooled:2, :] 
    tokens_even = grid_tokens[:, :, 1:W_pooled:2, :]
    
    # O(L) 线性合并 (求平均)
    pooled_tokens_grid = 0.5 * (tokens_odd + tokens_even)
    
    # 3. 展平回序列结构
    # [B * layers, H, W//2, D] -> [B * layers, H * W//2, D] -> [B, L_merged, D]
    L_merged = H * (W // 2) * layers
    merged_tokens = pooled_tokens_grid.view(B, L_merged, D)
    
    return merged_tokens


# 辅助函数：提取最后一层的 CLS token
def _get_cls_token(aligned_feature_last_layer: torch.Tensor) -> torch.Tensor:
    return aligned_feature_last_layer[:, 0:1, :] # 形状: [B, 1, D]

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# --- 1. 定义符合 train.py 接口要求的 Processor ---
# 必须包含 crop_size, image_mean 等属性，且能处理 PIL 图片
class SingleImageProcessor(object):
    def __init__(self, image_size=384, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.crop_size = {'height': image_size, 'width': image_size} # train.py 需要访问这个属性
        self.image_mean = mean # train.py 需要访问这个属性
        
        # 定义标准的预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def preprocess(self, image: Image.Image, return_tensors='pt') -> Dict[str, torch.Tensor]:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image) 
        if return_tensors == 'pt':
            return {'pixel_values': tensor.unsqueeze(0)} # [1, C, H, W]
        return tensor



import torch
import torch.nn.functional as F
from typing import List
class ImageProcessorMultipleEncoders(object):
    """
    负责在 AdaptiveConcatenationVisionTower 内部处理输入的图像张量。
    它的主要作用是：
    1. 确保输入的张量形状（H x W）能被所有 Vision Tower 的 patch_size 整除。
    2. 统一化处理，为所有 Tower 提供单一的、兼容的输入张量。
    Args:
        patch_size_list: 包含所有 Vision Tower 原始 patch size 的列表 (例如 [14, 16])。
        image_size: 目标统一输入尺寸 (例如 1024)。
    """
    def __init__(self, patch_size_list: List[int], image_size: int = 1024):
        self.patch_size_list = patch_size_list
        self.common_image_size = image_size
        # 记录最大 patch size，用于确定严格的整除性要求
        self.max_patch_size = max(patch_size_list)
        # 在初始化时进行检查，确保公共尺寸是最大 patch size 的倍数
        if self.common_image_size % self.max_patch_size != 0:
             print(f"Warning: Common image size ({self.common_image_size}) should be a multiple of the largest patch size ({self.max_patch_size}) for clean tokenization.")
    def process_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        接收来自 LazySupervisedDataset 的图像 Tensor。
        Args:
            image_tensor: 图像张量，形状通常为 [B, C, H, W] 或 [C, H, W]。
        Returns:
            处理后的图像张量，形状为 [B, C, common_image_size, common_image_size]。
        """
        # 1. 形状统一化：确保 Batch 维度存在
        if image_tensor.dim() == 3:
            # 如果输入是 [C, H, W]，添加 Batch 维度 [1, C, H, W]
            image_tensor = image_tensor.unsqueeze(0)
        B, C, H, W = image_tensor.shape
        # 2. 验证/调整尺寸 (关键步骤)
        # 理论上，外部处理器已将图像调整到 H=W=self.common_image_size
        if H != self.common_image_size or W != self.common_image_size:
            print(f"Internal Processor: Resizing image from {H}x{W} to {self.common_image_size}x{self.common_image_size}")
            # 使用插值进行强制调整，以确保所有 Tower 获得正确的输入尺寸
            image_tensor = F.interpolate(
                image_tensor,
                size=(self.common_image_size, self.common_image_size),
                mode='bilinear',
                align_corners=False
            )
        # 3. 兼容性检查 (可选，但在实际运行中，Vision Tower会自行处理)
        # 在 DINOv3 和 OryxViT 内部，它们会提取 patch tokens。
        # 这里只需要确保张量是连续的 (contiguous) 即可。
        return image_tensor.contiguous()
    

class AdaptiveConcatenationVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        # 1. 基础安全检查
        if not hasattr(args, 'vision_tower_dino') or args.vision_tower_dino is None:
            raise ValueError("❌ 错误：未通过 --vision_tower_dino 指定 DINO 路径！")
        self.is_loaded = False
        self.args = args
        self.global_dimension = getattr(args, "mm_hidden_size", 1024)
        self.compression_K = getattr(args, "compression_K", 4)
        self.num_heads = 8 
        self.mlp_ratio = 4.0
        self.target_image_size = 384
        self.image_processor = SingleImageProcessor(image_size=self.target_image_size)
        # 2. 动态加载子塔
        # DINO 塔 (A)
        self.dino_vision_tower = DinoVisionTower(args.vision_tower_dino, args)
        self.dino_vision_tower.load_model()
        
        # Oryx 塔 (B)
        self.oryx_vision_tower = OryxViTWrapper(args.vision_tower_oryx, args)
        self.oryx_vision_tower.load_model()

        # 3. 维度投影层 (对齐 DINO 768 和 Oryx 1152 到 1024)
        self.mlp_layers = nn.ModuleList([
            nn.Linear(self.dino_vision_tower.hidden_size, self.global_dimension),
            nn.Linear(self.oryx_vision_tower.hidden_size, self.global_dimension)
        ])

        # 6. 最终融合权重
        self.final_cls_weights = nn.Parameter(torch.ones(2))
        self._init_interaction_modules()
        if not delay_load:
            self.load_model()


    def load_model(self):
        if self.is_loaded: return
        self.dino_vision_tower.load_model()
        self.oryx_vision_tower.load_model()
        self.is_loaded = True

    def _init_interaction_modules(self):
        # 将之前的初始化逻辑封装在这里，保持 __init__ 干净
        self.N_layer_A = self.dino_vision_tower.layer_count 
        self.multi_cls_cross_attn_blocks_A = nn.ModuleList([
            CrossAttentionBlock(dim=self.global_dimension, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio) 
            for _ in range(self.N_layer_A)
        ])
        self.dino_cls_attn_weights = nn.Parameter(torch.ones(self.N_layer_A))

        self.N_layer_B = self.oryx_vision_tower.layer_count 
        self.b_pseudo_cls_head = WeightedPseudoCLSHead(dim=self.global_dimension) 
        self.multi_cls_cross_attn_blocks_B = nn.ModuleList([
            CrossAttentionBlock(dim=self.global_dimension, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio)
            for _ in range(self.N_layer_B)
        ])
        self.oryx_cls_attn_weights = nn.Parameter(torch.ones(self.N_layer_B))
        self.final_cls_weights = nn.Parameter(torch.ones(2))


    @property
    def hidden_size(self):
        return self.global_dimension

    @property
    def num_patches(self):
        # 逻辑：A_up(144) + B_up(144) + A_lo(144) + B_lo(144) = 576
        # K=4 时，每个部分是 (2 * 576) // 4 = 288, 分组后 A_up 是 144
        T = self.dino_vision_tower.target_N # 576
        K = self.compression_K
        return ((2 * T) // K + (2 * T) // K) * 2 

    @property
    def device(self):
        return self.mlp_layers[0].weight.device

    @property
    def dtype(self):
        return self.mlp_layers[0].weight.dtype

    @property
    def dummy_feature(self):
        # 1 (CLS) + num_patches
        return torch.zeros(1, 1 + self.num_patches, self.hidden_size, device=self.device, dtype=self.dtype)

    def forward(self, images):
        if images.shape[-1] != self.target_image_size:
            images = F.interpolate(
                images, 
                size=(self.target_image_size, self.target_image_size), 
                mode='bilinear', align_corners=False
            )
        # --- 1. 特征提取 ---
        _, A_inter = self.dino_vision_tower(images) # [B, 4*577, 768]
        _, B_inter = self.oryx_vision_tower(images) # [B, 4*577, 1152]

        # --- 2. 维度投影 ---
        A_proj = self.mlp_layers[0](A_inter) # [B, 4*577, 1024]
        B_proj = self.mlp_layers[1](B_inter) # [B, 4*577, 1024]

        B_batch = A_proj.shape[0]
        T_target = self.dino_vision_tower.target_N # 576

        # --- 3. 结构化拆分 ---
        # A 塔 (DINO)
        A_full = A_proj.view(B_batch, self.N_layer_A, 1 + T_target, -1)
        A_cls = A_full[:, :, 0:1, :]    
        A_patches = A_full[:, :, 1:, :] 

        # B 塔 (Oryx)
        B_full = B_proj.view(B_batch, self.N_layer_B, 1 + T_target, -1)
        B_patches = B_full[:, :, 1:, :] 

        # --- 4. 准备上下文 (Context) ---
        # B 塔压缩作为 A 的背景
        B_patches_flat = B_patches.flatten(0, 1) # [B*4, 576, 1024]
        B_merged_all = bipartite_soft_matching_merge(B_patches_flat, T_target // self.compression_K, B_patches_flat)
        B_kv_context = B_merged_all.view(B_batch, -1, self.global_dimension) # [B, 4*144, 1024]

        # A 塔压缩作为 B 的背景
        A_patches_flat = A_patches.flatten(0, 1)
        A_merged_all = bipartite_soft_matching_merge(A_patches_flat, T_target // self.compression_K, A_patches_flat)
        A_kv_context = A_merged_all.view(B_batch, -1, self.global_dimension) # [B, 4*144, 1024]

        # --- 5. 双向驱动增强 (Symmetric Enhancement) ---
        
        # A Driven by B
        enhanced_cls_A_list = []
        for i in range(self.N_layer_A):
            curr_cls_A = A_cls[:, i] # [B, 1, 1024]
            x_attn_A = torch.cat([curr_cls_A, B_kv_context], dim=1)
            enhanced_cls_A_list.append(self.multi_cls_cross_attn_blocks_A[i](x_attn_A)[:, 0:1, :])
        
        stacked_A = torch.cat(enhanced_cls_A_list, dim=1)
        w_A = F.softmax(self.dino_cls_attn_weights, dim=0).view(1, -1, 1)
        final_cls_A = torch.sum(stacked_A * w_A.to(stacked_A.dtype), dim=1, keepdim=True)

        # B Driven by A
        # 先生成 Oryx 的初始伪 CLS
        B_merged_for_head = B_merged_all.view(B_batch * self.N_layer_B, -1, self.global_dimension)
        pseudo_cls_B_all = self.b_pseudo_cls_head(B_merged_for_head).view(B_batch, self.N_layer_B, 1, -1)
        
        enhanced_cls_B_list = []
        for i in range(self.N_layer_B):
            curr_cls_B = pseudo_cls_B_all[:, i] # [B, 1, 1024]
            x_attn_B = torch.cat([curr_cls_B, A_kv_context], dim=1)
            enhanced_cls_B_list.append(self.multi_cls_cross_attn_blocks_B[i](x_attn_B)[:, 0:1, :])
            
        stacked_B = torch.cat(enhanced_cls_B_list, dim=1)
        w_B = F.softmax(self.oryx_cls_attn_weights, dim=0).view(1, -1, 1)
        final_cls_B = torch.sum(stacked_B * w_B.to(stacked_B.dtype), dim=1, keepdim=True)

        # --- 6. 最终全局 Token 融合 ---
        stacked_cls = torch.cat([final_cls_A, final_cls_B], dim=1)
        w_final = F.softmax(self.final_cls_weights, dim=0).view(1, 2, 1)
        enhanced_cls_token = torch.sum(stacked_cls * w_final.to(stacked_cls.dtype), dim=1, keepdim=True)

        # --- 7. Patch 分组压缩与拼接 (Upper/Lower 分组) ---
        # 按照 N_layer // 2 分成上下两组特征
        half_A = self.N_layer_A // 2
        A_up, A_lo = A_patches[:, :half_A].flatten(1, 2), A_patches[:, half_A:].flatten(1, 2)
        
        half_B = self.N_layer_B // 2
        B_up, B_lo = B_patches[:, :half_B].flatten(1, 2), B_patches[:, half_B:].flatten(1, 2)

        K = self.compression_K
        r_A = A_up.shape[1] - (A_up.shape[1] // K)
        r_B = B_up.shape[1] - (B_up.shape[1] // K)

        # 随机二分匹配合并
        mA_up, _ = random_bipartite_soft_matching(A_up, r=r_A)
        mA_lo, _ = random_bipartite_soft_matching(A_lo, r=r_A)
        mB_up, _ = random_bipartite_soft_matching(B_up, r=r_B)
        mB_lo, _ = random_bipartite_soft_matching(B_lo, r=r_B)
        final_embeddings = torch.cat([
                                    enhanced_cls_token, 
                                    mA_up(A_up), mB_up(B_up), 
                                    mA_lo(A_lo), mB_lo(B_lo)
                                ], dim=1)
        # 最终拼接：[1 (Global) + 144 (A_up) + 144 (B_up) + 144 (A_lo) + 144 (B_lo)] = 577 tokens
        return final_embeddings, None