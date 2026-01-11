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
        self.is_loaded = False
        self.args = args
        self.global_dimension = getattr(args, "mm_hidden_size", 1024)
        
        # [核心控制点] 
        # K=4 (默认) -> 577 tokens
        # K=2 -> 1153 tokens
        self.compression_K = getattr(args, "compression_K", 8)
        self.num_heads = 8 
        self.mlp_ratio = 4.0
        self.target_image_size = 384
        self.image_processor = SingleImageProcessor(image_size=self.target_image_size)
        # 1. 加载子塔
        if not hasattr(args, 'vision_tower_dino') or args.vision_tower_dino is None:
             raise ValueError("Please provide --vision_tower_dino")
             
        self.dino_vision_tower = DinoVisionTower(args.vision_tower_dino, args)
        self.oryx_vision_tower = OryxViTWrapper(args.vision_tower_oryx, args)

        # 2. 维度投影层 (DINO:768->1024, Oryx:1152->1024)
        self.mlp_layers = nn.ModuleList([
            nn.Linear(self.dino_vision_tower.hidden_size, self.global_dimension),
            nn.Linear(self.oryx_vision_tower.hidden_size, self.global_dimension)
        ])

        # 3. 初始化交互模块
        self._init_interaction_modules()
        
        if not delay_load:
            self.load_model()

    def _init_interaction_modules(self):
        # --- A 塔 (DINO) 组件 ---
        self.N_layer_A = self.dino_vision_tower.layer_count 
        self.multi_cls_cross_attn_blocks_A = nn.ModuleList([
            CrossAttentionBlock(dim=self.global_dimension, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio) 
            for _ in range(self.N_layer_A)
        ])
        self.dino_cls_attn_weights = nn.Parameter(torch.ones(self.N_layer_A))

        # --- B 塔 (Oryx) 组件 ---
        self.N_layer_B = self.oryx_vision_tower.layer_count 
        self.b_pseudo_cls_head = WeightedPseudoCLSHead(dim=self.global_dimension) 
        self.multi_cls_cross_attn_blocks_B = nn.ModuleList([
            CrossAttentionBlock(dim=self.global_dimension, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio)
            for _ in range(self.N_layer_B)
        ])
        self.oryx_cls_attn_weights = nn.Parameter(torch.ones(self.N_layer_B))
        
        # --- 最终融合 ---
        self.final_cls_weights = nn.Parameter(torch.ones(2))

    def load_model(self):
        if self.is_loaded: return
        self.dino_vision_tower.load_model()
        self.oryx_vision_tower.load_model()
        self.is_loaded = True

    @property
    def hidden_size(self):
        return self.global_dimension

    @property
    def num_patches(self):
        """
        动态计算 Projector 需要的输入 Token 数量。
        公式：(标准Grid数 / K) * 4个分组
        """
        T_standard = 576 # 24x24
        patches_per_group = T_standard // self.compression_K
        return patches_per_group * 4

    @property
    def device(self):
        return self.mlp_layers[0].weight.device
    
    @property
    def dtype(self):
        return self.mlp_layers[0].weight.dtype

    def forward(self, images):
        # ------------------------ 0. 预处理 ------------------------
        if images.shape[-1] != self.target_image_size:
            images = F.interpolate(
                images, 
                size=(self.target_image_size, self.target_image_size), 
                mode='bilinear', align_corners=False
            )

        # ------------------------ 1. 特征提取与投影 ------------------------
        # A_inter: [B, N_layer_A * 576, 768]
        # B_inter: [B, N_layer_B * 576, 1152]
        _, A_inter = self.dino_vision_tower(images) 
        _, B_inter = self.oryx_vision_tower(images) 
        
        # 投影到 1024
        A_proj = self.mlp_layers[0](A_inter) 
        B_proj = self.mlp_layers[1](B_inter) 

        B_batch = A_proj.shape[0]
        T_target = 576 # 标准 Grid Size

        # ------------------------ 2. 结构化拆分 ------------------------
        # 恢复层级结构: [B, Layers, 1+T, D]
        A_full = A_proj.view(B_batch, self.N_layer_A, 1 + T_target, -1)
        A_cls = A_full[:, :, 0:1, :]    
        A_patches = A_full[:, :, 1:, :] # [B, N_A, 576, D]

        B_full = B_proj.view(B_batch, self.N_layer_B, 1 + T_target, -1)
        B_patches = B_full[:, :, 1:, :] # [B, N_B, 576, D]

        # ------------------------ 3. 准备 Cross-Attn 上下文 ------------------------
        # 为了计算效率，上下文也进行一次预压缩 (这里保持和 K 一致的比例)
        target_context = T_target // self.compression_K
        
        # B 压缩后作为 A 的 Key/Value
        B_patches_flat = B_patches.flatten(0, 1) # [B*Layers, 576, D]
        B_merged_ctx = bipartite_soft_matching_merge(B_patches_flat, target_context, B_patches_flat)
        B_kv_context = B_merged_ctx.view(B_batch, -1, self.global_dimension) 

        # A 压缩后作为 B 的 Key/Value
        A_patches_flat = A_patches.flatten(0, 1)
        A_merged_ctx = bipartite_soft_matching_merge(A_patches_flat, target_context, A_patches_flat)
        A_kv_context = A_merged_ctx.view(B_batch, -1, self.global_dimension)

        # ------------------------ 4. 双向 CLS 增强 ------------------------
        
        # === A 塔 CLS 增强 (Query: DINO CLS, KV: Oryx Patches) ===
        enhanced_cls_A_list = []
        for i in range(self.N_layer_A):
            curr_cls = A_cls[:, i] 
            x_in = torch.cat([curr_cls, B_kv_context], dim=1)
            # 取出第一个 token (enhanced cls)
            out = self.multi_cls_cross_attn_blocks_A[i](x_in)[:, 0:1, :]
            enhanced_cls_A_list.append(out)
            
        stacked_A = torch.cat(enhanced_cls_A_list, dim=1) # [B, N_A, D]
        weights_A = F.softmax(self.dino_cls_attn_weights, dim=0).view(1, -1, 1).to(stacked_A.dtype)
        final_cls_A = torch.sum(stacked_A * weights_A, dim=1, keepdim=True)

        # === B 塔 CLS 增强 (Query: Oryx Pseudo CLS, KV: DINO Patches) ===
        # 先生成伪 CLS
        B_ctx_flat = B_merged_ctx.view(B_batch * self.N_layer_B, -1, self.global_dimension)
        pseudo_cls_B_all = self.b_pseudo_cls_head(B_ctx_flat).view(B_batch, self.N_layer_B, 1, -1)
        
        enhanced_cls_B_list = []
        for i in range(self.N_layer_B):
            curr_cls = pseudo_cls_B_all[:, i]
            x_in = torch.cat([curr_cls, A_kv_context], dim=1)
            out = self.multi_cls_cross_attn_blocks_B[i](x_in)[:, 0:1, :]
            enhanced_cls_B_list.append(out)

        stacked_B = torch.cat(enhanced_cls_B_list, dim=1)
        weights_B = F.softmax(self.oryx_cls_attn_weights, dim=0).view(1, -1, 1).to(stacked_B.dtype)
        final_cls_B = torch.sum(stacked_B * weights_B, dim=1, keepdim=True)

        # === 全局 CLS 融合 ===
        stacked_final = torch.cat([final_cls_A, final_cls_B], dim=1) # [B, 2, D]
        weights_final = F.softmax(self.final_cls_weights, dim=0).view(1, 2, 1).to(stacked_final.dtype)
        enhanced_cls_token = torch.sum(stacked_final * weights_final, dim=1, keepdim=True) # [B, 1, D]

        # ------------------------ 5. 动态 Patch 压缩 (核心修复区) ------------------------
        
        # 上下分组 (Upper/Lower Grouping)
        half_A = self.N_layer_A // 2
        A_up, A_lo = A_patches[:, :half_A].flatten(1, 2), A_patches[:, half_A:].flatten(1, 2)
        
        half_B = self.N_layer_B // 2
        B_up, B_lo = B_patches[:, :half_B].flatten(1, 2), B_patches[:, half_B:].flatten(1, 2)

        # [关键逻辑] 动态计算目标 Token 数量
        # 无论输入包含多少层，我们都要求每一组输出 "标准Grid / K" 个 Token
        target_tokens_per_group = T_target // self.compression_K 
        # 例如 K=4 -> target=144. 输入 1152 -> remove 1008. 结果=144.

        def get_merged_tokens(x, target_n):
            current_n = x.shape[1]
            if current_n <= target_n:
                # 理论上不会发生，除非 layer=0，但为了安全加个判断
                return x 
            
            # 计算需要移除的数量 r
            r = current_n - target_n
            
            # 执行随机二分匹配
            merge_func, _ = random_bipartite_soft_matching(x, r=r)
            return merge_func(x)

        # 分别对四组特征进行压缩
        res_A_up = get_merged_tokens(A_up, target_tokens_per_group)
        res_B_up = get_merged_tokens(B_up, target_tokens_per_group)
        res_A_lo = get_merged_tokens(A_lo, target_tokens_per_group)
        res_B_lo = get_merged_tokens(B_lo, target_tokens_per_group)
        #print(f"DEBUG: mA_up final result: {res_A_up.shape}")
        #print(f"DEBUG: mA_lo final result: {res_A_lo.shape}")
        #print(f"DEBUG: mB_up final result: {res_B_up.shape}")
        #print(f"DEBUG: mB_lo final result: {res_B_lo.shape}")
        #print(f"DEBUG: cls_token shape: {enhanced_cls_token.shape}")
        # ------------------------ 6. 最终拼接 ------------------------
        # 结构: [CLS, A_up, B_up, A_lo, B_lo]
        # K=4 时: 1 + 144 + 144 + 144 + 144 = 577
        final_embeddings = torch.cat([
            enhanced_cls_token, 
            res_A_up, res_B_up, 
            res_A_lo, res_B_lo
        ], dim=1)
        #print(f"DEBUG: [AdaptiveTower] Final Embeddings Shape: {final_embeddings.shape}")
        #print(f"DEBUG: [AdaptiveTower] Embeddings Mean: {final_embeddings.mean().item():.4f}")
        #print(f"[DEBUG END] =============================\n")
        return final_embeddings, None