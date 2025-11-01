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
from bunny.util.merge import bipartite_soft_matching_merge
from PIL import Image
from torchvision import transforms
from typing import Dict


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
class SingleImageProcessor(object):
    def __init__(self, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, image_size=1024):
        self.mean = mean
        self.std = std
        self.image_size = image_size
        # 定义 PIL Image 到 Tensor 的转换流程
        self.transform = transforms.Compose([
            # 将图像调整到目标尺寸 (例如 1024x1024)
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            # 转换为 PyTorch Tensor，并自动归一化到 [0, 1]
            transforms.ToTensor(),
            # 使用 ImageNet 参数进行标准化
            transforms.Normalize(mean=mean, std=std)
        ])
        # 方便 expand2square 函数访问裁剪尺寸
        self.crop_size = {'height': image_size, 'width': image_size}
        self.image_mean = mean # 用于 pad 操作
    def preprocess(self, image: Image.Image, return_tensors='pt') -> Dict[str, torch.Tensor]:
        if image.mode != 'RGB':
             image = image.convert('RGB')
        tensor = self.transform(image) # [C, H, W]
        if return_tensors == 'pt':
             # LazySupervisedDataset 的 __getitem__ 会取出 [0]
             return {'pixel_values': tensor.unsqueeze(0)} # [1, C, H, W]
        raise NotImplementedError("Only PyTorch tensors are supported.")
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
    def __init__(self,
                 vision_tower,
                 args,
                 grid_size=32):
        super().__init__()
        self.is_loaded = False
        self.grid_size = grid_size  #我们也可以设置,这个站所有分词一般大小
        self.num_tokens = self.grid_size ** 2
        self.max_image_size = 1152   #最大能处理的影像大小
        self.patch_size_list = [14]
        self.target_image_size = 384 # 明确目标输入尺寸
        self.vision_towers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.global_dimension = 1024  #将不同编码器的全局特征维度统一到1024这个上来
        vision_tower_name_list = []
        vision_tower_name_list.append("facebook/dinov3-convnext-large-pretrain-lvd1689m")
        vision_tower_name_list.append('oryx_vit:/mnt/THUdyhOryx-ViT/oryx_vit.pth')
         #假定只有两个视觉编码器，dinov3在前
        #self.input_image_size = 1024 # hardcode  多视觉编码器通常预期输入大小不一致（例如CLIP是336×336，ConvNeXt是224×224或者更大），
        #为了在多编码器融合时保证输入图像处理的一致性和特征空间匹配，这里在这个融合模块层面统一固定为 1024
        self.num_heads  = 8   # 多头自注意力
        self.mlp_ratio = 2.0   # MLP隐藏层大小是输入的4倍
        self.load_vision_towers(vision_tower_name_list, args)
        self.cross_attn_block = CrossAttentionBlock(dim=self.global_dimension, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio)
        self.image_processor = SingleImageProcessor(
            mean=IMAGENET_DEFAULT_MEAN, 
            std=IMAGENET_DEFAULT_STD, 
            image_size=self.target_image_size # 例如 1024
        )
     #直接写 
    def load_vision_towers(self, vision_tower_name_list, args):
        if self.is_loaded==False:
            self.mlp_layers = nn.ModuleList()  #用于投影映射全局特征
            dinov3_args = deepcopy(args)  #创建dinov配置
            oryx_args = deepcopy(args)  # 创建oryx配置文件
            dinov3_args.vision_tower = vision_tower_name_list[0]
            oryx_args.vision_tower = vision_tower_name_list[1]
            self.dino_vision_tower = DinoVisionTower(dinov3_args.vision_tower, dinov3_args)
            self.dino_vision_tower.load_model()
            self.patch_size_list.append(self.dino_vision_tower.patch_size)
            path = oryx_args.vision_tower.split(":")[1]
            oryx_args.mm_resampler_type = "dynamic_compressor" #默认使用这个
            self.oryx_vision_tower = OryxViTWrapper(oryx_args.vision_tower, path=path, args=oryx_args)
            self.oryx_vision_tower.load_model()
            self.patch_size_list.append(self.oryx_vision_tower.patch_size)
            self.vision_towers.append(self.dino_vision_tower)
            self.vision_towers.append(self.oryx_vision_tower)
            self.is_loaded = True
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(768, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.global_dimension)
            ))

            self.mlp_layers.append(nn.Sequential(
                    nn.Linear(1152, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, self.global_dimension)
            ))
            self.num_dino_layers = len(self.dino_vision_tower.interaction_indexes) # N_A，例如 4
            self.dino_cls_attn_weights = nn.Parameter(torch.ones(self.num_dino_layers))  #
            self.multi_cls_cross_attn_blocks = nn.ModuleList([
                    CrossAttentionBlock(dim=self.global_dimension, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio)
                    for _ in range(self.num_dino_layers)
                ])

        else:
            return 
                
    def load_model(self):
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"
    
    def forward(self, x):
        #将输入图像预处理到一个既能兼容所有模型patch_size的大小，又不超过各模型支持的最大输入尺寸的公共尺寸
        patch_size_list = list(dict.fromkeys(self.patch_size_list))  #去除重复
        processor_ = ImageProcessorMultipleEncoders(patch_size_list, image_size=self.target_image_size)
        processed_img = processor_.process_image(x) #转换后的影像大小
        #从第dinov3视觉编码器获取特征


        A_last_layer, A_intermediate_tokens = self.dino_vision_tower(processed_img) # A_intermediate_tokens: [B, N_A*(1+T_target), C_A]
        A_tokens_proj = self.mlp_layers[0](A_intermediate_tokens) # [B, N_A*(1+T_target), D_target]
        # OryxViT (B)
        B_last_layer,  _ , B_intermeidate_tokens = self.oryx_vision_tower(processed_img) # B_intermeidate_tokens: [B, N_B*T_target, C_B]
        B_tokens_proj = self.mlp_layers[1](B_intermeidate_tokens) # [B, N_B*T_target, D_target]
        
        N_A = len(self.dino_vision_tower.interaction_indexes)  # 总共有师曾
        T_target = self.dino_vision_tower.target_N

        A_layers_full = A_tokens_proj.view(
            A_tokens_proj.shape[0], N_A, (1 + T_target), self.global_dimension
        )

        all_dino_cls_tokens = A_layers_full[:, :, 0:1, :]

        B_len = B_tokens_proj.shape[1]
        target_B_len = B_len // 4 # 576
        B_tokens_proj_compressed = bipartite_soft_matching_merge(
                    B_tokens_proj, 
                    target_B_len, 
                    B_tokens_proj, 
                    mode="mean" # 或其他合适的模式
        )
        enhanced_cls_tokens_list = []
        for i in range(N_A):
            # Q: 当前层的 CLS Token [B, 1, D]
            current_cls = all_dino_cls_tokens[:, i, :, :]
            
            # K/V: 压缩后的 Oryx Tokens
            # 将 CLS 与压缩后的 B 塔特征拼接作为 Attention 输入
            x_for_cross_attn = torch.cat([current_cls, B_tokens_proj_compressed], dim=1)
            
            # 使用对应层的 CrossAttentionBlock
            # enhanced_cls: [B, 1, D]
            enhanced_cls = self.multi_cls_cross_attn_blocks[i](x_for_cross_attn)
            enhanced_cls_tokens_list.append(enhanced_cls)

        stacked_enhanced_cls = torch.cat(enhanced_cls_tokens_list, dim=1)
        weights = F.softmax(self.dino_cls_attn_weights, dim=0)
        weights = weights.view(1, N_A, 1, 1)
        enhanced_cls_token = torch.sum(stacked_enhanced_cls * weights, dim=1)
        #dino_cls_token_raw = _get_cls_token(A_last_layer) # [B, 1, 768]        
        #dino_cls_token_proj = self.mlp_layers[0](dino_cls_token_raw) # [B, 1, D_target]
        x_for_cross_attn = torch.cat([enhanced_cls_token, B_tokens_proj_compressed], dim=1) # [B, 1 + 576, D]
        enhanced_cls_token = self.cross_attn_block(x_for_cross_attn)
        #x_for_cross_attn = torch.cat([dino_cls_token_proj, B_tokens_proj], dim=1)
        #enhanced_cls_token = self.cross_attn_block(x_for_cross_attn)
        N_A = len(self.dino_vision_tower.interaction_indexes) 
        N_B = len(self.oryx_vision_tower.interaction_indexes)
        T_target = self.dino_vision_tower.target_N
        # 1. 核心操作：View/Reshape
        # 目标: 将序列还原成 [Batch, Layers, Tokens_per_Layer, Dim] 结构
        # Tokens_per_Layer = 1 (CLS) + T_target (Patches)
        A_layers = A_tokens_proj.view(A_tokens_proj.shape[0], # B
                                       N_A, # Layers N_A
                                       (1 + T_target), # Tokens per Layer (1 + T_target)
                                       self.global_dimension) # Dim D_target
        # 2. **剥离每层的 CLS token**，只保留 Patches
        # 形状: [B, N_A, T_target, D_target]
        A_patches = A_layers[:, :, 1:, :]     
        A_half = N_A // 2
        # A_upper_group: [B, A_half * T_target, D_target]
        A_upper_group = A_patches[:, :A_half].flatten(1, 2) 
        A_lower_group = A_patches[:, A_half:].flatten(1, 2)
        T_target_B = self.oryx_vision_tower.target_N
        B_layers = B_tokens_proj.view(B_tokens_proj.shape[0], 
                                     N_B, 
                                     T_target_B, # OryxViT 序列长度 T_target
                                     self.global_dimension)
        B_half = N_B // 2
        B_upper_group = B_layers[:, :B_half].flatten(1, 2) 
        B_lower_group = B_layers[:, B_half:].flatten(1, 2)
        r_A = A_upper_group.shape[1] // 2 # 合并 50% 的 tokens
        r_B = B_upper_group.shape[1] // 2
        A_upper_merged = bipartite_soft_matching_merge(A_upper_group, r_A, A_upper_group, mode="mean")
        A_lower_merged = bipartite_soft_matching_merge(A_lower_group, r_A, A_lower_group, mode="mean")
        B_upper_merged = bipartite_soft_matching_merge(B_upper_group, r_B, B_upper_group, mode="mean")
        B_lower_merged = bipartite_soft_matching_merge(B_lower_group, r_B, B_lower_group, mode="mean")
        #采用的策略不同
        final_upper_tokens = torch.cat([A_upper_merged, B_upper_merged], dim=1)
        final_lower_tokens = torch.cat([A_lower_merged, B_lower_merged], dim=1)
        final_tokens = torch.cat([enhanced_cls_token, final_upper_tokens, final_lower_tokens], dim=1)
        print(f" final_tokens shape is ......................: {final_tokens.shape}, dtype: {final_tokens.dtype}")
        return final_tokens,x
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    @property
    def config(self):
        assert NotImplementedError
        pass
    @property
    def hidden_size(self):
        # 融合后的特征维度 D_target，即 self.global_dimension
        # 统一后的特征维度 D_target (例如 1024)
        return self.global_dimension
    @property
    def num_patches(self):

        T_target = self.dino_vision_tower.target_N
        T_target_B = self.oryx_vision_tower.target_N
        N_A = len(self.dino_vision_tower.interaction_indexes)
        N_B = len(self.oryx_vision_tower.interaction_indexes)
        # Tokens 压缩了 50%
        L_final_patches = (N_A // 2 * T_target) + (N_B // 2 * T_target_B)
        L_final_patches = L_final_patches // 2
        return L_final_patches