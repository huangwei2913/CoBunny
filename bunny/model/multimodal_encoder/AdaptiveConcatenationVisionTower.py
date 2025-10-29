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
from  bunny.util.utils import CrossAttentionBlock, ImageProcessorMultipleEncoders
from .oryx_vit import OryxViTWrapper
from bunny.util.merge import bipartite_soft_matching_merge


# 辅助函数：提取最后一层的 CLS token
def _get_cls_token(aligned_feature_last_layer: torch.Tensor) -> torch.Tensor:
    return aligned_feature_last_layer[:, 0:1, :] # 形状: [B, 1, D]

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
        self.global_dimension = self.num_tokens  #将不同编码器的全局特征维度统一到1024这个上来
        vision_tower_name_list = vision_tower.split(";")  #假定只有两个视觉编码器，dinov3在前
        #self.input_image_size = 1024 # hardcode  多视觉编码器通常预期输入大小不一致（例如CLIP是336×336，ConvNeXt是224×224或者更大），
        #为了在多编码器融合时保证输入图像处理的一致性和特征空间匹配，这里在这个融合模块层面统一固定为 1024
        self.load_vision_towers(vision_tower_name_list, args)
        self.num_heads  = 8   # 多头自注意力
        self.mlp_ratio = 4.0   # MLP隐藏层大小是输入的4倍
        self.cross_attn_block = CrossAttentionBlock(dim=self.global_dimension, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio)
        self.vision_towers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

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
            oryx_args.mm_resampler_type = "dynamic_compressor" #默认使用这个
            self.oryx_vision_tower = OryxViTWrapper(oryx_args.vision_tower,oryx_args)
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
        else:
            return         

    def load_model(self):
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"

    def forward(self, x):
        #将输入图像预处理到一个既能兼容所有模型patch_size的大小，又不超过各模型支持的最大输入尺寸的公共尺寸
        patch_size_list = list(dict.fromkeys(self.patch_size_list))  #去除重复
        processor_ = ImageProcessorMultipleEncoders(patch_size_list)
        processed_img = processor_.process_image(x) #转换后的影像大小
        #从第dinov3视觉编码器获取特征
        A_last_layer, A_intermediate_tokens = self.dino_vision_tower(processed_img) # A_intermediate_tokens: [B, N_A*(1+T_target), C_A]
        A_tokens_proj = self.mlp_layers[0](A_intermediate_tokens) # [B, N_A*(1+T_target), D_target]
        
        # OryxViT (B)
        B_last_layer, B_intermeidate_tokens, _ = self.oryx_vision_tower(processed_img) # B_intermeidate_tokens: [B, N_B*T_target, C_B]
        B_tokens_proj = self.mlp_layers[1](B_intermeidate_tokens) # [B, N_B*T_target, D_target]

        dino_cls_token_raw = _get_cls_token(A_last_layer) # [B, 1, 768]        
        dino_cls_token_proj = self.mlp_layers[0](dino_cls_token_raw) # [B, 1, D_target]

        x_for_cross_attn = torch.cat([dino_cls_token_proj, B_tokens_proj], dim=1)
        enhanced_cls_token = self.cross_attn_block(x_for_cross_attn)

   
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

        return x
    

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
        # 注意：这个属性通常在 VLM 初始化时被 LLM 端用于确定占位符数量。
        # 由于 Token 数量在 forward 中是动态变化的（Token Merging），我们无法返回一个固定的准确值。
        
        # 最佳实践：返回融合后序列的最大可能长度 (用于创建占位符)
        # 最终序列长度 L_final = 1 (CLS) + L_upper_merged + L_lower_merged
        # 我们可以计算出在合并了 50% tokens 后，序列的理论长度。
        
        # L_per_layer = T_target (例如 576)
        # N_A_half = N_A // 2
        # N_B_half = N_B // 2
        # 合并前总 Patch Tokens: (N_A_half + N_B_half) * T_target * 2
        # 合并 50% 后: L_upper_merged = L_A_up * 0.5 + L_B_up * 0.5 
        
        # 由于无法在初始化时精确计算动态长度，通常有两种策略：
        # 1. 返回一个足够大的固定值（例如原始 T_target * N_A + N_B）作为 LLM 的占位符长度。
        # 2. 依赖 LLM 在 prepare_inputs_labels_for_multimodal 中使用 final_tokens.shape[1] 动态确定长度。
        
        # 采用策略 1: 返回原始未合并状态下的总 Patch Tokens 数量 (保守值)
        # (N_A * T_target) + (N_B * T_target)
        T_target = self.dino_vision_tower.target_N
        N_A = len(self.dino_vision_tower.interaction_indexes)
        N_B = len(self.oryx_vision_tower.interaction_indexes)
        
        # 返回的最大 token 数量（不含 CLS）： (N_A // 2 + N_B // 2) * T_target * 2
        # 由于合并了 50%，实际返回的 num_patches 应该是合并后的数量
        # L_merged = (N_A + N_B) / 2 * T_target * 0.5
        
        # 假设 LLM 端需要的是最终的序列长度 (不含 CLS)
        # L_final_patches = (N_A * T_target / 2) + (N_B * T_target / 2)
        
        # 鉴于动态合并的复杂性，最安全的方法是让 LLM 模块知道最终的 token 数量：
        
        # L_total_original_tokens_per_group = (N_A//2 * T_target) + (N_B//2 * T_target)
        L_final_patches = (N_A // 2 * T_target) + (N_B // 2 * T_target)
        
        # 最终序列长度 (不含 CLS)
        return L_final_patches
		




