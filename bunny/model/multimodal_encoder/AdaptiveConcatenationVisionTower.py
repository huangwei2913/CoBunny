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
from .siglip.siglip_encoder import SiglipVisionTowerS2
from timm.models.vision_transformer import  Mlp, Block

#建立一个处理不同输入影像大小的预处理类


from PIL import Image
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def list_lcm(numbers):
    from functools import reduce
    return reduce(lcm, numbers)

class ImageProcessorMultipleEncoders:
    def __init__(self, patch_size_list, max_size=1152, min_no_scale=384):
        self.patch_size_list = patch_size_list
        self.max_size = max_size
        self.min_no_scale = min_no_scale
        self.patch_lcm = list_lcm(patch_size_list)

    def process_image(self, image: Image.Image) -> Image.Image:
        # image is PIL.Image.Image
        W, H = image.size

        # 小于最小阈值，直接返回
        if H <= self.min_no_scale and W <= self.min_no_scale:
            return image

        # 384 ~ 1152 范围内保持不变
        if self.min_no_scale < H <= self.max_size and self.min_no_scale < W <= self.max_size:
            return image

        # 大于1152，重采样到不超过1152且为patch_lcm的最大倍数
        if H > self.max_size or W > self.max_size:
            new_H = (self.max_size // self.patch_lcm) * self.patch_lcm
            new_W = (self.max_size // self.patch_lcm) * self.patch_lcm
            image = image.resize((new_W, new_H), Image.BILINEAR)
            return image

        # 小于384但不满足最小公倍数倍数条件的，调整尺寸
        if H % self.patch_lcm != 0 or W % self.patch_lcm != 0:
            new_H = (H // self.patch_lcm) * self.patch_lcm
            new_W = (W // self.patch_lcm) * self.patch_lcm
            if new_H < 1: new_H = self.patch_lcm
            if new_W < 1: new_W = self.patch_lcm
            image = image.resize((new_W, new_H), Image.BILINEAR)

        return image
    

#使用cross Atttention模块
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


#建立一个自适应的多视觉编码器特征融合，首先合并不同编码器的token,然后通过找到与每一个token最相似的分层token
#然后将分层tokeen与分层token合并，可
class AdaptiveConcatenationVisionTower(nn.Module):
    def __init__(self,
                 vision_tower,
                 args,
                 grid_size=32):
        
        super().__init__()
        self.is_loaded = False
        self.grid_size = grid_size  #我们也可以设置
        self.num_tokens = self.grid_size ** 2
        self.max_image_size = 1152   #最大能处理的影像大小
        self.patch_size_list = [14]
        self.global_dimension = 1024  #将不同编码器的全局特征维度统一到1024这个上来
        vision_tower_name_list = vision_tower.split(";")
        #self.input_image_size = 1024 # hardcode  多视觉编码器通常预期输入大小不一致（例如CLIP是336×336，ConvNeXt是224×224或者更大），
        #为了在多编码器融合时保证输入图像处理的一致性和特征空间匹配，这里在这个融合模块层面统一固定为 1024
        self.load_vision_towers(vision_tower_name_list, args)
        self.num_heads  = 8   # 多头自注意力
        self.mlp_ratio = 4.0   # MLP隐藏层大小是输入的4倍
        self.cross_attn_block = CrossAttentionBlock(dim=self.global_dimension, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio)
      
    def load_vision_towers(self, vision_tower_name_list, args):
        self.vision_towers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()  #用于投影映射全局特征
        #首先要加载各个视觉编码器，然后对每个视觉编码器的输出做减半合并
        #然后使用辅助相似度，进行跨视觉编码器之间的token选择，先选择出于输入相似的，还是想合并在选择？？
        
        for name in vision_tower_name_list:
            if name =="facebook/dinov3-convnext-large-pretrain-lvd1689m":
                dinov3_args = deepcopy(args)
                # dinov3_args.freeze_vision = True  # Freeze vision not needed for DinoVisionTower
                dino_vision_tower = DinoVisionTower(name, dinov3_args)  # 224 image_size
                dino_vision_tower.load_model()
                self.vision_towers.append(dino_vision_tower) 
                self.patch_size_list.append(dino_vision_tower.patch_size)
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(1536, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, self.global_dimension)
                ))

            elif name=="/home/huangwei/siglip-so400m-patch14-384":  #自适应图像尺寸，patch_size=14是输入图像切分patch的大小。
                siglip_args = deepcopy(args)         #image_size 默认取384 patch_size=14 
                siglip_args.freeze_vision = True  
                siglip_vision_tower = SiglipVisionTowerS2(name, siglip_args)
                siglip_vision_tower.load_model()
                self.vision_towers.append(siglip_vision_tower)
                self.patch_size_list.append(siglip_vision_tower.patch_size)                            
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(1152, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, self.global_dimension)
                ))

        self.is_loaded = True        
                

    def load_model(self):
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"

    def forward(self, x):
        #统一处理影像，保证x
        #将输入图像预处理到一个既能兼容所有模型patch_size的大小，又不超过各模型支持的最大输入尺寸的公共尺寸
        patch_size_list = list(dict.fromkeys(self.patch_size_list))  #去除重复
        processor_ = ImageProcessorMultipleEncoders(patch_size_list)
        processed_img = processor_.process_image(x) #转换后的影像大小
        all_tokens = []
        all_cls_tokens = []
        all_patch_tokens = []
        token_lengths = [] #每一个视觉塔输出的tokens数量
        for i, vision_tower in enumerate(self.vision_towers):
            tokens = vision_tower(processed_img)  # (B, N, C_enc)
            # MLP映射到全局维度
            tokens_proj = self.mlp_layers[i](tokens)  # (B, N, global_dim)
            all_tokens.append(tokens_proj)
            token_lengths.append(tokens_proj.shape[1])
            # 取CLS token (第0个token)
            all_cls_tokens.append(tokens_proj[:, 0:1, :])  # (B,1,global_dim) 
            cls_token = tokens_proj[:, 0:1, :]  # CLS token
            patch_tokens = tokens_proj[:, 1:, :] # patch tokens
            all_cls_tokens.append(cls_token)
            all_patch_tokens.append(patch_tokens)

        # CrossAttention: 第一个视觉塔CLS对第二个视觉塔patch tokens
        # 输入形状 (B, N-1+1, C) 这里 +1是方便CrossAttention处理，拼接CLS token
        cross_input_1 = torch.cat([all_cls_tokens[0], all_patch_tokens[1]], dim=1)  
        enhanced_cls_1 = self.cross_attn_block(cross_input_1)  # 输出 (B,1,C)

        # CrossAttention: 第二个视觉塔CLS对第一个视觉塔patch tokens
        cross_input_2 = torch.cat([all_cls_tokens[1], all_patch_tokens[0]], dim=1)
        enhanced_cls_2 = self.cross_attn_block(cross_input_2)  # 输出 (B,1,C)

        # 返回融合后的增强CLS tokens
        enhanced_cls_tokens = torch.cat([enhanced_cls_1, enhanced_cls_2], dim=1)  # (B,2,C)
        #把中间层的token也使用起来，也就说中间层的
        









       

        return enhanced_cls_tokens
        
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.clip_vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.clip_vision_tower.parameters()).device

    @property
    def config(self):
        assert NotImplementedError
        pass

    @property
    def hidden_size(self):
        return sum([_.hidden_size for _ in self.vision_towers])

    @property
    def num_patches(self):
        return self.num_tokens
