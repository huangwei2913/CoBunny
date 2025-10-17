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
#建立一个自适应的多视觉编码器特征融合，首先合并不同编码器的token,然后通过找到与每一个token最相似的分层token
#然后将分层tokeen与分层token合并
class AdaptiveConcatenationVisionTower(nn.Module):
    def __init__(self,
                 vision_tower,
                 args,
                 grid_size=32):
        
        super().__init__()
        self.is_loaded = False
        self.grid_size = grid_size  #我们也可以设置
        self.num_tokens = self.grid_size ** 2
        
        vision_tower_name_list = vision_tower.split(";")
        self.input_image_size = 1024 # hardcode  多视觉编码器通常预期输入大小不一致（例如CLIP是336×336，ConvNeXt是224×224或者更大），
        #为了在多编码器融合时保证输入图像处理的一致性和特征空间匹配，这里在这个融合模块层面统一固定为 1024
        self.load_vision_towers(vision_tower_name_list, args)

      
    def load_vision_towers(self, vision_tower_name_list, args):
        self.vision_towers = nn.ModuleList()
        #首先要加载各个视觉编码器，然后对每个视觉编码器的输出做减半合并
        #然后使用辅助相似度，进行跨视觉编码器之间的token选择，先选择出于输入相似的，还是想合并在选择？？
        #
        for name in vision_tower_name_list:
            if name =="facebook/dinov3-convnext-large-pretrain-lvd1689m":
                dinov3_args = deepcopy(args)
                dinov3_args.freeze_vision = False
                dino_vision_tower = DinoVisionTower(name, dinov3_args)  # 224 image_size
                dino_vision_tower.load_model()
                self.vision_towers.append(dino_vision_tower)     
            elif name=="/home/huangwei/siglip-so400m-patch14-384":  #自适应图像尺寸，patch_size=14是输入图像切分patch的大小。
                siglip_args = deepcopy(args)         #image_size 默认取384 patch_size=14 
                siglip_args.freeze_vision = False  
                siglip_vision_tower = SiglipVisionTowerS2(name, siglip_args)
                siglip_vision_tower.load_model()
                self.vision_towers.append(siglip_vision_tower)                           
                pass

        self.is_loaded = True        
                

    def load_model(self):
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"

    def forward(self, x):
        #统一处理影像，保证x
        #将输入图像预处理到一个既能兼容所有模型patch_size的大小，又不超过各模型支持的最大输入尺寸的公共尺寸
        
       

        return features
        
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
