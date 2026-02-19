import torch
import torch.nn.functional as F
from modelscope import AutoConfig, AutoImageProcessor
from typing import Union, List, Tuple
import sys
import os
from .base_encoder import BaseVisionTower
from bunny.util.merge import bipartite_soft_matching_merge
from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vit7b16
from safetensors.torch import load_file  
import math

DINOv3_MODEL_FACTORIES = {
    "dinounet_s": dinov3_vits16,
    "dinounet_b": dinov3_vitb16,
    "dinounet_l": dinov3_vitl16,
    "dinounet_7b": dinov3_vit7b16,
}

DINOv3_MODEL_INFO = {
    "dinounet_s": {"embed_dim": 384, "depth": 12, "num_heads": 6, "params": "~22M"},
    "dinounet_b": {"embed_dim": 768, "depth": 12, "num_heads": 12, "params": "~86M"},
    "dinounet_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "params": "~300M"},
    "dinounet_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "params": "~7B"},
}

DINOv3_INTERACTION_INDEXES = {
    "dinounet_s": [2, 5, 8, 11],
    "dinounet_b": [2, 5, 8, 11],
    "dinounet_l": [4, 11, 17, 23],
    "dinounet_7b": [9, 19, 29, 39],
}

def load_dinov3_model(model_name, pretrained_path):
    model_factory = DINOv3_MODEL_FACTORIES[model_name]
    model = model_factory(pretrained=False)
    
    # 动态拼接 safetensors 路径
    st_path = os.path.join(pretrained_path, "model.safetensors")
    
    if os.path.exists(st_path):
        print(f"Loading weights from {st_path}")
        sd = load_file(st_path)
        # 注意：这里要确保 key 对齐，如果不一致需要处理
        model.load_state_dict(sd, strict=False)
    else:
        print(f"Warning: {st_path} not found, loading default weights.")
        model = model_factory(pretrained=True)
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from modelscope import AutoConfig, AutoImageProcessor
from .base_encoder import BaseVisionTower
from safetensors.torch import load_file

# 假设这些常量在外部定义或已导入
# DINOv3_MODEL_INFO, DINOv3_INTERACTION_INDEXES, load_dinov3_model 需确保在上下文中可用

class DinoVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super(DinoVisionTower, self).__init__(vision_tower, args, delay_load)
        
        # 1. 基础参数定义
        self._vision_tower_name = vision_tower
        self._image_size = 384    # DinoV3 默认常用尺寸
        self._patch_size = 16 
        self._num_patches_cached = None 
        self.is_loaded  = False
        self.model_name = "dinounet_b"
        self.interaction_indexes = [2, 5, 8, 11]
        
        # 混合编码器对齐关键参数
        self.target_grid_size = getattr(args, "mm_vision_grid_size", 24)
        self.target_N = self.target_grid_size * self.target_grid_size
        self._hidden_size = 768  # 这里的 hidden_size 指 Backbone 输出维度
        self.target_embed_dim = self._hidden_size # 内部投影目标维度
        
        self.pretrained_path = getattr(args, "dinov3_pretrained_path", "/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m")

        # 2. Config 信息预加载（确保 delay_load 模式下属性依然可用）
        try:
            self.cfg_only = AutoConfig.from_pretrained(self._vision_tower_name)
        except Exception as e:
            from transformers import PretrainedConfig
            self.cfg_only = PretrainedConfig(hidden_size=self._hidden_size, image_size=self._image_size)

        if not self.delay_load:
            self.load_model()


    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"✅ [DinoVisionTower] 状态已锁定，无需重复加载，保护当前显存权重。")
            return
        # 1. 灵魂探测：利用 Python 反射机制检查 builder.py 是否已完成权重注入
        has_soul = False
        
        # 检查属性是否存在（空降属性检查）
        if hasattr(self, 'vision_tower') and self.vision_tower is not None:
            try:
                # 获取第一个参数的权重分布
                param = next(self.vision_tower.parameters())
                # 如果 std != 1.0，说明这是一个被训练过或从 checkpoint 恢复的有灵魂的权重
                if param.numel() > 0 and torch.std(param.data).item() != 1.0:
                    has_soul = True
            except (StopIteration, AttributeError):
                pass

        # 2. 决策：如果有灵魂，坚决不加载底座，防止 0.65 成果被官方权重“污染”
        if has_soul:
            print(f"🚀 [DINO Safe Load] 确认：微调后的视觉灵魂已由外部注入，拒绝官方底座覆盖。")
        else:
            # 只有在架构确实为空（如 Stage 1）或注入失败时，才执行手动加载
            print(f"🏗️ [DINO Initial Load] 探测不到有效权重，正在加载官方底座: {self.model_name}")
            # 这里调用你原来的加载工具函数
            self.vision_tower = load_dinov3_model(self.model_name, self.pretrained_path)

        # 3. 意志同步：状态对齐与 T4 硬件适配 (Tesla T4 推理必须强制 FP16)
        self.vision_tower.to(device=self.device, dtype=torch.float16)

        if self.unfreeze_mm_vision_tower:
            self.vision_tower.requires_grad_(True)
            self.vision_tower.train()  # 必须开启训练模式以支持反向传播
            print(f"🔥 [DINO State] Full Parameter Fine-tuning Enabled (TRAIN模式).")
        else:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()   # 推理模式
            print(f"❄️ [DINO State] Inference Mode Enabled (EVAL模式).")

        # 4. 补全预处理器（注入机制通常不带这个）
        self.image_processor = AutoImageProcessor.from_pretrained(self._vision_tower_name)
            
        self.is_loaded = True

    def _forward(self, images):
        # 确保输入精度一致
        images = images.to(device=self.device, dtype=self.dtype)
        # 获取 4 层中间层特征 (List of Tuple: (feat, cls))
        all_layers = self.vision_tower.get_intermediate_layers(
            images, n=self.interaction_indexes, return_class_token=True
        )
        aligned_layers = []
        for layer_out in all_layers:
            if isinstance(layer_out, tuple):
                feat, cls = layer_out
            else:
                feat, cls = layer_out, None
            
            # 确保精度
            feat = feat.to(images.dtype)
            if cls is not None:
                cls = cls.to(images.dtype)

            # 维度处理: [B, H, W, C] -> [B, T, C]
            if feat.dim() == 4:
                B, C, H, W = feat.shape
                feat = feat.view(B, C, H * W).permute(0, 2, 1)
            
            # 空间插值对齐到 target_N (如 24x24=576)
            if feat.shape[1] != self.target_N:
                # [B, T, C] -> [B, C, T] -> [B, C, target_N] -> [B, target_N, C]
                B, T, C = feat.shape
                hw = int(math.sqrt(T)) # 算出原始的宽高，比如 24
                target_hw = int(math.sqrt(self.target_N)) # 目标宽高，比如 24
                feat = feat.view(B, hw, hw, C).permute(0, 3, 1, 2)
                feat = F.interpolate(
                    feat, 
                    size=(target_hw, target_hw), 
                    mode="bilinear", 
                    align_corners=False
                )
                feat = feat.permute(0, 2, 3, 1).view(B, -1, C).contiguous()

            # 拼接 CLS Token: [B, 1, C] + [B, target_N, C] -> [B, 1+target_N, C]
            if cls is not None:
                cls_tokens = cls.unsqueeze(1)
                feat_with_cls = torch.cat([cls_tokens, feat], dim=1)
            else:
                # 如果没有 CLS，伪造一个均值池化 CLS 保证混合编码器结构统一
                pseudo_cls = feat.mean(dim=1, keepdim=True)
                feat_with_cls = torch.cat([pseudo_cls, feat], dim=1)
                
            aligned_layers.append(feat_with_cls)

        # 拼接所有选定层的特征
        all_intermediate_features = torch.cat(aligned_layers, dim=1)
        
        # 更新缓存的 Patch 数量 (不含 CLS)
        if self._num_patches_cached is None:
            self._num_patches_cached = self.target_N

        return aligned_layers[-1], all_intermediate_features

    # --- 必须保留的核心属性 ---
    
    @property
    def hidden_size(self):
        """返回 Backbone 的原始维度，混合编码器会根据此值建立 MLP 映射到 1024"""
        return self._hidden_size

    @property
    def num_patches(self):
        """返回插值对齐后的 Patch 数量 (不含 CLS)"""
        return self.target_N

    @property
    def image_size(self):
        return self._image_size

    @property
    def patch_size(self):
        # 优先从加载后的模型获取真实 patch_size
        return getattr(self.vision_tower, "patch_size", self._patch_size)

    @property
    def layer_count(self):
        """返回提取的中间层数量"""
        return len(self.interaction_indexes)

    @property
    def num_patches_per_side(self):
        return int(self.num_patches ** 0.5)

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self.cfg_only