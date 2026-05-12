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
    def __init__(self, vision_tower, args,**kwargs):
        super(DinoVisionTower, self).__init__(vision_tower, args,**kwargs)
        
        # 1. 基础参数定义
        self.vision_tower_name = vision_tower
        self._image_size = 384    # DinoV3 默认常用尺寸
        self._patch_size = 16 
        self._num_patches_cached = None 
        self.is_loaded  = False
        self.model_name = "dinounet_b"
        self.interaction_indexes = [2, 5, 8, 11]
        self.training_stage = kwargs.get('training_stage', getattr(args, 'training_stage', 'inference'))  
        print(f"🎨 [DinoVisionTower] 成功识别training_stage: {self.training_stage}")
        self.delay_load = kwargs.get('delay_load', False) 
        print(f"🎨 [DinoVisionTower] 成功识别delay_load: {self.delay_load}")   
        # 混合编码器对齐关键参数
        self.target_grid_size = getattr(args, "mm_vision_grid_size", 24)
        self.target_N = self.target_grid_size * self.target_grid_size
        self._hidden_size = 768  # 这里的 hidden_size 指 Backbone 输出维度
        self.target_embed_dim = self._hidden_size # 内部投影目标维度
        
        self.pretrained_path = getattr(args, "dinov3_pretrained_path", "/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m")

        # 2. Config 信息预加载（确保 delay_load 模式下属性依然可用）
        try:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)
        except Exception as e:
            from transformers import PretrainedConfig
            self.cfg_only = PretrainedConfig(hidden_size=self._hidden_size, image_size=self._image_size)

        if not self.delay_load:
            self.load_model()


    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"✅ [DinoVisionTower] 状态已锁定，无需重复加载，保护当前显存权重。")
            return
        # 2. 核心决策逻辑：搭架子还是装权重？
        if self.training_stage  in ["finetune", "inference"]:
            # 【搭架子模式】：微调和推理时，我们只需要物理架构
            # 权重会由 BunnyMetaModel 后续通过全量 Checkpoint 注入
            print(f"🏗️ [DINO 执行层] 身份: {self.training_stage } -> 模式: 仅构建物理架构 (Skeleton Only)")
            
            # 使用工厂函数直接创建架构，pretrained 设为 False
            model_factory = DINOv3_MODEL_FACTORIES[self.model_name]
            self.vision_tower = model_factory(pretrained=False)
        else:  #第一个阶段
            # 【官方加载模式】：预训练第一阶段（Stage 1）
            # 此时没有全量 Checkpoint，必须加载官方原始权重
            print(f"🛒 [DINO 执行层] 身份: {self.training_stage} -> 模式: 加载官方预训练权重")
            self.vision_tower = load_dinov3_model(self.model_name, self.pretrained_path)
        
        self.vision_tower.to(device=self.device, dtype=torch.float16)
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.is_loaded = True
        print(f"✅ [DinoVisionTower] 装载任务执行完毕。")
       
    def _forward(self, images):
        # 确保输入精度一致
        images = images.to(device=self.device, dtype=self.dtype)
        
        # 获取多层中间层特征 (List of Tuple: (feat, cls))
        all_layers = self.vision_tower.get_intermediate_layers(
            images, n=self.interaction_indexes, return_class_token=True
        )
        
        raw_patch_list = []  # 存放空间对齐后的纯 Patch 特征 [B, target_N, C]
        cls_token_list = []  # 存放每一层提取或伪造的原始 CLS Token [B, 1, C]

        for layer_out in all_layers:
            if isinstance(layer_out, tuple):
                feat, cls = layer_out
            else:
                feat, cls = layer_out, None
            
            # 确保精度
            feat = feat.to(images.dtype)
            if cls is not None:
                cls = cls.to(images.dtype)

            # --- 1. 维度处理: [B, C, H, W] -> [B, T, C] ---
            # 如果 backbone 出来的是 4D 结构，统一展平为 1D 序列
            if feat.dim() == 4:
                B, C, H, W = feat.shape
                feat = feat.view(B, C, H * W).permute(0, 2, 1)

            # --- 2. 提取当前层 CLS (确保为 [B, 1, C]) ---
            # 注意：如果原本没有 CLS，就在没插值之前，利用最真实的原始特征算一个均值作为 CLS
            if cls is not None:
                cls_tokens = cls.unsqueeze(1)
            else:
                cls_tokens = feat.mean(dim=1, keepdim=True)
            cls_token_list.append(cls_tokens)

            # --- 3. 空间插值对齐到 target_N (如 24x24=576) ---
            B, T, C = feat.shape
            if T != self.target_N:
                hw = int(math.sqrt(T)) # 算出原始的宽高
                target_hw = int(math.sqrt(self.target_N)) # 目标宽高
                
                # 转换回 2D 结构用于插值 [B, C, hw, hw]
                feat = feat.view(B, hw, hw, C).permute(0, 3, 1, 2)
                
                # 动态选择插值模式：缩小用 area 保留笔画，放大用 bicubic 防止马赛克
                interp_mode = "area" if hw > target_hw else "bicubic"
                feat = F.interpolate(
                    feat, 
                    size=(target_hw, target_hw), 
                    mode=interp_mode, 
                    align_corners=False if interp_mode == "bicubic" else None
                )
                
                # 恢复为序列格式 [B, target_hw*target_hw, C] -> [B, target_N, C]
                feat = feat.permute(0, 2, 3, 1).view(B, -1, C).contiguous()
            
            # 将处理好的纯 Patch 特征存入列表
            raw_patch_list.append(feat)

        # ==========================================
        # 核心改动：DeepSeek 启发的 "全局 CLS 跨层聚合"
        # ==========================================
        # 此时 cls_token_list 包含了所有被选中层的 CLS
        # 我们把它们叠在一起求平均，生成一个融合了“浅层纹理”和“深层语义”的超级 CLS
        # 形状变化: [Layers, B, 1, C] -> mean(dim=0) -> [B, 1, C]
        merged_cls = torch.stack(cls_token_list, dim=0).mean(dim=0)

        # --- 4. 重新组装 ---
        aligned_layers = []
        for feat in raw_patch_list:
            # 现在，每一层的 Patch 前面，拼接的不再是它自己的偏科 CLS，
            # 而是这个跨越所有深度的最强全局指导信号 merged_cls
            feat_with_cls = torch.cat([merged_cls, feat], dim=1) # [B, 1+target_N, C]
            aligned_layers.append(feat_with_cls)

        # 拼接所有选定层的特征 (发往后续的 Connector)
        # 输出形状: [B, Layers * (1 + target_N), C]
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