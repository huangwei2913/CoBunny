import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
import math

class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_indices = [3,9,12,14,18,19,22,24]
        # 核心参数对齐
        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.target_N = 576  # 24x24, 与 DINO 对齐
        self._hidden_size = 1152 # SigLIP-SO400M 的标准维度

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            return
        
        # 加载 Processor 和 Model
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        
        # 冻结 Backbone，Pretrain 阶段只练 Projector 和融合层
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, images):
        """
        高性能前向传播：
        1. 一次性提取所有隐藏层。
        2. 自动进行空间缩放 (729 -> 576)。
        3. 拼接伪 CLS 以对齐混合编码器接口。
        """
        if type(images) is list:
            # 如果输入是列表，堆叠成 Batch 处理，提高 GPU 并行度
            images = torch.stack(images)

        # 1. 一次前向传播获取所有层结果 (显存高效)
        output = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True
        )

        # 2. 选择需要的层 (例如最后 4 层，或者根据 args 指定)
        # 这里为了配合 K=8 架构的多层融合，我们取最后 4 层
        all_hidden_states = output.hidden_states
    
        selected_layers = [all_hidden_states[i] for i in self.select_indices]

        aligned_layers = []
        for feat in selected_layers:
            # SigLIP 特征形状: [B, 729, 1152]
            b, n, d = feat.shape
            h = w = int(math.sqrt(n)) # 原生通常是 27
            
            # --- 空间对齐 (27x27 -> 24x24) ---
            # 转换为 [B, C, H, W] 进行 bicubic 插值
            feat = feat.view(b, h, w, d).permute(0, 3, 1, 2)
            feat = F.interpolate(
                feat, 
                size=(24, 24), 
                mode='bicubic', 
                align_corners=False
            )
            # 还原为 [B, 576, 1152]
            feat = feat.permute(0, 2, 3, 1).view(b, -1, d) 
            
            # --- 构造伪 CLS (对齐 DINO 接口) ---
            # 使用均值作为该层的全局表示，占据 index 0
            pseudo_cls = feat.mean(dim=1, keepdim=True)
            combined = torch.cat([pseudo_cls, feat], dim=1) # [B, 577, 1152]
            aligned_layers.append(combined)

        # 最后一层作为 image_features (用于主要特征参考)
        image_features = aligned_layers[-1]
        
        # 所有选定层拼接作为 gallery (用于 AdaptiveConcatenation 融合)
        # 结果形状: [B, 4 * 577, 1152]
        patch_tokens_gallery = torch.cat(aligned_layers, dim=1).contiguous()

        return image_features, patch_tokens_gallery

    @property
    def dummy_feature(self):
        return torch.zeros(1, 577, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_patches(self):
        """返回对齐后的 Patch 数量 (不含伪 CLS)"""
        return self.target_N
    @property
    def layer_count(self):
        """动态返回提取的层数，供融合塔自动对齐"""
        return len(self.select_indices)