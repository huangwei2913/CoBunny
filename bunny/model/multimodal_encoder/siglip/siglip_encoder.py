import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
import math
# 必须引入 BaseVisionTower
from ..base_encoder import BaseVisionTower

class SiglipVisionTower(BaseVisionTower): # 1. 改为继承 BaseVisionTower
    def __init__(self, vision_tower, args, delay_load=False):
        # 2. 调用父类 init，它会自动处理 self.unfreeze_mm_vision_tower 的赋值
        super(SiglipVisionTower, self).__init__(vision_tower, args, delay_load)

        self.vision_tower_name = vision_tower
        self.select_indices = [3,9,12,14,18,19,22,24]
        self.target_N = 576  
        self._hidden_size = 1152 

        if not delay_load:
            self.load_model()
        else:
            # 延迟加载时，只加载配置用于架构初始化
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            return
        
        # 加载核心组件
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        
        # 3. 核心解冻/冻结逻辑控制
        # self.unfreeze_mm_vision_tower 是从 args.unfreeze_mm_vision_tower 自动获取的
        # 🚨 修正点：确保从 args 准确读取，并显式转换或检查内容
 
        # 调试打印：让你在日志里一眼看到底传了什么
        print(f"DEBUG: unfreeze_mm_vision_tower value is.... {self.unfreeze_mm_vision_tower}")

        if self.unfreeze_mm_vision_tower:
            self.vision_tower.requires_grad_(True)
            self.vision_tower.train() # 别忘了开启 train 模式
            print(f"🔥 [SigLIP] Full Parameter Fine-tuning Enabled.")
        else:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()
            print(f"❄️ [SigLIP] Backbone Frozen.")

        self.is_loaded = True

    # 4. 补充一个 _forward 方法，这是为了兼容 BaseVisionTower 可能有的接口调用
    def _forward(self, images):
        return self.forward(images)

    def forward(self, images):
        if type(images) is list:
            images = torch.stack(images)

        # 确保数据在正确的设备和精度上
        images = images.to(device=self.device, dtype=self.dtype)

        output = self.vision_tower(
            images,
            output_hidden_states=True
        )

        all_hidden_states = output.hidden_states
        selected_layers = [all_hidden_states[i] for i in self.select_indices]

        aligned_layers = []
        for feat in selected_layers:
            b, n, d = feat.shape
            h = w = int(math.sqrt(n)) 
            
            # 空间对齐
            feat = feat.view(b, h, w, d).permute(0, 3, 1, 2)
            feat = F.interpolate(
                feat, 
                size=(24, 24), 
                mode='bicubic', 
                align_corners=False
            )
            feat = feat.permute(0, 2, 3, 1).view(b, -1, d) 
            
            # 伪 CLS 构造
            pseudo_cls = feat.mean(dim=1, keepdim=True)
            combined = torch.cat([pseudo_cls, feat], dim=1) 
            aligned_layers.append(combined)

        image_features = aligned_layers[-1]
        patch_tokens_gallery = torch.cat(aligned_layers, dim=1).contiguous()

        return image_features, patch_tokens_gallery

    @property
    def layer_count(self):
        return len(self.select_indices)
    
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