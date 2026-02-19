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
        self.is_loaded  = False
        self.vision_tower_name = vision_tower
        self.select_indices = [3,9,12,14,18,19,22,24]
        self.target_N = 576  
        self._hidden_size = 1152 

        if not delay_load:
            self.load_model()
        else:
            # 延迟加载时，只加载配置用于架构初始化
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"✅ [SigLIP] 状态已锁定，无需重复加载，保护当前显存权重。")
            return
        """
        核心逻辑：实现“防御性加载”。
        1. 检查灵魂（权重）是否已由 builder.py 的 from_pretrained 注入。
        2. 如果已被注入，则跳过官方权重加载，保护微调成果。
        3. 如果是空架构，则加载官方底座。
        """
        # 1. 探测“灵魂”注入状态
        has_injected_weights = False
        
        # 检查属性是否存在，且不是 None (应对 HuggingFace 的 setattr 注入)
        if hasattr(self, 'vision_tower') and self.vision_tower is not None:
            try:
                # 检查参数量和实质内容：如果 std != 1.0 且 > 0，说明是真实的微调权重
                param = next(self.vision_tower.parameters())
                if param.numel() > 0 and torch.std(param.data).item() != 1.0:
                    has_injected_weights = True
            except (StopIteration, AttributeError):
                pass

        # 2. 决策：执行加载或保护
        if has_injected_weights:
            print(f"🚀 [SigLIP Safe Load] 探测到有效的微调权重注入，跳过官方底座覆盖，保护 0.65 精度成果。")
            # 即使注入了权重，通常 processor 也需要手动补齐
            if self.image_processor is None:
                self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        else:
            # 只有在完全感知不到“灵魂”的情况下（如 Stage 1 或注入失败），才加载官方底座
            print(f"🏗️ [SigLIP Initial Load] 探测不到有效权重，正在加载官方预训练底座: {self.vision_tower_name}")
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
            self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        # 3. 意志同步：状态对齐与 T4 硬件适配
        # 强制转换到正确的设备和精度 (Tesla T4 必须用 FP16)
        self.vision_tower.to(device=self.device, dtype=torch.float16)

        # 根据配置决定是否开启梯度 (全量微调 vs 推理)
        if self.unfreeze_mm_vision_tower:
            self.vision_tower.requires_grad_(True)
            self.vision_tower.train()
            print(f"🔥 [SigLIP State] 视觉塔模式: TRAIN (微调中)")
        else:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()
            print(f"❄️ [SigLIP State] 视觉塔模式: EVAL (冻结/推理)")

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