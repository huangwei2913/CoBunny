import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
import math
# 必须引入 BaseVisionTower
from ..base_encoder import BaseVisionTower

class SiglipVisionTower(BaseVisionTower): # 1. 改为继承 BaseVisionTower
    def __init__(self, vision_tower, args, **kwargs):
        # 2. 调用父类 init，它会自动处理 self.unfreeze_mm_vision_tower 的赋值
        super(SiglipVisionTower, self).__init__(vision_tower, args, **kwargs)
        self.is_loaded  = False
        self.vision_tower_name = vision_tower
        self.image_processor = None
        self.select_indices = [3,9,12,14,18,19,22,24]
        self.target_N = 576  
        self._hidden_size = 1152 
        self.training_stage = kwargs.get('training_stage', getattr(args, 'training_stage', 'inference'))  
        print(f"🎨 [SiglipVisionTower] 成功识别training_stage: {self.training_stage}")
        self.delay_load = kwargs.get('delay_load', False) 
        print(f"🎨 [SiglipVisionTower] 成功识别delay_load: {self.delay_load}")   
        if not self.delay_load :
            self.load_model()
        else:
            # 延迟加载时，只加载配置用于架构初始化
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"✅ [SigLIP 执行层] 状态已锁定，无需重复加载。")
            return

        # 1. 获取顶层下发的身份指令 (由 __init__ 透传而来)


        # 2. 核心决策逻辑：架构初始化 vs. 权重加载
        if self.training_stage in ["finetune", "inference"]:
            # 【搭架子模式】：微调或推理阶段
            # 目标：仅在内存/显存中构建出 SigLIP 的物理结构，不加载任何官方预训练权重
            # 优势：防止 15GB Checkpoint 注入前，官方权重对 0.65 精度成果造成残留污染
            print(f"🏗️ [SigLIP 执行层] 身份确认: {self.training_stage} | 动作: 仅构建物理架构 (Skeleton Only)")
            
            # 使用 Config 初始化一个“空壳”模型
            config = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
            self.vision_tower = SiglipVisionModel(config) 
            
        else:
            # 【加载权重模式】：预训练第一阶段 (Stage 1)
            # 目标：此时没有全量 Checkpoint，必须依赖官方原始底座进行训练
            print(f"🛒 [SigLIP 执行层] 身份确认: {self.training_stage} | 动作: 加载官方预训练底座")
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

        # 3. 硬件与精度适配 (Tesla T4 建议强制 FP16)
        self.vision_tower.to(device=self.device, dtype=torch.float16)

        # 4. 补全预处理器（Processor 通常不包含在 state_dict 注入中，需要手动补全）
        if self.image_processor is None:
            self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        # 注意：梯度控制和 Train/Eval 模式已交给中间层项目经理 (_set_subtower_grad_state)
        # 底层只需要完成物理装载即可。

        self.is_loaded = True
        print(f"✅ [SigLIP 执行层] 物理装载任务执行完毕。")

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
            mean_feat = feat.mean(dim=1, keepdim=True)
            max_feat = feat.max(dim=1, keepdim=True)[0]
            energy = torch.norm(feat, dim=-1, keepdim=True) # [b, 576, 1]
            attn_weights = F.softmax(energy, dim=1)
            weighted_feat = torch.sum(feat * attn_weights, dim=1, keepdim=True)    
            pseudo_cls = (mean_feat + max_feat + weighted_feat) / 3.0
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