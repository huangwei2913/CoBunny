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

def load_dinov3_model(model_name: str, pretrained_path: str = None) -> DinoVisionTransformer:
    """Load DINOv3 model with pretrained weights"""
    
    if model_name not in DINOv3_MODEL_FACTORIES:
        supported_models = list(DINOv3_MODEL_FACTORIES.keys())
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported_models}")
    
    model_factory = DINOv3_MODEL_FACTORIES[model_name]
    
    # If pretrained path is provided, use custom weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading custom pretrained weights from {pretrained_path}")
        # Create model first
        model = model_factory(pretrained=False)
        # Load custom weights
        #state_dict = torch.load(pretrained_path, map_location="cpu")
        sd = load_file("/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m/model.safetensors")
        model.load_state_dict(sd, strict=False)
        print("Successfully loaded custom pretrained weights")
    else:
        # Use default pretrained weights
        print(f"Loading default pretrained weights for {model_name}")
        model = model_factory(pretrained=True)
        print("Successfully loaded default pretrained weights")

    return model


class DinoVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super(DinoVisionTower, self).__init__(vision_tower, args, delay_load)
        self._vision_tower_name = vision_tower
        self._image_size = 384   #取一个基本默认值，224, 384, 448, 512
        self._patch_size = 16    #这个不代表最后的隐藏层获得的tokens数量
        self._num_patches_cached = None  # 缓存动态计算的patch数
        self.select_feature = 'cls_patch'
        self.is_loaded = False
        self.model_name = "dinounet_b"    #默认设置成这个模型
        self.interaction_indexes = [2,  5,  8,  11]
        self.target_dim  = 768  #假设是这个大小
        self.target_embed_dim = getattr(self, "target_embed_dim", self.target_dim )
        self.target_grid_size = getattr(self, "target_grid_size", self._patch_size)
        self.target_N = self.target_grid_size * self.target_grid_size

        self.pretrained_path =  "/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m"   
        self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)  #直接用modelscope里面的配置
        if not self.delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)  #直接用modelscope里面的配置

    def load_model(self, device_map=None):
        model_info = DINOv3_MODEL_INFO[self.model_name]
        interaction_indexes = DINOv3_INTERACTION_INDEXES[self.model_name]
        print(f"🔧 Creating DINOv3 encoder: {self.model_name}")
        print(f"   Embedding dimension: {model_info['embed_dim']}")
        print(f"   Model depth: {model_info['depth']}")
        print(f"   Number of attention heads: {model_info['num_heads']}")
        print(f"   Parameter count: {model_info['params']}")
        print(f"   Interaction layer indices: {interaction_indexes}")
        self.dinov3_backbone = load_dinov3_model(self.model_name,self.pretrained_path)
        self._hidden_size = self.dinov3_backbone.embed_dim
        print(f"   _hidden_size  is...............: {self._hidden_size}")
        self.vision_tower = self.dinov3_backbone
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.image_processor = AutoImageProcessor.from_pretrained(self._vision_tower_name)  #直接用modelscope里面的imageprocessor
        self.is_loaded = True


    @property
    def image_size(self):
        return self._image_size

    def feature_select(self, outputs):
        sequence_output = outputs["last_hidden_state"]  # [B, seq_len, hidden_size]

        if self.select_feature == 'cls_patch':
            image_features = sequence_output
        elif self.select_feature == 'patch':
            image_features = sequence_output[:, 1:]
        elif self.select_feature == 'cls':
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


    # Layer 0 feat shape: torch.Size([8, 196, 768]), cls shape: torch.Size([8, 768]), dtype: torch.float32
    # Layer 1 feat shape: torch.Size([8, 196, 768]), cls shape: torch.Size([8, 768]), dtype: torch.float32
    # Layer 2 feat shape: torch.Size([8, 196, 768]), cls shape: torch.Size([8, 768]), dtype: torch.float32
    # Layer 3 feat shape: torch.Size([8, 196, 768]), cls shape: torch.Size([8, 768]), dtype: torch.float32

    def _forward(self, images):
        # 1) 统一使用 bf16 / float32 的自动混合精度上下文
        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                # 2) 获取中间层输出
                all_layers = self.vision_tower.get_intermediate_layers(
                    images, n=self.interaction_indexes, return_class_token=True
                )
                aligned_layers = []  #对每层进行投影与网格对齐
                # 3) 打印/调试信息（可删）
                for i, layer_out in enumerate(all_layers):
                    if isinstance(layer_out, tuple):
                        feat, cls = layer_out
                        feat = feat.to(images.dtype)
                        cls =  cls.to(images.dtype)
                        #print(f"Layer {i} feat shape: {feat.shape}, cls shape: {cls.shape}, dtype: {feat.dtype}")
                    else:
                        feat, cls = layer_out, None
                        feat = feat.to(images.dtype)
                        cls =  cls.to(images.dtype)
                        #print(f"Layer {i} output shape: {layer_out.shape}, dtype: {layer_out.dtype}")

                    if feat.dim() == 4:
                        B, C, H, W = feat.shape
                        feat = feat.view(B, C, H * W).permute(0, 2, 1)  # [B, T, C]
                    elif feat.dim() == 3:
                    # 已是 [B, T, C]
                        pass
                    else:
                        raise ValueError(f"Unsupported feat shape: {feat.shape}")
                    B, T, C = feat.shape

                    # 4b) 将通道投影到目标 embed_dim
                    if C != self.target_embed_dim:
                        # 使用一个共享投影头；若未初始化则创建并注册为子模块
                        if not hasattr(self, "_shared_proj_head"):
                            self._shared_proj_head = torch.nn.Linear(C, self.target_embed_dim, bias=True).to(images.device)
                        feat_proj = self._shared_proj_head(feat.to(images.dtype))  # [B, T, D]
                        feat_proj = feat_proj.to(images.dtype) 
                    else:
                        feat_proj = feat  # [B, T, D]
                        feat_proj = feat_proj.to(images.dtype) 

                    if feat_proj.shape[1] != self.target_N :
                    # 1D 插值，将序列长度从 T 调整到 target_N
                        D = feat_proj.shape[1]
                        feat_interp = torch.nn.functional.interpolate(
                            feat_proj.permute(0, 2, 1),  # [B, D, T]
                            size=self.target_N ,
                            mode="linear",
                            align_corners=False
                        ).permute(0, 2, 1).contiguous()  # [B, target_N, D]
                        feat_proj = feat_interp
                        
                    if cls is not None:
                    # 将 cls 投影到 target_embed_dim (如果需要的话，但通常 DINOv3 C=D)
                        cls_proj = cls.unsqueeze(1) # [B, 1, D]
                    # 拼接到 Patch Tokens 之前
                    feat_proj = torch.cat([cls_proj, feat_proj], dim=1)
                    aligned_layers.append(feat_proj.to(images.dtype) )  # [B, target_N, D]

                allintermidieaidfeatures = torch.cat(aligned_layers, dim=1)  
                #print(f" all patch features whoese output shape is ......................: {allintermidieaidfeatures.shape}, dtype: {allintermidieaidfeatures.dtype}")
                #print(f" aligned_layers[-1] output shape is ......................: {aligned_layers[-1].shape}, dtype: {aligned_layers[-1].dtype}")
                if self._num_patches_cached is None:
                    seq_len = aligned_layers[-1].shape[1]
                    self._num_patches_cached = seq_len - 1  # 一个图像被分解出的 patch 数量，不包括 CLS

                return aligned_layers[-1] , allintermidieaidfeatures
  

    @property
    def num_patches(self):
        if self._num_patches_cached is None:
            raise RuntimeError("Model not forwarded yet, num_patches unknown.")
        return self._num_patches_cached

    @property
    def num_patches_per_side(self):
        return int(self.num_patches ** 0.5)

    @property
    def patch_size(self):
        return self.vision_tower.patch_size

    @property
    def layer_count(self):
        return len(self.interaction_indexes)

    @property
    def hidden_size(self):
        return self._hidden_size
