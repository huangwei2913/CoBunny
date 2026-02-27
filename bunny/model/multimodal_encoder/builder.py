import os
from .eva_clip.eva_clip_encoder import EvaClipVisionTower
from .siglip.siglip_encoder import SiglipVisionTower
from .clip.clip_encoder import CLIPVisionTower
from .dfn_clip_encoder import DfnClipVisionTower
from .dino_encoder import DinoVisionTower
from .midas_encoder import MiDaSVisionTower
import logging
from .oryx_vit import OryxViTWrapper
from .AdaptiveConcatenationVisionTower import AdaptiveConcatenationVisionTower

#要明确知道每一个视觉编码器的输出
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    use_s2 = getattr(vision_tower_cfg, 'use_s2', False)
    if 'sig' in vision_tower.lower():
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)  
    elif 'mixedencoder' in vision_tower.lower():
        return AdaptiveConcatenationVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
