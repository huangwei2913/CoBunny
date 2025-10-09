import os
from .eva_clip.eva_clip_encoder import EvaClipVisionTower
from .siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
from .clip.clip_encoder import CLIPVisionTower
from .dfn_clip_encoder import DfnClipVisionTower
import logging
from .oryx_vit import OryxViTWrapper

#要明确知道每一个视觉编码器的输出
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    use_s2 = getattr(vision_tower_cfg, 'use_s2', False)

    if 'sig' in vision_tower.lower():
        if use_s2:
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'eva' in vision_tower.lower():
        if use_s2:
            raise ValueError(f'Currently not supporting S2 for EVA-CLIP')
        else:
            return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif "apple/dfn" in vision_tower.lower():
        logging.info(f"Loading **Apple DFN CLIP** Vision Tower: {vision_tower}")
        return DfnClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    elif "oryx_vit" in vision_tower:
        path = vision_tower.split(":")[1]
        return OryxViTWrapper(vision_tower, path=path, args=vision_tower_cfg, **kwargs)

    elif 'clip' in vision_tower.lower():
        if use_s2:
            raise ValueError(f'Currently not supporting S2 for CLIP')
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
