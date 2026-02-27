import torch

from .masked_drop import MaskedDrop
from .spatial_pool import SpatialPool
from .qformer import Qformer
from .vlm_attention import VlmAttention
from .perceiver import DynamicCompressor

class IdentityMap(torch.nn.Module):
    def __init__(self, *args, **kwargs): # 加上这个，吃掉所有传进来的指令
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}

def build_vision_resampler(model_args, **kwargs):
    resampler_type = getattr(model_args, 'mm_resampler_type', None)

    if resampler_type == 'masked_drop':
        return MaskedDrop(model_args)
    elif resampler_type == 'spatial_pool':
        return SpatialPool(model_args, **kwargs)
    elif resampler_type == 'qformer':
        return Qformer(model_args, **kwargs)
    elif resampler_type == 'vlm_attention':
        return VlmAttention(model_args, **kwargs)
    elif resampler_type == 'dynamic_compressor':
        return DynamicCompressor(model_args, **kwargs)
    elif resampler_type is None:
        # 现在的 IdentityMap 已经能吃下 kwargs 了
        return IdentityMap(**kwargs) 
    else:
        raise ValueError(f'Unknown resampler type: {resampler_type}')