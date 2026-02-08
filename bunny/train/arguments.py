from dataclasses import dataclass, field
from typing import Optional
import transformers  # 加上这一行
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)  #选择何种LLM
    version: Optional[str] = field(default=None)  #选择何种对话模版
    freeze_backbone: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)  #以前是在模型参数部分的，现在要搞出来  
    vision_tower: Optional[str] = field(default=None)
    use_s2: bool = field(default=False)  #是否使用S2
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')  #这个参数非常重要，它会指导如何建立投影层网络结构
    mm_resampler_type: Optional[str] = field(default=None) #采用何种重采样器
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    tune_mm_vision_resampler: bool = field(default=False)    
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_dense_connector_type: Optional[str] = field(default='dci')  #密集投影层类型
    vision_tower_dino: Optional[str] = field(default=None, metadata={"help": "DINOv2 子塔的权重路径"})
    vision_tower_siglip: Optional[str] = field(
        default=None, metadata={"help": "SigLIP 子塔的权重路径，例如 google/siglip-so400m-patch14-384"}
    )
    compression_K: int = field(default=8, metadata={"help": "ToMe 算法的压缩倍率"})
    mm_hidden_size: int = field(default=1024)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    save_mm_vision_tower: bool = field(default=False) #增加一个是否保留视觉塔模型部分的参数
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


