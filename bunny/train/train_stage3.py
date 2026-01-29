import os
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Optional

import torch

import transformers

from bunny.train.bunny_trainer import BunnyTrainer

from bunny import conversation as conversation_lib
from bunny.model import *
from bunny.util.data_utils import make_supervised_data_module, DataArguments


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)  #选择何种LLM
    version: Optional[str] = field(default=None)  #选择何种对话模版
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)  
    vision_tower: Optional[str] = field(default=None)
    unfreeze_vision_tower: bool = field(default=False)
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
    mm_vision_select_feature: Optional[str] = field(default="patch")
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
        default=512,
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


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.util.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def checkpoint_has_trainer_state(checkpoint_dir):
    return os.path.exists(os.path.join(checkpoint_dir, "trainer_state.json"))





def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    完整的、暴力可靠的权重保存函数。
    逻辑：
    1. 预训练阶段：自动抓取所有 requires_grad=True 的参数（含投影层和自定义融合层）。
    2. SFT 阶段：调用官方逻辑保存全量模型。
    """
    
    # 检查当前是否为“只练适配器”的预训练模式
    is_pretraining = getattr(trainer.args, "tune_mm_mlp_adapter", False)

    # ==========================================================
    # 场景 A: 预训练/对齐阶段 (只存增量参数)
    # ==========================================================
    if is_pretraining:
        if trainer.args.local_rank <= 0:
            print(f"\n[System] 启动暴力扫描保存模式...")

        # 暴力扫描：直接搜寻模型中所有开启了梯度的参数
        weight_to_save = {}
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                # 兼容 DeepSpeed Zero2/Zero3，确保拿到 CPU 上的数据
                clean_data = torch.nan_to_num(param.data.detach().cpu(), nan=0.0, posinf=65500, neginf=-65500)
                weight_to_save[name] = clean_data.cpu()
              

        # 主进程负责物理写入磁盘
        if trainer.args.local_rank <= 0:
            # 1. 保存模型配置 (config.json)
            trainer.model.config.save_pretrained(output_dir)
            
            # 2. 保存增量权重 (mm_projector.bin)
            save_path = os.path.join(output_dir, "mm_projector.bin")
            torch.save(weight_to_save, save_path)
            
            # 3. 打印统计报告，确认是否漏掉 key
            vt_count = sum(1 for k in weight_to_save.keys() if 'vision_tower' in k)
            pj_count = sum(1 for k in weight_to_save.keys() if 'mm_projector' in k)
        # 预训练模式任务完成，直接返回，不再执行后续全量保存
        return

    # ==========================================================
    # 场景 B: 全量微调阶段 (SFT) 或 其它模式
    # ==========================================================
    
    # 兼容用户可能需要的独立 Vision Tower 保存开关
    if getattr(trainer.args, "save_mm_vision_tower", False):
        # 即使在全量微调，也可以单独拎出一份视觉塔权重
        vt_weights = {n: p.data.cpu() for n, p in trainer.model.named_parameters() if 'vision_tower' in n}
        if trainer.args.local_rank <= 0:
            torch.save(vt_weights, os.path.join(output_dir, 'vision_tower_standalone.bin'))

    # 执行 HuggingFace 官方的全量保存逻辑（保存数 GB 的 pytorch_model.bin）
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
    else:
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)





def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """
    智能调整词表，解决 Phi-1.5 的 ID 冲突问题
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    # 关键：检查 <image> 是否还在 50256 这个坑里
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    if image_token_id == tokenizer.eos_token_id:
        print(f"🚨 警告: <image> 与 EOS 冲突 (ID: {image_token_id})。正在强制重分配...")
        # 强制添加一个独立的 Token，这将占用 50296 或更高
        # 注意：这里我们不需要再加 add_special_tokens，因为上面可能已经加了
        # 但为了保险，我们使用 add_tokens 强制分配新位置
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        new_id = tokenizer.convert_tokens_to_ids("<image>")
        print(f"✅ <image> 已迁移至新 ID: {new_id}")
    
    # 调整模型 Embedding 大小
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        # 确保新 Token 的 Embedding 均值初始化，而不是随机初始化（有助于收敛）
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
    return tokenizer



def train():
    global local_rank

    # 1. 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # 自动推断计算精度 (FP16/BF16/FP32)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # =========================================================
    # 2. BitsAndBytes 配置 (虽然 Stage 3 推荐全量，但保留逻辑以防万一)
    # =========================================================
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    # =========================================================
    # 3. Tokenizer 初始化 (完整保留原逻辑，处理特殊Token)
    # =========================================================
    assert model_args.vision_tower is not None
    # 根据模型类型选择加载方式
    if model_args.model_type in {'phi-1.5', 'phi-2', 'phi-3', 'qwen1.5-1.8b', 'minicpm', 'llama3-8b'}:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    elif model_args.model_type == 'stablelm-2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )

    # 补全 Token
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # Llama3 特殊处理
    if model_args.model_type == 'llama3-8b':
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token 

    # =========================================================
    # 4. 模型加载 (完整保留原逻辑，支持多种架构)
    # =========================================================
    if model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2':
        model = BunnyPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")



    # 关闭 Cache 以节省显存
    model.config.use_cache = False

    # =========================================================
    # 5. Stage 3 核心修改：全量解冻 (覆盖 freeze_backbone 参数)
    # =========================================================
    # 无论 model_args.freeze_backbone 是什么，Stage 3 强制解冻
    rank0_print("🔥 [Stage 3] 强制执行全量解冻策略 (Ignore freeze_backbone=True)...")
    model.requires_grad_(True) 

    # 处理量化训练的梯度准备 (如果开启 bits)
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 开启梯度检查点 (显存优化)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # =========================================================
    # 6. LoRA 初始化 (如果开启)
    # =========================================================
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # =========================================================
    # 7. 对话模板设置
    # =========================================================
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["default"]

    # 模板自检日志
    if local_rank == 0:
        rank0_print(f"🔍 激活模板: {model_args.version}")
        rank0_print(f"🔥 Roles: {conversation_lib.default_conversation.roles}")
        rank0_print(f"🔥 Sep: {repr(conversation_lib.default_conversation.sep)}")

    # =========================================================
    # 8. 视觉模块初始化 (必须调用，否则没有 vision_tower)
    # =========================================================
    rank0_print("👁️ 初始化视觉模块 (Vision Modules)...")
    model.get_model().initialize_vision_modules(model_args=model_args)
    
    # 调整 Token Embedding 大小
    model.resize_token_embeddings(len(tokenizer))

    # =========================================================
    # 9. 🛡️ 词表安全对齐 (防止梯度爆炸)
    # =========================================================
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight
    current_size = input_embeddings.shape[0]
    SAFE_VOCAB_SIZE = 50257 # Phi-2 的原始词表大小
    
    if current_size > SAFE_VOCAB_SIZE:
        rank0_print(f"🚨 词表对齐: {SAFE_VOCAB_SIZE} -> {current_size}")
        with torch.no_grad():
            in_avg = input_embeddings[:SAFE_VOCAB_SIZE].mean(dim=0, keepdim=True)
            out_avg = output_embeddings[:SAFE_VOCAB_SIZE].mean(dim=0, keepdim=True)
            input_embeddings[SAFE_VOCAB_SIZE:] = in_avg
            output_embeddings[SAFE_VOCAB_SIZE:] = out_avg
        rank0_print("✅ 新增 Token 已用均值初始化。")

    # =========================================================
    # 10. 🛡️ 参数 NaN/Inf 清洗
    # =========================================================
    rank0_print("🛡️ 执行参数坏点清洗...")
    for p in model.parameters():
        if p.requires_grad:
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=65500, neginf=-65500)

    # =========================================================
    # 11. 视觉塔配置与解冻 (Stage 3 关键)
    # =========================================================
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=compute_dtype, device=training_args.device)

    # 将配置同步到 data_args 和 model.config (为了保存)
    data_args.image_processor = vision_tower.image_processor
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    # 记录训练参数到 config
    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.use_s2 = model_args.use_s2
    model.config.unfreeze_vision_tower = model_args.unfreeze_vision_tower
    model.config.vision_tower_dino = model_args.vision_tower_dino
    model.config.vision_tower_siglip = model_args.vision_tower_siglip
    model.config.mm_projector_type = model_args.mm_projector_type
    model.config.model_type = model_args.model_type
    model.config.lora_enable = training_args.lora_enable

    # --- Stage 3 解冻逻辑 ---
    # 1. 解冻 Projector (必然)
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True
    
    # 2. 解冻 Vision Tower (Stage 3 必须)
    if model_args.unfreeze_vision_tower:
        rank0_print("🔥 [Stage 3] 解冻 Vision Tower 全量参数...")
        for p in model.get_model().vision_tower.parameters():
            p.requires_grad = True
        model.get_model().vision_tower.train() # 开启 Train 模式 (Dropout/BN 生效)
    
    # 3. 处理 LoRA 层的精度转换 (防止报错)
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # =========================================================
    # 12. 数据加载 (完整逻辑)
    # =========================================================
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # =========================================================
    # 13. Trainer 初始化
    # =========================================================
    trainer = BunnyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # 参数状态检查日志
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        rank0_print("\n" + "="*60)
        rank0_print("🔍 [Stage 3] 最终参数解冻状态检查")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        vt_active = any("vision_tower" in n for n in trainable_names)
        pj_active = any("mm_projector" in n for n in trainable_names)
        llm_active = any("layers" in n for n in trainable_names)
        
        rank0_print(f"   - Vision Tower Active: {vt_active}")
        rank0_print(f"   - Projector Active:    {pj_active}")
        rank0_print(f"   - LLM Backbone Active: {llm_active}")
        rank0_print(f"📊 总可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        rank0_print("="*60 + "\n")

    # =========================================================
    # 14. 训练执行 (支持断点续训)
    # =========================================================
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        latest_ckpt = str(sorted(checkpoints)[-1])
        if checkpoint_has_trainer_state(latest_ckpt):
            rank0_print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
            trainer.train(resume_from_checkpoint=latest_ckpt)
        else:
            rank0_print(f"⚠️ Checkpoint 损坏，重新开始训练。")
            trainer.train()
    else:
        rank0_print("🚀 Starting training from scratch.")
        trainer.train()

    # 保存最终状态
    trainer.save_state()
    
    # 开启 Cache 以便推理
    model.config.use_cache = True

    # =========================================================
    # 15. 最终全量保存 (Stage 3 核心)
    # =========================================================
    if training_args.local_rank <= 0:
        rank0_print("📢 [Stage 3] 训练结束，开始执行全量保存...")

        # 如果用了 LoRA，必须合并
        if training_args.lora_enable:
            rank0_print("📎 Merging LoRA weights back to base model...")
            model = model.merge_and_unload()
            model.config.lora_enable = False # 更新配置

        # 强制全量保存 (Config + Weights + Tokenizer)
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        
        # 确保 generation_config 也被保存
        if hasattr(model, "generation_config"):
            model.generation_config.save_pretrained(training_args.output_dir)

        rank0_print(f"✅ 全量模型已完整保存至: {training_args.output_dir}")

if __name__ == "__main__":
    train()
