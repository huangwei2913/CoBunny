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
from .arguments import TrainingArguments,ModelArguments

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)



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



def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))


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

    #跟序列的最大化长度相关，这里的padding同样最大长度max_length=10，输入7个token：,也就说model_max_length表示token的最大长度？？
    #当你输入的句子长度不足模型最大长度max_length时，需要用特殊的填充标记[PAD]把序列补齐到相同长度。这样，可以批量处理不等长的序列。
    assert model_args.vision_tower is not None
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

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.model_type == 'llama3-8b':
        tokenizer.eos_token_id = 128001  #该值不是随意确定的，而是对应模型词表中定义的特殊结束token。对于Llama3-8b模型，这个特殊token的id就是128001（根据模型词表和官方说明）。
        tokenizer.pad_token = tokenizer.eos_token 

    #看一下训练的时候，如何替代这些模型，任务13，非常重要，每一个模型都是多模态模型，因此，每一个模型都实现了类似于get_model().initialize_vision_modules(）
    #之类的函数，调用和得到对应的视觉编码器模块，重要的任务是在这里添加视觉或者模型块
    if model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2':
        model = BunnyPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'phi-3':
        model = BunnyPhi3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'stablelm-2':
        model = BunnyStableLMForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'qwen1.5-1.8b':
        model = BunnyQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'minicpm':
        model = BunnyMiniCPMForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_type == 'llama3-8b':
        model = BunnyLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.use_cache = False

    if model_args.freeze_backbone:   #是否冻结骨干
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()  #这是模型提供的一个方法，用来开启输入embedding层张量的requires_grad=True，允许对输入做梯度追踪。
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)  #这是模型中获取输入嵌入层（embedding layer）的接口，返回模型输入embedding模块，通常是一个nn.Embedding层


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
        # ---------------------------------------------------------
        # 🌟 关键加固：强制激活 LoRA 层梯度
        # ---------------------------------------------------------
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True # 确保 LoRA 层必开
            elif "mm_projector" in name:
                param.requires_grad = True # 确保投影层也必开
        # ---------------------------------------------------------

        # 打印一下，验证给学术论文看
        model.print_trainable_parameters()       


    #这段代码的作用正是为加载的大语言模型（LLM）选择对应的对话（聊天）模板
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["default"]



    # --- 在它下面插入这几行调试代码 ---
    rank0_print(f"\n" + "="*40)
    rank0_print(f"🔍 正在自检模板对齐情况...")
    rank0_print(f"🔥 命令行传入的 version: {model_args.version}")
    template_name = getattr(conversation_lib.default_conversation, 'version', 
                            getattr(conversation_lib.default_conversation, 'name', 'Unknown'))
    rank0_print(f"🔥 实际激活的模板名称: {template_name}")
    rank0_print(f"🔥 角色设定 (Roles): {conversation_lib.default_conversation.roles}")
    rank0_print(f"🔥 分隔符 (Sep): {repr(conversation_lib.default_conversation.sep)}")
    
    # 打印一个真实的预览，看看图片占位符和文字是怎么拼接的
    test_prompt = conversation_lib.default_conversation.get_prompt()
    rank0_print(f"🔥 模板预览:\n{test_prompt}")
    rank0_print("="*40 + "\n")


    model.get_model().initialize_vision_modules(model_args=model_args)
    model.resize_token_embeddings(len(tokenizer))

    # 2. 🛡️【核心修复】手动计算老词的均值，填补给新词
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight
    # 计算老词（原生 50257 个词）的平均值
    # 这样新词就长得像老词一样，不会惊吓到模型
    current_size = input_embeddings.shape[0]
    SAFE_VOCAB_SIZE = 50257
    if current_size > SAFE_VOCAB_SIZE:
        rank0_print(f"🚨 检测到词表差异！当前: {current_size}, 原生安全区: {SAFE_VOCAB_SIZE}")
        with torch.no_grad():
            # 计算原生词表的均值
            in_avg = input_embeddings[:SAFE_VOCAB_SIZE].mean(dim=0, keepdim=True)
            out_avg = output_embeddings[:SAFE_VOCAB_SIZE].mean(dim=0, keepdim=True)
            # 【关键操作】：把 50257 之后的所有位置（不管是 38 个还是 900 个）全部初始化
            input_embeddings[SAFE_VOCAB_SIZE:] = in_avg
            output_embeddings[SAFE_VOCAB_SIZE:] = out_avg
            
        rank0_print(f"✅ 已清理并初始化 {current_size - SAFE_VOCAB_SIZE} 个潜在危险槽位。")



    if training_args.local_rank == 0:
        print("✅ 已手动初始化新增 Token！梯度爆炸隐患已清除。")

    # 3. 🛡️【双重保险】把所有参数强制转为 float32 进行一次清洗，再转回 float16
    # 这能保证即便刚才 resize 产生了细微的 NaN，也被洗掉了
    for p in model.parameters():
        if p.requires_grad:
            # 只处理参与训练的参数
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=65500, neginf=-65500)
    # ...
    #################### ⭐️ 插入调试代码 ⭐️ ####################
    if model_args.pretrain_mm_mlp_adapter:
        rank0_print("Checking mm_projector parameters after loading pretrain weights...")

        # 假设 mm_projector 至少有一个权重层 (比如 weight)
        mm_projector_first_weight = model.get_model().mm_projector.parameters().__next__()

        # 尝试计算该权重的L2范数或某个统计量，证明它不是随机初始化
        # 注意：这只在 local_rank 0 上安全，因为它需要同步
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            try:
                # 检查权重的范数，如果是一个加载的权重，它的值应该是非零且非极小的
                weight_norm = torch.linalg.norm(mm_projector_first_weight).item()
                rank0_print(f"✅ mm_projector first weight norm: {weight_norm:.4f}")
                if weight_norm < 1.0: # 经验值，加载的权重通常不会这么小
                    rank0_print("⚠️ Warning: Weight norm seems very small, check if weights were correctly loaded.")
            except Exception as e:
                rank0_print(f"❌ Error checking mm_projector weight norm: {e}")

    # ... (继续后面的 vision_tower.to(...) 等代码)
    ####################应该是在这里添加视觉编码器？？？？？    
    vision_tower = model.get_vision_tower()
    #设备移动：模型必须移动到指定的训练设备（通常是GPU），否则计算无法加速。
    # 该调用确保vision_tower使用正确的硬件资源和数据格式，为训练或推理做准备。
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    #的主要作用是实现微调时只训练模型中视觉多模态MLP适配器（mm_projector）部分，而冻结模型其余参数。具体含义说明如下
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        if not training_args.lora_enable:
            print("❄️ [System] 全量冻结 Backbone，仅练 Projector...")
            model.requires_grad_(False)
        else:
            print("🚀 [System] 检测到 LoRA 已开启，仅冻结非 LoRA 的 LLM 权重...")
            # 这种情况下不需要 model.requires_grad_(False)，因为 get_peft_model 内部已经处理好了
            pass
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        rank0_print("🔥 [Custom] Unfreezing AdaptiveConcatenationVisionTower fusion layers...")
        if hasattr(model.get_model(), "vision_tower"):
            print("🔥 Unfreezing custom fusion layers in Vision Tower...")

            rank0_print("🔥 [Custom] 正在精准解冻混合视觉塔融合层...")
            v_tower = model.get_model().get_vision_tower() # 使用 getter 比较安全
            for name, p in v_tower.named_parameters():
                if any(k in name for k in ['mlp_layers', 'cross_attn', 'cls_weights', 'pseudo', 'score_predictor']):
                    p.requires_grad = True
                    print(f"   -> Unfrozen: {name}")


    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_projector_lr = training_args.mm_projector_lr

    model.config.use_s2 = model_args.use_s2

    #model.config.unfreeze_vision_tower = training_args.unfreeze_vision_tower = model_args.unfreeze_vision_tower
    #if training_args.unfreeze_vision_tower:
    #    for p in model.get_model().vision_tower.parameters():
    #        p.requires_grad = True


    if model_args.unfreeze_vision_tower:
            print("--- 🚀 尝试解冻视觉编码器参数 (Recipe-2) ---")
            # 实际解冻逻辑
            vision_tower = model.get_model().vision_tower
            for name, p in vision_tower.named_parameters():
                p.requires_grad = True
                


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


    ''' 
        #设置数据处理模块,这一部分是为了训练的时候，使用相关bunny数据集的
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                data_args=data_args)

        # 2. 从训练集中切出一小部分作为验证集 (例如 2000 条，足够反映收敛情况)
        full_train_dataset = data_module['train_dataset']
        num_val_samples = 2000 
        num_train_samples = len(full_train_dataset) - num_val_samples

        # 使用 torch.utils.data.random_split 进行随机切分
        train_dataset, eval_dataset = torch.utils.data.random_split(
                                            full_train_dataset, 
                                            [num_train_samples, num_val_samples],
                                            generator=torch.Generator().manual_seed(42) # 固定随机种子，确保多机训练时行为一致
                                            )

        # 3. 更新 data_module
        data_module['train_dataset'] = train_dataset
        data_module['eval_dataset'] = eval_dataset
            
    ''''''''' 

    # 1. 直接调用修改后的函数，它会一次性返回切分好的训练集和验证集,这个是为了使用那个sharegpt4v的
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args
    )

    # 2. 原本在 train.py 里的 random_split 逻辑全部删掉
    # 因为我们在 data_utils.py 内部已经处理好了属性透传

    model.config.vision_tower_dino = model_args.vision_tower_dino
    model.config.vision_tower_siglip = model_args.vision_tower_siglip
    model.config.mm_projector_type = model_args.mm_projector_type
    model.config.model_type = model_args.model_type
    # 额外建议：把 lora_enable 也同步进去，虽然保存时我们会强制改它
    model.config.lora_enable = training_args.lora_enable

    #   返回dict(train_dataset=train_dataset,
    #            eval_dataset=None,
    #            data_collator=data_collator)

    #可以把data_collator看成是批整合器，把LazySupervisedDataset看成是，也就是train_dataset这个对象看成是如何每次训练获取样本的集中管理器
    #开启训练过程
    trainer = BunnyTrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)
    

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        rank0_print("\n" + "="*80)
        rank0_print("🔍 [Parameter Check] 正在扫描可训练参数...")
        rank0_print(f"{'参数名称':<60} | {'形状':<20} | {'梯度'}")
        rank0_print("-" * 95)
        
        # 1. 修复：必须先初始化计数器
        trainable_params_count = 0
        trainable_params = []
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)
                trainable_params_count += 1
                # 2. 修复：变量名统一使用 p，而不是 param
                shape_str = str(list(p.shape))
                rank0_print(f"{name:<60} | {shape_str:<20} | {p.requires_grad}")
        
        rank0_print("-" * 95)
        rank0_print(f"📊 总计可训练参数项: {trainable_params_count}")
        
        # --- 逻辑验证 ---
        vision_tower_params = [n for n in trainable_params if "vision_tower" in n]
        projector_params = [n for n in trainable_params if "mm_projector" in n]
        
        fusion_found = len(vision_tower_params) > 0
        projector_found = len(projector_params) > 0
        
        if fusion_found and projector_found:
            rank0_print("🚀 状态确认：混合编码器融合层 和 Projector 已全部解冻！")
        else:
            if not fusion_found:
                rank0_print("❌ 警告：未发现 vision_tower 的可训练参数，请检查解冻逻辑！")
            if not projector_found:
                rank0_print("❌ 警告：未发现 mm_projector 的可训练参数！")
        
        rank0_print(f"🔍 逻辑明细：")
        rank0_print(f"   - 混合塔内部参数 (vision_tower): {len(vision_tower_params)} 项")
        rank0_print(f"   - 外部连接投影器 (mm_projector): {len(projector_params)} 项")
        rank0_print("="*80 + "\n")

# ==================== 🔍 更加稳健的自检 Debug 代码 ====================
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print("\n" + "="*50)
        print("🚀 [Debug] 正在抽样检查喂给模型的数据格式...")
        
        try:
            # 获取一个 batch
            sample_batch = next(iter(trainer.get_train_dataloader()))
            
            # 1. 获取 Input IDs 并移至 CPU 转换为 list
            input_ids = sample_batch['input_ids'][0].detach().cpu().tolist()
            print(f"👉 [Input IDs 前 10 个 Token]: {input_ids[:10]}")
            if any(tid < 0 for tid in input_ids):
                print(f"✅ 发现特殊的 Image Token Index!")
            else:
                print(f"⚠️ 警告：Input IDs 里全是正数，说明 <image> 没被正确转换成特殊索引！")
            # 2. 检查 Labels 并处理 -100
            labels = sample_batch['labels'][0].detach().cpu().tolist()
            
            # 找到非 -100 的部分（即模型真正学习的部分）
            # 我们把 -100 过滤掉，或者替换成一个可见字符
            filtered_input_ids = [tid for tid in input_ids if tid >= 0]
            decoded_text = tokenizer.decode(filtered_input_ids, skip_special_tokens=False)

            # 找到模型计算 Loss 的部分
            loss_mask_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
            decoded_loss_part = tokenizer.decode(loss_mask_tokens, skip_special_tokens=False)

            print(f"\n👉 [完整输入流解码] (含 Image Token 占位符):\n{decoded_text[:1000]}") # 截断前1000字符防止刷屏
            print(f"\n👉 [计算 Loss 的文本内容]:\n{decoded_loss_part}")
            
            if 'images' in sample_batch:
                print(f"\n👉 [图像 Tensor 形状]: {sample_batch['images'].shape}")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            import traceback
            print(f"❌ [Debug] 抽样检查依然失败: {e}")
            traceback.print_exc()
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        # 选最近的checkpoint
        latest_ckpt = str(sorted(checkpoints)[-1])
        if checkpoint_has_trainer_state(latest_ckpt):
            print(f"Resuming from checkpoint: {latest_ckpt}")
            trainer.train(resume_from_checkpoint=latest_ckpt)
        else:
            print(f"Checkpoint {latest_ckpt} missing trainer_state.json, training from scratch.")
            trainer.train()
    else:
        print("No checkpoint found, training from scratch.")
        trainer.train()   

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    # 2. 只在主进程 (Rank 0) 执行全量合并保存
    if training_args.local_rank <= 0:
        print("📢 [全量保存启动] 正在收集分布式权重并物理合并 LoRA...")

        # 如果模型是 PeftModel (开启了 LoRA)
        if training_args.lora_enable:
            # 核心逻辑：物理合并
            # merge_and_unload 会把 BA 矩阵加回 W，并返回一个正常的 BunnyPhiForCausalLM 对象
            model = model.merge_and_unload()
            
            # 强制更新 config，关闭推理时的 lora 搜索，因为它已经合进去了
            model.config.lora_enable = False
            
            # 此时的 model.state_dict() 已经包含了：
            # - 合并后的全量 LLM 权重
            # - 微调后的 Projector 权重 (无需手动 replace)
            # - 微调后的 Vision Tower 权重
            
            # 保存整个文件夹
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            
            print(f"✅ 全量模型已保存至: {training_args.output_dir}")
            print("ℹ️ 推理说明：直接使用 BunnyPhiForCausalLM.from_pretrained 加载此目录即可。")
        else:
            # 如果是全量微调，正常保存即可
            trainer.save_model()

if __name__ == "__main__":
    train()


