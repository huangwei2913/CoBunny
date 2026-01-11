import os
import warnings
import torch

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, logging

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

from bunny.model import *


def load_pretrained_model(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", **kwargs):
    
    # --- 1. 初始化配置 ---
    is_lora = 'lora' in model_name.lower()
    # 如果是 LoRA，必须提供 model_base；如果不是 LoRA，直接从 model_path 加载全量权重
    base_path = model_base if is_lora else model_path
    
    if is_lora and base_path is None:
        raise ValueError("❌ 错误：检测到 LoRA 模型，但未提供 model_base 路径！")

    print(f"🚀 加载模式: {'LoRA 叠加' if is_lora else '全量权重'}")
    print(f"📦 基础路径 (Base): {base_path}")
    print(f"📂 权重路径 (Weight): {model_path}")

    # --- 2. 硬件与精度适配 (针对 Tesla T4) ---
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        # T4 不支持 bf16，这里必须强制用 fp16
        kwargs['torch_dtype'] = torch.float16

    # --- 3. 实例化架构 (根据 model_type) ---
    # LoRA 模式下需要读取微调目录的 config
    cfg_pretrained = AutoConfig.from_pretrained(model_path if is_lora else base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True, trust_remote_code=True)

    print(f"🏗️ 正在构建 {model_type} 架构...")
    if model_type in ['phi-1.5', 'phi-2']:
        model = BunnyPhiForCausalLM.from_pretrained(base_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    elif model_type == 'phi-3':
        model = BunnyPhi3ForCausalLM.from_pretrained(base_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    elif model_type == 'stablelm-2':
        model = BunnyStableLMForCausalLM.from_pretrained(base_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    elif model_type == 'qwen1.5-1.8b':
        model = BunnyQwen2ForCausalLM.from_pretrained(base_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    elif model_type == 'minicpm':
        model = BunnyMiniCPMForCausalLM.from_pretrained(base_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    elif model_type == 'llama3-8b':
        model = BunnyLlamaForCausalLM.from_pretrained(base_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown Model Type {model_type}")

    # --- 4. 视觉塔初始化与 BF16 安全转换 (Tesla T4 救命逻辑) ---
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    
    print("🛡️ 正在执行 BF16 -> FP16 安全截断 (防止 T4 溢出 NaN)...")
    with torch.no_grad():
        for name, param in vision_tower.named_parameters():
            if param.dtype == torch.bfloat16:
            # 针对 Oryx (BF16) 进行截断处理，确保数值在 FP16 的 65504 范围内
                param.data = param.data.clamp(min=-65500, max=65500).to(torch.float16)
    
    vision_tower.to(device=device, dtype=torch.float16)

    # --- 5. 权重合并与注入 ---
    if is_lora:
        # A. 加载并合并 LoRA
        from peft import PeftModel
        print('🧪 合并 LoRA 适配器...')
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        
        # B. 注入 Non-LoRA 训练参数 (那 119 个参数)
        non_lora_bin = os.path.join(model_path, 'non_lora_trainables.bin')
        if os.path.exists(non_lora_bin):
            print('🔥 注入 Non-LoRA 融合层权重...')
            non_lora_trainables = torch.load(non_lora_bin, map_location='cpu')
            
            target_dict = model.state_dict()
            cleaned_weights = {}
            for k, v in non_lora_trainables.items():
                temp_k = k
                while temp_k not in target_dict and '.' in temp_k:
                    temp_k = temp_k.split('.', 1)[1]
                
                if temp_k in target_dict:
                    # 注入时同样进行安全截断，防止 Non-LoRA 参数里也有 BF16 遗毒
                    cleaned_weights[temp_k] = v.clamp(min=-65500, max=65500).to(torch.float16)
            
            model.load_state_dict(cleaned_weights, strict=False)
    else:
        # 如果不是 LoRA，且有 projector.bin (Stage 1)，手动加载
        projector_bin = os.path.join(model_path, 'mm_projector.bin')
        if os.path.exists(projector_bin):
            print('📽️ 加载 Projector 权重...')
            weights = torch.load(projector_bin, map_location='cpu')
            safe_weights = {
                k: v.clamp(min=-65500, max=65500).to(torch.float16) 
                for k, v in weights.items()
            }
            model.load_state_dict(safe_weights, strict=False)

    # --- 6. 最终兜底扫描 ---
    with torch.no_grad():
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                param.data.nan_to_num_(nan=0.0)
    
    # --- 7. 配置对齐 ---
    model.resize_token_embeddings(len(tokenizer))
    context_len = getattr(model.config, "max_sequence_length", 2048)
    image_processor = vision_tower.image_processor
    if model.generation_config is None:
        from transformers import GenerationConfig
        # 强制根据 config 创建生成配置，这会激活 generate() 内部的全部功能
        model.generation_config = GenerationConfig.from_model_config(model.config)
    print("✅ 加载完成！模型现在可以安全地在 T4 上进行推理。")
    return tokenizer, model, image_processor, context_len