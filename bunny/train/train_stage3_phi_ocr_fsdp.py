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
from arguments import ModelArguments,TrainingArguments
import json

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def checkpoint_has_trainer_state(checkpoint_dir):
    return os.path.exists(os.path.join(checkpoint_dir, "trainer_state.json"))

import re

def get_checkpoint_number(path):
    matches = re.findall(r"checkpoint-(\d+)", str(path))
    return int(matches[-1]) if matches else 0


def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
    
    # 定义全 FP16 策略，不给 FSDP 留任何 Upcast 到 FP32 的余地
    hardcore_fp16_policy = MixedPrecision(
        param_dtype=torch.float16,   # 权重强制 FP16
        reduce_dtype=torch.float16,  # 梯度规约强制 FP16
        buffer_dtype=torch.float16,  # 缓存强制 FP16
    )

    # 注入到 training_args 的 fsdp_config 中
    # 注意：这时候 training_args 已经存在了，这样改才能生效
    if training_args.fsdp_config is None:
        training_args.fsdp_config = {}
    
    training_args.fsdp_config["mixed_precision_policy"] = hardcore_fp16_policy
    
    model_args.unfreeze_mm_vision_tower = True
    training_args.freeze_mm_mlp_adapter = False
    import sys
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print("\n" + "=" * 50)
        print("🛠️ 原始命令行参数 (sys.argv):")
        print(sys.argv)
        print("-" * 50)
        print("📊 HfArgumentParser 解析结果:")
        v1 = getattr(model_args, 'unfreeze_mm_vision_tower', "MISSING")
        v2 = getattr(model_args, 'unfreeze_vision_tower', "MISSING")
        print(f">> model_args.unfreeze_mm_vision_tower: {v1} (Type: {type(v1)})")
        print(f">> model_args.unfreeze_vision_tower:    {v2} (Type: {type(v2)})")
        v4 = getattr(model_args, 'freeze_backbone', "MISSING")
        print(f">> model_args.freeze_backbone:    {v4} (Type: {type(v4)})")
        v3 = getattr(training_args, 'unfreeze_mm_vision_tower', "MISSING")
        print(f">> training_args.unfreeze_mm_vision_tower: {v3}")
        print("=" * 50 + "\n")
    model_args.freeze_backbone = False
    assert model_args.vision_tower is not None
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if model_args.model_type in ['phi-1.5', 'phi-2']:
        model = BunnyPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=compute_dtype,
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    model.get_model().initialize_vision_modules_stage3_fsdp(model_args=model_args)
    # ✨ 新增：强制全模型类型统一，解决 FP16/BF16 混合报错
    rank0_print(f"🧹 正在统一模型权重类型为.........................: {compute_dtype}...")
    model.to(torch.float16)

    rank0_print("🔥 [CRITICAL] 正在物理强制转换所有 LLM 层为 float16...")
    for name, param in model.named_parameters():
        param.data = param.data.to(torch.float16)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(torch.float16)

    # 3. 针对 Embedding 的特殊处理 (警告中经常提到的重灾区)
    model.get_input_embeddings().to(torch.float16)
    model.get_output_embeddings().to(torch.float16)    


    # 针对那些容易被忽略的 Buffer 也进行转换
    for module in model.modules():
        module.to(torch.float16)
    torch.cuda.empty_cache() # 清理转换过程产生的临时显存占用
    # =====================================================================
    # 🩹 [FSDP2 救命补丁]：强制解绑 Embedding 和 LM Head 权重
    # FSDP2 目前对 Tied Weights 支持有严重 Bug，会导致 device_mesh 报错。
    # 我们在这里通过 clone() 强行断开它们的内存共享，使其成为两个独立的 DTensor。
    # =====================================================================
    if getattr(model.config, "tie_word_embeddings", False):
        rank0_print("🔧 [FSDP2 Patch] 检测到 Tied Weights，正在强行解绑以兼容 FSDP2...")
        model.config.tie_word_embeddings = False
        
        # 获取输入和输出的 Embedding
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        
        if output_embeddings is not None and input_embeddings is not None:
            # 深拷贝一份权重给 LM Head，彻底切断物理联系
            output_embeddings.weight = torch.nn.Parameter(output_embeddings.weight.clone())
            rank0_print("✅ [FSDP2 Patch] LM Head 与 Embedding 解绑成功！")
    # =====================================================================

    rank0_print("🚀 [FSDP Mode] 正在解冻 15 亿参数全量梯度...")
    model.train()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()  
        model.enable_input_require_grads()  

    if not model_args.freeze_backbone:
        rank0_print("🚀 [FSDP Mode] 正在全量解冻 LLM 主干与视觉模块...")
        model.train()
        for param in model.parameters():
            param.requires_grad = True

    vision_tower = model.get_vision_tower() # 👈 所有进程都必须执行这一行
    vision_tower.to(torch.float16)
    if hasattr(vision_tower, 'image_processor'):
        data_args.image_processor = vision_tower.image_processor

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.use_s2 = model_args.use_s2
    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    model.config.vision_tower_dino = model_args.vision_tower_dino
    model.config.vision_tower_siglip = model_args.vision_tower_siglip
    model.config.mm_projector_type = model_args.mm_projector_type
    model.config.model_type = model_args.model_type
    model.config.lora_enable = training_args.lora_enable
    model.config.version = model_args.version
    model.config.training_stage = "finetune"
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        rank0_print(f"✅ 已成功匹配对话模板: {model_args.version}")
    else:
        raise ValueError(f"❌ 错误: 无法在 conversation_lib 中找到模板 '{model_args.version}'。 可用模板有: {list(conversation_lib.conv_templates.keys())}")
    if local_rank <= 0:
        test_conv = conversation_lib.default_conversation.copy()
        test_conv.append_message(test_conv.roles[0], "Check alignment.")
        test_conv.append_message(test_conv.roles[1], None)
        prompt = test_conv.get_prompt()
        rank0_print(f"📝 最终 Prompt 结构预览测试.............................:\n{'-' * 30}\n{prompt}\n{'-' * 30}")
    data_args.mm_use_im_start_end = getattr(model_args, 'mm_use_im_start_end', False)
    data_args.version = model_args.version
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

# ================= 🚀 终极地毯式精度转换开始 =================
    rank0_print("🛠️  开始对 Bunny-Phi 结构进行全量物理精度固化...")

    # 1. 第一步：全模型基础转换
    model.half() 

    # 2. 第二步：细致处理视觉塔 (Vision Tower)
    # 视觉塔往往包含很多复杂的 Buffer (如 RoPE 频率, mask)，必须逐个清理
    if hasattr(model.get_model(), "vision_tower"):
        rank0_print("  --> 正在固化 Vision Tower...")
        vision_tower = model.get_model().vision_tower
        vision_tower.to(torch.float16)
        for buf_name, buf in vision_tower.named_buffers():
            buf.data = buf.data.to(torch.float16)

    # 3. 第三步：细致处理适配器 (Projector/Multi-modal Projector)
    if hasattr(model.get_model(), "mm_projector"):
        rank0_print("  --> 正在固化 Multi-modal Projector...")
        model.get_model().mm_projector.to(torch.float16)
        for param in model.get_model().mm_projector.parameters():
            param.data = param.data.to(torch.float16)

    # 4. 第四步：死啃 LLM 底座 (PhiDecoderLayers)
    # 这是您发现警告最多的地方，我们要深入到每一个 DecoderLayer
    rank0_print("  --> 正在深入 Phi Decoder Layers...")
    llm_model = model.get_model() # 这里的层级根据您的模型结构可能微调
    
    # 处理词嵌入层 (Embedding) - 警告中的常客
    if hasattr(llm_model, "embed_tokens"):
        llm_model.embed_tokens.to(torch.float16)
        llm_model.embed_tokens.weight.data = llm_model.embed_tokens.weight.data.to(torch.float16)

    # 递归遍历所有 Decoder 层
    for i, layer in enumerate(llm_model.layers):
        # 强制该层所有参数为 fp16
        layer.to(torch.float16)
        # 逐个加固 LayerNorm (FP16 模式下最不稳定的地方)
        if hasattr(layer, "input_layernorm"):
            layer.input_layernorm.to(torch.float16)
        if hasattr(layer, "post_attention_layernorm"):
            layer.post_attention_layernorm.to(torch.float16)
        # 处理 Attention 和 MLP
        layer.self_attn.to(torch.float16)
        layer.mlp.to(torch.float16)
        
    # 5. 第五步：收尾处理 Final Norm 和 Head
    if hasattr(llm_model, "final_layernorm"):
        llm_model.final_layernorm.to(torch.float16)
    if hasattr(model, "lm_head"):
        model.lm_head.to(torch.float16)
        model.lm_head.weight.data = model.lm_head.weight.data.to(torch.float16)

    # 6. 最后的 Buffer 大扫除 (彻底消除 Upcast 诱因)
    for name, buf in model.named_buffers():
        # ✨ 关键：只有浮点类型的 Buffer 才转 FP16，整数型的索引必须保留 Long/Int
        if torch.is_floating_point(buf):
            buf.data = buf.data.to(torch.float16)
        else:
            # 确保 position_ids 等索引张量是 Long 类型
            buf.data = buf.data.to(torch.long)

    for m in model.modules():
        if "SiglipVisionEmbeddings" in m.__class__.__name__:
            if hasattr(m, "position_ids"):
                m.position_ids.data = m.position_ids.data.to(torch.long)


    torch.cuda.empty_cache()
    rank0_print("✅ 地毯式固化完成，已切断所有 Float32 路径。")
    # ================= 🚀 终极地毯式精度转换结束 =================



    trainer = BunnyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    torch.cuda.empty_cache()
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        rank0_print("\n" + "=" * 60)
        rank0_print("🔍 [Stage 3] 最终参数解冻状态检查")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        vt_active = any("vision_tower" in n for n in trainable_names)
        pj_active = any("mm_projector" in n for n in trainable_names)
        llm_active = any("layers" in n for n in trainable_names)
        rank0_print(f"   - Vision Tower Active: {vt_active}")
        rank0_print(f"   - Projector Active:    {pj_active}")
        rank0_print(f"   - LLM Backbone Active: {llm_active}")
        rank0_print(f"📊 总可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        rank0_print("=" * 60 + "\n")
    test_id = tokenizer.convert_tokens_to_ids("<img_content>")
    assert test_id == model.config.image_token_index, "Tokenizer ID 与模型配置不符！"
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        latest_ckpt = str(sorted(checkpoints, key=get_checkpoint_number)[-1])
        if checkpoint_has_trainer_state(latest_ckpt):
            rank0_print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
            trainer.train(resume_from_checkpoint=latest_ckpt)
        else:
            rank0_print(f"⚠️ Checkpoint 损坏，重新开始训练。")
            trainer.train()
    else:
        rank0_print("🚀 Starting training from scratch.")
        trainer.train()
    trainer.save_state()
    if training_args.local_rank <= 0:
        rank0_print("📢 [Stage 3] 训练完成，正在导出‘原子级回填包’以防权重被官方覆盖...")
        if training_args.lora_enable:
            rank0_print("📎 合并 LoRA 权重中...")
            model = model.merge_and_unload()
            model.config.lora_enable = False
        for name, module in model.named_modules():
            for buf_name, buf in module.named_buffers(recurse=False):
                module.register_buffer(buf_name, buf, persistent=True)
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        atomic_dir = os.path.join(training_args.output_dir, "atomic_weights_v365")
        os.makedirs(atomic_dir, exist_ok=True)
        raw_model = model.module if hasattr(model, "module") else model
        vision_tower = raw_model.get_vision_tower()
        if hasattr(vision_tower, 'dino_vision_tower'):
            rank0_print("💾 导出 DINO Backbone...")
            dino_m = vision_tower.dino_vision_tower.vision_tower
            torch.save(dino_m.state_dict(), os.path.join(atomic_dir, "sub_dino_backbone.pth"))
        if hasattr(vision_tower, 'siglip_vision_tower'):
            rank0_print("💾 导出 SigLIP Backbone...")
            siglip_m = vision_tower.siglip_vision_tower.vision_tower
            torch.save(siglip_m.state_dict(), os.path.join(atomic_dir, "sub_siglip_backbone.pth"))
        rank0_print("💾 导出 365 协议粘合层 (包含 Projector, CrossAttn, Sampler)...")
        tokenizer.save_pretrained(training_args.output_dir)
        if hasattr(model, "generation_config"):
            model.generation_config.save_pretrained(training_args.output_dir)
        rank0_print(f"✅ 全量模型已完整保存至: {training_args.output_dir}")


if __name__ == "__main__":
    train()
