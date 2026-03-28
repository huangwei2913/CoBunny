import os
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Optional
import torch
import transformers
from transformers import BitsAndBytesConfig
from bunny.train.bunny_trainer import BunnyTrainer
from bunny import conversation as conversation_lib
from bunny.model import *
from bunny.util.data_utils import make_supervised_data_module, DataArguments
from arguments import ModelArguments,TrainingArguments
import re

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def checkpoint_has_trainer_state(checkpoint_dir):
    return os.path.exists(os.path.join(checkpoint_dir, "trainer_state.json"))


import re

def get_checkpoint_number(path):
    # 正则匹配路径最后的数字
    matches = re.findall(r"checkpoint-(\d+)", str(path))
    return int(matches[-1]) if matches else 0




def train():
    global local_rank

    # 1. 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # 自动推断计算精度 (FP16/BF16/FP32)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # ==========================================
    # =========================================================
    # 3. Tokenizer 初始化 (完整保留原逻辑，处理特殊Token)
    # =========================================================
    assert model_args.vision_tower is not None
    # 根据模型类型选择加载方式
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # 3. 模型加载 (在修改词表前加载，确保权重对应)
    if model_args.model_type in ['Qwen2-1.5B']:  #本期只做qwen2的
        model = BunnyQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=compute_dtype,
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    model.config.training_stage = "pretrain"  # 确保传给 config
    # --- 针对 Qwen2.5 干净词表的适配逻辑 ---
    NEW_TOKENS = ["<img_content>", "<pad>"]
    tokenizer.add_tokens(NEW_TOKENS, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # 获取 ID
    img_content_id = tokenizer.convert_tokens_to_ids("<img_content>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    base_spirit_id = 151643  # 基于我们分词器的测试结果：eos_token_id

    # 强制覆盖与对齐
    with torch.no_grad():
        rank0_print(f"✨ 检测到 Qwen2 底座，正在基于 ID {base_spirit_id} 初始化新词表...")
        
        # 获取基准权重 (EOS)
        base_weight = model.get_input_embeddings().weight[base_spirit_id].detach().clone()
        
        # 赋予 <pad> 和 <img_content> 初始权重
        # 这样做是为了让模型在 Stage 1 开始时，对这两个新 Token 的反应是“中性”的
        model.get_input_embeddings().weight[pad_id] = base_weight
        model.get_input_embeddings().weight[img_content_id] = base_weight.clone()
        
        # Output Embeddings 同步
        output_weight = model.get_output_embeddings().weight[base_spirit_id].detach().clone()
        model.get_output_embeddings().weight[pad_id] = output_weight
        model.get_output_embeddings().weight[img_content_id] = output_weight.clone()

    # 更新配置
    tokenizer.pad_token = "<pad>"
    model.config.pad_token_id = pad_id
    model.config.image_token_index = img_content_id

    rank0_print(f"✅ 权重搬家与词表强制对齐完成。当前词表大小: {len(tokenizer)}")
    pad_id_in_tokenizer = tokenizer.pad_token_id
    pad_id_by_name = tokenizer.convert_tokens_to_ids("<pad>")

    rank0_print(f"🔍 [Tokenizer Check]")
    rank0_print(f"   - <pad> token ID: {pad_id_by_name}")
    rank0_print(f"   - tokenizer.pad_token_id: {pad_id_in_tokenizer}")

    if pad_id_in_tokenizer != pad_id_by_name:
        raise ValueError("🚨 严重错误：tokenizer.pad_token_id 与 <pad> 的实际 ID 不一致！")

    rank0_print(f"   - <img_content> ID: {model.config.image_token_index}")

    # 1. 【强行注入】把 model_args 的意志强加给 model.config
    # 这样即使内部代码错误地使用了 config，它也能读到 True
    print(f"💉 [Patch] 正在将 unfreeze_mm_vision_tower={model_args.unfreeze_mm_vision_tower} 注入到 model.config...")
    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    
    # 2. 【初始化】正常调用
    rank0_print("👁️ 初始化视觉模块...")
    model.get_model().initialize_vision_modules(model_args=model_args)

    # 3. 【强行覆盖】不管刚才 print 了什么 "Frozen"，现在我们手动接管控制权
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=compute_dtype, device=training_args.device)
        # =========================================================
    # 核心解冻策略：黑名单模式
    # =========================================================
    if model_args.tune_mm_mlp_adapter:
        # [1] 暴力全量冻结 (包括刚生成的 Embedding)
        model.requires_grad_(False)
        
        # [2] 解锁连接 LLM 的投影层
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
            
        # [3] 精准解锁混合视觉塔内部的粘合层 (使用“黑名单”策略)
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=compute_dtype, device=training_args.device)
        
        # 冻结黑名单：这两个前缀对应的才是原始预训练权重，必须冻结
        frozen_prefixes = ["dino_vision_tower", "siglip_vision_tower"]
        
        for name, param in vision_tower.named_parameters():
            if not any(prefix in name for prefix in frozen_prefixes):
                param.requires_grad = True # 这里解锁了 mlp_layers, cross_attn, gate_mlps 等
            else:
                param.requires_grad = False
        
        # [4] 额外保险：强制锁定 LLM Backbone 的关键组件
        # 确保 Phi-1.5 的大脑和词表不被 Stage 1 污染
        for name, param in model.named_parameters():
            if any(bk in name for bk in ["model.layers", "model.embed_tokens", "lm_head"]):
                param.requires_grad = False

    # 视觉塔子塔设为 eval
    vision_tower.dino_vision_tower.eval()
    vision_tower.siglip_vision_tower.eval()

    model.config.use_cache = False # 在 train 脚本里强制执行

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # 8. 🛡️ 参数坏点清洗 (NaN/Inf 处理)
    rank0_print("🛡️ 执行参数清洗，防止数值不稳定...")
    for p in model.parameters():
        if p.requires_grad:
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=65500, neginf=-65500)

    # 9. 配置同步：将关键元数据存入 config 供推理使用
    data_args.image_processor = vision_tower.image_processor
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    # 记录训练参数到 config
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
    # 10. 模板与数据加载
# =========================================================
    # 模板初始化与强校验
    # =========================================================
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        rank0_print(f"✅ 已成功匹配对话模板: {model_args.version}")
    else:
        # 兜底逻辑：如果匹配失败，强制报错，而不是用默认模板乱跑
        raise ValueError(f"❌ 错误: 无法在 conversation_lib 中找到模板 '{model_args.version}'。 "
                         f"可用模板有: {list(conversation_lib.conv_templates.keys())}")

    # 打印关键标志位，手动核对是否与 Stage 2 一致
    rank0_print(f"🔍 模板细节校验:")
    rank0_print(f"   - 系统提示词 (System): {conversation_lib.default_conversation.system}")
    rank0_print(f"   - 角色 A (User): {conversation_lib.default_conversation.roles[0]}")
    rank0_print(f"   - 角色 B (Assistant): {conversation_lib.default_conversation.roles[1]}")
    rank0_print(f"   - 分隔符 (Sep): '{conversation_lib.default_conversation.sep}'")


    if local_rank <= 0:
        # 灵魂自检：打印一段模拟对话，看看长什么样
        test_conv = conversation_lib.default_conversation.copy()
        test_conv.append_message(test_conv.roles[0], "Check alignment.")
        test_conv.append_message(test_conv.roles[1], None)
        prompt = test_conv.get_prompt()
        rank0_print(f"📝 最终 Prompt 结构预览测试.............................:\n{'-'*30}\n{prompt}\n{'-'*30}")

    # =========================================================
    # 12. 数据加载 (完整逻辑)
    # =========================================================
    data_args.mm_use_im_start_end = getattr(model_args, 'mm_use_im_start_end', False)
    data_args.version = model_args.version
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
    # 参数状态检查日志 (更精确的过滤)
    if training_args.local_rank <= 0:
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        
        # 真正检查 LLM Backbone (排除掉 vision_tower 的干扰)
        llm_backbone_active = any(("model.layers" in n or "lm_head" in n) for n in trainable_params)
        vision_tower_active = any("vision_tower" in n for n in trainable_params)
        projector_active = any("mm_projector" in n for n in trainable_params)

        rank0_print("\n" + "="*60)
        rank0_print("📊 [Stage 1] 最终参数审计报告")
        rank0_print(f"   - 视觉混合组件 (Vision Mixed): {vision_tower_active}")
        rank0_print(f"   - 连接投影层 (Projector):    {projector_active}")
        rank0_print(f"   - LLM 大脑骨干 (LLM Backbone): {llm_backbone_active} (预期应为 False)")
        rank0_print(f"📈 总可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        rank0_print("="*60 + "\n")
    


    # train_stage3.py
    test_id = tokenizer.convert_tokens_to_ids("<img_content>")
    assert test_id == model.config.image_token_index, "Tokenizer ID 与模型配置不符！"
    # =========================================================
    # 14. 训练执行 (支持断点续训)
    # =========================================================
    # =========================================================
    # 修复后的断点查找逻辑
    # =========================================================


    # 1. 获取所有 checkpoint 文件夹
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))

    #checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
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

    # 保存最终状态
    trainer.save_state()
    


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
