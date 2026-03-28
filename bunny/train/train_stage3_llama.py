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
import json

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


import hashlib
import json

def get_tensor_md5(tensor):
    # 取前 100 个元素的特征来快速生成指纹，防止全量计算太慢
    content = tensor.detach().cpu().numpy().tobytes()
    return hashlib.md5(content).hexdigest()


def save_v365_artifacts(model, output_dir, training_args):
    """
    通用函数：在指定的 output_dir 中生成原子包和审计报告
    """
    if training_args.local_rank > 0:
        return

    rank0_print(f"📦 [V365 Protocol] 正在目录 {output_dir} 中部署审计与回填包...")
    
    # 1. 准备目录
    atomic_dir = os.path.join(output_dir, "atomic_weights_v365")
    os.makedirs(atomic_dir, exist_ok=True)

    # 2. 获取视觉塔实体
    raw_model = model.module if hasattr(model, "module") else model
    vt = raw_model.get_vision_tower()
    
    # 3. 导出原子包 (Backbones & Glue)
    # --- DINO ---
    if hasattr(vt, 'dino_vision_tower'):
        dino_m = vt.dino_vision_tower.vision_tower
        torch.save(dino_m.state_dict(), os.path.join(atomic_dir, "sub_dino_backbone.pth"))
    
    # --- SigLIP ---
    if hasattr(vt, 'siglip_vision_tower'):
        siglip_m = vt.siglip_vision_tower.vision_tower
        torch.save(siglip_m.state_dict(), os.path.join(atomic_dir, "sub_siglip_backbone.pth"))

    # --- Glue (Projector, Resampler, etc.) ---
    glue_state = {}
    backbone_prefixes = ['dino_vision_tower.vision_tower', 'siglip_vision_tower.vision_tower']
    for name, param in vt.named_parameters():
        if not any(name.startswith(p) for p in backbone_prefixes):
            glue_state[name] = param.detach().cpu()
    for name, buf in vt.named_buffers():
        if not any(name.startswith(p) for p in backbone_prefixes):
            glue_state[name] = buf.detach().cpu()
    torch.save(glue_state, os.path.join(atomic_dir, "vision_glue_v365.pth"))

    # 4. 生成 vision_audit.json (MD5 报告)
    audit_report = {
        "model_type": "bunny-phi-v365",
        "path": output_dir,
        "weights": {}
    }
    
    for name, param in vt.named_parameters():
        audit_report["weights"][name] = {
            "shape": list(param.shape),
            "md5": get_tensor_md5(param),
            "is_buffer": False
        }
    for name, buf in vt.named_buffers():
        audit_report["weights"][name] = {
            "shape": list(buf.shape),
            "md5": get_tensor_md5(buf),
            "is_buffer": True
        }

    with open(os.path.join(output_dir, "vision_audit.json"), "w") as f:
        json.dump(audit_report, f, indent=4)
    
    rank0_print(f"✅ [V365 Done] 审计完成。该 Checkpoint 现在具备‘身份自证明’能力。")


def train():
    global local_rank

    # 1. 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # 自动推断计算精度 (FP16/BF16/FP32)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    model_args.unfreeze_mm_vision_tower = True  # 解冻视觉塔
    training_args.freeze_mm_mlp_adapter = False # 解冻 Projector     
    import sys
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print("\n" + "="*50)
        print("🛠️  原始命令行参数 (sys.argv):")
        print(sys.argv)
        print("-" * 50)
        
        print("📊 HfArgumentParser 解析结果:")
        # 检查 model_args 里的两个潜在冲突变量
        v1 = getattr(model_args, 'unfreeze_mm_vision_tower', "MISSING")
        v2 = getattr(model_args, 'unfreeze_vision_tower', "MISSING")
        
        print(f">> model_args.unfreeze_mm_vision_tower: {v1} (Type: {type(v1)})")
        print(f">> model_args.unfreeze_vision_tower:    {v2} (Type: {type(v2)})")
        
        # 检查 training_args 是否也被污染
        v3 = getattr(training_args, 'unfreeze_mm_vision_tower', "MISSING")
        print(f">> training_args.unfreeze_mm_vision_tower: {v3}")
        print("="*50 + "\n")
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
    if model_args.model_type in ['phi-1.5', 'phi-2']:
        model = BunnyPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=compute_dtype,
        )
    elif model_args.model_type == 'llama3-1b':
        model = BunnyLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        torch_dtype=compute_dtype,
    )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    NEW_TOKENS = ["<img_content>", "<pad>"]
    tokenizer.add_tokens(NEW_TOKENS, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    img_content_id = tokenizer.convert_tokens_to_ids("<img_content>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    old_img_id = tokenizer.convert_tokens_to_ids('<image>')  # 之前微调占用的 ID
    # 设置 Pad 属性
    tokenizer.pad_token = "<pad>"
    model.config.pad_token_id = tokenizer.pad_token_id  # 这个在后面会被用到
    model.config.image_token_index = tokenizer.convert_tokens_to_ids("<img_content>")
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 128001
  

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
    model.get_model().initialize_vision_modules_stage3(model_args=model_args)
    if training_args.local_rank <= 0:  # 仅在主进程运行
        print("\n" + "🔍" * 20)
        print("开始扫描 AdaptiveConcatenationVisionTower 及其子塔属性...")
        
        vision_tower = model.get_vision_tower()
        
        # 1. 检查容器层 (AdaptiveConcatenationVisionTower)
        print(f"\n[Level 1: 容器层] {type(vision_tower).__name__}")
        # 尝试从不同的坑位找这个参数
        container_unfreeze = getattr(vision_tower, 'unfreeze_mm_vision_tower', "未直接定义")
        print(f"  - 直接属性 self.unfreeze_mm_vision_tower: {container_unfreeze}")
        
        if hasattr(vision_tower, 'args'):
            v = getattr(vision_tower.args, 'unfreeze_mm_vision_tower', "不存在")
            print(f"  - 内部 self.args.unfreeze_mm_vision_tower: {v}")

        # 2. 检查子塔层 (DINO / SigLIP)
        # 假设你的子塔存在 self.siglip_vision_tower 这种变量里
        for sub_tower_name in ['siglip_vision_tower', 'dino_vision_tower']:
            if hasattr(vision_tower, sub_tower_name):
                sub_tower = getattr(vision_tower, sub_tower_name)
                print(f"\n[Level 2: 子塔层] {sub_tower_name}")
                
                # 检查子塔是否继承了 requires_grad
                has_grad = any(p.requires_grad for p in sub_tower.parameters())
                print(f"  - 实际梯度状态 (requires_grad): {'🔥 开启' if has_grad else '❄️ 冻结'}")
                
                # 检查子塔内部的 config/args
                if hasattr(sub_tower, 'config'):
                    v = getattr(sub_tower.config, 'unfreeze_mm_vision_tower', "不存在")
                    print(f"  - self.config.unfreeze_mm_vision_tower: {v}")
        
        # 3. 终极验证：检查 LLM 层的 config
        print(f"\n[Level 3: 全局配置] model.config")
        v_global = getattr(model.config, 'unfreeze_mm_vision_tower', "不存在")
        v_old = getattr(model.config, 'unfreeze_vision_tower', "不存在")
        print(f"  - model.config.unfreeze_mm_vision_tower: {v_global}")
        print(f"  - model.config.unfreeze_vision_tower (旧版标签): {v_old}")
        print("🔍" * 20 + "\n")
    # 3. 【强行覆盖】不管刚才 print 了什么 "Frozen"，现在我们手动接管控制权
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=compute_dtype, device=training_args.device)

    if model_args.unfreeze_mm_vision_tower:  #解冻
        rank0_print("🔨 [Override] 忽略之前的 Frozen 日志，正在强制执行解冻程序...")    
        # A. 解冻顶层
        vision_tower.requires_grad_(True)
        vision_tower.train()
        model.gradient_checkpointing_enable()  
        # B. 递归解冻所有子模块（针对双塔结构）
        for name, module in vision_tower.named_modules():
            # 跳过 batchnorm 等特殊层（可选，但全量微调通常也开）
            if "vision_tower" in name or "encoder" in name: 
                module.requires_grad_(True)
                module.train()
        
        # C. 【最终验尸】真相只有一个：检查梯度的 flag
        check_param = next(vision_tower.parameters())
        if check_param.requires_grad:
            rank0_print(f"✅ [SUCCESS] 视觉塔状态确认：UNFROZEN (requires_grad=True)")
            rank0_print(f"🔥 [Ready] 全量微调模式已就绪！")
        else:
            rank0_print(f"❌ [FAIL] 强制解冻失败！请检查代码逻辑！")
            exit(1) # 直接报错退出，别跑了
    else:
        # 如果确实是 False，那就冻结
        vision_tower.requires_grad_(False)
        vision_tower.eval()
        rank0_print("❄️ [Confirmed] 视觉塔保持冻结状态。")

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
    model.config.training_stage = "finetune" #历史原因哈，实际上就是应该确保config.json里面的这个值是finetue
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

    from transformers import TrainerCallback

    class V365AuditCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if args.local_rank <= 0:
                checkpoint_folder = f"checkpoint-{state.global_step}"
                output_path = os.path.join(args.output_dir, checkpoint_folder)
                
                # 安全获取 model，如果拿不到就从 trainer 里拿
                model = kwargs.get('model')
                if model is None and 'trainer' in kwargs:
                    model = kwargs['trainer'].model
                
                if model is not None:
                    # 已经彻底去掉了对 tokenizer 的依赖，安全！
                    save_v365_artifacts(model, output_path, args)
                else:
                    rank0_print("⚠️ [V365 Warning] 无法在回调中找到 model 实例，跳过审计保存。")
        # =========================================================
    # 13. Trainer 初始化
    # =========================================================
    trainer = BunnyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[V365AuditCallback()],
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
    
    # train_stage3.py
    test_id = tokenizer.convert_tokens_to_ids("<img_content>")
    assert test_id == model.config.image_token_index, "Tokenizer ID 与模型配置不符！"
    # =========================================================
    # 14. 训练执行 (支持断点续训)
    # =========================================================
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

    # 保存最终状态
    trainer.save_state()
    
    # =========================================================
    # 15. 最终全量保存 (Stage 3 增强版：原子级回填包)
    # =========================================================
    if training_args.local_rank <= 0:
        rank0_print("📢 [Stage 3] 训练完成，正在导出‘原子级回填包’以防权重被官方覆盖...")

        # 1. 处理 LoRA 合并 (如果启用)
        if training_args.lora_enable:
            rank0_print("📎 合并 LoRA 权重中...")
            model = model.merge_and_unload()
            model.config.lora_enable = False

        # 2. 关键：强制物化所有 Buffer 并标记为持久化
        # 这一步是为了防止 position_ids 等在保存时被漏掉
        for name, module in model.named_modules():
            for buf_name, buf in module.named_buffers(recurse=False):
                module.register_buffer(buf_name, buf, persistent=True)

        # 3. 维度 A：HuggingFace 标准全量保存 (from_pretrained 用)
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        # 1. 准备专门的目录
        atomic_dir = os.path.join(training_args.output_dir, "atomic_weights_v365")
        os.makedirs(atomic_dir, exist_ok=True)

        # 获取原始模型
        raw_model = model.module if hasattr(model, "module") else model
        vision_tower = raw_model.get_vision_tower()

        # --- (1) 导出子塔骨干权重 (Backbone) ---
        # 我们直接深入到最底层的 .vision_tower 成员，确保只存 Transformer 权重
        if hasattr(vision_tower, 'dino_vision_tower'):
            rank0_print("💾 导出 DINO Backbone...")
            dino_m = vision_tower.dino_vision_tower.vision_tower
            # 【核心修复】强制让所有参数出现在 state_dict 中
            for p in dino_m.parameters():
                p.requires_grad = True
            torch.save(dino_m.state_dict(), os.path.join(atomic_dir, "sub_dino_backbone.pth"))    
   

        if hasattr(vision_tower, 'siglip_vision_tower'):
            rank0_print("💾 导出 SigLIP Backbone...")
            siglip_m = vision_tower.siglip_vision_tower.vision_tower
            for p in siglip_m.parameters():
                p.requires_grad = True
            torch.save(siglip_m.state_dict(), os.path.join(atomic_dir, "sub_siglip_backbone.pth"))

        # --- (2) 导出 365 协议粘合层 (Glue) ---
        # 这里我们采用“排除法”，把 vision_tower 中不属于上面两个 backbone 的参数全抓出来
        rank0_print("💾 导出 365 协议粘合层 (包含 Projector, CrossAttn, Sampler)...")
        glue_state = {}
        backbone_prefixes = ['dino_vision_tower.vision_tower', 'siglip_vision_tower.vision_tower']
        
        for name, param in vision_tower.named_parameters():
            if not any(name.startswith(p) for p in backbone_prefixes):
                glue_state[name] = param.cpu().data
                
        # 同时抓取 Buffer (比如采样器里的 grid 或 pos_embed)
        for name, buf in vision_tower.named_buffers():
            if not any(name.startswith(p) for p in backbone_prefixes):
                glue_state[name] = buf.cpu().data

        torch.save(glue_state, os.path.join(atomic_dir, "vision_glue_v365.pth"))
        
        # --- (3) 保存词表 (非常关键，防止 Token ID 错乱) ---
        tokenizer.save_pretrained(training_args.output_dir)
        
        rank0_print(f"✅ 原子级回填包已保存至: {atomic_dir}")
      
        # 确保 generation_config 也被保存
        if hasattr(model, "generation_config"):
            model.generation_config.save_pretrained(training_args.output_dir)

        rank0_print(f"✅ 全量模型已完整保存至: {training_args.output_dir}")
        rank0_print("📊 正在生成 365 协议权重审计报告 (JSON)...")
    
        raw_model = model.module if hasattr(model, "module") else model
        vt = raw_model.get_vision_tower()
        
        audit_report = {
            "model_type": "bunny-phi-v365",
            "weights": {}
        }

        # 遍历视觉塔的所有参数和 Buffer
        for name, param in vt.named_parameters():
            audit_report["weights"][name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "md5": get_tensor_md5(param),
                "is_buffer": False
            }

        for name, buf in vt.named_buffers():
            audit_report["weights"][name] = {
                "shape": list(buf.shape),
                "dtype": str(buf.dtype),
                "md5": get_tensor_md5(buf),
                "is_buffer": True
            }

        # 保存 JSON 审计文件
        with open(os.path.join(training_args.output_dir, "vision_audit.json"), "w") as f:
            json.dump(audit_report, f, indent=4)
            
        rank0_print("✅ 审计报告已生成。推理时只需对比 MD5 即可判断权重是否被官方覆盖。")

if __name__ == "__main__":
    train()
