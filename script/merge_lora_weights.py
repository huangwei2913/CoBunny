import torch
import os
import argparse
from peft import PeftModel # <--- 必须手动引入
from bunny.model.builder import load_pretrained_model
from bunny.util.mm_utils import get_model_name_from_path

def merge_lora(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    print(f"🚀 正在加载基础模型: {args.model_base}")

    # 1. 先加载基础模型和配置
    # 注意：这里我们只加载基座，不让 builder 去处理复杂的 LoRA 逻辑
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name, 
        args.model_type,
        device_map={"": "cuda:0"}, 
        torch_dtype=torch.float16
    )

    # 2. 手动检查并加载 LoRA 权重
    # 如果目录下有 adapter_config.json，说明是 LoRA 模型
    if os.path.exists(os.path.join(model_path, 'adapter_config.json')):
        print("💎 检测到 LoRA 权重，正在手动包装并合并...")
        # 强制包装成 PeftModel 以获得 merge_and_unload 方法
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        print("✅ LoRA 权重已成功合并至主干。")
    else:
        print("⚠️ 未检测到 LoRA 权重，跳过合并步骤（可能已经是完整模型）。")

    # 3. 处理 non_lora_trainables (关键：视觉塔和投影层)
    # 这一步确保你的 Recipe-2 训练出的 Vision Tower 被保存
    non_lora_trainables_path = os.path.join(model_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        print("🎨 正在加载非 LoRA 可训练参数 (Vision Tower / Projector)...")
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        # 过滤掉不必要的 key 并载入
        model.load_state_dict(non_lora_trainables, strict=False)

    # 4. 修复 Phi-1.5 的 pad_token 问题
    if not hasattr(model.generation_config, 'pad_token_id') or model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id

    print(f"💾 正在保存合并后的完整模型至: {args.save_model_path}")
    
    # 5. 安全保存
    model.to("cpu")
    model.save_pretrained(args.save_model_path,safe_serialization=False)
    tokenizer.save_pretrained(args.save_model_path)
    
    print("🎊 恭喜！合并任务彻底圆满完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    args = parser.parse_args()
    merge_lora(args)