import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import copy
from PIL import Image
from transformers import AutoTokenizer
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
from bunny.util.mm_utils import tokenizer_image_token
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.model.multimodal_encoder.AdaptiveConcatenationVisionTower import ImageProcessorMultipleEncoders

def run_debug_inference():
    model_path = "/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-6619"
    image_path = "testt.jpg"
    device = "cuda"

    print(f"--- 🛠️ 开始深度诊断 ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = BunnyPhiForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)

    # --- 修复后的权重检查 ---
    print("\n🔍 [诊断 1: 融合层权重]")
    if hasattr(vision_tower, 'final_cls_weights'):
        weights = vision_tower.final_cls_weights.data
        print(f"融合层权重: {weights}")
        # 修复 dtype 不匹配报错
        is_initial = torch.allclose(weights, torch.tensor([0.5, 0.5], dtype=torch.float16, device=device), atol=1e-2)
        if is_initial:
            print("⚠️ 警告：权重接近初始值。")
        else:
            print("✅ 权重已偏离初始值，训练生效。")

    # --- 极简提示词 (针对小模型优化) ---
    # 格式：<image>\nUSER: What is in the image? ASSISTANT:
    question = "What is in the image?"
    prompt = f"{DEFAULT_IMAGE_TOKEN}\nUSER: {question} ASSISTANT:"
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    print("\n🔍 [诊断 2: Token 识别]")
    if IMAGE_TOKEN_INDEX in input_ids:
        pos = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1].item()
        print(f"✅ 成功识别图像占位符 (-200) 在位置: {pos}")
    else:
        print("❌ 错误：未识别到 -200")

    image = Image.open(image_path).convert("RGB")
    image_processor = ImageProcessorMultipleEncoders(patch_size_list=[14], target_size=384)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float16)

    print("\n🚀 [诊断 3: 推理测试]")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=64, # 先看短描述
            repetition_penalty=1.5,
            # 必须传 mask，防止 pad/eos 混淆
            attention_mask=torch.ones_like(input_ids).to(device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    response = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    print(f"\n✨ 推理结果:\n{response}")

if __name__ == "__main__":
    run_debug_inference()