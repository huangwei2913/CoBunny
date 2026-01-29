import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import glob
from PIL import Image
from transformers import AutoTokenizer, AutoConfig
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
from bunny.util.mm_utils import tokenizer_image_token
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.model.multimodal_encoder.AdaptiveConcatenationVisionTower import ImageProcessorMultipleEncoders

def run_final_full_inference():
    model_path = "/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune"
    image_path = "./gupiao.jpg"
    
    # 1. 加载 (已验证 0 缺失的完美方案)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    model = BunnyPhiForCausalLM(config).cuda().to(dtype=torch.float16)

    all_sd = {}
    for f in sorted(glob.glob(os.path.join(model_path, "*.bin"))):
        all_sd.update(torch.load(f, map_location="cpu"))
    model.load_state_dict(all_sd, strict=True) # 既然是对齐的，直接 strict 模式
    
    model.get_vision_tower().is_loaded = True
    model.eval()
    print("✅ 权重注入: 1147/1147 完美匹配。")

    # 2. 图像预处理
    image_processor = ImageProcessorMultipleEncoders(patch_size_list=[14], target_size=384)
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].cuda().to(dtype=torch.float16)

    # 3. 针对全量模型设计的 Prompt
    # 这种模型不喜欢太客套，它喜欢直接看图说话
    question = "What is the title written on the top?"
    prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n{question} ASSISTANT:"
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    attention_mask = torch.ones_like(input_ids).cuda()

    print("\n🚀 正在调用全量逻辑执行深度推理...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            do_sample=False,          # 开启采样
            temperature=0.4,         # 低温保持逻辑
            max_new_tokens=256,
            min_new_tokens=30,       # 【关键】强制它展开说，不要偷懒
            repetition_penalty=1.1,  # 适度的惩罚防止乱码
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 清理可能出现的复读标记
    if "USER" in response: response = response.split("USER")[0]
    
    print(f"\n✨ Stage 3 最终逻辑回复:\n{'-'*40}\n{response}\n{'-'*40}")

if __name__ == "__main__":
    run_final_full_inference()