import os
import sys

# 1. 环境与路径配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
sys.path.append(os.getcwd())

import torch
from PIL import Image
from safetensors.torch import load_file  # pip install safetensors
from transformers import AutoConfig, AutoTokenizer, SiglipImageProcessor
from transformers.cache_utils import DynamicCache

from bunny.model.language_model.bunny_phi import BunnyPhiConfig, BunnyPhiForCausalLM
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token

def test_inference():
    disable_torch_init()

    # --- 1. 路径设置 ---
    checkpoint_path = '/mnt/CoBunny/checkpoints-pretrain/bunny-phi1.5-mixed-pretrain/checkpoint-33300'
    base_llm_path = '/mnt/conda_data/microsoft/phi-1_5' 
    siglip_path = "/mnt/siglip-so400m-patch14-384"
    image_path = "Test.jpg"

    # --- 2. 构造模型骨架 ---
    print("🔄 正在加载配置并初始化模型骨架...")
    config = BunnyPhiConfig.from_pretrained(checkpoint_path)
    # 强制开启 cache 提高推理速度
    config.use_cache = True 
    model = BunnyPhiForCausalLM(config)

    # --- 3. 手动组装权重 (关键：解决 safetensors 加载问题) ---
    print("🔄 步骤 A: 加载基础 Phi-1.5 语言模型权重 (.safetensors)...")
    base_weight_file = os.path.join(base_llm_path, "model.safetensors")
    if os.path.exists(base_weight_file):
        state_dict_base = load_file(base_weight_file, device="cpu")
    else:
        # 备选路径：防止有些环境还是 .bin
        state_dict_base = torch.load(os.path.join(base_llm_path, "pytorch_model.bin"), map_location="cpu")
    
    model.load_state_dict(state_dict_base, strict=False)

    print("🔄 步骤 B: 注入预训练对齐权重 (Projector)...")
    projector_weights = torch.load(os.path.join(checkpoint_path, "mm_projector.bin"), map_location="cpu")
    model.load_state_dict(projector_weights, strict=False)

    # 如果有视觉塔微调权重，加载它
    vision_tuned_path = os.path.join(checkpoint_path, "vision_tower_tuned.bin")
    if os.path.exists(vision_tuned_path):
        print("🔄 步骤 C: 注入视觉塔微调权重...")
        vision_tuned_weights = torch.load(vision_tuned_path, map_location="cpu")
        model.load_state_dict(vision_tuned_weights, strict=False)

    # --- 4. 辅助补丁：Cache 兼容性修复 ---
    # 解决 transformers 内部调用 get_usable_length 时的参数报错
    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, seq_length=None, layer_idx=None):
            return self.get_seq_length(layer_idx if layer_idx is not None else 0)
        DynamicCache.get_usable_length = get_usable_length

    # --- 5. 加载 Tokenizer 和 图像处理器 ---
    print("🔄 加载 Tokenizer 与处理器...")
    tokenizer = AutoTokenizer.from_pretrained(base_llm_path, use_fast=True)
    image_processor = SiglipImageProcessor.from_pretrained(siglip_path)

    # 移动模型到 GPU
    device = torch.device("cuda")
    model.to(device, dtype=torch.float16)
    model.eval()
    print("✅ 模型组装成功，已就绪。")

    # --- 6. 图像处理 ---
    if not os.path.exists(image_path):
        print(f"❌ 找不到测试图片 {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    processed_output = image_processor(image, return_tensors="pt")
    image_tensor = processed_output["pixel_values"].to(device, dtype=torch.float16)

    # --- 7. 推理过程 ---
    # 注意：Pretrain 阶段建议使用简单的 Prompt 格式

    
    prompt = "A picture of" 
    input_ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors="pt").unsqueeze(0).to(device)

    print("🚀 启动混合推理引擎...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.7,        # 提高随机性，防止死循环
            top_p=0.9,
            max_new_tokens=64,
            use_cache=True,
            repetition_penalty=1.5, # 👈 关键：添加重复惩罚，强制模型不复读
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # --- 8. 结果展示 ---
    output_text = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    print("\n" + "=" * 40)
    print(f"🖼️ 模型推理结果: {output_text}")
    print("=" * 40)

if __name__ == "__main__":
    test_inference()