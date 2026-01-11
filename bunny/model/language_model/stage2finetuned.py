import torch
import os
from PIL import Image
from transformers.generation import GenerationMixin
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, IMAGE_TOKEN_INDEX

import torch
import os
import os
import sys
from PIL import Image
from transformers.generation import GenerationMixin # 导入生成类
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, IMAGE_TOKEN_INDEX
# --- 环境变量与兼容性设置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 强制单卡推理
# 确保能找到 bunny 模块（假设你在项目根目录运行）
sys.path.append(os.getcwd())
import argparse
from peft import PeftModel # <--- 必须手动引入
from bunny.model.builder import load_pretrained_model
from bunny.util.mm_utils import get_model_name_from_path
from PIL import Image
from transformers import logging
from transformers.cache_utils import DynamicCache

def patch_dynamic_cache():
    """修复 Transformers 库对 Phi 模型的 Cache 兼容性问题"""
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, seq_length=None, layer_idx=None):
            if layer_idx is None:
                return self.get_seq_length()
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = get_usable_length
        print("🔧 已修复 DynamicCache 兼容性补丁")


def run_triple_merge_inference():
    disable_torch_init()
    patch_dynamic_cache()
    # 路径定义
    checkpoint_dir = '/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k'
    base_llm_path = '/mnt/conda_data/microsoft/phi-1_5'
    
    print("1️⃣ 加载基础模型与 LoRA 插件...")
    # 这里 load_pretrained_model 会加载 base 权重，并根据 checkpoint_dir 加载 adapter_model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=checkpoint_dir,
        model_base=base_llm_path,
        model_name='bunny-phi-1.5-lora',
        model_type='phi-1.5',
        device_map="cuda",
        torch_dtype=torch.float16 
    )

    print("2️⃣ 注入 Non-LoRA 视觉权重 (Vision Tower + Projector)...")
    non_lora_bin = os.path.join(checkpoint_dir, 'non_lora_trainables.bin')
    
    if os.path.exists(non_lora_bin):
        # 加载这 222MB 的“视觉灵魂”
        non_lora_weights = torch.load(non_lora_bin, map_location='cuda')
        
        # 这一步非常重要：我们需要确保权重被分发到正确的子模块
        # 修正 key 的前缀问题（如果保存时带了 base_model.model 前缀）
        cleaned_weights = {}
        for k, v in non_lora_weights.items():
            new_k = k.replace('base_model.model.', '') # 移除 Peft 包装前缀
            cleaned_weights[new_k] = v
            
        # 强制更新模型，strict=False 允许只更新部分权重
        incompatible_keys = model.load_state_dict(cleaned_weights, strict=False)
        print(f"✅ 视觉注入完成！")
        print(f"   - 匹配成功的权重项: {len(cleaned_weights)}")
        print(f"   - 缺失项 (应该是正常的 LLM 权重): {len(incompatible_keys.missing_keys)}")
    else:
        raise FileNotFoundError(f"找不到关键文件: {non_lora_bin}")

    # 3. 动态修复 generate 属性
    if not hasattr(model, 'generate'):
        class FixedBunnyModel(model.__class__, GenerationMixin): pass
        model.__class__ = FixedBunnyModel

  

    if model.generation_config is None:
        from transformers import GenerationConfig
        try:
            # 尝试从模型配置创建一个默认的生成配置
            model.generation_config = GenerationConfig.from_model_config(model.config)
        except:
            # 如果失败，创建一个空的
            model.generation_config = GenerationConfig()
    
    # 确保 pad_token_id 存在，否则 phi 等模型生成会报错
    model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # 4. 执行推理

    model.eval()

    image = Image.open("Test.jpg").convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device='cuda', dtype=torch.float16)
    
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    print("🎯 开始生成（LoRA + 视觉补丁全开）...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=256,
            use_cache=True
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # 提取回复
    final_output = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response
    
    print("\n" + "="*50)
    print(f"🎉 最终结果:\n{final_output}")
    print("="*50)

if __name__ == "__main__":
    run_triple_merge_inference()