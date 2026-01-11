import os
import sys

# 关键：强制指定单卡环境，彻底解决 Runtime Error: Expected all tensors to be on the same device
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
from PIL import Image
from transformers import AutoConfig, logging
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin

# 确保能找到 bunny 模块
sys.path.append(os.getcwd())

from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from bunny.model.language_model.phi import PhiForCausalLM

def test_inference():
    disable_torch_init()

    # --- 1. 路径设置 ---
    checkpoint_path = '/mnt/CoBunny/checkpoints-pretrain/bunny-phi1.5-mixed-pretrain/checkpoint-33300'
    base_llm_path = '/mnt/conda_data/microsoft/phi-1_5' 
    dino_path = "/mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m"
    oryx_path = "oryx_vit:/mnt/THUdyhOryx-ViT/oryx_vit.pth"      #要特别注意的是，我们应该修改配置文件，这里是手动的，以后会变成自动的
    model_name = 'bunny-phi-1.5'
    model_type = 'phi-1.5'

    print(f"🔄 正在读取配置并注入混合编码器参数...")
    from transformers.cache_utils import DynamicCache
    
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    if not hasattr(DynamicCache, "get_usable_length"):
        print("🔧 正在修复 DynamicCache 兼容性 (get_usable_length 严谨版)...")
        def get_usable_length(self, seq_length=None, layer_idx=None):
            # 关键修复：如果 layer_idx 是 None，直接调用不带参数的 get_seq_length
            if layer_idx is None:
                return self.get_seq_length()
            return self.get_seq_length(layer_idx)
        
        DynamicCache.get_usable_length = get_usable_length

    # --- 2. 加载模型 ---
    print("🔄 正在通过混合逻辑加载模型 (强制单卡模式)...")
    # 注意：这里我们传入 config=cfg_pretrained 确保路径生效
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=checkpoint_path,   
        model_base=base_llm_path,    
        model_name=model_name,
        model_type=model_type
    )

    # --- 3. 核心补丁：类结构重塑与 Cache 兼容性 ---
    print("🔧 执行类结构重塑与 Cache 兼容性补丁...")
    
    # 修复 DynamicCache 属性名缺失
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    # 动态重塑类继承关系，找回 generate 等缺失属性
    class FullyFixedBunnyModel(model.__class__, PhiForCausalLM, GenerationMixin):
        pass
    model.__class__ = FullyFixedBunnyModel

    # 修复视觉塔接口
    if not hasattr(model, 'get_vision_tower'):
        model.get_vision_tower = lambda: model.model.get_vision_tower()

    # 强制将整个模型移动到同一设备并设为 eval 模式
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    # --- 4. 准备图片 ---
    image_path = "Test.jpg"
    if not os.path.exists(image_path):
        print(f"❌ 找不到测试图片 {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    processed_output = image_processor.preprocess(image, return_tensors="pt")
    
    # 这里的 Key 必须与你定义的 SingleImageProcessor 对应
    image_tensor = processed_output["pixel_values"].to(device, dtype=torch.float16)
    print(f"✅ 图像 Tensor 准备就绪，形状: {image_tensor.shape}")

    # --- 5. 构建推理 ---
    prompt = "A picture of"
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, -200, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )

    print("🚀 启动混合推理引擎...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=20,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # --- 6. 结果展示 ---
    output_text = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    
    print("\n" + "=" * 40)
    print(f"🖼️ 模型推理结果: {output_text}")
    print("=" * 40)

    # 逻辑验证
    if len(output_text) < 3 or (output_text.count('!') > 5):
        print("🚩 警告：输出疑似异常（感叹号过多或过短）。可能需要检查 Projector 训练状态。")
    else:
        print("✅ 成功：模型输出了有效文本，混合编码器逻辑已跑通。")

if __name__ == "__main__":
    test_inference()