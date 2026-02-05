import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import glob
from PIL import Image
from transformers import AutoTokenizer, AutoConfig
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
from bunny.util.mm_utils import tokenizer_image_token
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer, AutoConfig
# 1. 环境与路径设置

try:
    from bunny.model.multimodal_encoder.AdaptiveConcatenationVisionTower import ImageProcessorMultipleEncoders
    print("✅ 成功导入自定义处理器类: ImageProcessorMultipleEncoders")
except ImportError:
    print("❌ 警告：无法导入 AdaptiveConcatenationVisionTower！请检查文件路径。")

from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, get_model_name_from_path
from bunny.conversation import conv_templates



def calculate_anchors(full_len, target_len):
    """复刻 data_utils.py 中的动态锚点计算"""
    if full_len <= target_len:
        return [0, 0, 0, 0, 0]
    max_scroll = full_len - target_len
    return [
        0,                      # 左/上起点
        max_scroll // 4,        # 1/4
        max_scroll // 2,        # 中点
        3 * max_scroll // 4,    # 3/4
        max_scroll              # 右/下终点
    ]

def run_inference():
    # ---------------------------------------------------------
    # 1. 基础配置与模型加载
    # ---------------------------------------------------------
    model_path = '/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune-modified'
    image_path = "Test.jpg"
    conv_mode = "bunny" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print(f"🚀 [1/4] 初始化架构 (Token: {DEFAULT_IMAGE_TOKEN})...")
    disable_torch_init()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print(f"\n<image> ID: {tokenizer.convert_tokens_to_ids('<image>')}")
    print(f"### ID: {tokenizer.convert_tokens_to_ids('###')}")
    print(f"### endoftext: {tokenizer.convert_tokens_to_ids('<|endoftext|>')}")
    print(f"img_content: {tokenizer.convert_tokens_to_ids('<img_content>')}")

    if DEFAULT_IMAGE_TOKEN not in tokenizer.get_vocab():
        print(f"🚀 [1/4] 初始化架构 (Token: {DEFAULT_IMAGE_TOKEN})...")
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
    
    config = AutoConfig.from_pretrained(model_path)
    
    with torch.device('cuda'):
        model = BunnyPhiForCausalLM(config).to(dtype=dtype)
    model.resize_token_embeddings(len(tokenizer))

    # ---------------------------------------------------------
    # 2. 子塔初始化 (防止 NoneType)
    # ---------------------------------------------------------
    vision_tower = model.get_vision_tower()
    if hasattr(vision_tower, 'dino_vision_tower'):
        vision_tower.dino_vision_tower.load_model()
    if hasattr(vision_tower, 'siglip_vision_tower'):
        vision_tower.siglip_vision_tower.load_model()

    # ---------------------------------------------------------
    # 3. 权重灵魂注入 (Key 对齐修复)
    # ---------------------------------------------------------
    print("💉 [2/4] 注入 0.64 Loss 权重...")
    bin_files = sorted(glob.glob(os.path.join(model_path, "pytorch_model-*.bin")))
    model_keys = model.state_dict().keys()

    for f in bin_files:
        sd = torch.load(f, map_location="cpu")
        new_sd = { (k[6:] if (k.startswith("model.") and k[6:] in model_keys) else k): v for k, v in sd.items() }
        model.load_state_dict(new_sd, strict=False)
        del sd, new_sd

    model.to(device)
    vision_tower.is_loaded = True 
    model.eval()

    # ---------------------------------------------------------
    # 4. 核心：动态六子图切分 (复刻 data_utils.py)
    # ---------------------------------------------------------
    print("🎨 [3/4] 执行动态六子图高清切分...")
    raw_image = Image.open(image_path).convert('RGB')
    w, h = raw_image.size
    target_sz = 378 # 训练时的标准尺寸
    
    # 全局缩略图
    global_img = raw_image.resize((target_sz, target_sz), Image.BILINEAR)
    
    # 计算锚点并提取 5 个局部切片
    x_coords = calculate_anchors(w, target_sz)
    y_coords = calculate_anchors(h, target_sz)
    crops = [
        raw_image.crop((x_coords[0], y_coords[0], x_coords[0] + target_sz, y_coords[0] + target_sz)), # 左上
        raw_image.crop((x_coords[4], y_coords[0], x_coords[4] + target_sz, y_coords[0] + target_sz)), # 右上
        raw_image.crop((x_coords[0], y_coords[4], x_coords[0] + target_sz, y_coords[4] + target_sz)), # 左下
        raw_image.crop((x_coords[4], y_coords[4], x_coords[4] + target_sz, y_coords[4] + target_sz)), # 右下
        raw_image.crop((x_coords[2], y_coords[2], x_coords[2] + target_sz, y_coords[2] + target_sz)), # 正中心
    ]
    
    # 预处理 6 张图，形状变为 [6, 2, 3, 378, 378]
    image_processor = vision_tower.image_processor
    image_tensor = image_processor.preprocess([global_img] + crops, return_tensors='pt')['pixel_values']
    # 最终形状对齐训练: [1, 6, 2, 3, 378, 378]
    image_tensor = image_tensor.unsqueeze(0).to(device, dtype=dtype)
    print(f"📊 Tensor 就绪: {image_tensor.shape}")


    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nDescribe this image in detail.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    input_token_len = input_ids.shape[1]

    # 4. 深度诊断：必须确保看到 -200 夹在中间
    print(f"📊 [物理对齐检查] 总序列长度: {input_token_len}")
    print(f"📊 [物理对齐检查] 图像占位符位置: {torch.where(input_ids == -200)[1].tolist()}")
    # ---------------------------------------------------------
    # 5. 生成回复 (截断 Prompt 防止复读视觉假象)
    print("🎯 [4/4] 正在深度推理...")
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
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 关键：只取出 Assistant 回答的部分
    response = tokenizer.decode(output_ids[0][input_token_len:], skip_special_tokens=True).strip()
    
    print("\n🤖 Bunny 最终回复:")
    print("-" * 50)
    print(response if response else " (模型仍然沉默，请检查 vision_tower 内部拼接逻辑) ")
    print("-" * 50)

if __name__ == "__main__":
    run_inference()