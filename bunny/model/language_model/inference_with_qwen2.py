import torch
import os
from transformers import AutoTokenizer, AutoConfig
from bunny.model.language_model.bunny_qwen import BunnyQwen2ForCausalLM
from PIL import Image, ImageOps  # 必须引入 ImageOps 用于 pad 操作
from bunny import conversation as conversation_lib
from bunny.constants import DEFAULT_IMAGE_TOKEN
def get_inference_prompt(question):
    # 1. 强制获取 bunny 模版 (对应 --version bunny)
    conv = conversation_lib.conv_templates["bunny"].copy()
    
    # 2. 模仿训练时的输入逻辑
    # 注意：训练时 preprocess_multimodal 会把 <image> 换成 <img_content>
    # 这里的 DEFAULT_IMAGE_TOKEN 应该就是 "<img_content>"
    message = DEFAULT_IMAGE_TOKEN + "\n" + question
    
    conv.append_message(conv.roles[0], message) # USER: <img_content>\n{question}
    conv.append_message(conv.roles[1], None)    # ASSISTANT:
    
    # 3. 拿到 100% 对齐的字符串
    prompt = conv.get_prompt()
    return prompt
# 配置
#MODEL_PATH = '/mnt/conda_data/checkpoints-pretrain/pretrain_stage1_modified/checkpoint-31216'
MODEL_PATH = '/mnt/conda_data/checkpoints-finetuned/bunny-qwen2_finally'
IMAGE_PATH = "/mnt/CoBunny/bunny/model/language_model/cloth.jpg"
MYQUESTION = "Describe the image clearly."
DEVICE = "cuda:0"
IMAGE_TOKEN_INDEX = -200 
TARGET_ID = -200  # 锁死的逻辑 ID




# --- 定义全局常量 ---

def tokenizer_image_token_custom(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    """
    将文本按 <img_content> 切开，中间插入逻辑 ID 
    """
    # 1. 编码文本块
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<img_content>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    
    # 2. 处理 BOS Token
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])


    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to("cuda:0")





def get_six_crops(image_path, processor):
    """
    6图切分 - 完全对齐训练时的 V17 乐高切片逻辑
    """
    raw_image = Image.open(image_path).convert('RGB')
    target_sz = 378
    canvas_sz = 714
    
    # ==========================================
    # 1. 全局图逻辑 (与训练严格一致)
    # ==========================================
    global_img = ImageOps.pad(raw_image, (target_sz, target_sz), color=(122, 122, 122))

    # ==========================================
    # 2. 局部切片逻辑 (等同于训练的 get_v17_lego_crops)
    # ==========================================
    w, h = raw_image.size
    aspect_ratio = h / w if w > 0 else 1

    # --- 情况 1：极细长图 (手机截图类) ---
    if aspect_ratio > 1.6 or aspect_ratio < 0.6:
        main_dim = h if aspect_ratio > 1.6 else w
        cross_dim = w if aspect_ratio > 1.6 else h
        
        # 宽度/高度对齐到 target_sz
        scale = target_sz / cross_dim
        new_main = int(main_dim * scale)
        
        if aspect_ratio > 1.6:
            resized = raw_image.resize((target_sz, new_main), Image.Resampling.LANCZOS)
        else:
            resized = raw_image.resize((new_main, target_sz), Image.Resampling.LANCZOS)
            
        crops = []
        step = (new_main - target_sz) / 4 if new_main > target_sz else 0
        for i in range(5):
            s = int(i * step)
            if aspect_ratio > 1.6:
                crops.append(resized.crop((0, s, target_sz, s + target_sz)))
            else:
                crops.append(resized.crop((s, 0, s + target_sz, target_sz)))

    # --- 情况 2：标准比例 (十字咬合类) ---
    else:
        scale = canvas_sz / max(w, h)
        curr_w, curr_h = int(w * scale), int(h * scale)
        resized_714 = raw_image.resize((curr_w, curr_h), Image.Resampling.LANCZOS)

        def get_coords(cur, tgt):
            return (0, 0) if cur <= tgt else (0, cur - tgt)

        x_low, x_high = get_coords(curr_w, target_sz)
        y_low, y_high = get_coords(curr_h, target_sz)
        x_mid, y_mid = (curr_w - target_sz) // 2, (curr_h - target_sz) // 2

        coords = [(x_low, y_low), (x_high, y_low), (x_low, y_high), (x_high, y_high), (x_mid, y_mid)]
        
        crops = []
        for lx, ly in coords:
            crop = resized_714.crop((lx, ly, lx + target_sz, ly + target_sz))
            if crop.size != (target_sz, target_sz):
                # 最后的补白防线
                pad = Image.new('RGB', (target_sz, target_sz), (122, 122, 122))
                pad.paste(crop, (0, 0))
                crop = pad
            crops.append(crop)

    # ==========================================
    # 3. 拼接并送入 Processor
    # ==========================================
    six_images = [global_img] + crops
    pixel_values = processor.preprocess(six_images, return_tensors='pt')['pixel_values']
    
    if pixel_values.dim() == 5:
        pixel_values = pixel_values.unsqueeze(0)
        
    return pixel_values.to(DEVICE, dtype=torch.float16)




def get_processed_images(image_path, processor):
    # 按照 Bunny 的标准切分逻辑加载图片
    raw_image = Image.open(image_path).convert('RGB')
    target_sz = 378
    # 简化示例：仅返回 global_img。如需 6 图切分，请复用你之前的 get_six_crops 函数
    pixel_values = processor.preprocess(raw_image, return_tensors='pt')['pixel_values']
    return pixel_values.to(DEVICE, dtype=torch.float16)





def run_inference():
    print(f"🚀 [1/5] 正在通过“手动灌顶法”加载模型...")
    
    # 1. 直接加载 Config
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 2. 核心：在 CPU 上初始化一个“实心”模型（不使用 from_pretrained）
    # 这样可以确保参数不是 Meta Tensor，而是真正的 CPU 内存
    model = BunnyQwen2ForCausalLM(config)
    
    # 3. 手动载入那 4GB 的真材实料
    bin_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    print(f"📦 [2/5] 正在从硬盘灌入 4GB 权重: {bin_path}")
    state_dict = torch.load(bin_path, map_location="cpu")
    
    # strict=True 表示我们要完美对齐，既然扫描是完美的，这里绝对没问题
    model.load_state_dict(state_dict, strict=True)
    
    # 4. 搬运到 GPU
    print(f"🚀 [3/5] 正在整体搬运至 {DEVICE}...")
    model.to(DEVICE, dtype=torch.float16)
    
    # 特别注意：虽然加载了权重，但要触发子塔的内部初始化（比如 Processor）
    vision_tower = model.get_vision_tower()
    if not getattr(vision_tower, "is_loaded", False):
        vision_tower.load_model() 
    vision_tower.to(DEVICE, dtype=torch.float16)

    # 5. 开始推理流程
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    
    print(f"🖼️ [4/5] 准备图像与提示词: {IMAGE_PATH}")
    image_tensor = get_six_crops(IMAGE_PATH, vision_tower.image_processor)
    # 确保维度 [1, 6, 3, 378, 378]
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE, dtype=torch.float16)

    # 构造 Prompt (对齐训练时的 <img_content>)
    question = MYQUESTION
    prompt = get_inference_prompt(question)
    
    input_ids = tokenizer_image_token_custom(prompt, tokenizer, IMAGE_TOKEN_INDEX).to(DEVICE)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    print(f"🪄 [5/5] 模型思考中... (占位符数: {(input_ids == -200).sum().item()})")
      # 🛡️ 4. 解决 Pad Token 致命伤：使用 Stage 3 新增的 <pad>
    # 在 train_stage3.py 中你添加了 <pad>，现在必须显式使用它
    pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    if pad_token_id == tokenizer.unk_token_id:
        # 如果没搜到新 pad，尝试用原来的 eos 兜底
        pad_token_id = tokenizer.eos_token_id

    stop_token_ids = [tokenizer.eos_token_id]
    print(f"🚀 [设备检查] Model Device: {next(model.parameters()).device}")
    print(f"🚀 [设备检查] Input IDs Device: {input_ids.device}")
    attention_mask = torch.ones_like(input_ids).to(DEVICE)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=256, # 先看短描述
            # 必须传 mask，防止 pad/eos 混淆
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )

    

    clean_output_ids = [idx for idx in output_ids[0].tolist() if idx >= 0]
    
    # 现在解码就绝对不会报 TypeError 了
    response = tokenizer.decode(clean_output_ids, skip_special_tokens=True)
    
    print(f"\n📄 模型输出全文:\n{response}")

    if "ASSISTANT:" in response:
        answer = response.split("ASSISTANT:")[-1].strip()
        print(f"\n✅ 最终 AI 回答:\n{answer}")
    else:
        # Stage 1 没生成标志很正常，我们直接看 response 就行
        print("\n⚠️ 提示：模型处于预训练阶段，可能直接在续写描述而没有输出标志。")



if __name__ == "__main__":
    run_inference()

