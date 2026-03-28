import torch
import os
import sys
import warnings
from PIL import Image, ImageOps
from transformers import AutoTokenizer, AutoConfig, TextStreamer
from bunny.model.language_model.bunny_llama import BunnyLlamaForCausalLM
from bunny import conversation as conversation_lib
from bunny.constants import DEFAULT_IMAGE_TOKEN
from transformers.cache_utils import DynamicCache  # 引入缓存管理
# ================= 配置区 =================
BUNNY_REPO_PATH = '/mnt/CoBunny'
if BUNNY_REPO_PATH not in sys.path:
    sys.path.insert(0, BUNNY_REPO_PATH)

MODEL_PATH = '/mnt/CoBunny/checkpoints-stage3/llama'
IMAGE_PATH = "/mnt/CoBunny/bunny/model/language_model/testt.jpg"
DEVICE = "cuda:0" 
IMAGE_TOKEN_INDEX = -200 
warnings.filterwarnings("ignore")

# ================= 逻辑函数 (保持不变) =================
def get_inference_prompt(question):
    conv = conversation_lib.conv_templates["bunny"].copy()
    message = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def tokenizer_image_token_custom(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<img_content>')]
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)



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





# ================= 推理核心逻辑 =================
def run_inference():
    print(f"🚀 [1/4] 正在手动初始化 Llama-3.2-1B (避开 Meta Tensor)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 强制不使用底层自动加速加载，手动在内存创建真实权重张量
    model = BunnyLlamaForCausalLM(config) 
    
    # 手动加载权重文件 (pytorch_model.bin 或 model.safetensors)
    bin_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    print(f"📦 正在从硬盘加载权重到内存...")
    state_dict = torch.load(bin_path, map_location="cpu")
    
    # 把权重塞进刚才创建的实心模型里
    model.load_state_dict(state_dict, strict=False)
    
    # 现在它是一个有真实数据的 CPU 模型了，可以安全移动到 GPU
    print(f"⚙️ 正在将模型转为 FP16 并移至 {DEVICE}...")
    model = model.half().to(DEVICE)
    model.eval()

    vision_tower = model.get_vision_tower()
    if not getattr(vision_tower, "is_loaded", False):
        vision_tower.load_model()
    

    # === 修改 1：精准获取训练时注入的 Pad Token ===
    pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    if pad_token_id == tokenizer.unk_token_id or pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 128001

    # === 修改 2：构建多重停止符列表 ===
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 128001
    stop_token_ids = [tokenizer.eos_token_id]
    for tid in [tokenizer.convert_tokens_to_ids("<|eot_id|>"), tokenizer.convert_tokens_to_ids("<|end_of_text|>")]:
        if tid is not None: stop_token_ids.append(tid)
  
    print(f"🖼️ [2/4] 处理图像与 Prompt...")
    image_tensor = get_six_crops(IMAGE_PATH, vision_tower.image_processor)
    if image_tensor.dim() == 4 or image_tensor.dim() == 5: 
        # 兼容单塔或双塔，保证最终是 1 开头的 Batch 维度
        image_tensor = image_tensor.unsqueeze(0) if image_tensor.shape[0] != 1 else image_tensor
    
    prompt = get_inference_prompt("What is in the image?please describe it clearly.")
    input_ids = tokenizer_image_token_custom(prompt, tokenizer)

    print(f"🧠 [3/4] 模型开始生成 ↓")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    attention_mask = torch.ones_like(input_ids).to(DEVICE)

    with torch.inference_mode():
        model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=torch.ones_like(input_ids).to(DEVICE),
            do_sample=False, 
            max_new_tokens=512,          # 缩短长度，1B 模型长了必乱说
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            
            # --- 关键参数：解决 CUDA 越界报错 ---
            # 设为 1.0 (不惩罚)，因为 1B 模型在处理含有 -200 的 input_ids 时
            # 无法正确处理 repetition_penalty 的索引映射
            repetition_penalty=1.0, 
            
            # --- 关键参数：解决悬停和 <|path| 复读 ---
            # 既然模型喜欢复读训练集的 path，我们手动截断
            # 强制让它在生成较短内容后停止
            min_new_tokens=1,
            streamer=streamer
        )
    print("\n✅ [4/4] 推理任务完成。")

if __name__ == "__main__":
    run_inference()