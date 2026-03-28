import argparse
import torch
import os
import json
import math
from tqdm import tqdm
import shortuuid
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache

# --- 引入 Bunny 相关组件 ---
from bunny.constants import DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates
from bunny.util.utils import disable_torch_init
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM

# 全局常量，强制对齐你的微调脚本
IMAGE_TOKEN_INDEX = -200

# 🔧 修复 transformers 版本不兼容导致的 get_usable_length 报错
if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, seq_length=None, layer_idx=0):
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = get_usable_length


# --- 工具函数 ---
def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def tokenizer_image_token_custom(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    """
    完全对齐你推理脚本的 Tokenizer 逻辑，确保 <img_content> 被正确替换为逻辑 ID
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    return torch.tensor(input_ids, dtype=torch.long)


# --- 数据集类 ---
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, vision_tower, conv_mode="bunny"):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = vision_tower.image_processor
        self.conv_mode = conv_mode

    def get_six_crops_tensor(self, image_path):
        """完全对齐训练时的 V17 乐高切片逻辑 (6-crop)"""
        raw_image = Image.open(image_path).convert('RGB')
        target_sz = 378
        canvas_sz = 714
        
        # 1. 全局图
        global_img = ImageOps.pad(raw_image, (target_sz, target_sz), color=(122, 122, 122))

        # 2. 局部切片
        w, h = raw_image.size
        aspect_ratio = h / w if w > 0 else 1
        crops = []

        if aspect_ratio > 1.6 or aspect_ratio < 0.6:
            main_dim = h if aspect_ratio > 1.6 else w
            cross_dim = w if aspect_ratio > 1.6 else h
            scale = target_sz / cross_dim
            new_main = int(main_dim * scale)
            
            if aspect_ratio > 1.6:
                resized = raw_image.resize((target_sz, new_main), Image.Resampling.LANCZOS)
            else:
                resized = raw_image.resize((new_main, target_sz), Image.Resampling.LANCZOS)
                
            step = (new_main - target_sz) / 4 if new_main > target_sz else 0
            for i in range(5):
                s = int(i * step)
                if aspect_ratio > 1.6:
                    crops.append(resized.crop((0, s, target_sz, s + target_sz)))
                else:
                    crops.append(resized.crop((s, 0, s + target_sz, target_sz)))
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
            for lx, ly in coords:
                crop = resized_714.crop((lx, ly, lx + target_sz, ly + target_sz))
                if crop.size != (target_sz, target_sz):
                    pad = Image.new('RGB', (target_sz, target_sz), (122, 122, 122))
                    pad.paste(crop, (0, 0))
                    crop = pad
                crops.append(crop)

        # 3. 合并并交由 Processor 处理
        six_images = [global_img] + crops
        pixel_values = self.image_processor.preprocess(six_images, return_tensors='pt')['pixel_values']
        return pixel_values

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        question = line["text"]
        
        # 组装 Prompt
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 处理图像
        image_path = os.path.join(self.image_folder, image_file)
        image_tensor = self.get_six_crops_tensor(image_path) 

        # 处理 Token
        input_ids = tokenizer_image_token_custom(prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
        
        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)

def create_data_loader(questions, image_folder, tokenizer, vision_tower, args, batch_size=1, num_workers=4):
    assert batch_size == 1, "MME 评估必须使用 batch_size=1"
    dataset = CustomDataset(questions, image_folder, tokenizer, vision_tower, args.conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


# --- 核心推理循环 ---
def eval_model(args):
    disable_torch_init()
    MODEL_PATH = os.path.expanduser(args.model_path)
    model_name = os.path.basename(MODEL_PATH)
    
    # 动态设备分配
    num_gpus = torch.cuda.device_count()
    device_id = f"cuda:{args.chunk_idx % num_gpus}"
    DEVICE = torch.device(device_id)
    
    print(f"🚀 [1/5] 正在通过“手动灌顶法”初始化模型...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = BunnyPhiForCausalLM(config)
    
    bin_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    print(f"📦 [2/5] 正在从硬盘灌入 4GB 权重: {bin_path}")
    state_dict = torch.load(bin_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    
    print(f"🚀 [3/5] 正在搬运至 {DEVICE} (FP16)...")
    model.to(DEVICE, dtype=torch.float16)
    
    vision_tower = model.get_vision_tower()
    if not getattr(vision_tower, "is_loaded", False):
        print("💡 正在加载视觉塔权重...")
        vision_tower.load_model() 
    vision_tower.to(DEVICE, dtype=torch.float16)

    print(f"🔤 [4/5] 正在配置 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    
    # 🛡️ 解决 Pad Token 致命伤
    pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    if pad_token_id == tokenizer.unk_token_id:
        pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = pad_token_id

    model.eval()

    # --- 数据准备 ---
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    print(f"🖼️ [5/5] 正在准备数据集...")
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, vision_tower, args)

    # --- 推理循环 ---
    print(f"\n✅ 准备就绪，开始对 {len(questions)} 条 MME 数据进行测评！")
    
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(DEVICE)
        images = image_tensor.to(DEVICE, dtype=torch.float16)
        
        # 确保维度 [batch=1, 6, 3, 378, 378]
        if images.dim() == 4:
            images = images.unsqueeze(0)
            
        attention_mask = torch.ones_like(input_ids).to(DEVICE)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=False,              # MME 标准评估必须关闭随机性
                max_new_tokens=args.max_new_tokens,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=model.config.pad_token_id,
                use_cache=True                # 开启 KV Cache 加速
            )

        # 提取当前生成的 token (清洗掉负数 ID 防报错)
        clean_output_ids = [idx for idx in output_ids[0].tolist() if idx >= 0]
        response = tokenizer.decode(clean_output_ids, skip_special_tokens=True)
        
        # 按照你的逻辑提取 ASSISTANT 后面的真实回答
        if "ASSISTANT:" in response:
            answer = response.split("ASSISTANT:")[-1].strip()
        else:
            answer = response.strip()

        # 写入 MME 指定格式的 JSONL
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": answer,
            "answer_id": shortuuid.uuid(),
            "model_id": model_name,
            "metadata": {}
        }) + "\n")
        ans_file.flush() 

    ans_file.close()
    print(f"✨ MME 推理完成！结果已保存至: {answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="模型文件夹路径")
    parser.add_argument("--image-folder", type=str, required=True, help="MME_Benchmark 图像根目录")
    parser.add_argument("--question-file", type=str, required=True, help="MME 的 jsonl 问题文件")
    parser.add_argument("--answers-file", type=str, required=True, help="输出的预测结果文件")
    parser.add_argument("--conv-mode", type=str, default="bunny", help="对话模板")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)