import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math

# 保持 Bunny 基础工具的导入
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


import transformers
from transformers.cache_utils import DynamicCache

# 修复 transformers 版本不兼容导致的 get_usable_length 报错
if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, seq_length=None, layer_idx=0):
        """兼容旧版代码的补丁"""
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = get_usable_length
    print("🔧 已自动修复 Transformers.DynamicCache 的版本兼容性补丁")


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, model, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.model = model
        self.model_config = model_config
        
        # 定义一个稳健的图像处理 pipeline (适配 224x224 或 336x336)
        # 根据你训练时的分辨率修改 size，通常是 224 或 336
        self.transform = Compose([
            Resize((224, 224), interpolation=3), # 3 是 bicubic
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                      std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 读取并处理图像
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # 使用上面定义的 transform，不再调用 mm_utils.process_images
        image_tensor = self.transform(image) 

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, image_tensor
    def __len__(self):
        return len(self.questions)

def create_data_loader(questions, image_folder, tokenizer, model, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "MME evaluation typically requires batch_size=1"
    dataset = CustomDataset(questions, image_folder, tokenizer, model, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache

# --- 1. 强制兼容性补丁 ---
if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, seq_length=None, layer_idx=0):
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = get_usable_length



def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # --- 核心修改：动态分配显卡 ---
    # 逻辑：如果你有 8 张卡，chunk_idx 为 0 就用 cuda:0，为 1 就用 cuda:1，以此类推
    num_gpus = torch.cuda.device_count()
    gpu_id = args.chunk_idx % num_gpus 
    device_id = f"cuda:{gpu_id}"
    device = torch.device(device_id)
    
    print(f"🚀 [Chunk {args.chunk_idx}] 正在加载模型到 NVIDIA T4 ({device_id})...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # 强制所有分片进入计算出的 gpu_id，彻底停用自动分片
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map={"": device_id}, # 这里动态使用上面算出来的 device_id
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        local_files_only=True,
        use_cache=True
    )

    # 初始化视觉塔
    vision_tower = model.get_model().get_vision_tower()
    if not getattr(vision_tower, 'is_loaded', False):
        print("💡 正在加载 DINOv3/Oryx 权重...")
        vision_tower.load_model()
    
    # 再次确认全模型都在目标显卡上且为 bf16
    model.to(device=device, dtype=torch.bfloat16)
    
    # 修复 pad_token 警告
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    model.eval()

    # --- 数据准备 ---
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, model, model.config)

    # --- 推理循环 ---
    print(f"📊 显存占用已就绪，开始处理 {len(questions)} 条数据...")
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        # 数据移至 T4
        input_ids = input_ids.to(device=device, non_blocking=True)
        images = image_tensor.to(dtype=torch.bfloat16, device=device, non_blocking=True)

        with torch.inference_mode():
            # 使用 BF16 自动混合精度，节省显存并加速
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": shortuuid.uuid(),
            "model_id": model_name,
            "metadata": {}
        }) + "\n")
        ans_file.flush() 

    ans_file.close()
    print(f"✨ 处理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="bunny")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0) # MME 建议用 0
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)