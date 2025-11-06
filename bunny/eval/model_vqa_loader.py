import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, model, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.model = model
        self.model_config = model_config

 # 文件: /mnt/CoBunny/bunny/eval/model_vqa_loader.py (在 CustomDataset 类中)

    def __getitem__(self, index):
        line = self.questions[index]
        print("__getitem__.................",line)

        # 1. 提取图像ID和问题文本
        image_id = line["image_id"]
        qs = line["text"]

        # 2. 构造图像文件名 (解决 TypeError 问题)
        # 错误发生的原因是 image_id 是 int，os.path.join 不能处理 int。
        # 假设 COCO VQA 格式: 12 位零填充的 ID + .jpg 后缀。
        try:
            if isinstance(image_id, int):
                # 构造的文件名会是 "000000163845.jpg" (12位零填充)
                image_file ="COCO_test2015_"
                image_file1 = f"{image_id:012d}.jpg" 
                image_file ="COCO_test2015_" +image_file1
                
            elif isinstance(image_id, str):
                # 如果 image_id 已经是字符串，确保它有 .jpg 后缀
                image_file = image_id if image_id.endswith('.jpg') else image_id + '.jpg'
            else:
                # 极端情况处理
                raise TypeError(f"Unexpected type for image_id: {type(image_id)}")
        except Exception as e:
            # 兜底捕获格式化错误，以防 image_id 无法格式化
            print(f"Error formatting image_id {image_id}: {e}")
            # 尝试回退到 JSONL 中的 'image' 键（如果存在）
            image_file = line.get("image", str(image_id) + ".jpg") 
            

        # 3. 构造提示词 (Prompt)
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 4. 加载图像
        # os.path.join 此时 image_file 已经是字符串，不会报错
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        # 5. 处理图像和 tokenization
        image_tensor = self.model.process_images([image], self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, model, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, model, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    print("model_path is .....",model_path)
    model_name = get_model_name_from_path(model_path)
    print("model_name is .....",model_name)
    #tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
    #                                                                       args.model_type,tp_plan=None)


    from .modeling_bunny_phi import BunnyPhiForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BunnyPhiForCausalLM.from_pretrained(
        "/mnt/Bunny-v1_0-3B",
        low_cpu_mem_usage=True,
        tp_plan=None,
        use_safetensors=True,
        local_files_only=True,
    )
    model = model.cuda()
    #from transformers import AutoImageProcessor
    language_model = model.get_model() # 获取语言模型核心 (PhiForCausalLM)

    if hasattr(language_model, 'mm_projector') and language_model.mm_projector is not None:
        print("Forcibly moving all mm_projector parameters to CUDA...")
        
        # 递归地将所有子模块的参数移动到 CUDA
        for name, module in language_model.mm_projector.named_modules():
            if hasattr(module, 'weight'):
                module.weight.data = module.weight.data.to('cuda:0')
                if module.weight.grad is not None:
                    module.weight.grad.data = module.weight.grad.data.to('cuda:0')
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.to('cuda:0')
                if module.bias.grad is not None:
                    module.bias.grad.data = module.bias.grad.data.to('cuda:0')
    #image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, model, model.config)

    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=model.dtype, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
