import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Sequence, Optional, List

import torch
import transformers
from torch.utils.data import Dataset
from PIL import Image

# 引入我们定义好的常量
from bunny.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from bunny import conversation as conversation_lib
from bunny.util.mm_utils import tokenizer_image_token

# 分布式打印工具
def rank0_print(*args):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)
    mm_vision_tokens: int = field(default=365)  # 明确告诉数据加载器，视觉塔输出多少个tokens

def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    # 原始数据里的标签 (你 JSON 文件里写的是这个)
    RAW_JSON_TAG = "<image>"  #被污染的占位符

    for source in sources:
        for sentence in source:
            # 检查是否有图像占位符
            if RAW_JSON_TAG in sentence['value']:
                # 统计图片数量
                num_images = sentence['value'].count(RAW_JSON_TAG)
                
                # 情况 A: 单图 (保持简单，直接替换为 <img_content>)
                if num_images == 1:
                    # 先替换成新 Token 名字
                    new_val = sentence['value'].replace(RAW_JSON_TAG, DEFAULT_IMAGE_TOKEN)
                    # 移除多余的换行符，确保 <img_content> 在最前面或最合适的位置
                    # 这一步是为了防止 "<img_content>\n\nText" 这种双换行
                    sentence['value'] = new_val.strip()
                
                # 情况 B: 多图 (注入 "Image 1:" 锚点)
                elif num_images > 1:
                    parts = sentence['value'].split(RAW_JSON_TAG)
                    new_val = ""
                    for i in range(num_images):
                        # 注入逻辑: "Image 1: <img_content> "
                        # 注意末尾加个空格，帮助分词器隔离
                        new_val += f"{parts[i]}Image {i+1}: {DEFAULT_IMAGE_TOKEN} "
                    
                    new_val += parts[-1]
                    sentence['value'] = new_val.strip()
                    
    return sources

def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    # 加载对话模板 (bunny 模式)
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        
        # 此时 conv.get_prompt() 拿到的已经是 preprocess_multimodal 处理过
        # 带有 <img_content> 和 Image 1: ... 的文本了
        conversations.append(conv.get_prompt())

    # Tokenize 逻辑
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    
    # Mask 掉 User 的提问，只训练 Assistant 的回答
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 0 # Phi-1.5/Bunny 通常从 0 开始，如果有 BOS token 需改为 1
        
        # 如果 tokenizer 自动加了 BOS (比如 Phi-3), 这里要做调整
        # 对于标准的 Phi-1.5, 它没有强制 BOS，所以 cur_len = 0 是安全的
        
        for i, rou in enumerate(rounds):
            if rou == "": break

            parts = rou.split(sep)
            if len(parts) != 2: break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            
            # Phi-1.5 特殊修正：长度对齐
            round_len += 1 

            # 将 instruction (User部分) 设为 -100 (IGNORE)
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
            
        target[cur_len:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        rank0_print(f"Loading data from {data_path}...")
        list_data_dict = json.load(open(data_path, "r"))
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        # =================================================
        # 🚨 长度过滤核心逻辑 (防止 2048 溢出导致幻觉)
        # =================================================
        # 1. 设置单图消耗的 Token 数 (双塔输出 365)
        # 净增量 = 365 - 1 (原本的<img_content>占1个) = 364
        self.num_image_tokens = getattr(data_args, 'mm_vision_tokens', 365) #我们的视觉塔最大输出多少个token
        MAX_SEQ_LEN = tokenizer.model_max_length   
        PATCH_TOKENS = self.num_image_tokens 
        MAX_LEN = MAX_SEQ_LEN

        filtered_data = []
        discarded_count = 0
        
        rank0_print("Filtering data based on token length...")
        
        for entry in list_data_dict:
            full_text = ""
            for conv in entry['conversations']:
                full_text += conv['value']
            
            # 统计原始 JSON 里的图片数
            num_imgs = full_text.count("<image>")
            
            # 粗略计算文本长度 (字符数/3.5 是个经验值，或者直接用 encode 抽样)
            # 为了精准，我们这里做个简单的字符串替换模拟
            # "Image 1: " 大概占 4 个 token，如果多图，我们要加上这部分开销
            anchor_overhead = num_imgs * 5 if num_imgs > 1 else 0
            
            # 文本本身的 token 数 (不含图片特征)
            text_tokens = len(tokenizer.encode(full_text))
            
            # 真实总长度 = 文本 + 锚点开销 + 图片特征占位
            total_len = text_tokens + anchor_overhead + (num_imgs * (PATCH_TOKENS - 1))
            
            # 留一点 Buffer (比如 10 个 token)
            if total_len < (MAX_LEN - 10):
                filtered_data.append(entry)
            else:
                discarded_count += 1

        self.list_data_dict = filtered_data
        rank0_print(f"Loaded {len(self.list_data_dict)} samples. Discarded {discarded_count} samples due to length.")
        if not hasattr(self, 'modality_lengths'):
        # 如果没有这个属性，我们根据是否存在 'image' 字段给出一个参考长度
        # 图像 token 比较长，我们假设带图的为 1000，不带图的为 100
            self.modality_lengths = []
            for item in self.list_data_dict:
                if 'image' in item:
                    self.modality_lengths.append(400)
                else:
                    self.modality_lengths.append(100)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
            
        # ---------------------------------------------------------
        # 🚨 高清“动态六子图”切分核心逻辑 (378x378 方案)
        # ---------------------------------------------------------
        if 'image' in sources[0]:
            image_file = sources[0]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            
            # 支持多图列表或单图路径
            image_files = [image_file] if isinstance(image_file, str) else image_file
            
            pixel_values_list = []
            target_sz = 378  # 设定子图标准尺寸

            # 内部辅助函数：计算动态采样锚点，处理任意比例影像
            def calculate_anchors(full_len, target_len):
                if full_len <= target_len:
                    return [0, 0, 0, 0, 0]
                max_scroll = full_len - target_len
                return [
                    0,                      # 起点 (左/上)
                    max_scroll // 4,        # 1/4处
                    max_scroll // 2,        # 中点
                    3 * max_scroll // 4,    # 3/4处
                    max_scroll              # 终点 (右/下)
                ]

            for img_path in image_files:
                try:
                    # 1. 加载原始高清图
                    raw_image = Image.open(os.path.join(image_folder, img_path)).convert('RGB')
                    w, h = raw_image.size
                    
                    # 2. 生成全局缩略图 (保持语义完整性)
                    global_img = raw_image.resize((target_sz, target_sz), Image.BILINEAR)
                    
                    # 3. 计算动态锚点坐标
                    x_coords = calculate_anchors(w, target_sz)
                    y_coords = calculate_anchors(h, target_sz)
                    
                    # 4. 提取 5 个高清局部切片 (四角 + 绝对中心)
                    # 这种方式直接从原图扣取像素，最大程度保留手机拍照细节
                    crops = [
                        raw_image.crop((x_coords[0], y_coords[0], x_coords[0] + target_sz, y_coords[0] + target_sz)), # 左上
                        raw_image.crop((x_coords[4], y_coords[0], x_coords[4] + target_sz, y_coords[0] + target_sz)), # 右上
                        raw_image.crop((x_coords[0], y_coords[4], x_coords[0] + target_sz, y_coords[4] + target_sz)), # 左下
                        raw_image.crop((x_coords[4], y_coords[4], x_coords[4] + target_sz, y_coords[4] + target_sz)), # 右下
                        raw_image.crop((x_coords[2], y_coords[2], x_coords[2] + target_sz, y_coords[2] + target_sz)), # 正中心
                    ]
                    
                    # 5. 组合成 6 个子图序列并进行预处理
                    six_images = [global_img] + crops
                    # 对 6 张图执行 Normalize 和 ToTensor
                    sub_image_dict = processor.preprocess(six_images, return_tensors='pt')
                    
                    # 堆叠为 [6, 2，3, ,374, 374] 的张量包
                    pixel_values_list.append(sub_image_dict['pixel_values'])
                    
                except Exception as e:
                    print(f"Error loading image: {img_path}, {e}")
                    # 异常兜底：返回全黑的 6 子图张量
                    pixel_values_list.append(torch.zeros(6, 2, 3, target_sz, target_sz))

            # # 最终堆叠 [Num_Images, 6, 2, 3, 378, 378]
            # 这确保了 Model Forward 能够接收到 6 维张量
            image = torch.stack(pixel_values_list)
            
            # 调用原本的对话预处理逻辑
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # 文本 Tokenize 过程
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
            
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])

        # 将处理好的 6 维图像张量放入 data_dict
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # 非图样本的兜底填充，保持维度一致
            data_dict['image'] = torch.zeros(1, 6, 2, 3, 378, 378)
            
        return data_dict
# ---------------------------------------------------------
# 数据整理器 (Padding)
# ---------------------------------------------------------
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # 这是一个紧急出口，提醒你必须去外层初始化它
            raise ValueError("Tokenizer 缺少 pad_token！请确保在加载数据前执行了 tokenizer.add_tokens(['<pad>'])")
        
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        # 这里的 Padding 使用你新加的 <pad> (ID 50296)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
            
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        # 再次截断，双重保险
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def make_supervised_data_module(tokenizer, data_args) -> Dict:
    rank0_print("📂 [Data] 正在加载全量数据集 (2M)...")
    
    full_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    
    # --- 采用最稳健的切分方式，不使用随机索引拷贝 ---
    val_size = 2000
    train_size = len(full_dataset) - val_size
    
    # random_split 本身很快
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 关键补丁：只引用大列表，不进行循环拷贝！
    # 这样既保证了属性存在，又避免了 200 万次循环导致的悬停
    for dataset in [train_dataset, eval_dataset]:
        dataset.data_args = full_dataset.data_args
        dataset.modality_lengths = full_dataset.modality_lengths
        dataset.list_data_dict = full_dataset.list_data_dict
        dataset.tokenizer = full_dataset.tokenizer

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    rank0_print(f"✅ [Data] 准备就绪，开始训练...")
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

