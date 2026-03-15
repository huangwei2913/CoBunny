import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Sequence, Optional, List

import torch
import transformers
from torch.utils.data import Dataset
from PIL import Image, ImageOps
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

#除了返回被替换成的<img_content>之外，我们还要每一个样本中的影像数量
def preprocess_multimodal(sources, data_args):
    is_multimodal = data_args.is_multimodal
    RAW_JSON_TAG = "<image>"

    for source in sources:
        # --- 重点 1: 无论如何，先给这个样本打个“零图”底色 ---
        # 这样后面不管是纯文本，还是没匹配上，都不会报 KeyError
        source[0]['num_images'] = 0 

        if not is_multimodal:
            continue

        for sentence in source:
            if RAW_JSON_TAG in sentence['value']:
                num_images = sentence['value'].count(RAW_JSON_TAG)
                
                # --- 重点 2: 执行你的替换逻辑 ---
                if num_images == 1:
                    sentence['value'] = sentence['value'].replace(RAW_JSON_TAG, DEFAULT_IMAGE_TOKEN).strip()
                elif num_images > 1:
                    parts = sentence['value'].split(RAW_JSON_TAG)
                    new_val = ""
                    for i in range(num_images):
                        new_val += f"{parts[i]}Image {i+1}: {DEFAULT_IMAGE_TOKEN} "
                    sentence['value'] = (new_val + parts[-1]).strip()
                
                # --- 重点 3: 只有真正匹配到图片了，才更新这个数字 ---
                source[0]['num_images'] = num_images
    
    return sources
#用分词器得到每一个样本中的字符串对应的的input_ids和lables
#每一个样本的
#目前仅仅是支持目前的逻辑支持一个图片对应多个问题和回复（多轮对话）
#conv.append_message 把整个对话历史都拼成了一个长字符串（prompt），然后整个 prompt 只包含一个 <image> 占位符。
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

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     if i == 0: # 只打印这批数据的第一个样本，防止刷屏
        #         print("\n" + "👁️"*20)
        #         print("【DEBUG 1: 原始 Prompt 模板长这样】")
        #         print("请仔细检查里面是 USER: 还是 <|user|>，以及有没有特殊的 System Prompt。")
        #         print("-" * 40)
        #         print(repr(conv.get_prompt())) # 用 repr 打印，连 \n 都能显示出来
        #         print("👁️"*20 + "\n")



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

    # === 调试打印开始 ===
    # 随机选一个样本看一眼（比如第一个）
    debug_idx = 0 
    debug_input = input_ids[debug_idx]
    debug_label = targets[debug_idx]

    #print("\n" + "="*50)
    #print("🚀 [PREPROCESS DEBUG] 检查 Label 遮蔽是否精准:")
    
    decoded_output = []
    hardcore_tokens = [] # 存储 "ID:文本:状态" 的硬核信息
    for token_id, label_id in zip(debug_input, debug_label):
        # 【修改这里】：如果 token_id 是负数（如图像 token -200），手动转成文本
        if token_id < 0:
            token_text = f"<IMG_{token_id}>"
        else:
            token_text = tokenizer.decode([token_id])
        display_text = token_text.replace('\n', '\\n')
        
        if label_id == IGNORE_INDEX:
            decoded_output.append(f"\033[90m{token_text}\033[0m")
            hardcore_tokens.append(f"\033[90m[{token_id}:{display_text}:IGN]\033[0m")
        else:
            decoded_output.append(f"\033[92m{token_text}\033[0m")
            hardcore_tokens.append(f"\033[92m[{token_id}:{display_text}:LBL]\033[0m")
    #print("".join(decoded_output))
    #print("="*50 + "\n")
    #print("\n👉 [2. 机器硬核 Token 模式 (查 BOS/EOS 专用)]：")
    #print(" ".join(hardcore_tokens))
    input_ids = input_ids.view(-1)
    targets = targets.view(-1)

    return dict(input_ids=input_ids, labels=targets)


#获取乐高图的切片
def get_v17_lego_crops(raw_image, target_sz=378, canvas_sz=714):
    """
    统一的 V17 乐高切片逻辑：自动适配任何比例
    """
    w, h = raw_image.size
    aspect_ratio = h / w if w > 0 else 1

    # --- 情况 1：极细长图 (手机截图类) ---
    if aspect_ratio > 1.6 or aspect_ratio < 0.6:
        # 这种情况下，我们沿长边进行 1x5 的线性排布
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
        return crops

    # --- 情况 2：标准比例 (十字咬合类) ---
    else:
        # 也就是你最认可的 V17 四角+中心模式
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
        return crops



Image.MAX_IMAGE_PIXELS = None 
class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str,
                    tokenizer: transformers.PreTrainedTokenizer,
                    data_args: DataArguments,
                    list_data_dict=None
                    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_image_tokens = getattr(data_args, 'mm_vision_tokens', 365)
        MAX_SEQ_LEN = 2048  # 强制硬编码 2048，确保不溢出
        # --- 这里的逻辑是关键 ---
        if list_data_dict is not None:
            # 如果外部传了切分好的名单（train_list 或 val_list），直接用
            raw_data = list_data_dict
        else:
            # 2. 配置硬限制参数
            raw_data = json.load(open(data_path, "r"))

        self.list_data_dict = []
        self.modality_lengths = []
        # 3. 环境预处理：静音与安全设置
        # 临时把 tokenizer 的限制调到无穷大，防止它在扫描时乱报“3480 > 2048”
        old_max_len = tokenizer.model_max_length
        tokenizer.model_max_length = 9999999 
        # 允许 PIL 处理大图，我们手动过滤，不需要它报 Warning
        Image.MAX_IMAGE_PIXELS = None 
        rank0_print(f"📂 [Data Firewall] 正在深度扫描数据集: {data_path}")
        filtered_data = []
        # 计数器
        stats = {"too_long": 0, "empty": 0, "bad_img": 0, "total": len(raw_data)}
        for entry in raw_data:
            # --- 【检查 1：空 Label 过滤】 ---
            has_response = False
            full_text = ""
            if 'conversations' not in entry:
                stats["empty"] += 1
                continue
                
            for conv in entry['conversations']:
                full_text += conv['value'] + " "
                if conv['from'] in ['gpt', 'assistant']:
                    if conv['value'] and conv['value'].strip():
                        has_response = True
            
            if not has_response:
                stats["empty"] += 1
                continue

            # --- 【检查 2：图片尺寸与安全性过滤】 ---
            image_invalid = False
            num_imgs = full_text.count("<image>")
            
            if 'image' in entry:
                image_files = entry['image']
                if not isinstance(image_files, list):
                    image_files = [image_files]
                
                for img_path in image_files:
                    try:
                        if os.path.isabs(img_path) and os.path.exists(img_path):
                            full_path = img_path
                        else:
                            full_path = os.path.join(self.data_args.image_folder, img_path)
                      
                        # 只读 Header，极快
                        with Image.open(full_path) as img:
                            w, h = img.size
                            # 🚨 过滤条件：总像素 > 8000万 或 比例超过 15:1 (防止 OOM 和 畸变)
                            if (w * h) > 80000000 or (w / h) > 15 or (h / w) > 15:
                                image_invalid = True
                                break
                    except Exception:
                        rank0_print(f"🚨 [文件损坏] 无法读取图片: {full_path}, 错误原因: {e}")
                        image_invalid = True
                        break
            
            if image_invalid:
                stats["bad_img"] += 1
                continue

            # --- 【检查 3：Token 长度硬核扫描】 ---
            # 暴力替换法：把 <image> 换成 365 个占位符，模拟真实进入 LLM 的状态
            # 这里用 text_len 配合 image_overhead 复合计算，最稳
            pure_text = full_text.replace("<image>", "")
            text_token_ids = tokenizer.encode(pure_text, add_special_tokens=True)
            text_len = len(text_token_ids)
            
            # 每个图片 365 tokens + 每个图片 45 个模板开销 (处理多图索引和换行)
            total_real_len = text_len + (num_imgs * (self.num_image_tokens + 45))

            # 严格截断：留出 48 个 token 冗余 (给 System Prompt 和 BOS/EOS)
            if total_real_len > (MAX_SEQ_LEN - 48):
                stats["too_long"] += 1
                continue

            # --- 恭喜：通过所有检查 ---
            filtered_data.append(entry)
            # 记录长度，供训练采样器实现 Efficient Batching
            self.modality_lengths.append(total_real_len)

        # 4. 恢复环境设置
        tokenizer.model_max_length = old_max_len
        self.list_data_dict = filtered_data
        
        # 5. 打印最终扫描报告
        rank0_print(f"✅ [Data Firewall] 扫描完成！报告如下：")
        rank0_print(f"   - 原始总数: {stats['total']}")
        rank0_print(f"   - 最终保留: {len(self.list_data_dict)}")
        rank0_print(f"   - 垃圾清理: {stats['total'] - len(self.list_data_dict)}")
        rank0_print(f"     (其中 溢出:{stats['too_long']} | 空白:{stats['empty']} | 坏图:{stats['bad_img']} )")
        
        if len(self.list_data_dict) == 0:
            rank0_print("🚨🚨🚨 极其严重警告：所有数据都被滤掉了！请检查路径或逻辑！")

    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw_entry = self.list_data_dict[i]
        dialog_list = copy.deepcopy(raw_entry['conversations'])
        sources = [dialog_list] # 包装成 [source]
        
        # 1. 文本与多模态占位符处理
        has_image = 'image' in raw_entry
        if has_image:
            sources = preprocess_multimodal(sources, self.data_args)

        # 这里 preprocess 吐出来的 data_dict 包含了 input_ids 和 labels
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        # =================================================================
        # 🛡️ 核心质检逻辑：拦截“截断自杀”样本
        # =================================================================
        max_len = self.tokenizer.model_max_length
        # 模拟 DataCollator 的截断操作，只看模型能看到的那部分 labels
        truncated_labels = data_dict['labels'][:max_len]
        
        # 计算有效 Token 数（不等于 -100 的数量）
        # 如果是 Tensor 就用 .eq(...).sum()，如果是 List 就用 count
        if isinstance(truncated_labels, torch.Tensor):
            valid_label_count = truncated_labels.ne(IGNORE_INDEX).sum().item()
        else:
            valid_label_count = sum(1 for x in truncated_labels if x != IGNORE_INDEX)

        # 如果有效回答长度为 0，说明这个样本在截断后没东西可学
        if valid_label_count == 0:
            # 这里的 rank0_print 建议只在调试阶段开启，不然 22w 数据刷屏很快
            # rank0_print(f"🚮 跳过样本 {i}: 提问太长导致回答被截断。正在寻找下一个...")
            
            # 递归调用：找下一个样本（取模防止越界）
            # 注意：如果坏样本太多，可能导致递归过深，但在 22w 数据里一般不会
            return self.__getitem__((i + 1) % len(self.list_data_dict))
        
        # =================================================================

        # 2. 影像预处理核心逻辑 (只有活下来的样本才会走到这里，节省开销)
        if has_image:
            image_file = raw_entry['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            target_sz = 378

            if os.path.isabs(image_file) and os.path.exists(image_file):
                img_path = image_file
            else:
                img_path = os.path.join(image_folder, image_file)

            def calculate_anchors(full_len, target_len):
                if full_len <= target_len:
                    return [0, 0, 0, 0, 0]
                max_scroll = full_len - target_len
                return [0, max_scroll // 4, max_scroll // 2, 3 * max_scroll // 4, max_scroll]

            try:
                raw_image = Image.open(img_path).convert('RGB')
                global_img = ImageOps.pad(raw_image, (target_sz, target_sz), color=(122, 122, 122))
                crops = get_v17_lego_crops(raw_image, target_sz=378)
                six_images = [global_img] + crops
                sub_image_dict = processor.preprocess(six_images, return_tensors='pt')
                # 统一 Key 为 'image' (单数)，与 Collator 逻辑对齐
                data_dict['image'] = sub_image_dict['pixel_values'].squeeze(0) if sub_image_dict['pixel_values'].dim() == 6 else sub_image_dict['pixel_values']

            except Exception as e:
                rank0_print(f"🚨 图片处理失败: {img_path}, 错误: {e}")
                data_dict['image'] = torch.zeros(6, 2, 3, target_sz, target_sz)
        else:
            data_dict['image'] = torch.zeros(6, 2, 3, 378, 378)
        
        return data_dict

# ---------------------------------------------------------
# 数据整理器 (Padding)
###
## 将一批样本里面的所有样本都做inout_ids的对齐，lables的对齐
###一批只不过送给一个gpu的样本数量
###
# ---------------------------------------------------------
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 0. 安全检查：Tokenizer 是否就绪
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("🚨 Tokenizer 缺少 pad_token_id！请检查初始化代码。")
        
        # 1. 提取 input_ids 和 labels
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        # 2. 强制对齐 (Padding)
        # input_ids 使用 pad_token_id (50296) 填充
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
            
        # labels 使用 -100 填充 (不计算 loss)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        # 3. 截断 (最大长度限制)
        model_max_length = self.tokenizer.model_max_length
        input_ids = input_ids[:, :model_max_length]
        labels = labels[:, :model_max_length]
        
        # 4. 构建 Batch
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 5. 图像堆叠处理 [Batch, 6, 2, 3, 378, 378]
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # =================================================================
        # 🔍 深度调试区 (修复了 RuntimeError 问题)
        # =================================================================
        
        # [检查 A]：截断导致的 "全 -100" 风险
        # 计算每一行有多少个有效 label (即不等于 -100 的个数)
        # 使用 .long() 确保类型安全，使用 .tolist() 转为 Python 列表，彻底避免 Tensor Boolean Error
        valid_counts = (labels != IGNORE_INDEX).sum(dim=1).long().tolist()
        
        has_zero_label_sample = False
        for i, count in enumerate(valid_counts):
            if count == 0:
                has_zero_label_sample = True
                # 这是一个严重警告：说明截断太狠，把回答切没了
                rank0_print(f"🚨 [严重警告] 样本 {i} 的有效 Labels 数量为 0！(全被截断或原本就没有回答)")
                rank0_print(f"   - Input 长度: {len(instances[i]['input_ids'])} -> 截断后: {input_ids.shape[1]}")

        # [检查 B]：Padding 互补逻辑验证 (抽查第一个样本)
        # 找到 input_ids 中所有 pad 的位置
        first_sample_ids = input_ids[0]
        first_sample_labels = labels[0]
        
        # 找到 input_ids 等于 pad_token_id 的索引
        pad_indices = (first_sample_ids == pad_id).nonzero(as_tuple=True)[0]
        
        if len(pad_indices) > 0:
            # 只检查第一个 pad 出现的位置，验证逻辑是否闭环
            chk_idx = pad_indices[0].item()
            
            val_input = first_sample_ids[chk_idx].item()
            val_label = first_sample_labels[chk_idx].item()
            
            # 打印验证信息
            # 只有当 逻辑不对时 才打印警告，或者每隔一定步数打印一次（这里为了调试每次都打）
            # 为了防止刷屏，你可以只在发现不对时打印，但我这里加上是为了让你安心
            # print(f"🔍 [对齐检查] Index {chk_idx}: Input={val_input} (<pad>), Label={val_label} (-100)")
            
            if val_input == pad_id and val_label != IGNORE_INDEX:
                raise ValueError(f"🚨 Padding 逻辑崩溃！Input 是 <pad> ({val_input}) 但 Label 竟然是 {val_label} (不是 -100)！模型会学到错误的东西！")
        
        # [检查 C]：维度检查
        if 'images' in batch and isinstance(batch['images'], torch.Tensor):
             # 确保维度是 6 维
             if batch['images'].dim() != 6:
                 rank0_print(f"⚠️ [图像维度警告] 期望 6 维，实际得到: {batch['images'].shape}")


        # # --- 🚀 终极监控点 ---
        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     #print("\n" + "🚚" * 10)
        #     #print("【DataCollator 发货质检..............................................】")
        #     if 'images' in batch:
        #         print(f"   - 图像批次维度 (Images Shape): {batch['images'].shape}")
        #         # 预期应该是 [Batch, 6, 2, 3, 378, 378]
        #     if 'input_ids' in batch:
        #         print(f"   - 文本批次维度 (Input_ids Shape): {batch['input_ids'].shape}")
        #         # 这里的 SeqLen 应该是这个 Batch 里最长样本的长度
        #     print("🚚" * 10 + "\n")


        return batch

def make_supervised_data_module(tokenizer, data_args) -> Dict:
    rank0_print("📂 [Data] 正在读取数据名单...")
    
    # 1. 先读出原始的 list_data_dict (这是最轻量级的字符串列表)
    import json
    list_data_dict = json.load(open(data_args.data_path, "r"))
    
    # 2. 手动切分名单
    import random
    random.seed(42)
    random.shuffle(list_data_dict) # 随机打乱
    
    val_size = 2000
    val_list = list_data_dict[:val_size]
    train_list = list_data_dict[val_size:]

    rank0_print(f"📂 [Data] 名单切分完成: 训练集 {len(train_list)}, 验证集 {len(val_list)}")

    # 3. 分别创建两个独立的 LazySupervisedDataset 实例
    # 修改你的 Dataset 类，让它支持直接传入 list_data 而不是只读路径
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path, # 路径留着备用
        data_args=data_args,
        list_data_dict=train_list  # 直接传切好的列表
    )
    
    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        list_data_dict=val_list
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )