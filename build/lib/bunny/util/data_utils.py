import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Sequence, Optional

import torch

import transformers

from bunny.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from torch.utils.data import Dataset

from bunny import conversation as conversation_lib

from bunny.util.mm_utils import tokenizer_image_token

from PIL import Image


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)



#主要是对类似于对话数据进行处理
""" [
  [
    {"from": "human", "value": "<image>\nWhat is this?"},
    {"from": "gpt", "value": "Piece of dark jeans fabric Royalty Free Stock Photography"}
  ]
] """

def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

            replace_token = DEFAULT_IMAGE_TOKEN

            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources





# 首先看到conv_bunny = Conversation(
#     system="A chat between a curious user and an artificial intelligence assistant. "
#            "The assistant gives helpful, detailed, and polite answers to the user's questions.",
#     roles=("USER", "ASSISTANT"),
#     version="bunny",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.TWO,
#     sep=" ",
#     sep2="<|endoftext|>",
# )模版

# sources = [
#   {"from": "human", "value": "<image>\nWhat is this?"},
#   {"from": "gpt", "value": "Piece of dark jeans fabric Royalty Free Stock Photography"}
# ]
# 这段代码通过对“多轮对话”以及“文本加图片”的多模态输入，利用特定分隔符拆分成“提问”和“回答”，
# 利用tokenizer计算每部分token数量，进而对标签进行屏蔽（mask），
# 让模型只训练正确生成回答部分，提问部分不参与loss计算，保证训练有效且高效。
def preprocess_bunny(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

# 系统提示文本##
# human: <image>
# What is this?###
# gpt: Piece of dark jeans fabric Royalty Free Stock Photography###
# （根据get_prompt具体逻辑和sep2设置，分隔符会交替拼接）

# conversations列表会保存这个拼接的字符串。

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
    
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        end_token_cnt = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1
            end_token_cnt += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



# 拆出多轮对话的每个“提问+回答”块；

# 分别计算token数量；

# 对“提问”部分mask不计算loss，训练时只让模型预测“回答”部分；

# 防止超长和token不匹配带来训练异常。

def preprocess_bunny_with_bos(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

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

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        end_token_cnt = 0
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            end_token_cnt += 1
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# sources = [
#   {"from": "human", "value": "<image>\nWhat is this?"},
#   {"from": "gpt", "value": "Piece of dark jeans fabric Royalty Free Stock Photography"}
# ]
def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)


    #每次都是选择回答部分
    # tokenize conversations conversations = "<image>Piece of dark jeans fabric Royalty Free Stock Photography###"
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    #保留首个bos，然后将-200插入到这个token序列中，形成一个完整的tokens序列
    targets = copy.deepcopy(input_ids) #
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))  #先对样本中第一个对话消息的文本（即图片标记source['value']即"<image>"）做tokenize，计算得到它对应的token数量
        target[:tokenized_len] = IGNORE_INDEX  # 把标签序列 target 的这部分（图片对应的token区间）全部设置成 IGNORE_INDEX（通常是-100），这个索引是训练时告诉损失函数忽略计算这部分token的预测误差。

    return dict(input_ids=input_ids, labels=targets)  #封装plain text, 特别是一个样本中的回答部分

#input_ids 是给模型的输入，包含了图片特殊token和回答文本token的完整序列。
#labels 是训练目标，图片对应token被屏蔽为 IGNORE_INDEX，其余回答文本token作为训练的正确答案。
#这样的设计使得问题中的图片信息作为上下文输入被模型看到，但训练时不强制模型去预测图片token，从而避免干扰训练，提高训练效率和效果。
#这是多模态语言模型训练时常用的思路，让模型在看到“图片token + 回答文本”的条件下，专注于学会生成正确回答。
#preprocess_plain 函数实际上是一个较简单的训练逻辑例子，输入仅包含 <image> 和回答的拼接文本，不含问题文本本体，这和你理解的期望“问题作为上下文输入”不完全一致，但符合简单多模态训练的设计思路。


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)

    if conversation_lib.default_conversation.version == "bunny":
        return preprocess_bunny(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version in {"minicpm", "llama"}:
        return preprocess_bunny_with_bos(sources, tokenizer, has_image=has_image)
    # temporarily fix
    # Phi-3 June 2024 Update changes bos_token behavior
    elif conversation_lib.default_conversation.version == "phi3":
        if len(tokenizer('').input_ids) == 0:
            return preprocess_bunny(sources, tokenizer, has_image=has_image)
        else:
            return preprocess_bunny_with_bos(sources, tokenizer, has_image=has_image)


#首先读json数据文件
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        print("the data path is ........................................hhhhhhhhhhhhh",data_path)
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)



# "id": "0010278167", "image": "0010278167.jpg", "conversations":
#   [{"from": "human", "value": "<image>\nWhat is this?"}, 
#    {"from": "gpt", "value": "Piece of dark jeans fabric Royalty Free Stock Photography"}]},


    #假定单图假设单图像输入的粗略估算，计算每一条样本的token数量
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0   #设置成一个固定的128个tokens数量
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)   #统计value部分的粗略长度
        return length_list




# 设计modality_lengths输出纯文本模态对应负数，视觉模态对应正数，是为了区分不同类型的模态样本，方便训练时的一些采样策略。

# 原理和作用
# 多模态任务中，训练数据包含纯文本和视觉（图像+文本）两种样本。

# 训练时，有时需要对这两类样本分别处理或均衡采样，避免训练批次被纯文本或视觉样本单一占据，保证多模态模型学习均衡。

# 使用正数/负数区分模态是一种简单且高效的标记方式：

# 正数表示该样本含视觉模态（图像），代表多模态输入；

# 负数表示该样本纯文本模态，无视觉输入。

# 采样器或训练控制逻辑根据这个符号区分不同模态数据，能实现模态感知采样、分组或权重调整


    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    #这个函数非常重要.....后面需要依据音频输入来进行修改
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]  # pt 是pytorch张量的简写？？直接返回pixel_values代表，一批涨了，然后取第一个
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args) # 完全复制一份
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # sources=[{"from": "human", "value": "<image>\nWhat is this?"},
        #  {"from": "gpt", "value": "Piece of dark jeans fabric Royalty Free Stock Photography"}]   
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict




# 这里的 image 是经过图像处理器（processor）预处理后的 PyTorch 张量（tensor），代表图片以数值形式被加载并标准化，方便模型直接输入计算。

# 而 preprocess 函数主要是对对话中的文本内容做分词（tokenize）和编码，转换成模型能够理解的input_ids、labels等张量格式。

# preprocess 根据不同模板版本和规则，对对话文本进行一定的拼接整理，比如添加起始标识、分隔符，生成模型能接受的标准格式。

# 这个过程确保图像张量和文本编码都准备好，统一封装在 data_dict 字典里返回，作为模型训练或推理的输入。

# 简言之：
# 图像经过图像处理器变成张量；文本经过 preprocess 编码成ids；两者合并形成完整的多模态输入，针对模型要求做精细处理。

# 这就是你示例中 __getitem__ 函数的核心工作流程，保证每条样本都是结构合理标准的多模态输入，供模型高效学习和推理使用。是的，你理解得很准确。

# 这里的 image 是经过图像处理器（processor）预处理后，转成的 PyTorch 张量（tensor），方便后续模型直接输入。

# 而 preprocess 函数主要是对对话里面的文本内容进行分词（tokenize）、编码成模型可用的 input_ids、labels 等张量格式。这个过程中会根据不同的对话模板版本以及规则对文本做拼接，添加分隔符、起始标识等，生成符合模型输入要求的标准格式。

# 所以整体流程是：

# 先将图片用处理器转换成张量格式（即数值化的多维数组）；

# 再将对话文本编码tokenize成模型能理解的token序列；

# 最后将图像张量和文本编码一起放入字典，作为多模态模型训练或推理的输入。

# 这就是你代码中 __getitem__ 方法的核心作用，确保每条样本数据最终能形成标准化的多模态输入格式，满足训练和推理需求。



# @dataclasses.dataclass 装饰的 Conversation 类确实非常重要，理由如下：

# 结构清晰，方便管理多轮对话数据
# Conversation 类结构统一地保存了多轮对话的所有相关内容：系统信息、角色、消息、分隔符样式、版本号等。作为数据模型，它规范了对话消息存储，减少出错和混乱。

# 支持多样化的对话格式
# 通过 sep_style 和 version 等属性，可以灵活适配不同模型对多模态对话格式的需求，比如不同的消息分隔符、起始标志等。这是多模态对话模型兼容多种输入格式的基础。

# 封装对话处理逻辑
# 诸如 get_prompt() 用来生成拼接的对话字符串，append_message() 添加消息，get_images() 抽取图片，to_gradio_chatbot() 转成UI友好格式等，都是将多个功能封装在类里，方便调用和复用。

# 提升可维护性和扩展性
# 使用 dataclass 省去了繁琐的初始化代码，且提供拷贝、转字典等方法，提高代码可读性和维护效率，方便后期针对新需求做类的版本迭代。

# 多模态数据统一处理利器
# 该类不仅处理文本对话，也包含对消息中图片的拆分、图片预处理逻辑，符合多模态大模型的实际需要。

# 总结起来，Conversation 类是多模态对话系统中核心的数据结构和操作封装，决定了训练与推理阶段对话数据的组织和使用方式。深入研究它可以帮助理解整个多模态对话流程的根基和关键设计，准确合理地构造输入数据，保证训练与推理的顺利进行和模型性能。




# 完全复制一份 [
#   [
#     {"from": "human", "value": "<image>\nWhat is this?"},
#     {"from": "gpt", "value": "Piece of dark jeans fabric Royalty Free Stock Photography"}
#   ]
# ]  preprocess_multimodal 对这个进行处理

# 这里逐步解释：

# return_tensors='pt' 是告诉预处理器（processor）将处理后的数据转换成 PyTorch 张量（tensor）。

# ‘pt’ 是 PyTorch 的缩写。

# 这样返回的结果中，数据格式就是 PyTorch 的 Tensor，方便后续模型调用。

# processor.preprocess(image, return_tensors='pt') 返回的是一个字典（dict），其中包含了模型所需的预处理字段，比如pixel_values。

# ['pixel_values'] 是从返回的字典中取出关键字段 pixel_values，它通常是图像对应的张量（tensor）。这里pixel_values代表转换后、归一化处理后的图像数据。

# `` 是因为返回的 pixel_values 可能是一个批量(batch)的张量（即第一维是batch维度），即使这里只有一个图片样本，还是会被包装成长度为1的batch。

# `` 表示从这个批量里取出第一个（也是唯一一个）样本的张量。

# 这样最终image变量存的是形状为 (channel, height, width) 的单张图片张量，而不是包含batch维度的四维张量。

# 简单说，这一行代码完成了：

# 图像预处理，转换为模型需要的张量形式；

# 选取这一批次中的第一个图像张量，赋值给变量 image 用于后续模型输入。

# 这样可以保证传给模型的图像输入格式是单张图片的张量，而不是批量的张量，从而方便单样本推理或训练时的数据处理。

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        labels = labels[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

#在这个地方考虑如何处理样本和批次
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)






# 如果要加入语音模态样本，确实需要重新设计modality_lengths函数，因为它当前只考虑了纯文本和视觉图像模态。

# 为什么要重新设计？
# 多模态扩展
# 原来的modality_lengths只区分两类模态：

# 视觉模态（image存在）用正数表示

# 纯文本样本用负数表示

# 语音作为新的模态，需要用新的方式来区分，并且可能用不同的长度估计方式。

# 长度的量化不同

# 语音模态通常基于声学帧数或音频片段数来估计长度，和文本词数、视觉token数长度不在同一量级或标准。

# 需要设计统一或者归一化的长度度量，方便不同模态间做比较和分组采样。

# 多模态均衡采样要求

# 作为训练采样策略的基础，模态标签不仅要区分，还需要保证训练中不同模态样本的分布均衡。

# 如何重新设计？
# 1. 给不同模态类别赋予不同正负或符号区分
# 例如视觉+1前缀，文本-1前缀，语音可以用+2或-2，甚至多维度标记模态。

# 2. 长度计算改为针对每种模态单独计算并归一化
# 文本：用tokenizer编码长度

# 视觉：用预定义的视觉token数量或动态估计

# 语音：用音频帧数或切分片段数量

# 3. 更复杂的返回结构（可选）
# 返回元组或字典表示每个样本的多模态长度和模态标签

# 便于采样器调用时按模态和长度联合过滤或分组

# 示例改进伪代码
# python
# @property
# def modality_lengths(self):
#     length_list = []
#     for sample in self.list_data_dict:
#         text_len = sum(len(tokenizer.encode(conv['value'])) for conv in sample['conversations'])
#         if 'image' in sample:
#             length = +text_len  # 视觉模态保持正数
#         elif 'audio' in sample:
#             audio_len = estimate_audio_length(sample['audio'])  # 估计音频长度，单位帧数或秒
#             length = +audio_len * scale_factor  # 归一化，乘以系数匹配长度量级
#         else:
#             length = -text_len  # 纯文本负数区分
#         length_list.append(length)
#     return length_list
# 总结
# 重新设计modality_lengths是为了支持多模态标签区分和长度度量的多样性，满足多模态训练采样策略需求。

# 不同模态的长度估计原理不同，需要采用模态特定的方法计算并归一。

# 采用符号或标签区分模态方便采样时针对不同模态进行分组和均衡。

# 这样设计确保训练采样器能理解和合理利用多模态数据的异构特性，提高训练效果和效率。

# 这是多模态学习和多模态训练采样设计中的常见做法。

