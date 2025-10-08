from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .phi import PhiModel, PhiConfig, PhiForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM


class BunnyPhiConfig(PhiConfig):
    model_type = "bunny-phi"


# 这个 PhiModel 类是 PhiPreTrainedModel 的具体实现，它实现了一个基于 Transformer 的解码器结构，包含以下主要功能：

# 模型结构定义

# 包含 token 嵌入层 embed_tokens，将输入的 token id 转换为向量表示；

# 使用 dropout 进行正则化；

# 定义 config.num_hidden_layers 个 Transformer 解码层，每个层由 PhiDecoderLayer 实现；

# 在最后有 layer normalization 层进行归一化。

# 初始化

# 构造函数中初始化上述层和一些配置，调用父类初始化权重方法 post_init()；

# 设置 padding token id 和词表大小；

# 用于支持 flash attention 等加速机制。

# 输入嵌入接口

# 提供获取和设置 token 嵌入层的函数 get_input_embeddings 和 set_input_embeddings。

# 前向计算（forward）

# 接收输入 token 序列（input_ids）或预嵌入向量（inputs_embeds）；

# 支持传入注意力掩码、位置编码、缓存（past_key_values）等；

# 支持梯度检查点以减少显存开销；

# 根据是否开启缓存，管理缓存的格式转换和位置编码生成；

# 逐层调用定义的解码层处理隐藏状态；

# 最后经过 layernorm 归一化；

# 返回最后一层的隐藏状态、缓存和各层隐藏状态、注意力权重。

# 灵活的返回结构

# 支持返回字典或元组；

# 通过参数控制是否输出中间隐藏层和注意力权重。

# 简单总结
# PhiModel 实现了Transformer解码器核心结构功能：

# 输入token或嵌入，逐层计算Transformer解码层输出；

# 支持梯度显存优化、缓存和多种可配置的辅助特性；

# 输出隐藏层表示，支撑文本生成、推理和微调任务。

# 举例
# 当你调用

# python
# outputs = model(input_ids=torch.tensor([[1,2,3,4]]))
# 模型会先将token映射为向量，逐层经过self-attention等机制处理，最后给出文本表示的隐藏状态供后续生成或分类计算。



class BunnyPhiModel(BunnyMetaModel, PhiModel):
    config_class = BunnyPhiConfig

    def __init__(self, config: PhiConfig):
        super(BunnyPhiModel, self).__init__(config)


class BunnyPhiForCausalLM(PhiForCausalLM, BunnyMetaForCausalLM):
    config_class = BunnyPhiConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = BunnyPhiModel(config)  #内部封装了 PhiModel（基于 Transformer 的解码器核心）做前向计算；
        self.vocab_size = config.vocab_size  # 词汇表大小
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) #头部有一个线性层 lm_head，把隐藏状态映射到词表大小，用于预测下一个token的概率。
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("bunny-phi", BunnyPhiConfig)
AutoModelForCausalLM.register(BunnyPhiConfig, BunnyPhiForCausalLM)


# BunnyPhiForCausalLM 类是继承自 PhiForCausalLM 和 BunnyMetaForCausalLM 的多模态因果语言模型实现，主要实现了以下功能：

# 主要功能解释
# 多模态融合模型

# 内部封装了 BunnyPhiModel，这是基于 Transformer 的语言模型解码器核心，并集成了视觉编码器投影模块（继承自 BunnyMetaForCausalLM 功能）。

# 用于同时处理文本输入和视觉输入（图像特征）。

# 线性预测头

# 顶层用线性层 lm_head，将隐藏层输出映射到词表大小，做下一个token概率预测。

# 初始权重由self.post_init()初始化。

# 多模态输入准备

# 覆写了 forward 方法，增加了 images 参数，支持视觉图像作为输入。

# 如果没有直接传入 inputs_embeds，调用从 BunnyMetaForCausalLM 继承的 prepare_inputs_labels_for_multimodal 方法，将图像编码成视觉特征，与文本嵌入融合，输出统一嵌入和标签。

# 调用父类前向

# 处理完输入embedding和视觉特征后，调用父类 PhiForCausalLM 的 forward 方法完成语言模型的核心计算。

# 支持生成阶段视觉输入

# 覆写 prepare_inputs_for_generation 方法，支持生成时将视觉输入 images 一并传入，方便多模态生成推理。

# 总结
# BunnyPhiForCausalLM 是一个结合了视觉编码和基于Phi Transformer解码器的多模态大语言模型实现焦点。它完成了：

# 文本与视觉特征的统一嵌入；

# 多模态因果语言模型训练与推理；

# 支持训练时标签（labels）和注意力掩码的正确处理；

# 支持推理阶段多模态输入的连贯传递。

# 这是Bunny多模态大模型体系中的重要模块，实现了视觉-语言融合，构建了端到端的多模态生成模型。

# 换句话说，BunnyPhiForCausalLM 在 PhiForCausalLM 的基础上：

# 注入了视觉编码器处理能力（利用BunnyMetaForCausalLM的特性）；

# 同时保持了标准的语言模型接口与生成能力；

# 实现了多模态输入准备、前向推理和生成输入管理的完整流程。

# 这使模型既能处理文本也能理解图像，从而完成多模态语言生成和理解任务。这个类是 Bunny 多模态大语言模型的核心封装，实现了视觉特征编码与语言模型的无缝对接。它继承了 PhiForCausalLM 的文本因果语言建模能力，同时结合了 BunnyMetaForCausalLM 中的视觉编码与多模态输入融合方法，具体功能包括：
