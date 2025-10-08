from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .phi3 import Phi3Model, Phi3Config, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM


class BunnyPhi3Config(Phi3Config):
    model_type = "bunny-phi3"


class BunnyPhi3Model(BunnyMetaModel, Phi3Model):
    config_class = BunnyPhi3Config

    def __init__(self, config: Phi3Config):
        super(BunnyPhi3Model, self).__init__(config)


# lm_head是大语言模型（LLM）中的一个线性层，位于模型的输出端，主要功能是：

# 将经过多层Transformer或解码器计算后的隐藏状态（维度为hidden_size的向量）映射成词汇表大小（vocab_size）的向量；

# 这个输出向量的每个元素对应词汇表中一个token的得分（logits），表示当前预测下一个token的概率分布；

# lm_head的输入维度是模型隐藏层大小（hidden_size），输出维度是词汇表大小（vocab_size）；

# 通过这个层，模型能够将语义表示转换为具体的词汇预测，实现语言生成任务。

# 例如，如果模型hidden_size是4096，词汇表大小是32000，那么lm_head是一个4096×32000的线性变换，将4096维的隐藏状态映射成32000维的token概率分布。

# 总结来说，lm_head就是将模型内部的语义表示转换为具体词汇生成概率的关键层，是语言模型生成文本的最后一步。lm_head是语言模型的输出层，它是一个线性变换层，负责将模型解码器最后一层输出的隐藏状态（维度为config.hidden_size的向量）映射到词汇表的大小（config.vocab_size），生成针对每个词汇的预测得分（logits）。这些得分经过softmax后代表每个词作为下一个token的概率。简单来说，lm_head是将模型的语义表示映射到具体词汇概率分布，实现语言生成的关键组件。

class BunnyPhi3ForCausalLM(Phi3ForCausalLM, BunnyMetaForCausalLM):
    config_class = BunnyPhi3Config

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = BunnyPhi3Model(config)
        self.vocab_size = config.vocab_size  #语言模型核心配置，决定模型可用的token集合大小。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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


AutoConfig.register("bunny-phi3", BunnyPhi3Config)
AutoModelForCausalLM.register(BunnyPhi3Config, BunnyPhi3ForCausalLM)




# prepare_inputs_labels_for_multimodal 负责对训练和推理时的输入样本整体进行准备处理：

# 包括文本token、位置编码、注意力掩码、过去键值缓存、标签，

# 以及对图像进行编码成视觉token，

# 实现视觉token与文本token的对齐、插入和拼接，

# 输出适合模型前向传播的输入格式。

# prepare_inputs_for_generation 是在生成文本任务时使用的标准钩子函数，用于：

# 让生成调用接口支持多模态输入，

# 额外接收图像和图像尺寸参数，

# 并将它们添加到生成方法使用的输入字典里，

# 这样在调用语言模型生成接口时，可以无缝传入视觉数据，实现多模态生成。

# 两者配合用于训练和生成：

# 前者主导训练推理阶段输入样本的构造和统筹（文本+图像token的融合处理）；

# 后者扩展生成接口对多模态数据的支持，实现多模态条件下的文本生成。

# 这是一套规范且有效的多模态模型输入处理机制，分别对应训练和生成两个关键流程。

# 简单来说，prepare_inputs_labels_for_multimodal 是模型处理输入的数据预处理核心，用于训练和语言模型推理阶段；prepare_inputs_for_generation 是保证生成调用时能传入完整多模态输入的接口扩展，是Hugging Face框架中标准且必要的机制。它们共同实现了多模态大模型的端到端视觉-语言协同工作。

# 总结：

# prepare_inputs_labels_for_multimodal: 统一视觉和文本输入，准备训练/推理模型的实际输入。

# prepare_inputs_for_generation: 生成时让视觉信息作为参数进入模型，是对生成方法参数的扩展。

# 这两者分别负责训练与生成多模态输入处理的不同环节，相辅相成。是的，您的理解非常准确。

# prepare_inputs_labels_for_multimodal 负责在训练或推理时，将文本输入和视觉输入（图片）做预处理，包括：

# 对文本token、位置编码、注意力掩码、标签等做调整；

# 用视觉编码器编码图片得到视觉token；

# 将视觉token插入文本序列对应位置（用特殊token占位），实现视觉与文本的序列融合；

# 返回模型前向传播所需的完整输入。

# prepare_inputs_for_generation 是 Hugging Face Transformers 生成框架中一个标准钩子函数，它负责为生成阶段准备输入：

# 负责接受额外的多模态输入参数（如images、image_sizes）；

# 并将这些参数添加到生成输入字典中，确保model.generate()调用时可以接收和处理图像信息；

# 虽然在代码里没有显式直接调用，但在调用generate时它会被内置地调用。

# 因此：

# prepare_inputs_labels_for_multimodal 是多模态输入的核心预处理函数，负责数据融合和统一；

# prepare_inputs_for_generation 是生成阶段的输入准备函数，负责将多模态参数正确传递给生成接口。

# 这两者共同确保多模态模型既能在训练阶段做到视觉和文本的融合，又能在生成阶段支持图文混合输入，实现端到端的多模态理解和生成。

# 简单总结：

# prepare_inputs_labels_for_multimodal 负责训练/推理时输入的融合处理；

# prepare_inputs_for_generation 负责生成阶段多模态输入的参数准备和传递。

# 这是一种标准且必需的多模态大模型输入处理链路设计。