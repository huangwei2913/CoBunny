from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation import GenerationMixin # 必须导入这个

from .qwen2 import Qwen2Model, Qwen2Config, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM


class BunnyQwen2Config(Qwen2Config):
    model_type = "bunny-qwen2"


class BunnyQwen2Model(BunnyMetaModel, Qwen2Model):
    config_class = BunnyQwen2Config

    def __init__(self, config: Qwen2Config):
        super(BunnyQwen2Model, self).__init__(config)


class BunnyQwen2ForCausalLM(Qwen2ForCausalLM, BunnyMetaForCausalLM, GenerationMixin):
    config_class = BunnyQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = BunnyQwen2Model(config)
        self.vocab_size = config.vocab_size
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
        # 1. 提取图像张量并确保其能透传给 forward
        images = kwargs.pop("images", None)

        # 2. 推理状态监控 (Prefill 阶段自检)
        if past_key_values is None and images is not None:
            print("\n" + "="*50)
            print("🚀 [Qwen2 调度中心] 进入 Prefill (初始读图) 阶段")
            
            # 检查输入 ID 是否包含占位符
            # 这里的 -200 对应你代码中的 IMAGE_TOKEN_INDEX
            image_token_index = getattr(self.config, "image_token_index", -200) 
            num_images = (input_ids == image_token_index).sum().item()
            
            print(f"🎯 检测到图片占位符: {num_images > 0} (数量: {num_images})")
            print(f"🖼️  传入图像张量形状: {images.shape}")
            
            # 打印序列片段辅助调试
            if input_ids.shape[1] > 50:
                print(f"🔍 序列头部片段: {input_ids[0, :50].tolist()}...")
            else:
                print(f"🔍 完整序列: {input_ids[0].tolist()}")
            print("="*50 + "\n")

        # 3. KV Cache 兼容性桥接
        # 虽然 Qwen2 较新，但手动确保这些属性存在可以防止某些 transformers 版本的逻辑回退
        if past_key_values is not None:
            if not hasattr(past_key_values, "seen_tokens"):
                past_key_values.seen_tokens = past_key_values.get_seq_length()
            
            if not hasattr(past_key_values, "get_usable_length"):
                def get_usable_length(seq_len, layer_idx=None):
                    real_idx = layer_idx if layer_idx is not None else 0
                    return past_key_values.get_seq_length(real_idx)
                past_key_values.get_usable_length = get_usable_length

        # 4. 调用 Qwen2 原生的准备逻辑
        _inputs = super().prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            **kwargs
        )

        # 5. 🔥 核心修复：Attention Mask 强制对齐
        # 确保在多模态特征插入后，Mask 的长度始终覆盖 [文本 + 图像]
        if _inputs.get("attention_mask") is None:
            if attention_mask is not None:
                _inputs["attention_mask"] = attention_mask
            else:
                _inputs["attention_mask"] = torch.ones_like(_inputs["input_ids"])

        # 6. 将图像重新装回输入字典，确保传给 forward
        if images is not None:
            _inputs['images'] = images
            
        return _inputs

AutoConfig.register("bunny-qwen2", BunnyQwen2Config)
AutoModelForCausalLM.register(BunnyQwen2Config, BunnyQwen2ForCausalLM)
