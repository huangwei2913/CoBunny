from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .phi import PhiModel, PhiConfig, PhiForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM
from transformers.generation import GenerationMixin # 必须导入这个

class BunnyPhiConfig(PhiConfig):
    model_type = "bunny-phi"


class BunnyPhiModel(BunnyMetaModel, PhiModel):
    config_class = BunnyPhiConfig

    def __init__(self, config: PhiConfig):
        super(BunnyPhiModel, self).__init__(config)


class BunnyPhiForCausalLM(PhiForCausalLM, BunnyMetaForCausalLM, GenerationMixin):
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

    # 5. 【手动桥接】确保 get_vision_tower 能直接被找到
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()


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

        # 1. 预处理
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

        # ... (你关于 KV Cache 的处理逻辑保持不变) ...
            # --- 🚀 终极手动修复逻辑 ---
        # 如果存在 KV Cache (past_key_values)，我们需要确保 Mask 的长度是 [当前输入 + 历史缓存]
        if past_key_values is not None and inputs_embeds is not None:
            # 获取已经缓存的 Token 数量
            cache_length = past_key_values.get_seq_length()
            # 获取当前输入的 Token 数量
            current_length = inputs_embeds.shape[1]
            # 总长度
            total_length = cache_length + current_length
            
            # 强制构造一个全 1 的长 Mask，覆盖整个序列
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], total_length),
                dtype=torch.long,
                device=inputs_embeds.device
            )
            
            # 同时也强制对齐 position_ids，从缓存位置开始往后排
            position_ids = torch.arange(
                cache_length, total_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0).repeat(inputs_embeds.shape[0], 1)

        print(f"DEBUG: Final inputs_embeds shape...............................: {inputs_embeds.shape}")
        # 2. 调用内部模型
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # --- 🚀 核心修复：计算 Loss ---
        loss = None
        if labels is not None:
            
            #print(f"DEBUG: Initial labels snippet: {labels[0, :50].tolist()}")    
            #print(f"DEBUG: Sequence END labels.............: {labels[0, -10:].tolist()}")

            # 将 logits 移位以匹配 labels (Causal LM 标准操作)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            num_valid_labels = (shift_labels != -100).sum().item()
            #print(f"DEBUG: Valid tokens in this batch: {num_valid_labels}")
            #print(f"DEBUG: First 10 shift_labels: {shift_labels.view(-1)[:10]}")           
            # 展平进行计算
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # 确保在同一设备上
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if torch.distributed.get_rank() == 0: # 只让 0 号卡打印，避免刷屏
                print(f"🔥🔥🔥 [REAL-TIME CHECK] Step Loss: {loss.item():.4f}")
        # 3. 返回时带上计算好的 loss
        return CausalLMOutputWithPast(
            loss=loss,  # <--- 现在不再是 None 了！
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        images = kwargs.pop("images", None)

        # --- 🔥 终极修复：完美模拟旧版 Cache 接口 ---
        if past_key_values is not None:
            # 1. 模拟 seen_tokens
            if not hasattr(past_key_values, "seen_tokens"):
                past_key_values.seen_tokens = past_key_values.get_seq_length()
            
            # 2. 模拟 get_max_length
            if not hasattr(past_key_values, "get_max_length"):
                past_key_values.get_max_length = lambda: None 

            # 3. 模拟 get_usable_length (修正导致 TypeError 的地方)
            if not hasattr(past_key_values, "get_usable_length"):
                def get_usable_length(seq_len, layer_idx=None):
                    # 如果 layer_idx 是 None，传 0 或者不传，取决于你想获取哪个层的长度
                    # DynamicCache 需要明确的 layer_idx 才能工作
                    real_idx = layer_idx if layer_idx is not None else 0
                    return past_key_values.get_seq_length(real_idx)
                past_key_values.get_usable_length = get_usable_length

        # 调用父类 (Phi) 的原始方法
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

        # --- 🔥 修复：Attention Mask 传递 ---
        if _inputs.get("attention_mask") is None:
            if attention_mask is not None:
                _inputs["attention_mask"] = attention_mask
            else:
                _inputs["attention_mask"] = torch.ones_like(_inputs["input_ids"])

        if images is not None:
            _inputs['images'] = images
            
        return _inputs

AutoConfig.register("bunny-phi", BunnyPhiConfig)
AutoModelForCausalLM.register(BunnyPhiConfig, BunnyPhiForCausalLM)
