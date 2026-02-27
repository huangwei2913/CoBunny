from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .phi import PhiModel, PhiConfig, PhiForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM
from transformers.generation import GenerationMixin # 必须导入这个
import torch.distributed as dist
import sys

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

        # if dist.is_initialized() and dist.get_rank() == 0:
        #     sys.stdout.write("\n" + "🔔" * 10 + " [微调数据流监控] " + "🔔" * 10 + "\n")
            
        #     # 1. 检查文字原材料
        #     if input_ids is not None:
        #         sys.stdout.write(f"📝 input_ids 形状: {input_ids.shape}\n")
        #         # 检查你的 IMAGE_TOKEN_INDEX (-200)
        #         img_token_id = -200 
        #         num_img_tokens = (input_ids == img_token_id).sum().item()
        #         sys.stdout.write(f"🎯 占位符数量 (-200): {num_img_tokens}\n")
                
        #         # 看看第一个样本的前 50 个 Token (通常包含系统提示词和图片占位符)
        #         sys.stdout.write(f"🔍 序列片段: {input_ids[0, :].tolist()}\n")
            
        #     # 2. 检查图片原材料
        #     if images is not None:
        #         sys.stdout.write(f"🖼️  images 形状: {images.shape} (预期 [B, 3, H, W])\n")
        #     else:
        #         sys.stdout.write("⚠️  警告：images 是空的 (None)！视觉特征丢失！\n")
            
        #     # 3. 检查标签 (微调的关键)
        #     if labels is not None:
        #         # 统计非 -100 的数量，即真正参与计算 Loss 的文字数量
        #         valid_labels = (labels != -100).sum().item()
        #         sys.stdout.write(f"🏷️  有效 Label 数量: {valid_labels}\n")
                
        #     sys.stdout.write("-" * 50 + "\n")
        #     sys.stdout.flush()
        # # 1. 预处理
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

        #print(f"DEBUG: Final inputs_embeds shape...............................: {inputs_embeds.shape}")
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
            #if torch.distributed.get_rank() == 0: # 只让 0 号卡打印，避免刷屏
                #print(f"🔥🔥🔥 [REAL-TIME CHECK] Step Loss: {loss.item():.4f}")
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
        if past_key_values is None:
            print("\n" + "="*50)
            print("🚀 [调度中心] 进入 Prefill (初始读图) 阶段")

            # 1. 检查输入 ID (Tokenizer 的产物)
            print(f"📝 原始 input_ids: {input_ids[0].tolist()}")

            # 2. 检查特殊的图像占位符 (-200) 是否存在
            image_token_index = -200 # 对应你代码里的 IMAGE_TOKEN_INDEX
            has_image_token = (input_ids == image_token_index).any()
            print(f"🎯 是否检测到图片占位符 (-200): {has_image_token}")

            # 3. 统计占位符数量（如果是多图会大于 1）
            num_images = (input_ids == image_token_index).sum().item()
            print(f"📊 样本中占位符数量: {num_images}")

            # 4. 打印对应的文字片段（反向分词，看 Tokenizer 有没有乱分）
            # 注意：这里需要你环境中能访问到 tokenizer，如果访问不到可以注释掉
            if hasattr(self, 'tokenizer'):
                decoded_text = self.tokenizer.decode(input_ids[0])
                print(f"📖 还原后的提示词: {decoded_text}")
            print(f"🖼️  传入的图像张量形状: {images.shape if images is not None else 'None'}")
            print("="*50 + "\n")
        else:
            pass
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
