import os
import sys
import gc
import torch
import warnings
from PIL import Image
from typing import Optional, List, Union, Tuple  # 🔥 必须加上这一行

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoConfig
from transformers.generation import GenerationMixin
from transformers.cache_utils import DynamicCache

sys.path.append(os.getcwd())

from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, IMAGE_TOKEN_INDEX

from transformers.modeling_outputs import CausalLMOutputWithPast
import torch

def patch_bunny_phi_hardcore(model):
    MAX_POS = getattr(model.config, "max_position_embeddings", 2048)
    VOCAB_SIZE = getattr(model.config, "vocab_size", 51200)

    def fixed_forward(
        input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, images=None, return_dict=None, **kwargs
    ):
        # 1. 索引安全防护
        if input_ids is not None:
            input_ids = torch.where((input_ids < 0) | (input_ids >= VOCAB_SIZE), 
                                  torch.zeros_like(input_ids), input_ids)

        # 2. 图像融合
        if images is not None and input_ids is not None:
            (
                _, position_ids, attention_mask, _, inputs_embeds, labels
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )
            input_ids = None 

        # 3. 动态对齐 Mask
        past_length = 0
        if past_key_values is not None:
            try: past_length = past_key_values[0][0].shape[2]
            except: past_length = 0

        cur_seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else (input_ids.shape[1] if input_ids is not None else 0)
        
        if cur_seq_len > 0:
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            attention_mask = torch.ones((1, past_length + cur_seq_len), dtype=torch.bool, device=device)
            position_ids = torch.arange(past_length, past_length + cur_seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # 4. 核心推理
        outputs = model.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds.to(torch.float16) if inputs_embeds is not None else None,
            use_cache=use_cache,
            return_dict=True,
        )

        # 5. 输出映射与类型强制转换 (防止 OverflowError)
        logits = model.lm_head(outputs[0]).float() # 必须用 float()

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = fixed_forward
    
    def fixed_prepare_inputs(input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            return {"input_ids": input_ids[:, -1:], "past_key_values": past_key_values, "use_cache": kwargs.get("use_cache")}
        return {"input_ids": input_ids, "images": kwargs.get("images"), "attention_mask": attention_mask}

    model.prepare_inputs_for_generation = fixed_prepare_inputs



def patch_dynamic_cache():
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, seq_length=None, layer_idx=None: self.get_seq_length(layer_idx if layer_idx is not None else 0)

# --- 2. 推理主函数 ---

def run_t4_final_inference():
    checkpoint_dir = '/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k'
    base_llm_path = '/mnt/conda_data/microsoft/phi-1_5'
    device = 'cuda'
    dtype = torch.float16

    print("🚀 [1/5] 加载模型...")
    disable_torch_init()
    patch_dynamic_cache()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=checkpoint_dir,
        model_base=base_llm_path,
        model_name='bunny-phi-1.5-lora',
        model_type='phi-1.5',
        device_map=device,
        torch_dtype=dtype
    )

    # 应用最强力补丁
    patch_bunny_phi_hardcore(model)

    print("🛡️ [2/5] 视觉塔安全截断...")
    vision_tower = model.get_vision_tower()
    with torch.no_grad():
        for param in vision_tower.parameters():
            if param.abs().max() > 65500 or param.dtype == torch.bfloat16:
                param.data = param.data.clamp(min=-65500, max=65500).to(dtype)
    vision_tower.to(device=device, dtype=dtype)

    print("💉 [3/5] 注入 Non-LoRA 权重...")
    non_lora_bin = os.path.join(checkpoint_dir, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_bin):
        non_lora_weights = torch.load(non_lora_bin, map_location='cpu')
        target_dict = model.state_dict()
        cleaned_weights = {}
        for k, v in non_lora_weights.items():
            tk = k
            while tk not in target_dict and '.' in tk: tk = tk.split('.', 1)[1]
            if tk in target_dict:
                cleaned_weights[tk] = v.clamp(min=-65500, max=65500).to(dtype)
        model.load_state_dict(cleaned_weights, strict=False)
        print(f"✅ 已匹配 {len(cleaned_weights)} 个视觉融合参数")

    # 4. 设置生成配置
    if not hasattr(model, 'generate'):
        class FixedModel(model.__class__, GenerationMixin): pass
        model.__class__ = FixedModel
    model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.eval()

    print("🖼️ [4/5] 准备图像输入...")
    image = Image.open("Test.jpg").convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)
    
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    print("🎯 [5/5] 开始生成推理...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=512,
            use_cache=True,
        )

    out_list = output_ids[0].cpu().numpy().tolist()
    vocab_size = tokenizer.vocab_size
    safe_out_ids = [int(tid) for tid in out_list if 0 <= tid < vocab_size]
    try:
        response = tokenizer.decode(safe_out_ids, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"❌ 解码失败: {e}")
        response = "解码过程中出现异常，可能是生成的序列包含损坏的 Token。"

    print("-" * 30)
    print(f"🤖 Bunny 响应: {response}")
    print("-" * 30)

if __name__ == "__main__":
    run_t4_final_inference()