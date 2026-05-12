import transformers
from bunny.model import *
import torch

# 1. 加载您合并后的完整模型
path = "/mnt/CoBunny/checkpoints-stage3/llama_ocr"
tokenizer = transformers.AutoTokenizer.from_pretrained(path)
model = BunnyLlamaForCausalLM.from_pretrained(path)

# 2. 检查停止符到底是谁
print(f"🔍 官方定义的 EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"🔍 训练中定义的 PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# 3. 核心校验：检查模型的 lm_head 最后一层
# 如果 128001 是 EOS，检查它的权重是否与您训练时拷贝的 <pad> 一致
with torch.no_grad():
    embeds = model.get_input_embeddings().weight
    # 对比 EOS 和 <pad> 的权重 MD5
    eos_vec = embeds[tokenizer.eos_token_id]
    pad_vec = embeds[tokenizer.pad_token_id]
    is_same = torch.allclose(eos_vec, pad_vec)
    print(f"📊 权重一致性检查：EOS 与 PAD 权重是否相同? {'✅ 相同' if is_same else '❌ 不同'}")