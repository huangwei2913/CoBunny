from transformers import AutoModelForCausalLM, AutoConfig
import torch

model_path = "/mnt/conda_data/microsoft/phi-1_5"

print("--- 开始探测 Phi-1.5 内部结构 ---")

# 1. 尝试加载配置和模型
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# 我们只加载 Meta 数据或轻量化加载，不需要占满显存
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cpu", 
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 2. 打印模型完整结构（这个最直观）
print("\n[模型层级结构]:")
print(model)

# 3. 重点：寻找 Embedding 层
print("\n[Embedding 层探测结果]:")

# 探测方案 A: Llama 风格
if hasattr(model, 'get_model') and hasattr(model.get_model(), 'embed_tokens'):
    print("✅ 方案 A: 存在 model.get_model().embed_tokens")
elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
    print("✅ 方案 B: 存在 model.model.embed_tokens")
# 探测方案 C: Phi-1.5 常见的 transformer.emb 风格
elif hasattr(model, 'transformer') and hasattr(model.transformer, 'emb'):
    print("✅ 方案 C: 存在 model.transformer.emb (Phi 原生风格)")
# 探测方案 D: 通用的 get_input_embeddings 方法 (Hugging Face 标准)
elif hasattr(model, 'get_input_embeddings'):
    embed_layer = model.get_input_embeddings()
    print(f"✅ 方案 D: 找到通用方法 get_input_embeddings(), 实际类为: {type(embed_layer)}")
else:
    print("❌ 警告：未找到标准命名的 Embedding 层！")

# 4. 模拟一次 Embedding 转换测试
try:
    test_id = torch.tensor([198]) # 你的换行符
    if hasattr(model, 'get_input_embeddings'):
        test_emb = model.get_input_embeddings()(test_id)
        print(f"\n[功能测试]: 成功将 ID 198 转换为向量，形状为: {test_emb.shape}")
except Exception as e:
    print(f"\n[功能测试]: 转换失败，错误信息: {e}")