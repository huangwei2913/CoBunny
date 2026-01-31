import torch
from safetensors import safe_open

# 替换为你 Stage 2 的权重路径
path = "./checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-23476/model.safetensors"

# 1. 只读取元数据（不加载权重到内存，非常快）
with safe_open(path, framework="pt", device="cpu") as f:
    all_keys = f.keys()

# 2. 统计视觉塔相关的 Key
vision_keys = [k for k in all_keys if "vision_tower" in k]
projector_keys = [k for k in all_keys if "mm_projector" in k]
llm_keys = [k for k in all_keys if "model.layers" in k]

print(f"📊 总参数 Key 数量: {len(all_keys)}")
print(f"👁️ 视觉塔相关的 Key 数量: {len(vision_keys)}")
print(f"🔌 Projector 相关的 Key 数量: {len(projector_keys)}")
print(f"📖 LLM 层相关的 Key 数量: {len(llm_keys)}")

# 3. 打印前 10 个视觉 Key 看看具体路径
if vision_keys:
    print("\n🔍 视觉 Key 样例如下:")
    for k in vision_keys[:10]:
        print(f"  - {k}")
else:
    print("\n❌ 警告：未在 safetensors 中发现任何 vision_tower 相关的 Key！")