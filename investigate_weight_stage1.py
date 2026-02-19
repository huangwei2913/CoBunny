import torch
from safetensors.torch import load_file
import os
from collections import defaultdict

# 指向你保存的全量权重文件
checkpoint_path = "/mnt/conda_data/checkpoints-pretrain/pretrain_stage1_continued/model.safetensors"

if not os.path.exists(checkpoint_path):
    print(f"❌ 找不到权重文件: {checkpoint_path}")
    exit()

print(f"📖 正在深度解析权重文件: {checkpoint_path}")
print("=" * 80)

# 加载权重
weights = load_file(checkpoint_path)
all_keys = sorted(weights.keys())

# 用于分类存储
categories = {
    "Vision Tower (视觉编码器)": [],
    "Projector (模态投影层)": [],
    "LLM Backbone (语言模型骨架)": [],
    "Other (其他权重)": []
}

for key in all_keys:
    if "vision_tower" in key:
        categories["Vision Tower (视觉编码器)"].append(key)
    elif "mm_projector" in key:
        categories["Projector (模态投影层)"].append(key)
    elif "model.layers" in key or "lm_head" in key or "embed_tokens" in key or "final_layernorm" in key:
        categories["LLM Backbone (语言模型骨架)"].append(key)
    else:
        categories["Other (其他权重)"].append(key)

# --- 开始打印完整列表 ---
for cat_name, keys in categories.items():
    print(f"\n### {cat_name} - 共 {len(keys)} 个 Key")
    print("-" * 40)
    for k in keys:
        shape = list(weights[k].shape)
        dtype = weights[k].dtype
        # 打印名称、形状和数据类型
        print(f"  {k:<60} | {str(shape):<20} | {dtype}")

print("\n" + "=" * 80)
print(f"📊 扫描总结:")
print(f"  - 总 Key 数量: {len(all_keys)}")
for cat_name, keys in categories.items():
    print(f"  - {cat_name}: {len(keys)} 个")

# 验证几个核心组件的完整性
print("\n✅ 核心组件状态检查:")
critical_components = {
    "Vision Tower": any("vision_tower" in k for k in all_keys),
    "Projector": any("mm_projector" in k for k in all_keys),
    "LLM Layers": any("model.layers.0" in k for k in all_keys),
    "LM Head": "lm_head.weight" in all_keys
}

for component, status in critical_components.items():
    status_str = "OK" if status else "MISSING"
    print(f"  [{status_str}] {component}")

print("=" * 80)
print("💡 提示：你可以通过输出查看每一层具体名称，Stage 3 全解冻时这些 Key 都将被更新。")