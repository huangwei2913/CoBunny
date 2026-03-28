import json
import random
import os

# --- 路径配置 ---
MAMMOTH_PATH = "/data/MAmmoTH-VL-Instruct-12M/mammoth_500k_second_batch.json"
COLOR_PATH = "/mnt/conda_data/Bunny-v1.1-data/finetune/colorbench_data/colorbench_train.json"
# 绝对路径版 VG 数据
VG_ONLY_PATH = "/data/ShareGPT4V/sharegpt4v_vg_only_clean.json"

# 输出路径
OUTPUT_PATH = "/mnt/conda_data/Bunny-v1.1-data/finetune/Bunny_Stage3_Full_Mix_v1.json"

# 仅用于补全 ColorBench 的前缀
COLOR_BASE_PATH = "/mnt/conda_data/Bunny-v1.1-data/finetune/"

def load_json(path):
    print(f"正在读取: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 1. 加载所有原始数据
mammoth = load_json(MAMMOTH_PATH)
color = load_json(COLOR_PATH)
vg_only = load_json(VG_ONLY_PATH)

# 2. 处理 ColorBench 路径（因为它原先是相对路径）
print("正在修正 ColorBench 相对路径为绝对路径...")
for item in color:
    if "image" in item:
        # 补全为 /mnt/conda_data/Bunny-v1.1-data/finetune/colorbench_data/images/...
        item["image"] = os.path.join(COLOR_BASE_PATH, item["image"])

# 3. 处理 VG 和 猛犸路径
# 既然已经是绝对路径且在机器上真实存在，我们直接透传，无需额外处理
print("VG 与 猛犸路径已确认为绝对地址，直接导入...")

# 4. 抽样平衡策略
# 建议：颜色全量(5.5k)，VG全量或大比例抽样，猛犸抽样3万条作为通用基石
vg_sample_size = min(20000, len(vg_only)) # 视 VG 数据总量而定，如果不多可以全拿
mammoth_sample_size = 30000

sampled_vg = random.sample(vg_only, vg_sample_size)
sampled_mammoth = random.sample(mammoth, mammoth_sample_size)

# 5. 合并并随机打乱
print(f"--- 最终规模统计 ---")
print(f"📊 猛犸数据: {len(sampled_mammoth)} 条")
print(f"📊 颜色数据: {len(color)} 条")
print(f"📊 视觉定位 (VG): {len(sampled_vg)} 条")

final_mix = sampled_mammoth + color + sampled_vg
random.shuffle(final_mix)

# 6. 保存最终训练集
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(final_mix, f, ensure_ascii=False, indent=2)

print("-" * 30)
print(f"✅ Stage 3 混合数据集制作完成！")
print(f"📍 存储位置: {OUTPUT_PATH}")
print(f"🚀 总样本数: {len(final_mix)}")

# 抽样检查路径格式是否统一
print("\n🔍 路径样例检查:")
print(f"Color 样例: {color[0]['image']}")
print(f"VG 样例: {sampled_vg[0]['image']}")
print(f"猛犸 样例: {sampled_mammoth[0]['image']}")