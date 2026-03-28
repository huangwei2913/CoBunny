import json
import random
import os

# 路径配置
MAMMOTH_PATH = "/data/MAmmoTH-VL-Instruct-12M/mammoth_500k_second_batch.json"
COLOR_PATH = "/mnt/conda_data/Bunny-v1.1-data/finetune/colorbench_data/colorbench_train.json"
OUTPUT_PATH = "/mnt/conda_data/Bunny-v1.1-data/finetune/Bunny_Stage2_Color_Refine_v3.json"

# 绝对路径前缀
BASE_PREFIX = "/mnt/conda_data/Bunny-v1.1-data/finetune/"

# 加载数据
print("正在读取数据集...")
with open(MAMMOTH_PATH, 'r', encoding='utf-8') as f:
    mammoth = json.load(f)
with open(COLOR_PATH, 'r', encoding='utf-8') as f:
    color = json.load(f)

# --- 核心逻辑：替换 ColorBench 的图片路径为绝对地址 ---
print(f"正在转换 ColorBench 路径，目标目录: {BASE_PREFIX}")
for item in color:
    if "image" in item:
        # 原始格式: "colorbench_data/images/color_xxxx.jpg"
        # 目标格式: "/mnt/conda_data/Bunny-v1.1-data/finetune/colorbench_data/images/color_xxxx.jpg"
        rel_path = item["image"]
        item["image"] = os.path.join(BASE_PREFIX, rel_path)

# 核心策略：4:1 比例（ColorBench 约 5.5k，配比 27.5k 猛犸数据）
sample_size = min(27500, len(mammoth))
mammoth_sample = random.sample(mammoth, sample_size)

# 合并并彻底打乱
print("正在合并并随机打乱数据...")
final_mix = mammoth_sample + color
random.shuffle(final_mix)

# 保存结果
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(final_mix, f, ensure_ascii=False, indent=2)

print("-" * 30)
print(f"✅ 数据处理完成！")
print(f"📊 猛犸数据: {len(mammoth_sample)} 条")
print(f"📊 颜色数据: {len(color)} 条 (已补全绝对路径)")
print(f"🚀 总计规模: {len(final_mix)} 条")
print(f"📍 输出路径: {OUTPUT_PATH}")

# 路径校验示例
sample_img = next((item["image"] for item in final_mix if "color" in item["id"]), "None")
print(f"🔍 检查转换后的路径样例: {sample_img}")