import json
import random

# 定义文件路径
input_file = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_abs_fixed.json"
output_file = "/mnt/CoBunny/dataassert/sample_3000_test.json"

print(f"📖 正在加载大规模数据集...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"📊 总样本数: {len(data)}")

# 随机抽取 3000 个样本
if len(data) > 3000:
    sampled_data = random.sample(data, 3000)
    print(f"✅ 已成功随机抽取 3000 个样本。")
else:
    sampled_data = data
    print(f"⚠️ 总样本数不足 3000，已提取全部样本。")

# 保存结果
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"🚀 抽样文件已保存至: {output_file}")