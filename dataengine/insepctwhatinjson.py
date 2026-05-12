import json
import random

# 定义你的原始清洗后的文件路径
input_file = '/mnt/CoBunny/dataassert/stage1_pretrain_final_pure_100pct.json'
output_file = '/mnt/CoBunny/dataassert/sample_2000_for_analysis.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 随机抽取2000个
if len(data) > 2000:
    sampled_data = random.sample(data, 2000)
else:
    sampled_data = data

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)

print(f"抽样完成，已保存至: {output_file}")
print(f"样本包含字段示例: {list(sampled_data[0].keys())}")