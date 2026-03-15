import json
from tqdm import tqdm

input_path = "/data/MAmmoTH-VL-Instruct-12M/mammoth_si_10M_cleaned.json"
output_path = "/data/MAmmoTH-VL-Instruct-12M/mammoth_si_10M_cleaned.jsonl"

print("🚀 正在载入大 JSON 到内存（这步最吃内存，请观察 free -h）...")
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ 载入完成，共 {len(data)} 条数据。正在写入 JSONL...")
with open(output_path, 'w', encoding='utf-8') as f:
    for entry in tqdm(data):
        # 将每个字典转成单行字符串并写入
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"✨ 转换完成！新文件在: {output_path}")