import json

# 把你的 50k 抽样文件转为标准 JSON 格式
jsonl_file = '/data/MAmmoTH-VL-Instruct-12M/mammoth_500k_second_batch.jsonl'
output_json = '/data/MAmmoTH-VL-Instruct-12M/mammoth_500k_second_batch.json'

with open(jsonl_file, 'r', encoding='utf-8') as f:
    # 逐行读取并转为 list
    data = [json.loads(line) for line in f]

with open(output_json, 'w', encoding='utf-8') as f:
    # 一次性写入为一个标准的 JSON List
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 转换完成，请将训练参数中的 data_path 改为 .json 后缀的文件")