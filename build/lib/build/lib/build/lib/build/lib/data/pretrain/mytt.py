import json

input_file = 'filtered_data.json'
output_file = 'filtered_data_compact.json'

with open(input_file, 'r', encoding='utf-8') as fin:
    data = json.load(fin)

with open(output_file, 'w', encoding='utf-8') as fout:
    json.dump(data, fout, ensure_ascii=False, separators=(',', ':'))

