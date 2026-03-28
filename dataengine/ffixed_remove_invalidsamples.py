import json
import re

original_path = "/mnt/CoBunny/dataassert/v365_stage3_mcp_final_clean.json"
fixed_path = "/mnt/CoBunny/dataassert/v365_stage3_mcp_final_clean_fixed.json"

with open(original_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

clean_data = []
dropped_ids = []

for item in data:
    # 统计图片数
    actual_img = 1 if ('image' in item and item['image']) else (len(item['images']) if 'images' in item else 0)
    
    # 统计占位符
    total_text = ""
    for conv in item['conversations']:
        total_text += conv['value']
    placeholder_count = len(re.findall(r'<image>', total_text))

    # 只有完全对齐的才保留
    if placeholder_count == actual_img:
        clean_data.append(item)
    else:
        dropped_ids.append(item.get('id', 'N/A'))

with open(fixed_path, 'w', encoding='utf-8') as f:
    json.dump(clean_data, f, indent=2, ensure_ascii=False)

print(f"🎉 清洗完成！")
print(f"原始规模: {len(data)}")
print(f"剔除坏样本: {len(dropped_ids)} 个")
print(f"最终规模: {len(clean_data)}")
print(f"新文件已保存至: {fixed_path}")