import json
import re

def has_chinese(text):
    """检测字符串中是否包含中文字符"""
    return len(re.findall(r'[\u4e00-\u9fa5]', text)) > 0

def format_instruction(text):
    """
    强制格式化为: <image>\nYour Question
    去掉所有多余的空白、首尾空格，确保只有单个 \n 分隔
    """
    # 1. 先去掉所有的 <image> 占位符及其前后的空白
    content = text.replace("<image>", "").strip()
    # 2. 重新拼接成标准的格式
    return f"<image>\n{content}"

input_path = '/mnt/CoBunny/dataassert/cobunny_stage2_final_mixed_ocr.json'
output_path = '/mnt/CoBunny/dataassert/cobunny_stage2_final_pure_ocr_en.json'

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

cleaned_data = []
removed_chinese_count = 0
formatted_count = 0

print(f"开始处理，原始总计: {len(data)} 条")

for entry in data:
    # --- 1. 中文过滤逻辑 ---
    # 同时检查人类的问题和模型的回答，只要有中文就干掉
    instruction = entry['conversations'][0]['value']
    response = entry['conversations'][1]['value']
    
    if has_chinese(instruction) or has_chinese(response):
        removed_chinese_count += 1
        continue
    
    # --- 2. 格式强制转换逻辑 ---
    # 确保 human 的 value 严格遵循 <image>\nContent
    original_val = entry['conversations'][0]['value']
    new_val = format_instruction(original_val)
    
    if original_val != new_val:
        entry['conversations'][0]['value'] = new_val
        formatted_count += 1
    
    cleaned_data.append(entry)

# 保存处理后的数据
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

print("-" * 30)
print(f"处理完成！保存至: {output_path}")
print(f"✅ 最终剩余条目: {len(cleaned_data)}")
print(f"🚫 剔除中文条目: {removed_chinese_count}")
print(f"🛠️ 格式修正条目: {formatted_count}")
print(f"🎯 现在的格式样例:\n{cleaned_data[0]['conversations'][0]['value'][:30]}...")