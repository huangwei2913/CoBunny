import json
import re

def is_chinese(text):
    # 再次检查，确保没有漏网的中文
    return re.search(r'[\u4e00-\u9faf]', text) is not None

def refine_to_single_turn():
    input_file = "/data/MAmmoTH-VL-Instruct-12M/pure_english_ocr_stage1.json"
    output_file = "/data/MAmmoTH-VL-Instruct-12M/pure_english_ocr_stage1_single_turn.json"
    
    print(f"正在清洗 {input_file}，剔除多轮对话和残留中文...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    final_data = []
    multi_turn_count = 0
    chinese_count = 0
    
    for item in data:
        convs = item.get('conversations', [])
        
        # 1. 核心卡点：只要单轮 (1个Human问 + 1个GPT答 = 2)
        if len(convs) != 2:
            multi_turn_count += 1
            continue
            
        # 2. 再次语义检查：防止之前的脚本漏掉中文
        full_text = "".join([c['value'] for c in convs])
        if is_chinese(full_text):
            chinese_count += 1
            continue
            
        # 3. 指令归一化 (可选，建议做)：
        # 第一阶段任务越纯粹越好，把零散的提问统一成强指令
        # item['conversations'][0]['value'] = "<image>\nOCR: Extract all text from this image to Markdown."
        
        final_data.append(item)
        
    print(f"清洗完成！")
    print(f"✅ 保留单轮纯英样本: {len(final_data)}")
    print(f"❌ 剔除多轮对话: {multi_turn_count}")
    print(f"❌ 剔除残留中文: {chinese_count}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    refine_to_single_turn()