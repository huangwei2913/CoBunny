import json
import os
import re
from collections import Counter
import pandas as pd
from tqdm import tqdm

def analyze_file_quality(file_path):
    if not os.path.exists(file_path):
        return None
    
    print(f"\n[正在挖掘] {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            print(f"❌ 解析失败: {file_path}")
            return None

    total = len(data)
    stats = {
        "filename": os.path.basename(file_path),
        "total_samples": total,
        "chinese_samples": 0,
        "ocr_instruction_count": 0,
        "too_short_response": 0,  # 只有几个单词的垃圾回复
        "avg_length": 0
    }

    # OCR 关键词挖掘
    ocr_keywords = ['ocr', 'read', 'text', 'transcribe', 'extract', 'write', '文字', '识别']
    
    for entry in tqdm(data, desc="扫描中"):
        convs = entry.get("conversations", [])
        if len(convs) < 2: continue
        
        prompt = convs[0].get("value", "").lower()
        response = convs[1].get("value", "")

        # 1. 统计中文 (只要包含中文字符就计入)
        if re.search(r'[\u4e00-\u9fa5]', response):
            stats["chinese_samples"] += 1
            
        # 2. 统计 OCR 指令 (判断 Prompt 是否在要求识别文字)
        if any(kw in prompt for kw in ocr_keywords):
            stats["ocr_instruction_count"] += 1
            
        # 3. 统计过短回复 (如果是 OCR，通常不会只有一两个词，除非是极简 Caption)
        if len(response.split()) < 5:
            stats["too_short_response"] += 1
            
        stats["avg_length"] += len(response)

    stats["avg_length"] /= total if total > 0 else 1
    stats["zh_ratio"] = f"{(stats['chinese_samples']/total)*100:.2f}%"
    stats["ocr_ratio"] = f"{(stats['ocr_instruction_count']/total)*100:.2f}%"
    
    return stats

# 待检查的核心文件列表 (根据你 ll 的结果挑选)
files_to_check = [
    "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_cleaned_ABS.json",
    "/mnt/CoBunny/dataassert/mixed_stage1_2m_stable.json",
    "/mnt/CoBunny/dataassert/ocr_5_ready_fixed_invalid.json",
    "/mnt/CoBunny/dataassert/v365_stage3_mcp_final_clean_fixed.json"
]

all_results = []
for f in files_to_check:
    res = analyze_file_quality(f)
    if res:
        all_results.append(res)

# 打印对比结果
df = pd.DataFrame(all_results)
print("\n" + "="*50)
print("📊 数据集质量横向对比表")
print("="*50)
print(df[['filename', 'total_samples', 'zh_ratio', 'ocr_ratio', 'too_short_response', 'avg_length']])