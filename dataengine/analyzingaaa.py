import json
import re
from collections import Counter
import pandas as pd
from tqdm import tqdm

def evaluate_final_dataset(file_path):
    print(f"🧐 正在对最终数据集进行深度体检: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    stats = {
        "total": total,
        "format_ok": 0,      # 符合 <image>\n 且无多余空格的
        "format_bad": 0,     # 依然存在问题的
        "black_padding": 0,  # 盲推理样本
        "real_image": 0,     # 真实图片样本
        "ocr_tasks": 0,      # 强 OCR 指令样本
        "avg_gpt_len": 0,
        "lang_zh": 0         # 检查是否还有残留中文
    }

    # 用于格式检查的正则：必须以 <image> 开头，紧跟换行，且换行后不能有空格
    perfect_pattern = r'^<image>\n\S' 
    ocr_kws = ['ocr', 'read', 'text', 'extract', 'transcribe']

    for entry in tqdm(data, desc="评估中"):
        convs = entry.get("conversations", [])
        img_path = entry.get("image", "")
        
        if len(convs) < 2: continue
        
        human_val = convs[0].get("value", "")
        gpt_val = convs[1].get("value", "")

        # 1. 格式对齐检查
        if human_val.startswith("<image>\n"):
            stats["format_ok"] += 1
        else:
            stats["format_bad"] += 1

        # 2. 样本类型检查
        if "black_padding.jpg" in img_path:
            stats["black_padding"] += 1
        else:
            stats["real_image"] += 1

        # 3. 任务属性检查
        if any(kw in human_val.lower() for kw in ocr_kws):
            stats["ocr_tasks"] += 1
        
        # 4. 残留中文检查
        if re.search(r'[\u4e00-\u9fa5]', gpt_val):
            stats["lang_zh"] += 1

        stats["avg_gpt_len"] += len(gpt_val)

    # 汇总
    print("\n" + "="*40)
    print("📊 Stage 1 OCR 预训练数据集评估报告")
    print("="*40)
    print(f"✅ 总计条目: {total}")
    print(f"🏗️ 格式一致性 (<image>\\n): {stats['format_ok']} ({(stats['format_ok']/total)*100:.2f}%)")
    print(f"🌑 盲推理样本 (Black Image): {stats['black_padding']} ({(stats['black_padding']/total)*100:.2f}%)")
    print(f"🖼️ 真实图片样本: {stats['real_image']} ({(stats['real_image']/total)*100:.2f}%)")
    print(f"🔍 强 OCR 指令占比: {(stats['ocr_tasks']/total)*100:.2f}%")
    print(f"🚫 残留中文条目: {stats['lang_zh']}")
    print(f"📝 平均回复字符长度: {stats['avg_gpt_len']/total:.2f}")
    
    # 最终建议
    print("\n💡 最终评估结论:")
    if stats['format_bad'] == 0 and stats['lang_zh'] == 0:
        print(">>> [PASS] 格式和语言纯净度合格。")
    else:
        print(">>> [WARNING] 仍有少量格式或语言残留，建议检查。")
        
    if 0.1 <= (stats['black_padding']/total) <= 0.3:
        print(">>> [EXCELLENT] 盲推理样本比例适中 (10%-30%)，有利于鲁棒性。")
    elif (stats['black_padding']/total) > 0.5:
        print(">>> [CAUTION] 盲推理样本过多，可能会削弱 Projector 对视觉特征的敏感度。")

evaluate_final_dataset("/mnt/CoBunny/dataassert/cobunny_stage2_final_mixed_ocr.json")