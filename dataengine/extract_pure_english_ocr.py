import json
import re

def is_chinese(text):
    """判断字符串中是否包含中文"""
    return re.search(r'[\u4e00-\u9faf]', text) is not None

def extract_pure_ocr():
    input_file = "/data/MAmmoTH-VL-Instruct-12M/mammoth_si_10M.json"
    output_file = "/data/MAmmoTH-VL-Instruct-12M/pure_english_ocr_stage1.json"
    
    # 1. 来源白名单：这些就是您指定的硬核 OCR 源
    ocr_sources = [
        "ocr_junpeng", 
        "textocr", 
        "ureader_ocr", 
        "ureader_chart",
        "tinychart",
        "dvqa",      # 图表类 OCR
        "chartqa",   # 图表类 OCR
        "docvqa"     # 文档类 OCR
    ]
    
    # 2. 补充关键词：防止漏掉一些没有标注 source 但实际是 OCR 的样本
    ocr_keywords = ["ocr", "extract text", "transcribe", "read all", "markdown table"]
    
    extracted_data = []
    
    print("开始从 21G Mammoth 库中打捞纯英 OCR 数据...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # 建议内存足够时直接 load，T4 机器通常有 256G 内存则无压力
        data = json.load(f)
        
        for item in data:
            source = item.get('source', '').lower()
            convs = item.get('conversations', [])
            
            # 合并所有文本用于检测
            full_text = "".join([c['value'] for c in convs])
            
            # --- 过滤逻辑 ---
            # A. 语种检查：只要包含中文，直接毙掉
            if is_chinese(full_text):
                continue
                
            # B. 长度检查：防止超长样本爆显存（设为 2500 字符）
            if len(full_text) > 2500:
                continue
            
            # C. 核心打捞：
            # 逻辑：属于白名单来源 OR 对话里明确提到 OCR 任务
            is_target_source = any(s in source for s in ocr_sources)
            is_ocr_task = any(kw in full_text.lower() for kw in ocr_keywords)
            
            if is_target_source or is_ocr_task:
                # 统一 ID 格式，方便追踪
                item['id'] = f"mammoth_pure_en_{len(extracted_data)}"
                
                # 统一 Task Prompt (FireRed 风格)
                # 如果是 OCR 任务，确保 human 指令里有强引导词
                if "<image>" in convs[0]['value']:
                    # 可以在这里对指令进行微调，让它更统一
                    pass
                
                extracted_data.append(item)
            
            if len(extracted_data) % 100000 == 0 and len(extracted_data) > 0:
                print(f"已打捞 {len(extracted_data)} 条纯英 OCR 样本...")

    print(f"打捞完成！共获得 {len(extracted_data)} 条高质量纯英 OCR 样本。")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    extract_pure_ocr()