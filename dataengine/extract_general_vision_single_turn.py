import json
import re

def is_chinese(text):
    """判断是否包含中文，用于过滤掉非目标样本"""
    return re.search(r'[\u4e00-\u9faf]', text) is not None

def extract_general_vision():
    input_file = "/data/MAmmoTH-VL-Instruct-12M/mammoth_si_10M.json"
    output_file = "/data/MAmmoTH-VL-Instruct-12M/general_vision_stage1_single_turn.json"
    
    # 核心通用视觉来源：ShareGPT4V (最顶级的描述) 和 SVIT (空间感知)
    vision_sources = [
        "sharegpt4v", 
        "svit", 
        "llava_instruct", 
        "allava"
    ]
    
    refined_data = []
    max_vision_samples = 300000  # 建议通用视觉在第一阶段保持在 20-30万条左右即可
    
    print(f"开始打捞通用视觉样本 (目标源: {vision_sources})...")

    # 依然建议使用 ijson 或者如果内存够大直接处理
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for item in data:
            if len(refined_data) >= max_vision_samples:
                break
                
            convs = item.get('conversations', [])
            source = item.get('source', '').lower()
            
            # --- 严格过滤逻辑 ---
            # 1. 只要单轮 (1 Human + 1 GPT)
            if len(convs) != 2:
                continue
            
            # 2. 来源过滤：只从通用视觉源里捞
            if not any(vs in source for vs in vision_sources):
                continue
                
            # 3. 语种过滤：绝对不要中文
            full_text = "".join([c['value'] for c in convs])
            if is_chinese(full_text):
                continue

            # 4. 长度限制
            if len(full_text) > 2000:
                continue

            # 统一任务前缀（通用视觉不需要强制 OCR 指令，保持原样或微调）
            # 这里的目的是让模型学会描述图片，比如 "Describe this image in detail."
            
            refined_data.append(item)
            
            if len(refined_data) % 50000 == 0:
                print(f"已打捞通用视觉样本: {len(refined_data)} 条...")

    print(f"打捞完成！共获取 {len(refined_data)} 条通用视觉单轮样本。")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    extract_general_vision()