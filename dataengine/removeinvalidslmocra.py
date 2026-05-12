import json
import re

def contains_non_english(text):
    if not text: return False
    return re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text) is not None

def is_invalid_content(text):
    """检测是否是无效内容（空、太短、或者只有引导词）"""
    if not text: return True
    clean_text = text.strip().lower()
    # 如果去掉空格后太短（比如小于 5 个字符），或者是纯引导词
    if len(clean_text) < 5: return True
    if clean_text in ["a:", "answer:", "yes.", "no.", "ok."]: return True
    return False

def convert_refined_v3():
    input_file = "/data/SlimOrca/SlimOrca_for_Bunny_Stage1.json"
    output_file = "/data/SlimOrca/SlimOrca_for_Bunny_Stage1_Refined.json"
    
    MAX_CHAR_LIMIT = 2000 
    
    print("正在加载数据并启动终极清洗...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    converted_data = []
    skip_non_en = 0
    skip_long = 0
    skip_cot = 0
    skip_invalid = 0 # 统计残缺或无效数据
    
    for item in raw_data:
        try:
            convs = item['conversations']
            # 必须有两个角色
            if len(convs) < 2:
                skip_invalid += 1
                continue
                
            human_val = convs[0]['value']
            gpt_val = convs[1]['value']
            
            # --- 1. 完整性检查（剔除只有问或只有答，或者回答太短的） ---
            if is_invalid_content(human_val) or is_invalid_content(gpt_val):
                skip_invalid += 1
                continue

            # --- 2. 语言过滤 ---
            if contains_non_english(gpt_val) or contains_non_english(human_val):
                skip_non_en += 1
                continue
            
            # --- 3. 剔除思维链冗余 (CoT) ---
            low_quality_keywords = [
                "to come up with the answer", 
                "step 1:", "step 1.", 
                "let's think step by step",
                "the original sentence",
                "translate the"
            ]
            if any(kw in gpt_val.lower() for kw in low_quality_keywords):
                skip_cot += 1
                continue

            # --- 4. 长度过滤 ---
            total_length = len(human_val) + len(gpt_val)
            if total_length > MAX_CHAR_LIMIT:
                skip_long += 1
                continue
            
            # 记录有效的单轮对话
            converted_data.append({
                "id": f"slimorca_final_{len(converted_data)}",
                "image": "", 
                "conversations": convs
            })
                
        except Exception:
            skip_invalid += 1
            continue

    print(f"✅ 终极清洗完成！")
    print(f"保留纯净单轮逻辑: {len(converted_data)}")
    print(f"❌ 剔除残缺/无效(只有问或只有答): {skip_invalid}")
    print(f"❌ 剔除外语(日/中): {skip_non_en}")
    print(f"❌ 剔除冗余解释(CoT): {skip_cot}")
    print(f"❌ 剔除超长文本: {skip_long}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    convert_refined_v3()