import json
import random
import os
from tqdm import tqdm

def is_perfect_single_turn(item):
    """
    最硬核的结构质检：
    1. 必须有 id 和 conversations
    2. conversations 必须且只能有 2 轮 (Human/GPT)
    3. 内容不能为空
    """
    try:
        # 字段存在性
        if not all(k in item for k in ['id', 'conversations']):
            return False
        
        convs = item['conversations']
        # 强制单轮 (1 Human + 1 GPT = 2)
        if not isinstance(convs, list) or len(convs) != 2:
            return False
            
        # 角色对齐检查
        if convs[0].get('from') != 'human' or convs[1].get('from') != 'gpt':
            return False
            
        # 文本内容完整性检查 (排除碎裂样本)
        if not str(convs[0].get('value', '')).strip() or not str(convs[1].get('value', '')).strip():
            return False
            
        return True
    except:
        return False

def merge_datasets_pure():
    ocr_file = "/data/MAmmoTH-VL-Instruct-12M/pure_english_ocr_stage1_single_turn.json"
    vision_file = "/data/MAmmoTH-VL-Instruct-12M/general_vision_stage1_single_turn.json"
    logic_file = "/mnt/CoBunny/dataassert/SlimOrca_Stage1_Fixed_With_Images.json"
    output_file = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_cleaned.json"

    final_list = []
    stats = {"ocr": 0, "vision": 0, "logic": 0, "dropped": 0}

    def process_source(file_path, tag):
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过缺失文件: {file_path}")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        for item in tqdm(raw_data, desc=f"合并 {tag}"):
            # 1. 结构化质检 (自动剔除你之前发现的碎片数据和多轮数据)
            if not is_perfect_single_turn(item):
                stats["dropped"] += 1
                continue
            
            # 2. 物理重组：手动提取字段，彻底杀掉“重复Key”和“畸形Key”
            # 直接使用原始的 value，不准乱加 OCR 指令
            clean_item = {
                "id": str(item['id']),
                "conversations": [
                    {"from": "human", "value": item['conversations'][0]['value']},
                    {"from": "gpt", "value": item['conversations'][1]['value']}
                ]
            }
            
            # 如果有图片，保留图片字段
            if 'image' in item and item['image']:
                clean_item["image"] = item['image']
                
            final_list.append(clean_item)
            stats[tag] += 1

    # 按顺序合并
    process_source(ocr_file, "ocr")
    process_source(vision_file, "vision")
    process_source(logic_file, "logic")

    print("\n" + "="*50)
    print(f"📊 抢救性合并报告")
    print("-" * 50)
    print(f"✅ OCR 样本: {stats['ocr']}")
    print(f"✅ Vision 样本: {stats['vision']}")
    print(f"✅ Logic 样本: {stats['logic']}")
    print(f"❌ 剔除坏账(多轮/碎裂/空值): {stats['dropped']}")
    print(f"🌟 最终可用总量: {len(final_list)}")
    print("="*50)

    # 随机打乱
    random.seed(42)
    random.shuffle(final_list)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    
    print(f"🚀 合并完成！已存至 {output_file}")

if __name__ == "__main__":
    merge_datasets_pure()