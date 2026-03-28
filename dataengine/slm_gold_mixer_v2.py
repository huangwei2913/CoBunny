import json
import random
import os
from tqdm import tqdm

# --- 配置区 ---
DATA_DIR = "/mnt/CoBunny/dataassert"
OUTPUT_FILE = "/mnt/CoBunny/dataassert/v365_stage3_mcp_final.json"
MCP_PROB = 0.15  # 15% 的 Fashion 和 MAmmoTH 数据会被转化为选择题格式

# 精准配方 (根据审计报告调整)
DATA_RECIPE = {
    # 核心资产：加强与保量
    "fashion_full_scan_alignment.json": 1.0,  # 8.6w 核心，全留
    "echo4o_hard_vqa_refined.json": 1.5,      # 4.4w -> 6.6w 纠错特效药
    "blip3o_final_sft.json": 1.0,             # 5.8w 高质量底色
    
    # 逻辑与对话：降噪保质
    "mammoth_500k_pilot.json": 0.3,           # 50w -> 15w (防止逻辑过载)
    "bunny_high_quality_final.json": 0.7,     # 21w -> 14.7w (维持对话感)
    "vqa_9_ready.json": 1.0,                  # 4.6w
    
    # OCR 军团：极度压制长文本，防止复读
    "ocr_4_ready.json": 0.08,                 # 18.8w -> 1.5w (G-Len 1666太沉了)
    "ocr_5_ready.json": 0.08,                 # 19.3w -> 1.5w
    "ocr_8_ready.json": 0.1,                  # 5.7w -> 0.5w
    "ocr_1_ready.json": 1.0,                  # 1.4w 小文件全留
    "ocr_2_ready.json": 1.0,                  # 2.1w
    "ocr_3_ready.json": 1.0,                  # 1.4w
    "mathvista_for_train.json": 1.0           # 0.1w
}

# 预定义的干扰项池 (针对 Fashion 数据)
FASHION_DISTRACTORS = ["skirt", "t-shirt", "dress", "jacket", "pants", "blouse", "coat", "jeans", "sweater"]
COLOR_DISTRACTORS = ["red", "blue", "green", "yellow", "black", "white", "pink", "purple", "navy"]

def apply_mcp(item, file_type):
    """
    将普通 QA 转换为 MCP (Multiple Choice Prompt) 格式
    """
    try:
        convs = item['conversations']
        if len(convs) < 2: return item
        
        original_ans = convs[1]['value']
        # 简单清洗，只针对短答案进行 MCP 转化，效果最好
        if len(original_ans) > 50: return item
        
        # 选择干扰项池
        pool = COLOR_DISTRACTORS if any(c in original_ans.lower() for c in COLOR_DISTRACTORS) else FASHION_DISTRACTORS
        distractors = random.sample([d for d in pool if d not in original_ans.lower()], 3)
        options = distractors + [original_ans]
        random.shuffle(options)
        
        correct_idx = options.index(original_ans)
        correct_letter = "ABCD"[correct_idx]
        
        mcp_question = (
            f"{convs[0]['value']}\n"
            f"Please choose the correct answer from the following options:\n"
            f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n"
            f"Answer with the option letter directly."
        )
        
        item['conversations'] = [
            {"from": "human", "value": mcp_question},
            {"from": "gpt", "value": correct_letter}
        ]
    except:
        pass
    return item

def main():
    final_data = []
    random.seed(42)

    for filename, weight in DATA_RECIPE.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path): continue
        
        print(f"🛠️ 处理中: {filename} (权重: {weight})")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 1. 采样逻辑
            sample_count = int(len(data) * weight)
            if weight >= 1.0:
                sampled = (data * (int(weight) + 1))[:sample_count]
            else:
                sampled = random.sample(data, sample_count)
            
            # 2. MCP 注入逻辑 (针对 Fashion 和 MAmmoTH)
            if filename in ["fashion_full_scan_alignment.json", "mammoth_500k_pilot.json"]:
                mcp_count = 0
                for i in range(len(sampled)):
                    if random.random() < MCP_PROB:
                        sampled[i] = apply_mcp(sampled[i], filename)
                        mcp_count += 1
                print(f"   └─ 成功注入 MCP 样本: {mcp_count}")

            final_data.extend(sampled)
            print(f"   └─ 最终贡献条数: {len(sampled):,}")

    # 3. 全局混洗 (打断数据聚集，防止学习偏见)
    print("\n🎲 正在进行全量数据 Global Shuffle...")
    random.shuffle(final_data)

    # 4. 写入文件
    print(f"💾 正在保存至: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"\n✨ 炼丹材料准备完毕！总样本数: {len(final_data):,}")
    print(f"🚀 建议 1.5B 模型微调参数：LR=8e-6, Epoch=1, LoRA Rank=64")

if __name__ == "__main__":
    main()