import json
import os
import re
from collections import Counter
from tqdm import tqdm

# --- 配置区 ---
DATA_DIR = "/mnt/CoBunny/dataassert"
# 关注的关键词类别，用于识别数据“偏向”
KEYWORDS = {
    "color": ["red", "green", "blue", "yellow", "white", "black", "burgundy", "navy", "pink"],
    "count": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "spatial": ["top", "bottom", "left", "right", "grid", "position", "coordinate", "bracket"],
    "fashion": ["skirt", "dress", "jacket", "pants", "shirt", "blouse", "sleeve", "neck", "fabric"],
    "logic": ["table", "chart", "calculate", "math", "reason", "because", "therefore"]
}

def analyze_json(file_path):
    stats = {
        "count": 0,
        "avg_len_human": 0,
        "avg_len_gpt": 0,
        "keyword_hits": Counter(),
        "total_human_chars": 0,
        "total_gpt_chars": 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            stats["count"] = len(data)
            
            for item in data:
                convs = item.get("conversations", [])
                for conv in convs:
                    role = conv.get("from")
                    val = conv.get("value", "").lower()
                    
                    if role == "human":
                        stats["total_human_chars"] += len(val)
                    elif role == "gpt":
                        stats["total_gpt_chars"] += len(val)
                    
                    # 关键词扫描
                    for cat, words in KEYWORDS.items():
                        for w in words:
                            if w in val:
                                stats["keyword_hits"][cat] += 1

            if stats["count"] > 0:
                stats["avg_len_human"] = stats["total_human_chars"] / stats["count"]
                stats["avg_len_gpt"] = stats["total_gpt_chars"] / stats["count"]
                
    except Exception as e:
        return f"Error: {e}"
    
    return stats

def main():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    print(f"🔍 开始审计目录: {DATA_DIR}")
    print(f"📊 扫描目标: {len(files)} 个 JSON 文件\n")

    results = {}
    for filename in tqdm(files, desc="Analyzing"):
        path = os.path.join(DATA_DIR, filename)
        results[filename] = analyze_json(path)

    # --- 打印报告 ---
    header = f"{'Filename':<35} | {'Count':>10} | {'H-Len':>6} | {'G-Len':>6} | {'Top Category'}"
    print("\n" + "="*95)
    print(header)
    print("-"*95)

    for name, s in results.items():
        if isinstance(s, str):
            print(f"{name:<35} | {s}")
            continue
            
        # 找出命中率最高的关键词类别
        top_cat = "None"
        if s["keyword_hits"]:
            top_cat = s["keyword_hits"].most_common(1)[0][0]
            
        print(f"{name:<35} | {s['count']:>10,} | {int(s['avg_len_human']):>6} | {int(s['avg_len_gpt']):>6} | {top_cat}")

    print("="*95)
    print("\n💡 审计建议：")
    print("1. 如果 G-Len (GPT回答长度) 过短，说明该数据集倾向于简单识别，容易导致复读。")
    print("2. 重点观察 MAmmoTH 和 Fashion 的 G-Len 差异，这决定了模型说话的‘详细程度’。")
    print("3. 如果某 OCR 文件的 Top Category 是 spatial，说明其坐标数据极多，需谨慎配比。")

if __name__ == "__main__":
    main()