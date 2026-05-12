import json
import os
from tqdm import tqdm

INPUT_FILE = "/mnt/CoBunny/dataassert/stage1_final_cleaned_fixed_.json"
OUTPUT_FILE = "/mnt/CoBunny/dataassert/stage1_pretrain_final_pure_100pct.json"

def is_perfect_format(entry):
    convs = entry.get("conversations", [])
    if not convs:
        return False
    
    # 严格检查：human 端必须以 "<image>\n" 开头
    human_val = convs[0].get("value", "")
    return human_val.startswith("<image>\n")

def run_clean():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"🚀 正在从 {len(data)} 条数据中筛选 100% 合规样本...")
    
    # 只保留格式完美的
    clean_data = [item for item in tqdm(data) if is_perfect_format(item)]
    
    removed_count = len(data) - len(clean_data)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*40)
    print(f"✨ 物理删除完成！")
    print(f"🗑️ 已剔除格式不规范样本: {removed_count} 条")
    print(f"✅ 最终剩余完美样本: {len(clean_data)} 条")
    print(f"💾 最终训练文件路径: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    run_clean()