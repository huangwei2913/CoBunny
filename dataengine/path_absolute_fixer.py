import json
import os
from tqdm import tqdm

# --- 配置区 ---
JSON_PATH = "/mnt/CoBunny/dataassert/v365_stage3_mcp_final.json"
OUTPUT_PATH = "/mnt/CoBunny/dataassert/v365_stage3_mcp_final_abs.json"

# 扫描这两个大根目录下的所有子文件夹
SEARCH_ROOTS = [
    "/data",
    "/mnt/conda_data/Bunny-v1.1-data/finetune/images"
]

def build_file_index(roots):
    """
    建立文件名到绝对路径的映射索引，解决层级对不上的问题
    """
    index = {}
    print(f"📂 正在建立全域文件索引，请稍候...")
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    # 键名使用: 文件夹名/文件名，增加匹配精度
                    parent_dir = os.path.basename(dirpath)
                    index[f"{parent_dir}/{f}"] = os.path.join(dirpath, f)
                    # 同时保留纯文件名索引作为兜底
                    if f not in index:
                        index[f] = os.path.join(dirpath, f)
    print(f"✅ 索引建立完毕，共收录 {len(index):,} 个图像文件。")
    return index

def fix_paths_v2():
    if not os.path.exists(JSON_PATH):
        return

    # 1. 建立索引
    file_cache = build_file_index(SEARCH_ROOTS)

    print(f"📖 正在读取 JSON: {JSON_PATH}")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_count = 0
    already_abs_count = 0
    missing_samples = []

    print("🔍 正在利用全域索引补全路径...")
    for item in tqdm(data):
        if 'image' not in item: continue
        rel_path = item['image']

        # 已经是绝对路径
        if rel_path.startswith('/'):
            already_abs_count += 1
            continue

        # 尝试匹配策略 1: 直接拼接 (最快)
        found = False
        for root in SEARCH_ROOTS:
            if os.path.exists(os.path.join(root, rel_path)):
                item['image'] = os.path.join(root, rel_path)
                updated_count += 1
                found = True
                break
        
        if found: continue

        # 尝试匹配策略 2: 文件名索引 (解决层级偏移)
        fname = os.path.basename(rel_path)
        parent = os.path.basename(os.path.dirname(rel_path))
        key = f"{parent}/{fname}"
        
        if key in file_cache:
            item['image'] = file_cache[key]
            updated_count += 1
        elif fname in file_cache:
            item['image'] = file_cache[fname]
            updated_count += 1
        else:
            missing_samples.append(rel_path)

    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"📊 最终路径处理报告：")
    print(f"✅ 成功补全: {updated_count:,}")
    print(f"⚪ 原本即绝对路径: {already_abs_count:,}")
    print(f"❌ 依然找不到: {len(missing_samples):,}")
    print("="*50)

    if missing_samples:
        print("\n🧐 抽样检查依然找不到的路径（前 5 个）：")
        for s in missing_samples[:5]:
            print(f"  - {s}")

if __name__ == "__main__":
    fix_paths_v2()