import json
import os
from tqdm import tqdm

# --- 配置区 ---
# 输入：你刚才合并完成的文件
INPUT_JSON = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_cleaned.json"
# 输出：最终可以直接喂给训练脚本的文件
OUTPUT_JSON = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_cleaned_ABS.json"

# 图像搜索的根目录（根据你之前的路径信息汇总）
SEARCH_ROOTS = [
    "/data/MAmmoTH-VL-Instruct-12M/single_image_data",
    "/data/MAmmoTH-VL-Instruct-12M",
    "/data"
]

def build_fast_index(roots):
    """
    建立文件名索引，解决类似 'tinychart_train/1335489.png' 这种层级对不上的问题
    """
    index = {}
    print(f"📂 正在扫描图像文件并建立索引...")
    for root in roots:
        if not os.path.exists(root): continue
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    # 1. 存储 '文件夹/文件名' 格式 (如 tinychart_train/1335489.png)
                    parent_dir = os.path.basename(dirpath)
                    index[f"{parent_dir}/{f}"] = os.path.join(dirpath, f)
                    # 2. 存储 '纯文件名' 格式作为保底
                    if f not in index:
                        index[f] = os.path.join(dirpath, f)
    print(f"✅ 索引建立完毕，共收录 {len(index):,} 个文件。")
    return index

def fix_image_paths():
    if not os.path.exists(INPUT_JSON):
        print(f"❌ 找不到输入文件: {INPUT_JSON}")
        return

    # 1. 预构建索引
    file_index = build_fast_index(SEARCH_ROOTS)

    print(f"📖 正在读取数据样本...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fixed_count = 0
    already_abs_count = 0
    failed_count = 0
    failed_list = []

    print("🔍 正在转换相对路径为绝对路径...")
    for item in tqdm(data):
        if 'image' not in item or not item['image']:
            continue
        
        rel_path = item['image']

        # --- 策略 0: 如果已经是绝对路径，直接跳过 ---
        if rel_path.startswith('/'):
            already_abs_count += 1
            continue

        # --- 策略 1: 直接拼接尝试 (最快) ---
        found = False
        for root in SEARCH_ROOTS:
            full_path = os.path.join(root, rel_path)
            if os.path.exists(full_path):
                item['image'] = full_path
                fixed_count += 1
                found = True
                break
        
        if found: continue

        # --- 策略 2: 索引模糊匹配 (解决路径层级错位) ---
        # 提取文件名 (1335489.png) 和 父目录 (tinychart_train)
        fname = os.path.basename(rel_path)
        parent = os.path.basename(os.path.dirname(rel_path))
        key = f"{parent}/{fname}"
        
        if key in file_index:
            item['image'] = file_index[key]
            fixed_count += 1
        elif fname in file_index:
            item['image'] = file_index[fname]
            fixed_count += 1
        else:
            failed_count += 1
            failed_list.append(rel_path)

    # 3. 保存结果
    print(f"📝 正在保存最终结果到: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(f"📊 路径修复报告")
    print("-" * 50)
    print(f"✅ 成功补全为绝对路径: {fixed_count}")
    print(f"⚪ 已经是绝对路径:     {already_abs_count}")
    print(f"❌ 依然找不到物理文件: {failed_count}")
    print("="*50)

    if failed_list:
        print("\n🧐 错误样例（检查一下这些路径是否真的存在）：")
        for p in failed_list[:5]:
            print(f"  - {p}")

if __name__ == "__main__":
    fix_image_paths()