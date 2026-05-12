import json
import os
from tqdm import tqdm
import re

# --- 配置区 ---
# 1. 原始合并后的数据集
JSON_PATH = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_cleaned.json"
# 2. 修复后的输出路径
OUTPUT_PATH = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_abs_fixed.json"

# 3. 扫描根目录（包含所有可能放图片的地方）
SEARCH_ROOTS = [
    "/data",
    "/mnt/conda_data/Bunny-v1.1-data/finetune/images",
    "/data/MAmmoTH-VL-Instruct-12M" # 建议加上 OCR 数据的根目录
]

def build_file_index(roots):
    index = {}
    print(f"📂 正在建立全域文件索引...")
    for root in roots:
        if not os.path.exists(root): continue
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    fname = f
                    parent_dir = os.path.basename(dirpath)
                    # 索引 key 1: 文件夹/文件名 (更精准)
                    index[f"{parent_dir}/{fname}"] = os.path.join(dirpath, f)
                    # 索引 key 2: 纯文件名 (兜底)
                    if fname not in index:
                        index[fname] = os.path.join(dirpath, f)
    print(f"✅ 索引建立完毕，共收录 {len(index):,} 个图像文件。")
    return index

def fix_and_clean():
    # 1. 建立全局索引
    file_cache = build_file_index(SEARCH_ROOTS)

    print(f"📖 正在读取 JSON 并执行深度修复...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {"path_fixed": 0, "tag_removed": 0, "already_ok": 0, "missing_but_text_only": 0}
    
    for item in tqdm(data):
        has_image_field = 'image' in item and item['image']
        # 统计文本中 <image> 数量
        conv_text = "".join([c['value'] for c in item['conversations']])
        has_image_tag = '<image>' in conv_text

        # --- 核心逻辑：路径绝对化 ---
        found_abs_path = None
        if has_image_field:
            rel_path = item['image']
            # 如果已经是绝对路径且存在
            if rel_path.startswith('/') and os.path.exists(rel_path):
                found_abs_path = rel_path
                stats["already_ok"] += 1
            else:
                # 尝试从索引查找
                fname = os.path.basename(rel_path)
                parent = os.path.basename(os.path.dirname(rel_path))
                key = f"{parent}/{fname}"
                
                if key in file_cache:
                    found_abs_path = file_cache[key]
                elif fname in file_cache:
                    found_abs_path = file_cache[fname]

        # --- 核心逻辑：一致性对齐 ---
        if found_abs_path:
            # 找到了图：确保路径是绝对的，且文本里有标签
            item['image'] = found_abs_path
            stats["path_fixed"] += 1
            if not has_image_tag:
                # 如果有图但没标签，补上（针对某些异常样本）
                item['conversations'][0]['value'] = "<image>\n" + item['conversations'][0]['value']
        else:
            # 硬盘里没找到图：
            item['image'] = "" # 清空 image 字段
            if has_image_tag:
                # ‼️ 重点：如果没图但有标签，必须删掉标签，否则训练必崩
                for conv in item['conversations']:
                    conv['value'] = conv['value'].replace("<image>", "").strip()
                stats["tag_removed"] += 1
            else:
                stats["missing_but_text_only"] += 1

    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"📊 修复报告：")
    print(f"✅ 路径成功绝对化: {stats['path_fixed']}")
    print(f"✂️ 删除了非法 <image> 标签 (硬盘无图): {stats['tag_removed']}")
    print(f"⚪ 纯文本样本 (本身无图无标签): {stats['missing_but_text_only']}")
    print(f"🚀 修复后的文件已保存至: {OUTPUT_PATH}")
    print("="*50)

if __name__ == "__main__":
    fix_and_clean()