import json
import os
import hashlib
from tqdm import tqdm

# --- 配置区 ---
SOURCE_JSON = "/mnt/CoBunny/dataassert/v365_stage3_mcp_final.json"
CLEAN_JSON = "/mnt/CoBunny/dataassert/v365_stage3_mcp_final_clean.json"

SEARCH_ROOTS = [
    "/data",
    "/mnt/conda_data/Bunny-v1.1-data/finetune/images"
]

def build_file_index(roots):
    """建立物理文件快速索引"""
    index = {}
    print("📂 正在扫描物理磁盘，建立文件索引...")
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    # 索引策略：全路径存储，键为 文件名 或 文件夹/文件名
                    fname = f
                    parent = os.path.basename(dirpath)
                    abs_path = os.path.join(dirpath, f)
                    
                    index[f"{parent}/{fname}"] = abs_path
                    if fname not in index:
                        index[fname] = abs_path
    print(f"✅ 索引构建完毕，共发现 {len(index):,} 个物理文件。")
    return index

def get_sample_hash(item, abs_path):
    """生成样本的唯一指纹用于去重"""
    # 组合图片路径和对话内容
    content = abs_path + json.dumps(item['conversations'], sort_keys=True)
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def sanitize():
    if not os.path.exists(SOURCE_JSON):
        print(f"❌ 找不到源文件: {SOURCE_JSON}")
        return

    # 1. 建立物理索引
    file_index = build_file_index(SEARCH_ROOTS)

    print(f"📖 正在读取数据: {SOURCE_JSON}")
    with open(SOURCE_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final_data = []
    seen_hashes = set()
    
    stats = {
        "total": len(data),
        "missing_img": 0,
        "duplicate": 0,
        "valid": 0
    }

    print("🚿 开始执行‘脱水’清洗与去重...")
    for item in tqdm(data):
        if 'image' not in item: continue
        rel_path = item['image']
        abs_path = None

        # --- 第一步：路径补全与物理校验 ---
        # 尝试直接拼接
        for root in SEARCH_ROOTS:
            test_path = os.path.join(root, rel_path)
            if os.path.exists(test_path):
                abs_path = test_path
                break
        
        # 尝试索引匹配 (容错层级差异)
        if not abs_path:
            fname = os.path.basename(rel_path)
            parent = os.path.basename(os.path.dirname(rel_path))
            key = f"{parent}/{fname}"
            abs_path = file_index.get(key) or file_index.get(fname)

        if not abs_path or not os.path.exists(abs_path):
            stats["missing_img"] += 1
            continue  # 找不到图，直接丢弃

        # 更新为绝对路径
        item['image'] = abs_path

        # --- 第二步：内容去重 ---
        sample_hash = get_sample_hash(item, abs_path)
        if sample_hash in seen_hashes:
            stats["duplicate"] += 1
            continue  # 重复样本，直接丢弃
        
        seen_hashes.add(sample_hash)
        final_data.append(item)
        stats["valid"] += 1

    # 保存清洗后的结果
    print(f"💾 正在保存清洗后的数据至: {CLEAN_JSON}")
    with open(CLEAN_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    # --- 最终报告 ---
    print("\n" + "="*50)
    print(f"📊 数据清洗最终报告：")
    print(f"📥 原始样本总数: {stats['total']:,}")
    print(f"❌ 物理缺失(已删除): {stats['missing_img']:,}")
    print(f"👯 内容重复(已删除): {stats['duplicate']:,}")
    print(f"✅ 最终可用样本: {stats['valid']:,}")
    print(f"📉 整体压缩率: {((stats['total']-stats['valid'])/stats['total'])*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    sanitize()