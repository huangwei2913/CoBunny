import json
import os
import re
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# --- 配置区 ---
INPUT_FILE = "/mnt/conda_data/Bunny-v1.1-data/pretrain/PRIMA_pretrain_merged.json"
INPUT_FILE = "/mnt/CoBunny/dataassert/mixed_stage1_2m_stable.json"
OUTPUT_FILE = "/mnt/conda_data/Bunny-v1.1-data/pretrain/PRIMA_pretrain_merged_en_only.json"
IMAGE_ROOT = "/mnt/CoBunny/dataassert/mixed_stage1_2m_stable_nonenghlish.json" 
MAX_WORKERS = 32  # 内存大，线程直接拉满，建议 64-128

# 极速英文/符号检测正则（只要包含基本英文字符和数字，且不含中文即视为通过）
# 这样比 langdetect 快 100 倍，且能保留 OCR 关注的数字和符号
def is_english_relaxed(text):
    if not text: return True
    # 过滤掉含有中文字符的样本 (匹配中文范围)
    if re.search(r'[\u4e00-\u9fa5]', text):
        return False
    return True

def process_item(item):
    try:
        # 1. 提取并格式化对话
        if 'conversations' not in item: return None
        
        full_text = ""
        for conv in item['conversations']:
            val = conv['value']
            # 纠正 <image>\n\n 问题
            if "<image>" in val:
                new_val = val.replace("<image>", "").strip()
                conv['value'] = f"<image>\n{new_val}"
            if conv['from'] == 'human':
                full_text += val

        # 2. 快速过滤中文/乱码 (不再用慢速库)
        if not is_english_relaxed(full_text):
            return None

        # 3. 图片尺寸与坏图审计
        if 'image' in item:
            img_path = item['image']
            image_files = img_path if isinstance(img_path, list) else [img_path]
            for rel_path in image_files:
                full_path = rel_path if os.path.isabs(rel_path) else os.path.join(IMAGE_ROOT, rel_path)
                
                # 只要图片存在，就读一下 Header
                with Image.open(full_path) as img:
                    w, h = img.size
                    # 按照您的要求排除超大图和畸形图
                    if (w * h) > 80000000 or (w / h) > 15 or (h / w) > 15:
                        return None
        return item
    except Exception:
        return None

def clean_dataset():
    print(f"📦 正在加载原始数据...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_count = len(data)
    print(f"📊 原始样本数: {total_count}")

    print(f"⚡ 开启 {MAX_WORKERS} 线程极速处理...")
    clean_data = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 list 包装以配合 tqdm 实时显示
        results = list(tqdm(executor.map(process_item, data), total=total_count, desc="清理进度"))
        
    clean_data = [r for r in results if r is not None]

    print(f"\n✅ 清理完成! 保留: {len(clean_data)}, 剔除: {total_count - len(clean_data)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_dataset()