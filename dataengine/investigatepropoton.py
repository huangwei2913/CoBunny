import json
from tqdm import tqdm

INPUT_FILE = "/mnt/conda_data/Bunny-v1.1-data/pretrain/PRIMA_pretrain_merged_en_only.json"

def check_text_only():
    print(f"📦 正在加载洗好的数据...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    text_only_count = 0
    multimodal_count = 0
    
    print(f"🧐 正在扫描 205 万样本中的纯文本比例...")
    for item in tqdm(data):
        # 条件 1: 检查 key 是否存在
        has_image_key = 'image' in item and item['image']
        
        # 条件 2: 检查文本里有没有占位符
        has_image_tag = False
        if 'conversations' in item:
            for conv in item['conversations']:
                if '<image>' in conv['value']:
                    has_image_tag = True
                    break
        
        # 只有两个都没有，才算纯文本
        if not has_image_key and not has_image_tag:
            text_only_count += 1
        else:
            multimodal_count += 1

    ratio = (text_only_count / total) * 100
    print(f"\n✅ 统计完成！")
    print(f"📊 总样本数: {total}")
    print(f"💬 纯文本样本: {text_only_count}")
    print(f"🖼️ 多模态样本: {multimodal_count}")
    print(f"📈 纯文本占比: {ratio:.2f}%")

if __name__ == "__main__":
    check_text_only()

    
