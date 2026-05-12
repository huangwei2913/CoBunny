import json
from tqdm import tqdm

file_path = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_abs_fixed.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

bad_samples = []

for i, item in enumerate(tqdm(data)):
    # 1. 统计 JSON 里 image 字段给出的图片数量
    img_list = item.get('image', [])
    if isinstance(img_list, str):
        img_count = 1 if img_list else 0
    else:
        img_count = len(img_list)
    
    # 2. 统计文本里 <image> 标签出现的总次数
    text_content = "".join([conv['value'] for conv in item['conversations']])
    tag_count = text_content.count("<image>")
    
    # 3. 逻辑比对：如果不相等，就是坏样本
    if img_count != tag_count:
        bad_samples.append({
            "index": i,
            "img_count": img_count,
            "tag_count": tag_count,
            "id": item.get('id', 'unknown')
        })

print(f"\n🚨 审计完成！共发现 {len(bad_samples)} 条对齐不一致的样本。")
if bad_samples:
    print("样例预览:", bad_samples[:3])