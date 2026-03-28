import json
import re

# 指向你当前报错的数据集路径
data_path = "/mnt/CoBunny/dataassert/v365_stage3_mcp_final_clean_fixed.json"

def analyze_mismatch():
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🚀 开始扫描数据集：{data_path}")
    print("-" * 60)
    
    mismatch_samples = []

    for idx, item in enumerate(data):
        # 1. 统计图片数量 (Bunny 逻辑：通常是 item['image'] 存在则为1，或者 len(item['images']))
        # 根据你的报错，视觉塔只给了 1 组特征，说明 item 里可能只有一个图片 key
        actual_image_count = 0
        if 'image' in item and item['image']:
            actual_image_count = 1
        elif 'images' in item:
            actual_image_count = len(item['images'])

        # 2. 统计文本中 <image> 出现的总次数 (涵盖所有对话轮次)
        total_text = ""
        if 'conversations' in item:
            for conv in item['conversations']:
                total_text += conv['value']
        
        # 使用正则精确匹配 <image> 标签
        placeholder_count = len(re.findall(r'<image>', total_text))

        # 3. 核心比对
        if placeholder_count != actual_image_count:
            sample_info = {
                "index": idx,
                "id": item.get('id', 'N/A'),
                "placeholders": placeholder_count,
                "actual_images": actual_image_count,
                "text_snippet": total_text[:200] + "..." # 只打印前200字看个大概
            }
            mismatch_samples.append(sample_info)
            
            print(f"‼️ 发现异常 [Index: {idx}] [ID: {sample_info['id']}]")
            print(f"   - 文本占位符 <image> 数量: {placeholder_count}")
            print(f"   - 实际提供图片数量: {actual_image_count}")
            print(f"   - 预览内容: {sample_info['text_snippet']}")
            print("-" * 30)

    print(f"\n✅ 扫描结束！共发现 {len(mismatch_samples)} 个不匹配样本。")
    return mismatch_samples

if __name__ == "__main__":
    anomalies = analyze_mismatch()