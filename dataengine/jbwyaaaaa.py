import json
from bunny.constants import DEFAULT_IMAGE_TOKEN

def check_my_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    bad_samples = []
    print(f"开始检查数据：{json_path}，总计 {len(data)} 条")

    for i, entry in enumerate(data):
        # 1. 计算文本中 <image> 标签出现的总次数
        text_content = ""
        for conversation in entry.get('conversations', []):
            text_content += conversation.get('value', "")
        
        tag_count = text_content.count(DEFAULT_IMAGE_TOKEN)
        
        # 2. 计算 JSON 字段里实际提供的图片张数
        # 如果没有 'image' 字段，实际张数就是 0（对应您补的全0图逻辑，此时标签应为1）
        # 如果有 'image' 字段，检查它是字符串还是列表
        image_data = entry.get('image', None)
        if image_data is None:
            actual_image_count = 0 
        elif isinstance(image_data, list):
            actual_image_count = len(image_data)
        else:
            actual_image_count = 1

        # 3. 核心逻辑判断：
        # 对于 Bunny 这种架构，通常一个样本只能处理 1 张图
        # 如果标签数 > 1，或者标签数与图片数不匹配（除开您补全0图的特殊情况）
        if tag_count > 1:
            bad_samples.append({
                "line": i,
                "id": entry.get('id', 'unknown'),
                "reason": f"文本中有 {tag_count} 个 <image> 标签，Bunny 架构只支持 1 个",
                "content": text_content[:100] # 打印前100字方便定位
            })
        elif actual_image_count > 0 and tag_count == 0:
             bad_samples.append({
                "line": i,
                "id": entry.get('id', 'unknown'),
                "reason": "有图片数据但文本里没写 <image> 标签",
            })

    # 输出结果
    if bad_samples:
        print(f"\n❌ 发现 {len(bad_samples)} 个脏样本！")
        for bug in bad_samples:
            print(f"行号: {bug['line']} | ID: {bug['id']} | 原因: {bug['reason']}")
    else:
        print("\n✅ 数据集非常健康，没有标签冲突！")

if __name__ == "__main__":
    check_my_dataset("/mnt/CoBunny/dataassert/test_sample_4000.json")