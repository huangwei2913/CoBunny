import json
from tqdm import tqdm

def check_multiple_images():
    # 填入您要排查的文件（建议先用 test_sample_4000.json 测一下，也可以直接上全量文件）
    file_path = "/mnt/CoBunny/dataassert/test_sample_4000.json" 
    
    print(f"🔍 开始地毯式搜索多个 <image> 标签...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    multi_image_samples = []
    
    for item in tqdm(data, desc="扫描中"):
        # 将该样本中所有的对话内容拼接成一段长文本
        total_text = ""
        for conv in item.get('conversations', []):
            total_text += str(conv.get('value', ''))
            
        # 纯物理级字符串统计：数一数到底有几个 <image>
        img_count = total_text.count("<image>")
        
        if img_count > 1:
            multi_image_samples.append(item['id'])
            print(f"\n⚠️ 抓到现行！ID: {item.get('id', 'N/A')} | 文本中包含的 <image> 数量: {img_count}")
            # 打印出具体的文本片段看看它藏在哪
            for conv in item.get('conversations', []):
                if "<image>" in conv.get('value', ''):
                    snippet = conv['value'].replace('\n', ' ')[:100]
                    print(f"  [{conv['from']}]: {snippet}...")
    
    print("\n" + "="*50)
    print(f"📊 扫描结果总结")
    print("-" * 50)
    print(f"总计扫描样本: {len(data)}")
    print(f"发现多个 <image> 的嫌疑样本数: {len(multi_image_samples)}")
    print("="*50)

    if len(multi_image_samples) == 0:
        print("\n🎉 恭喜！数据层面的嫌疑被彻底洗清了！")
        print("👉 文本里绝对没有混入多余的占位符。这就意味着，1 vs 2 的报错，100% 是模型底层代码（比如双塔的拼接逻辑）在作怪！")
    else:
        print("\n🚨 警报！确实是数据清洗的锅！赶紧看看上面打印出来的样本，到底是怎么混进去多个标签的！")

if __name__ == "__main__":
    check_multiple_images()