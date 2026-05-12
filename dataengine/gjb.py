import json
import random
import os

def merge_and_sample_datasets():
    # 1. 定义路径
    ocr_data_path = "/mnt/CoBunny/dataassert/final_stage1_v365_all_cleaned.json"
    prima_data_path = "/mnt/conda_data/Bunny-v1.1-data/pretrain/PRIMA_pretrain_merged_en_only.json"
    output_path = "/mnt/CoBunny/dataassert/mixed_stage1_2m_stable.json"
    
    sample_target = 2000000  # 目标样本数
    
    print("正在加载数据集...")
    
    # 2. 加载数据
    with open(ocr_data_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    print(f"加载 OCR 数据: {len(ocr_data)} 条")
    
    with open(prima_data_path, 'r', encoding='utf-8') as f:
        prima_data = json.load(f)
    print(f"加载 PRIMA 数据: {len(prima_data)} 条")
    
    # 3. 全局去重 (依据 image 字段)
    # 使用字典，以 image 路径为键，后出现的会覆盖前面的，保证路径唯一
    combined_dict = {}
    
    print("正在进行全局去重...")
    # 先放 OCR 数据，保证如果路径重复，优先保留 OCR（或者根据需求调整顺序）
    for item in ocr_data:
        if 'image' in item:
            combined_dict[item['image']] = item
            
    for item in prima_data:
        if 'image' in item:
            # 如果 PRIMA 中有重复路径，这里不会重复添加
            if item['image'] not in combined_dict:
                combined_dict[item['image']] = item

    all_unique_data = list(combined_dict.values())
    total_unique = len(all_unique_data)
    print(f"去重后总样本数: {total_unique}")

    # 4. 随机采样
    if total_unique <= sample_target:
        print(f"警告：总样本数不足 {sample_target}，将保留全部 {total_unique} 条数据。")
        final_data = all_unique_data
    else:
        print(f"正在随机抽取 {sample_target} 条样本...")
        final_data = random.sample(all_unique_data, sample_target)
    
    # 5. 打乱顺序 (Shuffle)
    # 这一步非常重要！确保训练时 OCR 和通用数据交替出现，平滑梯度
    random.shuffle(final_data)
    
    # 6. 保存结果
    print(f"正在保存至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print("完成！数据已准备就绪。")

if __name__ == "__main__":
    merge_and_sample_datasets()