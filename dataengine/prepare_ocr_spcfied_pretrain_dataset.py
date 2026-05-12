import json
import os

def final_merge_for_pretrain():
    # --- 1. 路径配置 ---
    # 基础数据集（需要补全路径）
    base_dir = "/mnt/conda_data/Bunny-v1.1-data/pretrain"
    base_json_path = os.path.join(base_dir, "bunny_stage1_cleaned.json")
    base_image_prefix = "/mnt/conda_data/Bunny-v1.1-data/pretrain/images/"

    # OCR 数据集（路径已经是绝对地址，不需要处理）
    ocr_dir = "/mnt/CoBunny/dataassert"
    ocr_files = ["ocr_1_ready.json", "ocr_2_ready.json", "ocr_3_ready.json"]

    output_path = os.path.join(base_dir, "PRIMA_pretrain_merged.json")

    merged_data = []

    # --- 2. 处理逻辑：修正 <image> 位置 ---
    def fix_image_placeholder(item):
        """强制将 <image> 移到 Human 对话的最前面"""
        for conv in item['conversations']:
            if conv['from'] == 'human':
                val = conv['value']
                # 移除所有位置的 <image> 标签及其前后的换行
                clean_val = val.replace("<image>", "").strip()
                # 重新拼接到最前面
                conv['value'] = f"<image>\n{clean_val}"
        return item

    # --- 3. 处理 bunny_stage1 (补全 + 修正) ---
    if os.path.exists(base_json_path):
        print(f"正在处理基础数据 (补全路径至 {base_image_prefix})...")
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
            for item in base_data:
                # 补全绝对路径
                img_name = item['image']
                item['image'] = os.path.join(base_image_prefix, img_name)
                # 修正位置
                merged_data.append(fix_image_placeholder(item))
    else:
        print(f"❌ 找不到基础文件: {base_json_path}")

    # --- 4. 处理 OCR 数据 (只修正位置，路径绝对不动) ---
    for ocr_file in ocr_files:
        full_ocr_path = os.path.join(ocr_dir, ocr_file)
        if os.path.exists(full_ocr_path):
            print(f"正在融合 OCR 数据 (保持原路径): {ocr_file}...")
            with open(full_ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
                for item in ocr_data:
                    # 路径不做任何修改，直接处理占位符
                    merged_data.append(fix_image_placeholder(item))
        else:
            print(f"⚠️ 跳过不存在的 OCR 文件: {full_ocr_path}")

    # --- 5. 保存最终结果 ---
    print(f"✅ 处理完成！总计样本数: {len(merged_data)}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print(f"🚀 最终混合预训练文件已生成: {output_path}")

if __name__ == "__main__":
    final_merge_for_pretrain()