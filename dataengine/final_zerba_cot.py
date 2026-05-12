import os
import pandas as pd
import json
from PIL import Image
import io
from tqdm import tqdm

def process_zebra_full(base_dir, output_json, image_save_dir):
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    all_data = []
    # 锁定您选定的两个最能打的子集
    target_configs = [
        {"name": "2D Visual Reasoning - Visual Jigsaw", "img_count": 1},
        {"name": "2D Visual Reasoning - Visual Search", "img_count": 1}
    ]
    
    for config in target_configs:
        folder_name = config["name"]
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"警告: 找不到目录 {folder_name}")
            continue
        
        parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
        for p_file in parquet_files:
            file_path = os.path.join(folder_path, p_file)
            df = pd.read_parquet(file_path)
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {folder_name[:15]}"):
                try:
                    # --- 1. 核心图像拼接逻辑 ---
                    p_img_bytes = row['problem_image_1']['bytes']
                    # 默认取第一张推理图，它是“撒盐”的关键
                    r_img_bytes = row['reasoning_image_1']['bytes'] if row['reasoning_image_1'] is not None else p_img_bytes
                    
                    p_img = Image.open(io.BytesIO(p_img_bytes)).convert("RGB")
                    r_img = Image.open(io.BytesIO(r_img_bytes)).convert("RGB")
                    
                    # 统一高度为 512，宽度按比例缩放，确保拼接后六子图采样不畸变
                    target_h = 512
                    p_w = int(p_img.width * target_h / p_img.height)
                    r_w = int(r_img.width * target_h / r_img.height)
                    
                    p_img_res = p_img.resize((p_w, target_h), Image.LANCZOS)
                    r_img_res = r_img.resize((r_w, target_h), Image.LANCZOS)
                    
                    # 横向拼接
                    combined_img = Image.new('RGB', (p_w + r_w, target_h))
                    combined_img.paste(p_img_res, (0, 0))
                    combined_img.paste(r_img_res, (p_w, 0))
                    
                    # 保存图片
                    img_filename = f"{folder_name.replace(' ', '_')}_{p_file.split('-')[1]}_{idx}.jpg"
                    final_path = os.path.join(image_save_dir, img_filename)
                    combined_img.save(final_path, "JPEG", quality=95)

                    # --- 2. 文本清洗逻辑 ---
                    # 清洗问题中的图片占位符
                    clean_q = row['Question']
                    for i in range(1, 5):
                        clean_q = clean_q.replace(f"<image_start>[problem_image_{i}]<image_end>", "")
                    clean_q = clean_q.strip()

                    # 清洗推理链中的所有图片占位符，只保留思维逻辑
                    clean_trace = row['Text Reasoning Trace']
                    for i in range(1, 25): # Zebra有些子集占位符很多，一次性清干净
                        clean_trace = clean_trace.replace(f"<image_start>[problem_image_{i}]<image_end>", "")
                        clean_trace = clean_trace.replace(f"<image_start>[reasoning_image_{i}]<image_end>", "")
                    
                    # --- 3. 构造 Bunny 训练格式 ---
                    all_data.append({
                        "id": f"zebra_{idx}_{hash(img_filename) % 100000}",
                        "image": final_path,
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{clean_q}"},
                            {"from": "gpt", "value": f"Thinking Process: {clean_trace.strip()}\nFinal Answer: {row['Final Answer']}"}
                        ]
                    })
                except Exception as e:
                    print(f"跳过损坏样本 {idx}: {e}")
                    continue

    # 保存结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 处理完成！共生成 {len(all_data)} 条 SFT 数据。")
    print(f"数据文件保存在: {output_json}")

if __name__ == "__main__":
    # 配置您的路径
    DATA_ROOT = "/data/Zebra-CoT"
    SAVE_JSON = "/data/Zebra-CoT/zebra_sft_stage2.json"
    SAVE_IMG = "/data/Zebra-CoT/zebra_processed_images"
    
    process_zebra_full(DATA_ROOT, SAVE_JSON, SAVE_IMG)