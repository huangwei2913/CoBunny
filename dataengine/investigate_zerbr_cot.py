import os
import pandas as pd
import json
from PIL import Image
import io
from tqdm import tqdm

def process_zebra_v2(base_dir, output_json, image_save_dir):
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    all_data = []
    # 锁定您指定的两个子集
    target_folders = [
        "2D Visual Reasoning - Visual Jigsaw",
        "2D Visual Reasoning - Visual Search"
    ]
    
    for folder in target_folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path): continue
        
        parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
        for p_file in parquet_files:
            df = pd.read_parquet(os.path.join(folder_path, p_file))
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {folder[:15]}"):
                # --- 1. 图像拼接逻辑 ---
                # 提取主图和第一张推理图
                p_img_bytes = row['problem_image_1']['bytes']
                r_img_bytes = row['reasoning_image_1']['bytes'] if row['reasoning_image_1'] is not None else p_img_bytes
                
                p_img = Image.open(io.BytesIO(p_img_bytes)).convert("RGB")
                r_img = Image.open(io.BytesIO(r_img_bytes)).convert("RGB")
                
                # 将两张图横向拼接 (对齐高度)
                target_height = 512
                p_img = p_img.resize((int(p_img.width * target_height / p_img.height), target_height))
                r_img = r_img.resize((int(r_img.width * target_height / r_img.height), target_height))
                
                combined_img = Image.new('RGB', (p_img.width + r_img.width, target_height))
                combined_img.paste(p_img, (0, 0))
                combined_img.paste(r_img, (p_img.width, 0))
                
                img_name = f"{folder.replace(' ', '_')}_{idx}.jpg"
                final_img_path = os.path.join(image_save_dir, img_name)
                combined_img.save(final_img_path, "JPEG", quality=90)

                # --- 2. 文本清洗 ---
                # 移除 Zebra 原生的图片占位符，替换为 Bunny 的 <image>
                clean_q = row['Question'].replace("<image_start>[problem_image_1]<image_end>", "").strip()
                # 移除推理链中的图片标记，保持纯文本思维流
                clean_trace = row['Text Reasoning Trace']
                for i in range(1, 5):
                    clean_trace = clean_trace.replace(f"<image_start>[problem_image_{i}]<image_end>", "")
                    clean_trace = clean_trace.replace(f"<image_start>[reasoning_image_{i}]<image_end>", "")

                # --- 3. 构造 Bunny 格式 ---
                all_data.append({
                    "id": f"zebra_{idx}",
                    "image": final_img_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{clean_q}"},
                        {"from": "gpt", "value": f"Thinking Process: {clean_trace}\nFinal Answer: {row['Final Answer']}"}
                    ]
                })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

# 执行处理
process_zebra_v2("/data/Zebra-CoT", "/data/Zebra-CoT/zebra_2d_sft.json", "/data/Zebra-CoT/zebra_combined_images")