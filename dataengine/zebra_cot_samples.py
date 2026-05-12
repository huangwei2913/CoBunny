import os
import pandas as pd
import json
from PIL import Image
import io

def extract_and_preview(base_dir):
    # 锁定目标子集
    targets = {
        "Jigsaw": "2D Visual Reasoning - Visual Jigsaw",
        "Search": "2D Visual Reasoning - Visual Search"
    }
    
    for key, folder_name in targets.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"跳过: 找不到目录 {folder_name}")
            continue

        # 获取该目录下第一个 parquet 文件
        files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
        if not files: continue
        
        # 读取第一行数据
        df = pd.read_parquet(os.path.join(folder_path, files[0]))
        row = df.iloc[0] # 取第一个样本

        # --- 1. 处理并保存预览图片 ---
        # 提取主图 (problem_image_1)
        p_img_bytes = row['problem_image_1']['bytes']
        p_img = Image.open(io.BytesIO(p_img_bytes)).convert("RGB")
        
        # 提取第一张推理图 (reasoning_image_1)，如果不存在则用主图占位
        r_img = p_img
        if 'reasoning_image_1' in row and row['reasoning_image_1'] is not None:
            r_img_bytes = row['reasoning_image_1']['bytes']
            r_img = Image.open(io.BytesIO(r_img_bytes)).convert("RGB")

        # 将两张图横向拼接，方便您一眼对比
        target_h = 512
        p_img_res = p_img.resize((int(p_img.width * target_h / p_img.height), target_h))
        r_img_res = r_img.resize((int(r_img.width * target_h / r_img.height), target_h))
        
        combined = Image.new('RGB', (p_img_res.width + r_img_res.width, target_h))
        combined.paste(p_img_res, (0, 0))
        combined.paste(r_img_res, (p_img_res.width, 0))
        
        save_path = f"sample_{key.lower()}.jpg"
        combined.save(save_path)

        # --- 2. 打印清洗后的对话 ---
        clean_q = row['Question'].replace("<image_start>[problem_image_1]<image_end>", "").strip()
        print(f"\n{'='*50}")
        print(f"【子集类型】: {key}")
        print(f"【保存图片】: {os.path.abspath(save_path)}")
        print(f"【清洗后的问题】: {clean_q}")
        print(f"【思维链 (Trace)】: \n{row['Text Reasoning Trace']}")
        print(f"【最终答案】: {row['Final Answer']}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    # 请确保路径指向您的 Zebra-CoT 根目录
    extract_and_preview("/data/Zebra-CoT")