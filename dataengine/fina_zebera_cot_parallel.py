import os
import pandas as pd
import json
import io
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- 配置区 ---
DATA_ROOT = "/data/Zebra-CoT"
SAVE_JSON = "/data/Zebra-CoT/zebra_sft_stage2.json"
SAVE_IMG = "/data/Zebra-CoT/zebra_processed_images"
NUM_WORKERS = 16  # 根据您的 CPU 核心数调整

def process_single_row(args):
    """
    单行数据处理函数（多进程核心）
    """
    row_data, folder_name, idx, image_save_dir = args
    try:
        # 1. 提取图像字节
        p_img_bytes = row_data['problem_image_1']['bytes']
        r_img_bytes = row_data['reasoning_image_1']['bytes'] if row_data['reasoning_image_1'] is not None else p_img_bytes
        
        # 使用 OpenCV 解码速度通常快于 PIL 直接 open
        nparr_p = np.frombuffer(p_img_bytes, np.uint8)
        nparr_r = np.frombuffer(r_img_bytes, np.uint8)
        p_img = cv2.imdecode(nparr_p, cv2.IMREAD_COLOR)
        r_img = cv2.imdecode(nparr_r, cv2.IMREAD_COLOR)

        if p_img is None or r_img is None: return None

        # 2. 图像缩放与拼接 (目标高度 512)
        target_h = 512
        h_p, w_p = p_img.shape[:2]
        h_r, w_r = r_img.shape[:2]
        
        new_w_p = int(w_p * target_h / h_p)
        new_w_r = int(w_r * target_h / h_r)
        
        p_img_res = cv2.resize(p_img, (new_w_p, target_h), interpolation=cv2.INTER_AREA)
        r_img_res = cv2.resize(r_img, (new_w_r, target_h), interpolation=cv2.INTER_AREA)
        
        # 横向拼接
        combined_img = np.hstack((p_img_res, r_img_res))
        
        # 保存图像
        img_filename = f"{folder_name.replace(' ', '_')}_{idx}.jpg"
        final_path = os.path.join(image_save_dir, img_filename)
        cv2.imwrite(final_path, combined_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 3. 文本清洗
        clean_q = row_data['Question']
        for i in range(1, 3): # 针对 Jigsaw/Search 重点清洗前两张
            clean_q = clean_q.replace(f"<image_start>[problem_image_{i}]<image_end>", "")
        
        clean_trace = row_data['Text Reasoning Trace']
        for i in range(1, 6): 
            clean_trace = clean_trace.replace(f"<image_start>[problem_image_{i}]<image_end>", "")
            clean_trace = clean_trace.replace(f"<image_start>[reasoning_image_{i}]<image_end>", "")

        return {
            "id": f"zebra_{folder_name[:3]}_{idx}",
            "image": final_path,
            "conversations": [
                {"from": "human", "value": f"<image>\n{clean_q.strip()}"},
                {"from": "gpt", "value": f"Thinking Process: {clean_trace.strip()}\nFinal Answer: {row_data['Final Answer']}"}
            ]
        }
    except Exception:
        return None

def main():
    if not os.path.exists(SAVE_IMG): os.makedirs(SAVE_IMG)
    all_results = []
    target_folders = ["2D Visual Reasoning - Visual Jigsaw", "2D Visual Reasoning - Visual Search"]

    for folder in target_folders:
        folder_path = os.path.join(DATA_ROOT, folder)
        if not os.path.exists(folder_path): continue
        
        parquet_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.parquet')])
        
        for p_file in parquet_files:
            print(f"正在加载文件: {p_file}")
            df = pd.read_parquet(os.path.join(folder_path, p_file))
            
            # 准备多进程参数
            tasks = [(row.to_dict(), folder, i, SAVE_IMG) for i, row in df.iterrows()]
            
            # 使用进程池并行处理
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                results = list(tqdm(executor.map(process_single_row, tasks), total=len(tasks), desc=f"并行处理 {p_file}"))
                all_results.extend([r for r in results if r is not None])

    # 写入 JSON
    with open(SAVE_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"全部完成，共提取 {len(all_results)} 条数据。")

if __name__ == "__main__":
    main()