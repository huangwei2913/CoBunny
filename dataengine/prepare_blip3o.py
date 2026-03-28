import os
import json
from tqdm import tqdm

def generate_sft_json(root_path, output_name):
    # 这里指向你刚刚创建并解压好的目录
    base_dir = os.path.join(root_path, "images_curated")
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    final_json = []

    print(f"Found categories: {categories}")

    for cat in categories:
        cat_path = os.path.join(base_dir, cat)
        # 获取所有图片文件
        imgs = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Converting {cat}: {len(imgs)} images found.")
        
        for img in tqdm(imgs):
            # 这里的逻辑：1000.jpg -> 1000.txt
            txt_file = os.path.splitext(img)[0] + ".txt"
            txt_path = os.path.join(cat_path, txt_file)
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                
                # 针对顶刊实验设计的智能指令分配
                if 'text' in cat:
                    q = "Identify and transcribe all the text present in this image accurately."
                elif 'gestures' in cat:
                    q = "What is the specific human action or gesture shown here?"
                else:
                    q = "Provide an extremely detailed and high-fidelity description of this scene."

                final_json.append({
                    "id": f"blip3o_{cat}_{os.path.splitext(img)[0]}",
                    "image": f"BLIP3o-60k/images_curated/{cat}/{img}", # 确保路径与你训练代码的 data_root 匹配
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{q}"},
                        {"from": "gpt", "value": caption}
                    ]
                })

    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"\n[Success] Created {output_name} with {len(final_json)} samples!")

if __name__ == "__main__":
    generate_sft_json('/data/BLIP3o-60k', '/mnt/CoBunny/dataassert/blip3o_final_sft.json')