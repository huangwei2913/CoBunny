import json
import os
import random
from tqdm import tqdm

def reconstruct_fashion_anti_lazy():
    source_dir = "/data/fashion/FashionRec/basic_recommendation/train"
    output_json = "/mnt/CoBunny/dataassert/fashion_full_scan_alignment.json"

    
    # --- 1. 提问池：防止模型对问题产生抗药性 ---
    question_pool = [
        "What fashion items can you see in this picture?",
        "Identify and describe all the clothing pieces shown here.",
        "Give me a detailed list of all the garments in this image.",
        "Could you break down the fashion items visible in this photo?",
        "What is featured in this fashion display? Provide details for everything."
    ]

    reconstructed_data = []
    print(f"🚀 开始构建【防偷懒】版时尚数据...")
    
    files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    
    for filename in tqdm(files):
        idx = filename.replace('.json', '')
        json_path = os.path.join(source_dir, filename)
        image_path = os.path.join(source_dir, f"{idx}.jpg")
        
        if not os.path.exists(image_path): continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = [it.get('description', '').strip() for it in data.get('partial_outfit', []) if it.get('description')]
                
                if not items: continue

                # --- 2. 随机策略：打乱描述顺序 ---
                # 防止模型建立“位置 -> 固定顺序”的偷懒逻辑
                random.shuffle(items)

                # --- 3. 动态开头：防止句式固化 ---
                openings = [
                    f"I can identify {len(items)} items here: ",
                    f"The image showcases {len(items)} distinct fashion pieces, including ",
                    f"There are {len(items)} garments visible: ",
                    "The collection consists of "
                ]
                
                response_text = random.choice(openings)
                response_text += ", ".join([f"a {desc}" for desc in items]) + "."

                # --- 4. 随机选提问 ---
                user_question = random.choice(question_pool)

                reconstructed_data.append({
                    "id": f"fashion_anti_lazy_{idx}",
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{user_question}"},
                        {"from": "gpt", "value": response_text}
                    ]
                })

        except Exception:
            continue

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(reconstructed_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！成功构建 {len(reconstructed_data)} 条【防偷懒】强对齐数据。")

if __name__ == "__main__":
    reconstruct_fashion_anti_lazy()