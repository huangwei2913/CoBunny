import json
import os
import random
import glob
from tqdm import tqdm

# ================= 配置区 =================
TOTAL_TARGET_SAMPLES = 220000 

# 科学配比：OCR 占比 30%，Pilot 占比 25%，通用 15%，VG 15%，业务 10%，逻辑 5%
DATA_RATIOS = {
    "ShareGPT4V_General": 0.15, 
    "MAmmoTH_Pilot_500k": 0.25, 
    "MAmmoTH_Ready_All": 0.30,  # 核心抗降智全集
    "VG_HighRes": 0.15,         
    "Fashion_PhD": 0.10,        
    "Logic_Math": 0.05          
}

# --- 请根据你的 ll 结果再次确认路径 ---
FILE_MAP = {
    "ShareGPT4V_General": "/data/ShareGPT4V/sharegpt4v_matched_772k_fixed.json",
    "MAmmoTH_Pilot_500k": "/data/MAmmoTH-VL-Instruct-12M/mammoth_500k_pilot.json",
    "MAmmoTH_Ready_Dir": "/data/MAmmoTH-VL-Instruct-12M/nvidiadataset/merged_finetune_data/", 
    "VG_HighRes": "/data/ShareGPT4V/sharegpt4v_vg_only_clean.json",
    "Logic_Math": "/data/MathVista/mathvista_for_train.json",
    "Fashion_PhD": "/data/fashion/FashionRec/fashion_visual_alignment_gold.json" # 请确保它在 /data/ShareGPT4V 下
}

def clean_conversations(conversations):
    new_convs = []
    img_tag_inserted = False
    for conv in conversations:
        role, content = conv['from'], conv['value']
        if role == 'human':
            clean_text = str(content).replace('<image>', '').strip()
            if not img_tag_inserted:
                new_value = f"<image>\n{clean_text}"
                img_tag_inserted = True
            else:
                new_value = clean_text
            new_convs.append({"from": "human", "value": new_value})
        else:
            new_convs.append({"from": "gpt", "value": str(content)})
    return new_convs

def main():
    final_pool = []
    sources = {}
    
    # 1. 强力预装载逻辑
    print("🔍 开始扫描并加载数据源...")
    
    # --- A. 特别处理 MAmmoTH 目录 ---
    ready_dir = FILE_MAP["MAmmoTH_Ready_Dir"]
    ready_files = glob.glob(os.path.join(ready_dir, "*.json"))
    if ready_files:
        print(f"📂 发现 {len(ready_files)} 个 OCR 子集，正在合并装载...")
        combined_ready = []
        for f in ready_files:
            with open(f, 'r', encoding='utf-8') as jf:
                combined_ready.extend(json.load(jf))
        sources["MAmmoTH_Ready_All"] = combined_ready
    else:
        print(f"❌ 错误：在 {ready_dir} 没看到 JSON！请检查路径拼写。")

    # --- B. 处理其他单文件 ---
    for key in ["ShareGPT4V_General", "MAmmoTH_Pilot_500k", "VG_HighRes", "Logic_Math", "Fashion_PhD"]:
        path = FILE_MAP[key]
        if os.path.exists(path):
            print(f"📖 正在装载 {key}: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.jsonl'):
                    sources[key] = [json.loads(line) for line in f]
                else:
                    sources[key] = json.load(f)
        else:
            print(f"❌ 找不到文件: {path}")

    # 2. 配比混合
    for label, ratio in DATA_RATIOS.items():
        if label not in sources or not sources[label]:
            print(f"⚠️ 跳过 {label}，因为没有装载到数据。")
            continue
        
        target_count = int(TOTAL_TARGET_SAMPLES * ratio)
        current_source = sources[label]
        
        # 采样/上采样
        if len(current_source) < target_count:
            sampled = (current_source * (target_count // len(current_source) + 1))[:target_count]
        else:
            sampled = random.sample(current_source, target_count)
            
        valid_count = 0
        for item in tqdm(sampled, desc=f"   对齐 {label}"):
            img_path = item.get('image') or item.get('img')
            if not img_path: continue
            
            # 路径修正逻辑 (适配 1B 模型的物理存储)
            if not os.path.isabs(img_path):
                # 尝试多个可能的根目录
                for root in ["/data/ShareGPT4V", "/data/MAmmoTH-VL-Instruct-12M"]:
                    test_p = os.path.join(root, img_path)
                    if os.path.exists(test_p):
                        img_path = test_p
                        break

            item['image'] = img_path
            if not os.path.exists(item['image']): continue
            
            item['conversations'] = clean_conversations(item['conversations'])
            final_pool.append(item)
            valid_count += 1
        print(f"   ✅ {label} 有效注入: {valid_count}")

    # 3. 最终 Shuffle 并保存
    print(f"🎲 全量 Shuffle (最终规模: {len(final_pool)})...")
    random.shuffle(final_pool)
    
    # 存到你有权限的目录
    output_name = "/data/ShareGPT4V/Bunny_TPAMI_Full_Mega_Mix_v3.json"
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(final_pool, f, ensure_ascii=False, indent=2)
    
    print(f"\n🏆 【TPAMI 级】全量数据集准备完毕！")
    print(f"📍 最终文件: {output_name}")
    print(f"📈 最终总数: {len(final_pool)} 条")

if __name__ == "__main__":
    main()