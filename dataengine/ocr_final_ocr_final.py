import json
import random
import os

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- 1. 定义数据路径与目标数量 ---
# 基础 OCR 数据全量保留 (73,803 条)
OCR_PATH = "/mnt/CoBunny/dataassert/ocr_en_only_stage3.json"

# 辅助数据集及其目标采样数
DATA_CONFIGS = [
    {"path": "/data/Zebra-CoT/zebra_sft_stage2.json", "target": 25000},       # 逻辑推理核心
    {"path": "/data/fashion/FashionRec/fashion_visual_alignment_gold.json", "target": 10000}, # 细节描述
    {"path": "/data/ShareGPT4V/sharegpt4v_vg_only_clean.json", "target": 8000}, # 空间定位
    {"path": "/mnt/CoBunny/dataassert/echo4o_hard_vqa_refined_abspath.json", "target": 4000}, # 复杂指令
    {"path": "/mnt/CoBunny/dataassert/blip3o_final_sft_abs.json", "target": 4000},  # 高精审美
    {"path": "/mnt/CoBunny/dataassert/SlimOrca_Stage1_Fixed_With_Images.json", "target": 15000}  # 常规会话
]

OUTPUT_PATH = "/mnt/CoBunny/dataassert/cobunny_stage2_final_mixed_ocr.json"

def main():
    final_data = []

    # 1. 加载主力 OCR 数据
    print(f"正在加载主力 OCR 数据: {OCR_PATH}")
    ocr_data = load_json(OCR_PATH)
    final_data.extend(ocr_data)
    print(f"✅ 已存入 OCR 样本: {len(ocr_data)} 条")

    # 2. 循环处理辅助数据
    for config in DATA_CONFIGS:
        path = config["path"]
        target = config["target"]
        
        if not os.path.exists(path):
            print(f"⚠️ 跳过: 找不到文件 {path}")
            continue
            
        data = load_json(path)
        source_name = os.path.basename(path)
        
        # 如果现有数据不足目标数，则全量取；否则随机采样
        if len(data) <= target:
            sampled = data
            print(f"📦 {source_name}: 全量获取 {len(sampled)} 条")
        else:
            sampled = random.sample(data, target)
            print(f"🎲 {source_name}: 随机采样 {len(sampled)} 条 (总数 {len(data)})")
            
        final_data.extend(sampled)

    # 3. 关键步骤：全局随机打乱
    # 必须打乱，防止训练时模型在某个 epoch 只盯着一种数据看
    random.shuffle(final_data)

    # 4. 保存最终训练文件
    print(f"正在整合并写入最终文件: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n🚀 混合完成！")
    print(f"最终总样本数: {len(final_data)}")
    print(f"OCR 占比: {(len(ocr_data)/len(final_data))*100:.2f}%")

if __name__ == "__main__":
    main()