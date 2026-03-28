import json
import random
import os

# === 配置区 ===
INPUT_FILE = '/data/MAmmoTH-VL-Instruct-12M/mammoth_si_10M.json'
OUTPUT_FILE = '/mnt/CoBunny/dataassert/mammoth_500k_pilot.json'
SAMPLE_COUNT = 500000 
SEED = 42

def main():
    random.seed(SEED)
    
    # 1. 直接加载全量数据 (10M条数据，内存占用可能在 20GB-40GB 左右，请确保服务器内存充足)
    print(f"📖 正在加载全量 JSON 数据 (这需要几分钟，请耐心等待)...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except MemoryError:
        print("❌ 内存不足！10M 数据太大，无法一次性读入。建议增加 Swap 或联系管理员。")
        return

    total_items = len(full_data)
    print(f"✅ 成功加载！总共有 {total_items:,} 条样本。")

    # 2. 随机抽样
    print(f"🎲 正在随机抽取 {SAMPLE_COUNT:,} 条样本...")
    if total_items < SAMPLE_COUNT:
        print("⚠️ 警告：总样本数少于要求的抽样数，将返回全部数据。")
        sampled_data = full_data
    else:
        sampled_data = random.sample(full_data, SAMPLE_COUNT)

    # 3. 释放原始大列表内存 (可选)
    del full_data

    # 4. 写入新文件
    print(f"💾 正在写入新文件: {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # 保持缩进为 2，格式和你给出的例子一模一样
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

    print(f"✨ 抽样完成！50万条标准 JSON 样本已就绪。")

if __name__ == "__main__":
    main()