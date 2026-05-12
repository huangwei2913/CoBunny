import json
import random
import os

def sample_for_test():
    # 输入：您刚才清洗完的全量干净数据
    input_file = "/mnt/CoBunny/dataassert/final_stage1_v365_all_cleaned.json"
    # 输出：仅用于快速测试流程的 4000 条小样
    output_file = "/mnt/CoBunny/dataassert/test_sample_4000.json"
    
    sample_size = 4000

    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件: {input_file}")
        return

    print(f"📖 正在加载全量数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_count = len(data)
    print(f"📊 当前全量数据总计: {total_count:,} 条")

    # 安全检查：如果总数还不到 4000，就直接全量保存
    if total_count <= sample_size:
        print(f"⚠️ 数据总量不足 {sample_size}，将使用全量数据进行测试。")
        final_sample = data
    else:
        print(f"🎲 正在随机抽取 {sample_size} 个样本...")
        # 使用随机种子确保如果重复运行，结果是一致的（方便 Debug）
        random.seed(42)
        final_sample = random.sample(data, sample_size)

    # 保存抽样结果
    print(f"📝 正在写入测试文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_sample, f, ensure_ascii=False, indent=2)

    print("\n" + "="*40)
    print(f"✅ 抽样完成！测试集：{len(final_sample)} 条")
    print(f"🚀 建议测试指令：")
    print(f"   --data_path {output_file}")
    print("="*40)

if __name__ == "__main__":
    sample_for_test()