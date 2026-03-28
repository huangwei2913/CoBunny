import json
import os

# --- 配置路径 ---
input_file = "/data/Echo-4o-Image/Instruction-Following-Image/Instruction-Following-Image.jsonl"
output_file = "/mnt/CoBunny/dataassert/echo4o_hard_vqa_refined.json"
# 注意：Bunny训练时读取图片的根路径，通常建议设置为绝对路径
img_root = "/data/Echo-4o-Image/Instruction-Following-Image/images"

# --- 转换逻辑 ---
converted_data = []
hard_count = 0
total_count = 0

print("🚀 开始扫描 Echo-4o 矿石，提取 Hard 级精华...")

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        total_count += 1
        item = json.loads(line)
        
        # 核心策略：只选 Hard 样本，且任务类型是 text-to-image 的指令转换
        if item.get("type") == "hard":
            instruction = item.get("instruction", "")
            img_name = os.path.basename(item.get("output_image", ""))
            img_path = os.path.join("Echo-4o-Image/Instruction-Following-Image/images", img_name)
            
            # 构造高质量的 VQA 对话
            # Question: 引导模型进行精确计数和描述
            # Answer: 使用富有逻辑的句式，打破复读
            entry = {
                "id": f"echo4o_hard_{hard_count}",
                "image": img_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nObserve this image carefully. What objects are present, and what is their exact count?"
                    },
                    {
                        "from": "gpt",
                        "value": f"The image features {instruction}. The composition clearly highlights these elements, ensuring a precise representation of the requested subjects."
                    }
                ]
            }
            converted_data.append(entry)
            hard_count += 1

# 写入文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print("-" * 50)
print(f"✅ 处理完成！")
print(f"📊 扫描总数: {total_count}")
print(f"💎 提取 Hard 样本: {hard_count}")
print(f"📂 输出路径: {output_file}")
print("-" * 50)