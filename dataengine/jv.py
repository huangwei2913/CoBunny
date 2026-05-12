import json
import os
from tqdm import tqdm

def refine_slimorca():
    input_file = "/data/SlimOrca/SlimOrca_for_Bunny_Stage1_Refined.json"
    output_file = "/mnt/CoBunny/dataassert/SlimOrca_Stage1_Fixed_With_Images.json"
    black_image_path = "/mnt/CoBunny/dataassert/placeholders/black_padding.jpg"

    print(f"🚀 开始处理 SlimOrca 数据集...")

    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"❌ JSON 解析失败，文件可能已损坏: {e}")
            return

    refined_list = []
    
    for item in tqdm(data, desc="补全图片与替换占位符"):
        # 只要单轮对话，且结构必须完整
        if 'conversations' not in item or len(item['conversations']) < 2:
            continue

        # 1. 提取原始对话
        human_val = str(item['conversations'][0].get('value', ''))
        gpt_val = str(item['conversations'][1].get('value', ''))

        # 2. 执行替换逻辑：将 "General Chat: " 替换为 "<image>"
        # 如果文本开头不是这个，我们也强制在最前面加上 <image>，确保模型对齐
        if "General Chat:" in human_val:
            # 使用正则或直接替换，这里直接 replace 掉所有的前缀
            new_human_val = human_val.replace("General Chat:", "<image>").strip()
        else:
            new_human_val = "<image>\n" + human_val

        # 3. 构造干净的新样本 (彻底解决 Key 碎裂和多轮问题)
        new_item = {
            "id": item.get("id", "slimorca_unk"),
            "image": black_image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": new_human_val
                },
                {
                    "from": "gpt",
                    "value": gpt_val
                }
            ]
        }
        
        refined_list.append(new_item)

    # 保存结果
    print(f"📝 正在保存... 共计 {len(refined_list)} 条样本")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(refined_list, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 任务完成！输出路径: {output_file}")

if __name__ == "__main__":
    refine_slimorca()