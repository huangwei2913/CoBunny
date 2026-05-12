import json
import os
from tqdm import tqdm

def strict_pairs_hook(pairs):
    """
    在 JSON 加载阶段就拦截重复键位。
    如果一个字典里出现了两次 'from' 或者重复的 'gpt' 逻辑，这里能直接察觉。
    """
    keys = [p[0] for p in pairs]
    # 如果在同一层级发现了重复的 key（比如两个 "from"），这数据就是脏的
    if len(keys) != len(set(keys)):
        return "__BAD_DATA_REPEATED_KEY__"
    return dict(pairs)

def is_legal_single_turn(item):
    """
    严格校验：
    1. 必须是单轮 (Human + GPT)
    2. 角色顺序必须对
    3. 没有任何被钩子标记的坏账
    """
    if item == "__BAD_DATA_REPEATED_KEY__":
        return False
        
    try:
        if not isinstance(item, dict) or 'conversations' not in item:
            return False
            
        convs = item.get('conversations', [])
        
        # 1. 严格数量检查：只能是 2 个元素
        if not isinstance(convs, list) or len(convs) != 2:
            return False
            
        # 2. 深度扫描：检查 conversations 内部是否存在重复 key 标记
        if "__BAD_DATA_REPEATED_KEY__" in str(convs):
            return False
            
        # 3. 角色与内容校验
        c0, c1 = convs[0], convs[1]
        if c0.get('from') != 'human' or c1.get('from') != 'gpt':
            return False
            
        if not str(c0.get('value', '')).strip() or not str(c1.get('value', '')).strip():
            return False
            
        return True
    except:
        return False

def rescue_clean_data_final():
    input_file = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_cleaned_ABS.json"
    output_file = "/mnt/CoBunny/dataassert/final_stage1_v365_all_cleaned.json"

    print(f"🚀 启动‘硬核级’坏账清除计划...")

    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件: {input_file}")
        return

    # 使用 object_pairs_hook 钩子加载 JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            print("📖 正在进行内存扫描与重复键检测...")
            data = json.load(f, object_pairs_hook=strict_pairs_hook)
        except Exception as e:
            print(f"❌ JSON 物理损坏，无法解析: {e}")
            return

    clean_list = []
    dropped_count = 0

    for item in tqdm(data, desc="深度扫描样本内容"):
        # 过滤掉被钩子标记的坏账和逻辑错误的单轮
        if not is_legal_single_turn(item):
            dropped_count += 1
            continue

        # 重新封装，确保输出的 JSON 物理结构绝对纯净
        clean_node = {
            "id": str(item['id']),
            "conversations": [
                {"from": "human", "value": item['conversations'][0]['value']},
                {"from": "gpt", "value": item['conversations'][1]['value']}
            ]
        }
        
        if 'image' in item and item['image']:
            clean_node["image"] = item['image']
            
        clean_list.append(clean_node)

    print(f"\n" + "="*50)
    print(f"📊 最终质检报告")
    print("-" * 50)
    print(f"✅ 纯净样本: {len(clean_list):,}")
    print(f"❌ 拦截坏账 (含重复Key/错位): {dropped_count:,}")
    print("="*50)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_list, f, ensure_ascii=False, indent=2)

    print(f"🚀 清理完成！干净的数据已存至: {output_file}")

if __name__ == "__main__":
    rescue_clean_data_final()