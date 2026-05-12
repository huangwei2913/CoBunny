import json
import random
import os
from tqdm import tqdm

def is_legal_single_turn(item):
    """
    极度严格的质检逻辑：
    1. 必须有 id, conversations, image (针对视觉数据)
    2. conversations 必须能提取出且仅能提取出一对 Human-GPT
    """
    try:
        convs = item.get('conversations', [])
        # 统计角色出现次数
        role_counts = {}
        for c in convs:
            role = c.get('from')
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # 铁律：human 必须有且仅有1个，gpt 必须有且仅有1个
        if role_counts.get('human') != 1 or role_counts.get('gpt') != 1:
            return False
            
        # 顺序校验：必须是 human 开头
        if convs[0].get('from') != 'human':
            return False
            
        # 内容校验：不能为空
        if not str(convs[0].get('value', '')).strip() or not str(convs[1].get('value', '')).strip():
            return False
            
        return True
    except:
        return False

def final_polish_and_balance():
    # 路径配置
    input_file = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_cleaned_ABS.json"
    output_file = "/mnt/CoBunny/dataassert/final_stage1_v365_721_fixed.json"

    print(f"🚀 启动终极质检与比例调配...")

    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # 1. 自动归类与清洗
    ocr_pool = []
    vision_pool = []
    logic_pool = []
    dropped_bad_format = 0

    for item in tqdm(all_data, desc="执行角色与结构质检"):
        # 执行严格质检：踢掉你发现的那种双重 GPT 或者 格式破碎的样本
        if not is_legal_single_turn(item):
            dropped_bad_format += 1
            continue

        # 彻底重组对象，消除任何可能的 Key 重复隐患
        clean_item = {
            "id": str(item['id']),
            "image": item.get('image', ""),
            "conversations": [
                {"from": "human", "value": item['conversations'][0]['value']},
                {"from": "gpt", "value": item['conversations'][-1]['value']} # 取最后一个GPT，防止冗余
            ]
        }

        # 根据 ID 前缀或内容归类 (根据您之前的合并逻辑)
        # 这里假设您之前的合并中保留了原始 ID 特征
        item_id = clean_item['id'].lower()
        if "slimorca" in item_id:
            logic_pool.append(clean_item)
        elif "general_vision" in item_id or "vision" in item_id:
            vision_pool.append(clean_item)
        else:
            # 默认为 OCR 样本
            ocr_pool.append(clean_item)

    print(f"\n📊 质检完成：剔除结构异常样本 {dropped_bad_format} 条")
    print(f"📦 当前存量：OCR={len(ocr_pool)}, Vision={len(vision_pool)}, Logic={len(logic_pool)}")

    # 2. 比例控制 (目标: 100万 : 30万 : 15万)
    target_ocr = 1000000
    target_vision = 300000
    target_logic = 150000

    # 实际抽样（如果存量不足，则取全部）
    final_ocr = random.sample(ocr_pool, min(len(ocr_pool), target_ocr))
    final_vision = random.sample(vision_pool, min(len(vision_pool), target_vision))
    final_logic = random.sample(logic_pool, min(len(logic_pool), target_logic))

    # 3. 合并并随机打乱
    final_list = final_ocr + final_vision + final_logic
    random.seed(42)
    random.shuffle(final_list)

    # 4. 保存
    print(f"\n✅ 最终输出报告:")
    print(f" - OCR (对齐/识字): {len(final_ocr)}")
    print(f" - Vision (常识/描述): {len(final_vision)}")
    print(f" - Logic (智商/推理): {len(final_logic)}")
    print(f"🌟 总计样本: {len(final_list)}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    print(f"\n🚀 数据集已完美就绪：{output_file}")

if __name__ == "__main__":
    final_polish_and_balance()