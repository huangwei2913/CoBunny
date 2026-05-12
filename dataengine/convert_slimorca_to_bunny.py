import json

def convert():
    input_file = "/data/SlimOrca/oo-labeled_correct.gpt4.sharegpt.jsonl"
    output_file = "/data/SlimOrca/SlimOrca_for_Bunny_Stage1.json"
    
    # --- 新增：长度阈值设置 ---
    # 建议设为 2500 字符。如果整条对话（问+答）超过这个值，就直接丢弃。
    MAX_CHAR_LIMIT = 2500 
    
    converted_data = []
    skipped_count = 0 # 统计跳过了多少条
    
    print(f"开始转换 SlimOrca 数据 (过滤阈值: {MAX_CHAR_LIMIT} 字符)...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                raw_item = json.loads(line)
                orig_convs = raw_item['conversations']
                
                system_msg = ""
                human_msg = ""
                gpt_msg = ""
                
                for c in orig_convs:
                    if c['from'] == 'system':
                        system_msg = c['value']
                    elif c['from'] == 'human':
                        human_msg = c['value']
                    elif c['from'] == 'gpt':
                        gpt_msg = c['value']
                
                # 组合成 Bunny 格式的内容
                combined_human = f"General Chat: {system_msg}\nQuestion: {human_msg}".strip()
                
                # --- 新增：超长过滤逻辑 ---
                total_length = len(combined_human) + len(gpt_msg)
                if total_length > MAX_CHAR_LIMIT:
                    skipped_count += 1
                    continue # 跳过当前循环，不加入 converted_data
                
                new_convs = [
                    {"from": "human", "value": combined_human},
                    {"from": "gpt", "value": gpt_msg}
                ]
                
                converted_data.append({
                    "id": f"slimorca_{len(converted_data)}",
                    "image": "", 
                    "conversations": new_convs
                })
                
            except Exception as e:
                continue

    print(f"转换完成！")
    print(f"✅ 保留样本数: {len(converted_data)}")
    print(f"❌ 因太长被剔除: {skipped_count}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"文件已保存至: {output_file}")

if __name__ == "__main__":
    convert()