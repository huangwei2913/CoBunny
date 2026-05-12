import json
import re
import os

# --- 配置路径 ---
DATA_DIR = '/mnt/CoBunny/dataassert'
# 仅保留 OCR 专项数据，剔除 VQA，确保任务单一性
FILES_TO_MERGE = ['ocr_1_ready.json', 'ocr_3_ready.json']
OUTPUT_FILE = os.path.join(DATA_DIR, 'ocr_en_only_specialized.json')

def has_chinese(text):
    """使用正则匹配中文字符范围"""
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def fix_newlines(text):
    """
    针对格式问题进行修复：
    将 <image>\n\n 替换为 <image>\n，确保符合 Bunny 训练的标准 Input Template
    """
    # 替换 <image> 后面可能跟着的多个换行符为单个换行
    text = re.sub(r'<image>\n+', '<image>\n', text)
    # 同时清理掉字符串首尾多余的空白
    return text.strip()

def clean_and_merge_ocr_only():
    merged_data = []
    total_scanned = 0
    removed_chinese_count = 0
    fixed_format_count = 0

    print(f"🚀 开始 OCR 专项数据扫描、清理并修复格式...")

    for file_name in FILES_TO_MERGE:
        file_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过不存在的文件: {file_name}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ 文件格式错误，无法解析: {file_name}")
                continue

            total_scanned += len(data)
            
            for item in data:
                conv_text_full = ""
                is_fixed = False
                
                # 遍历对话内容进行格式修复
                for conv in item.get('conversations', []):
                    old_val = conv.get('value', '')
                    new_val = fix_newlines(old_val)
                    
                    if old_val != new_val:
                        conv['value'] = new_val
                        is_fixed = True
                    
                    conv_text_full += new_val
                
                # 严格过滤逻辑：只要包含一个中文字符，直接剔除
                if has_chinese(conv_text_full):
                    removed_chinese_count += 1
                else:
                    if is_fixed:
                        fixed_format_count += 1
                    merged_data.append(item)
        
        print(f"   ✅ 已处理并修复 OCR 数据: {file_name}")

    # --- 保存最终结果 ---
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"\n--- OCR 专项数据审计报告 ---")
    print(f"📊 总扫描 OCR 样本数: {total_scanned}")
    print(f"🚫 剔除含中文字符样本: {removed_chinese_count}")
    print(f"🔧 修复换行符格式样本数: {fixed_format_count}")
    print(f"🏆 最终入库纯英 OCR 样本: {len(merged_data)}")
    print(f"💾 专项训练数据已就绪: {OUTPUT_FILE}")
    print(f"💡 建议：现在可以使用这个文件重新开始训练，Loss 下降速度应该会明显改善。")

if __name__ == "__main__":
    clean_and_merge_ocr_only()