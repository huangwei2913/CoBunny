import json
import re
from tqdm import tqdm

FILE_PATH = "/mnt/CoBunny/dataassert/stage1_final_cleaned_fixed_.json"
FINAL_FILE = "/mnt/CoBunny/dataassert/stage1_ready_to_train_v1.json"

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("正在执行最后的 0.36% 格式强制对齐...")

for entry in tqdm(data):
    convs = entry["conversations"]
    val = convs[0]["value"]
    
    # 无论 <image> 在哪，都把它拎到最前面，并统一格式
    if "<image>" in val:
        # 1. 移除所有现有的 <image> 标签及其前后的空白
        clean_content = re.sub(r'\s*<image>\s*', '', val).strip()
        # 2. 强制插回开头，标准格式：<image>\n + 内容
        convs[0]["value"] = f"<image>\n{clean_content}"

with open(FINAL_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✨ 格式已 100% 对齐！最终文件：{FINAL_FILE}")