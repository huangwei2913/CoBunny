import json
from tqdm import tqdm

# 你刚才生成的那个带绝对路径的 JSON
file_path = "/mnt/CoBunny/dataassert/final_stage1_mix_ocr_abs_fixed.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in tqdm(data):
    # 逻辑很简单：如果 image 字段是空的，或者根本不是个文件路径
    if 'image' in item and (not item['image'] or item['image'] == ""):
        # 1. 彻底拔掉这个 Key，这样代码就不会去尝试 Image.open() 了
        del item['image']
        
        # 2. 把文本里的 <image> 删掉，因为它现在是纯对话了
        for conv in item['conversations']:
            conv['value'] = conv['value'].replace("<image>", "").strip()

# 保存回去
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 修复完成！现在纯对话样本已经转为纯文本格式，不会再触发路径报错了。")