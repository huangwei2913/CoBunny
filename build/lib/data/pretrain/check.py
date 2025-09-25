import json

filename = "filtered_data.json"
with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()
    
try:
    data = json.loads(content)
    print("整体加载成功")
except json.JSONDecodeError as e:
    print(f"JSONDecodeError at line {e.lineno}, col {e.colno}, msg: {e.msg}")

# 如果是列表，可以尝试逐条加载
import ijson

filename = "filtered_data.json"
with open(filename, 'r', encoding='utf-8') as f:
    objects = ijson.items(f, 'item')  # 假设根是数组
    for i, obj in enumerate(objects):
        # 可以添加类型和字段校验，catch异常
        if i % 10000 == 0:
            print(f"Processed {i} items")

