import os
import json

def filter_records_with_images(data, image_dir):
    filtered = []
    for record in data:
        image_path = os.path.join(image_dir, record.get("image", ""))
        if os.path.isfile(image_path):
            filtered.append(record)
    return filtered

# 读取json文件
json_path = "/home/huangwei/Bunny/data/pretrain/bunny_pretrain_laion_2m.json"
image_directory = "/home/huangwei/Bunny/data/pretrain/images"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_data = filter_records_with_images(data, image_directory)

print(f"Original total records: {len(data)}")
print(f"Records with existing images: {len(filtered_data)}")

# 如果需要写回文件，可以取消下面注释：
with open("filtered_data.json", "w", encoding="utf-8") as f_out:
    json.dump(filtered_data, f_out, ensure_ascii=False, indent=2)

