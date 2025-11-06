# fix_jsonl_key.py
import json
import os

def fix_jsonl_file(input_path, output_path, dataset_split):
    """
    读取 JSONL 文件，将 'image_id' 转换为 'image' (文件名)。
    """
    if 'test2015' in dataset_split:
        prefix = "COCO_test2015_"
    elif 'train2014' in dataset_split:
        prefix = "COCO_train2014_"
    else:
        # 如果不是标准COCO，可能需要自定义前缀
        prefix = "" 
        print(f"Warning: Unknown split '{dataset_split}', using empty prefix.")

    fixed_lines = []
    
    with open(input_path, 'r') as infile:
        for line in infile:
            try:
                data = json.loads(line)
                
                if 'image_id' in data and 'image' not in data:
                    image_id = data['image_id']
                    
                    # 构造 COCO 格式的文件名 (12位零填充)
                    image_filename = f"{prefix}{image_id:012d}.jpg"
                    
                    # 移除旧键并添加新键
                    # data.pop('image_id', None) # 可以选择移除旧键
                    data['image'] = image_filename
                
                fixed_lines.append(json.dumps(data) + '\n')
                
            except json.JSONDecodeError:
                print(f"Skipping badly formatted line: {line.strip()}")
            
    with open(output_path, 'w') as outfile:
        outfile.writelines(fixed_lines)
    
    print(f"Successfully fixed {len(fixed_lines)} lines.")
    print(f"Saved to: {output_path}")

# --- 执行修复 ---
INPUT_FILE = "/mnt/CoBunny/eval/vqav2/bunny_vqav2_mscoco_test-dev2015.jsonl"
OUTPUT_FILE = "/mnt/CoBunny/eval/vqav2/bunny_vqav2_mscoco_test-dev2015_fixed.jsonl"
DATASET_SPLIT = "test-dev2015" # 用于确定前缀

fix_jsonl_file(INPUT_FILE, OUTPUT_FILE, DATASET_SPLIT)
