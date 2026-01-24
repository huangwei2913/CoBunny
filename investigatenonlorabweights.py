import torch
import os

# 指向你最新的 checkpoint 目录
path = "/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/non_lora_trainables.bin"

if not os.path.exists(path):
    print("❌ 找不到文件，请检查路径")
else:
    print(f"📦 正在解剖: {path}")
    data = torch.load(path, map_location='cpu')
    
    # 统计信息
    mm_keys = [k for k in data.keys() if 'mm_projector' in k]
    vision_keys = [k for k in data.keys() if 'vision_tower' in k]
    embed_keys = [k for k in data.keys() if 'embed_tokens' in k]
    
    print(f"📊 统计结果:")
    print(f"  - Projector 相关 Key 数量: {len(mm_keys)}")
    print(f"  - Vision Tower 相关 Key 数量: {len(vision_keys)}")
    print(f"  - Embedding 相关 Key 数量: {len(embed_keys)}")
    print(f"  - 总 Key 数量: {len(data.keys())}")

    print("\n📝 所有 non_lora_trainables Key 样板 (用来检查命名是否混乱):")
    for k in list(data.keys())[:]:
        print(f"  {k}")