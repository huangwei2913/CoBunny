import torch
import os

path = '/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/non_lora_trainables.bin'
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 正在使用 {device} 深度扫描权重文件...")

# 1. 加载数据
data = torch.load(path, map_location=device)

nan_keys = []
inf_keys = []

# 2. 快速迭代检查
for k, v in data.items():
    if not isinstance(v, torch.Tensor):
        continue
        
    # 在 GPU 上这行代码是毫秒级的
    has_nan = torch.isnan(v).any().item()
    has_inf = torch.isinf(v).any().item()
    
    if has_nan:
        nan_keys.append(k)
        print(f"❌ 发现 NaN: {k}")
    if has_inf:
        inf_keys.append(k)
        print(f"⚠️ 发现 Inf: {k}")

print("-" * 50)
if not nan_keys and not inf_keys:
    print(f"✅ 扫描完成！{len(data)} 个权重全部健康，无 NaN/Inf。")
    print("这意味着：权重文件本身是好的，问题百分之百出在 builder.py 加载或 Base Model 身上。")
else:
    print(f"🚨 结论：权重文件已损坏。共计 {len(nan_keys)} 项 NaN，{len(inf_keys)} 项 Inf。")
    print("这意味着：训练阶段（Stage 2）的融合层参数已经练炸了。")