import torch

# 加载两个版本的权重
p1000 = torch.load("/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-4000/mm_projector.bin", map_location='cpu')
p2000 = torch.load("/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-6000/mm_projector.bin", map_location='cpu')

# 计算第一个权重的差值
key = list(p1000.keys())[0] # 取第一个 tensor
diff = (p1000[key].float() - p2000[key].float()).abs().mean()

print(f"权重 {key} 的平均更新量: {diff.item():.10f}")
