import torch
# 指向 6619 具体的 mm_projector.bin
p_path = "/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-6619/mm_projector.bin"
weights = torch.load(p_path, map_location='cpu')

print(f"📊 Projector 审计报告:")
for k, v in weights.items():
    print(f"Key: {k:40} | Max: {v.max().item():.4f} | Min: {v.min().item():.4f} | Mean: {v.mean().item():.4f}")
    



