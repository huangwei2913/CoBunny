import torch
# 加载权重字典（只读 key，不占内存）
state_dict = torch.load('/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-forzeLLM_ocr/checkpoint-4728/pytorch_model.bin/pytorch_model.bin', map_location='cpu')

# 检查是否含有 LoRA 关键字
has_lora = any('lora_' in k for k in state_dict.keys())
print(f"是否有 LoRA 权重: {has_lora}")

# 检查是否有视觉塔权重
has_vision = any('vision_tower' in k for k in state_dict.keys())
print(f"是否有视觉塔权重: {has_vision}")

del state_dict # 释放内存