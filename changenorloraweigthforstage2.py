import torch
import os

original_path = "/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/non_lora_trainables.bin"
save_path = "/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/fixed_non_lora.bin"

print("✂️ 正在开始 Key 命名手术...")
old_data = torch.load(original_path, map_location='cpu')
new_data = {}

for k, v in old_data.items():
    # 核心修复逻辑：去掉冗余的 base_model.model.model. 前缀
    # 对齐到 Bunny 期待的 model.xxx 格式
    new_k = k.replace('base_model.model.model.', 'model.')
    
    # 如果还是没对上，尝试更激进的替换
    if not new_k.startswith('model.'):
        new_k = 'model.' + new_k.split('model.')[-1]
    
    new_data[new_k] = v

# 同时检查是否有极值并净化
for k in new_data:
    new_data[k] = torch.nan_to_num(new_data[k].half(), nan=0.0, posinf=65000)

torch.save(new_data, save_path)
print(f"✅ 手术完成！修正后的权重已保存至: {save_path}")
print("👉 现在请修改你的 merge 脚本，指向这个 fixed_non_lora.bin 再尝试合并。")