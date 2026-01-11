import torch
from safetensors.torch import save_file
import os

model_path = "/mnt/CoBunny/checkpoints-finetune/phi-1.5-bunny-mixed-lora-695"
bin_file = os.path.join(model_path, "pytorch_model.bin")
save_file_path = os.path.join(model_path, "model.safetensors")

print("🚀 正在加载 .bin 文件 (已跳过安全检查)...")

# 关键修正：添加 weights_only=False
state_dict = torch.load(bin_file, map_location="cpu", weights_only=False)

# 有时候 state_dict 可能会嵌套在 'model' 或 'state_dict' 键下，
# 检查一下以确保我们保存的是纯权重字典
if "model" in state_dict:
    state_dict = state_dict["model"]

print("📦 正在转换为真正的 .safetensors...")
save_file(state_dict, save_file_path)
print(f"✅ 转换完成！文件已保存至: {save_file_path}")