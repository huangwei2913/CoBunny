import torch
import os
from bunny.model import BunnyPhiForCausalLM
from transformers import AutoConfig, AutoTokenizer

# 1. 路径设置 (对齐你 ll 命令里的路径)
model_path = "/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune_modified/checkpoint-2000"
# 这是 zero_to_fp32 脚本生成的那个包含分片的文件夹
weights_dir = model_path 
output_dir = "/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune-2000-fp16"

print("🔥 内存已就位，开始暴力物理合并...")

# 2. 暴力读取两个分片到内存
shard1_path = os.path.join(weights_dir, "pytorch_model-00001-of-00002.bin")
shard2_path = os.path.join(weights_dir, "pytorch_model-00002-of-00002.bin")

print(f"📦 正在搬运分片 1 (4.9G)...")
state_dict1 = torch.load(shard1_path, map_location="cpu")
print(f"📦 正在搬运分片 2 (3.2G)...")
state_dict2 = torch.load(shard2_path, map_location="cpu")

# 3. 字典合体
print("⚡ 正在进行字典大融合...")
full_state_dict = {**state_dict1, **state_dict2}
del state_dict1, state_dict2 # 即使内存大，随手清理也是好习惯

# 4. 初始化结构并注入权重
print("🧠 初始化模型结构并注入灵魂...")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

with torch.device("cpu"):
    model = BunnyPhiForCausalLM(config)
    # 这一步是关键：把刚才合并的字典塞进去
    model.load_state_dict(full_state_dict)
    
    # 5. 暴力转 FP16 (这步最烧内存，但你没问题)
    print("✨ 正在进行 FP32 -> FP16 终极转换...")
    model = model.to(torch.float16)

# 6. 保存为单文件
print(f"💾 正在导出最终单文件模型至: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# 这一步会生成一个约 3.8GB 的单文件 pytorch_model.bin
model.save_pretrained(output_dir, max_shard_size="10GB", safe_serialization=False)
tokenizer.save_pretrained(output_dir)

print("\n✅ 合并完成！现在这个目录就是一个完美的、全量的 FP16 模型了！")