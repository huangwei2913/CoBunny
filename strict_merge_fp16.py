import torch
import os
import json
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
from transformers import AutoConfig, AutoTokenizer

# ================= 配置路径 =================
# 指向包含 config.json 和那两个 7.7GB 分片的目录
MODEL_PATH = "/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune-final/checkpoint-3000"
# 最终输出目录
OUTPUT_DIR = "/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune-final-fp16"

print(f"🚀 [系统启动] 内存资源充足 ({684}GB)，启动全量镜像合并模式...")

# 1. 物理读取分片（严格顺序读取）
shard1_path = os.path.join(MODEL_PATH, "pytorch_model-00001-of-00002.bin")
shard2_path = os.path.join(MODEL_PATH, "pytorch_model-00002-of-00002.bin")

print(f"📦 正在载入分片 1: {shard1_path}")
state_dict = torch.load(shard1_path, map_location="cpu")

print(f"📦 正在载入分片 2: {shard2_path}")
state_dict_shard2 = torch.load(shard2_path, map_location="cpu")

# 严格合并字典
print("🔄 正在执行字典合并...")
state_dict.update(state_dict_shard2)
del state_dict_shard2 # 及时清理

# 2. 构造模型结构
print("🏗️ 正在构造模型骨架...")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 【核心修复】在注入前修正 Config
config.unfreeze_mm_vision_tower = True
config.tune_mm_vision_resampler = True

# 构造空模型
with torch.device("cpu"):
    model = BunnyPhiForCausalLM(config)

# 3. 严格注入权重
print("💉 正在执行权重注入 (load_state_dict)...")
# 使用 strict=True 确保每一个参数都对上，如果有 Key 缺失立刻报错
info = model.load_state_dict(state_dict, strict=True)
print(f"✅ 注入完成！状态报告: {info}")

# 4. 精度转换
print("✨ 正在将模型全量转换为 FP16...")
model.half()

# 5. 保存最终形态
print(f"💾 正在保存全量模型至: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存模型权重 (取消安全序列化以保持兼容性，或设为 True 生成 safetensors)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

# 保存 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# 6. 验证保存结果
final_bin_path = os.path.join(OUTPUT_DIR, "model.safetensors")
if os.path.exists(final_bin_path):
    size_gb = os.path.getsize(final_bin_path) / (1024**3)
    print(f"🏁 转换成功！最终模型大小: {size_gb:.2f} GB")
else:
    print("⚠️ 警告：未能找到生成的权重文件，请检查目录。")

print("🌟 任务圆满完成。现在你可以使用 extract_weights.py 提取真正的原子包了。")