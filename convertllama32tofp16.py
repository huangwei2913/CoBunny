import torch
import os
import json
from bunny.model import BunnyLlamaForCausalLM
from transformers import AutoConfig, AutoTokenizer

# 1. 配置路径
# 这是包含 config.json 的目录
model_path = "/mnt/CoBunny/checkpoints-stage3/bunny-llama-full-finetune/checkpoint-7674"
# 这是 zero_to_fp32 生成的那个 7.7GB 的文件夹路径
weights_dir = os.path.join(model_path, "pytorch_model.bin")
# 最终输出
output_dir = "/mnt/CoBunny/checkpoints-stage3/llama"

print(" 开始物理合并权重...")

# 2. 加载配置和 Tokenizer
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 3. 构造空的模型 (在 CPU 上)
# 注意：先不加载权重，只初始化结构
with torch.device("cpu"):
        model = BunnyLlamaForCausalLM(config)

        # 4.  手动加载那两个分片 (4.7G 和 3.0G 的那两个)
        # 绕过 transformers 的 index 加载逻辑，直接暴力读取数据
        shard1_path = os.path.join(weights_dir, "pytorch_model-00001-of-00002.bin")
        shard2_path = os.path.join(weights_dir, "pytorch_model-00002-of-00002.bin")

        print(f" 正在读取分片 1: {shard1_path}")
        state_dict1 = torch.load(shard1_path, map_location="cpu")
        print(f" 正在读取分片 2: {shard2_path}")
        state_dict2 = torch.load(shard2_path, map_location="cpu")

        # 合并 state_dict
        full_state_dict = {**state_dict1, **state_dict2}
        del state_dict1, state_dict2 # 释放内存

        # 5. 将权重注入模型并转为 FP16
        print(" 正在将权重注入结构并转换为 FP16...")
        model.load_state_dict(full_state_dict)
        model = model.to(torch.float16)

        # 6. 保存
        print(f" 正在保存最终单文件模型至: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)

        print("✅ 恭喜！全量 FP16 模型已物理合成成功！")
