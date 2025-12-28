# 阶段3：权重合并与平均（Merge LoRA & Averaging）
#这是一个综合脚本，先运行 merge_lora_weights.py，然后运行那段 Python 的平均逻辑（Safetensors 处理）。
#目的：把训练过程中的临时 LoRA 权重变成一个可以直接推理的完整模型。
#价值：这一步是把“训练代码”变成“作品”的过程。