import os
import torch
from safetensors.torch import load_file
from collections import defaultdict

# =================配置区域=================
CHECKPOINT_PATH = "/mnt/conda_data/checkpoints-pretrain/pretrain_stage1_ocr_hard/checkpoint-29000"
# ==========================================

def inspect_checkpoint(checkpoint_dir):
    # 1. 寻找权重文件
    st_file = os.path.join(checkpoint_dir, "model.safetensors")
    bin_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    state_dict = {}
    if os.path.exists(st_file):
        print(f"🔍 正在加载 Safetensors 权重: {st_file}")
        state_dict = load_file(st_file, device="cpu")
    elif os.path.exists(bin_file):
        print(f"🔍 正在加载 PyTorch Bin 权重: {bin_file}")
        state_dict = torch.load(bin_file, map_location="cpu")
    else:
        print(f"❌ 错误：在 {checkpoint_dir} 中没找到 model.safetensors 或 pytorch_model.bin")
        return

    # 2. 分类统计
    stats = defaultdict(lambda: {"count": 0, "params": 0})
    
    print("\n" + "="*80)
    print(f"{'权重名称':<60} | {'形状':<20}")
    print("-"*80)

    for name, param in state_dict.items():
        num_params = param.numel()
        shape_str = str(list(param.shape))
        
        # 简单分类逻辑
        if "embed_tokens" in name or "lm_head" in name:
            category = "Embeddings & Head (词表相关)"
        elif "layers" in name:
            category = "LLM Backbone (Transformer层)"
        elif "vision_tower" in name:
            category = "Vision Tower (视觉塔)"
        elif "mm_projector" in name:
            category = "MM Projector (投影层)"
        else:
            category = "Others (其他)"

        stats[category]["count"] += 1
        stats[category]["params"] += num_params
        
        # 打印每一个具体权重的名称（如果太多可以根据需要注释掉这一行）
        print(f"{name:<60} | {shape_str:<20}")

    # 3. 汇总报告
    print("\n" + "="*80)
    print(f"{'分类统计':<40} | {'层数':<10} | {'总参数量 (亿)'}")
    print("-"*80)
    total_all = 0
    for cat, data in stats.items():
        billion_params = data["params"] / 1e8
        total_all += data["params"]
        print(f"{cat:<40} | {data['count']:<10} | {billion_params:.4f} 亿")
    
    print("-"*80)
    print(f"{'总计':<40} | {'-':<10} | {total_all/1e8:.4f} 亿")
    print("="*80)

if __name__ == "__main__":
    inspect_checkpoint(CHECKPOINT_PATH)