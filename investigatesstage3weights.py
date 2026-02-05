import torch
import json
import os
import glob
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
from transformers import AutoConfig

def check_weights_quality(model_path):
    print(f"🔍 开始全量扫描权重路径: {model_path}")
    
    # 1. 加载 Index 文件看看官方定义的映射
    index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    
    bin_files = sorted(list(set(index_data["weight_map"].values())))
    print(f"Found {len(bin_files)} bin files.")

    # 2. 实例化一个空模型用于 Key 对比
    print("🏗️ 正在构建空模型结构用于对比...")
    config = AutoConfig.from_pretrained(model_path)
    model = BunnyPhiForCausalLM(config)
    model_keys = set(model.state_dict().keys())
    
    all_bin_keys = []
    
    # 3. 逐个 bin 文件扫描
    print("\n" + "="*60)
    print(f"{'Weight Key Name':<50} | {'Status':<10} | {'Mean Value'}")
    print("-" * 80)

    for bin_file in bin_files:
        bin_path = os.path.join(model_path, bin_file)
        print(f"\n📦 正在扫描文件: {bin_file}")
        state_dict = torch.load(bin_path, map_location="cpu")
        
        for k, v in state_dict.items():
            all_bin_keys.append(k)
            
            # 检查数值健康度
            has_nan = torch.isnan(v).any().item()
            is_all_zero = (v.abs().sum() == 0).item()
            mean_val = v.float().mean().item()
            
            status = "✅ OK"
            if has_nan: status = "❌ NaN"
            elif is_all_zero: status = "⚠️ ZERO"
            
            # 重点关注关键层
            if any(name in k for name in ["mm_projector", "embed_tokens", "vision_tower"]):
                print(f"{k[:50]:<50} | {status:<10} | {mean_val:.6f}")

        del state_dict # 释放内存

    # 4. 汇总对比报告
    print("\n" + "="*60)
    print("📊 最终对齐报告")
    
    matched = [k for k in all_bin_keys if k in model_keys]
    # 尝试去掉 "model." 前缀后再对比
    matched_with_prefix = [k for k in all_bin_keys if (k.startswith("model.") and k[6:] in model_keys)]
    
    print(f"1. 原始 Key 直接匹配成功: {len(matched)}")
    print(f"2. 去掉 'model.' 前缀匹配成功: {len(matched_with_prefix)}")
    print(f"3. 模型总参数项 (model_keys): {len(model_keys)}")
    
    # 找出模型里哪些关键层可能没被喂饱
    missing_layers = [mk for mk in model_keys if mk not in matched and mk not in [k[6:] for k in matched_with_prefix]]
    
    critical_missing = [m for m in missing_layers if "mm_projector" in m]
    if critical_missing:
        print("\n🚨 警告！以下关键 Projector 层在 bin 文件里找不到匹配:")
        for cm in critical_missing[:5]:
            print(f"   - {cm}")
    else:
        print("\n✨ 恭喜：所有 Projector 层均有对应权重。")

if __name__ == "__main__":
    MODEL_PATH = '/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune-modified'
    check_weights_quality(MODEL_PATH)