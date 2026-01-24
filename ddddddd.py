import torch
import os

def audit_model_weights():
    # --- 1. 定义路径 ---
    merged_model_bin = "/mnt/CoBunny/models/bunny-phi1.5-stage2-final_rv/pytorch_model.bin"
    ckpt_dir = "/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-6619"
    
    print(f"🔍 正在读取合并后的全量模型: {merged_model_bin}")
    merged_state_dict = torch.load(merged_model_bin, map_location="cpu")
    merged_keys = sorted(list(merged_state_dict.keys()))

    print(f"📊 合并模型总参数槽位: {len(merged_keys)}")
    
    # --- 2. 分类统计合并后的模型 Key ---
    llm_keys = [k for k in merged_keys if 'model.layers' in k]
    proj_keys = [k for k in merged_keys if 'mm_projector' in k]
    vision_keys = [k for k in merged_keys if 'vision_tower' in k]
    
    print(f"\n📑 架构分布预览:")
    print(f"  - 语言模型层 (LLM): {len(llm_keys)} 个 Key")
    print(f"  - 投影层 (Projector): {len(proj_keys)} 个 Key")
    print(f"  - 视觉塔 (Vision Tower): {len(vision_keys)} 个 Key")

    # --- 3. 打印关键层级的具体命名 (用于对齐分析) ---
    print("\n📝 [关键层级命名审计]")
    print(f"  - LLM 示例: {llm_keys[0] if llm_keys else 'N/A'}")
    print(f"  - Projector 示例: {proj_keys[0] if proj_keys else 'N/A'}")
    print(f"  - Vision Tower 示例: {vision_keys[0] if vision_keys else 'N/A'}")
    
    # --- 4. 对比训练产出的原始权重 ---
    print("\n📂 [对比原始训练产物]")
    for bin_file in ['adapter_model.bin', 'vision_tower_tuned.bin', 'mm_projector.bin']:
        bin_path = os.path.join(ckpt_dir, bin_file)
        if os.path.exists(bin_path):
            original_weights = torch.load(bin_path, map_location="cpu")
            orig_keys = list(original_weights.keys())
            print(f"\n📦 文件: {bin_file}")
            print(f"  - 原始 Key 示例: {orig_keys[0]}")
            
            # 自动寻找匹配规律
            sample_key = orig_keys[0]
            # 尝试暴力匹配
            matched = False
            for mk in merged_keys:
                if sample_key.split('.')[-2:] == mk.split('.')[-2:]:
                    print(f"  🎯 发现匹配规律!")
                    print(f"    [源] {sample_key}")
                    print(f"    [目] {mk}")
                    matched = True
                    break
            if not matched:
                print("  ❌ 未发现直接匹配规律，可能需要路径重组。")

if __name__ == "__main__":
    audit_model_weights()