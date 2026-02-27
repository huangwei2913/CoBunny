import torch
import os
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
from transformers import AutoConfig

MODEL_PATH = '/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune-2000-fp16'

def scan_model_and_weights():
    print(f"🔍 正在扫描路径: {MODEL_PATH}")
    
    # 1. 加载配置和模型骨架 (实心初始化到 CPU)
    print("🏗️ 1. 正在初始化模型骨架...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = BunnyPhiForCausalLM(config)
    
    # 获取模型预期的所有 Key
    model_keys = set(model.state_dict().keys())
    
    # 2. 加载物理权重文件
    bin_file = os.path.join(MODEL_PATH, "pytorch_model.bin")
    print(f"📦 2. 正在读取物理权重文件 ({os.path.getsize(bin_file)/1024**3:.2f} GB)...")
    state_dict = torch.load(bin_file, map_location="cpu")
    weight_keys = set(state_dict.keys())

    # 3. 统计比对
    missing_keys = model_keys - weight_keys
    unexpected_keys = weight_keys - model_keys
    
    print("\n" + "="*50)
    print(f"📊 统计结果:")
    print(f"  - 模型预期总参数项: {len(model_keys)}")
    print(f"  - 权重文件总参数项: {len(weight_keys)}")
    print(f"  - 完美匹配项: {len(model_keys & weight_keys)}")
    print(f"  - 缺失项 (Missing): {len(missing_keys)}")
    print(f"  - 多余项 (Unexpected): {len(unexpected_keys)}")
    print("="*50 + "\n")

    # 4. 专项检查：视觉塔 (Vision Tower)
    print("📸 正在针对 [视觉塔] 进行专项审计...")
    vt_model_keys = [k for k in model_keys if 'vision_tower' in k.lower()]
    vt_weight_keys = [k for k in weight_keys if 'vision_tower' in k.lower()]
    
    print(f"  - 模型结构中视觉塔参数项: {len(vt_model_keys)}")
    print(f"  - 权重文件中视觉塔参数项: {len(vt_weight_keys)}")
    
    if len(vt_model_keys) > 0 and len(vt_weight_keys) == 0:
        print("❌ 严重错误：权重文件中完全没有视觉塔权重！你的合并脚本丢包了。")
    elif len(vt_model_keys) != len(vt_weight_keys):
        print("⚠️ 警告：视觉塔参数数量不一致，可能存在命名错位或部分缺失。")
    else:
        print("✅ 视觉塔参数数量基本一致。")

    # 5. 采样命名对齐情况
    if vt_model_keys:
        print("\n🔍 命名对齐采样 (前3项):")
        print(f"  模型预期样例: {vt_model_keys[:3]}")
        # 找找权重文件里最像的
        sample_vt_weight = [k for k in vt_weight_keys[:3]]
        print(f"  文件实际样例: {sample_vt_weight}")

    # 6. 检查是否有 Meta Tensor 残留
    meta_params = [n for n, p in model.named_parameters() if p.device.type == 'meta']
    if meta_params:
        print(f"\n👻 发现 {len(meta_params)} 个参数仍处于 Meta 状态！")

if __name__ == "__main__":
    scan_model_and_weights()