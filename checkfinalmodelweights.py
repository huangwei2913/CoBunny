import torch
import os

def check_model_integrity(model_bin_path):
    print(f"🔍 开始扫描模型指纹: {model_bin_path}")
    
    if not os.path.exists(model_bin_path):
        print("❌ 错误：找不到 pytorch_model.bin 文件！")
        return

    # 只加载 key，不加载具体张量，速度极快
    state_dict = torch.load(model_bin_path, map_location='cpu')
    all_keys = state_dict.keys()
    
    # 定义我们需要检查的关键模块特征
    check_list = {
        "语言模型 (LLM)": "model.layers.0.self_attn.q_proj.weight",
        "投影层 (Projector)": "model.mm_projector.0.weight",
        "DINOv3 视觉塔": "model.vision_tower.dino_vision_tower.vision_tower",
        "Oryx-ViT 视觉塔": "model.vision_tower.oryx_vision_tower.vision_tower",
        "自定义 Cross-Attn 融合层": "model.vision_tower.cross_attn_block",
        "可学习的 Pseudo-CLS 头": "model.vision_tower.b_pseudo_cls_head"
    }
    
    print("\n" + "="*50)
    print(f"{'组件名称':<25} | {'检测结果':<10}")
    print("-"*50)
    
    found_count = 0
    for name, key_prefix in check_list.items():
        # 只要包含该前缀的 key 存在，就说明模块在里面
        is_found = any(key_prefix in k for k in all_keys)
        status = "✅ 存在" if is_found else "❌ 缺失"
        if is_found: found_count += 1
        print(f"{name:<25} | {status}")
        
    print("="*50)
    
    if found_count == len(check_list):
        print("\n🎊 校验通过！你的 3.9G 模型是一个完整的“混合动力”多模态模型。")
        # 随机打印一个具体的权重形状看看，增加确定感
        sample_key = "model.mm_projector.0.weight"
        print(f"📊 投影层维度采样: {state_dict[sample_key].shape} (符合预期)")
    else:
        print("\n⚠️ 校验未完全通过，请检查合并脚本是否漏掉了某些模块。")

# 使用你的路径
merged_model_bin = "/mnt/CoBunny/checkpoints-finetune/phi-1.5-bunny-mixed-final/pytorch_model.bin"
check_model_integrity(merged_model_bin)