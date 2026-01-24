import torch
import os

def check_bunny_weights(projector_path, vision_tower_path=None):
    print(f"\n{'='*20} 1. 投影层检查 (Projector) {'='*20}")
    if os.path.exists(projector_path):
        weights = torch.load(projector_path, map_location='cpu')
        keys = list(weights.keys())
        print(f"文件路径: {projector_path}")
        print(f"参数总量: {len(keys)}")
        print(f"前 10 个 Key 示例:")
        for k in keys[:10]:
            # 打印 key 的名字和维数，方便对齐
            print(f"  - {k} \t shape: {list(weights[k].shape)}")
        
        # 自动分析前缀
        if any('mm_projector' in k for k in keys):
            print("⚠️ 发现含有 'mm_projector.' 前缀，加载时需 split 后缀。")
        elif any('0.weight' == k or '0.bias' == k for k in keys):
            print("✅ 发现 Key 是纯净的序号开头 (如 0.weight)，这是 Sequential 期望的格式。")
    else:
        print(f"❌ 未找到投影层文件: {projector_path}")

    if vision_tower_path and os.path.exists(vision_tower_path):
        print(f"\n{'='*20} 2. 视觉塔检查 (Vision Tower) {'='*20}")
        vt_weights = torch.load(vision_tower_path, map_location='cpu')
        vt_keys = list(vt_weights.keys())
        print(f"文件路径: {vision_tower_path}")
        print(f"参数总量: {len(vt_keys)}")
        print(f"融合层相关 Key 示例:")
        # 看看有没有你定义的那些融合层关键字
        fusion_keywords = ['mlp_layers', 'cross_attn', 'cls_weights']
        found_fusion = [k for k in vt_keys if any(kw in k for kw in fusion_keywords)]
        for k in found_fusion[:]:
            print(f"  - {k}")
        if not found_fusion:
            print("⚠️ 未发现明显的融合层 Key，请检查保存逻辑。")
    elif vision_tower_path:
        print(f"\n❌ 未找到视觉塔权重文件: {vision_tower_path}")

# --- 请修改为你实际的路径 ---
projector_file = "/mnt/CoBunny/checkpoints-pretrain/bunny-phi1.5-mixed-pretrain/checkpoint-33300/mm_projector.bin"
vision_file = "/mnt/CoBunny/checkpoints-pretrain/bunny-phi1.5-mixed-pretrain/checkpoint-33300/vision_tower_tuned.bin"

check_bunny_weights(projector_file, vision_file)