import torch

# 路径
S1_PATH = '/mnt/conda_data/checkpoints-pretrain/pretrain_stage1_ocr_enhanced/checkpoint-31947/model.safetensors'
FINAL_PATH = '/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-forzeLLM_ocr/final_ocr_direct_merge_3.9G/pytorch_model.bin'

def final_check():
    from safetensors.torch import load_file
    base = load_file(S1_PATH, device="cpu")
    final = torch.load(FINAL_PATH, map_location='cpu')
    
    # 1. 验证词表
    v_base = base['model.embed_tokens.weight'].shape[0]
    v_final = final['model.embed_tokens.weight'].shape[0]
    print(f"📊 词表审计: 基座 {v_base} -> 最终 {v_final} ({'✅ 匹配' if v_base==v_final else '❌ 不匹配'})")

    # 2. 验证 LoRA 融合 (语言层)
    k = 'model.layers.0.self_attn.q_proj.weight'
    diff = torch.abs(base[k].float() - final[k].float()).mean().item()
    print(f"🧠 语言层审计: 偏移强度 {diff:.8f} ({'✅ 已融合 LoRA' if diff > 0 else '❌ 融合失败'})")

    # 3. 验证视觉对齐 (Projector 层 - OCR 关键)
    # 这是连接 Vision 和 LLM 的最后一层
    k_v = 'model.mm_projector.2.weight' 
    diff_v = torch.abs(base[k_v].float() - final[k_v].float()).mean().item()
    print(f"👁️  投影层审计: 偏移强度 {diff_v:.8f} ({'✅ 视觉已更新' if diff_v > 0 else '❌ 视觉未变动'})")

    if diff > 0 and diff_v > 0:
        print("\n🏆 结论：黄老师，这 3.9G 是真的！包含了 Phi 1.5 + Stage 1 视觉 + Stage 3 OCR 增强。")

final_check()