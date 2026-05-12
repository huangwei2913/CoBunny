import os
import torch
from transformers import AutoConfig, AutoTokenizer
from peft import PeftConfig, get_peft_model
from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
from bunny.util.utils import disable_torch_init
from safetensors.torch import load_file

# --- 1. 路径定义 ---
# 基座：Stage 1 (4.1GB, 词表 51200, 结构最全)
BASE_S1_PATH = '/mnt/conda_data/checkpoints-pretrain/pretrain_stage1_ocr_enhanced/checkpoint-31947'
# 灵魂：Stage 3 (3.3GB, 包含您最新的 OCR 成果和 LoRA)
SOUL_S3_PATH = '/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-forzeLLM_ocr/checkpoint-4728'

SAVE_DIR = '/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-forzeLLM_ocr/final_ocr_direct_merge_3.9G'

def direct_merge_logic():
    disable_torch_init()

    print("🚀 第一步：加载 Stage 1 基座结构与权重 (51200 词表)...")
    config = AutoConfig.from_pretrained(BASE_S1_PATH, trust_remote_code=True)
    # 直接初始化，确保在物理内存中
    model = BunnyPhiForCausalLM(config)
    model = model.to(torch.float16)

    # 加载 Stage 1 的真实权重
    s1_file = os.path.join(BASE_S1_PATH, 'model.safetensors')
    if os.path.exists(s1_file):
        s1_weights = load_file(s1_file, device="cpu")
    else:
        s1_weights = torch.load(os.path.join(BASE_S1_PATH, 'pytorch_model.bin'), map_location='cpu')
    
    model.load_state_dict(s1_weights, strict=False)
    print("   └─ Stage 1 基座加载完成。")
    del s1_weights

    print("🚀 第二步：挂载 Stage 3 LoRA 骨架...")
    peft_config = PeftConfig.from_pretrained(SOUL_S3_PATH)
    model = get_peft_model(model, peft_config)

    print("🚀 第三步：用 Stage 3 (4728步) 权重全量覆盖...")
    # 这里加载您用 zero_to_fp32 拼好的 3.3GB bin
    s3_weights = torch.load(os.path.join(SOUL_S3_PATH, 'pytorch_model.bin'), map_location='cpu')
    
    # 这一步会精准地把 S3 里的视觉权重、Projector 权重和 LoRA 权重填入 S1 的基座中
    model.load_state_dict(s3_weights, strict=False)
    print("   └─ Stage 3 权重注入完成。")
    del s3_weights

    print("🚀 第四步：执行物理融合 (Merge & Unload)...")
    # 将 LoRA 权重永久合并进主权重
    model = model.merge_and_unload()

    # --- 最终导出 ---
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    print(f"🚀 第五步：导出大一统 3.9GB 模型到 {SAVE_DIR}...")
    
    # 保存为 pytorch_model.bin
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pytorch_model.bin"))
    
    # 拷贝所有配置文件
    model.config.save_pretrained(SAVE_DIR)
    os.system(f"cp {SOUL_S3_PATH}/*.json {SAVE_DIR}/")
    os.system(f"cp {SOUL_S3_PATH}/tokenizer* {SAVE_DIR}/")

    print(f"\n✅ 任务达成！这个模型就是：[S1 完整结构] 直接吸收 [S3 OCR 强化结果]。")
    print(f"请检查 {SAVE_DIR} 目录下的文件大小。")

if __name__ == "__main__":
    direct_merge_logic()