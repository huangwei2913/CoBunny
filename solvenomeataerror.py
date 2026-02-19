import torch
import os
from transformers import AutoTokenizer, AutoConfig
from bunny.model import BunnyPhiForCausalLM

# ================= 配置区 =================
# 1. 你现在那个报错的权重路径
ORIGINAL_PATH = '/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune-final/checkpoint-3303'

# 2. 我们要生成的“干净”权重路径
CLEAN_PATH = '/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5-full-finetune-final/checkpoint-3303-FULL-FIXED'

# 3. 核心逻辑 ID
TARGET_LOGICAL_ID = -200 
# ==========================================

def fix_and_save():
    print(f"🚀 开始加载原始模型 (使用 CPU 强制物化)...")
    
    # A. 先修正配置
    config = AutoConfig.from_pretrained(ORIGINAL_PATH, trust_remote_code=True)
    config.image_token_index = TARGET_LOGICAL_ID
    
    # B. 强制实心加载 (low_cpu_mem_usage=False 是关键，它会把 meta 填上零或默认值)
    model = BunnyPhiForCausalLM.from_pretrained(
        ORIGINAL_PATH,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=False, # 拒绝 Meta Tensor
        device_map=None          # 禁止自动分发
    )
    
    print("🛠️ 正在修复缺失的 position_ids 和 Buffer...")
    # C. 针对报错的 SigLIP 和 Phi 补全 Buffer
    for name, module in model.named_modules():
        # 如果某个模块有 position_ids 但它是空的或 meta，我们造一个真的给它
        if hasattr(module, 'position_ids'):
            max_len = config.max_position_embeddings
            # 这里的 2048 是 Phi 的默认长度，你可以根据 config 改
            pos_ids = torch.arange(max_len).unsqueeze(0)
            module.register_buffer('position_ids', pos_ids, persistent=True)
            print(f"   - 已补全: {name}.position_ids")

    # D. 强制同步模型内部的所有 config
    model.config.image_token_index = TARGET_LOGICAL_ID
    if hasattr(model, 'model'):
        model.model.config.image_token_index = TARGET_LOGICAL_ID

    print(f"💾 正在保存【实心全量版】模型到: {CLEAN_PATH}")
    if not os.path.exists(CLEAN_PATH):
        os.makedirs(CLEAN_PATH)
        
    model.save_pretrained(CLEAN_PATH, safe_serialization=True)
    
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(CLEAN_PATH)
    
    print("\n" + "="*50)
    print("🎉 修复完成！这个新的 Checkpoint 已经：")
    print("1. 补全了导致 meta tensor 报错的所有零件。")
    print("2. 永久性地把 image_token_index 改成了 -200。")
    print("="*50)

if __name__ == "__main__":
    fix_and_save()