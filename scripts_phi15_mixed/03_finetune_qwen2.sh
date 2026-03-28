#!/bin/bash
# ========================================================
# 1. 显卡锁定 (只使用 2, 3, 4, 5 号卡)
# ========================================================
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# ========================================================

# 2. 环境变量优化 (针对 16GB T4 显存)
# ========================================================

# 防止显存碎片化导致 OOM

export PYTORCH_ALLOC_CONF="expandable_segments:True"

# 禁用 NCCL 的 P2P（T4 在某些拓扑下 P2P 通信会卡死）
export NCCL_P2P_DISABLE=1          # T4 必关，防止 PCIe 冲突死锁
export NCCL_IB_DISABLE=1           # 禁用 Infiniband (如果环境没有 IB)
export NCCL_DEBUG=INFO             # 开启调试，出问题能看清是谁挂了
export NCCL_TIMEOUT=7200           # 超时时间拉长到 1 小时，防止 ZeRO-3 同步太慢

# ========================================================

# 3. 路径与模型定义

# ========================================================

MODEL_TYPE="Qwen2-1.5B"

#BASE_MODEL="/mnt/conda_data/checkpoints-pretrain/pretrain_stage1_qwen2/checkpoint-62431"
#BASE_MODEL="/mnt/conda_data/checkpoints-stage3/bunny-qwen2"
BASE_MODEL="/mnt/conda_data/checkpoints-finetuned/bunny-qwen2_continue"

#OUTPUT_DIR="/mnt/conda_data/checkpoints-stage3/bunny-qwen-full-finetune_color_shape"
OUTPUT_DIR="/mnt/conda_data/checkpoints-stage3/bunny-qwen-full-finetune_win"
#第一次用500K微调
#DATA_PATH="/data/MAmmoTH-VL-Instruct-12M/mammoth_500k_pilot.json"
#第二次用核心策略：4:1 比例（ColorBench 约 5.5k，配比 27.5k 猛犸数据）
#DATA_PATH="/mnt/conda_data/Bunny-v1.1-data/finetune/Bunny_Stage2_Color_Refine_v3.json"
DATA_PATH="/mnt/CoBunny/dataassert/v365_stage3_mcp_final_clean_fixed.json"
IMAGE_PATH="/"

RANDOM_PORT=$((RANDOM % 1000 + 20000))

mkdir -p $OUTPUT_DIR


deepspeed \
    --master_port $RANDOM_PORT \
    --include localhost:0,1,2,3,4,5,6,7 \
    bunny/train/train_stage3_qwen2.py \
    --deepspeed ./script/deepspeed/zero3_t4_qwen.json \
    --model_name_or_path $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --version bunny \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --vision_tower mixedencoder \
    --vision_tower_dino /mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m \
    --vision_tower_siglip /mnt/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --freeze_backbone False \
    --unfreeze_mm_vision_tower True \
    --lora_enable False \
    --bf16 False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 5e-6 \
    --max_grad_norm 1.0 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --group_by_modality_length True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none 2>&1 | tee $OUTPUT_DIR/finetunefull.log 