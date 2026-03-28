#!/bin/bash

# 1. 强制逻辑隔离坏卡 (假设 7 号卡损坏)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# 2. 环境变量补丁 (针对 T4/云主机稳定性)
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export NCCL_IB_DISABLE=1      # 禁用 IB 协议，走以太网
export NCCL_P2P_DISABLE=1     # 关键：禁用 P2P 降低总线压力
export NCCL_TIMEOUT=12000     # 防止 NFS 读取慢导致超时
export DS_SKIP_CUDA_CHECK=1   # 跳过 DeepSpeed 的硬件检查

# 3. 参数定义
MODEL_TYPE="llama3-1b"
BASE_MODEL="/mnt/conda_data/Llama-3.2-1B"
VISION_TOWER="mixedencoder"
OUTPUT_DIR="/mnt/conda_data/checkpoints-pretrain/pretrain_llama"
mkdir -p $OUTPUT_DIR

# 4. 启动 DeepSpeed (只给 7 张卡)
# 这里的 --include 是关键，它会让进程只在 0-6 号卡上启动
deepspeed \
    --include localhost:0,1,2,3,4,5,6 \
    --master_port 29505 \
    bunny/train/train_stage1_llama.py \
    --deepspeed ./script/deepspeed/zero2_mixencoders_pretraing.json \
    --model_name_or_path $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --version bunny \
    --vision_tower $VISION_TOWER \
    --vision_tower_dino /mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m \
    --vision_tower_siglip /mnt/siglip-so400m-patch14-384 \
    --data_path /mnt/conda_data/Bunny-v1.1-data/pretrain/bunny_stage1_cleaned.json \
    --image_folder /mnt/conda_data/Bunny-v1.1-data/pretrain/images \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_backbone True \
    --bf16 False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --learning_rate 2e-4 \
    --max_grad_norm 0.5 \
    --lr_scheduler_type  "cosine"\
    --logging_steps 10 \
    --warmup_ratio 0.1  \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --greater_is_better False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --group_by_modality_length True \
    --dataloader_num_workers 16 \
    --report_to none 2>&1 | tee $OUTPUT_DIR/pretrain.log



    