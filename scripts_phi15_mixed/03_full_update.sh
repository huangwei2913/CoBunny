#!/bin/bash

# ========================================================
# 1. 基础环境配置 (保持你之前的网络优化)
# ========================================================
MASTER_ADDR=${MASTER_ADDR:-"192.168.0.3"}
MASTER_PORT=${MASTER_PORT:-"29501"}
HOSTFILE="./script/deepspeed/hostfile"
INCLUDE_STR="192.168.0.3:0,1,2,3,4,5,6,7"

# 显存碎片优化 (关键)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
# NCCL 稳定性配置 (防止多机死锁)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0 
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=12000
export NCCL_ASYNC_ERROR_HANDLING=1

# 其他优化
export PYTHONUNBUFFERED=1
export DS_SKIP_CUDA_CHECK=1
export DEEPSPEED_USE_TORCH_ADAM=1

# ========================================================
# 2. 路径定义
# ========================================================
MODEL_TYPE="phi-1.5"

# [输入] 指向 Stage 1 (或 Stage 2) 的产出目录,也就说当存储一个checkpoint后，我们应该用整理好的这个final-fp16
# 确保这个目录下有 config.json, model.safetensors 等完整文件
BASE_MODEL="/mnt/CoBunny/checkpoints-stage3/bunny-phi1.5"
#由于重新开有问题，那我们就
# [输出] Stage 3 的保存位置
OUTPUT_DIR="./checkpoints-stage3/bunny-phi1.5-full-finetune_fashion"

# [数据] Stage 3 的高质量混合数据
DATA_PATH="/data/fashion/FashionRec/fashion_visual_alignment_gold.json"
IMAGE_PATH="/"

# ========================================================
# 3. 启动全量微调 (Full Fine-tuning)yuuuuuuuuuuuuuuuuuuuuuuuuu
# ========================================================
# 继承 Stage 2 的成功参数：LR=2e-5, Cosine Scheduler
# 区别：关闭 LoRA，开启全参数更新

deepspeed \
    --hostfile $HOSTFILE \
    --include "$INCLUDE_STR" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    bunny/train/train_stage3.py \
    --deepspeed ./script/deepspeed/zero3_mixedencoders_full.json \
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
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --group_by_modality_length True \
    --dataloader_num_workers 8\
    --lazy_preprocess True \
    --report_to none 2>&1 | tee $OUTPUT_DIR/finetunefull.log