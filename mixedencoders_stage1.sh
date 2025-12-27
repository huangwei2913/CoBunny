#!/bin/bash

# --- 1. 分布式环境配置 (沿用第一阶段的设置) ---
MASTER_ADDR="192.168.0.3" 
MASTER_PORT=29500 
export DS_MASTER_PORT=$MASTER_PORT

# --- 2. 模型和项目配置 ---
MODEL_TYPE=phi-1.5
# 第一阶段的预对齐权重目录， 这个会在后面去修改
PRETRAIN_PROJ_DIR=mixed_encoder_multinode 

# 第二阶段的指令微调项目名称
PROJECT_NAME=phi-1.5-lora-finetune-multinode 
echo ${PROJECT_NAME}

# 定义输出目录
OUTPUT_DIR=./checkpoints-finetune/$PROJECT_NAME
mkdir -p $OUTPUT_DIR

# --- 3. 训练启动命令 (DeepSpeed 分布式 + LoRA 指令微调) ---

# ⭐️ 关键修改：将分布式参数放在 DeepSpeed 命令之后，Python 脚本之前
deepspeed \
    --hostfile ./script/deepspeed/hostfile \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path /mnt/conda_data/microsoft/phi-1_5 \
    --model_type $MODEL_TYPE \
    --version phi3  \
    --pretrain_mm_mlp_adapter /mnt/CoBunny/checkpoints-pretrain/oryx_phi/checkpoint-81250/mm_projector.bin \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --data_path /mnt/conda_data/Bunny-v1.1-data/finetune/bunny_695k.json \
    --image_folder /mnt/conda_data/Bunny-v1.1-data/finetune/images \
    --vision_tower mixedencoder \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --resume_from_checkpoint "$OUTPUT_DIR/checkpoint-15000" \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --tf32 False \
    --report_to none \
    | tee 2>&1 $OUTPUT_DIR/log.txt