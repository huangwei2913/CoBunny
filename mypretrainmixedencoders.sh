#!/bin/bash

MASTER_ADDR="192.168.0.3" 
MASTER_PORT=29500 
export DS_MASTER_PORT=$MASTER_PORT

MODEL_TYPE=phi-1.5
PROJECT_NAME=mixed_encoder_multinode # 使用一个新的 PROJECT_NAME 避免冲突
echo ${PROJECT_NAME}

# 注意：这里使用 $PROJECT_NAME
mkdir -p ./checkpoints-pretrain/$PROJECT_NAME

# --------------------------------------------------------------------------
# ⭐️ 关键修改：将 --hostfile, --master_addr, --master_port 放在 DeepSpeed 命令之后
#/mnt/CoBunny/checkpoints-pretrain/oryx_phi/checkpoint-81250 这个里面的权重是最小的 
# 且在 Python 脚本 bunny/train/train.py 之前
# --------------------------------------------------------------------------
deepspeed \
    --hostfile ./script/deepspeed/hostfile \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path /mnt/conda_data/microsoft/phi-1_5\
    --model_type $MODEL_TYPE \
    --version plain \
    --data_path /mnt/conda_data/Bunny-v1.1-data/pretrain/bunny_pretrain_laion_2m.json \
    --image_folder /mnt/conda_data/Bunny-v1.1-data/pretrain/images \
    --vision_tower mixedencoder \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --image_aspect_ratio square \
    --bf16 True \
    --output_dir ./checkpoints-pretrain/$PROJECT_NAME \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --tf32 False \
    --report_to none \
    | tee 2>&1 ./checkpoints-pretrain/$PROJECT_NAME/log.txt