#!/bin/bash
MODEL_TYPE=phi-1.5
OUTPUT_DIR=bunny-$MODEL_TYPE-pretrain
PROJECT_NAME=oryx_phi
echo ${PROJECT_NAME}

mkdir -p ./checkpoints-pretrain/$OUTPUT_DIR

deepspeed bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path /mnt/microsoft/phi-1_5 \
    --model_type $MODEL_TYPE \
    --version plain \
    --data_path /mnt/Bunny-v1.1-data/pretrain/bunny_pretrain_laion_2m.json \
    --image_folder /mnt/Bunny-v1.1-data/pretrain/images \
    --vision_tower  mixedencoder \
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
    --tf32 False  \
    --report_to none | tee 2>&1 ./checkpoints-pretrain/$PROJECT_NAME/log.txt
