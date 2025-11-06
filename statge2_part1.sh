#!/bin/bash

#视觉编码器：冻结。
#投影映射层：解冻 通过 --mm_projector_lr 2e-5 指定了学习率）。
#LLAMA部分解冻（通过 --lora_enable True 启用 $\text{LoRA}$ 权重）。
# 在高质量指令数据上，训练 $\text{LLM}$ 的推理和生成能力，同时精调投影层，使模型能够响应复杂的指令。

MODEL_TYPE=phi-3

PRETRAIN_DIR=bunny-$MODEL_TYPE-pretrain
OUTPUT_DIR=bunny-lora-$MODEL_TYPE-recipe-1

mkdir -p ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR

deepspeed bunny/train/train.py \
    --lora_enable True  \
    --lora_r 128  \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /mnt/microsoft/phi-1_5\
    --model_type $MODEL_TYPE \
    --version phi3 \
    --data_path /mnt/Bunny-v1.1-data/finetune/bunny_695k.json \
    --image_folder /mnt/Bunny-v1.1-data/finetune/images \
    --vision_tower /mnt/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt