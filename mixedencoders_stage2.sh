#!/bin/bash

# --- 分布式配置 (保持不变) ---
RECIPE1_PROJECT_NAME=phi-1.5-lora-finetune-multinode-recipe2
RECIPE1_OUTPUT_DIR=./checkpoints-finetune/$RECIPE1_PROJECT_NAME
LAST_CHECKPOINT_PATH="$RECIPE1_OUTPUT_DIR/checkpoint-18000" 
MASTER_ADDR="192.168.0.3" 
MASTER_PORT=29500 
export DS_MASTER_PORT=$MASTER_PORT
MODEL_TYPE=phi-1.5 
PROJECT_NAME=phi-1.5-lora-finetune-multinode-recipe2
echo ${PROJECT_NAME}
OUTPUT_DIR=./checkpoints-finetune/$PROJECT_NAME
mkdir -p $OUTPUT_DIR



# --- 3. 训练启动命令 (DeepSpeed 分布式 + LoRA 指令微调) ---

deepspeed \
    --hostfile ./script/deepspeed/hostfile \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /mnt/conda_data/microsoft/phi-1_5 \
    --model_type $MODEL_TYPE \
    --version phi3 \
    --pretrain_mm_mlp_adapter /mnt/CoBunny/checkpoints-pretrain/oryx_phi/checkpoint-81250/mm_projector.bin \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --mm_projector_lr 2e-5 \
    --data_path /mnt/conda_data/Bunny-v1.1-data/finetune/bunny_llava_allava_2m.json \
    --image_folder /mnt/conda_data/Bunny-v1.1-data/finetune/images \
    --vision_tower mixedencoder \
    --unfreeze_vision_tower True \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --report_to none \
    | tee 2>&1 $OUTPUT_DIR/log.txt