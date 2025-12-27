#!/bin/bash
set -e

# --- 路径配置 ---
MODEL_TYPE=phi-1.5 
PROJECT_NAME_ZERO2=phi-1.5-lora-finetune-optimized
OUTPUT_DIR=./checkpoints-finetune/$PROJECT_NAME_ZERO2
NUM_GPUS=8
MERGED_MODEL_PATH="/mnt/CoBunny/checkpoints-finetune/phi-1.5-lora-finetune-multinode-recipe2/zero3_merged_model" # Step 1 的输出路径

mkdir -p $OUTPUT_DIR
echo "🌟 启动 DeepSpeed Zero-2 高性能训练"

# --- 3. 训练启动命令 ---

deepspeed \
    --num_gpus=$NUM_GPUS \
    bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path $MERGED_MODEL_PATH \
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
    --per_device_train_batch_size 2 \
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
    --dataloader_num_workers 32 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to none \
    | tee 2>&1 $OUTPUT_DIR/log.txt