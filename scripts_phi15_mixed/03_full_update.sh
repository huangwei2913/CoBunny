#!/bin/bash

# ========================================================
# 1. 基础配置
# ========================================================
MASTER_ADDR=${MASTER_ADDR:-"192.168.0.3"}
MASTER_PORT=${MASTER_PORT:-"29501"}
# 你的 hostfile 配置
HOSTFILE="./script/deepspeed/hostfile"
# 确保所有卡都参与
INCLUDE_STR="192.168.0.3:0,1,2,3,4,5,6,7"

# ========================================================
# 2. 路径定义
# ========================================================
MODEL_TYPE="phi-1.5"
BASE_MODEL="./checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-23476"
OUTPUT_DIR="./checkpoints-stage3/bunny-phi1.5-full-finetune-modified"  #我们重新设计了第三个阶段代码
DATA_PATH="/mnt/conda_data/Bunny-v1.1-data/finetune/bunny_stage3_cleaned.json"
IMAGE_PATH="/"
# 关键：指向 Stage 1 跑出来的那个包含 117 个 Key 的文件
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1
export DEEPSPEED_USE_TORCH_ADAM=1
export NCCL_DEBUG=INFO  # 开启调试模式，这样卡住时能看到为什么卡
export NCCL_SOCKET_IFNAME=eth0 
export GLOO_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=9600
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
# ========================================================
# 3. 启动训练 (Stage 2: Instruction Tuning)
# ========================================================
# 注意：这里我们使用 Zero-3 (如果显存够用 Zero-2 也可以，但 LoRA + 2M 数据建议 Zero-3 更稳)
# 增加了 --lora_enable 等参数，用的是经过精选后的数据

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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-7 \
    --max_grad_norm 1.0 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --group_by_modality_length True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none 2>&1 | tee $OUTPUT_DIR/finetunefull.log


###--group_by_modality_length True  这个必须要的