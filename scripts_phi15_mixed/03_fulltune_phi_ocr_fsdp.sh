#!/bin/bash

# ========================================================
# 1. 分布式环境配置 (适配 FSDP)
# ========================================================
# 既然您已经完成了 accelerate config，很多参数会自动从 yaml 读取
# 但为了多机/多卡稳定性，我们显式指定关键环境变量
export MASTER_ADDR=${MASTER_ADDR:-"192.168.0.3"}
export MASTER_PORT=${MASTER_PORT:-"29501"}

# 显存与网络优化 (针对 8xT4 环境)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=12000

# 其他基础配置
export PYTHONUNBUFFERED=1

# ========================================================
# 2. 路径定义 (保持您的 OCR 专项路径)
# ========================================================
MODEL_TYPE="phi-1.5"
# 基础模型：指向您 pretrain 阶段产出的 checkpoint
BASE_MODEL="/mnt/conda_data/checkpoints-pretrain/pretrain_stage1_ocr_hard/checkpoint-29000"
# 输出目录：Stage 3 FSDP 专项输出
OUTPUT_DIR="./checkpoints-stage3/bunny-phi1.5-fsdp-ocr-v365"
# 混合数据路径
DATA_PATH="/mnt/CoBunny/dataassert/cobunny_stage2_final_mixed_ocr.json"
IMAGE_PATH="/"

# 确保输出目录存在，否则 tee 会报错
mkdir -p $OUTPUT_DIR

# ========================================================
# 3. FSDP 正式启动命令
# ========================================================
# 注意：我们使用 accelerate launch 代替 deepspeed
# 它会自动读取 /home/huangwei/.cache/huggingface/accelerate/default_config.yaml

accelerate launch \
    --config_file /home/huangwei/.cache/huggingface/accelerate/default_config.yaml \
    bunny/train/train_stage3_phi_ocr_fsdp.py \
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
    --bf16 False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none 2>&1 | tee $OUTPUT_DIR/finetunefull.log