# 阶段1：预训练（只练投影层 Projector）
#任务：建立混合编码器（Mixed Vision Tower）与 Phi-1.5 的联系。
#关键点：--version plain。
# 注意：一定要在这里解决你之前的 IncompatibleKeys 报错

#!/bin/bash

# ========================================================
# 1. 硬件与分布式环境配置 (支持多卡加速)
# ========================================================
# 如果是单机多卡，DeepSpeed 会自动识别。如果有 hostfile 请取消注释。
# HOSTFILE="./script/deepspeed/hostfile"
MASTER_ADDR=${MASTER_ADDR:-"192.168.0.3"}
MASTER_PORT=${MASTER_PORT:-"29501"}
HOSTFILE="./script/deepspeed/hostfile"
INCLUDE_STR="192.168.0.3:0,1,2,3,4,5,6,7@192.168.0.2:2,3,4,5,6,7"
# ========================================================
# 2. 模型与架构参数
# ========================================================
MODEL_TYPE="phi-1.5"
BASE_MODEL="/mnt/conda_data/microsoft/phi-1_5"
# 关键：这里传你代码中定义的逻辑开关名称
VISION_TOWER="mixedencoder" 
OUTPUT_DIR="./checkpoints-pretrain/bunny-phi1.5-mixed-pretrain"
mkdir -p $OUTPUT_DIR
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1
export DEEPSPEED_USE_TORCH_ADAM=1
export NCCL_DEBUG=INFO  # 开启调试模式，这样卡住时能看到为什么卡
export NCCL_SOCKET_IFNAME=eth0 
export GLOO_SOCKET_IFNAME=eth0
# ========================================================
# 3. 启动训练 (使用 DeepSpeed)
# ========================================================
# 注意：Pretrain 阶段通常建议使用 Zero-2 性能更佳，显存极度紧张才用 Zero-3
deepspeed \
    --hostfile $HOSTFILE \
    --include "$INCLUDE_STR" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero2_mixencoders_pretraing.json \
    --model_name_or_path $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --version plain \
    --vision_tower $VISION_TOWER \
    --vision_tower_dino /mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m \
    --vision_tower_siglip /mnt/siglip-so400m-patch14-384 \
    --data_path /mnt/conda_data/Bunny-v1.1-data/pretrain/bunny_pretrain_laion_2m.json \
    --image_folder /mnt/conda_data/Bunny-v1.1-data/pretrain/images \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_backbone True \
    --bf16 False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "steps" \
    --eval_steps 5000 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --learning_rate 2e-4 \
    --max_grad_norm 0.5 \
    --lr_scheduler_type  "cosine"\
    --logging_steps 20 \
    --warmup_ratio 0.1  \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --greater_is_better False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to none 2>&1 | tee $OUTPUT_DIR/pretrain.log