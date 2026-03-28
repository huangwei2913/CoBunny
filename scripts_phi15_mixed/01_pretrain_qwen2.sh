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
MASTER_ADDR=${MASTER_ADDR:-"192.168.0.2"}
MASTER_PORT=${MASTER_PORT:-"29501"}
HOSTFILE="./script/deepspeed/hostfile"
INCLUDE_STR="192.168.0.2:2,3,4,5"
# ========================================================
# 2. 模型与架构参数
# ========================================================
MODEL_TYPE="Qwen2-1.5B"
BASE_MODEL="/mnt/conda_data/Qwen2-1.5B"
# 关键：这里传你代码中定义的逻辑开关名称
VISION_TOWER="mixedencoder" 
OUTPUT_DIR="/mnt/conda_data/checkpoints-pretrain/pretrain_stage1_qwen2"
mkdir -p $OUTPUT_DIR
# 1. 基础环境
export PYTHONUNBUFFERED=1
export DS_SKIP_CUDA_CHECK=1
export DEEPSPEED_USE_TORCH_ADAM=1

# 2. 显存管理（重点修正）
# 注意：PYTORCH_CUDA_ALLOC_CONF 只能定义一次，不要写散了
# max_split_size_mb:512 能减少显存碎片，防止莫名其妙的 OOM
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
# 3. NCCL 多机通讯优化（稳定性关键）
export NCCL_DEBUG=INFO  # 既然报错了，保持 INFO 很有必要
export NCCL_SOCKET_IFNAME=eth0 
export GLOO_SOCKET_IFNAME=eth0

# 核心稳定性补丁：如果你的网络不是昂贵的 InfiniBand (IB)，请务必禁用以下两项
export NCCL_IB_DISABLE=1      # 禁用 IB 协议，强制走以太网，防止 NCCL 找错网卡
export NCCL_P2P_DISABLE=1     # 在普通以太网或虚拟化网络中，P2P 通讯极易崩溃

# 超时控制：2M 数据启动慢，增加超时阈值防止没跑就开始报错
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=12000     # 增加到 12000 秒
export NCCL_ASYNC_ERROR_HANDLING=1
# ========================================================
# 3. 启动训练 (使用 DeepSpeed)
# ========================================================
# 注意：Pretrain 阶段通常建议使用 Zero-2 性能更佳，显存极度紧张才用 Zero-3
#这个地方即便是用bunny也是没有关系的
deepspeed \
    --hostfile $HOSTFILE \
    --include "$INCLUDE_STR" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    bunny/train/train_stage1_qwen2.py \
    --deepspeed ./script/deepspeed/zero2_mixencoders_pretraing_qwen2.json \
    --model_name_or_path $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --version bunny \
    --vision_tower $VISION_TOWER \
    --vision_tower_dino /mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m \
    --vision_tower_siglip /mnt/siglip-so400m-patch14-384 \
    --data_path /mnt/conda_data/Bunny-v1.1-data/pretrain/bunny_stage1_cleaned.json \
    --image_folder /mnt/conda_data/Bunny-v1.1-data/pretrain/images \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_backbone True \
    --bf16 False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10 \
    --load_best_model_at_end True \
    --learning_rate 2e-4 \
    --max_grad_norm 0.5 \
    --lr_scheduler_type  "cosine"\
    --logging_steps 10 \
    --warmup_ratio 0.1  \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --greater_is_better False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --group_by_modality_length True \
    --dataloader_num_workers 16 \
    --report_to none 2>&1 | tee $OUTPUT_DIR/pretrain.log