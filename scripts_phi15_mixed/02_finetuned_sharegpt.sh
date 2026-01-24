# 阶段2：微调（包含 Recipe-1 和 Recipe-2）

#可以把 Recipe-1 和 Recipe-2 写在一个脚本里，用 && 连接，确保第一步成功后自动跑第二步：

#Recipe-1：--unfreeze_vision_tower False。先让语言模型学会多模态指令。

#Recipe-2：--unfreeze_vision_tower True。打开视觉塔，微调全链路。

#关键修正：全部统一使用 --version bunny，彻底告别 phi3。

#🎯 第二阶段（Stage 2）的核心定义：我们在练什么？
#你之前的理解部分正确，但不完全完整。在第二阶段，我们不再是简单的“训练映射层”，而是在进行一次**“三位一体”的协同进化**。

#具体来说，显存里发生的事情是这样的：

#🧠 大脑 (LLM - Phi-1.5)：

#本体：冻结 (Frozen)。

#挂件 (LoRA)：🔥 训练 (Trainable)。这是本阶段的重点。LoRA 模块插入在 LLM 的每一层中，学习如何处理复杂的指令逻辑（如“解释为什么”、“提取文字”）。

#👀 眼睛 (Vision Tower)：

#视网膜 (DINO/Oryx Backbone)：冻结 (Frozen)。保护基础视觉能力。

#神经束 (Fusion Layers 113 参数)：🔥 训练 (Trainable)。这是你独有的优势。它们必须继续进化，学会根据 LoRA 的指令需求，动态调整 DINO 和 Oryx 的融合权重（比如问颜色时多听 Oryx 的，看结构时多听 DINO 的）。

#🌉 桥梁 (Projector 4 参数)：

#本体：🔥 训练 (Trainable)。继续精调，修正 Stage 1 的“指鹿为马”现象。


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
BASE_MODEL="/mnt/conda_data/microsoft/phi-1_5"
OUTPUT_DIR="./checkpoints-finetune/bunny-phi1.5-mixed-lora-695k"
# 关键：指向 Stage 1 跑出来的那个包含 117 个 Key 的文件
PRETRAIN_ADAPTER="/mnt/CoBunny/checkpoints-pretrain/bunny-phi1.5-mixed-pretrain/checkpoint-33300/mm_projector.bin"
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
    bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero2_mixencoders_finetune.json \
    --model_name_or_path $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --version bunny \
    --data_path /data/ShareGPT4V/sharegpt4v_matched_772k.json \
    --image_folder /data/ShareGPT4V/images \
    --vision_tower mixedencoder \
    --vision_tower_dino /mnt/facebook/dinov3-convnext-large-pretrain-lvd1689m \
    --vision_tower_siglip /mnt/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter $PRETRAIN_ADAPTER \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_backbone False \
    --unfreeze_vision_tower True \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
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
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none 2>&1 | tee $OUTPUT_DIR/finetunesharegpt.log