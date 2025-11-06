#!/bin/bash

SPLIT="test"
MODEL_TYPE=phi-2
TARGET_DIR=bunny-phi-2

python -m bunny.eval.model_vqa_mmmu \
    --model-path  /mnt/Bunny-v1_0-3B\
    --model-type $MODEL_TYPE \
    --data-path /mnt/CoBunny/MMMU \
    --config-path ./eval/mmmu/config.yaml \
    --output-path ./eval/mmmu/answers_upload/$SPLIT/$TARGET_DIR.json \
    --split $SPLIT \
    --conv-mode bunny
