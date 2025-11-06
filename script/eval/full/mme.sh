#!/bin/bash
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

MODEL_TYPE=phi-2
TARGET_DIR=bunny-phi-2
#在这里我们先用训练好的模型来看下效果
python -m bunny.eval.model_vqa_loader_mme \
    --model-path /mnt/Bunny-v1_0-3B \
    --model-type $MODEL_TYPE \
    --image-folder ./eval/mme/MME_Benchmark_release_version/MME_Benchmark \
    --question-file ./eval/mme/bunny_mme.jsonl \
    --answers-file ./eval/mme/answers/$TARGET_DIR.jsonl \
    --temperature 0 \
    --conv-mode bunny

#cd ./eval/mme

#python convert_answer_to_mme.py --experiment $TARGET_DIR

#python calculation_mme.py --results_dir answers_upload/$TARGET_DIR \
#| tee 2>&1 answers_upload/$TARGET_DIR/res.txt
#也就是说，如果以本地路径中的模型来进行推理的，模型有关的参数都定义在类似于config.json文件中
#因此，我们在加载视觉塔模型的时候，需要极其注意这个细节，要修改config.json中的mm_vision_tower参数

# (base) huangwei@huangwei-System-Product-Name:~/Bunny-v1_0-3B$ ll
# total 6220500
# drwxrwxr-x  3 huangwei huangwei       4096 10月 10 14:19 ./
# drwxr-xr-x 77 huangwei huangwei       4096 10月 10 22:46 ../
# -rw-rw-r--  1 huangwei huangwei       1080 10月 10 11:54 added_tokens.json
# -rw-rw-r--  1 huangwei huangwei    1532921 10月 10 11:54 comparison.png
# -rw-rw-r--  1 huangwei huangwei       1272 10月 10 11:54 config.json
# -rw-rw-r--  1 huangwei huangwei      11604 10月 10 11:54 configuration_bunny_phi.py
# -rw-rw-r--  1 huangwei huangwei        119 10月 10 11:54 generation_config.json
# -rw-rw-r--  1 huangwei huangwei     456318 10月 10 11:54 merges.txt
# -rw-rw-r--  1 huangwei huangwei 4990950824 10月 10 12:03 model-00001-of-00002.safetensors
# -rw-rw-r--  1 huangwei huangwei 1373673760 10月 10 12:06 model-00002-of-00002.safetensors
# -rw-rw-r--  1 huangwei huangwei     102157 10月 10 14:19 modeling_bunny_phi.py
# -rw-rw-r--  1 huangwei huangwei      88677 10月 10 11:54 model.safetensors.index.json
# drwxrwxr-x  2 huangwei huangwei       4096 10月 10 14:19 __pycache__/
# -rw-rw-r--  1 huangwei huangwei        441 10月 10 12:06 special_tokens_map.json
# -rw-rw-r--  1 huangwei huangwei        370 10月 10 14:17 test.py
# -rw-rw-r--  1 huangwei huangwei       7339 10月 10 12:06 tokenizer_config.json
# -rw-rw-r--  1 huangwei huangwei    2114924 10月 10 12:06 tokenizer.json
# -rw-rw-r--  1 huangwei huangwei     798156 10月 10 12:06 vocab.json
# (base) huangwei@huangwei-System-Product-Name:~/Bunny-v1_0-3B$ cat config.json
# {
#   "_name_or_path": "BAAI/bunny-phi-2-siglip",
#   "architectures": [
#     "BunnyPhiForCausalLM"
#   ],
#   "attention_dropout": 0.0,
#   "auto_map": {
#     "AutoConfig": "configuration_bunny_phi.BunnyPhiConfig",
#     "AutoModelForCausalLM": "modeling_bunny_phi.BunnyPhiForCausalLM"
#   },
#   "bos_token_id": 50256,
#   "embd_pdrop": 0.0,
#   "eos_token_id": 50256,
#   "freeze_mm_mlp_adapter": false,
#   "hidden_act": "gelu_new",
#   "hidden_size": 2560,
#   "image_aspect_ratio": "pad",
#   "initializer_range": 0.02,
#   "intermediate_size": 10240,
#   "layer_norm_eps": 1e-05,
#   "max_position_embeddings": 2048,
#   "mm_hidden_size": 1152,
#   "mm_projector_lr": 2e-05,
#   "mm_projector_type": "mlp2x_gelu",
#   "mm_vision_tower": "google/siglip-so400m-patch14-384",
#   "model_type": "bunny-phi",
#   "num_attention_heads": 32,
#   "num_hidden_layers": 32,
#   "num_key_value_heads": 32,
#   "pad_token_id": 50256,
#   "partial_rotary_factor": 0.4,
#   "qk_layernorm": false,
#   "resid_pdrop": 0.1,
#   "rope_scaling": null,
#   "rope_theta": 10000.0,
#   "tie_word_embeddings": false,
#   "tokenizer_model_max_length": 2048,
#   "tokenizer_padding_side": "right",
#   "torch_dtype": "float16",
#   "transformers_version": "4.36.2",
#   "tune_mm_mlp_adapter": false,
#   "use_cache": true,
#   "use_mm_proj": true,
#   "vocab_size": 50295
# }
# (base) huangwei@huangwei-System-Product-Name:~/Bunny-v1_0-3B$ 
