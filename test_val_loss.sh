#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

# model_and_tokenizer=Qwen/Qwen2-7B
# model_and_tokenizer=google/gemma-2-9b
model_and_tokenizer=meta-llama/Meta-Llama-3.1-8B
# model_and_tokenizer=openai-community/gpt2
# python3 python/train.py \

accelerate launch --config_file fsdp_config.yaml python/train.py \
    --model $model_and_tokenizer \
    --use_wandb \
    --wandb_run_name test_val_loss_clip1 \
    --tokenizer-name $model_and_tokenizer \
    --train-data data_lmflow/train_\*.json \
    --val-data data_lmflow/test.json \
    --optimizer "name=adamw, lr=1e-5, weight_decay=0.0" \
    --bf16 \
    --warmup-ratio 0.03 \
    --model-type Llama \
    --pseudo_random \
    --logging_conf_file conf/common.log_conf \
    --seed 1234 \
    --max-length 1024 \
    --epoch 1 \
    --val_batch_size 4 \
    --eval_frequency 5 \
    --response_loss_only \
    --save_dir ./test_model/ \
    --global_batch_size 64 \
    --lmflow-format \
    --chat_template llama3 \
    --micro_batch_size 4 \
    --clip_grad_norm 1.0 

    # --max-steps 400 \