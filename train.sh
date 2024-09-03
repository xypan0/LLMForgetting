#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

# model_and_tokenizer=meta-llama/Meta-Llama-3.1-8B
model_and_tokenizer=openai-community/gpt2

accelerate launch --config_file fsdp_config.yaml python/train.py \
    --model $model_and_tokenizer \
    --tokenizer-name $model_and_tokenizer \
    --train-data data/train_\*.json \
    --val-data data/val.json \
    --optimizer "name=adam, lr=1e-4, weight_decay=0.0" \
    --norm 0.25 \
    --bf16 \
    --pseudo_random \
    --logging_conf_file conf/common.log_conf \
    --seed 1234 \
    --max-steps 50 \
    --epoch 1 \
    --diff_norm \
    --val_batch_size 4 \
    --eval_frequency 5 \
    --response_loss_only \
    --save_dir ./test_model/ \
    --global_batch_size 16 \
    --sharegpt_format \
    --micro_batch_size 4