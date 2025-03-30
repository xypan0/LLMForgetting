#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

# model_and_tokenizer=Qwen/Qwen2-7B
# model_and_tokenizer=google/gemma-2-9b
model_and_tokenizer=meta-llama/Meta-Llama-3.1-8B
# model_and_tokenizer=openai-community/gpt2
# python3 python/train.py \

# accelerate launch --config_file fsdp_config.yaml python/train.py \
python3 python/train.py \
    --model $model_and_tokenizer \
    --tokenizer-name $model_and_tokenizer \
    --train-data data_lmflow/train_\*.json \
    --val-data data_lmflow/val.json \
    --optimizer "name=adam, lr=1e-1, weight_decay=0.0" \
    --bf16 \
    --norm 1e8 \
    --warmup-ratio 0.00 \
    --model-type Llama \
    --pseudo_random \
    --logging_conf_file conf/common.log_conf \
    --seed 1234 \
    --max-steps 400 \
    --max-length 1024 \
    --epoch 1 \
    --val_batch_size 4 \
    --eval_frequency 500000 \
    --response_loss_only \
    --save_dir ./test_model/ \
    --global_batch_size 2 \
    --chat_template llama3 \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --lora_target_modules "q_proj", "v_proj" \
    --lora_bias "none" \
    --lora_task_type CAUSAL_LM \
    --clip_grad_norm 1.0 \
    --lora \
    --lmflow-format \
    --micro_batch_size 1