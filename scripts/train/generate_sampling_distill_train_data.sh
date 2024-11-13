#!/bin/bash
watermark=$1
llama=${2:-"meta-llama/Llama-2-7b-hf"}
name=$3

output_file="data/sampling-distill-train-data-dicts/${watermark}_${name}_owt_len256_640k_samples_dict.json"
output_train_file="data/sampling-distill-train-data/sampling-distill-train-data-${watermark}.json"
watermark_config_file="experiments/watermark-configs/${watermark}-config.json"

echo "Watermark used: $watermark"
echo "Llama model used: $llama"

srun -p a800 --gres=gpu:1 --cpus-per-task=12 --mem-per-cpu=8G --pty python experiments/generate_sampling_distill_train_data.py \
    --model_name "${llama}" \
    --dataset_name Skylion007/openwebtext \
    --dataset_split train \
    --data_field "text" \
    --streaming \
    --output_file "${output_file}" \
    --output_train_file "${output_train_file}" \
    --num_samples 256000 \
    --min_new_tokens 256 \
    --max_new_tokens 256 \
    --prompt_length 50 \
    --seed 42 \
    --watermark_config_file "${watermark_config_file}" \
    --save_interval 64000 \
    --fp16 \
    --dataloader_batch_size 10000 \
    --batch_size 128
