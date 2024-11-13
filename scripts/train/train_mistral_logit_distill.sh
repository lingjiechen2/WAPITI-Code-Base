#!/bin/bash
watermark= $1
out_dir=$2
port= $3
mistral= ${4:-"mistralai/Mistral-7B-v0.3"}

model_name="Mistral-7B-v0.3-logit-watermark-distill-${watermark}"

if [ "$watermark" = "aar-k2" ]; then
    watermark_args="--watermark_type aar --aar_watermark_k 2"
elif [ "$watermark" = "aar-k3" ]; then
    watermark_args="--watermark_type aar --aar_watermark_k 3"
elif [ "$watermark" = "aar-k4" ]; then
    watermark_args="--watermark_type aar --aar_watermark_k 4"
elif [ "$watermark" = "kgw-k0-gamma0.25-delta1" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 1.0 \
    --kgw_watermark_seeding_scheme simple_0"
elif [ "$watermark" = "kgw-k0-gamma0.25-delta2" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 2.0 \
    --kgw_watermark_seeding_scheme simple_0"
elif [ "$watermark" = "kgw-k1-gamma0.25-delta1" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 1.0 \
    --kgw_watermark_seeding_scheme simple_1"
elif [ "$watermark" = "kgw-k1-gamma0.25-delta2" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 2.0 \
    --kgw_watermark_seeding_scheme simple_1"
elif [ "$watermark" = "kgw-k2-gamma0.25-delta2" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 2.0 \
    --kgw_watermark_seeding_scheme simple_2"
elif [ "$watermark" = "kth-shift1" ]; then
    watermark_args="--watermark_type kth \
    --kth_watermark_key_len 256 \
    --kth_watermark_num_shifts 1"
elif [ "$watermark" = "kth-shift2" ]; then
    watermark_args="--watermark_type kth \
    --kth_watermark_key_len 256 \
    --kth_watermark_num_shifts 2"
elif [ "$watermark" = "kth-shift4" ]; then
    watermark_args="--watermark_type kth \
    --kth_watermark_key_len 256 \
    --kth_watermark_num_shifts 4"
elif [ "$watermark" = "kth-shift256" ]; then
    watermark_args="--watermark_type kth \
    --kth_watermark_key_len 256 \
    --kth_watermark_num_shifts 256"
else
    echo "Unsupported watermark type ${watermark}."
    exit 1
fi

if [[ "$watermark" == kth* ]]; then
    batch_size=32
    block_size=256
else
    batch_size=16
    block_size=512
fi

srun -p a800 --gres=gpu:2 --cpus-per-task=12 --mem-per-cpu=8G --pty \
torchrun --nproc_per_node=2 --master_port=29544  train_logit_distill.py \
# torchrun --nproc_per_node=4 --master_port=${port} train_logit_distill.py \
    --model_name_or_path "${mistral}" \
    --dataset_name Skylion007/openwebtext \
    --streaming \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --do_train \
    --max_steps 5000 \
    --logging_steps 1 \
    --output_dir "${out_dir}${model_name}" \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 500 \
    --block_size ${block_size} \
    --save_steps 1000 \
    --save_total_limit 1 \
    --tf32 True \
    --bf16 True \
    --gradient_checkpointing True \
    ${watermark_args} \
    --watermark_seed 42 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "MistralDecoderLayer"




srun -p a800 --gres=gpu:2 --cpus-per-task=12 --mem-per-cpu=8G --pty \
torchrun --nproc_per_node=2 --master_port=29544  train_logit_distill.py \
    --model_name_or_path mistralai/Mistral-7B-v0.3 \
    --dataset_name Skylion007/openwebtext \
    --streaming \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --max_steps 5000 \
    --logging_steps 1 \
    --output_dir /remote-home1/miintern1/watermark-learnability/data/mistral-7B-v0.3/Mistral-7B-v0.3-logit-watermark-distill-kgw-k0-gamma0.25-delta2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 500 \
    --block_size 512 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --tf32 True \
    --bf16 True \
    --gradient_checkpointing True \
    --watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 2.0 \
    --kgw_watermark_seeding_scheme simple_0 \
    --watermark_seed 42 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "MistralDecoderLayer"
--output=mistral.log