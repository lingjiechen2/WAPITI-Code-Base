import argparse
import os
import re
import copy
os.environ['http_proxy'] = "http://10.176.58.101:7890"
os.environ['https_proxy'] = "http://10.176.58.101:7890"
os.environ['all_proxy'] = "socks5://10.176.58.101:7891"
import sys
sys.path.append(('../'))
sys.path.append(('../../'))
sys.path.append('/remote-home1/miintern1/watermark-learnability')
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import json
import time
from typing import Dict
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformer_lens import HookedTransformer
from task_vector import TaskVector
import plotly.express as px
from safetensors import safe_open

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType

from huggingface_hub import login
login(token="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")
from transformers import AutoConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
watermark_name = 'kgw-k0-gamma0.25-delta2'
watermarked_model_name = "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# dataset = load_dataset("allenai/c4", "realnewslike", "validation")
dataset = load_dataset("allenai/c4", "realnewslike", split="validation")

max_length = 250
min_length = 250
num_samples = 512
batch_size = 16

def filter_length(example):
        return len(tokenizer(example['text'], truncation=True, max_length=max_length)["input_ids"]) >= min_length

def encode(examples):
    trunc_tokens = tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    # Examples are truncated to max_length, which comprises the possible generation prompt and the text to be generated
    examples["text"] = tokenizer.batch_decode(trunc_tokens["input_ids"], skip_special_tokens=True)
    prompt = tokenizer(
        examples["text"], truncation=True, padding=True, max_length=50, return_tensors="pt",
    ).to(device)
    examples["prompt_text"] = tokenizer.batch_decode(prompt["input_ids"], skip_special_tokens=True)
    examples["input_ids"] = prompt["input_ids"]
    examples["attention_mask"] = prompt["attention_mask"]
    examples["text_completion"] = tokenizer.batch_decode(
        trunc_tokens["input_ids"][:, 50:], skip_special_tokens=True
    )
    return examples

dataset = dataset.filter(filter_length)
# Set how many samples will be skipped
dataset = dataset.map(encode, batched=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size)



prompts = []
human_text = []
prompt_text = []
full_human_text = []
for batch in dataloader:
    if len(human_text) >= num_samples:
        break
    if (type(batch["input_ids"]) == list):
        batch["input_ids"] = torch.stack(batch["input_ids"], dim=1).to(device)
    if (type(batch["attention_mask"]) == list):
        batch["attention_mask"] = torch.stack(batch["attention_mask"], dim=1).to(device)
    prompts.append(batch)
    human_text.extend(batch["text_completion"])
    prompt_text.extend(batch["prompt_text"])
    full_human_text.extend(batch["text"])
human_text = human_text[:num_samples]
prompt_text = prompt_text[:num_samples]
full_human_text = full_human_text[:num_samples]
raw_input = {
    "prompts": prompts,
    "human_text": human_text,
    "prompt_text": prompt_text,
    "full_human_text": full_human_text,
}


DO_SAMPLE = True
temperature=1.0
top_p=0.9
top_k=0


watermark_configs = {
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2":{"type": "kgw", "k": 2, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_2", "kgw_device": "cpu"},
}

watermark_config = watermark_configs[watermarked_model_name]
detector = WatermarkDetector(
                              device=watermark_config.get("kgw_device", 'cpu'),
                              tokenizer=tokenizer,
                              vocab=tokenizer.get_vocab().values(),
                              gamma=watermark_config["gamma"],
                              seeding_scheme=watermark_config["seeding_scheme"],
                              normalizers=[]
                         )


original_model_name = "OnAnOrange/llama-gms8k-watermarked-model"
save_path = f'/remote-home1/miintern1/watermark-learnability/data/refinetuning/correct_gms8k_watermark_distillation_{watermark_name}.json'
base_path = '/remote-home1/miintern1/watermark-learnability/data/refinetuning/llama-2-7b-sampling-watermark-distill-kgw-k0-gamma0.25-delta2/checkpoint-8500'

if os.path.exists(save_path):
    with open(save_path, 'r') as f:
        refinetuning_watermark_ability_decay = json.load(f)
else:
    refinetuning_watermark_ability_decay = {}

if "original_model" not in refinetuning_watermark_ability_decay:
    generation_outputs = []
    watermarked_scores = []
    original_model = AutoModelForCausalLM.from_pretrained(original_model_name)
    original_model.to(device)
    for batch in tqdm(prompts):
        watermarked_output = original_model.generate(
            input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                do_sample=DO_SAMPLE,
                                min_new_tokens=200,
                                max_new_tokens=200,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                pad_token_id=tokenizer.eos_token_id,
                            )
        watermarked_text = tokenizer.batch_decode(watermarked_output, skip_special_tokens=True)
        generation_outputs.extend(watermarked_text)
        watermark_score = [detector.detect(sample)['p_value'] for sample in watermarked_text]
        watermarked_scores.extend(watermark_score)
        # print(len(watermarked_scores))
        # print(len(watermarked_text))
        # print(watermarked_scores)
        # break
        refinetuning_watermark_ability_decay['original_model'] = {
            # "prompts": prompts[:num_samples],
            "generation_outputs": generation_outputs,
            "watermarked_scores": watermarked_scores,
        }
    del original_model

pattern = re.compile(r'model-step-\d+')
matched_folders = []
for root, dirs, files in os.walk(base_path):
    for dir_name in dirs:
        if pattern.match(dir_name):
            full_path = os.path.join(root, dir_name)
            matched_folders.append(full_path)

for sharded_weights_dir in matched_folders:
    if sharded_weights_dir in refinetuning_watermark_ability_decay:
        print(f"Skipping {sharded_weights_dir}")
        continue
    print(f"Now processing {sharded_weights_dir}")
    refinetuning_watermark_ability_decay[sharded_weights_dir] = {}
    shard_filenames = [f for f in os.listdir(sharded_weights_dir) if f.startswith('model-') and f.endswith('.safetensors')]
    shard_filenames.sort()
    full_state_dict = {}
    for shard_filename in shard_filenames:
        shard_state_dict = {}
        shard_path = os.path.join(sharded_weights_dir, shard_filename)
        with safe_open(shard_path, framework='pt', device = 'cpu') as f:
            for k in f.keys():
                shard_state_dict[k] = f.get_tensor(k)
        full_state_dict.update(shard_state_dict)

    config = AutoConfig.from_pretrained(watermarked_model_name)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(full_state_dict)
    model.to(device)

    generation_outputs = []
    watermarked_scores = []
    for batch in tqdm(prompts):
        watermarked_output = model.generate(
            input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                do_sample=DO_SAMPLE,
                                min_new_tokens=200,
                                max_new_tokens=200,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                pad_token_id=tokenizer.eos_token_id,
                            )
        watermarked_text = tokenizer.batch_decode(watermarked_output, skip_special_tokens=True)
        generation_outputs.extend(watermarked_text)
        watermark_score = [detector.detect(sample)['p_value'] for sample in watermarked_text]
        watermarked_scores.extend(watermark_score)
        # print(len(watermarked_scores))
        # print(len(watermarked_text))
        # print(watermarked_scores)
        # break

    refinetuning_watermark_ability_decay[sharded_weights_dir] = {
        # "prompts": prompts[:num_samples],
        "generation_outputs": generation_outputs,
        "watermarked_scores": watermarked_scores,
    }
    del model
with open(save_path, 'w') as f:
    json.dump(refinetuning_watermark_ability_decay, f, indent=4)