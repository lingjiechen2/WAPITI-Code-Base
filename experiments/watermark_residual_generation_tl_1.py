import argparse
import os
import copy
os.environ['http_proxy'] = "http://10.176.52.116:7890"
os.environ['https_proxy'] = "http://10.176.52.116:7890"
os.environ['all_proxy'] = "socks5://10.176.52.116:7891"
import sys
# sys.path.append(('../'))
# sys.path.append(('../../'))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import json
from typing import Dict
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from task_vector import TaskVector
from itertools import chain
from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType
import logging
from transformer_lens import HookedTransformer
import torch.nn.functional as F


logging.basicConfig(
    filename='/remote-home/miintern1/watermark-learnability/logs/residual_generation.txt',  # Specify the log file name
    level=logging.INFO,          # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Set the log message format
)

final_result_folder = '/remote-home/miintern1/watermark-learnability/data/c4/'

max_length = 250
min_length = 250
# num_samples = 512
# batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
# dataset = load_dataset("allenai/c4", "realnewslike", split="validation", streaming=False)
os.environ['HF_DATASETS_CACHE'] = '/remote-home/miintern1/watermark-learnability/experiments/.cache/huggingface/dataset/'
dataset = load_dataset('allenai/c4', 'realnewslike', split='validation')


block_size = 512
batch_size = 16
# Initialize lists to hold the tokenized and grouped data
all_input_ids = []
all_attention_masks = []
analyze_layers = [f'blocks.{i}.hook_resid_post' for i in range(28,32)]

# Tokenize and concatenate texts
for example in tqdm(dataset):
    text = example["text"]
    # Tokenize the text
    tokenized_text = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=block_size)
    
    # Add the tokenized text to the lists
    all_input_ids.append(tokenized_text['input_ids'].squeeze().tolist())
    all_attention_masks.append(tokenized_text['attention_mask'].squeeze().tolist())


# Ensure the total length is a multiple of block_size
all_input_ids = list(chain(*all_input_ids))
all_attention_masks = list(chain(*all_attention_masks))

total_length = (len(all_input_ids) // block_size) * block_size
all_input_ids = all_input_ids[:total_length]
all_attention_masks = all_attention_masks[:total_length]

# Split the tokenized texts into chunks of block_size
grouped_input_ids = [all_input_ids[i:i + block_size] for i in range(0, total_length, block_size)]
grouped_attention_masks = [all_attention_masks[i:i + block_size] for i in range(0, total_length, block_size)]

# Create batches of size batch_size
batched_input_ids = [grouped_input_ids[i:i + batch_size] for i in range(0, len(grouped_input_ids), batch_size)]
batched_attention_masks = [grouped_attention_masks[i:i + batch_size] for i in range(0, len(grouped_attention_masks), batch_size)]


batched_input_ids = [torch.tensor(batch) for batch in batched_input_ids]
batched_attention_masks = [torch.tensor(batch) for batch in batched_attention_masks]
logging.info("Data tokenized and grouped successfully.")

print(f"{batched_input_ids[0].shape=}, {batched_attention_masks[0].shape=}")

# vanilla_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map = 'cpu')
# model = HookedTransformer.from_pretrained('Llama-2-7b', device="gpu", hf_model=vanilla_model)
# model.eval()
# logging.info("Model and tokenizer loaded successfully.")

# vanilla_outputs = dict()
# vanilla_logits = []

# test_time = 10
# test_index = 0
# progress_bar = tqdm(total=len(batched_input_ids), desc="Processing batches", leave=True)
# for input_ids, attention_mask in zip(batched_input_ids, batched_attention_masks):
#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#     with torch.no_grad():
#         tl_logits, cache = model.run_with_cache(input_ids, return_type='logits')
#     vanilla_logits.append(tl_logits[:, -1, :].cpu())
#     for layer in analyze_layers:
#         if layer not in vanilla_outputs:
#             vanilla_outputs[layer] = []
#         vanilla_outputs[layer].append(cache[layer].cpu())
#     del tl_logits, cache
#     torch.cuda.empty_cache()
#     progress_bar.update(1)

#     test_index += 1
#     if test_index == test_time:
#         break

# vanilla_logits = torch.cat(vanilla_logits, dim=0)
# logging.info("Vanilla residuals generated successfully.")
# del vanilla_model
# torch.cuda.empty_cache()
# torch.save(vanilla_outputs, final_result_folder + "tl_vanilla_residuals.pt")
# torch.save(vanilla_logits, final_result_folder + "tl_vanilla_logits.pt")

watermark_config = {
    #  "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
    #  "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
    #  "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2":{"type": "kgw", "k": 2, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_2", "kgw_device": "cpu"},
}

watermark_outputs_dict = torch.load(final_result_folder + "tl_watermark_residuals.pt") if os.path.exists(final_result_folder + "tl_watermark_residuals.pt") else dict()
watermark_logits_dict = torch.load(final_result_folder + "tl_watermark_logits.pt") if os.path.exists(final_result_folder + "tl_watermark_logits.pt") else dict()


for watermark_name, watermark_config in watermark_config.items():
    if watermark_name in watermark_outputs_dict:
        logging.info(f"Watermark logits for {watermark_name[40:]} already generated.")
        print(f"Watermark logits for {watermark_name[40:]} already generated.")
        continue
    watermarked_model = AutoModelForCausalLM.from_pretrained(watermark_name, device_map = 'auto')   
    logging.info(f"Watermarked model {watermark_name[40:]} loaded successfully.")
    watermark_outputs = dict()
    watermark_logits = []
    
    progress_bar = tqdm(total=len(batched_input_ids), desc="Processing batches", leave=True)
    for input_ids, attention_mask in zip(batched_input_ids, batched_attention_masks):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            tl_watermark_logits, cache = watermarked_model.run_with_cache(input_ids, return_type='logits')
        watermark_logits.append(tl_watermark_logits[:, -1, :].cpu())
        for layer in analyze_layers:
            if layer not in watermark_outputs:
                watermark_outputs[layer] = []
            watermark_outputs[layer].append(cache[layer].cpu())
        del tl_watermark_logits, cache
        torch.cuda.empty_cache()
        progress_bar.update(1)


    watermark_outputs_dict[watermark_name] = watermark_outputs
    watermark_logits_dict[watermark_name] = torch.cat(watermark_logits, dim=0)
    del watermarked_model
    torch.cuda.empty_cache()
    progress_bar.close()
    logging.info(f"Watermark logits for {watermark_name[40:]} generated successfully.")

    torch.save(watermark_outputs_dict, final_result_folder + "tl_watermark_residuals_1.pt")
    torch.save(watermark_logits_dict, final_result_folder + "tl_watermark_logits_1.pt")
