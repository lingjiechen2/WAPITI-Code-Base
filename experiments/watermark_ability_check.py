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
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, LogitsProcessorList
from transformer_lens import HookedTransformer
from task_vector import TaskVector
import plotly.express as px
from safetensors import safe_open
import argparse

from watermarks.kgw.watermark_processor import WatermarkLogitsProcessor
from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.aar.aar_watermark import AarWatermark
from watermarks.watermark_types import WatermarkType

from huggingface_hub import login
login(token="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")
from transformers import AutoConfig

parser = argparse.ArgumentParser(description='Watermark ability check')
parser.add_argument('--save_name', type=str, required=True, help='The name to save the file as')
parser.add_argument('--tested_model_name', type=str, required=True, help='The name of the tested model')
parser.add_argument('--watermark_type', type=str, required=True, help='The type of watermark to use')
args = parser.parse_args()
tested_model_name = args.tested_model_name
save_name = args.save_name
watermark_type = args.watermark_type
print(f"Tested model name: {tested_model_name}")
print(f"Watermark type: {watermark_type}")
print(f"Save name: {save_name}")

device = 'cuda' # if torch.cuda.is_available() else 'cpu'
# llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("cygu/pythia-1.4b-sampling-watermark-distill-kgw-k0-gamma0.25-delta1")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b ")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# dataset = load_dataset("allenai/c4", "realnewslike", "validation")
dataset = load_dataset("allenai/c4", "realnewslike", split="validation")


DO_SAMPLE = True
temperature=1.0
top_p=0.9
top_k=0
max_length = 250
min_length = 250
num_samples = 512
batch_size = 16


def load_model_from_safetensors(sharded_weights_dir, model_name, device='cpu'):
    """
    Loads a model from sharded SafeTensors files.

    Args:
        sharded_weights_dir (str): Directory containing sharded SafeTensors files.
        model_name (str): The name or path of the model configuration to use.
        device (str): Device to load the model on (default is 'cpu').

    Returns:
        model: The loaded model with the combined state_dict.
    """
    # Get list of SafeTensors files in the specified directory
    shard_filenames = [f for f in os.listdir(sharded_weights_dir) if f.startswith('model-') and f.endswith('.safetensors')]
    shard_filenames.sort()  # Ensure files are loaded in order
    full_state_dict = {}  # To store the complete state dict
    print(f"Found {len(shard_filenames)} shard files")

    for shard_filename in shard_filenames:
        shard_state_dict = {}
        shard_path = os.path.join(sharded_weights_dir, shard_filename)
        with safe_open(shard_path, framework='pt', device='cpu') as f:
            for k in f.keys():
                shard_state_dict[k] = f.get_tensor(k)
        full_state_dict.update(shard_state_dict)

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(full_state_dict)
    model.to(device)
    return model

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

if watermark_type == "kgw":
    # watermark_configs = {
    # "cygu/pythia-1.4b-sampling-watermark-distill-kgw-k0-gamma0.25-delta1":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
    # "cygu/pythia-1.4b-sampling-watermark-distill-kgw-k0-gamma0.25-delta2":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
    # "cygu/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta1":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
    # "cygu/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
    # "cygu/pythia-1.4b-sampling-watermark-distill-kgw-k2-gamma0.25-delta2":{"type": "kgw", "k": 2, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_2", "kgw_device": "cpu"},
    # }
    watermark_configs = {
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_0", "kgw_device": "cuda"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_0", "kgw_device": "cuda"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_1", "kgw_device": "cuda"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_1", "kgw_device": "cuda"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2":{"type": "kgw", "k": 2, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_2", "kgw_device": "cuda"},
}
elif watermark_type =='aar':
    # watermark_configs = {
    # "cygu/pythia-1.4b-sampling-watermark-distill-aar-k2":{"type": "aar", "k": 2, "seed": 42},
    # "cygu/pythia-1.4b-sampling-watermark-distill-aar-k3":{"type": "aar", "k": 3, "seed": 42},
    # "cygu/pythia-1.4b-sampling-watermark-distill-aar-k4":{"type": "aar", "k": 4, "seed": 42},
    # }
    watermark_configs = {
        "cygu/llama-2-7b-logit-watermark-distill-aar-k2":{"type": "aar", "k": 2, "seed": 42},
        "cygu/llama-2-7b-logit-watermark-distill-aar-k3":{"type": "aar", "k": 3, "seed": 42},
        "cygu/llama-2-7b-logit-watermark-distill-aar-k4":{"type": "aar", "k": 4, "seed": 42},
    }
else:
    raise ValueError(f"Unknown watermark type {watermark_type}")

def move_to_device(batch, device):
    """Move batch to the specified device."""
    new_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            new_batch[key] = value.to(device)
        elif isinstance(value, list):
            # Assuming lists are lists of tensors, move each tensor to the device
            new_batch[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        else:
            new_batch[key] = value
    return new_batch


def compute_p_value(samples, detector, type='kgw'):
    score_list = []
    for s in tqdm(samples):
        if s == "":
            continue
        score = detector.detect(s)
        if type=='kgw':
            if 'p_value' in score:
                score_list.append(score['p_value'])
        elif type=='aar':
            score_list.append(score)
        else:
            print(f"Error in computing p-value for {s}")
    return score_list

full_result = {}


def compute_seq_rep_n(samples, tokenizer, n=3):
    """compute seq-rep-n metric"""
    n_gram_reps = []
    
    for s in samples:
        n_grams = []
        tokens = tokenizer(s, add_special_tokens=False).input_ids
        for i in range(len(tokens)):
            if i <= len(tokens) - n:
                n_grams.append(tuple(tokens[i:i + n]))
        if len(n_grams) == 0:
            continue
        rep = 1 - len(set(n_grams)) / len(n_grams)
        n_gram_reps.append(rep)
            
    median_rep = np.median(n_gram_reps)
    mean_rep = np.mean(n_gram_reps)
    return {
        f"median_seq_rep_{n}": median_rep,
        f"mean_seq_rep_{n}": mean_rep,
        f"list_seq_rep_{n}": n_gram_reps,
    }


def compute_total_rep_n(samples, tokenizer, n=3):
    """compute total-rep-n metric"""
    n_grams = []
    
    for s in samples:
        tokens = tokenizer(s, add_special_tokens=False).input_ids
        for i in range(len(tokens)):
            if i <= len(tokens) - n:
                n_grams.append(tuple(tokens[i:i + n]))

    total_rep = 1 - len(set(n_grams)) / len(n_grams)        

    return {f"total_rep_{n}": total_rep}


def compute_repetition(samples_dict, tokenizer):
    """Compute repetition metrics."""
    samples = samples_dict['watermarked_output']
    samples_dict.update(compute_seq_rep_n(samples, tokenizer, n=3))
    samples_dict.update(compute_total_rep_n(samples, tokenizer, n=3))
    # print(f"Model name: {model_name}\nMedian seq rep 3: {samples['median_seq_rep_3']}\nTotal rep 3: {samples['total_rep_3']}")
    return f"Median seq rep 3: {samples_dict['median_seq_rep_3']}\nTotal rep 3: {samples_dict['total_rep_3']}"


def compute_ppl(samples_dict, prompts,  tokenizer, model, batch_size, fp16=True):
    """Compute perplexities under `ppl_model_name`."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model.device.type != device:
        original_device = model.device
        model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    samples = samples_dict["full_watermarked_output"]

    for i in tqdm(range(0, len(samples), batch_size)):
        s = samples[i:i + batch_size]
        encodings = tokenizer(
            s,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_batch = encodings["input_ids"]
        attn_mask = encodings["attention_mask"]

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        prompt_text = prompts[i:i + batch_size]
        # print(prompt_text)
        # print(type(prompt_text))
        # print(len(prompt_text))
        
        prompt_encodings = tokenizer(
            prompt_text,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        prompt_attn_mask = prompt_encodings["attention_mask"]

        # match shape of prompt_attn_mask and attn_mask by padding with 0
        padding = torch.zeros(
            (attn_mask.shape[0], attn_mask.shape[1] - prompt_attn_mask.shape[1]),
        ).to(device)
        padded_prompt_attn_mask = torch.cat([prompt_attn_mask, padding], dim=1)
        prompt_mask = (padded_prompt_attn_mask == 1)
        
        # don't score prompt tokens
        attn_mask[prompt_mask] = 0

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    mean_perplexity = np.mean(ppls)
    median_perplexity = np.median(ppls)
    samples_dict["mean_perplexity"] = mean_perplexity
    samples_dict["median_perplexity"] = median_perplexity
    samples_dict["perplexities"] = ppls
    if original_device!=device:
        model.to(original_device)
    return f"mean perplexity: {mean_perplexity}, median perplexity: {median_perplexity}"


# watermark_model_path = '/remote-home1/miintern1/watermark-learnability/data/finetune_ability/math_gsm8k/llama-2-7b-sampling-watermark-distill-kgw-k0-gamma0.25-delta2/'
# watermark_model_name = "neuralmagic/Llama-2-7b-gsm8k"
# watermark_model = load_model_from_safetensors(watermark_model_path, watermark_model_name, device)
tested_model = AutoModelForCausalLM.from_pretrained(tested_model_name, device_map=device)
vanilla_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map='cpu')

all_watermark_results = {}
for watermark_name, watermark_config in watermark_configs.items():    
    print(f"Processing {watermark_name}")
    full_result = {}
    full_result['watermarked_output'] = []
    full_result['full_watermarked_output'] = []
    full_result['vanilla_output'] = []
    full_result['watermarked_scores'] = []
    full_result['vanilla_scores'] = []
    full_result['watermark_config'] = watermark_config

    if watermark_config["type"] == WatermarkType.AAR:
        watermark = AarWatermark(
            vocab_size=len(tokenizer),
            k=watermark_config["k"],
            seed=watermark_config["seed"],
            device=device,
        )
        do_sample = False
    elif watermark_config["type"] == WatermarkType.KGW:
        watermark = WatermarkLogitsProcessor(
            vocab=tokenizer.get_vocab().values(),
            gamma=watermark_config["gamma"],
            delta=watermark_config["delta"],
            seeding_scheme=watermark_config["seeding_scheme"],
            device=device,
        )
        do_sample = True

    for batch in tqdm(prompts):
        if len(full_result) >= num_samples:
            break
        with torch.no_grad():
            batch = move_to_device(batch, "cuda")
            watermarked_output = tested_model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            do_sample=DO_SAMPLE,
                            min_new_tokens=200,
                            max_new_tokens=200,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            pad_token_id=tokenizer.eos_token_id,
                            logits_processor=LogitsProcessorList([watermark])
                        )
            vanilla_output = tested_model.generate(
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
            
        watermarked_output = watermarked_output.cpu()
        vanilla_output = vanilla_output.cpu()
        n_input_tokens = batch["input_ids"].shape[1]
        del batch
        torch.cuda.empty_cache()

        watermarked_text = watermarked_output[:, n_input_tokens:]
        vanilla_text = vanilla_output[:, n_input_tokens:]

        full_result['watermarked_output'].extend(tokenizer.batch_decode(watermarked_text, skip_special_tokens=True))
        full_result['full_watermarked_output'].extend(tokenizer.batch_decode(watermarked_output, skip_special_tokens=True))
        full_result['vanilla_output'].extend(tokenizer.batch_decode(vanilla_text, skip_special_tokens=True))
        # break
    if watermark_type == "kgw":
        detector = WatermarkDetector(
                            device=watermark_config.get("kgw_device", 'cuda'),
                            tokenizer=tokenizer,
                            vocab=tokenizer.get_vocab().values(),
                            gamma=watermark_config["gamma"],
                            seeding_scheme=watermark_config["seeding_scheme"],
                            normalizers=[],
                        )
    elif watermark_config["type"] == "aar":
        detector = AarWatermarkDetector(tokenizer=tokenizer, k=watermark_config['k'], seed=watermark_config['seed'], eps=1e-20)
    
    # print("Computing p-values")
    watermarked_scores = compute_p_value(full_result['watermarked_output'], detector, type=watermark_config["type"])
    vanilla_scores = compute_p_value(full_result['vanilla_output'], detector, type=watermark_config["type"])
    full_result['watermarked_scores'] = watermarked_scores
    full_result['vanilla_scores'] = vanilla_scores
    rep_output = compute_repetition(full_result, tokenizer)
    ppl_output = compute_ppl(full_result ,prompt_text, tokenizer, vanilla_model, batch_size)

    all_watermark_results[watermark_name] = full_result
    print(all_watermark_results.keys())
    with open(f"/remote-home1/miintern1/watermark-learnability/Essay_data/Main_watermark_result/{watermark_type}_watermark_original_{save_name}.json", "w") as f:
        json.dump(all_watermark_results, f)

    



# for watermark_name, watermark_config in watermark_configs.items():
#     print(f"Processing {watermark_name}")
#     full_result[watermark_name] = {}
#     watermarked_output_results = []
#     full_watermarked_output_results = []
#     watermarked_model = AutoModelForCausalLM.from_pretrained(watermark_name, device_map='cpu')    
#     watermarked_model.half()                                                
#     watermarked_model = watermarked_model.to(device)
#     watermarked_model.eval()
#     print(f"Model {watermark_name} loaded")
#     for batch in tqdm(prompts):
#         if len(watermarked_output_results) >= num_samples:
#             break
#         with torch.no_grad():
#             # print(batch)            
#             batch = move_to_device(batch, "cuda")
#             # print(batch["input_ids"].shape)
#             watermarked_output = watermarked_model.generate(
#                             input_ids=batch["input_ids"],
#                             attention_mask=batch["attention_mask"],
#                             do_sample=DO_SAMPLE,
#                             min_new_tokens=200,
#                             max_new_tokens=200,
#                             temperature=temperature,
#                             top_p=top_p,
#                             top_k=top_k,
#                             pad_token_id=tokenizer.eos_token_id,
#                         )
#             # print(watermarked_output.shape)
#             watermarked_output = watermarked_output.cpu()
#             n_input_tokens = batch["input_ids"].shape[1]
#             del batch
#             torch.cuda.empty_cache()

            
#             watermarked_text = watermarked_output[:, n_input_tokens:]
#             full_watermarked_output_results.extend(tokenizer.batch_decode(watermarked_output, skip_special_tokens=True))
#             watermarked_output_results.extend(tokenizer.batch_decode(watermarked_text, skip_special_tokens=True))
    
#     full_result[watermark_name]['full_watermarked_output'] = full_watermarked_output_results
#     full_result[watermark_name]['watermarked_output'] = watermarked_output_results

#     if watermark_type == "kgw":
#         detector = WatermarkDetector(
#                             device='cuda',
#                             tokenizer=llama_tokenizer,
#                             vocab=llama_tokenizer.get_vocab().values(),
#                             gamma=watermark_config["gamma"],
#                             seeding_scheme=watermark_config["seeding_scheme"],
#                             normalizers=[],
#                         )
#     elif watermark_config["type"] == "aar":
#         detector = AarWatermarkDetector(tokenizer=llama_tokenizer, k=watermark_config['k'], seed=watermark_config['seed'], eps=1e-20)
    
#     # print("Computing p-values")
#     watermarked_scores = compute_p_value(watermarked_output_results, detector, type=watermark_config["type"])
#     full_result[watermark_name]['watermarked_scores'] = watermarked_scores
#     # print(watermarked_scores)

#     del watermarked_model
#    # break

with open(f"/remote-home1/miintern1/watermark-learnability/Essay_data/Main_watermark_result/{watermark_type}_watermark_original_{save_name}.json", "w") as f:
    json.dump(all_watermark_results, f)


# with open(f'/remote-home1/miintern1/watermark-learnability/pythia_watermark_ability_check_{watermark_type}.json', 'w') as f:
#     json.dump(full_result, f)