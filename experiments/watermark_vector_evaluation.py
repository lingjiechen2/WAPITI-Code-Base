import argparse
import os
import copy
os.environ['http_proxy'] = "http://10.176.52.116:7890"
os.environ['https_proxy'] = "http://10.176.52.116:7890"
os.environ['all_proxy'] = "socks5://10.176.52.116:7891"
import sys
sys.path.append(('../'))
sys.path.append(('../../'))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import json
from typing import Dict
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from task_vector import TaskVector

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType
import logging


logging.basicConfig(
    filename='/remote-home/miintern1/watermark-learnability/logs/kgw_watermark_vector_NousResearch_Llama-2-7b-chat-hf_log.txt',  # Specify the log file name
    level=logging.INFO,          # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Set the log message format
)

logging.info(f"{torch.cuda.device_count()=}")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# dataset = load_dataset("allenai/c4", "realnewslike", "validation")
dataset = load_dataset("allenai/c4", "realnewslike", split="validation", streaming=True)

max_length = 250
min_length = 250
num_samples = 512
batch_size = 16
save_path = '/remote-home/miintern1/watermark-learnability/data/c4/kgw_watermark_vector_NousResearch_Llama-2-7b-chat-hf.json'
if os.path.exists(save_path):
    with open(save_path, 'r') as json_file:
        all_model_dict = json.load(json_file)
else:
    all_model_dict = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)
logging.info("Using device: {}".format(device))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
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
logging.info("Data loaded and processed successfully")

watermark_configs = {
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2":{"type": "kgw", "k": 2, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_2", "kgw_device": "cpu"},
}

# watermark_configs = {
#     "cygu/llama-2-7b-logit-watermark-distill-aar-k2":{"type": "aar", "k": 2, "seed": 42},
#     "cygu/llama-2-7b-logit-watermark-distill-aar-k3":{"type": "aar", "k": 3, "seed": 42},
#     "cygu/llama-2-7b-logit-watermark-distill-aar-k4":{"type": "aar", "k": 4, "seed": 42},
# }

vanilla_model_name = "meta-llama/Llama-2-7b-hf"
# tested_model_name = "meta-llama/Llama-2-7b-chat-hf"
tested_model_name = 'NousResearch/Llama-2-7b-chat-hf'

def compute_p_value(samples, detector, type='kgw'):
    score_list = []
    for s in tqdm(samples):
        score = detector.detect(s)
        score_list.append(score['p_value']) if type=='kgw' else score_list.append(score)
    return score_list


def compute_seq_rep_n(samples, tokenizer, n=3):
    """compute seq-rep-n metric"""
    n_gram_reps = []
    
    for s in samples:
        n_grams = []
        tokens = tokenizer(s, add_special_tokens=False).input_ids
        for i in range(len(tokens)):
            if i <= len(tokens) - n:
                n_grams.append(tuple(tokens[i:i + n]))
                    
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
    # if original_device!=device:
    #     model.to(original_device)
    return f"mean perplexity: {mean_perplexity}, median perplexity: {median_perplexity}"


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

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

DO_SAMPLE = True
temperature=1.0
top_p=0.9
top_k=0

# coefficient_list = np.arange(0.0, 1.0, 0.1)
coefficient_list = np.concatenate((np.arange(0.0, 1.0, 0.1) , np.arange(1.0, 4.0, 0.3)))

tested_model = AutoModelForCausalLM.from_pretrained(tested_model_name)
tested_model = tested_model.half()

for watermark_name, watermark_config in watermark_configs.items():
    logging.info(f"Processing watermark {watermark_name}")
    if watermark_name not in all_model_dict:
        all_model_dict[watermark_name] = dict()
    elif [float(value) for value in all_model_dict[watermark_name].keys() if is_float(value)]:
        logging.info(f"Skip processing {watermark_name} since it already exists")
        continue

    watermarked_model = AutoModelForCausalLM.from_pretrained(watermark_name)                                                    
    watermarked_model = watermarked_model.half()
    
    vanilla_model= AutoModelForCausalLM.from_pretrained(vanilla_model_name)                                                       
    vanilla_model = vanilla_model.half()
    task_vector = TaskVector(vanilla_model, watermarked_model)
    vanilla_model.to("cuda")
    # task_vector.to("cuda:1")
    # watermarked_model.to("cuda:0")
    # vanilla_model.to("cuda:1")

    logging.info(f"Processing watermark with coefficients {coefficient_list}")
    all_model_dict[watermark_name]['watermark_config'] = watermark_config
    # coefficient_list = [1]
    
    for i,coefficient in enumerate(coefficient_list):
        logging.info(f"Processing coefficient {coefficient}")
        vanilla_output_results = []
        watermarked_output_results = []
        full_watermarked_output_results = []
        if coefficient in all_model_dict[watermark_name]:
            logging.info(f"Skip processing {coefficient} since it already exists")
            continue
        else:
            all_model_dict[watermark_name][coefficient] = dict()

        # Load the tested model
        all_model_dict[watermark_name][coefficient] = dict()
        tested_model_copy = copy.deepcopy(tested_model)
        tested_model_copy.to('cpu')
        coefficient_watermarked_model = task_vector.apply_to(tested_model_copy, scaling_coef = coefficient)
        coefficient_watermarked_model.to("cuda")
        tested_model.to('cuda')

        for batch in tqdm(prompts):
            if len(watermarked_output_results) >= num_samples:
                break
            with torch.no_grad():
                # print(batch)            
                batch = move_to_device(batch, "cuda")
                # logging.info(f"Vanilla model input device: {batch['input_ids'].device}")
                # logging.info(f"Vanilla model device: {next(vanilla_model.parameters()).device}")
                # logging.info(f"Watermarked model input device: {batch['input_ids'].device}")
                # logging.info(f"Watermarked model device: {next(coefficient_watermarked_model.parameters()).device}")

                # Used to check whether task vector is correctly applied
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
                watermarked_output = coefficient_watermarked_model.generate(
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
            
        
            # torch.cuda.empty_cache()
            n_input_tokens = batch["input_ids"].shape[1]
            vanilla_text = vanilla_output[:, n_input_tokens:]
            watermarked_text = watermarked_output[:, n_input_tokens:]

            vanilla_output_results.extend(tokenizer.batch_decode(vanilla_text, skip_special_tokens=True))
            watermarked_output_results.extend(tokenizer.batch_decode(watermarked_text, skip_special_tokens=True))
            full_watermarked_output_results.extend(tokenizer.batch_decode(watermarked_output, skip_special_tokens=True))
            # break

        all_model_dict[watermark_name][coefficient]["watermarked_output"] = watermarked_output_results[:num_samples]
        all_model_dict[watermark_name][coefficient]["full_watermarked_output"] = full_watermarked_output_results[:num_samples]
        all_model_dict[watermark_name][coefficient]["vanilla_output"] = vanilla_output_results

        vanilla_output_results = vanilla_output_results[:num_samples]
        watermarked_output_results = watermarked_output_results[:num_samples]
        full_watermarked_output_results = full_watermarked_output_results[:num_samples]

        if watermark_config["type"] == "kgw":
            detector = WatermarkDetector(
                            device=watermark_config.get("kgw_device", 'cpu'),
                            tokenizer=tokenizer,
                            vocab=tokenizer.get_vocab().values(),
                            gamma=watermark_config["gamma"],
                            seeding_scheme=watermark_config["seeding_scheme"],
                            normalizers=[],
                        )
        elif watermark_config["type"] == "aar":
            detector = AarWatermarkDetector(tokenizer=tokenizer, k=watermark_config['k'], seed=watermark_config['seed'], eps=1e-20)
        
        vanilla_scores = compute_p_value(vanilla_output_results, detector, type=watermark_config["type"])
        watermarked_scores = compute_p_value(watermarked_output_results, detector, type=watermark_config["type"])
        all_model_dict[watermark_name][coefficient]["vanilla_scores"] = vanilla_scores
        all_model_dict[watermark_name][coefficient]["watermarked_scores"] = watermarked_scores
        # print(f"{type(vanilla_scores)=}")
        # print(f"{type(watermarked_scores)=}")
        rep_output = compute_repetition(all_model_dict[watermark_name][coefficient], tokenizer)
        logging.info(f"{rep_output}")
        ppl_output = compute_ppl(all_model_dict[watermark_name][coefficient],prompt_text, tokenizer, vanilla_model, batch_size)
        logging.info(f"{ppl_output}")
        logging.info(f"Finished processing coefficient {coefficient}")
        # break
        with open(save_path, 'w') as json_file:
            json.dump(all_model_dict, json_file, indent=4)
    del vanilla_model
    del watermarked_model
    logging.info(f"Finished processing watermark {watermark_name}".center(80, "="))


# with open(save_path, 'w') as json_file:
#     json.dump(all_model_dict, json_file, indent=4)

# logging.info(f"Dictionary has been saved to {save_path}")

    
    
