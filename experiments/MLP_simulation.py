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
from datasets import load_dataset,load_from_disk
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from task_vector import TaskVector

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType
import logging
import torch
from torch.utils.data import Dataset, DataLoader,IterableDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import os
import glob

logging.basicConfig(
    filename='/remote-home/miintern1/watermark-learnability/logs/MLP_simulation.txt',  # Specify the log file name
    level=logging.INFO,          # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Set the log message format
)

vanilla_outputs = {}
substitution_outputs = {}
watermark_outputs = {}
vanill_generation_results = {}
watermarked_generation_results = {}
substitution_generation_results = {}
hook_layer = [31]  # Specify the layers to hook
max_length = 250
min_length = 250
num_samples = 512
batch_size = 16
save_path = '/remote-home/miintern1/watermark-learnability/data/c4/simulation_generation.json'
hook_handles = {}
if os.path.exists('/remote-home/miintern1/watermark-learnability/data/c4/simulation_generation.json'):
    # If the file exists, open it and load its content
    with open('/remote-home/miintern1/watermark-learnability/data/c4/simulation_generation.json', 'r') as f:
        generation_result_dict = json.load(f)
else:
    generation_result_dict = {}
vanilla_model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(vanilla_model_name)


# dataset = load_dataset("allenai/c4", "realnewslike", split="validation", streaming=False)
# os.environ['HF_DATASETS_CACHE'] = '/remote-home/miintern1/watermark-learnability/experiments/.cache/huggingface/dataset/'
# dataset = load_dataset("allenai/c4", "realnewslike", split="validation", streaming=False)
dataset = load_from_disk("/remote-home/share/data/c4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Using device: {}".format(device))
set_seed(42)
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
    )
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

print("Filtering and encoding dataset")
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

def extract_files(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    files = [f for f in files if os.path.isfile(f)]
    return files

def compute_p_value(samples, detector, type='kgw'):
    score_list = []
    for s in samples:
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
    samples = samples_dict['truncate_prompt_output']
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

    samples = samples_dict["full_output"]

    # for i in tqdm(range(0, len(samples), batch_size)):
    for i in range(0, len(samples), batch_size):
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


def generate_and_evaluate(prompts, num_samples, model, tokenizer, generation_result_dict, watermark_config):
    output_results = []
    full_output_results = []

    # Initialize tqdm progress bar outside the loop
    with tqdm(total=len(prompts), desc="Generating outputs") as pbar:
        for batch in prompts:
            if len(output_results) >= num_samples:
                break
            with torch.no_grad():
                batch = move_to_device(batch, "cuda")
                vanilla_output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    do_sample=True,
                    min_length=50,
                    max_length=100,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
                n_input_tokens = batch["input_ids"].shape[1]
                truncated_prompt_output = vanilla_output[:, n_input_tokens:]

                output_results.extend(tokenizer.batch_decode(truncated_prompt_output, skip_special_tokens=True))
                full_output_results.extend(tokenizer.batch_decode(vanilla_output, skip_special_tokens=True))

            generation_result_dict['full_output'] = full_output_results[:num_samples]
            generation_result_dict["truncate_prompt_output"] = output_results[:num_samples]

            output_results = output_results[:num_samples]
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
            
            output_scores = compute_p_value(output_results, detector, type=watermark_config["type"])
        
            generation_result_dict['watermark_scores'] = output_scores
            rep_output = compute_repetition(generation_result_dict, tokenizer)
            # print(f"{rep_output}")
            # ppl_output = compute_ppl(generation_result_dict, prompt_text, tokenizer, ppl_model, batch_size)
            # print(f"{ppl_output}")

            # Update the tqdm progress bar
            pbar.update(1)
    return generation_result_dict

def save_json(new_data, path):
    # Read existing data from the file if it exists
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data.update(new_data)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

vanilla_model = AutoModelForCausalLM.from_pretrained(vanilla_model_name)
vanilla_model.half()
vanilla_model.to(device)

if 'vanilla_model' not in generation_result_dict:
    logging.info("Generating and evaluating vanilla model")
    generation_result_dict = dict()
    generation_result_dict['vanilla_model'] = dict()
    generate_and_evaluate(prompts, num_samples, vanilla_model, tokenizer, generation_result_dict['vanilla_model'], {'type': 'kgw', 'gamma': 0.25, 'seeding_scheme': 'simple_0'})
    save_json(generation_result_dict, save_path)
else:
    logging.info("Vanilla model already evaluated")
vanilla_model.to('cpu')


folder_path = '/remote-home/miintern1/watermark-learnability/data/model_weights_2'
MLP_model_list = extract_files(folder_path)
MLP_model_path_dict = defaultdict(list)
for path in MLP_model_list:
    MLP_model_path_dict[path[83:106]].append(path)
watermark_list = list(MLP_model_path_dict.keys())

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = out + x 
        return out

class TransformModel(nn.Module):
    def __init__(self, num_layers=4, input_dim=1024, hidden_dim=500, output_dim=300):
        super(TransformModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(ResidualBlock(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
    
def substitute_layer_hook(layer_name, mlp_model=None):
    def hook_fn(module, input, output):
        # Store the original output
        if layer_name not in vanilla_outputs:
            vanilla_outputs[layer_name] = []
        activation = output[0][0]
        # print(f"{isinstance(activation, torch.Tensor)=}")
        # vanilla_outputs[layer_name].append(output)
        
        # If mlp_model is provided, use it to transform the output
        if mlp_model is not None:
            # print(f"Transforming output of layer {layer_name}")
            with torch.no_grad():
                # Ensure the output is on the same device as the mlp_model
                activation = output[0]
                transformed_output = mlp_model(activation).to(output[0].device)
                # transformed_output = transformed_output.to(torch.float16)
                # logging.info(f"{transformed_output.device=}")
                # logging.info(f"{output[0].device=}")
                modified_output = (transformed_output, *output[1:])
                # if layer_name not in modified_outputs:
                #     modified_outputs[layer_name] = []
                # modified_outputs[layer_name].append(modified_output[0].cpu())
                return modified_output
        else:
            return output
    return hook_fn

def remove_hooks(hook_handles):
    for handle in hook_handles.values():
        handle.remove()

for watermark in watermark_list:
    logging.info(f"Processing watermark: {watermark}")
    watermark_config_name = f"/remote-home/miintern1/watermark-learnability/experiments/watermark-configs/{watermark}-config.json"
    with open(watermark_config_name, "r") as f:
        watermark_config = json.load(f)
    if watermark not in generation_result_dict:
        watermarked_model = AutoModelForCausalLM.from_pretrained(f"cygu/llama-2-7b-logit-watermark-distill-{watermark}")
        watermarked_model.half()
        watermarked_model.to(device)
        generation_result_dict[watermark] = dict()
        generation_result_dict[watermark]['watermarked_model'] = dict()
        generate_and_evaluate(prompts, num_samples, watermarked_model, tokenizer, generation_result_dict[watermark]['watermarked_model'], watermark_config)        
        save_json(generation_result_dict, save_path)
        del watermarked_model
    else:
        logging.info(f"Watermarked model {watermark} already evaluated")
    MLP_model_list = MLP_model_path_dict[watermark]
    # vanilla_model = AutoModelForCausalLM.from_pretrained(vanilla_model_name)
    # vanilla_model.half()
    vanilla_model.to(device)
    for MLP_model_path in MLP_model_list:
        if MLP_model_path[-42:-14] not in generation_result_dict[watermark]:
            logging.info(f"Processing MLP model: {MLP_model_path}")
            loaded_dict = torch.load(MLP_model_path, map_location='cpu')
            model_state_dict = loaded_dict["model_state_dict"]
            loaded_hyperparameters = loaded_dict["hyperparameters"]
            loaded_num_layers = loaded_hyperparameters["num_layers"]
            loaded_hidden_dim = loaded_hyperparameters["hidden_dim"]
            loaded_learning_rate = loaded_hyperparameters["learning_rate"]
            loaded_num_epochs = loaded_hyperparameters["num_epochs"]
            MLP_model = TransformModel(
                num_layers=loaded_num_layers,
                input_dim=4096,  # Change this as needed
                hidden_dim=loaded_hidden_dim,
                output_dim=4096  # Change this as needed
            )
            MLP_model.load_state_dict(model_state_dict)
            MLP_model.half()
            MLP_model.to(device)
            if hook_handles:
                remove_hooks(hook_handles)
            # logging.info(f"Substituting layer {hook_layer} with MLP model")
            for i in hook_layer:
                layer_name = f"model.layers.{i}"
                layer = dict([*vanilla_model.named_modules()])[layer_name]
                hook_handle = layer.register_forward_hook(substitute_layer_hook(layer_name, MLP_model))
                hook_handles[layer_name] = hook_handle
                # logging.info(f"Hooked layer {layer_name}")
            # hyperparameter_string = f"num_layers_{loaded_num_layers}_hidden_dim_{loaded_hidden_dim}"
            generation_result_dict[watermark][MLP_model_path[-42:-14]] = dict()
            generate_and_evaluate(prompts, num_samples, vanilla_model, tokenizer, generation_result_dict[watermark][MLP_model_path[-42:-14]], watermark_config)
            save_json(generation_result_dict, save_path)
            del MLP_model
        else:
            logging.info(f"MLP model {MLP_model_path} already evaluated")
    vanilla_model.to('cpu')


     
                                                                                 

