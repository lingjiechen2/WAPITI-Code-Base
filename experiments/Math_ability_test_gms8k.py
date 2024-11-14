import argparse
import os
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
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
from transformer_lens import HookedTransformer
from task_vector import TaskVector
import plotly.express as px
from safetensors import safe_open

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType

from huggingface_hub import login
login(token="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")


math_dataset = load_dataset('gsm8k','main')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

test_samples = 400
save_path = '/remote-home1/miintern1/watermark-learnability/data/finetune_ability/math_gsm8k/math_ability_test.json'
math_model_name = "neuralmagic/Llama-2-7b-gsm8k"
# watermarked_math_model_name = 'OnAnOrange/llama-gms8k-watermarked-model'
# base_model = 'meta-llama/Llama-2-7b-chat-hf'
# model = AutoModelForCausalLM.from_pretrained(math_model_name, device_map = 'cpu')
# math_tokenizer = AutoTokenizer.from_pretrained(math_model_name)   
# model.to(device)
output_path = f"/remote-home1/miintern1/watermark-learnability/data/finetune_ability/math_gsm8k/{math_model_name.split('/')[-1]}_watermark.json"

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


watermark_configs = {
        "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
        "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
        "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
        "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
        "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2":{"type": "kgw", "k": 2, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_2", "kgw_device": "cpu"},
        "cygu/llama-2-7b-logit-watermark-distill-aar-k2":{"type": "aar", "k": 2, "seed": 42},
        "cygu/llama-2-7b-logit-watermark-distill-aar-k3":{"type": "aar", "k": 3, "seed": 42},
        "cygu/llama-2-7b-logit-watermark-distill-aar-k4":{"type": "aar", "k": 4, "seed": 42}
    }


def extract_content_after_hashes(text):
    # Split the string at the first occurrence of '###'
    parts = text.split("####", 1)
    # Return the part after the '###', which is the second element in the list
    return parts[1] if len(parts) > 1 else ""

def evaluate_model_on_gms8k(model, tokenizer, dataset, num_samples=5):
    correct = 0
    results = [] 

    for i in tqdm(range(num_samples)):
        question = dataset['test']['question'][i]
        prompt_prefix = "Output the answer to the following math question after ###:"
        expected_answer = dataset['test']['answer'][i]
        prompt = f"{prompt_prefix} {question}"

        # Tokenize input and generate model output
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs['input_ids'], 
                                 attention_mask = inputs['attention_mask'],
                                 max_new_tokens=200)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the content after hashes (###) for both expected and generated answers
        extracted_content = extract_content_after_hashes(generated_answer)
        expected_answer = extract_content_after_hashes(expected_answer)
        
        # Check if the expected answer matches the generated answer
        if expected_answer.strip() == extracted_content.strip():
            correct += 1

        # Store the result
        results.append({
            "question": question,
            "prompt": prompt,
            "expected_answer": expected_answer.strip(),
            "generated_answer": generated_answer.strip()
        })

    accuracy = correct / num_samples
    return accuracy, results


# vanilla_model_name = 'meta-llama/Llama-2-7b-hf'
# vanilla_model = AutoModelForCausalLM.from_pretrained(vanilla_model_name, device_map = 'cpu')


# if os.path.exists(save_path):
#     print("Detect previous results")
#     with open(save_path, 'r') as f:
#         accuracy_file = json.load(f)
# else:
#     accuracy_file = {}


# if os.path.exists(output_path):
#     print("Detect previous results")
#     with open(output_path,'r') as f:
#         output_file = json.load(f)
# else:
#     output_file = {}


# for watermark_name, watermark_config in watermark_configs.items():
#     print(f"Now processing {watermark_name}")
#     print(f"{list(accuracy_file[math_model_name].keys())}")
#     if watermark_name in list(accuracy_file[math_model_name].keys()):
#         print(f"Already processed {watermark_name}, skip it.")
#         continue
#     watermarked_model = AutoModelForCausalLM.from_pretrained(watermark_name)
#     task_vector = TaskVector(vanilla_model, watermarked_model)
#     math_model = copy.deepcopy(model)
#     tested_model = task_vector.apply_to(math_model, scaling_coef=1.0)
#     tested_model.to(device)

#     accuracy, results = evaluate_model_on_gms8k(tested_model, math_tokenizer, math_dataset, num_samples=test_samples)
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     accuracy_file[math_model_name][watermark_name] = accuracy
#     output_file[math_model_name][watermark_name] = results
#     with open(save_path, 'w') as f:
#         json.dump(accuracy_file, f)

#     with open(output_path, 'w') as f:
#         json.dump(output_file, f)
#     del tested_model, watermarked_model

# if 'original_model' in list(accuracy_file.keys()):
#     print("Already processed original model, skip it.")
# else:
#     model.to(device)
#     accuracy, results = evaluate_model_on_gms8k(model, math_tokenizer, math_dataset, num_samples=test_samples)
#     accuracy_file['original_model'] = accuracy
#     output_file['original_model'] = results
#     print(f"The accuracy is {100*accuracy:.2f}%")
#     del model

# if 'base_model' in list(accuracy_file.keys()):
#     print("Already processed base model, skip it.")
# else:
#     vanilla_model.to(device)
#     accuracy, results = evaluate_model_on_gms8k(vanilla_model, math_tokenizer, math_dataset, num_samples=test_samples)
#     accuracy_file['base_model'] = accuracy
#     output_file['base_model'] = results
#     print(f"The accuracy is {100*accuracy:.2f}%")
#     del vanilla_model


# with open(save_path, 'w') as f:
#     json.dump(accuracy_file, f)

# with open(output_path, 'w') as f:
#     json.dump(output_file, f)

# coefficient_list = np.arange(0.5, 1.5, 0.1)
# accuracy_dict = {}
# for coefficient in coefficient_list:
#     # start_time = time.time()
#     # print(f"coefficient: {coefficient}")
#     tested_model_copy = copy.deepcopy(tested_model)
#     tested_model_copy.to('cpu')
#     math_model = task_vector.apply_to(tested_model_copy, scaling_coef= coefficient)
#     math_model.eval()
#     math_model.to(device)
#     accuracy = evaluate_model_on_gms8k(math_model, math_tokenizer, math_dataset, num_samples=200)
#     print(f"coefficient: {coefficient}, accuracy: {accuracy * 100:.2f}%")
#     accuracy_dict[coefficient] = accuracy
#     del math_model
#     # end_time = time.time()
# print(accuracy_dict)

# model_name = 'andreaskoepf/pythia-1.4b-gpt4all-pretrain'
# save_path = f'/remote-home1/miintern1/watermark-learnability/data/finetune_ability/math_gsm8k/{model_name.replace('/', '_')}.json'
# print(save_path)
# start_time = time.time()
# watermarked_math_model = AutoModelForCausalLM.from_pretrained(model_name)
# math_tokenizer = AutoTokenizer.from_pretrained(model_name)
# watermarked_math_model.eval()
# # watermarked_math_model.half()
# watermarked_math_model.to(device)
# accuracy, results = evaluate_model_on_gms8k(watermarked_math_model, math_tokenizer, math_dataset, num_samples=400)
# print(f"The accuracy is {100*accuracy:.2f}%")
# print(f"Total accuracy is {100*accuracy:.2f}%")
# with open(save_path, 'w') as f:
#     json.dump(results, f)
# del watermarked_math_model

math_model_name = 'neuralmagic/Llama-2-7b-gsm8k'
math_tokenizer = AutoTokenizer.from_pretrained(math_model_name)
watermarked_math_model =load_model_from_safetensors('data/finetune_ability/aar_math_gsm8k/llama-2-7b-sampling-watermark-distill-arr-k2/checkpoint-333', math_model_name, device)
accuracy, results = evaluate_model_on_gms8k(watermarked_math_model, math_tokenizer, math_dataset, num_samples=400)
print(f"The accuracy is {100*accuracy:.2f}%")