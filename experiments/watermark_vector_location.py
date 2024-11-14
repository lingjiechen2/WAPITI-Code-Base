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

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType

# from huggingface_hub import login
# login(token="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")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
watermarked_model = AutoModelForCausalLM.from_pretrained(model_path)
watermarked_model.half()
watermarked_model.to(device)
watermarked_model.train()

watermark_configs = {
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2":{"type": "kgw", "k": 2, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_2", "kgw_device": "cpu"},
}

def topk_elements_by_dot_product(model, k, input_text = "The quick brown fox jumps over the lazy dog."):
    elementwise_products = []
    indices = []
    param_names = []
    
    inputs = tokenizer(input_text, return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()


    for name, param in model.named_parameters():
        if param.grad is not None:
            product = param * param.grad
            flattened_product = product.view(-1)
            elementwise_products.append(flattened_product)
            indices.append(torch.arange(flattened_product.numel())) 
            param_names.append([name] * flattened_product.numel())  

    # Concatenate all the element-wise products and indices into one tensor
    elementwise_products = torch.cat(elementwise_products)
    indices = torch.cat(indices)
    param_names = [item for sublist in param_names for item in sublist]  # Flatten param_names list

    # Get the top K element-wise products and their corresponding indices
    topk_values, topk_indices = torch.topk(elementwise_products, k)
    topk_param_names = [param_names[i] for i in topk_indices]
    topk_param_indices = [indices[i].item() for i in topk_indices]  # Convert indices to integers

    return topk_values, topk_param_names, topk_param_indices

watermark_config = watermark_configs["cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1"]
detector = WatermarkDetector(
                            device=watermark_config.get("kgw_device", 'cpu'),
                            tokenizer=tokenizer,
                            vocab=tokenizer.get_vocab().values(),
                            gamma=watermark_config["gamma"],
                            seeding_scheme=watermark_config["seeding_scheme"],
                            normalizers=[],
                        )

# Vanilla Text
vanilla_text = "The quick brown fox jumps over the lazy dog."
print("Vanilla Text detection result")
print(detector.detect(vanilla_text))

vanilla_inputs = tokenizer(vanilla_text, return_tensors="pt")
for key in vanilla_inputs:
    vanilla_inputs[key] = vanilla_inputs[key].to(device)
vanilla_outputs = watermarked_model(**vanilla_inputs, labels=vanilla_inputs["input_ids"])
loss = vanilla_outputs.loss
loss.backward()

k = 15  # Specify how many top elements you want
topk_values, topk_param_names, topk_param_indices = topk_elements_by_dot_product(watermarked_model, k)
print(f"Top {k} elements by dot product:{topk_param_names}")
print(f"Top {k} elements by dot product:{topk_values}")
print(f"Top {k} elements by dot product:{topk_param_indices}")