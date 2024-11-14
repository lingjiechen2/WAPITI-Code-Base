import argparse
import os
import sys
import copy
import json
import logging

# Set up environment variables for proxies
os.environ['http_proxy'] = "http://10.176.52.116:7890"
os.environ['https_proxy'] = "http://10.176.52.116:7890"
os.environ['all_proxy'] = "socks5://10.176.52.116:7891"

# Adjust the system path
sys.path.append('/remote-home1/miintern1/watermark-learnability')

# Import necessary modules
from typing import Dict
from itertools import chain

# PyTorch and related imports
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# Data handling and processing
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset,load_from_disk

# Transformers and model-related imports
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformer_lens import HookedTransformer

# Watermarking imports
from task_vector import TaskVector
from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType

# Visualization imports
# import seaborn as sns
import matplotlib.pyplot as plt

tested_model_path_dict = {
    'QA': "PKU-Alignment/alpaca-7b-reproduced-llama-2",
    'math':"neuralmagic/Llama-2-7b-gsm8k",
    'instruct': "meta-llama/Llama-2-7b-chat-hf"
}
watermark_path_dict = {
    "k0-gamma0.25-delta1": "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1",
    "k0-gamma0.25-delta2": "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2",
    "k1-gamma0.25-delta1": "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1",
    "k1-gamma0.25-delta2": "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2",
    "k2-gamma0.25-delta2": "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2",
    "aar-k2": "cygu/llama-2-7b-logit-watermark-distill-aar-k2",
    "aar-k3": "cygu/llama-2-7b-logit-watermark-distill-aar-k3",
    "aar-k4": "cygu/llama-2-7b-logit-watermark-distill-aar-k4"
}

all_dict = {**tested_model_path_dict, **watermark_path_dict}
vanilla_model_name = "meta-llama/Llama-2-7b-hf"
vanilla_model = AutoModelForCausalLM.from_pretrained(vanilla_model_name)
cosine_similarity_df = pd.DataFrame(index=all_dict.keys(), columns=all_dict.keys())
task_vector_dict = dict()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for name, model_path in all_dict.items():
    model = AutoModelForCausalLM.from_pretrained(model_path)
    task_vector = TaskVector(vanilla_model, model)
    task_vector_flattened = torch.cat([v.flatten() for v in task_vector.vector.values()])
    task_vector_flattened = task_vector_flattened / torch.norm(task_vector_flattened)
    task_vector_dict[name] = task_vector_flattened.double().to(device)

    del model
    torch.cuda.empty_cache()

assert len(task_vector_dict) == len(all_dict)

print("Calculating cosine similarity")
for name1, vec1 in task_vector_dict.items():
    for name2, vec2 in task_vector_dict.items():
        cosine_sim = F.cosine_similarity(vec1, vec2, dim=0).item()
        cosine_similarity_df.loc[name1, name2] = cosine_sim


cosine_similarity_df.to_csv("cuda_cosine_similarity.csv")
