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
from transformers import AutoTokenizer
from datasets import load_dataset
from itertools import chain
from collections import defaultdict, Counter
from transformers import AutoTokenizer
from pprint import pprint


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False

def calculate_ngrams(sequence, n):
    ngrams = zip(*[sequence[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def calculate_ngrams(sequence, n):
    ngrams = zip(*[sequence[i:] for i in range(n)])
    return [list(ngram) for ngram in ngrams] 

with open('/remote-home/miintern1/watermark-learnability/data/c4/aronson_watermark_vector.json', 'r') as f:
    aar_data = json.load(f)

watermark_names = ['cygu/llama-2-7b-logit-watermark-distill-aar-k2',
                     'cygu/llama-2-7b-logit-watermark-distill-aar-k3',
                     'cygu/llama-2-7b-logit-watermark-distill-aar-k4']

watermark_configs = {
    "cygu/llama-2-7b-logit-watermark-distill-aar-k2":{"type": "aar", "k": 2, "seed": 42},
    "cygu/llama-2-7b-logit-watermark-distill-aar-k3":{"type": "aar", "k": 3, "seed": 42},
    "cygu/llama-2-7b-logit-watermark-distill-aar-k4":{"type": "aar", "k": 4, "seed": 42},
}
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

aar_watermark_distillation_generation_ngrams = {}

for watermark_name, watermark_data in aar_data.items():
    watermark_config = watermark_configs[watermark_name]
    n = watermark_config['k'] + 1
    print(f"Now processing {watermark_name} with {n}-grams")
    aar_watermark_distillation_generation_ngrams[watermark_name] = dict()
    coefficient_list = [coefficient for coefficient in watermark_data.keys() if is_float(coefficient)]
    print(coefficient_list)
    for coefficient in coefficient_list:
        coefficient_ngram_count = defaultdict(int)
        generation = watermark_data[coefficient]['watermarked_output']
        # print(f"There are total {len(generation)} generations")
        for example in generation:
            # decoded_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            tokens_data = tokenizer(example, add_special_tokens=False)  # add_special_tokens=False to avoid [CLS], [SEP], etc.
            input_ids = tokens_data['input_ids']  # Get the input_ids
            ngrams = calculate_ngrams(input_ids, n)
            for ngram in ngrams:
                coefficient_ngram_count[ngram] += 1
        
        sorted_coefficient_ngram_count = dict(sorted(dict(coefficient_ngram_count).items(), key=lambda item: item[1], reverse=True))
        aar_watermark_distillation_generation_ngrams[watermark_name][coefficient] = sorted_coefficient_ngram_count

        # print(f"Finished processing coefficient {coefficient}, total ngrams: {len(aar_watermark_distillation_generation_ngrams[watermark_name][coefficient])}")

for watermark_name, watermark_data in aar_watermark_distillation_generation_ngrams.items():
    ngram_number = [len(ngram_count) for ngram_count in watermark_data.values()]
    print(f"Watermark: {watermark_name}, Number of ngrams: {ngram_number}")

