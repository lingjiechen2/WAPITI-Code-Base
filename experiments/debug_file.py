import argparse
import os
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
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from task_vector import TaskVector

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType
import logging


logging.info(f"{torch.cuda.device_count()=}")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
watermarked_model = AutoModelForCausalLM.from_pretrained(f"cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1")
watermarked_model.half()
watermarked_model.to("cuda")

text = "Please introduce yourself to me:"
inputs = tokenizer(text, return_tensors='pt', truncation=True)
input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
input_ids, attention_mask = input_ids.to("cuda"), attention_mask.to("cuda")
substututed_output = watermarked_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    do_sample=True,  # or False, depending on your requirement
    min_length=50,
    max_length=100,
    temperature=1.0,
    top_p=0.95,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
)

substututed_text = tokenizer.decode(substututed_output[0], skip_special_tokens=True)
print(substututed_text)
print('-'*100)
watermarked_model.to("cpu")

vanilla_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
vanilla_model.half()
vanilla_model.to("cuda")
generated_output = vanilla_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    do_sample=True,  # or False, depending on your requirement
    min_length=50,
    max_length=100,
    temperature=1.0,
    top_p=0.95,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
)

generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
print(generated_text)
print('-'*100)
