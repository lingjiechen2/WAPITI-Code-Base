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

from huggingface_hub import login
login(token="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")
import numpy as np
import pandas as pd
import time
# from crop import crop

choices = ["A", "B", "C", "D"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

print(f"device: {device}")

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def alpaca_format_prompt(df, idx, include_answer=True):
    prompt_template = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:'
    prompt = df.iloc[idx, 0]
    k= df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    final_prompt = prompt_template.format(input=prompt)
    # print(final_prompt)
    return final_prompt

def eval(ntrain, subject, tokenizer, model, dev_df, test_df, demo_size = 20, save_path = False):
    prompt_demo = []
    generation_demo = []
    cors = []
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = alpaca_format_prompt(test_df, i, include_answer=False)
        prompt = prompt_end
        # print(f"prompt is {prompt}")

        # prompt_end = format_example(test_df, i, include_answer=False)
        # prompt = prompt_end
        # prompt = train_prompt + prompt_end
        prompt_demo.append(prompt)

        # while crop(prompt) != prompt:
        #     k -= 1
        #     train_prompt = gen_prompt(dev_df, subject, k)
        #     prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        try:
            # Replace the OpenAI API call with a call to the local model's generate method
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            # print(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
            completion = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens=1,
                # temperature = 0.1,
            )
            # print(completion.shape)
            # print(f"Response:{tokenizer.decode(completion[0], skip_special_tokens=True)}")
            # print(completion.shape)
            # print(tokenizer.decode(completion[0], skip_special_tokens=True))
            pred = tokenizer.decode(completion[0, input_ids.shape[-1]:], skip_special_tokens=True)#[0]
            # print(f"The prefiction is {pred} and actual label is {label}")
            generation_demo.append(pred)
            cor = pred == label
            cors.append(cor)
            # print(pred)
            # print('='*100)
        except Exception as e:
            print(f"Error generating with the model: {e}")
            continue

    acc = np.mean(cors)
    generation_demo = generation_demo[:demo_size]
    prompt_demo = prompt_demo[:demo_size]
    # print(f"Average accuracy {acc:.3f} - {subject}")

    return cors, acc, generation_demo, prompt_demo

data_dir = '/remote-home1/miintern1/watermark-learnability/data/finetune_ability/MMLU'
ntrain = 5
subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f])
print(subjects)

vanilla_model_name = 'meta-llama/Llama-2-7b-hf'
# watermarked_model_name = "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1"
vanilla_model = AutoModelForCausalLM.from_pretrained(vanilla_model_name, device_map = 'cpu')
# watermarked_model = AutoModelForCausalLM.from_pretrained(watermarked_model_name)
# task_vector = TaskVector(vanilla_model, watermarked_model)

# QA_model_name = 'kalyan003/Llama-2-7b-chat-finetune-QA'
# QA_model_name = "meta-llama/Llama-2-7b-chat-hf"
# QA_model_name = "meta-llama/Llama-2-7b-hf"
# QA_model_name = 'tlc4418/pythia_1.4b_sft_policy'
QA_model_name = 'PKU-Alignment/alpaca-7b-reproduced-llama-2'
# QA_model_name = '/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B-Instruct/'
# save_path = '/remote-home1/miintern1/watermark-learnability/data/finetune_ability/MMLU/kalyan_QA_finetuned_MMLU_accuracy.csv'
# save_path = '/remote-home1/miintern1/watermark-learnability/data/finetune_ability/MMLU/llama_2_7b_instruct_MMLU_accuracy.csv'
# save_path = '/remote-home1/miintern1/watermark-learnability/data/finetune_ability/MMLU/pythia_1.4b_sft_policy_MMLU_accuracy.csv'
cor_save_path = '/remote-home1/miintern1/watermark-learnability/data/finetune_ability/MMLU/full_zero_shot_alpaca_MMLU_accuracy.json'
# demo_save_path = '/remote-home1/miintern1/watermark-learnability/data/finetune_ability/MMLU/demo_zero_shot_alpaca_MMLU_accuracy.csv'
demo_save_path = None
tokenizer = AutoTokenizer.from_pretrained(QA_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
chat_model = AutoModelForCausalLM.from_pretrained(QA_model_name, device_map = 'cpu')
# model = task_vector.apply_to(chat_model, scaling_coef = 1.0)

# if not os.path.exists(save_dir):
#     os.mkdir(args.save_dir)

if os.path.exists(cor_save_path):
    with open(cor_save_path, 'r') as f:
        all_cors_dict = json.load(f)
else:
    all_cors_dict = {}

if demo_save_path and os.path.exists(demo_save_path):
    with open(demo_save_path) as f:
        all_demo_dict = json.load(f)
else:
    all_demo_dict = {}

for watermark_name, watermark_config in watermark_configs.items():
    # if watermark_name in all_cors_dict:
    #     print(f"Already processed {watermark_name}")
    #     continue
    watermarked_model = AutoModelForCausalLM.from_pretrained(watermark_name)
    task_vector = TaskVector(vanilla_model, watermarked_model)
    chat_model_copy = copy.deepcopy(chat_model)
    tested_model = task_vector.apply_to(chat_model_copy, scaling_coef= 1.0)
    # tested_model = chat_model
    tested_model.to(device)

    cors_list = []
    demo_file = {}
    subject_accuracy = {}

    for subject in tqdm(subjects, desc=f"Processing {watermark_name}"):
        dev_df = pd.read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None)[:ntrain]
        test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)
        # print(dev_df.shape, test_df.shape)
        cors, acc, generation_demo, prompt_demo = eval(5, subject, tokenizer, tested_model, dev_df, test_df)
        # print(f"{subject}:{acc}")
        cors_list.append(cors)
        subject_accuracy[subject] = acc
        demo_file[subject] = {"prompt_demo":        prompt_demo, 
                            "generation_demo": generation_demo
                            }
    weighted_acc = np.mean(np.concatenate(cors_list))
    subject_accuracy['weighted_acc'] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))
    all_cors_dict[watermark_name] = subject_accuracy
    all_demo_dict[watermark_name] = demo_file

    with open(cor_save_path, 'w') as f:
        json.dump(all_cors_dict, f)
    if demo_save_path:
        with open(demo_save_path, 'w') as f:
            json.dump(all_demo_dict, f)
    del watermarked_model, tested_model
    # torch.cuda.empty_cache()

cors_list = []
demo_file = {}
subject_accuracy = {}
chat_model.to(device)
for subject in tqdm(subjects):
    dev_df = pd.read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None)[:ntrain]
    test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)
    # print(dev_df.shape, test_df.shape)
    cors, acc, generation_demo, prompt_demo = eval(5, "mathematics", tokenizer, chat_model, dev_df, test_df)
    cors_list.append(cors)
    subject_accuracy[subject] = acc
    demo_file[subject] = {"prompt_demo": prompt_demo, 
                        "generation_demo": generation_demo
                        }
weighted_acc = np.mean(np.concatenate(cors_list))
subject_accuracy['weighted_acc'] = weighted_acc
print("Average accuracy: {:.3f}".format(weighted_acc))
all_cors_dict['original_model'] = subject_accuracy
all_demo_dict['original_model'] = demo_file
del chat_model

cors_list = []
demo_file = {}
subject_accuracy = {}
vanilla_model.to(device)
for subject in tqdm(subjects):
    dev_df = pd.read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None)[:ntrain]
    test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)
    # print(dev_df.shape, test_df.shape)
    cors, acc, generation_demo, prompt_demo = eval(5, "mathematics", tokenizer, vanilla_model, dev_df, test_df)
    cors_list.append(cors)
    subject_accuracy[subject] = acc
    demo_file[subject] = {"prompt_demo": prompt_demo, 
                        "generation_demo": generation_demo
                        }
weighted_acc = np.mean(np.concatenate(cors_list))
subject_accuracy['weighted_acc'] = weighted_acc
print("Average accuracy: {:.3f}".format(weighted_acc))
all_cors_dict['base_model'] = subject_accuracy
all_demo_dict['base_model'] = demo_file

with open(cor_save_path, 'w') as f:
    json.dump(all_cors_dict, f)
if demo_save_path:
    with open(demo_save_path, 'w') as f:
        json.dump(all_demo_dict, f)

# with open(save_path.replace("accuracy.csv", "demo.json"), 'w') as f:
#     json.dump(demo_file, f)

