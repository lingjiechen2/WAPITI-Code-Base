import torch
import os
os.environ['http_proxy'] = "http://10.176.52.116:7890"
os.environ['https_proxy'] = "http://10.176.52.116:7890"
os.environ['all_proxy'] = "socks5://10.176.52.116:7891"
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,Cache
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
sys.path.append(('../'))
sys.path.append(('../../'))
import logging
from task_vector import TaskVector
import logging
import copy
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./runs/refinetuning")


import subprocess
# from FileNotFoundError import FileNotFoundError

logging.basicConfig(
    filename='/remote-home/miintern1/watermark-learnability/logs/refinetuning.txt',  # Specify the log file name
    level=logging.INFO,          # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Set the log message format
)

def get_gpu_info():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], encoding='utf-8')
        logging.info(result)
    except FileNotFoundError:
        logging.info("nvidia-smi not found. Make sure you have NVIDIA drivers installed and nvidia-smi in your PATH.")

get_gpu_info()

max_length = 250
min_length = 250
num_samples = 512
batch_size = 8
coefficient = 1.0
save_path = "/remote-home/share/data/openwebtext"
checkpoint_save_path = "/remote-home/miintern1/watermark-learnability/data/openwebtext/kgw_k1_delta_2_math"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

accelerator = Accelerator()
vanilla_model_name = "meta-llama/Llama-2-7b-hf"
tested_model_name = "RohitSahoo/llama-2-7b-chat-hf-math-ft-V1"
watermarked_model_name = "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"
vanilla_model = AutoModelForCausalLM.from_pretrained(vanilla_model_name, device_map = 'cpu')
tested_model = AutoModelForCausalLM.from_pretrained(tested_model_name, device_map = 'cpu')
watermarked_model = AutoModelForCausalLM.from_pretrained(watermarked_model_name, device_map = 'cpu')
task_vector = TaskVector(vanilla_model, watermarked_model)
# model = task_vector.apply_to(tested_model, scaling_coef=coefficient)
model = copy.deepcopy(vanilla_model)
model.half()
model.to(device)
del vanilla_model, tested_model, watermarked_model, task_vector
tokenizer = AutoTokenizer.from_pretrained(tested_model_name)
logging.info(f"Model and tokenizer loaded: {tested_model_name[40:]}")
get_gpu_info()

dataset = load_from_disk(save_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# def filter_length(example):
#         return len(tokenizer(example['text'], truncation=True, max_length=max_length)["input_ids"]) >= min_length

# def encode(examples):
#     trunc_tokens = tokenizer(
#         examples['text'],
#         truncation=True,
#         padding=True,
#         max_length=max_length,
#         return_tensors="pt"
#     ).to(device)
#     # Examples are truncated to max_length, which comprises the possible generation prompt and the text to be generated
#     examples["text"] = tokenizer.batch_decode(trunc_tokens["input_ids"], skip_special_tokens=True)
#     prompt = tokenizer(
#         examples["text"], truncation=True, padding=True, max_length=50, return_tensors="pt",
#     ).to(device)
#     examples["prompt_text"] = tokenizer.batch_decode(prompt["input_ids"], skip_special_tokens=True)
#     examples["input_ids"] = prompt["input_ids"]
#     examples["attention_mask"] = prompt["attention_mask"]
#     examples["text_completion"] = tokenizer.batch_decode(
#         trunc_tokens["input_ids"][:, 50:], skip_special_tokens=True
#     )
#     return examples

# dataset = dataset.filter(filter_length)
# # Set how many samples will be skipped
# dataset = dataset.map(encode, batched=True)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size)

# prompts = []
# human_text = []
# prompt_text = []
# full_human_text = []
# for batch in dataloader:
#     if len(human_text) >= num_samples:
#         break
#     if (type(batch["input_ids"]) == list):
#         batch["input_ids"] = torch.stack(batch["input_ids"], dim=1).to(device)
#     if (type(batch["attention_mask"]) == list):
#         batch["attention_mask"] = torch.stack(batch["attention_mask"], dim=1).to(device)
#     prompts.append(batch)
#     human_text.extend(batch["text_completion"])
#     prompt_text.extend(batch["prompt_text"])
#     full_human_text.extend(batch["text"])
# human_text = human_text[:num_samples]
# prompt_text = prompt_text[:num_samples]
# full_human_text = full_human_text[:num_samples]
# raw_input = {
#     "prompts": prompts,
#     "human_text": human_text,
#     "prompt_text": prompt_text,
#     "full_human_text": full_human_text,
# }

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999))
total_steps = int(2500 * (64/batch_size))
warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
# model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
model.train()

global_step = 0
save_steps = int(500 * (64/batch_size))
logging.info(f"Save model checkpoint every {save_steps} steps, Total steps: {total_steps}")
# logging.info(f"The length of dataloader is {len(prompts)}")
logging.info(f"Length of dataloader: {len(dataloader)}")

logging.info(f"Start training for {total_steps} steps")
pbar = tqdm(total=total_steps, desc="Training Progress", dynamic_ncols=True)
for epoch in range(total_steps // len(dataloader) + 1):
    # logging.info(f"Epoch {epoch}/{total_steps // len(prompts)}")
    logging.info(f"Epoch {epoch}/{total_steps // len(dataloader)}")
    # for batch in dataloader:
        # inputs = {key: val.to(device) for key, val in batch.items()}
    # for batch in prompts:
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # get_gpu_info()
        optimizer.zero_grad()
        # outputs = model(**inputs, labels=inputs['input_ids'])
        # print(batch.keys())
        # cache = Cache(model, batch['input_ids'], batch['attention_mask'])
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels=input_ids)#, cache=cache)
        loss = outputs.loss
        logging.info(f"Global step: {global_step}, Loss: {loss.item()}")
        logging.info(f"{batch['input_ids'].shape=}")
        # accelerator.backward(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1
        writer.add_scalar("Loss/train", loss.item(), global_step)
        if global_step % save_steps == 0:
            accelerator.save_state(f"{checkpoint_save_path}/checkpoint-{global_step}")
        pbar.set_description(f"Training Progress (loss: {loss.item():.4f})")
        pbar.update(1)
        del batch, outputs, loss
        torch.cuda.empty_cache()

        if global_step >= total_steps:
            break
    if global_step >= total_steps:
        break
pbar.close()
writer.close()