import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import itertools
from tqdm import tqdm

os.environ['http_proxy'] = "http://10.176.52.116:7890"
os.environ['https_proxy'] = "http://10.176.52.116:7890"
os.environ['all_proxy'] = "socks5://10.176.52.116:7891"

num_samples = 10
hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf") # , local_files_only=True
hf_model.half()
model = HookedTransformer.from_pretrained('Llama-2-7b', device="cpu", hf_model=hf_model)
del hf_model
# torch.cuda.empty_cache()
model.eval()

with open("/remote-home/miintern1/watermark-learnability/data/c4/paraphrased_results.json","r") as f:
    data = json.load(f)

original_text = [a.replace('\n', '') for a in data['original_text'][:num_samples]]
paraphrased_text = data['paraphrased_text'][:num_samples]

def prompt_cache_analysis(original_input):
    original_input_ids = model.to_tokens(original_input, prepend_bos=False)#.to('cuda')
    original_logits, original_cache = model.run_with_cache(original_input_ids, return_type='logits')
    original_logits = original_logits.cpu()
    original_cache = {k: v.cpu() for k, v in original_cache.items()}
    return original_cache, original_logits

output_cache = {}
original_cache_list = []
paraphrased_cache_list = []

total_items = len(original_text)
with tqdm(total=total_items) as pbar:
    for original, paraphrase in zip(original_text, paraphrased_text):
        original_cache, original_logits = prompt_cache_analysis(original)
        paraphrased_cache, paraphrased_logits = prompt_cache_analysis(paraphrase)
        original_cache_list.append(original_cache)
        paraphrased_cache_list.append(paraphrased_cache)
        pbar.update(1)
    

analyze_list = [f'blocks.{i}.hook_resid_post' for i in range(32)]
cos_sim_average = []
cos_sim_median = []
for name in analyze_list:
    layer_cos_sim = []
    for i in range(len(original_cache_list)):
        cos_similarity = F.cosine_similarity(original_cache_list[i][name][:,-1,:].squeeze(), paraphrased_cache_list[i][name][:,-1,:].squeeze(), dim=-1)
        layer_cos_sim.append(cos_similarity)
    cos_sim_average.append(np.mean(layer_cos_sim))
    cos_sim_median.append(np.median(layer_cos_sim))

range_limit = len(original_text)
k = num_samples  # Number of unique pairs you want to generate
all_pairs = [(i, j) for i, j in itertools.combinations(range(range_limit), 2)]
selected_pairs = random.sample(all_pairs, k)
print(f"{len(selected_pairs)=}")

comparison_cos_sim_average = []
comparison_cos_sim_median = []
for name in analyze_list:
    layer_cos_sim = []
    for i, j in selected_pairs:
        cos_similarity = F.cosine_similarity(original_cache_list[i][name][:,-1,:].squeeze(), original_cache_list[j][name][:,-1,:].squeeze(), dim=-1)
        # print(cos_similarity)
        layer_cos_sim.append(cos_similarity)
    comparison_cos_sim_average.append(np.mean(layer_cos_sim))
    comparison_cos_sim_median.append(np.median(layer_cos_sim))
print(f"{len(comparison_cos_sim_average)=}")
print(comparison_cos_sim_average)
print(f"{len(comparison_cos_sim_median)=}")
print(comparison_cos_sim_median)

x_labels = list(range(len(analyze_list)))

plt.figure(figsize=(10, 6))
plt.plot(x_labels, cos_sim_average, label='Average Cosine Similarity', color='blue')
plt.plot(x_labels, comparison_cos_sim_average, label='Comparison', color='red')
plt.ylim(0, 1)
plt.xlabel('X Label')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Plot')
plt.legend()
save_path = '/remote-home/miintern1/watermark-learnability/data/Figures/average_cos_sim.png'  
plt.savefig(save_path, format='png')
print(f"Saved to {save_path}")

plt.figure(figsize=(10, 6))
plt.plot(x_labels, cos_sim_median, label='Median Cosine Similarity', color='blue')
plt.plot(x_labels, comparison_cos_sim_median, label='Comparison', color='orange')
plt.ylim(0, 1)
plt.xlabel('X Label')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Plot')
plt.legend()
save_path = '/remote-home/miintern1/watermark-learnability/data/Figures/median_cos_sim.png' 
plt.savefig(save_path, format='png')
print(f"Saved to {save_path}")

