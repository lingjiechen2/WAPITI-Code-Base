import json
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from watermarks.watermark_types import WatermarkType
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


watermark_file_path = "/remote-home/miintern1/watermark-learnability/data/c4/aronson_watermark_vector.json"
# watermark_type = WatermarkType.KGW
ref_data_path = "/remote-home/miintern1/watermark-learnability/data/c4/auroc_ref_distribution_llama_c4.json"

with open(ref_data_path, "r") as f:
    ref_dist_data = json.load(f)
    ref_dist = ref_dist_data["data"]
# with open('/remote-home/miintern1/watermark-learnability/data/c4/watermark_vector_evaluation.json', "r") as f:
with open(watermark_file_path, "r") as f:
    samples_dict = json.load(f)

final_result_dict = {}

for watermark, watermark_data in samples_dict.items():
    final_result_dict[watermark] = {}
    wc = watermark_data['watermark_config']
    for name, ref_dist_data in ref_dist.items():
        ref_dist_wc = ref_dist_data["watermark_config"]
        if (
                # ref_dist_wc["type"] == wc['type'] and 
                # ref_dist_wc["gamma"] == wc["gamma"] and
                # ref_dist_wc["seeding_scheme"] == wc["seeding_scheme"]
                ref_dist_wc == wc
            ):
                print(name)
                watermark_scores = watermark_data
                null_scores = ref_dist_data["p_values"]
                break

    print(watermark_data.keys())
    coefficient_list = [i for i in watermark_scores if is_float(i)]
    auroc_list = []
    fpr_list = []
    tpr_list = []
    # print(coefficient_list)
    for coefficient in coefficient_list:
        watermark_scores_coefficient = watermark_scores[coefficient]['watermarked_scores']
        null_scores_coefficient = null_scores[:len(watermark_scores_coefficient)]
        # print(f"{len(watermark_scores_coefficient)=}, {len(null_scores_coefficient)=}")
        y_true = np.concatenate([np.zeros_like(watermark_scores_coefficient), np.ones_like(null_scores_coefficient)])
        y_scores = np.concatenate([watermark_scores_coefficient, null_scores_coefficient])
        # print(f"{len(y_true)=}, {len(y_scores)=}")
        auroc = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        # print(f"{auroc=}")

        auroc_list.append(auroc)
        fpr_list.append(np.array(fpr))
        tpr_list.append(np.array(tpr))
        # print(f"{type(fpr_interp)=}, {type(tpr_interp)=},{type(auroc)=}")
    final_result_dict[watermark]['auroc_list'] = auroc_list
    final_result_dict[watermark]['fpr_list'] = fpr_list
    final_result_dict[watermark]['tpr_list'] = tpr_list

    # print(f"{len(fpr_list)=}, {len(tpr_list)=}")

    colormap = plt.cm.viridis  # You can choose other colormaps like plt.cm.plasma, plt.cm.inferno, etc.
    num_colors = 20  # Number of different coefficients
    # Normalize the color range
    norm = plt.Normalize(vmin=0, vmax=num_colors - 1)

    # Generate the gradient colors
    color_list = [colormap(norm(i)) for i in range(num_colors)]
    for i, coefficient in enumerate(coefficient_list):
        plt.plot(fpr_list[i], tpr_list[i], color=color_list[i], label=f'coefficient: {coefficient}')
    plt.title('ROC curves for different coefficients')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True, which='both', linestyle='--')
    # plt.legend()
    plt.savefig('roc_curves.png')

    plt.figure()
    plt.plot(coefficient_list, auroc_list)
    plt.title('AUROC score for different coefficients')
    plt.xlabel('Coefficient')
    plt.ylabel('AUROC')
    plt.grid(True, which='both', linestyle='--')
    plt.savefig('auroc_scores.png')

with open("/remote-home/miintern1/watermark-learnability/data/c4/aronson_final_result_dict.pkl", "wb") as f:
    pickle.dump(final_result_dict, f)
