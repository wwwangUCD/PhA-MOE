
import os
import numpy as np
import sys

import pytorch_lightning.utilities.cli

project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
sys.path.append(project_dir)
import math
from utils.metrics import *
# from utils.plots import *
"""
Compare estimated aphy signal vs the ground truth
"""
from utils.plots import *

# base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21\PACE\robust_wholeband_log_wholeband\MDN\seed_53'
# model_type = "MDN" # top3: ['seed_53', 'seed_57', 'seed_44']

base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb_PACEft\PACE\robust_wholeband_log_wholeband\MDN_MOE_s\52_8_4_1\seed_45'
model_type = "PhA-MOE" # top3:['seed_45', 'seed_62', 'seed_67']

# base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21\PACE\robust_wavelength_log_wavelength\VAE_1000\seed_45'
# model_type = "VAE"

# base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21\PACE\no_wavelength_no_wavelength\MLP_1000\seed_48'
# model_type = "MLP"

test_targets_path = os.path.join(base_dir, 'all_test_targets.npy')
test_outputs_path = os.path.join(base_dir, 'all_test_outputs.npy')
test_targets = np.load(test_targets_path)
test_outputs = np.load(test_outputs_path)

# Calculate test metrics
test_nrmse_per_element_ = calculate_nrmse_per_element(test_targets, test_outputs,avg=False)
test_mdsa_ = calculate_mdsa_wl(test_targets, test_outputs,avg=False)
test_sspb_ = calculate_sspb_wl(test_targets, test_outputs,avg=False)
test_slope_, test_slope_deviation_ = calculate_slope_wl(test_targets, test_outputs,avg=False)

# Calculate average values
avg_test_nrmse = np.mean(test_nrmse_per_element_)
avg_test_mdsa = np.mean(test_mdsa_)
avg_test_sspb = np.mean(np.abs(test_sspb_))
avg_test_slope_deviation = np.mean(test_slope_deviation_)

numbers = [
    403, 405, 408, 410, 413, 415, 418, 420, 422, 425, 427, 430, 432, 435,
    437, 440, 442, 445, 447, 450, 452, 455, 457, 460, 462, 465, 467, 470, 472,
    475, 477, 480, 482, 485, 487, 490, 492, 495, 497, 500, 502, 505, 507, 510,
    512, 515, 517, 520, 522, 525, 527, 530, 532, 535, 537, 540, 542, 545, 547,
    550, 553, 555, 558, 560, 563, 565, 568, 570, 573, 575, 578, 580, 583, 586,
    588, 591, 593, 596, 598, 601, 603, 605, 608, 610, 613, 615, 618, 620, 623,
    625, 627, 630, 632, 635, 637, 640, 641, 642, 643, 645, 646, 647, 648, 650,
    651, 652, 653, 655, 656, 657, 658, 660, 661, 662, 663, 665, 666, 667, 668,
    670, 671, 672, 673, 675, 676, 677, 678, 679, 681, 682, 683, 684, 686, 687,
    688, 689, 691, 692, 693, 694, 696, 697, 698, 699
]  # PACE selected wavelength
initial_wl=401
integers = [math.floor(num) for num in numbers]
selected_wl = integers

save_dir = r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\scatters'
os.makedirs(save_dir, exist_ok=True)


def calculate_nrmse_per_sample(y_true, y_pred):
    """
    Compute NRMSE for each sample in the dataset.

    Parameters:
    - y_true: np.array of shape (num_samples, num_features) - ground truth values
    - y_pred: np.array of shape (num_samples, num_features) - predicted values

    Returns:
    - nrmse_values: np.array of shape (num_samples,) containing NRMSE for each sample
    """
    mse = np.mean((y_true - y_pred) ** 2, axis=1)  # Mean squared error per sample
    rmse = np.sqrt(mse)  # Root mean squared error per sample
    norm_factor = np.max(y_true, axis=1) - np.min(y_true, axis=1)  # Normalization factor per sample
    norm_factor[norm_factor == 0] = 1  # Prevent division by zero
    nrmse_values = rmse / norm_factor  # Normalized RMSE per sample

    return nrmse_values

nrmse_per_sample = calculate_nrmse_per_sample(test_targets, test_outputs)

# Rank the samples based on NRMSE (ascending order, lower is better)
sorted_indices = np.argsort(nrmse_per_sample)

# Get indices of the best, 50th best, and 100th best samples
best_idx = sorted_indices[0]
best_50_idx = sorted_indices[49] if len(sorted_indices) > 49 else None
best_100_idx = sorted_indices[99] if len(sorted_indices) > 99 else None


file_name_1 = f'PACE_{model_type}_curve1.pdf'
file_name_50 = f'PACE_{model_type}_curve50.pdf'
file_name_100 = f'PACE_{model_type}_curve100.pdf'

save_path_1 = os.path.join(save_dir, file_name_1)
save_path_50 = os.path.join(save_dir, file_name_50)
save_path_100 = os.path.join(save_dir, file_name_100)

best_idx=289
best_50_idx=139
best_100_idx=54
test_IDs=np.load("D:/Research/EnvironmentalData/1800Data/test_IDs.npy")
# the old version uses plot_aphy_comparison1
plt = plot_aphy_comparison1_ID(selected_wl, test_targets, test_outputs, best_idx, IDs=test_IDs, linewidth=4, fontsize=44)
plt.savefig(save_path_1, bbox_inches='tight', dpi=300)  # Save the figure
plt.show()

plt = plot_aphy_comparison1_ID(selected_wl, test_targets, test_outputs, best_50_idx, IDs=test_IDs, linewidth=4, fontsize=44)
plt.savefig(save_path_50, bbox_inches='tight', dpi=300)  # Save the figure
plt.show()

plt = plot_aphy_comparison1_ID(selected_wl, test_targets, test_outputs, best_100_idx, IDs=test_IDs, linewidth=4, fontsize=44)
plt.savefig(save_path_100, bbox_inches='tight', dpi=300)  # Save the figure
plt.show()


