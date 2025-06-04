
import os
import numpy as np
import sys
project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
sys.path.append(project_dir)
import math
from utils.metrics import *
# from utils.plots import *
"""
Compare estimated aphy signal vs the ground truth
"""
from utils.plots import *

# base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\MDN\seed_44'
# model_type = "MDN"

base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\MDN_MOE_s\seed_51'
model_type = "PhA-MOE"

# base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\VAE_1000\seed_52'
# model_type = "VAE"

# base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21\EMIT\robust_wholeband_robust_wholeband\MLP_1000\seed_51'
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
    403.2254, 410.638, 418.0536, 425.47214, 432.8927, 440.31726, 447.7428,
    455.17035, 462.59888, 470.0304, 477.46292, 484.89743, 492.33292, 499.77142,
    507.2099, 514.6504, 522.0909, 529.5333, 536.9768, 544.42126, 551.8667,
    559.3142, 566.7616, 574.20905, 581.6585, 589.108, 596.55835, 604.0098,
    611.4622, 618.9146, 626.36804, 633.8215, 641.2759, 648.7303, 656.1857,
    663.6411, 671.09753, 678.5539, 686.0103, 693.4677
]  # EMIT selected wavelength
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


file_name_1 = f'EMIT_{model_type}_curve1.pdf'
file_name_50 = f'EMIT_{model_type}_curve50.pdf'
file_name_100 = f'EMIT_{model_type}_curve100.pdf'

save_path_1 = os.path.join(save_dir, file_name_1)
save_path_50 = os.path.join(save_dir, file_name_50)
save_path_100 = os.path.join(save_dir, file_name_100)


test_IDs=np.load("D:/Research/EnvironmentalData/1800Data/test_IDs.npy")


# best_idx
# 289
# best_50_idx
# 139
# best_100_idx
# 54



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
