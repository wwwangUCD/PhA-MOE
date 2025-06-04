
import os
import numpy as np
import sys
project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
sys.path.append(project_dir)
import math
from utils.metrics import *
# from utils.plots import *
"""
Plot scatter plot to visualize the regression quality of estimated aphy
"""
from utils.plots import *
is_second=False
# base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\MDN\seed_44'
# model_type = "MDN" # top3 seeds: 44,45,48
# is_second=True


base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\MDN_MOE_s\seed_51'
model_type = "PhA-MOE"# top3 seeds: 51,52,45

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

wl1=440
closest_idx = min(range(len(selected_wl)), key=lambda i: abs(selected_wl[i] - wl1))
closest_wl = selected_wl[closest_idx]
y_true = test_targets[:,closest_idx]
y_est = test_outputs[:,closest_idx]
nrmse = test_nrmse_per_element_[closest_idx]
mdsa = test_mdsa_[closest_idx]
sspb = test_sspb_[closest_idx]
slope = test_slope_[closest_idx]

plt_obj =plot_scatter(y_true, y_est, nrmse, mdsa, sspb, slope, closest_wl, model_type,linewidth=4, fontsize=46)
if is_second:
    filename = f'{save_dir}\\EMIT_2nd_{model_type}_{closest_wl}nm_scatter.pdf'
else:
    filename = f'{save_dir}\\EMIT_{model_type}_{closest_wl}nm_scatter.pdf'
#plt_obj.savefig(filename, dpi=300, bbox_inches='tight')
plt_obj.show()

wl1=621
closest_idx = min(range(len(selected_wl)), key=lambda i: abs(selected_wl[i] - wl1))
closest_wl = selected_wl[closest_idx]
y_true = test_targets[:,closest_idx]
y_est = test_outputs[:,closest_idx]
nrmse = test_nrmse_per_element_[closest_idx]
mdsa = test_mdsa_[closest_idx]
sspb = test_sspb_[closest_idx]
slope = test_slope_[closest_idx]


plt_obj =plot_scatter(y_true, y_est, nrmse, mdsa, sspb, slope, closest_wl, model_type,linewidth=4, fontsize=44)
# filename = f'{save_dir}\\EMIT_{model_type}_{closest_wl}nm_scatter.png'
if is_second:
    filename = f'{save_dir}\\EMIT_2nd_{model_type}_{closest_wl}nm_scatter.pdf'
else:
    filename = f'{save_dir}\\EMIT_{model_type}_{closest_wl}nm_scatter.pdf'
#plt_obj.savefig(filename, dpi=300, bbox_inches='tight')
plt_obj.show()
# plt_obj.savefig(filename, dpi=300, bbox_inches='tight')


wl1=673
closest_idx = min(range(len(selected_wl)), key=lambda i: abs(selected_wl[i] - wl1))
closest_wl = selected_wl[closest_idx]
y_true = test_targets[:,closest_idx]
y_est = test_outputs[:,closest_idx]
nrmse = test_nrmse_per_element_[closest_idx]
mdsa = test_mdsa_[closest_idx]
sspb = test_sspb_[closest_idx]
slope = test_slope_[closest_idx]


plt_obj =plot_scatter(y_true, y_est, nrmse, mdsa, sspb, slope, closest_wl, model_type,linewidth=4, fontsize=42)
if is_second:
    filename = f'{save_dir}\\EMIT_2nd_{model_type}_{closest_wl}nm_scatter.pdf'
else:
    filename = f'{save_dir}\\EMIT_{model_type}_{closest_wl}nm_scatter.pdf'
#plt_obj.savefig(filename, dpi=300, bbox_inches='tight')
plt_obj.show()
