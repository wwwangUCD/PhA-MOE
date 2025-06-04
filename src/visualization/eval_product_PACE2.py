
import os
import numpy as np
import sys
project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
sys.path.append(project_dir)
import math
from utils.metrics import *
# from utils.plots import *
"""
Evaluate estimated aphy on each wavelength and render using corresponding light colors
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

colors = [wavelength_to_rgb(wl) for wl in selected_wl]


save_name=['NRMSE','MDSA','SSPB','Slop']
metric_names = ['NRMSE', 'MDSA', '|SSPB|', '|Slop-1|']
metric_values = [test_nrmse_per_element_, test_mdsa_, np.abs(test_sspb_), test_slope_deviation_]
y_axis_limits = [(0, 5), (0, 60), (0, 35), (0, 0.3)]
for i in range(len(metric_names)):

    fig = plot_metrics_new(metric_values[i], selected_wl, colors, metric_names[i], bar_width=2, fontsize=44)
    ax = fig.axes[0]
    ax.set_ylim(y_axis_limits[i])
    file_name = f'PACE_{model_type}_rainbow_{save_name[i]}.pdf'
    save_path = os.path.join(save_dir, file_name)
    # fig.savefig(save_path, bbox_inches='tight', dpi=300)
    fig.show()


# filename = f'{save_dir}\\PACE_{model_type}_{closest_wl}nm_scatter.pdf'
# plt_obj.savefig(filename, dpi=300, bbox_inches='tight')
# plt_obj.show()
