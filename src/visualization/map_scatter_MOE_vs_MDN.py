"""
Compare the best performance seeds of PhA-MOE and MDN after finetuning on the new 35 station dataset
The result is illustrated as scatter plots visuslizing the regression quality
"""

import pandas as pd
import os
import numpy as np
import sys
project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
sys.path.append(project_dir)
import math
from utils.metrics import *
from utils.plots import *
# base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_map' \
#          r'\finetunedModel_forMaps_Feb23_9_2\PACE\robust_wholeband_log_wholeband\seed_45'


plt.close()


# base_dir=fr"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_MOE_PACE_FT_9_4\seed_68"
# model_type = "PhA_MOE"

base_dir=fr"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_PACE_FT_9_4\seed_45"
model_type = "MDN"

base_dir2 = base_dir
base_save_dir = r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\Map_May24'
os.makedirs(base_save_dir, exist_ok=True)
# save_path = os.path.join(base_save_dir, "PACE_map_scatter_MOE.pdf")
save_path = os.path.join(base_save_dir, "PACE_map_scatter_MDN.pdf")


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
]# PACE wavelength
initial_wl=401
integers = [math.floor(num) for num in numbers]
selected_wl = integers

test_targets_path = os.path.join(base_dir, 'all_test_targets.npy')
test_outputs_path = os.path.join(base_dir, 'all_test_outputs.npy')
test_targets = np.load(test_targets_path)
test_outputs = np.load(test_outputs_path)

test_targets_reshaped = test_targets.reshape(-1, 1)  # Reshape to (:, 1)
test_outputs_reshaped = test_outputs.reshape(-1, 1)

test_nrmse_per_element_ = calculate_nrmse_per_element(test_targets_reshaped, test_outputs_reshaped,avg=False)
test_mdsa_ = calculate_mdsa_wl(test_targets_reshaped, test_outputs_reshaped,avg=False)
test_sspb_ = calculate_sspb_wl(test_targets_reshaped, test_outputs_reshaped,avg=False)
test_slope_, test_slope_deviation_ = calculate_slope_wl(test_targets_reshaped, test_outputs_reshaped,avg=False)

nrmse = test_nrmse_per_element_[0]
mdsa = test_mdsa_[0]
sspb = test_sspb_[0]
slope = test_slope_[0]

plot1 = plot_scatter3(test_targets, test_outputs, selected_wl,nrmse, mdsa, sspb, slope,
                     fontsize=42)

# Save the figure
#plot1.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the plot with high quality
plot1.show()
