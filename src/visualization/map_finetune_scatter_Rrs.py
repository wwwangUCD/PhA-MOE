"""
Use scatter plots to compare the field Rrs with the PACE Rrs.
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
data_dir = "D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers"
base_dir=fr"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_MOE_PACE_FT_9_4\seed_68"
base_dir2 = base_dir
base_save_dir = r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\Map_May24'
model_type = "PhA_MOE"

remove_idx = 4 # the station b02 data does not present in 1024.nc's map
rrs_path = os.path.join(data_dir,'fine_tune_rrs.npy')
rrs_data_array_ft_test = np.load(rrs_path)
rrs_data_array_ft_test = np.delete(rrs_data_array_ft_test, remove_idx, axis=0)




rrs_fromMap_path = os.path.join(data_dir,'ft_Rrs_all_stations.npy')
Rrs_data_fromMap_ft_test = np.load(rrs_fromMap_path)

remove_idx = [ 6, 16] # the station b02 data does not present in 1024.nc's map
rrs_path = os.path.join(data_dir,'evaluation_rrs.npy')
rrs_data_array_ft_train = np.load(rrs_path)
rrs_data_array_ft_train = np.delete(rrs_data_array_ft_train, remove_idx, axis=0)




rrs_fromMap_path = os.path.join(data_dir,'ft_train_Rrs_all_stations.npy')
Rrs_data_fromMap_ft_train = np.load(rrs_fromMap_path)



# rrs_data_array=rrs_data_array_ft_train
#
# Rrs_data_fromMap=Rrs_data_fromMap_ft_train

# Concatenate Rrs data arrays
rrs_data_array = np.concatenate([rrs_data_array_ft_train, rrs_data_array_ft_test], axis=0)

# Concatenate Rrs-from-map arrays
Rrs_data_fromMap = np.concatenate([Rrs_data_fromMap_ft_train, Rrs_data_fromMap_ft_test], axis=0)





initial_wl = 400  # initial wavelength of the loaded data, both aphy and Rrs

selected_wl_range = [
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
]  # PACE wavelength

selected_wl_indices = [i - initial_wl for i in selected_wl_range]

Rrs_data = rrs_data_array[:, selected_wl_indices]


"""
remove outlier

Rrs_data = np.clip(Rrs_data, 1e-5, None)
Rrs_data_fromMap = np.clip(Rrs_data_fromMap, 1e-5, None)

log_Rrs = np.log10(Rrs_data)
log_Rrs_hat = np.log10(Rrs_data_fromMap)

relative_diff = (log_Rrs_hat - log_Rrs) / log_Rrs
avg_relative_diff_per_sample = np.mean(np.abs(relative_diff), axis=1)

sorted_indices = np.argsort(avg_relative_diff_per_sample)
sorted_avg_values = avg_relative_diff_per_sample[sorted_indices]

idx_to_remove=sorted_indices[-2:]
worest six:
30, 18, 17, 12, 21, 20
"""
idx_to_remove=[30, 18, 17, 12, 21, 20]

# test_targets = Rrs_data
# test_outputs = Rrs_data_fromMap

test_targets = np.delete(Rrs_data, idx_to_remove, axis=0)
test_outputs = np.delete(Rrs_data_fromMap, idx_to_remove, axis=0)



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


plot1 = plot_scatter3_Rrs(test_targets, test_outputs, selected_wl_range,nrmse, mdsa, sspb, slope,
                     fontsize=42)

save_path = os.path.join(base_save_dir, "PACE_map_scatter_Rrs.pdf")

# Save the figure
plot1.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the plot with high quality
plot1.show()
