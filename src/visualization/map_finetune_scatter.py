"""
Use the best performance seeds of PhA-MOE finetuning on the new 35 station dataset, and
compare its performance on field Rrs vs PACE Rrs
The result is illustrated as scatter plots visuslizing the regression quality, and spectra signals to compare the
estimation quality

To make a fair comparision, we remove the idx=4 station as it does not have corresponding PACE Rrs measurement
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

test_targets_path = os.path.join(base_dir, 'aphy_evalStation_target.npy')
test_outputs_path = os.path.join(base_dir, 'aphy_evalStation_output.npy')
test_targets = np.load(test_targets_path)
test_outputs = np.load(test_outputs_path)
remove_idx = 4 # the station b02 data does not present in 1024.nc's map
test_targets = np.delete(test_targets, remove_idx, axis=0)
test_outputs = np.delete(test_outputs, remove_idx, axis=0)

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

test_outputs_map_path = os.path.join(base_dir2, 'aphy_evalStation_mapRrs_output.npy')
test_outputs_map = np.load(test_outputs_map_path)
test_outputs_map_reshaped = test_outputs_map.reshape(-1, 1)

test_nrmse_per_element_map_ = calculate_nrmse_per_element(test_targets_reshaped, test_outputs_map_reshaped, avg=False)
test_mdsa_map_ = calculate_mdsa_wl(test_targets_reshaped, test_outputs_map_reshaped, avg=False)
test_sspb_map_ = calculate_sspb_wl(test_targets_reshaped, test_outputs_map_reshaped, avg=False)
test_slope_map_, test_slope_deviation_map_ = calculate_slope_wl(test_targets_reshaped, test_outputs_map_reshaped, avg=False)

nrmse_map = test_nrmse_per_element_map_[0]
mdsa_map = test_mdsa_map_[0]
sspb_map = test_sspb_map_[0]
slope_map = test_slope_map_[0]

# plot1 = plot_scatter3(test_targets, test_outputs, selected_wl,nrmse, mdsa, sspb, slope,
#                      fontsize=30, title_str=r'$a_{phy}$ from measured $R_{rs}$ ')
plot1 = plot_scatter3(test_targets, test_outputs, selected_wl,nrmse, mdsa, sspb, slope,
                     fontsize=42)

save_path = os.path.join(base_save_dir, "PACE_map_scatter.pdf")

# Save the figure
plot1.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the plot with high quality
plot1.show()

plot2 = plot_scatter3(test_targets, test_outputs_map, selected_wl,nrmse_map, mdsa_map, sspb_map, slope_map,
                     fontsize=42)#, title_str='$a_{phy}$ from PACE map \'s $R_{rs}$ ')
save_path = os.path.join(base_save_dir, "PACE_map_scatter_fromRrs.pdf")
plot2.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the plot with high quality
plot2.show()


rrs_path = os.path.join(data_dir,'fine_tune_rrs.npy')
rrs_data_array = np.load(rrs_path)
rrs_data_array = np.delete(rrs_data_array, remove_idx, axis=0)

rrs_fromMap_path = os.path.join(data_dir,'ft_Rrs_all_stations.npy')
Rrs_data_fromMap = np.load(rrs_fromMap_path)



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

# load the station and date labels
csv_path = os.path.join(data_dir,"ft_data.csv")
eval_data = pd.read_csv(csv_path)
eval_data = eval_data.drop(index=remove_idx)
eval_dates = eval_data.iloc[:, 0].values
eval_stations = eval_data.iloc[:, 1].values

for i in range(len(eval_dates)):
    date = eval_dates[i]
    station = eval_stations[i]

    plt=plot_rrs_comparison(selected_wl, Rrs_data, Rrs_data_fromMap, i, linewidth=5, fontsize=46,
                          title_str=r'Station:'+station)
    save_dir = os.path.join(base_save_dir, str(date))
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    save_path = os.path.join(save_dir, f'Rrs_Station_{station}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

for i in range(len(eval_dates)):
    date = eval_dates[i]
    station = eval_stations[i]

    # Call the plot_aphy_comparison2 function to generate the plot
    plt = plot_aphy_comparison2(selected_wl, test_targets, test_outputs, test_outputs_map, i, linewidth=5, fontsize=46,
                                title_str=f'Station: {station}')

    # Define the path to save the plot
    save_dir = os.path.join(base_save_dir, str(date))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'Aphy_Station_{station}.pdf')

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the plot with high quality

    # Show the figure
    plt.show()

