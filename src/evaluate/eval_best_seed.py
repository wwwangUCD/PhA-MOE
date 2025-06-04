"""

"""
import os
import numpy as np
import sys
project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
sys.path.append(project_dir)

from utils.metrics import *


def process_folder1(folder1_path):
    """
    Process the folder1 directory to calculate average metrics.

    Args:
    folder1_path (str): Path to the folder1 directory

    Returns:
    tuple: Average test metrics and average validation metrics
    """
    file_names = []
    test_metrics = []
    val_metrics = []
    idx=0 # only evaluate the first 50 random seeds
    for folder2 in os.listdir(folder1_path):
        folder2_path = os.path.join(folder1_path, folder2)
        if os.path.isdir(folder2_path):
            test_targets_path = os.path.join(folder2_path, 'all_test_targets.npy')
            test_outputs_path = os.path.join(folder2_path, 'all_test_outputs.npy')
            test_targets = np.load(test_targets_path)
            test_outputs = np.load(test_outputs_path)

            test_nrmse_per_element = calculate_nrmse_per_element(test_targets, test_outputs)
            test_mdsa = calculate_mdsa_wl(test_targets, test_outputs)

            test_sspb_ = calculate_sspb_wl(test_targets, test_outputs, avg=False)
            test_slope_, test_slope_deviation_ = calculate_slope_wl(test_targets, test_outputs, avg=False)
            avg_test_sspb = np.mean(np.abs(test_sspb_))
            avg_test_slope_deviation = np.mean(test_slope_deviation_)

            # test_sspb = calculate_sspb_wl(test_targets, test_outputs)
            # test_slope = calculate_slope_wl(test_targets, test_outputs)

            test_metrics.append([test_nrmse_per_element, test_mdsa, avg_test_sspb, avg_test_slope_deviation])

            # Load validation targets and outputs
            val_targets_path = os.path.join(folder2_path, 'all_val_targets.npy')
            val_outputs_path = os.path.join(folder2_path, 'all_val_outputs.npy')
            val_targets = np.load(val_targets_path)
            val_outputs = np.load(val_outputs_path)

            # Calculate validation metrics
            val_nrmse_per_element = calculate_nrmse_per_element(val_targets, val_outputs)
            val_mdsa = calculate_mdsa_wl(val_targets, val_outputs)
            val_sspb_ = calculate_sspb_wl(val_targets, val_outputs, avg=False)
            val_slope_, val_slope_deviation_ = calculate_slope_wl(val_targets, val_outputs, avg=False)
            avg_val_sspb = np.mean(np.abs(val_sspb_))
            avg_val_slope_deviation = np.mean(val_slope_deviation_)
            # val_sspb = calculate_sspb_wl(val_targets, val_outputs)
            # val_slope = calculate_slope_wl(val_targets, val_outputs)

            val_metrics.append([val_nrmse_per_element, val_mdsa, avg_val_sspb, avg_val_slope_deviation])
            file_names.append(folder2)
    test_metrics = np.array(test_metrics)
    val_metrics = np.array(val_metrics)
    best_idx = np.argmin(val_metrics[:, 0])
    # best_idx = np.argmin(test_metrics[:, 0])
    best_val_metrics = val_metrics[best_idx]
    best_test_metrics = test_metrics[best_idx]
    best_file_name = file_names[best_idx]

    top3_indices = np.argsort(val_metrics[:, 0])[:3]
    top3_val_metrics = val_metrics[top3_indices]
    top3_test_metrics = test_metrics[top3_indices]
    top3_file_names = [file_names[i] for i in top3_indices]
    # return top3_test_metrics, top3_val_metrics, top3_file_names
    return best_test_metrics, best_val_metrics,best_file_name
import numpy as np


def print_average_metrics(avg_test_metrics, avg_val_metrics):
    # Print Validation Metrics
    print(f"Validation Metrics - NRMSE per element: {avg_val_metrics[0]:.4f}, "
          f" MDSA: {avg_val_metrics[1]:.4f}, "
          f"|SSPB|: {avg_val_metrics[2]:.4f}, |Slope-1|: {avg_val_metrics[3]:.4f}")

    # Print Test Metrics
    print(f"Test Metrics - NRMSE per element: {avg_test_metrics[0]:.4f}, "
          f"MDSA: {avg_test_metrics[1]:.4f}, "
          f"|SSPB|: {avg_test_metrics[2]:.4f}, |Slope-1|: {avg_test_metrics[3]:.4f}")



best_test_metrics, best_val_metrics,best_file_name = process_folder1(base_dir)
print_average_metrics(best_test_metrics, best_val_metrics)
print(f"Best model folder: {best_file_name}")


## EMIT
# MDN
# base_dir=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\MDN'
# Validation Metrics - NRMSE per element: 1.0809,  MDSA: 26.7771, |SSPB|: 6.7556, |Slope-1|: 0.0574
# Test Metrics - NRMSE per element: 0.9123, MDSA: 25.9890, |SSPB|: 7.5819, |Slope-1|: 0.0602
# Best model folder: seed_44

# top3_val_metrics
# array([[ 1.08087549, 26.77705944,  6.75558954,  0.05740012],
#        [ 1.12238697, 29.5280984 ,  5.47608137,  0.0718214 ],
#        [ 1.1281552 , 24.9858892 ,  3.00947994,  0.05315392]])
# top3_test_metrics
# array([[ 0.91226413, 25.98903328,  7.58186042,  0.06015282],
#        [ 0.84309159, 26.03609413,  3.83142292,  0.06367731],
#        [ 1.03431874, 26.79827362,  3.10043275,  0.04166133]])
# top3_file_names
# ['seed_44', 'seed_45', 'seed_48']


# base_dir=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\MDN_MOE_s'


# MDN MOE
# base_dir=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\MDN_MOE_s'
# Validation Metrics - NRMSE per element: 1.0919,  MDSA: 28.7022, |SSPB|: 4.8839, |Slope-1|: 0.0586
# Test Metrics - NRMSE per element: 0.8808, MDSA: 26.2904, |SSPB|: 4.5633, |Slope-1|: 0.0546
# Best model folder: seed_51

# top3_val_metrics
# array([[ 1.09187457, 28.70216191,  4.88388449,  0.05856009],
#        [ 1.20335757, 31.7712599 ,  5.00920683,  0.10270883],
#        [ 1.25765821, 33.08037043,  4.46371078,  0.1328463 ]])
# top3_test_metrics
# array([[ 0.88082843, 26.29044533,  4.56325531,  0.05460476],
#        [ 0.90180059, 27.57410973,  4.78276372,  0.09360884],
#        [ 0.99624205, 28.84642988,  3.94899815,  0.11014365]])
# top3_file_names
# ['seed_51', 'seed_52', 'seed_45']




# VAE
# base_dir=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\VAE_1000'
# Validation Metrics - NRMSE per element: 3.3565,  MDSA: 45.9448, |SSPB|: 12.8563, |Slope-1|: 0.1382
# Test Metrics - NRMSE per element: 4.0981, MDSA: 47.2836, |SSPB|: 14.8445, |Slope-1|: 0.1366
# Best model folder: seed_52

# MLP
# base_dir=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21\EMIT\robust_wholeband_robust_wholeband\MLP_1000'
# Validation Metrics - NRMSE per element: 1.8503,  MDSA: 58.0092, |SSPB|: 21.3360, |Slope-1|: 0.2562
# Test Metrics - NRMSE per element: 2.1211, MDSA: 55.7351, |SSPB|: 12.1650, |Slope-1|: 0.2627
# Best model folder: seed_51

## PACE
# MDN
# base_dir=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21\PACE\robust_wholeband_log_wholeband\MDN'
# Validation Metrics - NRMSE per element: 1.3961,  MDSA: 35.0878, |SSPB|: 8.6757, |Slope-1|: 0.0923
# Test Metrics - NRMSE per element: 1.2405, MDSA: 34.5668, |SSPB|: 8.7202, |Slope-1|: 0.0881
# Best model folder: seed_53

# second best model:
# top3_val_metrics
# array([[ 1.39608961, 35.08776617,  8.67572294,  0.09233065],
#        [ 1.74933666, 37.02832171, 13.07148164,  0.04138449],
#        [ 1.83686068, 41.13487684,  7.44619924,  0.14278379]])
#
# top3_test_metrics
# array([[ 1.24047816, 34.56678647,  8.72017783,  0.08814852],
#        [ 1.63257574, 43.49079471, 21.13020652,  0.04468794],
#        [ 1.46781814, 39.08446108, 10.24031184,  0.13647551]])
#
# top3_file_names
# ['seed_53', 'seed_57', 'seed_44']

# MDN MoE
# base_dir=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb_PACEft\PACE\robust_wholeband_log_wholeband\MDN_MOE_s\52_8_4_1'
# Validation Metrics - NRMSE per element: 1.2177,  MDSA: 36.4607, |SSPB|: 6.7909, |Slope-1|: 0.1256
# Test Metrics - NRMSE per element: 1.2718, MDSA: 35.6033, |SSPB|: 4.9613, |Slope-1|: 0.1216
# Best model folder: seed_45

# top3_val_metrics
# array([[ 1.21772595, 36.46067497,  6.79090164,  0.12563395],
#        [ 1.37532163, 39.05945147,  8.55082166,  0.08925783],
#        [ 1.40873915, 42.81162065,  6.99636547,  0.15887474]])
# top3_test_metrics
# array([[ 1.27177947, 35.60334759,  4.96126206,  0.1216416 ],
#        [ 1.25475206, 37.6488554 , 10.60015096,  0.09195941],
#        [ 1.33485382, 47.49322385, 10.73183649,  0.15545145]])
# top3_file_names
# ['seed_45', 'seed_62', 'seed_67']




# MLP
# base_dir=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21\PACE\no_wavelength_no_wavelength\MLP_1000'
# Validation Metrics - NRMSE per element: 2.8077,  MDSA: 46.7230, |SSPB|: 11.3483, |Slope-1|: 0.2393
# Test Metrics - NRMSE per element: 3.0774, MDSA: 47.2969, |SSPB|: 10.0739, |Slope-1|: 0.2312
# Best model folder: seed_48

# VAE
# base_dir=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21\PACE\robust_wavelength_log_wavelength\VAE_1000'
# Validation Metrics - NRMSE per element: 3.6224,  MDSA: 50.3092, |SSPB|: 13.6219, |Slope-1|: 0.1788
# Test Metrics - NRMSE per element: 3.2771, MDSA: 45.6928, |SSPB|: 12.8682, |Slope-1|: 0.1611
# Best model folder: seed_45

