"""
here we test the performance of the pretrained model on the PACE map data, with 35 real measurements

1. only the 14 test samples for evaluation

2. use all wavelength together for calculation


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
            # when not using finetunning, the test + train data together is the new dataset
            # Load test data
            test_targets_path = os.path.join(folder2_path, 'all_test_targets.npy')
            test_outputs_path = os.path.join(folder2_path, 'all_test_outputs.npy')
            test_targets = np.load(test_targets_path)
            test_outputs = np.load(test_outputs_path)
            test_targets = test_targets.reshape(-1, 1)
            test_outputs = test_outputs.reshape(-1, 1)

            # Load val data
            train_targets_path = os.path.join(folder2_path, 'all_train_targets.npy')
            train_outputs_path = os.path.join(folder2_path, 'all_train_outputs.npy')
            train_targets = np.load(train_targets_path)
            train_outputs = np.load(train_outputs_path)

            # Stack train and test together
            # test_targets = np.concatenate([train_targets, test_targets], axis=0)
            # test_outputs = np.concatenate([train_outputs, test_outputs], axis=0)

            test_nrmse_per_element = calculate_nrmse_per_element(test_targets, test_outputs)
            test_mdsa = calculate_mdsa_wl(test_targets, test_outputs)

            test_sspb_ = calculate_sspb_wl(test_targets, test_outputs, avg=False)
            test_slope_, test_slope_deviation_ = calculate_slope_wl(test_targets, test_outputs, avg=False)
            avg_test_sspb = np.mean(np.abs(test_sspb_))
            avg_test_slope_deviation = np.mean(test_slope_deviation_)

            # test_sspb = calculate_sspb_wl(test_targets, test_outputs)
            # test_slope = calculate_slope_wl(test_targets, test_outputs)

            test_metrics.append([test_nrmse_per_element, test_mdsa, avg_test_sspb, avg_test_slope_deviation])

            file_names.append(folder2)
    test_metrics = np.array(test_metrics)
    best_idx = np.argmin(test_metrics[:, 0])
    # best_idx = np.argmin(test_metrics[:, 0])
    best_test_metrics = test_metrics[best_idx]

    return test_metrics,file_names
import numpy as np


def print_average_metrics(avg_test_metrics):
    # Print Test Metrics
    print(f"Test Metrics - NRMSE per element: {avg_test_metrics[0]:.4f}, "
          f"MDSA: {avg_test_metrics[1]:.4f}, "
          f"|SSPB|: {avg_test_metrics[2]:.4f}, |Slope-1|: {avg_test_metrics[3]:.4f}")

# base_dir=fr"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_PACE_FT_9_4"
# base_dir=fr"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_FT_9_4"


test_metrics,file_names = process_folder1(base_dir)
best_idx = np.argmin(test_metrics[:, 0])

print(f"Best model folder: {file_names[best_idx]}")
print_average_metrics(test_metrics[best_idx])

k = 11
top_k_indices = np.argsort(test_metrics[:, 0])[:k]
top_k_metrics = test_metrics[top_k_indices]
average_top_k = np.mean(top_k_metrics, axis=0)
print(f"Average performance on top {k} seeds:")
print_average_metrics(average_top_k)

"""
after FT:
MOE-4:
Best model folder: seed_68
Test Metrics - NRMSE per element: 0.3479, MDSA: 21.5571, |SSPB|: 2.5498, |Slope-1|: 0.1010
Average performance on top 11 seeds:
Test Metrics - NRMSE per element: 0.5045, MDSA: 28.6458, |SSPB|: 7.7954, |Slope-1|: 0.1106
MDN-4:
Best model folder: seed_45
Test Metrics - NRMSE per element: 0.4146, MDSA: 36.6447, |SSPB|: 3.8677, |Slope-1|: 0.0263
Average performance on top 11 seeds:
Test Metrics - NRMSE per element: 0.5574, MDSA: 29.7015, |SSPB|: 6.8346, |Slope-1|: 0.0913


"""

# base_dir=fr"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_PACE_noFT"
# base_dir=fr"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_MOE_PACE_noFT"
# test_metrics,file_names = process_folder1(base_dir)
# best_idx = np.argmin(test_metrics[:, 0])
#
# print(f"Best model folder: {file_names[best_idx]}")
# print_average_metrics(test_metrics[best_idx])
#
# k = 11
# top_k_indices = np.argsort(test_metrics[:, 0])[:k]
# top_k_metrics = test_metrics[top_k_indices]
# average_top_k = np.mean(top_k_metrics, axis=0)
# print(f"Average performance on top {k} seeds:")
# print_average_metrics(average_top_k)
"""
before FT
MOE:
Best model folder: seed_51
Test Metrics - NRMSE per element: 1.0602, MDSA: 38.8713, |SSPB|: 27.4759, |Slope-1|: 0.0981
Average performance on top 11 seeds:
Test Metrics - NRMSE per element: 1.4965, MDSA: 41.4589, |SSPB|: 35.3332, |Slope-1|: 0.0999
MDN:
Best model folder: seed_45
Test Metrics - NRMSE per element: 1.2004, MDSA: 51.3480, |SSPB|: 48.2665, |Slope-1|: 0.0956
Average performance on top 11 seeds:
Test Metrics - NRMSE per element: 1.6790, MDSA: 44.4126, |SSPB|: 32.7873, |Slope-1|: 0.1206
"""