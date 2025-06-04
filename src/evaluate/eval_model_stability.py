"""
Compare the performance on each model across different random seeds.

Average performance and coefficience of variance are calculated across seeds on each evaluation metric.

"""
import os
import numpy as np
import sys
import argparse
import pandas as pd
# project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
# sys.path.append(project_dir)

# from utils.metrics import *


def process_folder1(folder1_path,k):
    """
    Process the folder1 directory to calculate average metrics.

    Args:
    folder1_path (str): Path to the folder1 directory

    Returns:
    tuple: Average test metrics and average validation metrics
    """
    test_metrics = []
    val_metrics = []
    val_nrmse_values = []
    val_mdsa_values = []
    for folder2 in os.listdir(folder1_path): # folder1 is the model type
        folder2_path = os.path.join(folder1_path, folder2) # folder2 is the seed
        if os.path.isdir(folder2_path):
            test_targets_path = os.path.join(folder2_path, 'all_test_targets.npy')
            test_outputs_path = os.path.join(folder2_path, 'all_test_outputs.npy')
            test_targets = np.load(test_targets_path)
            test_outputs = np.load(test_outputs_path)

            # Calculate test metrics
            test_nrmse_per_element = calculate_nrmse_per_element(test_targets, test_outputs)
            # test_nrmse_per_element = calculate_nrmse_wl(test_targets, test_outputs,'relative')
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
            # val_nrmse_per_element = calculate_nrmse_wl(val_targets, val_outputs, 'relative')
            val_mdsa = calculate_mdsa_wl(val_targets, val_outputs)
            val_sspb_ = calculate_sspb_wl(val_targets, val_outputs, avg=False)
            val_slope_, val_slope_deviation_ = calculate_slope_wl(val_targets, val_outputs, avg=False)
            avg_val_sspb = np.mean(np.abs(val_sspb_))
            avg_val_slope_deviation = np.mean(val_slope_deviation_)
            # val_sspb = calculate_sspb_wl(val_targets, val_outputs)
            # val_slope = calculate_slope_wl(val_targets, val_outputs)

            val_metrics.append([val_nrmse_per_element, val_mdsa, avg_val_sspb, avg_val_slope_deviation])
            val_nrmse_values.append(val_nrmse_per_element)
            val_mdsa_values.append(val_mdsa)
    # Convert lists to numpy arrays and calculate average metrics
    test_metrics = np.array(test_metrics)
    val_metrics = np.array(val_metrics)
    val_nrmse_values = np.array(val_nrmse_values)
    # Get indices of the top k models based on val_nrmse_per_element
    top_k_indices = np.argsort(val_nrmse_values)[:k]  # Select k smallest NRMSE values

    # Compute average metrics for top k models
    avg_test_metrics = np.mean(test_metrics[top_k_indices], axis=0)
    avg_val_metrics = np.mean(val_metrics[top_k_indices], axis=0)


    # return avg_test_metrics, avg_val_metrics
    top_k_test = test_metrics[top_k_indices]
    top_k_val = val_metrics[top_k_indices]
    # Compute coefficient of variation: std / mean
    # Use np.where to avoid division by zero
    epsilon = 1e-8  # to avoid zero division
    std_test = np.std(top_k_test, axis=0)
    std_val = np.std(top_k_val, axis=0)

    cv_test_metrics = np.where(avg_test_metrics != 0, std_test / (avg_test_metrics + epsilon), 0)
    cv_val_metrics = np.where(avg_val_metrics != 0, std_val / (avg_val_metrics + epsilon), 0)

    return avg_test_metrics, avg_val_metrics, cv_test_metrics, cv_val_metrics
def evaluate_hyperparameters(base_dir):
    # Initialize lists
    folder_names = []
    test_metrics_list = []
    val_metrics_list = []

    # Loop through direct subfolders in base_dir
    for folder1 in os.listdir(base_dir):
        folder1_path = os.path.join(base_dir,folder1)
        if os.path.isdir(folder1_path):
            folder_names.append(folder1)
            avg_test_metrics, avg_val_metrics = process_folder1(folder1_path)
            test_metrics_list.append(avg_test_metrics)
            val_metrics_list.append(avg_val_metrics)

    return folder_names, test_metrics_list, val_metrics_list


def save_results(save_dir, base_dir, folder_names, test_metrics_list, val_metrics_list):
    # Extract last two terms from base_dir to create filename
    dir_parts = base_dir.strip(os.sep).split(os.sep)
    filename = f"{dir_parts[-2]}_{dir_parts[-1]}.txt"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w') as f:
        for i, folder_name in enumerate(folder_names):
            # Write folder name
            f.write(f"Folder: {folder_name}\n")

            # Write Validation Metrics
            avg_val_metrics = val_metrics_list[i]
            f.write(f"Validation Metrics - NRMSE per element: {avg_val_metrics[0]:.4f}, "
                    f"MDSA: {avg_val_metrics[1]:.4f}, "
                    f"|SSPB|: {avg_val_metrics[2]:.4f}, |Slope-1|: {avg_val_metrics[3]:.4f}\n")

            # Write Test Metrics
            avg_test_metrics = test_metrics_list[i]
            f.write(f"Test Metrics - NRMSE per element: {avg_test_metrics[0]:.4f}, "
                    f"MDSA: {avg_test_metrics[1]:.4f}, "
                    f"|SSPB|: {avg_test_metrics[2]:.4f}, |Slope-1|: {avg_test_metrics[3]:.4f}\n\n")

    print(f"Results saved to {filepath}")


def print_average_metrics(avg_test_metrics, avg_val_metrics):
    # Print Validation Metrics
    print(f"Validation Metrics - NRMSE per element: {avg_val_metrics[0]:.4f}, "
          f" MDSA: {avg_val_metrics[1]:.4f}, "
          f"|SSPB|: {avg_val_metrics[2]:.4f}, |Slope-1|: {avg_val_metrics[3]:.4f}")

    # Print Test Metrics
    print(f"Test Metrics - NRMSE per element: {avg_test_metrics[0]:.4f}, "
          f"MDSA: {avg_test_metrics[1]:.4f}, "
          f"|SSPB|: {avg_test_metrics[2]:.4f}, |Slope-1|: {avg_test_metrics[3]:.4f}")

# base_dir=rf'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\kF_test\PACE\MDN'
# folder_names, test_metrics_list, val_metrics_list = evaluate_hyperparameters(base_dir)
def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate K-folder cross-validation performance.")
    parser.add_argument("--basedir", type=str, default="/home/weiwei/NASA/1800Data/",
                        help="Base directory for cross-validation folders.")
    parser.add_argument("--savedir", type=str, default=r"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\PACE",
                        help="Directory to save the results file.")
    parser.add_argument("--projectdir", type=str, default=r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3',
                        help="Project directory to include in the path.")
    parser.add_argument("--system", type=str, default="PACE", help="System name (e.g., PACE, HICO, EMIT).")
    return parser.parse_args()

# Parse arguments and set up sys.path
args = parse_arguments()
sys.path.append(args.projectdir)
from utils.metrics import *


def sort(folder_names, test_metrics_list, val_metrics_list):
    # Combine results into tuples to sort by validation NRMSE
    combined = list(zip(folder_names, test_metrics_list, val_metrics_list))

    # Sort by validation NRMSE (lower is better)
    combined.sort(key=lambda x: x[2][0])

    # Unpack the sorted results
    sorted_folder_names, sorted_test_metrics_list, sorted_val_metrics_list = zip(*combined)
    return list(sorted_folder_names), list(sorted_test_metrics_list), list(sorted_val_metrics_list)

def main():
    k = 11
    save_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\paper_review'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Selected directories for PACE
    selected_dirs_pace = [
        ("PACE", "robust_wholeband_log_wholeband", "MDN_MOE_s", r"52_8_4_1"),
        ("PACE", "robust_wholeband_log_wholeband", "MDN", None),
        ("PACE", "no_wavelength_no_wavelength", "MLP_1000", None),
        ("PACE", "robust_wavelength_log_wavelength", "VAE_1000", None),
    ]

    # Selected directories for EMIT
    selected_dirs_emit = [
        ("EMIT", "robust_wavelength_log_wavelength", "MDN_MOE_s", None),
        ("EMIT", "robust_wavelength_log_wavelength", "MDN", None),
        ("EMIT", "robust_wholeband_robust_wholeband", "MLP_1000", None),
        ("EMIT", "robust_wavelength_log_wavelength", "VAE_1000", None),
    ]

    base_root = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21'
    base_root_moe_pace = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb_PACEft'

    def build_path(dataset, pp, model, subfolder):
        if dataset == "PACE" and model == "MDN_MOE_s":
            base = base_root_moe_pace
        else:
            base = base_root
        path = os.path.join(base, dataset, pp, model)
        if subfolder:
            path = os.path.join(path, subfolder)
        return path

    def evaluate_and_collect(selected_dirs, dataset_name):
        metrics_dict = {}
        metric_names = ["NRMSE", "MDSA", "SSPB", "|SLOPE-1|"]

        for dataset, pp, model, subfolder in selected_dirs:
            model_display = model if subfolder is None else f"{model}/{subfolder}"
            full_path = build_path(dataset, pp, model, subfolder)
            if not os.path.isdir(full_path):
                print(f"[Warning] Path not found: {full_path}")
                continue

            avg_test, avg_val, cv_test, cv_val = process_folder1(full_path, k)
            metrics_dict[model_display] = {
                "test": avg_test,
                "val": avg_val,
                "cv_test": cv_test,
                "cv_val": cv_val
            }

        def build_dataframe(metrics_type):
            rows = []
            for model, result in metrics_dict.items():
                metrics = result.get(metrics_type, ["N/A"] * len(metric_names))
                row = [model] + [f"{v:.4f}" if isinstance(v, (int, float)) else "N/A" for v in metrics]
                rows.append(row)

            return pd.DataFrame(rows, columns=["Model"] + metric_names)

        # Create and save mean metric tables
        df_test = build_dataframe("test")
        df_val = build_dataframe("val")
        df_test.to_csv(os.path.join(save_dir, f"{dataset_name}_test_metrics_k{k}.csv"), index=False)
        df_val.to_csv(os.path.join(save_dir, f"{dataset_name}_val_metrics_k{k}.csv"), index=False)

        # Create and save CV tables
        df_cv_test = build_dataframe("cv_test")
        df_cv_val = build_dataframe("cv_val")
        df_cv_test.to_csv(os.path.join(save_dir, f"{dataset_name}_cv_test_metrics_k{k}.csv"), index=False)
        df_cv_val.to_csv(os.path.join(save_dir, f"{dataset_name}_cv_val_metrics_k{k}.csv"), index=False)

        print(f"[{dataset_name}] Test/Val + CV CSVs saved in row-by-metric format.")

    evaluate_and_collect(selected_dirs_pace, "PACE")
    evaluate_and_collect(selected_dirs_emit, "EMIT")
if __name__ == "__main__":
    main()

