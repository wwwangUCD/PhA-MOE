"""
We use the MDN MOE code on the PACE map data directly without finetuning.
"""


import pandas as pd
import sys

project_dir = 'D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
sys.path.append(project_dir)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
from models.MDN_MoE_s import *
from tqdm import tqdm
from sklearn import preprocessing
from utils._CustomTransformer import _CustomTransformer
from utils.data_utils import *  # use the same preprocessing as in the MDN code
import math
import pickle
from sklearn.model_selection import train_test_split


def main(hidden=[100] * 5, band_width=10, num_experts=6, k=4, gamma=100, seed=42):
    # the user may not have the Aphy ground truth.
    # in this case set use_aphy_gt=False
    use_aphy_gt = True
    batch_size = 128

    save_dir = os.path.join(save_folder, rf'seed_{seed}')
  # preTrainModel_dir = rf'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb21\EMIT\robust_wavelength_log_wavelength\MDN_MOE_s\seed_{seed}'
    preTrainModel_dir = rf'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Jan\EMIT\robust_wavelength_log_wavelength\MDN_MOE_s\seed_{seed}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # seed = 42  # You can choose any integer value as the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create a MLP model
    input_size = 40  # Number of features in the input data
    output_size = 40
    # model = MDN(n_inputs=input_size, n_targets=output_size, hidden=hidden)
    model = MDN_MoE(n_inputs=input_size, n_targets=output_size, hidden=hidden, lr=1e-3, band_width=band_width,
                    num_experts=num_experts, k=k)

    MDN_criterion = model.loss  # Mean Squared Error loss for regression
    optimizer = model.optimizer  # Adam optimizer
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(preTrainModel_dir, 'best_model_state_dict.pth')))

    all_gates_list = []  # Initialize empty list

    model.eval()  # Ensure model is in eval mode

    with torch.no_grad():  # No gradients needed during inference
        for inputs, targets in train_loader:
            if len(inputs.shape) == 3:
                inputs = inputs.view(inputs.shape[0], -1)
            if len(targets.shape) == 3:
                targets = targets.view(targets.shape[0], -1)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Compute routing weights for this batch
            model.compute_routing_weights(inputs)  # this calls MoE_net.forward internally
            batch_gates = model.MoE_net.batch_gates.detach().cpu().numpy()  # extract as NumPy

            all_gates_list.append(batch_gates)

    # Stack all gates
    all_gates = np.vstack(all_gates_list)

    # Save final result
    final_save_path = os.path.join(save_folder, "train_moe_routing_weights.npy")
    np.save(final_save_path, all_gates)

    print(f"Final stacked routing weights saved at: {final_save_path}")
    print("Final shape:", all_gates.shape)  # Expected shape: (total_samples, num_experts)



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, current_loss):
        if self.best_loss is None or (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True  # Stop training
        return False


if __name__ == "__main__":
    # load fine-tuning data
    train_data = np.load(r'D:\Research\EnvironmentalData\1800Data\train_data.npy')
    train_labels = np.load(r'D:\Research\EnvironmentalData\1800Data\train_labels.npy')
    save_folder = r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\MOE_May'

    initial_wl = 401  # initial wavelength of the loaded data, both aphy and Rrs

    numbers = [
        403.2254, 410.638, 418.0536, 425.47214, 432.8927, 440.31726, 447.7428,
        455.17035, 462.59888, 470.0304, 477.46292, 484.89743, 492.33292, 499.77142,
        507.2099, 514.6504, 522.0909, 529.5333, 536.9768, 544.42126, 551.8667,
        559.3142, 566.7616, 574.20905, 581.6585, 589.108, 596.55835, 604.0098,
        611.4622, 618.9146, 626.36804, 633.8215, 641.2759, 648.7303, 656.1857,
        663.6411, 671.09753, 678.5539, 686.0103, 693.4677
    ]  # EMIT
    integers = [math.floor(num) for num in numbers]
    selected_wl_range = np.array(integers)

    selected_wl_indices = [i - initial_wl for i in selected_wl_range]
    train_data = train_data[:, selected_wl_indices]
    train_labels = train_labels[:, selected_wl_indices]







    import joblib

    scaler_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\scalers\EMIT\WL'
    scalery_path = os.path.join(scaler_dir, 'y_preprocessor.pkl')
    scalerx_path = os.path.join(scaler_dir, 'x_preprocessor.pkl')

    # Load the saved preprocessors using joblib
    scalery = joblib.load(scalery_path)
    scalerx = joblib.load(scalerx_path)

    train_data = scalerx.transform(train_data)
    train_labels = scalery.transform(train_labels)


    if len(train_data.shape) == 3:
        train_data = train_data.squeeze(1)
        train_labels = train_labels.squeeze(1)

    train_tensor = TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_labels).float())
    train_loader = DataLoader(train_tensor, batch_size=128, shuffle=False)

    main(hidden=[256] * 5, band_width=40, num_experts=8, k=6, gamma=1, seed=51)
