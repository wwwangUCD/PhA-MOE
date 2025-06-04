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
from models.MDN_MoE_s_new import *
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
    # preTrainModel_dir = rf'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb\PACE\robust_wholeband_log_wholeband\MDN_MOE_s\seed_{seed}'
    preTrainModel_dir = rf'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\Results_Feb_PACEft\PACE\robust_wholeband_log_wholeband\MDN_MOE_s\52_8_4_1\seed_{seed}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # seed = 42  # You can choose any integer value as the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create a MLP model
    input_size = 144  # Number of features in the input data
    output_size = 144
    # model = MDN(n_inputs=input_size, n_targets=output_size, hidden=hidden)
    model = MDN_MoE(n_inputs=input_size, n_targets=output_size, hidden=hidden, lr=1e-4, band_width=band_width,
                    num_experts=num_experts, k=k)

    MDN_criterion = model.loss  # Mean Squared Error loss for regression
    optimizer = model.optimizer  # Adam optimizer
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(preTrainModel_dir, 'best_model_state_dict.pth')))
    all_targets_test, all_outputs_test, avg_test_loss = evaluate_model(model, test_loader, MDN_criterion, device,
                                                             record_outputs=True)
    if all_outputs_test.dim() == 3:
        all_outputs_test = all_outputs_test.squeeze(1)
    all_targets_test_np = all_targets_test.numpy()
    all_outputs_test_np = all_outputs_test.numpy()
    all_targets_test_np = scalery.inverse_transform(all_targets_test_np)
    all_outputs_test_np = scalery.inverse_transform(all_outputs_test_np)
    np.save(os.path.join(save_dir, 'all_test_targets.npy'), all_targets_test_np)
    np.save(os.path.join(save_dir, 'all_test_outputs.npy'), all_outputs_test_np)

    # Evaluate on train set
    all_targets_train, all_outputs_train, avg_val_loss = evaluate_model(model, train_loader, MDN_criterion, device,
                                                                        record_outputs=True)
    if all_outputs_train.dim() == 3:
        all_outputs_train = all_outputs_train.squeeze(1)
    all_targets_train_np = all_targets_train.numpy()
    all_outputs_train_np = all_outputs_train.numpy()
    all_targets_train_np = scalery.inverse_transform(all_targets_train_np)
    all_outputs_train_np = scalery.inverse_transform(all_outputs_train_np)
    np.save(os.path.join(save_dir, 'all_train_targets.npy'), all_targets_train_np)
    np.save(os.path.join(save_dir, 'all_train_outputs.npy'), all_outputs_train_np)

    # Concatenate test and train results
    all_targets_combined = np.concatenate([all_targets_train_np,all_targets_test_np], axis=0)
    all_outputs_combined = np.concatenate([all_outputs_train_np,all_outputs_test_np], axis=0)
    np.save(os.path.join(save_dir, 'all_targets.npy'), all_targets_combined)
    np.save(os.path.join(save_dir, 'all_outputs.npy'), all_outputs_combined)




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
    evaluation_aphy = np.load(r'..\papers\fine_tune_aphy.npy')
    evaluation_rrs = np.load(r'..\papers\fine_tune_rrs.npy')
    fine_tune_aphy = np.load(r'..\papers\evaluation_aphy.npy')
    fine_tune_rrs = np.load(r'..\papers\evaluation_rrs.npy')
    # the fine tune npy has 14 samples and the evaluation has 21 samples.
    # so we want to switch them, i.e. use 21 for training and 14 for testing

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
    fine_tune_aphy = fine_tune_aphy[:, selected_wl_indices]
    fine_tune_rrs = fine_tune_rrs[:, selected_wl_indices]
    evaluation_aphy = evaluation_aphy[:, selected_wl_indices]
    evaluation_rrs = evaluation_rrs[:, selected_wl_indices]


    save_folder = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_MOE_PACE_noFT'

    new_train_data = np.concatenate([fine_tune_rrs])
    new_train_labels = np.concatenate([fine_tune_aphy])

    new_test_data = np.concatenate([evaluation_rrs])
    new_test_labels = np.concatenate([evaluation_aphy])

    import joblib

    scaler_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\scalers\PACE\WB'
    scalery_path = os.path.join(scaler_dir, 'y_preprocessor.pkl')
    scalerx_path = os.path.join(scaler_dir, 'x_preprocessor.pkl')

    # Load the saved preprocessors using joblib
    scalery = joblib.load(scalery_path)
    scalerx = joblib.load(scalerx_path)

    train_data = scalerx.transform(new_train_data)
    train_labels = scalery.transform(new_train_labels)

    test_data = scalerx.transform(new_test_data)
    test_labels = scalery.transform(new_test_labels)


    if len(train_data.shape) == 3:
        train_data = train_data.squeeze(1)
        train_labels = train_labels.squeeze(1)
        test_data = test_data.squeeze(1)
        test_labels = test_labels.squeeze(1)
    train_tensor = TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_labels).float())
    train_loader = DataLoader(train_tensor, batch_size=128, shuffle=True)
    test_tensor = TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_labels).float())
    test_loader = DataLoader(test_tensor, batch_size=128, shuffle=False)
    for seed in range(42, 72):
        print(f"Finetunning seed {seed}\n")
        main(hidden=[256] * 6, band_width=144, num_experts=8, k=4, gamma=1, seed=seed)
