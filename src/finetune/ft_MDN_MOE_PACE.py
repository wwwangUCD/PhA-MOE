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
# from models.MDN_MoE_s import *
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
    model = MDN_MoE(n_inputs=input_size, n_targets=output_size, hidden=hidden, lr=1e-3, band_width=band_width,
                    num_experts=num_experts, k=k)

    MDN_criterion = model.loss  # Mean Squared Error loss for regression
    optimizer = model.optimizer  # Adam optimizer
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(preTrainModel_dir, 'best_model_state_dict.pth')))

    early_stopper = EarlyStopping(patience=20, min_delta=1e-3)
    num_epochs = 1000

    train_losses = []
    test_losses = []
    best_epoch = -1
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    best_val_loss = float('inf')
    warm_up_epoch = 50
    for epoch in range(num_epochs):
        # Training
        model.train()  # Set the model to train mode
        train_loss_sum = 0.0
        num_train_samples = 0
        for inputs, targets in train_loader:
            # targets = targets.squeeze(1)
            # inputs = inputs.squeeze(1)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            # outputs = model(inputs)
            outputs, Moe_loss = model(inputs)
            # Compute loss
            loss = MDN_criterion(outputs, targets)
            total_loss = loss + gamma * Moe_loss
            optimizer.zero_grad()
            (gamma * Moe_loss).backward(retain_graph=True)  # Perform backward on MoE loss first, retain the graph

            # Backpropagation for main loss
            loss.backward()  # Perform backward on the main loss

            optimizer.step()

            # Accumulate training loss
            train_loss_sum += total_loss.item() * inputs.shape[0]
            num_train_samples += inputs.shape[0]
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss_sum / num_train_samples
        train_losses.append(avg_train_loss)
        avg_val_loss = evaluate_model(model, val_loader, MDN_criterion, device, gamma=gamma)
        avg_test_loss = evaluate_model(model, test_loader, MDN_criterion, device, gamma=gamma)
        # Print epoch statistics
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
        if avg_val_loss < best_val_loss:
            best_epoch = epoch
            best_train_loss = avg_train_loss
            best_test_loss = avg_test_loss
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()
        if epoch >= warm_up_epoch:
            if early_stopper.step(avg_val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break
    print("Best epoch:", best_epoch)
    print("Best train loss:", best_train_loss)
    print("Best val loss:", best_val_loss)
    print("Best test loss:", best_test_loss)

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        torch.save(best_model_state_dict, os.path.join(save_dir, f'best_model_state_dict.pth'))

    all_targets, all_outputs, avg_test_loss = evaluate_model(model, test_loader, MDN_criterion, device,
                                                             record_outputs=True)
    if all_outputs.dim() == 3:
        all_outputs = all_outputs.squeeze(1)
    all_targets_np = all_targets.numpy()
    all_outputs_np = all_outputs.numpy()
    all_targets_np = scalery.inverse_transform(all_targets_np)
    all_outputs_np = scalery.inverse_transform(all_outputs_np)
    np.save(os.path.join(save_dir, f'all_test_targets.npy'), all_targets_np)
    np.save(os.path.join(save_dir, f'all_test_outputs.npy'), all_outputs_np)

    all_targets, all_outputs, avg_val_loss = evaluate_model(model, val_loader, MDN_criterion, device,
                                                            record_outputs=True)
    if all_outputs.dim() == 3:
        all_outputs = all_outputs.squeeze(1)
    all_targets_np = all_targets.numpy()
    all_outputs_np = all_outputs.numpy()
    all_targets_np = scalery.inverse_transform(all_targets_np)
    all_outputs_np = scalery.inverse_transform(all_outputs_np)
    np.save(os.path.join(save_dir, f'all_val_targets.npy'), all_targets_np)
    np.save(os.path.join(save_dir, f'all_val_outputs.npy'), all_outputs_np)


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
    evaluation_aphy = np.load(r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\fine_tune_aphy.npy')
    evaluation_rrs = np.load(r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\fine_tune_rrs.npy')
    fine_tune_aphy = np.load(r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\evaluation_aphy.npy')
    fine_tune_rrs = np.load(r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\evaluation_rrs.npy')
    # the fine tune npy has 14 samples and the evaluation has 21 samples.
    # so we want to switch them, i.e. use 21 for training and 14 for testing

    initial_wl = 400  # initial wavelength of the loaded data, both aphy and Rrs

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
    ]  # PACE wavelength
    integers = [math.floor(num) for num in numbers]
    selected_wl_range = np.array(integers)

    selected_wl_indices = [i - initial_wl for i in selected_wl_range]
    fine_tune_aphy = fine_tune_aphy[:, selected_wl_indices]
    fine_tune_rrs = fine_tune_rrs[:, selected_wl_indices]
    evaluation_aphy = evaluation_aphy[:, selected_wl_indices]
    evaluation_rrs = evaluation_rrs[:, selected_wl_indices]

    # load original data
    data_dir = "D:/Research/EnvironmentalData/1800Data"
    test_data = np.load(f"{data_dir}/test_data.npy")
    test_labels = np.load(f"{data_dir}/test_labels.npy")
    train_data = np.load(f"{data_dir}/train_data.npy")
    train_labels = np.load(f"{data_dir}/train_labels.npy")
    val_data = np.load(f"{data_dir}/val_data.npy")
    val_labels = np.load(f"{data_dir}/val_labels.npy")

    # Dataset preparation
    # 1. Split original train, val, and test datasets
    ori_train_subset_data, _, ori_train_subset_labels, _ = train_test_split(
        train_data, train_labels, train_size=0.2, random_state=42
    )
    # ori_train_subset_data = train_data
    # ori_train_subset_labels = train_labels

    # Similarly, split the original validation data and labels together
    # ori_val_subset_data, _, ori_val_subset_labels, _ = train_test_split(
    #     val_data, val_labels, train_size=1.0, random_state=42
    # )
    ori_val_subset_data = val_data
    ori_val_subset_labels = val_labels

    # Similarly, split the original test data and labels together
    ori_test_subset_data, _, ori_test_subset_labels, _ = train_test_split(
        test_data, test_labels, train_size=0.2, random_state=42
    )
    initial_wl = 401
    selected_wl_indices = [i - initial_wl for i in selected_wl_range]

    ori_train_subset_data = ori_train_subset_data[:, selected_wl_indices]
    ori_val_subset_data = ori_val_subset_data[:, selected_wl_indices]
    ori_test_subset_data = ori_test_subset_data[:, selected_wl_indices]
    ori_train_subset_labels = ori_train_subset_labels[:, selected_wl_indices]
    ori_val_subset_labels = ori_val_subset_labels[:, selected_wl_indices]
    ori_test_subset_labels = ori_test_subset_labels[:, selected_wl_indices]
    # 2. Create new datasets
    new_train_data = np.concatenate([ori_train_subset_data, fine_tune_rrs])
    new_train_labels = np.concatenate([ori_train_subset_labels, fine_tune_aphy])
    save_folder = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_MOE_PACE_FT_9_2'

    # # 7: lr=1e-3 20% original data
    # new_train_data = np.concatenate([ori_train_subset_data, fine_tune_rrs])
    # new_train_labels = np.concatenate([ori_train_subset_labels, fine_tune_aphy])
    #
    # new_val_data = np.concatenate([evaluation_rrs])
    # new_val_labels = np.concatenate([evaluation_aphy])
    #
    # new_test_data = np.concatenate([evaluation_rrs])
    # new_test_labels = np.concatenate([evaluation_aphy])
    # 9 100% original data
    # 9-2 20% original data
    # 9-3 100% original data +1e-4 lr
    # 9-4 20% original data + 1e-4 lr
    new_train_data = np.concatenate([ori_train_subset_data, fine_tune_rrs])
    new_train_labels = np.concatenate([ori_train_subset_labels, fine_tune_aphy])

    new_val_data = np.concatenate([ori_val_subset_data, evaluation_rrs])
    new_val_labels = np.concatenate([ori_val_subset_labels, evaluation_aphy])

    new_test_data = np.concatenate([evaluation_rrs])
    new_test_labels = np.concatenate([evaluation_aphy])

    import joblib

    scaler_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\scalers\PACE\WB'
    scalery_path = os.path.join(scaler_dir, 'y_preprocessor.pkl')
    scalerx_path = os.path.join(scaler_dir, 'x_preprocessor.pkl')

    # Load the saved preprocessors using joblib
    scalery = joblib.load(scalery_path)
    scalerx = joblib.load(scalerx_path)

    test_data = scalerx.transform(new_test_data)
    test_labels = scalery.transform(new_test_labels)

    train_data = scalerx.transform(new_train_data)
    train_labels = scalery.transform(new_train_labels)

    val_data = scalerx.transform(new_val_data)
    val_labels = scalery.transform(new_val_labels)

    if len(train_data.shape) == 3:
        train_data = train_data.squeeze(1)
        val_data = val_data.squeeze(1)
        test_data = test_data.squeeze(1)
        train_labels = train_labels.squeeze(1)
        val_labels = val_labels.squeeze(1)
        test_labels = test_labels.squeeze(1)
    test_tensor = TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_labels).float())
    test_loader = DataLoader(test_tensor, batch_size=128, shuffle=False)
    train_tensor = TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_labels).float())
    train_loader = DataLoader(train_tensor, batch_size=128, shuffle=True)
    val_tensor = TensorDataset(torch.tensor(val_data).float(), torch.tensor(val_labels).float())
    val_loader = DataLoader(val_tensor, batch_size=128, shuffle=False)
    for seed in range(42, 72):
        print(f"Finetunning seed {seed}\n")
        main(hidden=[256] * 6, band_width=40, num_experts=8, k=4, gamma=1, seed=seed)
