"""
MDN model, transcribed from the tensorflow version

Use Preprocessing


"""
import sys
project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
sys.path.append(project_dir)
import os
import torch
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
from models.MLP import *
from tqdm import tqdm
from sklearn import preprocessing
from utils._CustomTransformer import _CustomTransformer
from utils.data_utils import * # use the same preprocessing as in the MDN code
import math


def main(system,numbers,hidden_mult=[1] * 5,neuron=128, seed=42,
         scaler_x_type='robust',
         scaler_y_type='log',
         scaler_x_scope='wavelength',
         scaler_y_scope='wavelength',
         keep_shape=False,
         keep_1st_derivative=False,
         normalize_1st_derivative=False,
         keep_2nd_derivative=False,
         normalize_2nd_derivative=False):
    folder_name = f"{scaler_x_type}_{scaler_x_scope}_{scaler_y_type}_{scaler_y_scope}"
    save_dir = f'../Results_Jan/{system}/{folder_name}/MLP_1000/seed_{seed}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # seed = 42  # You can choose any integer value as the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = "D:\\Research\\EnvironmentalData\\1800Data"

    best_epochs = []
    best_train_losses = []
    best_valid_losses = []
    best_test_losses = []


    integers = [math.floor(num) for num in numbers]
    selected_wl = integers
    # selected_wl=401
    if scaler_x_type=='no':
        is_preproc=False
    else:
        is_preproc = True
    if is_preproc is True:
        train_loader, val_loader, test_loader, x_len, y_len,x_channel, scalerx, scalery = prepare_dataloaders(data_dir, batch_size=128,
                                                                                                              selected_wl=selected_wl,
                                                                                                              is_preproc=is_preproc,
                                                                                                              scaler_x_type=scaler_x_type,
                                                                                                              scaler_y_type=scaler_y_type,
                                                                                                              scaler_x_scope=scaler_x_scope,
                                                                                                              scaler_y_scope=scaler_y_scope,
                                                                                                              keep_shape=keep_shape,
                                                                                                              keep_1st_derivative=keep_1st_derivative,
                                                                                                              normalize_1st_derivative=normalize_1st_derivative,
                                                                                                              keep_2nd_derivative=keep_2nd_derivative,
                                                                                                              normalize_2nd_derivative=normalize_2nd_derivative)
    else:
        train_loader, val_loader, test_loader, x_len,  y_len, x_channel = prepare_dataloaders(data_dir,
                                                                                                    batch_size=128,
                                                                                                    selected_wl=selected_wl,
                                                                                                    is_preproc=is_preproc)
    # create a MLP model
    input_size = x_len  # Number of features in the input data
    output_size = y_len
    model = MLP(input_size=input_size, hidden_size=neuron, output_size=output_size,mult=hidden_mult, activation='relu')
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    # Define the number of epochs
    num_epochs = 1000
    # Define lists to store training and validation losses
    train_losses = []
    valid_losses = []
    test_losses = []
    best_valid_loss = float('inf')
    best_epoch = -1
    best_train_loss = float('inf')
    best_test_loss = None
    model.to(device)
    # for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
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
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_loss_sum += loss.item() * inputs.shape[0]
            num_train_samples += inputs.shape[0]
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss_sum / num_train_samples
        train_losses.append(avg_train_loss)

        # Validation

        avg_valid_loss = evaluate_model(model, val_loader, criterion, device)
        valid_losses.append(avg_valid_loss)
        avg_test_loss = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(avg_test_loss)

        # Print epoch statistics
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')
        # if avg_valid_loss < best_valid_loss and avg_train_loss < best_train_loss:
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_epoch = epoch
            best_train_loss = avg_train_loss
            best_test_loss = avg_test_loss
            best_model_state_dict = model.state_dict()
        # best_valid_loss = avg_valid_loss
        # best_epoch = epoch
        # best_train_loss = avg_train_loss
        # best_test_loss = avg_test_loss
        # best_model_state_dict = model.state_dict()
    print("Best epoch:", best_epoch)
    print("Best train loss:", best_train_loss)
    print("Best validation loss:", best_valid_loss)
    print("Best test loss:", best_test_loss)
    best_epochs.append(best_epoch)
    best_train_losses.append(best_train_loss)
    best_valid_losses.append(best_valid_loss)
    best_test_losses.append(best_test_loss)
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        torch.save(best_model_state_dict, os.path.join(save_dir, f'best_model_state_dict.pth'))
    all_targets, all_outputs, avg_test_loss = evaluate_model(model, test_loader, criterion, device, record_outputs=True)
    if all_outputs.dim() == 3:
        all_outputs = all_outputs.squeeze(1)
    all_targets_np = all_targets.numpy()
    all_outputs_np = all_outputs.numpy()
    if is_preproc is True:
        all_targets_np = scalery.inverse_transform(all_targets_np)
        all_outputs_np = scalery.inverse_transform(all_outputs_np)
    np.save(os.path.join(save_dir, f'all_test_targets.npy'), all_targets_np)
    np.save(os.path.join(save_dir, f'all_test_outputs.npy'), all_outputs_np)

    all_targets, all_outputs, avg_val_loss = evaluate_model(model, val_loader, criterion, device, record_outputs=True)
    if all_outputs.dim() == 3:
        all_outputs = all_outputs.squeeze(1)
    all_targets_np = all_targets.numpy()
    all_outputs_np = all_outputs.numpy()
    if is_preproc is True:
        all_targets_np = scalery.inverse_transform(all_targets_np)
        all_outputs_np = scalery.inverse_transform(all_outputs_np)
    np.save(os.path.join(save_dir, f'all_val_targets.npy'), all_targets_np)
    np.save(os.path.join(save_dir, f'all_val_outputs.npy'), all_outputs_np)


if __name__ == "__main__":
    import time
    systems = ['EMIT','PACE']
    for system in systems:
        if system == 'HICO':
            numbers = [
                404.080, 409.808, 415.536, 421.264, 426.992, 432.720, 438.448, 444.176,
                449.904, 455.632, 461.360, 467.088, 472.816, 478.544, 484.272, 490.000,
                495.728, 501.456, 507.184, 512.912, 518.640, 524.368, 530.096, 535.824,
                541.552, 547.280, 553.008, 558.736, 564.464, 570.192, 575.920, 581.648,
                587.376, 593.104, 598.832, 604.560, 610.288, 616.016, 621.744, 627.472,
                633.200, 638.928, 644.656, 650.384, 656.112, 661.840, 667.568, 673.296,
                679.024, 684.752, 690.480, 696.208
            ]  # selected wavelength
            hidden_mult=[1, 2, 3, 2, 1]  # Example: Slightly varied pyramid
            neuron = 128
        elif system == 'EMIT':
            numbers = [
                403.2254, 410.638, 418.0536, 425.47214, 432.8927, 440.31726, 447.7428,
                455.17035, 462.59888, 470.0304, 477.46292, 484.89743, 492.33292, 499.77142,
                507.2099, 514.6504, 522.0909, 529.5333, 536.9768, 544.42126, 551.8667,
                559.3142, 566.7616, 574.20905, 581.6585, 589.108, 596.55835, 604.0098,
                611.4622, 618.9146, 626.36804, 633.8215, 641.2759, 648.7303, 656.1857,
                663.6411, 671.09753, 678.5539, 686.0103, 693.4677
            ]  # EMIT
            hidden_mult = [1, 1, 1, 1, 1, 1]
            neuron = 256
        elif system == 'PACE':
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
            ]
            hidden_mult = [1, 2, 2, 1]
            neuron = 256
        else:
            print(f"Error: The specified system '{system}' does not exist or is not recognized.")
        seeds = list(range(42, 53))

        parameter_combinations = [
            # {'scaler_x_type': 'robust', 'scaler_y_type': 'log', 'scaler_x_scope': 'wavelength',
            #  'scaler_y_scope': 'wavelength'},
            {'scaler_x_type': 'robust', 'scaler_y_type': 'log', 'scaler_x_scope': 'wholeband',
             'scaler_y_scope': 'wholeband'},
            # {'scaler_x_type': 'robust', 'scaler_y_type': 'robust', 'scaler_x_scope': 'wavelength',
            #  'scaler_y_scope': 'wavelength'},
            {'scaler_x_type': 'robust', 'scaler_y_type': 'robust', 'scaler_x_scope': 'wholeband',
             'scaler_y_scope': 'wholeband'},
            # {'scaler_x_type': 'log', 'scaler_y_type': 'log', 'scaler_x_scope': 'wavelength',
            #  'scaler_y_scope': 'wavelength'},
            {'scaler_x_type': 'log', 'scaler_y_type': 'log', 'scaler_x_scope': 'wholeband',
             'scaler_y_scope': 'wholeband'},
            # {'scaler_x_type': 'no', 'scaler_y_type': 'no', 'scaler_x_scope': 'wavelength',
            #  'scaler_y_scope': 'wavelength'},
            # {'scaler_x_type': 'log', 'scaler_y_type': 'robust', 'scaler_x_scope': 'wholeband',
            #  'scaler_y_scope': 'wholeband'},
        ]

        from tqdm import tqdm


        total_iterations = len(seeds) * len(parameter_combinations)

        # Initialize the progress bar
        with tqdm(total=total_iterations, desc="Total Progress") as pbar:
                for seed in seeds:
                    for params in parameter_combinations:
                        start_time = time.time()
                        main(system,numbers,hidden_mult=hidden_mult,neuron=neuron, seed=seed,
                             scaler_x_type=params['scaler_x_type'],
                             scaler_y_type=params['scaler_y_type'],
                             scaler_x_scope=params['scaler_x_scope'],
                             scaler_y_scope=params['scaler_y_scope'], )
                        elapsed_time = time.time() - start_time
                        pbar.set_postfix({"last_epoch_time": f"{elapsed_time:.2f}s"})
                        pbar.update(1)
