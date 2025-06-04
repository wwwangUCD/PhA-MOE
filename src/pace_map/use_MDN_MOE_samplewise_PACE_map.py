"""
Author: Weiwei Wang
August 2024

MOE-based MDN Model

This script is intended for using a pre-trained Mixture of Experts (MoE)-based MDN model. Users can either test the model using the provided open dataset or apply their own datasets.

Dataset Structure:
In the provided `data_dir`, the following data files are expected:
    - `test_data = np.load(os.path.join(data_dir, 'test_data.npy'))`
    - `test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))`

Wavelength Information:
The Rrs and Aphy data cover wavelengths from 401 nm to 699 nm at a 1 nm resolution.
If your dataset uses the same PACE wavelengths (401-699 nm, 144 wavelengths in total), set the `select_wl` parameter to `None`.
"""

import sys
project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
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
from models.MDN_MoE_s import MDN_MoE, evaluate_model_v2
from tqdm import tqdm
from sklearn import preprocessing
from utils._CustomTransformer import _CustomTransformer
from utils.data_utils import * # use the same preprocessing as in the MDN code
import math
import pickle
def main(hidden=[100] * 5, band_width = 10, num_experts = 6,k = 4, gamma = 100):
    # the user may not have the Aphy ground truth.
    # in this case set use_aphy_gt=False
    use_aphy_gt=False
    batch_size=128

    # ['seed_47', 'seed_71', 'seed_48', 'seed_42', 'seed_45', 'seed_59', 'seed_51', 'seed_63', 'seed_66', 'seed_70']


    seed=68

    preTrainModel_dir = r"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_MOE_PACE_FT_9_4\seed_68"

    # save_dir=r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\Map'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = r"D:\Research\HyperCoast\PACEmap\4Seasons\npyData"
    # save_dir = data_dir
    save_dir =  r"D:\Research\HyperCoast\PACEmap\4Seasons\npyData_May24"
    os.makedirs(save_dir, exist_ok=True)
    date='0425'# 0425,0607,0923,0929,1229
    test_data = np.load(os.path.join(data_dir, f'Rrs_All_{date}.npy'))
    mask_data = np.load(os.path.join(data_dir, f'Rrs_All_nan_mask_{date}.npy'))
    mask = mask_data == 1
    N = np.sum(mask)
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
    initialWaveLength=401
    integers = [math.floor(num) for num in numbers]


    # Load scalery
    import joblib

    scaler_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\scalers\PACE\WB'
    scalery_path = os.path.join(scaler_dir, 'y_preprocessor.pkl')
    scalerx_path = os.path.join(scaler_dir, 'x_preprocessor.pkl')

    # Load the saved preprocessors using joblib
    scalery = joblib.load(scalery_path)
    scalerx = joblib.load(scalerx_path)

    valid_test_data = test_data[mask].reshape(N, 144)
    valid_test_data = scalerx.transform(valid_test_data)
    test_tensor = TensorDataset(torch.tensor(valid_test_data).float())
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    # create a MLP model
    input_size = 144  # Number of features in the input data
    output_size = 144
    # model = MDN(n_inputs=input_size, n_targets=output_size, hidden=hidden)
    model = MDN_MoE(n_inputs=input_size, n_targets=output_size, hidden=hidden, band_width=band_width, num_experts=num_experts, k=k)

    MDN_criterion = model.loss # Mean Squared Error loss for regression
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(preTrainModel_dir, 'best_model_state_dict.pth')))
    all_outputs, avg_test_loss = evaluate_model_v2(model, test_loader, MDN_criterion, device, record_outputs=True, use_aphy_gt=use_aphy_gt)
    if all_outputs.dim() == 3:
        all_outputs = all_outputs.squeeze(1)
    all_outputs_np = all_outputs.numpy()
    all_outputs_np = scalery.inverse_transform(all_outputs_np)

    outputs = np.full(test_data.shape, np.nan)  # Initialize outputs with NaN
    outputs[mask] = all_outputs_np
    # np.save(os.path.join(save_dir, f'Aphy_All_{date}.npy'), outputs)
    np.save(os.path.join(save_dir, f'Aphy_All_{date}_seed{seed}.npy'), outputs)
if __name__ == "__main__":
    # main(hidden=[256] * 6, band_width=144, num_experts=8, k=6, gamma=1)
    main(hidden=[256] * 6, band_width=144, num_experts=8, k=4, gamma=1)