"""
Author: Weiwei Wang
August 2024

apply the pretrained and finetuned PACE MOE model on the station's evaluation set, but the Rrs now comes from the PACE map data instead of the field Rrs data
when compared with the
    evaluation_rrs = np.load('evaluation_rrs.npy')
this dataset
    Rrs_all_stations.npy doesnot contain evaluation_rrs[16], because it is NAN we dropped it.
should pay attentino when comparing the two results.

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
from models.MDN_MoE_s_new import MDN_MoE, evaluate_model_v2
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
    seed=68

    preTrainModel_dir = "D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers_review\MDN_MOE_PACE_FT_9_4\seed_68"

    save_dir=preTrainModel_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = "D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers"
    test_data = np.load(os.path.join(data_dir, 'ft_Rrs_all_stations.npy'))
    # test_data = np.load(os.path.join(data_dir, 'fine_tune_rrs.npy'))
    # selected_wl_range = [
    #     403, 405, 408, 410, 413, 415, 418, 420, 422, 425, 427, 430, 432, 435,
    #     437, 440, 442, 445, 447, 450, 452, 455, 457, 460, 462, 465, 467, 470, 472,
    #     475, 477, 480, 482, 485, 487, 490, 492, 495, 497, 500, 502, 505, 507, 510,
    #     512, 515, 517, 520, 522, 525, 527, 530, 532, 535, 537, 540, 542, 545, 547,
    #     550, 553, 555, 558, 560, 563, 565, 568, 570, 573, 575, 578, 580, 583, 586,
    #     588, 591, 593, 596, 598, 601, 603, 605, 608, 610, 613, 615, 618, 620, 623,
    #     625, 627, 630, 632, 635, 637, 640, 641, 642, 643, 645, 646, 647, 648, 650,
    #     651, 652, 653, 655, 656, 657, 658, 660, 661, 662, 663, 665, 666, 667, 668,
    #     670, 671, 672, 673, 675, 676, 677, 678, 679, 681, 682, 683, 684, 686, 687,
    #     688, 689, 691, 692, 693, 694, 696, 697, 698, 699
    # ]# PACE wavelength
    # initial_wl = 400
    # selected_wl_indices = [i - initial_wl for i in selected_wl_range]
    # test_data = test_data[:, selected_wl_indices]

    # Load scalery
    import joblib

    scaler_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\scalers\PACE\WB'
    scalery_path = os.path.join(scaler_dir, 'y_preprocessor.pkl')
    scalerx_path = os.path.join(scaler_dir, 'x_preprocessor.pkl')

    # Load the saved preprocessors using joblib
    scalery = joblib.load(scalery_path)
    scalerx = joblib.load(scalerx_path)

    valid_test_data = test_data
    valid_test_data = scalerx.transform(valid_test_data)
    
    if len(valid_test_data.shape) == 3:
        valid_test_data = valid_test_data.squeeze(1)
    
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

    np.save(os.path.join(save_dir, f'aphy_evalStation_mapRrs_output.npy'), all_outputs_np)
if __name__ == "__main__":
    # main(hidden=[256] * 6, band_width=144, num_experts=8, k=6, gamma=1)
    main(hidden=[256] * 6, band_width=144, num_experts=8, k=4, gamma=1)