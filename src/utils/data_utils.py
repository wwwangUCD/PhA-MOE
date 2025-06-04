"""
Use different preprocessing methods
"""

import sys
project_dir='D:\\Research\\EnvironmentalData\\BenchmarkEvaluation_r3'
sys.path.append(project_dir)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from utils._CustomTransformer import *
from sklearn.preprocessing import MinMaxScaler

def prepare_dataloaders(data_dir, batch_size=32,selected_wl=None,is_preproc=False,
                 scaler_x_type='robust',
                 scaler_y_type='log',
                 scaler_x_scope='wavelength',
                 scaler_y_scope='wavelength',
                 keep_shape=False,
                 keep_1st_derivative=False,
                 normalize_1st_derivative=False,
                 keep_2nd_derivative=False,
                 normalize_2nd_derivative=False):
    initialWaveLength=401
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'))
    val_labels = np.load(os.path.join(data_dir, 'val_labels.npy'))
    test_data = np.load(os.path.join(data_dir, 'test_data.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))


    if selected_wl is not None:
        if isinstance(selected_wl, range):
            indices = [i - initialWaveLength for i in selected_wl]
        elif isinstance(selected_wl, list):
            indices = [i - initialWaveLength for i in selected_wl]
        elif isinstance(selected_wl, int):
            indices = [selected_wl - initialWaveLength]
        else:
            raise ValueError("selected_wl must be an integer, a range, or a list of integers")

        train_labels = train_labels[:, indices]
        val_labels = val_labels[:, indices]
        test_labels = test_labels[:, indices]
        train_data = train_data[:, indices]
        val_data = val_data[:, indices]
        test_data = test_data[:, indices]



    if is_preproc is True:
        # Define scalers
        x_preprocesser = Preprocessor(
                 scaler_type=scaler_x_type,
                 scaler_scope=scaler_x_scope,
                 keep_shape=keep_shape,
                 keep_1st_derivative=keep_1st_derivative,
                 normalize_1st_derivative=normalize_1st_derivative,
                 keep_2nd_derivative=keep_2nd_derivative,
                 normalize_2nd_derivative=normalize_2nd_derivative)
        y_preprocesser = Preprocessor(
                 scaler_type=scaler_y_type,
                 scaler_scope=scaler_y_scope)
        # Fit the scalers on the training data
        x_preprocesser.fit(train_data)
        y_preprocesser.fit(train_labels)
        train_data = x_preprocesser.transform(train_data)
        val_data = x_preprocesser.transform(val_data)
        test_data = x_preprocesser.transform(test_data)

        train_labels = y_preprocesser.transform(train_labels)
        val_labels = y_preprocesser.transform(val_labels)
        test_labels = y_preprocesser.transform(test_labels)

    if len(train_data.shape) == 3:
        train_data = train_data.squeeze(1)
        val_data = val_data.squeeze(1)
        test_data = test_data.squeeze(1)
        train_labels = train_labels.squeeze(1)
        val_labels = val_labels.squeeze(1)
        test_labels = test_labels.squeeze(1)
    train_tensor = TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_labels).float())
    val_tensor = TensorDataset(torch.tensor(val_data).float(), torch.tensor(val_labels).float())
    test_tensor = TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_labels).float())

    # Create data loaders
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    x_len = train_data.shape[1]
    y_len = train_labels.shape[1]
    if is_preproc is True:
        x_channel = x_preprocesser.data_channel
        return train_loader, val_loader, test_loader, x_len, y_len,x_channel, x_preprocesser, y_preprocesser
    else:
        x_channel=None
        return train_loader, val_loader, test_loader, x_len, y_len,x_channel


"""
How to save and load the scalers:
import joblib
import os

# Define save directory
save_dir = r"D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\scalers\PACE\WB"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Save the preprocessors
joblib.dump(x_preprocesser, os.path.join(save_dir, "x_preprocessor.pkl"))
joblib.dump(y_preprocesser, os.path.join(save_dir, "y_preprocessor.pkl"))
print("Preprocessors saved successfully!")

# Load the preprocessors
x_preprocesser = joblib.load(os.path.join(save_dir, "x_preprocessor.pkl"))
y_preprocesser = joblib.load(os.path.join(save_dir, "y_preprocessor.pkl"))

print("Preprocessors loaded successfully!")


"""