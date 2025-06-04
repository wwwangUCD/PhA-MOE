import numpy as np
import os
"""
# in total there are 14 stations
# 0925+1023
# the idx=4 one is NAN, and removed
"""



# Base directory
base_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers'

# File names
files = ['ft_Rrs_Stations_0925.npy', 'ft_Rrs_Stations_1023.npy']

# Initialize an empty list to collect the arrays
data_list = []

# Iterate through the files
for file in files:
    file_path = os.path.join(base_dir, file)

    # Load the array and validate its dimensions
    data = np.load(file_path)
    print(f"Loaded {file} with shape {data.shape}")

    # Ensure the array has consistent dimensions
    if len(data_list) > 0 and data.shape[1:] != data_list[0].shape[1:]:
        raise ValueError(f"Dimension mismatch in file {file}. Expected shape {data_list[0].shape[1:]}, but got {data.shape[1:]}")

    # Append the array to the list
    data_list.append(data)

# Concatenate all the arrays along the first dimension
stacked_data = np.concatenate(data_list, axis=0)

# Identify rows with NaN values
nan_rows = np.any(np.isnan(stacked_data), axis=1)

# Print row indices containing NaN
nan_indices = np.where(nan_rows)[0]
if len(nan_indices) > 0:
    print(f"Rows containing NaN values: {nan_indices}")
else:
    print("No rows with NaN values found.")

# Remove rows containing NaN
cleaned_data = stacked_data[~nan_rows]

# Save the cleaned array
output_path = os.path.join(base_dir, 'ft_Rrs_all_stations.npy')
np.save(output_path, cleaned_data)

print(f"Cleaned data shape: {cleaned_data.shape}")
print(f"Cleaned data saved to: {output_path}")
