import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bar

"""
Find the water class label of each training sample.
"""


# === Step 1: Load and Clean Original Data ===
csv_path = "D:/Research/EnvironmentalData/OurData/Rrs_400_760nm.csv"
original_data = pd.read_csv(csv_path)

# Identify relevant wavelengths
wavelength_cols = [col for col in original_data.columns if col.startswith("Rrs_")]
selected_wavelengths = [col for col in wavelength_cols if 401 <= int(col.split("_")[1]) <= 699]

# **Step 1.1: Remove rows where any wavelength ≤ 699 has NaN**
cleaned_data = original_data.dropna(subset=selected_wavelengths)

# **Step 1.2: Convert cleaned data to NumPy**
original_data_array = cleaned_data[selected_wavelengths].to_numpy()

# Print new shape after filtering
print("Original data shape before NaN removal:", original_data.shape)
print("Shape after removing NaNs from 401-699 nm range:", cleaned_data.shape)

# === Step 2: Load the Train Data ===
train_data_path = "D:/Research/EnvironmentalData/1800Data/train_data.npy"
train_data = np.load(train_data_path)

# === Step 3: Classification Function ===
def classify_water_type(Rrs):
    """
    Classifies water type using the decision tree based on Rrs values.
    Assumes Rrs is a dictionary with keys as wavelengths (401-760 nm).
    Missing data for λ > 700 is treated as 0.
    """
    R492 = Rrs.get(492, 0)
    R560 = Rrs.get(560, 0)
    R665 = Rrs.get(665, 0)
    R740 = Rrs.get(740, 0) if not np.isnan(Rrs.get(740, np.nan)) else 0  # Treat NaN as 0

    # Step 1: Check if R(665) is between R(492) and R(560)
    if R665 < R560 and R665 > R492:
        # Step 2: If true, proceed to Node 1
        if R560 < R492:
            return "Type 1 (Blue-Green Water)"
        else:
            return "Type 2 (Green Water)"

    # Step 3: If R(665) is not between R(492) and R(560), check for brown water
    elif R665 > R560:
        if R740 > 0.01:
            return "Type 3 (Brown Water)"
    if R560 < R492:
        return "Type 1 (Blue-Green Water)"
    else:
        return "Type 2 (Green Water)"


# === Step 4: Match Train Data to Cleaned Original Data and Classify ===
classified_results = []

for sample_idx in tqdm(range(len(train_data)), desc="Processing Samples"):
    sample = train_data[sample_idx]

    # Find the closest match by minimizing the total squared error
    errors = np.sum((original_data_array - sample) ** 2, axis=1)
    min_error_idx = np.argmin(errors)  # Get the index of the smallest error
    min_error_value = errors[min_error_idx]  # Get the actual minimum error

    print(f"Sample {sample_idx}: Closest match is row {min_error_idx+1} with minimum error {min_error_value:.10f}")

    # Retrieve the matching row from the cleaned dataset (ensures 401-699 nm are valid)
    closest_sample = cleaned_data.iloc[min_error_idx]

    # Convert to dictionary for classification (including Rrs_740)
    sample_dict = {int(col.split("_")[1]): closest_sample[col] for col in wavelength_cols}

    # Perform classification
    water_type = classify_water_type(sample_dict)

    # Store results
    classified_results.append({"Sample_ID": min_error_idx + 1, "Water_Type": water_type})

water_type_mapping = {
    "Type 1 (Blue-Green Water)": 1,
    "Type 2 (Green Water)": 2,
    "Type 3 (Brown Water)": 3
}

# Convert classified_results to a NumPy array of shape (N, 1)
class_np = np.array([[water_type_mapping[result["Water_Type"]]] for result in classified_results])

# Print shape to verify
print("Shape of class_np:", class_np.shape)  # Should be (N, 1)
print("First 5 classifications:\n", class_np[:5])

save_path = "D:/Research/EnvironmentalData/1800Data/train_water_class.npy"

np.save(save_path, class_np)

print(f"Classification labels saved successfully at: {save_path}")