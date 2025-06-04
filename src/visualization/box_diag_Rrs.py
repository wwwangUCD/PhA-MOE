import numpy as np
import math
import os
import matplotlib.pyplot as plt
# Step 1: Define the data directory and load Rrs.npy and Aphy.npy
# data_dir = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\WB'
data_dir=r'D:\Research\EnvironmentalData\1800Data'
save_base_dir = r"D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing"
folder = 'pp'
# filename = 'Rrs_WB_pp.pdf'
filename = 'Rrs_no_pp.pdf'
save_path = os.path.join(save_base_dir, folder, filename)

# Ensure the folder exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)
# Load the Rrs and Aphy arrays
Rrs = np.load(f'{data_dir}\\train_data.npy').squeeze()
Aphy = np.load(f'{data_dir}\\train_labels.npy').squeeze()

numbers=list(range(401, 699, 10))
# Step 3: Convert wavelengths to integers and calculate indices
initial_wl = 401  # Starting wavelength
selected_wl = [math.floor(num) for num in numbers]

# Compute indices by subtracting the initial wavelength
indices = [wl - initial_wl for wl in selected_wl]

# Step 4: Take the selected indices part of Rrs and Aphy
Rrs_selected = Rrs[:, indices]
Aphy_selected = Aphy[:, indices]

# Now you have Rrs_selected and Aphy_selected with the chosen wavelengths
print(f"Selected Rrs shape: {Rrs_selected.shape}")
print(f"Selected Aphy shape: {Aphy_selected.shape}")


# Assuming selected_wl and Rrs_selected are already defined
fontsize = 28  # Set the desired fontsize

fig, ax = plt.subplots(figsize=(12, 6))

# Customize the boxplot appearance
boxprops = dict(linestyle='-', linewidth=2, color='black')  # Increase edge width
medianprops = dict(linestyle='-', linewidth=2, color='red')  # Make the median line red and thicker
flierprops = dict(marker='o', color='blue', markersize=8, markerfacecolor='none', markeredgewidth=1.5)   # Change outlier style
whiskerprops = dict(linewidth=2)

# Create the boxplot for the Rrs_selected data
ax.boxplot(Rrs_selected, positions=selected_wl, vert=True, patch_artist=True, widths=6,
           boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops)

# Make the boxes transparent for better clarity
for patch in ax.artists:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.6)  # Add transparency

# Add labels and title with specified fontsize
ax.set_xlabel('Wavelength (nm)', fontsize=fontsize)
ax.set_ylabel(r'$R_{rs}(sr^{-1})$', fontsize=fontsize)
# ax.set_title(r'Box plots of $R_{rs}$ data values across wavelengths', fontsize=fontsize)

# Set x-axis ticks to every second wavelength (show fewer values)
ax.set_xticks(selected_wl[::2])
ax.set_xticklabels(selected_wl[::2], rotation=45, ha='right', fontsize=fontsize)

# Set x-axis limits to tighten space between the first and last boxes and the y-axis
ax.set_xlim(min(selected_wl) - 5, max(selected_wl) + 5)

# Adjust the font size for the y-axis ticks
ax.tick_params(axis='y', labelsize=fontsize)

# Display the plot with darker gridlines
plt.grid(True, linewidth=1.5)

plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

