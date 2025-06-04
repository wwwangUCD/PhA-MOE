import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Define file paths
train_moe_routing_weights_path = r"D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\MOE\train_moe_routing_weights.npy"
train_water_class_path = r"D:\Research\EnvironmentalData\1800Data\train_water_class.npy"
save_dir = r"D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\MOE"

# Load the data
train_moe_routing_weights = np.load(train_moe_routing_weights_path)  # Shape: (N, 8)
train_water_class = np.load(train_water_class_path).squeeze()  # Shape: (N,), representing class labels

# Define colors for Water Types 1, 2, and 3
water_colors = {1: 'red', 2: 'blue', 3: 'green'}

# Perform t-SNE to reduce gate weights to 2D
tsne_2d = TSNE(n_components=2, random_state=42)
train_gate_2d = tsne_2d.fit_transform(train_moe_routing_weights)  # Shape: (N, 2)

# Create and save 2D t-SNE scatter plot
plt.figure(figsize=(10, 8))
scatter_colors = [water_colors[label] for label in train_water_class]

plt.scatter(train_gate_2d[:, 0], train_gate_2d[:, 1], c=scatter_colors, alpha=0.7, linewidth=3)
# plt.title("2D t-SNE Scatter of Gate Weights (Colored by Water Type)", fontsize=32)
plt.xlabel("t-SNE Dimension 1", fontsize=32)
plt.ylabel("t-SNE Dimension 2", fontsize=32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.grid(True, linewidth=1.5)

# Create legend
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markersize=15, markerfacecolor=color, label=f"Water Type {label}")
                  for label, color in water_colors.items()]
plt.legend(handles=legend_patches, fontsize=24, loc='center left', bbox_to_anchor=(1, 0.5))

# Save 2D figure
save_path_2d = os.path.join(save_dir, "EMIT_WaterClass_tSNE_2D.pdf")
plt.savefig(save_path_2d, bbox_inches='tight')
plt.show()

# Perform t-SNE to reduce gate weights to 3D
tsne_3d = TSNE(n_components=3, random_state=42)
train_gate_3d = tsne_3d.fit_transform(train_moe_routing_weights)  # Shape: (N, 3)

# Create and save 3D t-SNE scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_gate_3d[:, 0], train_gate_3d[:, 1], train_gate_3d[:, 2], c=scatter_colors, alpha=0.7, linewidth=3)

# ax.set_title("3D t-SNE Scatter of Gate Weights (Colored by Water Type)", fontsize=32)
# ax.set_xlabel("t-SNE Dimension 1", fontsize=28)
# ax.set_ylabel("t-SNE Dimension 2", fontsize=28)
# ax.set_zlabel("t-SNE Dimension 3", fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=24)

# Create legend for 3D plot
legend_patches_3d = [mpatches.Patch(color=color, label=f"Water Type {label}") for label, color in water_colors.items()]
plt.legend(handles=legend_patches_3d, fontsize=24, loc='center left', bbox_to_anchor=(1, 0.5))

# Save 3D figure
save_path_3d = os.path.join(save_dir, "EMIT_WaterClass_tSNE_3D.pdf")
plt.savefig(save_path_3d, bbox_inches='tight')
plt.show()

print("2D and 3D t-SNE scatter plots saved successfully!")
