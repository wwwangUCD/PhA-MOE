import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

save_dir = r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\MOE'

# Define file paths
train_data_path = r"D:\Research\EnvironmentalData\1800Data\train_data.npy"
train_moe_routing_weights_path = r"D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\MOE\train_moe_routing_weights.npy"

# Load the data
train_data = np.load(train_data_path)  # Shape: (N, 40)
train_moe_routing_weights = np.load(train_moe_routing_weights_path)  # Shape: (N, 8)

N = train_moe_routing_weights.shape[0]  # Number of samples
top_3_gates = np.zeros((N, 3), dtype=int)
top_8_gates = np.zeros((N, 8), dtype=int)
# Loop over each sample and find the top 3 expert gates
for i in range(N):
    sorted_indices = np.argsort(train_moe_routing_weights[i])[::-1]  # Sort in descending order
    top_3_gates[i] = sorted_indices[:3]
    top_8_gates[i] = sorted_indices[:8]
# Perform t-SNE to reduce dimensions from 40 to 3
tsne_3d = TSNE(n_components=3, random_state=42)
train_data_3d = tsne_3d.fit_transform(train_data)  # Shape: (N, 3)

# Define colors for 8 gates
gate_colors = {
    0: 'red', 1: 'blue', 2: 'green', 3: 'orange',
    4: 'purple', 5: 'brown', 6: 'pink', 7: 'gray'
}

# Create and save separate 3D scatter plots for the top 3 gates
for i in range(8):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Assign colors based on the gate index
    scatter_colors = [gate_colors[idx] for idx in top_8_gates[:, i]]

    # Scatter plot
    ax.scatter(train_data_3d[:, 0], train_data_3d[:, 1], train_data_3d[:, 2], c=scatter_colors, alpha=0.7, linewidth=3)

    # Title and labels
    ax.set_title(f"3D t-SNE Scatter for Top {i + 1} Routing Probabilities", fontsize=32)
    # ax.set_xlabel("t-SNE Dimension 1", fontsize=28)
    # ax.set_ylabel("t-SNE Dimension 2", fontsize=28)
    # ax.set_zlabel("t-SNE Dimension 3", fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=24)

    # Create legend and move it to the right outside the plot
    legend_patches = [mpatches.Patch(color=color, label=f"Expert {idx}") for idx, color in gate_colors.items()]
    plt.legend(handles=legend_patches, fontsize=24, loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the figure
    save_path = os.path.join(save_dir, f"EMIT_gate_top_{i + 1}_3D.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
