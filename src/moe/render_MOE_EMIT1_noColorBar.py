import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib.patches as mpatches
save_dir=r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\MOE'
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

top_3_probs = np.zeros((N, 3))
top_8_probs = np.zeros((N, 8))
for i in range(N):
    top_3_probs[i] = train_moe_routing_weights[i, top_3_gates[i]]
    top_8_probs[i] = train_moe_routing_weights[i, top_8_gates[i]]

# save_path = os.path.join(save_dir,'EMIT_gate_box.pdf')
# # Plot a box plot for the top 3 gate probabilities
# plt.figure(figsize=(10, 8))
# plt.boxplot(
#     [top_3_probs[:, 0], top_3_probs[:, 1], top_3_probs[:, 2]],
#     labels=['Top 1', 'Top 2', 'Top 3'],
#     boxprops=dict(linewidth=3),
#     whiskerprops=dict(linewidth=3),
#     capprops=dict(linewidth=3),
#     medianprops=dict(linewidth=3)
# )
#
# # Customize plot appearance
# plt.title("Box Plot of Top 3 Routing Probabilities", fontsize=32)
# plt.ylabel("Probability", fontsize=32)
# plt.xlabel("Gate Ranking", fontsize=32)
# plt.xticks(fontsize=28)
# plt.yticks(fontsize=28)
# plt.grid(True, linewidth=1.5)
#
# # Save the figure
# plt.savefig(save_path, bbox_inches='tight')
# plt.show()

save_path = os.path.join(save_dir, 'EMIT_gate_box_top8.pdf')

# Adjust the figure size if needed
plt.figure(figsize=(12, 8))

# Plot a box plot for the top 8 gate probabilities
plt.boxplot(
    [top_8_probs[:, i] for i in range(8)],  # Extract each top-k probability
    labels=[f'{i+1}' for i in range(8)],  # Dynamic labels
    boxprops=dict(linewidth=3),
    whiskerprops=dict(linewidth=3),
    capprops=dict(linewidth=3),
    medianprops=dict(linewidth=3)
)

# Customize plot appearance
# plt.title("Box Plot of Routing Probabilities", fontsize=32)
plt.ylabel("Probability", fontsize=44)
plt.xlabel("Ranking", fontsize=44)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.grid(True, linewidth=1.5)

# Save the figure
plt.savefig(save_path, bbox_inches='tight')
plt.show()

# Perform t-SNE to reduce dimensions from 40 to 2
tsne = TSNE(n_components=2, random_state=42)
train_data_2d = tsne.fit_transform(train_data)  # Shape: (N, 2)

# Define colors for 8 gates
gate_colors = {
    0: 'red', 1: 'blue', 2: 'green', 3: 'orange',
    4: 'purple', 5: 'brown', 6: 'pink', 7: 'gray'
}

# # Create and save separate scatter plots for the top 3 gates
# for i in range(3):
#     plt.figure(figsize=(10, 8))
#
#     # Assign colors based on the gate index
#     scatter_colors = [gate_colors[idx] for idx in top_8_gates[:, i]]
#
#     # Scatter plot
#     plt.scatter(train_data_2d[:, 0], train_data_2d[:, 1], c=scatter_colors, alpha=0.7, linewidth=3)
#
#     # Title and labels
#     # plt.title(f"t-SNE Scatter for Top {i + 1} Routing Probabilities", fontsize=44)
#     plt.xlabel("t-SNE Dimension 1", fontsize=44)
#     plt.ylabel("t-SNE Dimension 2", fontsize=44)
#     plt.xticks(fontsize=36)
#     plt.yticks(fontsize=36)
#     plt.grid(True, linewidth=1.5)
#
#     # # Create legend and move it to the right outside the plot
#     # legend_patches = [mpatches.Patch(color=color, label=f"Expert {idx}") for idx, color in gate_colors.items()]
#     # plt.legend(handles=legend_patches, fontsize=24, loc='center left', bbox_to_anchor=(1, 0.5))
#
#     # Save the figure
#     save_path = os.path.join(save_dir, f"EMIT_gate_top{i + 1}.pdf")
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.show()
#
# # Create a standalone figure for the legend
# plt.figure(figsize=(6, 6))
#
# # Create patches for each expert
# legend_patches = [mpatches.Patch(color=color, label=f"Expert {idx}") for idx, color in gate_colors.items()]
#
# # Plot the legend only
# legend_fig = plt.legend(handles=legend_patches, fontsize=44, loc='center')
# plt.axis('off')  # Hide axes
#
# # Save the legend as a separate figure
# legend_save_path = os.path.join(save_dir, "EMIT_gate_legend.pdf")
# plt.savefig(legend_save_path, bbox_inches='tight')
# plt.show()
# plt.close()
#
