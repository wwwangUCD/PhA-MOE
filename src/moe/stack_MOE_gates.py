import numpy as np
import os

# Define directory containing batch files
save_dir = r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\MOE'

# Generate file names in the correct order
batch_files = [os.path.join(save_dir, f'gates_batch{i}.npy') for i in range(1, 12)]  # 1 to 11

# Load and stack all batches
all_gates = np.vstack([np.load(batch_file) for batch_file in batch_files])

# Save the final stacked routing weights
final_save_path = os.path.join(save_dir, "train_moe_routing_weights.npy")
np.save(final_save_path, all_gates)

print(f"Final stacked routing weights saved at: {final_save_path}")
print("Final shape:", all_gates.shape)  # Expected: (Total_N, 8)
