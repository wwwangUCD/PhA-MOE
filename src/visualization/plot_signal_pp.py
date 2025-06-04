import os
import numpy as np
import matplotlib.pyplot as plt
import math

# Define paths
base_path = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\2nm'# 2nm for PACE and 7nm for EMIT
save_path = r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\pp\2nm'
folders = ['no', 'WL', 'WB']

# Define wavelength axis
# numbers = [
#     403.2254, 410.638, 418.0536, 425.47214, 432.8927, 440.31726, 447.7428,
#     455.17035, 462.59888, 470.0304, 477.46292, 484.89743, 492.33292, 499.77142,
#     507.2099, 514.6504, 522.0909, 529.5333, 536.9768, 544.42126, 551.8667,
#     559.3142, 566.7616, 574.20905, 581.6585, 589.108, 596.55835, 604.0098,
#     611.4622, 618.9146, 626.36804, 633.8215, 641.2759, 648.7303, 656.1857,
#     663.6411, 671.09753, 678.5539, 686.0103, 693.4677
# ]# EMIT
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
]  # PACE


selected_wl = [math.floor(num) for num in numbers]

# Load data
data_dict = {}
for folder in folders:
    data_path = os.path.join(base_path, folder)
    Rrs = np.load(os.path.join(data_path, 'train_data.npy'))
    Aphy = np.load(os.path.join(data_path, 'train_labels.npy'))
    data_dict[folder] = {'Rrs': Rrs, 'Aphy': Aphy}

# Define plotting function
def plot_spectral(data, wavelengths, signal_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, data.flatten(), linewidth=3)
    ax.set_xlabel("Wavelength (nm)", fontsize=34)
    ylabel = r"$R_{rs}$" if signal_type == "Rrs" else r"$a_{phy}$"
    ax.set_ylabel(ylabel, fontsize=34)
    ax.tick_params(axis='both', labelsize=34)
    ax.grid(True, linestyle='--', alpha=0.5)  # Add this line
    fig.tight_layout()
    return fig

# Set index
idx = 0

# Generate and save plots
os.makedirs(save_path, exist_ok=True)
for folder in folders:
    for signal_type in ['Rrs', 'Aphy']:
        data = data_dict[folder][signal_type][idx]
        fig = plot_spectral(data, selected_wl, signal_type)
        filename = f"{signal_type}_{folder}.pdf"
        #fig.savefig(os.path.join(save_path, filename), bbox_inches='tight')
        fig.show()
        plt.close(fig)
