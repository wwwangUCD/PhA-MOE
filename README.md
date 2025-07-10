# PhA-MOE
This repository contains the source code for the **PhA-MOE** model for **phytoplankton absorption estimation**.

## Model Overview

The **PhA-MOE** model is built on a **Mixture Density Network (MDN)** backbone, inspired by the work:
> _["Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A machine-learning approach"](https://www.sciencedirect.com/science/article/pii/S0034425719306248)_  
> N. Pahlevan, et al. (2020), Remote Sensing of Environment. DOI: [10.1016/j.rse.2019.111604](https://doi.org/10.1016/j.rse.2019.111604)

- The original MDN code is available at: [https://github.com/BrandonSmithJ/MDN](https://github.com/BrandonSmithJ/MDN)
- We have **rewritten** the code in **PyTorch** (original version was in TensorFlow).

## Mixture-of-Experts (MoE)

However their MDN code is based on tensorflow planton form, and we have rewrote and transfered it into a Phytorch based version.

The PhA-MOE model integrates **Sparsely-Gated Mixture-of-Experts (MoE)** layers based on:

> _["Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538)_  
> Shazeer et al., 2017 (Google Brain)

- TensorFlow implementation reference:  
  [tensor2tensor/utils/expert_utils.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py)
- Pytorch implementation reference:  
  [mixture-of-experts/moe.py](https://github.com/davidmrau/mixture-of-experts)

  
## Visualization with HyperCoast

This project uses the [**HyperCoast**](https://hypercoast.org) library for analyzing and visualizing hyperspectral data.
- Free software: MIT License  
- Documentation: [https://hypercoast.org](https://hypercoast.org)

## Pretrained Models & Data Access

All pretrained model checkpoints, sample datasets, and prediction outputs (e.g., PACE maps) are available at the following Google Drive link:

**[Access Google Drive Folder](https://drive.google.com/drive/folders/1D2guZCrUm5PufIiwwIjijB6xteLeFkf6?usp=sharing)**

Contents include:

### 1. `Data/`
Contains all raw and processed datasets used in training, evaluation, and visualization.

#### a. `1800 Data/`
- `data_Rrs.csv`: Raw Rrs data (401–700 nm).
- `data_aPHY.csv`: Ground-truth aPHY data.
- Train/val/test splits: Provided for both input (`Rrs_*.npy`) and label (`aPHY_*.npy`) data.
- `WaterType_classifier.py` and outputs: For water type classification on training set.

#### b. `preprocessed 1800 Data/`
Processed versions of the raw Rrs and aPHY:
- **Rrs**: Robust scaling.
- **aPHY**: Log + min-max scaling.
- Organized under different spectral resolutions:
  - `7nm`: EMIT-like resolution.
  - `2nm`: PACE-like resolution.
- Each resolution folder includes:
  - `WL/`: Wavelength-wise preprocessing.
  - `WB/`: Whole-band preprocessing.
  - `no/`: No preprocessing.
- Useful for plotting distributions or model comparisons.

#### c. `MOE Data/`
- MOE routing weights saved during EMIT-resolution model training.
- Supports visualization of expert selection behavior in the Mixture-of-Experts framework.

#### d. `PACE Map Data/`
- Satellite-derived data including:
  - `original_map_data`: Raw PACE map files.
  - `Rrs_map`: Rrs extracted from the PACE satellite images.
  - `aphy_estimation`: Our model’s output estimation for aPHY.
- Field experiment data:
  - `Estuary_HPLC sites_Fall_2024.xlsx`: Measurements at 35 stations.
  - `eval_data.csv`: 21 stations, **used for fine-tuning**.
  - `ft_data.csv`: 14 stations, **used for evaluation**.

> **Note**: The filenames `eval_data.csv` and `ft_data.csv` are intentionally swapped relative to their actual use. We keep the original names for consistency with the codebase.

- Extracted Rrs from PACE map:
  - `ft_Rrs_all_stations`: Matches locations from `ft_data.csv` (some locations missing, e.g., index 4).
  - `Rrs_data_fromMap_ft_train`: Matches locations from `eval_data.csv` (e.g., indices 6, 16 missing).
- See `map_finetune_scatter_Rrs.py` for precise details on indexing and missing entries.



Please download the necessary files and place them into the appropriate folders before running the evaluation or visualization scripts.
**Please refer to our paper for more details on the dataset structure, preprocessing steps, and training/fine-tuning procedures.**

# Cite

If you find this work useful, please cite our paper:

```bibtex
@Article{rs17122103,
  AUTHOR = {Wang, Weiwei and Liu, Bingqing and Gao, Song and Li, Jiang and Zhou, Yueling and Zhang, Songyang and Ding, Zhi},
  TITLE = {PhA-MOE: Enhancing Hyperspectral Retrievals for Phytoplankton Absorption Using Mixture-of-Experts},
  JOURNAL = {Remote Sensing},
  VOLUME = {17},
  YEAR = {2025},
  NUMBER = {12},
  ARTICLE-NUMBER = {2103}
}
