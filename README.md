# PhA-MOE
This repository contains the source code for the **PhA-MOE** model for **phytoplankton absorption estimation** 

W. Wang, B. Liu, S. Gao, J. Li, Y. Zhou, S. Zhang, and Z. Ding “PhA-MOE: Enhancing Hyperspectral Retrievals for Phytoplankton Absorption Using Mixture-of-Experts”, in MDPI Remote Sensing, 2025 [Link](https://www.mdpi.com/2072-4292/17/12/2103).

## Requirements

- Python: 3.9
- Pytorch: 2.2.0+cu118

## Source Code (`src/`)

This directory contains the full implementation for model training, evaluation, fine-tuning, and visualization.

### 1. `evaluate/`
Contains code to evaluate model performance on test data. 
Metrics include:
- NRMSE (Normalized Root Mean Square Error)
- MDSA (Mean Derivative Spectral Angle)
- SSPB (Spectral Similarity Preserving Bias)
- Regression slope between predicted and ground-truth aPHY.

### 2. `finetune/`
Scripts for fine-tuning pretrained models (PhA-MOE or MDN) on the 35-station experimental dataset.  
Also includes code to:
- Apply the pretrained model directly to field data.
- Apply the fine-tuned model to PACE satellite data.

These scripts can be adapted to use your own dataset for fine-tuning and inference.

### 3. `models/`
Contains implementations of all model architectures used in this work:
- PhA-MOE (Mixture-of-Experts)
- MDN (Mixture Density Network)
- MLP (Multi-layer Perceptron)
- VAE (Variational Autoencoder)

### 4. `moe/`
Utility scripts specific to the PhA-MOE architecture, including:
- Extracting routing weights across datasets
- Visualizing the routing distribution of experts

### 5. `pace_map/`
Code for processing and visualizing PACE satellite map data, including:
- Reading and preprocessing satellite Rrs maps
- Applying models to generate estimated aPHY maps
- Patching and rendering RGB-Rrs-aPHY overlays for spatial visualization

### 6. `train/`
Scripts to train PhA-MOE, MDN, MLP, and VAE models on the Rrs–aPHY dataset.  
Supports configurable training via arguments or config files.

### 7. `utils/`
Includes:
- Metric computation functions
- Data loading utilities
- Helper functions for plotting and normalization

### 8. `visualization/`
Visualization scripts to support:
- Boxplots of data distributions
- Regression analysis of estimated vs ground-truth aPHY
- Rainbow plots showing wavelength-wise performance
- Map visualizations on PACE data

If you have any questions with our work, feel free to contact us:

Email: (wwwang1915@gmail.com)

# Acknowledgement
1. **HyperCoast**: [https://hypercoast.org](https://hypercoast.org)
  
2. **MDN**: [https://github.com/BrandonSmithJ/MDN](https://github.com/BrandonSmithJ/MDN)
   
   N. Pahlevan, et al. (2020), Remote Sensing of Environment,
   ["Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A machine-learning approach"](https://www.sciencedirect.com/science/article/pii/S0034425719306248)


3. **MoE**: [mixture-of-experts/moe.py](https://github.com/davidmrau/mixture-of-experts)

   Shazeer et al., 2017 (Google Brain),
   ["Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538)


# Reference

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
