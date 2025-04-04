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
