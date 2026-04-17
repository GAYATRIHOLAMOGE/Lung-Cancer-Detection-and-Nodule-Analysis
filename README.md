# 🫁 Lung Cancer Detection & Nodule Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch)
![LUNA16](https://img.shields.io/badge/Dataset-LUNA16-green?style=flat-square)
![VRAM](https://img.shields.io/badge/VRAM-4GB%20Constrained-orange?style=flat-square)

## Overview

This project implements a complete, memory-efficient two-stage 3D deep learning pipeline for automated pulmonary nodule detection and malignancy classification using the LUNA16 dataset (888 CT scans across 10 subsets). The first stage employs a 3D U-Net with residual convolutional blocks and instance normalisation, trained on 64³ voxel patches using a combined Focal-Dice loss to handle extreme foreground-background imbalance, performing sliding-window inference over full CT volumes to generate candidate nodule heatmaps. The second stage passes detected candidates to a lightweight 3D ResNet-10 classifier with Squeeze-and-Excitation attention blocks, trained on 32³ crops to predict malignancy probability. The entire pipeline — from raw `.mhd` CT volumes through isotropic resampling, HU normalisation, patch extraction, augmentation, training, and evaluation — is constrained to a 4GB VRAM budget using FP16 automatic mixed precision and gradient checkpointing, making it fully reproducible on consumer-grade hardware. Spatial explainability is provided through 3D Grad-CAM visualisations that generate voxel-level activation heatmaps highlighting regions most influential to each malignancy prediction, constituting a complete Computer-Aided Detection (CAD) system prototype.

---
