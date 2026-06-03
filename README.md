# Vision Transformer Knowledge Distillation

A modular PyTorch framework for studying knowledge distillation techniques in Vision Transformers (ViTs). This project investigates how effectively a lightweight student Vision Transformer can recover the representational power of a larger teacher Vision Transformer using multiple distillation strategies.

## Project Overview

Knowledge Distillation (KD) is a model compression technique where a compact student model learns from a larger teacher model. This repository is designed to compare multiple distillation approaches for Vision Transformers and evaluate their impact on downstream computer vision tasks.

### Teacher Model

* ViT-Large (Teacher)

### Student Model

* ViT-Small / ViT-Tiny (Student)

## Distillation Strategies

The project is structured to support the following experiments:

1. Baseline Student Training
2. Logit-Based Knowledge Distillation
3. Feature-Based Distillation
4. Attention Transfer
5. Relational Knowledge Distillation (RKD)
6. Vision Transformer Knowledge Distillation (ViTKD)
7. Combined Distillation Strategy

## Repository Structure

```
configs/
datasets/
evaluation/
experiments/
losses/
models/
trainers/

train.py
requirements.txt
README.md
```

## Current Implementation

Implemented:

* Teacher Vision Transformer module
* Student Vision Transformer module
* Knowledge Distillation loss module
* Modular project structure
* Git-based experiment tracking

Planned:

* Feature Distillation
* Attention Transfer
* RKD
* ViTKD
* Experiment tracking
* Evaluation pipelines

## Research Motivation

Large Vision Transformers provide excellent performance but require substantial computational resources. This project explores how much of the teacher's performance can be transferred to smaller student models while maintaining efficiency.

## Technologies

* Python
* PyTorch
* TIMM
* TorchVision
* NumPy
* Matplotlib

## Future Work

* Oxford-IIIT Pet Classification
* Pascal VOC Segmentation
* NYUv2 Depth Estimation
* Comparative Distillation Analysis
* Ablation Studies

## References

* Hinton et al., Distilling the Knowledge in a Neural Network
* Park et al., Relational Knowledge Distillation
* Yang et al., ViTKD: Feature-based Knowledge Distillation for Vision Transformers (CVPRW 2024)

## Author

Saine Dehingia

M.Tech, Computer Science & Engineering

Research Interests:

* Computer Vision
* Deep Learning
* Vision Transformers
* Knowledge Distillation
* Explainable AI
