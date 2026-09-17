# LeJEPA: Self-Supervised Vision Architecture

A clean, high-performance PyTorch reproduction of the **Latent Euclidian Joint-Embedding Predictive Architecture (LeJEPA)** paired with **SIGReg** regularization to prevent representation collapse. 

This repository features custom, ground-up implementations of modern vision backbones trained across full-scale and domain-specific datasets using distributed training pipelines.

---

## Key Features

* **Custom Vision Backbones:**
  * **ViT-Tiny:** Equipped with modern LLM/Vision Transformer optimizations—**FlashAttention**, **RoPE** (Rotary Position Embeddings), **SwiGLU** as feed-forward network, and a **Dynamic erf** (normalization-free) layer.
  * **ConvNeXtV2-Femto:** Fully custom implementation of the modern convolutional network.
* **Self-Supervised Learning Setup:**
  * Joint-Embedding Predictive Architecture (JEPA) integrated with **SIGReg** (Sketched-Isotropic Gaussian Regularization).
  * Linear probe evaluation (`detach()`'d computational graph) to benchmark representation quality.
* **Distributed ML Infrastructure:**
  * Multi-GPU training via PyTorch **Distributed Data Parallel (DDP)**.
  * **Mixed Precision (AMP)** for memory efficiency.
* **Modern MLOps Stack:**
  * Reproducible environment management via **`uv`**.
  * Hierarchical experiment configuration via **Hydra**.
  * Complete metric logging and hyperparameter tracking with **Weights & Biases**.

## Project Structure

```text
├── configs/          # Hydra configuration files
├── src/
│   ├── models/       # Custom ViT-Tiny & ConvNeXtV2-Femto modules + SIGReg
│   ├── data.py       # Custom HuggingFace dataset
|   ├── train.py          # Training pipeline
│   └── utils.py      # DDP & logging helpers
└── pyproject.toml    # Managed via uv
```

---

## Linear Probe across models and datasets

Both **ViT-Tiny** and **ConvNeXtV2-Femto** architectures are pretrained and evaluated across four distinct datasets:
To be done...

---

## Quick Start

### 1. Installation

Install dependencies using `uv`:

```bash
# Sync dependencies
uv sync
```

### 2. Distributed Training

Run DDP pretraining across GPUs using Hydra configurations:

```bash
uv run torchrun --nproc_per_node=2 train.py --config_name vit_tiny8_inet10.yaml
```