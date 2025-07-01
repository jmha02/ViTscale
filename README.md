# Simple ViT CIFAR-100 Training

Minimal Vision Transformer training scripts for CIFAR-100 dataset - both regular and LoRA versions.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Regular ViT Training
```bash
python train_vit_cifar100.py
```

### LoRA ViT Training
```bash
python train_vit_lora_cifar100.py
```

### Performance Comparison
```bash
python compare_vit_lora.py
```

### Continual Learning (Split-CIFAR100)
```bash
# Baseline continual learning (no regularization)
python train_vit_continual_baseline.py
python train_vit_lora_continual_baseline.py

# Compare baseline approaches
python compare_continual_baseline.py  # Baseline ViT vs LoRA

# Advanced continual learning (with EWC + replay)
python train_vit_continual.py
python train_vit_lora_continual.py
```

The training scripts will:
- Download CIFAR-100 dataset automatically
- Train a ViT-Small model for 50 epochs
- Save the best model
- Print training progress every 100 batches

## What it does

- **Model**: ViT-Small (Vision Transformer Small) from timm
- **Dataset**: CIFAR-100 (100 classes, 32x32 images resized to 224x224)
- **Training**: 50 epochs with AdamW optimizer
- **LoRA**: Uses HuggingFace PEFT library with rank=8, alpha=16
- **No configs, no arguments** - just run and train!

The LoRA version trains only ~1-2% of parameters compared to full fine-tuning while maintaining similar performance.

## Performance Comparison

The `compare_vit_lora.py` script provides detailed benchmarks:

- **Memory Usage**: GPU and CPU memory consumption
- **Training Speed**: Per epoch, per batch, and per step timing
- **Parameter Efficiency**: Total vs trainable parameter counts
- **Side-by-side Comparison**: Direct comparison table with improvement ratios

This helps you understand the practical benefits of LoRA in terms of memory efficiency and training performance.

## Continual Learning

The continual learning scripts demonstrate **online learning** scenarios:

- **Split-CIFAR100**: 10 sequential tasks, 10 classes each
- **Catastrophic Forgetting**: How models forget previous tasks

**Baseline Scripts** (Pure sequential learning):
- `train_vit_continual_baseline.py` - Regular ViT without any regularization
- `train_vit_lora_continual_baseline.py` - LoRA ViT without any regularization
- Shows raw catastrophic forgetting effect

**Advanced Scripts** (With mitigation strategies):
- `train_vit_continual.py` - Regular ViT with EWC + Experience Replay
- `train_vit_lora_continual.py` - LoRA ViT with adapters + Experience Replay

**Key Features:**
- Sequential task learning (online learning)
- Catastrophic forgetting measurement
- LoRA vs full fine-tuning comparison in continual setting
- Baseline vs advanced continual learning strategies