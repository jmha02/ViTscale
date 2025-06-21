# ViT LoRA Comparison

A comprehensive comparison framework for analyzing Vision Transformer (ViT) performance with LoRA (Low-Rank Adaptation) vs Full Fine-tuning.


## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate vitscale
```

## Usage

### **Run ViT LoRA Comparison**
```bash
# Run full comparison (baseline + LoRA)
python vit_lora_comparison.py

# Use quick test configuration (5 epochs)
python vit_lora_comparison.py --config configs/quick_test.yaml

# Use custom configuration
python vit_lora_comparison.py --config configs/my_custom_config.yaml
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Model and dataset (any timm model supported)
model_name: "vit_small_patch16_224"  # or "vit_base_patch16_224", etc.
dataset: "cifar100"                  # or "cifar10"
image_size: 224
batch_size: 32
epochs: 100

# LoRA parameters
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
target_modules: ["qkv", "proj"]  # ViT attention modules

# Training parameters
learning_rate: 0.001
weight_decay: 0.05
optimizer: "adamw"
scheduler: "cosine"
```