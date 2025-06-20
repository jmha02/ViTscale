# ViTscale: ViT Performance Benchmarking with LoRA

A comprehensive benchmarking suite for analyzing Vision Transformer (ViT) performance with LoRA (Low-Rank Adaptation) optimizations. This project provides detailed analysis of both **inference** and **training** performance characteristics.

## 🚀 Features

### 🔍 **Dual Performance Analysis**
- **Inference Benchmarking**: Measures forward pass speed, memory usage, and FLOPs
- **Training Benchmarking**: Measures full training step (forward + backward + optimizer) performance
- **Comparative Analysis**: Side-by-side comparison of baseline vs LoRA models

### ⚡ **Key Metrics**
- **Speed**: Inference time, training time per batch, throughput (FPS/samples per sec)
- **Memory**: Peak GPU memory usage during inference and training
- **Efficiency**: Parameter count, trainable parameter percentage, computational load (FLOPs)
- **Scalability**: LoRA rank impact analysis (ranks 4, 8, 16, 32, 64)

### 📊 **Comprehensive Reporting**
- Interactive HTML reports with detailed analysis
- Visualization plots and charts
- Performance trade-off analysis
- Recommendations for optimal configurations

## 🛠️ Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate vitscale

# Verify CUDA setup (recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📈 Usage

### 1. **Inference Performance Benchmarking**
```bash
# Run inference benchmark (forward pass only)
python run_inference_benchmark.py

# Analyze inference results
python run_inference_analysis.py
```
**Measures**: Forward pass speed, memory consumption, FLOPs computation

### 2. **Training Performance Benchmarking**
```bash
# Run training benchmark (forward + backward + optimizer)
python run_training_benchmark.py

# Analyze training results  
python run_training_analysis.py
```
**Measures**: Full training step performance, gradient computation time, memory efficiency

### 3. **Basic Training (Optional)**
```bash
# Train baseline ViT
python main.py --config configs/default.yaml --mode baseline

# Train ViT with LoRA
python main.py --config configs/default.yaml --mode lora
```

## 📁 Project Structure

```
ViTscale/
├── 🔧 Core Scripts
│   ├── run_inference_benchmark.py    # Inference performance benchmarking
│   ├── run_training_benchmark.py     # Training performance benchmarking  
│   ├── run_inference_analysis.py     # Inference results analysis
│   ├── run_training_analysis.py      # Training results analysis
│   └── main.py                       # Basic training script
│
├── 🏗️ Framework
│   ├── benchmarks/                   # Core benchmark implementations
│   │   ├── inference.py              # Inference benchmark logic
│   │   └── training.py               # Training benchmark logic
│   ├── analysis/                     # Analysis implementations
│   │   ├── inference.py              # Inference analysis logic
│   │   └── training.py               # Training analysis logic
│   ├── models/                       # ViT and LoRA model implementations
│   ├── utils/                        # Performance measurement utilities
│   └── configs/                      # Configuration files
│
├── 📊 Results
│   ├── plots/                        # Benchmark results and visualizations
│   └── analysis/                     # Detailed analysis reports
│       └── results/                  # Generated analysis outputs
│           ├── inference/            # Inference-specific analysis
│           └── training/             # Training-specific analysis
│
└── 📋 Setup
    ├── environment.yml               # Conda environment setup
    └── README.md                    # This file
```

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Model and dataset
model_name: "vit_base_patch16_224"
dataset: "cifar10"
batch_size: 32

# LoRA parameters  
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
target_modules: ["qkv", "proj"]  # ViT attention modules

# Performance settings
warmup_iterations: 5
benchmark_iterations: 100
```
