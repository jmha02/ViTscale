import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

from models import create_model
from utils import (
    get_dataset, BenchmarkTimer, FLOPsCalculator, 
    compare_models_performance, visualize_performance_comparison
)

def comprehensive_benchmark(config_path: str = 'configs/default.yaml'):
    """Run comprehensive benchmark comparing different LoRA configurations"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force CUDA usage if available
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(0)  # Use first GPU
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    print(f"Running benchmark on device: {device}")
    
    # Create directories
    plot_dir = config['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load dataset for input shape
    _, _, num_classes = get_dataset(
        config['dataset'],
        data_dir=config['data_dir'],
        batch_size=1,  # Single batch for benchmarking
        num_workers=0,
        image_size=config['image_size']
    )
    
    input_shape = (3, config['image_size'], config['image_size'])
    
    # Different LoRA configurations to test
    lora_ranks = [4, 8, 16, 32, 64]
    results = []
    
    # Baseline model
    print("Benchmarking baseline model...")
    baseline_model = create_model(
        config['model_name'],
        num_classes,
        mode='baseline'
    ).to(device)
    
    baseline_results = benchmark_single_model(
        baseline_model, input_shape, device, "Baseline"
    )
    results.append(baseline_results)
    
    # LoRA models with different ranks
    for rank in lora_ranks:
        print(f"Benchmarking LoRA model with rank {rank}...")
        
        lora_config = {
            'lora_rank': rank,
            'lora_alpha': config['lora_alpha'],
            'lora_dropout': config['lora_dropout'],
            'target_modules': config['target_modules']
        }
        
        lora_model = create_model(
            config['model_name'],
            num_classes,
            mode='lora',
            lora_config=lora_config
        ).to(device)
        
        lora_results = benchmark_single_model(
            lora_model, input_shape, device, f"LoRA-{rank}"
        )
        results.append(lora_results)
        
        # Clean up GPU memory
        del lora_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(plot_dir, 'benchmark_results.csv')
    df.to_csv(results_path, index=False)
    print(f"Benchmark results saved to: {results_path}")
    
    # Create visualizations
    create_benchmark_plots(df, plot_dir)
    
    # Print summary
    print_benchmark_summary(df)
    
    return df

def benchmark_single_model(model, input_shape, device, model_name):
    """Benchmark a single model"""
    
    # Parameter count
    if hasattr(model, 'get_trainable_parameters'):
        params_info = model.get_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        params_info = {
            'trainable_params': trainable_params,
            'all_params': total_params,
            'trainable_percentage': 100.0 * trainable_params / total_params
        }
    
    # FLOPs calculation
    flops_calc = FLOPsCalculator()
    flops_info = flops_calc.calculate_model_flops(model, input_shape, device)
    
    # Timing benchmark
    timer = BenchmarkTimer(warmup_iterations=10)
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    def forward_pass():
        with torch.no_grad():
            return model(dummy_input)
    
    timing_info = timer.benchmark_function(forward_pass, iterations=100)
    
    # Memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model(dummy_input)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        # For CPU, we can't measure GPU memory
        model(dummy_input)
        peak_memory = 0.0
    
    return {
        'model_name': model_name,
        'trainable_params': params_info['trainable_params'],
        'total_params': params_info['all_params'],
        'trainable_percentage': params_info['trainable_percentage'],
        'total_flops': flops_info['total_flops'],
        'inference_time_ms': timing_info['mean_time'] * 1000,
        'inference_std_ms': timing_info['std_time'] * 1000,
        'peak_memory_mb': peak_memory,
        'throughput_fps': 1.0 / timing_info['mean_time']
    }

def create_benchmark_plots(df, plot_dir):
    """Create comprehensive benchmark plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('ViT-LoRA Comprehensive Benchmark Results', fontsize=16, fontweight='bold')
    
    # 1. Trainable Parameters
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['model_name'], df['trainable_params'])
    ax1.set_title('Trainable Parameters')
    ax1.set_ylabel('Number of Parameters')
    ax1.tick_params(axis='x', rotation=45)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Add value labels on bars
    for bar, value in zip(bars1, df['trainable_params']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1e}', ha='center', va='bottom', fontsize=8)
    
    # 2. Trainable Percentage
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['model_name'], df['trainable_percentage'])
    ax2.set_title('Trainable Parameters Percentage')
    ax2.set_ylabel('Percentage (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, df['trainable_percentage']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. FLOPs
    ax3 = axes[0, 2]
    bars3 = ax3.bar(df['model_name'], df['total_flops'])
    ax3.set_title('FLOPs per Forward Pass')
    ax3.set_ylabel('FLOPs')
    ax3.tick_params(axis='x', rotation=45)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    for bar, value in zip(bars3, df['total_flops']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1e}', ha='center', va='bottom', fontsize=8)
    
    # 4. Inference Time
    ax4 = axes[1, 0]
    bars4 = ax4.bar(df['model_name'], df['inference_time_ms'], 
                   yerr=df['inference_std_ms'], capsize=5)
    ax4.set_title('Inference Time')
    ax4.set_ylabel('Time (ms)')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars4, df['inference_time_ms']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Memory Usage
    ax5 = axes[1, 1]
    bars5 = ax5.bar(df['model_name'], df['peak_memory_mb'])
    ax5.set_title('Peak Memory Usage')
    ax5.set_ylabel('Memory (MB)')
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars5, df['peak_memory_mb']):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Throughput
    ax6 = axes[1, 2]
    bars6 = ax6.bar(df['model_name'], df['throughput_fps'])
    ax6.set_title('Throughput')
    ax6.set_ylabel('FPS')
    ax6.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars6, df['throughput_fps']):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plot_dir, 'comprehensive_benchmark.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive benchmark plot saved to: {plot_path}")
    
    plt.show()
    
    # Create efficiency plot (parameters vs performance)
    plt.figure(figsize=(12, 8))
    
    # Efficiency scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(df['trainable_params'], df['inference_time_ms'], 
               c=range(len(df)), cmap='viridis', s=100)
    for i, model in enumerate(df['model_name']):
        plt.annotate(model, (df['trainable_params'].iloc[i], df['inference_time_ms'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Trainable Parameters')
    plt.ylabel('Inference Time (ms)')
    plt.title('Parameters vs Inference Time')
    plt.xscale('log')
    
    # Parameter reduction comparison
    plt.subplot(2, 2, 2)
    baseline_params = df[df['model_name'] == 'Baseline']['trainable_params'].iloc[0]
    param_reduction = (baseline_params - df['trainable_params']) / baseline_params * 100
    lora_models = df[df['model_name'] != 'Baseline']
    plt.bar(lora_models['model_name'], param_reduction[1:])
    plt.title('Parameter Reduction vs Baseline')
    plt.ylabel('Reduction (%)')
    plt.tick_params(axis='x', rotation=45)
    
    # Speed comparison
    plt.subplot(2, 2, 3)
    baseline_time = df[df['model_name'] == 'Baseline']['inference_time_ms'].iloc[0]
    speedup = baseline_time / df['inference_time_ms']
    plt.bar(df['model_name'], speedup)
    plt.title('Speed-up vs Baseline')
    plt.ylabel('Speed-up (x)')
    plt.tick_params(axis='x', rotation=45)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
    plt.legend()
    
    # Efficiency ratio (FPS per parameter)
    plt.subplot(2, 2, 4)
    efficiency = df['throughput_fps'] / (df['trainable_params'] / 1e6)  # FPS per million parameters
    plt.bar(df['model_name'], efficiency)
    plt.title('Efficiency (FPS per Million Parameters)')
    plt.ylabel('FPS / Million Params')
    plt.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    efficiency_plot_path = os.path.join(plot_dir, 'efficiency_analysis.png')
    plt.savefig(efficiency_plot_path, dpi=300, bbox_inches='tight')
    print(f"Efficiency analysis plot saved to: {efficiency_plot_path}")
    
    plt.show()

def print_benchmark_summary(df):
    """Print benchmark summary"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("="*80)
    
    baseline_row = df[df['model_name'] == 'Baseline'].iloc[0]
    
    print(f"\nBaseline Model Performance:")
    print(f"  Trainable Parameters: {baseline_row['trainable_params']:,}")
    print(f"  FLOPs: {baseline_row['total_flops']:,}")
    print(f"  Inference Time: {baseline_row['inference_time_ms']:.2f} ± {baseline_row['inference_std_ms']:.2f} ms")
    print(f"  Peak Memory: {baseline_row['peak_memory_mb']:.1f} MB")
    print(f"  Throughput: {baseline_row['throughput_fps']:.1f} FPS")
    
    print(f"\nLoRA Model Comparisons:")
    lora_models = df[df['model_name'] != 'Baseline']
    
    for _, row in lora_models.iterrows():
        param_reduction = (baseline_row['trainable_params'] - row['trainable_params']) / baseline_row['trainable_params'] * 100
        flops_reduction = (baseline_row['total_flops'] - row['total_flops']) / baseline_row['total_flops'] * 100
        speedup = baseline_row['inference_time_ms'] / row['inference_time_ms']
        memory_reduction = (baseline_row['peak_memory_mb'] - row['peak_memory_mb']) / baseline_row['peak_memory_mb'] * 100
        
        print(f"\n  {row['model_name']}:")
        print(f"    Trainable Parameters: {row['trainable_params']:,} ({param_reduction:.1f}% reduction)")
        print(f"    Trainable Percentage: {row['trainable_percentage']:.2f}%")
        print(f"    FLOPs: {row['total_flops']:,} ({flops_reduction:.1f}% reduction)")
        print(f"    Inference Time: {row['inference_time_ms']:.2f} ms ({speedup:.2f}x speedup)")
        print(f"    Peak Memory: {row['peak_memory_mb']:.1f} MB ({memory_reduction:.1f}% reduction)")
        print(f"    Throughput: {row['throughput_fps']:.1f} FPS")
    
    # Best performing LoRA model
    best_efficiency_idx = (lora_models['throughput_fps'] / (lora_models['trainable_params'] / 1e6)).idxmax()
    best_model = lora_models.loc[best_efficiency_idx]
    
    print(f"\nMost Efficient LoRA Configuration: {best_model['model_name']}")
    print(f"  Efficiency: {best_model['throughput_fps'] / (best_model['trainable_params'] / 1e6):.2f} FPS per million parameters")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive ViT-LoRA Benchmark')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run benchmark
    results_df = comprehensive_benchmark(args.config)
    
    print("\nBenchmark completed successfully!")

if __name__ == '__main__':
    main()
