#!/usr/bin/env python3

import argparse
import yaml
import os
import torch
import numpy as np
import random
from pathlib import Path

from models import create_model
from utils import (
    get_dataset, Trainer, create_optimizer, create_scheduler,
    PerformanceMonitor, compare_models_performance, visualize_performance_comparison
)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='ViT Fine-tuning with LoRA Performance Analysis')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['baseline', 'lora', 'compare'],
                       default='compare', help='Training mode')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (overrides config)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lora_rank', type=int, default=None,
                       help='LoRA rank (overrides config)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['model_name'] = args.model
    if args.dataset:
        config['dataset'] = args.dataset
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lora_rank:
        config['lora_rank'] = args.lora_rank
    if args.device:
        config['device'] = args.device
    
    # Set device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['plot_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {config['dataset']}")
    train_loader, val_loader, num_classes = get_dataset(
        config['dataset'],
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )
    
    if args.mode == 'baseline':
        print("Training baseline ViT model...")
        train_baseline(config, train_loader, val_loader, num_classes, device)
    
    elif args.mode == 'lora':
        print("Training ViT model with LoRA...")
        train_lora(config, train_loader, val_loader, num_classes, device)
    
    elif args.mode == 'compare':
        print("Comparing baseline and LoRA models...")
        compare_models(config, train_loader, val_loader, num_classes, device)

def train_baseline(config, train_loader, val_loader, num_classes, device):
    """Train baseline ViT model"""
    
    # Create model
    model = create_model(
        config['model_name'],
        num_classes,
        mode='baseline'
    )
    
    print(f"Model parameters: {model.get_trainable_parameters()}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        config['optimizer'],
        config['learning_rate'],
        config['weight_decay']
    )
    
    scheduler = create_scheduler(
        optimizer,
        config['scheduler'],
        config['epochs']
    )
    
    # Create trainer
    trainer = Trainer(model, device, optimizer=optimizer, scheduler=scheduler)
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    
    # Train
    save_path = os.path.join(config['model_dir'], 'baseline_best.pth')
    history = trainer.fit(
        train_loader,
        val_loader,
        config['epochs'],
        config['log_interval'],
        save_path
    )
    
    print("\nBaseline training completed!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Average epoch time: {np.mean(history['epoch_times']):.2f}s")

def train_lora(config, train_loader, val_loader, num_classes, device):
    """Train ViT model with LoRA"""
    
    # LoRA configuration
    lora_config = {
        'lora_rank': config['lora_rank'],
        'lora_alpha': config['lora_alpha'],
        'lora_dropout': config['lora_dropout'],
        'target_modules': config['target_modules']
    }
    
    # Create model
    model = create_model(
        config['model_name'],
        num_classes,
        mode='lora',
        lora_config=lora_config
    )
    
    print(f"Model parameters: {model.get_trainable_parameters()}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        config['optimizer'],
        config['learning_rate'],
        config['weight_decay']
    )
    
    scheduler = create_scheduler(
        optimizer,
        config['scheduler'],
        config['epochs']
    )
    
    # Create trainer
    trainer = Trainer(model, device, optimizer=optimizer, scheduler=scheduler)
    
    # Train
    save_path = os.path.join(config['model_dir'], 'lora_best.pth')
    history = trainer.fit(
        train_loader,
        val_loader,
        config['epochs'],
        config['log_interval'],
        save_path
    )
    
    print("\nLoRA training completed!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Average epoch time: {np.mean(history['epoch_times']):.2f}s")

def compare_models(config, train_loader, val_loader, num_classes, device):
    """Compare baseline and LoRA models"""
    
    # Create baseline model
    baseline_model = create_model(
        config['model_name'],
        num_classes,
        mode='baseline'
    )
    
    # LoRA configuration
    lora_config = {
        'lora_rank': config['lora_rank'],
        'lora_alpha': config['lora_alpha'],
        'lora_dropout': config['lora_dropout'],
        'target_modules': config['target_modules']
    }
    
    # Create LoRA model
    lora_model = create_model(
        config['model_name'],
        num_classes,
        mode='lora',
        lora_config=lora_config
    )
    
    # Move models to device
    baseline_model = baseline_model.to(device)
    lora_model = lora_model.to(device)
    
    # Compare performance
    print("Comparing model performance...")
    input_shape = (3, config['image_size'], config['image_size'])
    
    comparison = compare_models_performance(
        baseline_model,
        lora_model,
        input_shape,
        "Baseline ViT",
        "LoRA ViT",
        device
    )
    
    # Print comparison results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*60)
    
    baseline_data = comparison['models']['Baseline ViT']
    lora_data = comparison['models']['LoRA ViT']
    comp_data = comparison['comparison']
    
    print(f"\nBaseline ViT:")
    print(f"  Trainable Parameters: {baseline_data['parameters']['trainable_params']:,}")
    print(f"  Total FLOPs: {baseline_data['flops']['total_flops']:,}")
    print(f"  Inference Time: {baseline_data['timing']['mean_time']*1000:.2f} ± {baseline_data['timing']['std_time']*1000:.2f} ms")
    
    print(f"\nLoRA ViT:")
    print(f"  Trainable Parameters: {lora_data['parameters']['trainable_params']:,}")
    print(f"  Trainable Percentage: {lora_data['parameters']['trainable_percentage']:.2f}%")
    print(f"  Total FLOPs: {lora_data['flops']['total_flops']:,}")
    print(f"  Inference Time: {lora_data['timing']['mean_time']*1000:.2f} ± {lora_data['timing']['std_time']*1000:.2f} ms")
    
    print(f"\nPerformance Improvements:")
    print(f"  Parameter Reduction: {comp_data['parameter_reduction']:.2f}%")
    print(f"  FLOPs Reduction: {comp_data['flops_reduction']:.2f}%")
    print(f"  Speed-up: {comp_data['speedup']:.2f}x")
    
    # Create visualization
    if config['save_plots']:
        plot_path = os.path.join(config['plot_dir'], 'performance_comparison.png')
        visualize_performance_comparison(comparison, plot_path)
        print(f"\nPerformance comparison plot saved to: {plot_path}")
    
    # Train both models for accuracy comparison
    print("\n" + "="*60)
    print("TRAINING BOTH MODELS FOR ACCURACY COMPARISON")
    print("="*60)
    
    # Train baseline
    print("\nTraining baseline model...")
    baseline_optimizer = create_optimizer(
        baseline_model, config['optimizer'], config['learning_rate'], config['weight_decay']
    )
    baseline_trainer = Trainer(baseline_model, device, optimizer=baseline_optimizer)
    baseline_history = baseline_trainer.fit(
        train_loader, val_loader, config['epochs'], config['log_interval']
    )
    
    # Train LoRA
    print("\nTraining LoRA model...")
    lora_optimizer = create_optimizer(
        lora_model, config['optimizer'], config['learning_rate'], config['weight_decay']
    )
    lora_trainer = Trainer(lora_model, device, optimizer=lora_optimizer)
    lora_history = lora_trainer.fit(
        train_loader, val_loader, config['epochs'], config['log_interval']
    )
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL TRAINING RESULTS")
    print("="*60)
    
    baseline_best_acc = max(baseline_history['val_acc'])
    lora_best_acc = max(lora_history['val_acc'])
    baseline_avg_time = np.mean(baseline_history['epoch_times'])
    lora_avg_time = np.mean(lora_history['epoch_times'])
    
    print(f"\nBaseline ViT:")
    print(f"  Best Validation Accuracy: {baseline_best_acc:.2f}%")
    print(f"  Average Epoch Time: {baseline_avg_time:.2f}s")
    print(f"  Total Training Time: {sum(baseline_history['epoch_times']):.2f}s")
    
    print(f"\nLoRA ViT:")
    print(f"  Best Validation Accuracy: {lora_best_acc:.2f}%")
    print(f"  Average Epoch Time: {lora_avg_time:.2f}s")
    print(f"  Total Training Time: {sum(lora_history['epoch_times']):.2f}s")
    
    print(f"\nTraining Comparison:")
    print(f"  Accuracy Difference: {lora_best_acc - baseline_best_acc:.2f}%")
    print(f"  Training Speed-up: {baseline_avg_time / lora_avg_time:.2f}x")
    print(f"  Total Time Reduction: {(sum(baseline_history['epoch_times']) - sum(lora_history['epoch_times'])):.2f}s")

if __name__ == '__main__':
    main()
