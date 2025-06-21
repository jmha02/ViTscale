#!/usr/bin/env python3
"""
ViT-Small/16 + CIFAR-100 Training Comparison
Com            # Create base model
            base_model = timm.create_model(
                self.config['model_name'],
                pretrained=True,  # Use pretrained model for fine-tuning
                num_classes=self.num_classes
            )aseline vs LoRA training performance
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import argparse
from pathlib import Path

from models.vit_models import create_model
from utils.data import get_dataset
from utils.performance import BenchmarkTimer
from utils.training import train_epoch, evaluate_model

# PEFT imports for LoRA
import timm
from peft import LoraConfig, get_peft_model

class ViTLoRAComparator:
    def __init__(self, config_path):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Create directories
        self.results_dir = Path(self.config['results_dir'])
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.train_loader, self.val_loader, self.num_classes = get_dataset(
            self.config['dataset'],
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            image_size=self.config['image_size']
        )
        
        print(f"Dataset: {self.config['dataset']}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
    
    def create_model_and_optimizer(self, mode='baseline'):
        """Create model and optimizer for given mode"""
        print(f"\\nCreating {mode} model...")
        
        if mode == 'baseline':
            # Create baseline model using timm
            model = timm.create_model(
                self.config['model_name'],
                pretrained=True,  # Use pretrained model for fine-tuning
                num_classes=self.num_classes
            ).to(self.device)
            
        elif mode == 'lora':
            # Create base model
            base_model = timm.create_model(
                self.config['model_name'],
                pretrained=True,
                num_classes=self.num_classes
            )
            
            # Configure LoRA using PEFT
            lora_config = LoraConfig(
                # For vision models, we don't specify task_type or use None
                inference_mode=False,
                r=self.config.get('lora_rank', 8),
                lora_alpha=self.config.get('lora_alpha', 16),
                lora_dropout=self.config.get('lora_dropout', 0.1),
                target_modules=self.config.get('target_modules', ['qkv', 'proj'])
            )
            
            # Apply LoRA to the model
            model = get_peft_model(base_model, lora_config).to(self.device)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Create optimizer (only for trainable parameters in LoRA case)
        if mode == 'lora':
            # Only optimize LoRA parameters
            optimizer_params = [p for p in model.parameters() if p.requires_grad]
        else:
            optimizer_params = model.parameters()
            
        if self.config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                optimizer_params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            optimizer = optim.SGD(
                optimizer_params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                momentum=0.9
            )
        
        # Create scheduler
        if self.config.get('scheduler') == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['epochs']
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        
        return model, optimizer, scheduler, {'total_params': total_params, 'trainable_params': trainable_params}
    
    def train_model(self, mode='baseline'):
        """Train model for given mode"""
        print(f"\\n{'='*60}")
        print(f"TRAINING {mode.upper()} MODEL")
        print(f"{'='*60}")
        
        # Create model and optimizer
        model, optimizer, scheduler, param_info = self.create_model_and_optimizer(mode)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_time': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        best_model_path = self.checkpoint_dir / f"{mode}_best_model.pth"
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss, train_acc = train_epoch(
                model, self.train_loader, criterion, optimizer, self.device
            )
            
            # Validation phase
            val_loss, val_acc = evaluate_model(
                model, self.val_loader, criterion, self.device
            )
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            epoch_time = time.time() - epoch_start
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['epoch_time'].append(epoch_time)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.config.get('save_best_model', True):
                    if mode == 'lora':
                        # For PEFT LoRA models, save only the LoRA adapters
                        model.save_pretrained(best_model_path.parent / f"{mode}_best_lora")
                        # Also save training info
                        torch.save({
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_acc': best_val_acc,
                            'config': self.config
                        }, best_model_path.parent / f"{mode}_training_info.pth")
                    else:
                        # For baseline models, save normally
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_acc': best_val_acc,
                            'config': self.config
                        }, best_model_path)
            
            # Print progress
            if (epoch + 1) % self.config.get('eval_frequency', 10) == 0:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                print(f"  Epoch Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"  Best Val Acc: {best_val_acc:.2f}%")
        
        total_time = time.time() - start_time
        
        print(f"\\nTraining completed!")
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Combine results
        results = {
            'mode': mode,
            'total_params': param_info['total_params'],
            'trainable_params': param_info['trainable_params'],
            'best_val_acc': best_val_acc,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'total_training_time': total_time,
            'avg_epoch_time': np.mean(history['epoch_time']),
            'history': history
        }
        
        return results
    
    def run_comparison(self):
        """Run training comparison between baseline and LoRA"""
        print("\\n" + "="*80)
        print("ViT-Small/16 + CIFAR-100 TRAINING COMPARISON")
        print("="*80)
        
        all_results = {}
        
        # Train both baseline and LoRA models
        for mode in ['baseline', 'lora']:
            print(f"\\n🚀 Training {mode.upper()} model...")
            results = self.train_model(mode)
            all_results[mode] = results
            
            # Save individual results
            results_file = self.results_dir / f"{mode}_results.pkl"
            import pickle
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
        
        # Show results comparison
        print("\\n" + "="*80)
        print("TRAINING COMPARISON RESULTS")
        print("="*80)
        
        for mode, result in all_results.items():
            print(f"\\n{mode.upper()} Results:")
            print(f"  Best Accuracy: {result['best_accuracy']:.4f}")
            print(f"  Training Time: {result['training_time']:.2f}s")
            print(f"  Trainable Parameters: {result['trainable_params']:,}")
            if 'total_params' in result:
                trainable_pct = (result['trainable_params'] / result['total_params']) * 100
                print(f"  Trainable %: {trainable_pct:.2f}%")
        
        # Create analysis and plots
        if len(all_results) > 1:
            self.create_comparison_analysis(all_results)
        
        return all_results
    
    def create_comparison_analysis(self, results):
        """Create comprehensive comparison analysis"""
        print("\\nCreating comparison analysis...")
        
        # Extract comparison data
        modes = list(results.keys())
        comparison_data = []
        
        for mode in modes:
            data = results[mode]
            comparison_data.append({
                'Mode': mode.capitalize(),
                'Total Params (M)': data['total_params'] / 1e6,
                'Trainable Params (M)': data['trainable_params'] / 1e6,
                'Trainable %': 100 * data['trainable_params'] / data['total_params'],
                'Best Val Acc (%)': data['best_val_acc'],
                'Final Train Acc (%)': data['final_train_acc'],
                'Final Val Acc (%)': data['final_val_acc'],
                'Total Training Time (h)': data['total_training_time'] / 3600,
                'Avg Epoch Time (s)': data['avg_epoch_time']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_file = self.results_dir / 'comparison_summary.csv'
        comparison_df.to_csv(comparison_file, index=False)
        
        # Print comparison table
        print("\\n" + "="*80)
        print("TRAINING COMPARISON SUMMARY")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        
        # Create visualizations
        self.create_comparison_plots(results)
        
        # Create HTML report
        self.create_html_report(comparison_df, results)
        
        print(f"\\nComparison analysis saved to: {self.results_dir}")
    
    def create_comparison_plots(self, results):
        """Create comparison visualization plots"""
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Training curves comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ViT-Small/16 + CIFAR-100: Baseline vs LoRA Training Comparison', fontsize=16, fontweight='bold')
        
        modes = list(results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Training accuracy
        ax = axes[0, 0]
        for i, mode in enumerate(modes):
            history = results[mode]['history']
            epochs = range(1, len(history['train_acc']) + 1)
            ax.plot(epochs, history['train_acc'], label=f'{mode.capitalize()}', 
                   color=colors[i], linewidth=2)
        ax.set_title('Training Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation accuracy
        ax = axes[0, 1]
        for i, mode in enumerate(modes):
            history = results[mode]['history']
            epochs = range(1, len(history['val_acc']) + 1)
            ax.plot(epochs, history['val_acc'], label=f'{mode.capitalize()}', 
                   color=colors[i], linewidth=2)
        ax.set_title('Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Training loss
        ax = axes[1, 0]
        for i, mode in enumerate(modes):
            history = results[mode]['history']
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], label=f'{mode.capitalize()}', 
                   color=colors[i], linewidth=2)
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation loss
        ax = axes[1, 1]
        for i, mode in enumerate(modes):
            history = results[mode]['history']
            epochs = range(1, len(history['val_loss']) + 1)
            ax.plot(epochs, history['val_loss'], label=f'{mode.capitalize()}', 
                   color=colors[i], linewidth=2)
        ax.set_title('Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ViT-Small/16 + CIFAR-100: Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        mode_names = [mode.capitalize() for mode in modes]
        
        # Best validation accuracy
        ax = axes[0, 0]
        best_accs = [results[mode]['best_val_acc'] for mode in modes]
        bars = ax.bar(mode_names, best_accs, color=colors[:len(modes)])
        ax.set_title('Best Validation Accuracy')
        ax.set_ylabel('Accuracy (%)')
        for bar, acc in zip(bars, best_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Trainable parameters
        ax = axes[0, 1]
        trainable_params = [results[mode]['trainable_params']/1e6 for mode in modes]
        bars = ax.bar(mode_names, trainable_params, color=colors[:len(modes)])
        ax.set_title('Trainable Parameters')
        ax.set_ylabel('Parameters (M)')
        for bar, params in zip(bars, trainable_params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{params:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Training time
        ax = axes[1, 0]
        training_times = [results[mode]['total_training_time']/3600 for mode in modes]
        bars = ax.bar(mode_names, training_times, color=colors[:len(modes)])
        ax.set_title('Total Training Time')
        ax.set_ylabel('Time (hours)')
        for bar, time_h in zip(bars, training_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                   f'{time_h:.2f}h', ha='center', va='bottom', fontweight='bold')
        
        # Average epoch time
        ax = axes[1, 1]
        epoch_times = [results[mode]['avg_epoch_time'] for mode in modes]
        bars = ax.bar(mode_names, epoch_times, color=colors[:len(modes)])
        ax.set_title('Average Epoch Time')
        ax.set_ylabel('Time (seconds)')
        for bar, time_s in zip(bars, epoch_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{time_s:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comparison plots saved!")
    
    def create_html_report(self, comparison_df, results):
        """Create comprehensive HTML report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ViT-Small/16 + CIFAR-100 Training Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #4CAF50; color: white; }}
                .highlight {{ background-color: #ffff99; }}
                .img-container {{ text-align: center; margin: 20px 0; }}
                .config {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ViT-Small/16 + CIFAR-100 Training Comparison Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Model:</strong> {self.config['model_name']}</p>
                <p><strong>Dataset:</strong> {self.config['dataset'].upper()}</p>
                <p><strong>Epochs:</strong> {self.config['epochs']}</p>
            </div>
            
            <div class="section">
                <h2>🎯 Comparison Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Baseline</th>
                        <th>LoRA</th>
                        <th>Difference</th>
                    </tr>
        """
        
        # Add comparison rows
        baseline_results = results.get('baseline', {})
        lora_results = results.get('lora', {})
        
        if baseline_results and lora_results:
            comparisons = [
                ('Best Val Accuracy (%)', 'best_val_acc', '{:.2f}%'),
                ('Trainable Params (M)', 'trainable_params', '{:.1f}M', 1e6),
                ('Training Time (h)', 'total_training_time', '{:.2f}h', 3600),
                ('Avg Epoch Time (s)', 'avg_epoch_time', '{:.1f}s'),
            ]
            
            for metric_name, key, format_str, *divisor in comparisons:
                div = divisor[0] if divisor else 1
                baseline_val = baseline_results[key] / div
                lora_val = lora_results[key] / div
                
                if 'Params' in metric_name:
                    diff = f"{(lora_val - baseline_val):.1f}M ({(lora_val/baseline_val - 1)*100:+.1f}%)"
                elif 'Accuracy' in metric_name:
                    diff = f"{(lora_val - baseline_val):+.2f}%"
                else:
                    diff = f"{(lora_val - baseline_val):+.2f} ({(lora_val/baseline_val - 1)*100:+.1f}%)"
                
                html_content += f"""
                    <tr>
                        <td><strong>{metric_name}</strong></td>
                        <td>{format_str.format(baseline_val)}</td>
                        <td>{format_str.format(lora_val)}</td>
                        <td>{diff}</td>
                    </tr>
                """
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>📊 Training Curves</h2>
                <div class="img-container">
                    <img src="training_curves_comparison.png" alt="Training Curves" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Performance Metrics</h2>
                <div class="img-container">
                    <img src="performance_comparison.png" alt="Performance Comparison" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>⚙️ Configuration</h2>
                <div class="config">
                    <pre>{yaml.dump(self.config, default_flow_style=False)}</pre>
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Detailed Results</h2>
                {comparison_df.to_html(index=False, classes='table', table_id='comparison-table')}
            </div>
            
            <div class="section">
                <h2>💡 Key Insights</h2>
                <ul>
        """
        
        if baseline_results and lora_results:
            param_reduction = (1 - lora_results['trainable_params'] / baseline_results['trainable_params']) * 100
            acc_diff = lora_results['best_val_acc'] - baseline_results['best_val_acc']
            time_diff = (lora_results['total_training_time'] / baseline_results['total_training_time'] - 1) * 100
            
            html_content += f"""
                    <li><strong>Parameter Efficiency:</strong> LoRA reduces trainable parameters by {param_reduction:.1f}%</li>
                    <li><strong>Accuracy Impact:</strong> LoRA {"improves" if acc_diff > 0 else "decreases"} accuracy by {abs(acc_diff):.2f}%</li>
                    <li><strong>Training Speed:</strong> LoRA training is {abs(time_diff):.1f}% {"faster" if time_diff < 0 else "slower"}</li>
                    <li><strong>Best Configuration:</strong> {"LoRA" if lora_results['best_val_acc'] > baseline_results['best_val_acc'] else "Baseline"} achieves better validation accuracy</li>
            """
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_file = self.results_dir / 'training_comparison_report.html'
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='ViT LoRA vs Full Fine-tuning Comparison')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create comparator and run comparison
    comparator = ViTLoRAComparator(args.config)
    results = comparator.run_comparison()
    
    print("\\n🎉 Training comparison completed successfully!")
    print(f"Results saved in: {comparator.results_dir}")

if __name__ == '__main__':
    main()
