import torch
import time
import psutil
import numpy as np
from fvcore.nn import flop_count
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import threading

class PerformanceMonitor:
    """Monitor training performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'epoch_times': [],
            'batch_times': [],
            'memory_usage': [],
            'gpu_memory': [],
            'cpu_usage': [],
            'throughput': []
        }
    
    @contextmanager
    def measure_epoch(self):
        """Context manager for measuring epoch time"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        yield
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        epoch_time = end_time - start_time
        self.metrics['epoch_times'].append(epoch_time)
        self.metrics['memory_usage'].append(end_memory)
    
    @contextmanager
    def measure_batch(self, batch_size: int):
        """Context manager for measuring batch processing time"""
        start_time = time.time()
        
        yield
        
        end_time = time.time()
        batch_time = end_time - start_time
        throughput = batch_size / batch_time
        
        self.metrics['batch_times'].append(batch_time)
        self.metrics['throughput'].append(throughput)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {
            'cpu_memory_mb': psutil.virtual_memory().used / 1024 / 1024,
            'cpu_memory_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_memory_percent': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            })
        
        return memory_info
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        if self.metrics['epoch_times']:
            summary['avg_epoch_time'] = np.mean(self.metrics['epoch_times'])
            summary['std_epoch_time'] = np.std(self.metrics['epoch_times'])
        
        if self.metrics['batch_times']:
            summary['avg_batch_time'] = np.mean(self.metrics['batch_times'])
            summary['std_batch_time'] = np.std(self.metrics['batch_times'])
        
        if self.metrics['throughput']:
            summary['avg_throughput'] = np.mean(self.metrics['throughput'])
            summary['std_throughput'] = np.std(self.metrics['throughput'])
        
        return summary

class FLOPsCalculator:
    """Calculate FLOPs for model operations"""
    
    @staticmethod
    def calculate_model_flops(
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cuda'
    ) -> Dict[str, int]:
        """Calculate FLOPs for a single forward pass"""
        
        model.eval()
        dummy_input = torch.randn(1, *input_shape).to(device)
        model = model.to(device)
        
        try:
            with torch.no_grad():
                flops_dict, _ = flop_count(
                    model,
                    (dummy_input,),
                    supported_ops={
                        "aten::add": None,
                        "aten::addmm": None,
                        "aten::bmm": None,
                        "aten::mm": None,
                        "aten::mul": None,
                        "aten::conv2d": None,
                    }
                )
            
            total_flops = sum(flops_dict.values())
        except Exception as e:
            print(f"Warning: FLOPs calculation failed: {e}")
            # For LoRA models or unsupported models, return approximate FLOPs
            # based on parameter count as fallback
            total_params = sum(p.numel() for p in model.parameters())
            total_flops = total_params * 2  # Rough approximation
            flops_dict = {"estimated": total_flops}
        
        return {
            'total_flops': total_flops,
            'detailed_flops': flops_dict
        }
    
    @staticmethod
    def calculate_training_flops(
        forward_flops: int,
        backward_multiplier: float = 2.0
    ) -> Dict[str, int]:
        """Estimate training FLOPs (forward + backward)"""
        
        backward_flops = forward_flops * backward_multiplier
        total_training_flops = forward_flops + backward_flops
        
        return {
            'forward_flops': forward_flops,
            'backward_flops': backward_flops,
            'total_training_flops': total_training_flops
        }

class BenchmarkTimer:
    """Precise timing utilities"""
    
    def __init__(self, warmup_iterations: int = 5):
        self.warmup_iterations = warmup_iterations
        self.times = []
    
    def benchmark_function(
        self,
        func,
        iterations: int = 100,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """Benchmark a function execution time"""
        
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            func(*args, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times)
        }

def compare_models_performance(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    input_shape: Tuple[int, ...],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Compare performance between two models"""
    
    # Parameter count comparison
    params1 = model1.get_trainable_parameters()
    params2 = model2.get_trainable_parameters()
    
    # FLOPs comparison
    flops_calc = FLOPsCalculator()
    flops1 = flops_calc.calculate_model_flops(model1, input_shape, device)
    flops2 = flops_calc.calculate_model_flops(model2, input_shape, device)
    
    # Timing comparison
    timer = BenchmarkTimer()
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    def forward_pass(model):
        with torch.no_grad():
            return model(dummy_input)
    
    timing1 = timer.benchmark_function(forward_pass, iterations=100, model=model1)
    timing2 = timer.benchmark_function(forward_pass, iterations=100, model=model2)
    
    comparison = {
        'models': {
            model1_name: {
                'parameters': params1,
                'flops': flops1,
                'timing': timing1
            },
            model2_name: {
                'parameters': params2,
                'flops': flops2,
                'timing': timing2
            }
        },
        'comparison': {
            'parameter_reduction': (params1['trainable_params'] - params2['trainable_params']) / params1['trainable_params'] * 100,
            'flops_reduction': (flops1['total_flops'] - flops2['total_flops']) / flops1['total_flops'] * 100,
            'speedup': timing1['mean_time'] / timing2['mean_time']
        }
    }
    
    return comparison

def visualize_performance_comparison(comparison_data: Dict[str, Any], save_path: Optional[str] = None):
    """Create visualization of performance comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    models = list(comparison_data['models'].keys())
    model1_data = comparison_data['models'][models[0]]
    model2_data = comparison_data['models'][models[1]]
    
    # Parameters comparison
    params_data = [
        model1_data['parameters']['trainable_params'],
        model2_data['parameters']['trainable_params']
    ]
    axes[0, 0].bar(models, params_data, color=['blue', 'orange'])
    axes[0, 0].set_title('Trainable Parameters')
    axes[0, 0].set_ylabel('Number of Parameters')
    axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # FLOPs comparison
    flops_data = [
        model1_data['flops']['total_flops'],
        model2_data['flops']['total_flops']
    ]
    axes[0, 1].bar(models, flops_data, color=['blue', 'orange'])
    axes[0, 1].set_title('FLOPs per Forward Pass')
    axes[0, 1].set_ylabel('FLOPs')
    axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Timing comparison
    timing_data = [
        model1_data['timing']['mean_time'] * 1000,  # Convert to ms
        model2_data['timing']['mean_time'] * 1000
    ]
    timing_std = [
        model1_data['timing']['std_time'] * 1000,
        model2_data['timing']['std_time'] * 1000
    ]
    axes[1, 0].bar(models, timing_data, yerr=timing_std, color=['blue', 'orange'], capsize=5)
    axes[1, 0].set_title('Inference Time')
    axes[1, 0].set_ylabel('Time (ms)')
    
    # Comparison metrics
    comparison_metrics = comparison_data['comparison']
    metrics_names = ['Parameter\nReduction (%)', 'FLOPs\nReduction (%)', 'Speedup\n(x times)']
    metrics_values = [
        comparison_metrics['parameter_reduction'],
        comparison_metrics['flops_reduction'],
        comparison_metrics['speedup']
    ]
    
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=['green', 'red', 'purple'])
    axes[1, 1].set_title('Performance Improvements')
    axes[1, 1].set_ylabel('Improvement')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig
