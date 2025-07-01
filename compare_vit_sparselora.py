import subprocess
import time
import argparse
import os
import json
import torch

def run_training_script(script_name, method_name):
    """Run a training script and capture its output"""
    print(f"\n{'='*70}")
    print(f"RUNNING {method_name.upper()} TRAINING")
    print(f"{'='*70}")
    
    cmd = ["python", script_name]
    print(f"Command: {' '.join(cmd)}")
    print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.perf_counter()
        
        metrics = parse_output(result.stdout, method_name)
        metrics['wall_clock_time'] = end_time - start_time
        metrics['success'] = True
        
        print(f"\n{method_name} completed successfully!")
        print(f"Wall clock time: {metrics['wall_clock_time']:.2f}s")
        
        return metrics, result.stdout, result.stderr
        
    except subprocess.CalledProcessError as e:
        end_time = time.perf_counter()
        print(f"\n{method_name} failed with error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        
        metrics = {
            'method': method_name,
            'wall_clock_time': end_time - start_time,
            'success': False,
            'error': str(e)
        }
        return metrics, e.stdout, e.stderr

def parse_output(output, method_name):
    """Parse training output to extract metrics"""
    metrics = {'method': method_name}
    lines = output.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Parameter information
        if "Total parameters:" in line:
            try:
                params = line.split(":")[-1].strip().replace(',', '')
                metrics['total_parameters'] = int(params)
            except ValueError:
                pass
        
        if "Trainable parameters:" in line:
            try:
                params = line.split(":")[-1].strip().replace(',', '')
                metrics['trainable_parameters'] = int(params)
            except ValueError:
                pass
        
        if "Parameter efficiency:" in line:
            try:
                efficiency = line.split(":")[-1].strip().rstrip('%')
                metrics['parameter_efficiency'] = float(efficiency) / 100
            except ValueError:
                pass
        
        # Performance metrics
        if "Total training time:" in line:
            try:
                time_str = line.split(":")[-1].strip().split()[0]
                metrics['training_time'] = float(time_str)
            except ValueError:
                pass
        
        if "Average epoch time:" in line:
            try:
                time_str = line.split(":")[-1].strip().split()[0]
                metrics['avg_epoch_time'] = float(time_str)
            except ValueError:
                pass
        
        if "Average batch time:" in line:
            try:
                time_str = line.split(":")[-1].strip().split()[0]
                metrics['avg_batch_time'] = float(time_str)
            except ValueError:
                pass
        
        if "Max GPU memory used:" in line:
            try:
                memory = line.split(":")[-1].strip().split()[0]
                metrics['max_gpu_memory'] = float(memory)
            except ValueError:
                pass
        
        if "Best test accuracy:" in line:
            try:
                acc = line.split(":")[-1].strip().rstrip('%')
                metrics['best_accuracy'] = float(acc)
            except ValueError:
                pass
        
        if "Training throughput:" in line:
            try:
                throughput = line.split(":")[-1].strip().split()[0]
                metrics['training_throughput'] = float(throughput)
            except ValueError:
                pass
        
        # SparseLoRA specific metrics
        if "Final overall sparsity:" in line:
            try:
                sparsity = line.split(":")[-1].strip().rstrip('%')
                metrics['final_sparsity'] = float(sparsity) / 100
            except ValueError:
                pass
        
        if "Final active LoRA parameters:" in line:
            try:
                params = line.split(":")[-1].strip().replace(',', '')
                metrics['active_lora_parameters'] = int(params)
            except ValueError:
                pass
        
        if "Effective parameter efficiency:" in line:
            try:
                efficiency = line.split(":")[-1].strip().split()[0].rstrip('%')
                metrics['effective_parameter_efficiency'] = float(efficiency) / 100
            except ValueError:
                pass
    
    return metrics

def compare_results(baseline_metrics, lora_metrics, sparselora_metrics):
    """Compare results between all three methods"""
    print(f"\n{'='*80}")
    print("VIT TRAINING COMPARISON RESULTS")
    print("Baseline ViT vs LoRA vs SparseLoRA")
    print(f"{'='*80}")
    
    # Parameter Efficiency Comparison
    print("\nPARAMETER EFFICIENCY COMPARISON:")
    print("-" * 50)
    
    methods = [
        ("Baseline", baseline_metrics),
        ("LoRA", lora_metrics), 
        ("SparseLoRA", sparselora_metrics)
    ]
    
    print(f"{'Method':<12} {'Total Params':<15} {'Trainable':<15} {'Efficiency':<12}")
    print("-" * 54)
    
    for name, metrics in methods:
        if metrics.get('success', False):
            total = metrics.get('total_parameters', 0)
            trainable = metrics.get('trainable_parameters', 0)
            efficiency = metrics.get('parameter_efficiency', 0)
            
            print(f"{name:<12} {total:>14,} {trainable:>14,} {efficiency:>11.2%}")
    
    # Timing Comparison
    print(f"\nTIMING COMPARISON:")
    print("-" * 50)
    print(f"{'Method':<12} {'Total Time':<12} {'Avg Epoch':<12} {'Batch Time':<12}")
    print("-" * 48)
    
    for name, metrics in methods:
        if metrics.get('success', False):
            total_time = metrics.get('training_time', metrics.get('wall_clock_time', 0))
            epoch_time = metrics.get('avg_epoch_time', 0)
            batch_time = metrics.get('avg_batch_time', 0)
            
            print(f"{name:<12} {total_time:>9.1f}s {epoch_time:>9.1f}s {batch_time:>9.1f}ms")
    
    # Memory Comparison
    print(f"\nMEMORY USAGE COMPARISON:")
    print("-" * 50)
    print(f"{'Method':<12} {'Max GPU Memory':<15}")
    print("-" * 27)
    
    for name, metrics in methods:
        if metrics.get('success', False):
            max_memory = metrics.get('max_gpu_memory', 0)
            print(f"{name:<12} {max_memory:>12.2f} GB")
    
    # Accuracy Comparison
    print(f"\nACCURACY COMPARISON:")
    print("-" * 50)
    print(f"{'Method':<12} {'Best Accuracy':<15}")
    print("-" * 27)
    
    baseline_acc = baseline_metrics.get('best_accuracy', 0) if baseline_metrics.get('success') else 0
    
    for name, metrics in methods:
        if metrics.get('success', False):
            accuracy = metrics.get('best_accuracy', 0)
            if baseline_acc > 0 and name != "Baseline":
                diff = accuracy - baseline_acc
                print(f"{name:<12} {accuracy:>12.2f}% ({diff:+.2f}%)")
            else:
                print(f"{name:<12} {accuracy:>12.2f}%")
    
    # SparseLoRA Specific Analysis
    if sparselora_metrics.get('success', False):
        print(f"\nSPARSELORA ANALYSIS:")
        print("-" * 50)
        
        sparsity = sparselora_metrics.get('final_sparsity', 0)
        active_params = sparselora_metrics.get('active_lora_parameters', 0)
        effective_efficiency = sparselora_metrics.get('effective_parameter_efficiency', 0)
        
        print(f"Final sparsity: {sparsity:.1%}")
        print(f"Active LoRA parameters: {active_params:,}")
        print(f"Effective parameter efficiency: {effective_efficiency:.4%}")
        
        # Compare with LoRA
        if lora_metrics.get('success', False):
            lora_trainable = lora_metrics.get('trainable_parameters', 0)
            if active_params > 0:
                param_reduction = (lora_trainable - active_params) / lora_trainable
                print(f"Parameter reduction vs LoRA: {param_reduction:.1%}")
    
    # Efficiency Summary
    print(f"\nEFFICIENCY SUMMARY:")
    print("-" * 50)
    
    if all(m.get('success', False) for m in [baseline_metrics, lora_metrics, sparselora_metrics]):
        baseline_time = baseline_metrics.get('training_time', baseline_metrics.get('wall_clock_time', 0))
        lora_time = lora_metrics.get('training_time', lora_metrics.get('wall_clock_time', 0))
        sparselora_time = sparselora_metrics.get('training_time', sparselora_metrics.get('wall_clock_time', 0))
        
        if all(t > 0 for t in [baseline_time, lora_time, sparselora_time]):
            lora_speedup = baseline_time / lora_time
            sparselora_speedup = baseline_time / sparselora_time
            
            print(f"LoRA speedup vs Baseline: {lora_speedup:.2f}x")
            print(f"SparseLoRA speedup vs Baseline: {sparselora_speedup:.2f}x")
            print(f"SparseLoRA vs LoRA: {lora_time / sparselora_time:.2f}x")
        
        baseline_mem = baseline_metrics.get('max_gpu_memory', 0)
        lora_mem = lora_metrics.get('max_gpu_memory', 0)
        sparselora_mem = sparselora_metrics.get('max_gpu_memory', 0)
        
        if all(m > 0 for m in [baseline_mem, lora_mem, sparselora_mem]):
            print(f"Memory reduction - LoRA: {(1 - lora_mem/baseline_mem):.1%}")
            print(f"Memory reduction - SparseLoRA: {(1 - sparselora_mem/baseline_mem):.1%}")

def main():
    parser = argparse.ArgumentParser(description="Compare ViT training methods")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline training")
    parser.add_argument("--skip-lora", action="store_true", help="Skip LoRA training") 
    parser.add_argument("--skip-sparselora", action="store_true", help="Skip SparseLoRA training")
    parser.add_argument("--output-dir", default="./comparison_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training scripts
    baseline_metrics = {}
    lora_metrics = {}
    sparselora_metrics = {}
    
    if not args.skip_baseline:
        print("Training Baseline ViT...")
        baseline_metrics, baseline_stdout, baseline_stderr = run_training_script(
            "train_vit_cifar100.py", "Baseline ViT"
        )
        
        # Save logs
        with open(os.path.join(args.output_dir, 'baseline_stdout.log'), 'w') as f:
            f.write(baseline_stdout)
        with open(os.path.join(args.output_dir, 'baseline_stderr.log'), 'w') as f:
            f.write(baseline_stderr)
    
    if not args.skip_lora:
        print("Training LoRA ViT...")
        lora_metrics, lora_stdout, lora_stderr = run_training_script(
            "train_vit_lora_cifar100.py", "LoRA ViT"
        )
        
        # Save logs
        with open(os.path.join(args.output_dir, 'lora_stdout.log'), 'w') as f:
            f.write(lora_stdout)
        with open(os.path.join(args.output_dir, 'lora_stderr.log'), 'w') as f:
            f.write(lora_stderr)
    
    if not args.skip_sparselora:
        print("Training SparseLoRA ViT...")
        sparselora_metrics, sparselora_stdout, sparselora_stderr = run_training_script(
            "train_vit_sparselora_cifar100.py", "SparseLoRA ViT"
        )
        
        # Save logs
        with open(os.path.join(args.output_dir, 'sparselora_stdout.log'), 'w') as f:
            f.write(sparselora_stdout)
        with open(os.path.join(args.output_dir, 'sparselora_stderr.log'), 'w') as f:
            f.write(sparselora_stderr)
    
    # Compare results
    if any([baseline_metrics, lora_metrics, sparselora_metrics]):
        compare_results(baseline_metrics, lora_metrics, sparselora_metrics)
        
        # Save results to JSON
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'baseline': baseline_metrics,
            'lora': lora_metrics,
            'sparselora': sparselora_metrics,
        }
        
        output_file = os.path.join(args.output_dir, 'comparison_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo training was performed.")

if __name__ == "__main__":
    main()