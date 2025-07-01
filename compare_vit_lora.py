import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
from peft import LoraConfig, get_peft_model
import time
import psutil
import os
import gc

# Memory and Performance Comparison: ViT vs ViT+LoRA

# Hyperparameters
BATCH_SIZE = 16  # Reduced for memory
TEST_EPOCHS = 2  # Just test for 2 epochs for comparison
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LoRA hyperparameters
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3  # GB
    else:
        gpu_memory = 0
        gpu_memory_max = 0
    
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024**3  # GB
    
    return {
        'gpu_memory': gpu_memory,
        'gpu_memory_max': gpu_memory_max,
        'cpu_memory': cpu_memory
    }

def create_dataset():
    """Create CIFAR-100 dataset"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    return trainloader

def create_regular_vit():
    """Create regular ViT model"""
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100)
    return model

def create_lora_vit():
    """Create LoRA ViT model"""
    base_model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100)
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["qkv", "proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none"
    )
    
    model = get_peft_model(base_model, lora_config)
    return model

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def benchmark_model(model, model_name, trainloader):
    """Benchmark a model for memory and time"""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*60}")
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    # Memory before training
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    initial_memory = get_memory_usage()
    print(f"Initial GPU memory: {initial_memory['gpu_memory']:.2f} GB")
    
    # Training benchmarks
    epoch_times = []
    batch_times = []
    step_times = []
    
    model.train()
    
    for epoch in range(TEST_EPOCHS):
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_start = time.time()
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass timing
            step_start = time.time()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            step_end = time.time()
            
            batch_end = time.time()
            
            batch_times.append(batch_end - batch_start)
            step_times.append(step_end - step_start)
            
            # Only test first 20 batches per epoch for speed
            if batch_idx >= 19:
                break
        
        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)
        
        # Memory during training
        current_memory = get_memory_usage()
        print(f"Epoch {epoch+1}: {epoch_times[-1]:.2f}s, GPU memory: {current_memory['gpu_memory']:.2f} GB")
    
    # Final memory usage
    final_memory = get_memory_usage()
    
    # Calculate statistics
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_step_time = sum(step_times) / len(step_times)
    
    results = {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': 100 * trainable_params / total_params,
        'initial_gpu_memory': initial_memory['gpu_memory'],
        'max_gpu_memory': final_memory['gpu_memory_max'],
        'final_gpu_memory': final_memory['gpu_memory'],
        'avg_epoch_time': avg_epoch_time,
        'avg_batch_time': avg_batch_time,
        'avg_step_time': avg_step_time,
    }
    
    print(f"\nRESULTS SUMMARY:")
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    print(f"Average batch time: {avg_batch_time*1000:.1f} ms")
    print(f"Average step time: {avg_step_time*1000:.1f} ms")
    print(f"Max GPU memory: {final_memory['gpu_memory_max']:.2f} GB")
    
    return results

def main():
    print("ViT vs ViT+LoRA Performance Comparison")
    print("="*60)
    
    # Create dataset
    print("Loading CIFAR-100 dataset...")
    trainloader = create_dataset()
    
    # Test Regular ViT
    print("\nCreating Regular ViT model...")
    regular_vit = create_regular_vit()
    regular_results = benchmark_model(regular_vit, "Regular ViT", trainloader)
    
    # Clear memory
    del regular_vit
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Test LoRA ViT
    print("\nCreating LoRA ViT model...")
    lora_vit = create_lora_vit()
    lora_vit.print_trainable_parameters()
    lora_results = benchmark_model(lora_vit, "LoRA ViT", trainloader)
    
    # Final Comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Metric':<25} {'Regular ViT':<15} {'LoRA ViT':<15} {'Improvement':<15}")
    print("-" * 80)
    
    # Parameters
    print(f"{'Total Parameters':<25} {regular_results['total_params']:,<15} {lora_results['total_params']:,<15} {'Same':<15}")
    print(f"{'Trainable Parameters':<25} {regular_results['trainable_params']:,<15} {lora_results['trainable_params']:,<15} {regular_results['trainable_params']/lora_results['trainable_params']:.1f}x less")
    print(f"{'Trainable Ratio':<25} {regular_results['trainable_ratio']:.1f}%{'':<15} {lora_results['trainable_ratio']:.1f}%{'':<15} {regular_results['trainable_ratio']/lora_results['trainable_ratio']:.1f}x less")
    
    # Memory
    print(f"{'Max GPU Memory (GB)':<25} {regular_results['max_gpu_memory']:.2f}{'':<15} {lora_results['max_gpu_memory']:.2f}{'':<15} {regular_results['max_gpu_memory']/lora_results['max_gpu_memory']:.2f}x less")
    
    # Time
    faster_epoch = 'faster' if lora_results['avg_epoch_time'] < regular_results['avg_epoch_time'] else 'slower'
    faster_batch = 'faster' if lora_results['avg_batch_time'] < regular_results['avg_batch_time'] else 'slower'
    faster_step = 'faster' if lora_results['avg_step_time'] < regular_results['avg_step_time'] else 'slower'
    
    print(f"{'Avg Epoch Time (s)':<25} {regular_results['avg_epoch_time']:.2f}{'':<15} {lora_results['avg_epoch_time']:.2f}{'':<15} {regular_results['avg_epoch_time']/lora_results['avg_epoch_time']:.2f}x {faster_epoch}")
    print(f"{'Avg Batch Time (ms)':<25} {regular_results['avg_batch_time']*1000:.1f}{'':<15} {lora_results['avg_batch_time']*1000:.1f}{'':<15} {regular_results['avg_batch_time']/lora_results['avg_batch_time']:.2f}x {faster_batch}")
    print(f"{'Avg Step Time (ms)':<25} {regular_results['avg_step_time']*1000:.1f}{'':<15} {lora_results['avg_step_time']*1000:.1f}{'':<15} {regular_results['avg_step_time']/lora_results['avg_step_time']:.2f}x {faster_step}")
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"LoRA reduces trainable parameters by {regular_results['trainable_params']/lora_results['trainable_params']:.1f}x")
    print(f"LoRA reduces GPU memory usage by {regular_results['max_gpu_memory']/lora_results['max_gpu_memory']:.2f}x")
    if lora_results['avg_epoch_time'] < regular_results['avg_epoch_time']:
        print(f"LoRA is {regular_results['avg_epoch_time']/lora_results['avg_epoch_time']:.2f}x faster per epoch")
    else:
        print(f"Regular ViT is {lora_results['avg_epoch_time']/regular_results['avg_epoch_time']:.2f}x faster per epoch")

if __name__ == "__main__":
    main()