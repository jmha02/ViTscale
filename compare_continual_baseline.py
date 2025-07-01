import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm
from peft import LoraConfig, get_peft_model
import time
import psutil
import os
import gc
from collections import defaultdict

# Comparison: Baseline Continual Learning ViT vs ViT+LoRA

# Hyperparameters
BATCH_SIZE = 32
EPOCHS_PER_TASK = 3  # Reduced for comparison speed
LEARNING_RATE = 0.001
NUM_TASKS = 3  # Test on first 3 tasks for speed
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

def create_split_cifar100():
    """Create Split-CIFAR100 datasets (reduced for comparison)"""
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load full CIFAR-100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    # Split into tasks
    tasks_train = []
    tasks_test = []
    
    for task_id in range(NUM_TASKS):
        # Classes for this task (10 classes per task)
        start_class = task_id * 10
        end_class = (task_id + 1) * 10
        task_classes = list(range(start_class, end_class))
        
        # Filter training data for this task
        train_indices = [i for i, (_, label) in enumerate(trainset) if label in task_classes]
        test_indices = [i for i, (_, label) in enumerate(testset) if label in task_classes]
        
        # Create task-specific datasets
        task_trainset = Subset(trainset, train_indices)
        task_testset = Subset(testset, test_indices)
        
        # Create dataloaders
        train_loader = DataLoader(task_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(task_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        tasks_train.append(train_loader)
        tasks_test.append(test_loader)
        
        print(f"Task {task_id}: Classes {start_class}-{end_class-1}, "
              f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    return tasks_train, tasks_test

def create_regular_vit():
    """Create regular ViT model"""
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100)
    return model

def create_lora_vit():
    """Create ViT model with LoRA"""
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

def train_task_with_timing(model, train_loader, task_id, model_name):
    """Train model on a single task with detailed timing"""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining Task {task_id} - {model_name}")
    print("-" * 40)
    
    task_start_time = time.time()
    epoch_times = []
    batch_times = []
    step_times = []
    
    for epoch in range(EPOCHS_PER_TASK):
        epoch_start = time.time()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start = time.time()
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Time forward+backward pass
            step_start = time.time()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            step_end = time.time()
            
            batch_end = time.time()
            
            # Record timings
            batch_times.append(batch_end - batch_start)
            step_times.append(step_end - step_start)
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Only train on first 20 batches for speed
            if batch_idx >= 19:
                break
        
        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)
        
        epoch_acc = 100. * correct / total
        print(f'Task {task_id}, Epoch {epoch+1}: {epoch_times[-1]:.2f}s, '
              f'Loss: {epoch_loss/min(20, len(train_loader)):.4f}, Acc: {epoch_acc:.2f}%')
    
    task_end_time = time.time()
    total_task_time = task_end_time - task_start_time
    
    return {
        'total_task_time': total_task_time,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'avg_batch_time': sum(batch_times) / len(batch_times),
        'avg_step_time': sum(step_times) / len(step_times),
        'epoch_times': epoch_times,
        'batch_times': batch_times,
        'step_times': step_times
    }

def test_all_tasks_with_timing(model, all_test_loaders, current_task_id, model_name):
    """Test model on all seen tasks with timing"""
    model.eval()
    
    test_start = time.time()
    task_accuracies = []
    
    with torch.no_grad():
        for task_id in range(current_task_id + 1):
            test_loader = all_test_loaders[task_id]
            correct = 0
            total = 0
            
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            accuracy = 100. * correct / total
            task_accuracies.append(accuracy)
    
    test_end = time.time()
    total_test_time = test_end - test_start
    avg_test_time = total_test_time / (current_task_id + 1)
    
    return task_accuracies, total_test_time, avg_test_time

def benchmark_continual_learning(model, model_name, tasks_train, tasks_test):
    """Benchmark continual learning approach"""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*60}")
    
    model = model.to(DEVICE)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    
    # Memory before training
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    initial_memory = get_memory_usage()
    
    # Store all timing results
    all_training_times = []
    all_testing_times = []
    all_task_results = []
    
    total_continual_start = time.time()
    
    # Train on each task sequentially
    for task_id in range(NUM_TASKS):
        print(f"\n{'='*50}")
        print(f"TASK {task_id} - {model_name}")
        print(f"{'='*50}")
        
        # Train on current task
        train_timing = train_task_with_timing(model, tasks_train[task_id], task_id, model_name)
        all_training_times.append(train_timing)
        
        # Test on all tasks seen so far
        task_accs, test_time, avg_test_time = test_all_tasks_with_timing(model, tasks_test, task_id, model_name)
        all_testing_times.append({'total_test_time': test_time, 'avg_test_time': avg_test_time})
        
        avg_acc = sum(task_accs) / len(task_accs)
        all_task_results.append({
            'task_id': task_id,
            'task_accuracies': task_accs,
            'average_accuracy': avg_acc
        })
        
        print(f"Task {task_id} completed in {train_timing['total_task_time']:.2f}s")
        print(f"Test accuracies: {[f'{acc:.1f}' for acc in task_accs]}")
        print(f"Average accuracy: {avg_acc:.2f}%")
    
    total_continual_end = time.time()
    total_continual_time = total_continual_end - total_continual_start
    
    # Final memory usage
    final_memory = get_memory_usage()
    
    # Compute overall statistics
    total_training_time = sum([t['total_task_time'] for t in all_training_times])
    total_testing_time = sum([t['total_test_time'] for t in all_testing_times])
    
    avg_task_time = total_training_time / NUM_TASKS
    avg_epoch_time = sum([t['avg_epoch_time'] for t in all_training_times]) / len(all_training_times)
    avg_batch_time = sum([t['avg_batch_time'] for t in all_training_times]) / len(all_training_times)
    avg_step_time = sum([t['avg_step_time'] for t in all_training_times]) / len(all_training_times)
    
    results = {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': 100 * trainable_params / total_params,
        'total_continual_time': total_continual_time,
        'total_training_time': total_training_time,
        'total_testing_time': total_testing_time,
        'avg_task_time': avg_task_time,
        'avg_epoch_time': avg_epoch_time,
        'avg_batch_time': avg_batch_time,
        'avg_step_time': avg_step_time,
        'initial_gpu_memory': initial_memory['gpu_memory'],
        'max_gpu_memory': final_memory['gpu_memory_max'],
        'final_gpu_memory': final_memory['gpu_memory'],
        'task_results': all_task_results,
        'training_times': all_training_times,
        'testing_times': all_testing_times
    }
    
    print(f"\n{model_name} SUMMARY:")
    print(f"Total continual learning time: {total_continual_time:.2f}s")
    print(f"Average task training time: {avg_task_time:.2f}s")
    print(f"Average epoch time: {avg_epoch_time:.2f}s")
    print(f"Average batch time: {avg_batch_time*1000:.1f}ms")
    print(f"Average step time: {avg_step_time*1000:.1f}ms")
    print(f"Max GPU memory: {final_memory['gpu_memory_max']:.2f}GB")
    
    return results

def main():
    print("Baseline Continual Learning Comparison: ViT vs ViT+LoRA")
    print("=" * 60)
    print(f"Testing {NUM_TASKS} tasks, {EPOCHS_PER_TASK} epochs each")
    print("=" * 60)
    
    # Create datasets
    tasks_train, tasks_test = create_split_cifar100()
    
    # Test Regular ViT
    print("\nCreating Regular ViT model...")
    regular_vit = create_regular_vit()
    regular_results = benchmark_continual_learning(regular_vit, "Regular ViT", tasks_train, tasks_test)
    
    # Clear memory
    del regular_vit
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Test LoRA ViT
    print("\nCreating LoRA ViT model...")
    lora_vit = create_lora_vit()
    lora_results = benchmark_continual_learning(lora_vit, "LoRA ViT", tasks_train, tasks_test)
    
    # Final Comparison
    print(f"\n{'='*80}")
    print("BASELINE CONTINUAL LEARNING COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Metric':<30} {'Regular ViT':<15} {'LoRA ViT':<15} {'Improvement':<15}")
    print("-" * 80)
    
    # Parameters
    print(f"{'Total Parameters':<30} {regular_results['total_params']:,<15} {lora_results['total_params']:,<15} {'Same':<15}")
    print(f"{'Trainable Parameters':<30} {regular_results['trainable_params']:,<15} {lora_results['trainable_params']:,<15} {regular_results['trainable_params']/lora_results['trainable_params']:.1f}x less")
    print(f"{'Trainable Ratio (%)':<30} {regular_results['trainable_ratio']:.1f}{'':<15} {lora_results['trainable_ratio']:.1f}{'':<15} {regular_results['trainable_ratio']/lora_results['trainable_ratio']:.1f}x less")
    
    # Memory
    print(f"{'Max GPU Memory (GB)':<30} {regular_results['max_gpu_memory']:.2f}{'':<15} {lora_results['max_gpu_memory']:.2f}{'':<15} {regular_results['max_gpu_memory']/lora_results['max_gpu_memory']:.2f}x")
    
    # Time Comparisons
    faster_continual = 'faster' if lora_results['total_continual_time'] < regular_results['total_continual_time'] else 'slower'
    faster_task = 'faster' if lora_results['avg_task_time'] < regular_results['avg_task_time'] else 'slower'
    faster_epoch = 'faster' if lora_results['avg_epoch_time'] < regular_results['avg_epoch_time'] else 'slower'
    faster_batch = 'faster' if lora_results['avg_batch_time'] < regular_results['avg_batch_time'] else 'slower'
    faster_step = 'faster' if lora_results['avg_step_time'] < regular_results['avg_step_time'] else 'slower'
    
    print(f"{'Total CL Time (s)':<30} {regular_results['total_continual_time']:.1f}{'':<15} {lora_results['total_continual_time']:.1f}{'':<15} {regular_results['total_continual_time']/lora_results['total_continual_time']:.2f}x {faster_continual}")
    print(f"{'Avg Task Time (s)':<30} {regular_results['avg_task_time']:.1f}{'':<15} {lora_results['avg_task_time']:.1f}{'':<15} {regular_results['avg_task_time']/lora_results['avg_task_time']:.2f}x {faster_task}")
    print(f"{'Avg Epoch Time (s)':<30} {regular_results['avg_epoch_time']:.2f}{'':<15} {lora_results['avg_epoch_time']:.2f}{'':<15} {regular_results['avg_epoch_time']/lora_results['avg_epoch_time']:.2f}x {faster_epoch}")
    print(f"{'Avg Batch Time (ms)':<30} {regular_results['avg_batch_time']*1000:.1f}{'':<15} {lora_results['avg_batch_time']*1000:.1f}{'':<15} {regular_results['avg_batch_time']/lora_results['avg_batch_time']:.2f}x {faster_batch}")
    print(f"{'Avg Step Time (ms)':<30} {regular_results['avg_step_time']*1000:.1f}{'':<15} {lora_results['avg_step_time']*1000:.1f}{'':<15} {regular_results['avg_step_time']/lora_results['avg_step_time']:.2f}x {faster_step}")
    
    # Accuracy Comparison
    print(f"\n{'='*60}")
    print("ACCURACY COMPARISON")
    print(f"{'='*60}")
    
    regular_final_acc = regular_results['task_results'][-1]['average_accuracy']
    lora_final_acc = lora_results['task_results'][-1]['average_accuracy']
    
    print(f"Regular ViT Final Avg Accuracy: {regular_final_acc:.2f}%")
    print(f"LoRA ViT Final Avg Accuracy: {lora_final_acc:.2f}%")
    print(f"Accuracy Difference: {abs(regular_final_acc - lora_final_acc):.2f}% ({'LoRA better' if lora_final_acc > regular_final_acc else 'Regular better'})")
    
    # Catastrophic Forgetting Analysis
    print(f"\n{'='*60}")
    print("CATASTROPHIC FORGETTING ANALYSIS")
    print(f"{'='*60}")
    
    for model_name, results in [("Regular ViT", regular_results), ("LoRA ViT", lora_results)]:
        if len(results['task_results']) > 1:
            task_0_initial = results['task_results'][0]['task_accuracies'][0]
            task_0_final = results['task_results'][-1]['task_accuracies'][0]
            forgetting = task_0_initial - task_0_final
            
            print(f"{model_name}:")
            print(f"  Task 0: {task_0_initial:.1f}% → {task_0_final:.1f}% (Forgot: {forgetting:.1f}%)")
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"LoRA reduces trainable parameters by {regular_results['trainable_params']/lora_results['trainable_params']:.1f}x")
    print(f"LoRA continual learning is {regular_results['total_continual_time']/lora_results['total_continual_time']:.2f}x {'faster' if lora_results['total_continual_time'] < regular_results['total_continual_time'] else 'slower'}")
    print(f"Memory usage is {regular_results['max_gpu_memory']/lora_results['max_gpu_memory']:.2f}x different")

if __name__ == "__main__":
    main()