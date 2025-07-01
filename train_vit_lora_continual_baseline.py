import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm
from peft import LoraConfig, get_peft_model
import numpy as np
from collections import defaultdict
import time
import psutil
import os

# Baseline Continual Learning with ViT + LoRA on Split-CIFAR100 (No Regularization)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS_PER_TASK = 10
LEARNING_RATE = 0.001
NUM_TASKS = 10  # Split CIFAR-100 into 10 tasks (10 classes each)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LoRA hyperparameters
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Latency tracking variables
task_times = []
epoch_times = []
batch_times = []
step_times = []

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
    """Create Split-CIFAR100 datasets (10 tasks, 10 classes each)"""
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

def create_lora_model():
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

def train_task(model, train_loader, task_id):
    """Train LoRA model on a single task (baseline - no regularization)"""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining Task {task_id} with LoRA (Baseline)")
    print("-" * 40)
    
    task_start = time.time()
    
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
            
            if batch_idx % 50 == 0:
                avg_batch_time = sum(batch_times[-50:]) / min(50, len(batch_times))
                avg_step_time = sum(step_times[-50:]) / min(50, len(step_times))
                print(f'Epoch {epoch+1}, Batch {batch_idx}: Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%, Batch: {avg_batch_time*1000:.1f}ms, Step: {avg_step_time*1000:.1f}ms')
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        
        epoch_acc = 100. * correct / total
        print(f'Task {task_id}, Epoch {epoch+1}: {epoch_time:.2f}s, Loss: {epoch_loss/len(train_loader):.4f}, '
              f'Acc: {epoch_acc:.2f}%')
    
    task_end = time.time()
    task_time = task_end - task_start
    task_times.append(task_time)
    
    return task_time

def test_all_tasks(model, all_test_loaders, current_task_id):
    """Test model on all seen tasks so far"""
    model.eval()
    task_accuracies = []
    
    print(f"\nTesting after Task {current_task_id}:")
    print("-" * 50)
    
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
            print(f'Task {task_id} Accuracy: {accuracy:.2f}%')
    
    avg_accuracy = sum(task_accuracies) / len(task_accuracies)
    print(f'Average Accuracy: {avg_accuracy:.2f}%')
    
    return task_accuracies, avg_accuracy

def main():
    print("Baseline Continual Learning with ViT + LoRA on Split-CIFAR100")
    print("=" * 60)
    print("NO REGULARIZATION - Pure Sequential Learning")
    print(f"Number of tasks: {NUM_TASKS}")
    print(f"Classes per task: 10")
    print(f"Epochs per task: {EPOCHS_PER_TASK}")
    print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print("=" * 60)
    
    # Create LoRA model
    model = create_lora_model()
    model = model.to(DEVICE)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Create split datasets
    tasks_train, tasks_test = create_split_cifar100()
    
    # Store results
    all_results = []
    
    # Train on each task sequentially
    for task_id in range(NUM_TASKS):
        print(f"\n{'='*60}")
        print(f"LEARNING TASK {task_id}")
        print(f"{'='*60}")
        
        # Train on current task only
        train_task(model, tasks_train[task_id], task_id)
        
        # Test on all tasks seen so far
        task_accs, avg_acc = test_all_tasks(model, tasks_test, task_id)
        all_results.append({
            'task_id': task_id,
            'task_accuracies': task_accs,
            'average_accuracy': avg_acc
        })
    
    # Final results summary
    print(f"\n{'='*80}")
    print("FINAL BASELINE CONTINUAL LEARNING RESULTS (LoRA)")
    print(f"{'='*80}")
    
    print(f"{'Task':<6} {'Avg Acc':<10} {'Individual Task Accuracies'}")
    print("-" * 60)
    
    for result in all_results:
        task_id = result['task_id']
        avg_acc = result['average_accuracy']
        task_accs = result['task_accuracies']
        
        accs_str = " ".join([f"{acc:.1f}" for acc in task_accs])
        print(f"{task_id:<6} {avg_acc:<10.2f} {accs_str}")
    
    # Compute final metrics
    final_avg_acc = all_results[-1]['average_accuracy']
    print(f"\nFinal Average Accuracy: {final_avg_acc:.2f}%")
    
    # Catastrophic forgetting analysis
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("CATASTROPHIC FORGETTING ANALYSIS (LoRA)")
        print(f"{'='*60}")
        
        initial_accs = []
        final_accs = []
        
        for task_id in range(NUM_TASKS - 1):
            # Get accuracy on this task right after learning it
            initial_acc = all_results[task_id]['task_accuracies'][task_id]
            # Get accuracy on this task after learning all tasks
            final_acc = all_results[-1]['task_accuracies'][task_id]
            
            initial_accs.append(initial_acc)
            final_accs.append(final_acc)
            
            forgetting = initial_acc - final_acc
            print(f"Task {task_id}: {initial_acc:.1f}% → {final_acc:.1f}% (Forgot: {forgetting:.1f}%)")
        
        avg_initial = sum(initial_accs) / len(initial_accs)
        avg_final = sum(final_accs) / len(final_accs)
        avg_forgetting = avg_initial - avg_final
        
        print(f"\nAverage Initial Accuracy: {avg_initial:.2f}%")
        print(f"Average Final Accuracy: {avg_final:.2f}%")
        print(f"Average Forgetting: {avg_forgetting:.2f}%")
        
        # Show LoRA forgetting effect
        task_0_final = all_results[-1]['task_accuracies'][0]
        task_0_initial = all_results[0]['task_accuracies'][0]
        print(f"\nTask 0 Performance Drop: {task_0_initial:.1f}% → {task_0_final:.1f}% "
              f"(Lost {task_0_initial - task_0_final:.1f}%)")
        
        print(f"\nLoRA Specific Analysis:")
        print(f"- Trained only {model.num_parameters(only_trainable=True):,} parameters per task")
        print(f"- Base model kept {model.num_parameters() - model.num_parameters(only_trainable=True):,} parameters frozen")
    
    # Performance summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY (LoRA)")
    print(f"{'='*80}")
    
    total_time = sum(task_times)
    avg_task_time = total_time / len(task_times)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    
    print(f'Total continual learning time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)')
    print(f'Average task time: {avg_task_time:.2f} seconds')
    print(f'Average epoch time: {avg_epoch_time:.2f} seconds')
    print(f'Average batch time: {avg_batch_time*1000:.1f} ms')
    print(f'Average step time: {avg_step_time*1000:.1f} ms')
    
    # Memory summary
    final_memory = get_memory_usage()
    print(f'Final GPU memory: {final_memory["gpu_memory"]:.2f} GB')
    print(f'Max GPU memory used: {final_memory["gpu_memory_max"]:.2f} GB')
    print(f'CPU memory: {final_memory["cpu_memory"]:.2f} GB')
    
    # LoRA efficiency metrics
    print(f'\nLoRA Efficiency:')
    print(f'Trainable parameters: {model.num_parameters(only_trainable=True):,}')
    print(f'Total parameters: {model.num_parameters():,}')
    print(f'Parameter efficiency: {model.num_parameters() / model.num_parameters(only_trainable=True):.1f}x reduction')
    
    # Task timing breakdown
    print(f"\nTask timing breakdown:")
    for i, task_time in enumerate(task_times):
        print(f"Task {i}: {task_time:.2f}s ({task_time/60:.1f}min)")
    
    print("\nBaseline Continual Learning with LoRA completed!")
    print("This shows how LoRA handles catastrophic forgetting without any mitigation.")

if __name__ == "__main__":
    main()