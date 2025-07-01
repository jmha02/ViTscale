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
import copy

# Continual Learning with ViT + LoRA on Split-CIFAR100

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

# Continual Learning Strategy
REPLAY_BUFFER_SIZE = 500  # Examples per task to store for replay

class LoRAContinualLearner:
    def __init__(self, base_model):
        self.base_model = base_model
        self.lora_adapters = {}  # Store LoRA adapters for each task
        self.current_model = None
        self.previous_tasks_data = []  # Replay buffer
        self.task_accuracies = defaultdict(list)
        
    def create_task_adapter(self, task_id):
        """Create a new LoRA adapter for the task"""
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=["qkv", "proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none"
        )
        
        # Apply LoRA to base model
        model = get_peft_model(copy.deepcopy(self.base_model), lora_config)
        self.lora_adapters[task_id] = model
        self.current_model = model
        
        print(f"Created LoRA adapter for Task {task_id}")
        model.print_trainable_parameters()
        return model
    
    def merge_adapters(self, task_ids):
        """Merge multiple LoRA adapters for multi-task inference"""
        # For simplicity, we'll use the most recent adapter
        # In practice, you might want to ensemble or average adapters
        if task_ids:
            latest_task = max(task_ids)
            return self.lora_adapters[latest_task]
        return self.current_model
    
    def save_task_data(self, dataloader, task_id):
        """Save representative samples for replay"""
        samples_per_class = REPLAY_BUFFER_SIZE // 10  # 10 classes per task
        task_data = {'inputs': [], 'targets': []}
        class_counts = defaultdict(int)
        
        for inputs, targets in dataloader:
            for i, target in enumerate(targets):
                target_class = target.item()
                if class_counts[target_class] < samples_per_class:
                    task_data['inputs'].append(inputs[i])
                    task_data['targets'].append(targets[i])
                    class_counts[target_class] += 1
        
        self.previous_tasks_data.append({
            'task_id': task_id,
            'inputs': torch.stack(task_data['inputs']),
            'targets': torch.stack(task_data['targets'])
        })
        
        print(f"Saved {len(task_data['inputs'])} samples for task {task_id}")

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

def train_task(model, train_loader, learner, task_id):
    """Train LoRA adapter on a single task"""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining Task {task_id} with LoRA")
    print("-" * 40)
    
    for epoch in range(EPOCHS_PER_TASK):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Current task loss
            current_loss = criterion(outputs, targets)
            total_loss = current_loss
            
            # Add replay loss for previous tasks
            if learner.previous_tasks_data:
                replay_loss = 0
                for prev_task_data in learner.previous_tasks_data:
                    prev_inputs = prev_task_data['inputs'].to(DEVICE)
                    prev_targets = prev_task_data['targets'].to(DEVICE)
                    
                    # Random sample from replay buffer
                    replay_indices = torch.randperm(len(prev_inputs))[:min(32, len(prev_inputs))]
                    replay_inputs = prev_inputs[replay_indices]
                    replay_targets = prev_targets[replay_indices]
                    
                    replay_outputs = model(replay_inputs)
                    replay_loss += criterion(replay_outputs, replay_targets)
                
                total_loss += 0.5 * replay_loss  # Weight replay loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}: Loss: {total_loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_acc = 100. * correct / total
        print(f'Task {task_id}, Epoch {epoch+1}: Loss: {epoch_loss/len(train_loader):.4f}, '
              f'Acc: {epoch_acc:.2f}%')

def test_all_tasks(learner, all_test_loaders, current_task_id):
    """Test model on all seen tasks so far"""
    task_accuracies = []
    
    print(f"\nTesting after Task {current_task_id}:")
    print("-" * 50)
    
    with torch.no_grad():
        for task_id in range(current_task_id + 1):
            # Use the current model (with all learned adapters)
            model = learner.current_model
            model.eval()
            
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
    print("Continual Learning with ViT + LoRA on Split-CIFAR100")
    print("=" * 60)
    print(f"Number of tasks: {NUM_TASKS}")
    print(f"Classes per task: 10")
    print(f"Epochs per task: {EPOCHS_PER_TASK}")
    print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"Replay buffer size: {REPLAY_BUFFER_SIZE}")
    print("=" * 60)
    
    # Create base model
    base_model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100)
    base_model = base_model.to(DEVICE)
    
    # Create continual learner
    learner = LoRAContinualLearner(base_model)
    
    # Create split datasets
    tasks_train, tasks_test = create_split_cifar100()
    
    # Store results
    all_results = []
    
    # Train on each task sequentially
    for task_id in range(NUM_TASKS):
        print(f"\n{'='*60}")
        print(f"LEARNING TASK {task_id}")
        print(f"{'='*60}")
        
        # Create LoRA adapter for this task
        model = learner.create_task_adapter(task_id)
        model = model.to(DEVICE)
        
        # Train on current task
        train_task(model, tasks_train[task_id], learner, task_id)
        
        # Save data for replay
        learner.save_task_data(tasks_train[task_id], task_id)
        
        # Test on all tasks seen so far
        task_accs, avg_acc = test_all_tasks(learner, tasks_test, task_id)
        all_results.append({
            'task_id': task_id,
            'task_accuracies': task_accs,
            'average_accuracy': avg_acc
        })
    
    # Final results summary
    print(f"\n{'='*80}")
    print("FINAL CONTINUAL LEARNING RESULTS (LoRA)")
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
    
    # Backward transfer (forgetting)
    if len(all_results) > 1:
        initial_accs = []
        final_accs = []
        
        for task_id in range(NUM_TASKS - 1):
            # Get accuracy on this task right after learning it
            initial_acc = all_results[task_id]['task_accuracies'][task_id]
            # Get accuracy on this task after learning all tasks
            final_acc = all_results[-1]['task_accuracies'][task_id]
            
            initial_accs.append(initial_acc)
            final_accs.append(final_acc)
        
        avg_forgetting = sum(initial_accs) / len(initial_accs) - sum(final_accs) / len(final_accs)
        print(f"Average Forgetting: {avg_forgetting:.2f}%")
    
    # LoRA specific metrics
    total_adapters = len(learner.lora_adapters)
    print(f"\nLoRA Metrics:")
    print(f"Total LoRA adapters created: {total_adapters}")
    print(f"Memory efficient: Each adapter trains only ~1% of parameters")
    
    print("\nContinual Learning with LoRA completed!")

if __name__ == "__main__":
    main()