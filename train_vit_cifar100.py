import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
import time
import psutil
import os

# Simple CIFAR-100 ViT Training Script

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Latency tracking variables
batch_times = []
epoch_times = []
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

# Data transforms
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

# Load CIFAR-100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load ViT model
model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

# Training function
def train():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            avg_batch_time = sum(batch_times[-100:]) / min(100, len(batch_times))
            avg_step_time = sum(step_times[-100:]) / min(100, len(step_times))
            print(f'Batch {batch_idx}: Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, '
                  f'Batch Time: {avg_batch_time*1000:.1f}ms, Step Time: {avg_step_time*1000:.1f}ms')
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    test_start = time.time()
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_end = time.time()
    test_time = test_end - test_start
    
    test_loss /= len(testloader)
    test_acc = 100. * correct / total
    return test_loss, test_acc, test_time

# Main training loop
print(f"Training ViT on CIFAR-100 using {DEVICE}")
print(f"Model: {model.__class__.__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Initial memory usage
initial_memory = get_memory_usage()
print(f"Initial GPU memory: {initial_memory['gpu_memory']:.2f} GB")

# Training start time
training_start = time.time()

best_acc = 0
for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch+1}/{EPOCHS}')
    print('-' * 50)
    
    # Time epoch
    epoch_start = time.time()
    
    # Train
    train_loss, train_acc = train()
    
    # Test
    test_loss, test_acc, test_time = test()
    
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    epoch_times.append(epoch_time)
    
    # Current memory usage
    current_memory = get_memory_usage()
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    print(f'Epoch Time: {epoch_time:.2f}s, Test Time: {test_time:.2f}s')
    print(f'GPU Memory: {current_memory["gpu_memory"]:.2f} GB')
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_vit_cifar100.pth')
        print(f'New best accuracy: {best_acc:.2f}% - Model saved!')

training_end = time.time()
total_training_time = training_end - training_start

# Final memory usage
final_memory = get_memory_usage()

# Performance summary
print(f'\n{"="*60}')
print("TRAINING PERFORMANCE SUMMARY")
print(f'{"="*60}')
print(f'Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.1f} minutes)')
print(f'Average epoch time: {sum(epoch_times)/len(epoch_times):.2f} seconds')
print(f'Average batch time: {sum(batch_times)/len(batch_times)*1000:.1f} ms')
print(f'Average step time: {sum(step_times)/len(step_times)*1000:.1f} ms')
print(f'Max GPU memory used: {final_memory["gpu_memory_max"]:.2f} GB')
print(f'Final GPU memory: {final_memory["gpu_memory"]:.2f} GB')
print(f'Best test accuracy: {best_acc:.2f}%')
print(f'Training throughput: {len(trainloader)*EPOCHS/(total_training_time/3600):.1f} batches/hour')