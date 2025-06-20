import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import numpy as np
from tqdm import tqdm
import time

class Trainer:
    """Training utility class"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            if hasattr(output, 'logits'):
                output = output.logits
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(accuracy)
        self.history['epoch_times'].append(epoch_time)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                if hasattr(output, 'logits'):
                    output = output.logits
                
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        log_interval: int = 100,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs"""
        
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch, log_interval)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            print(f'Epoch {epoch}:')
            print(f'  Train Loss: {train_metrics["loss"]:.6f}, Train Acc: {train_metrics["accuracy"]:.2f}%')
            print(f'  Val Loss: {val_metrics["loss"]:.6f}, Val Acc: {val_metrics["accuracy"]:.2f}%')
            print(f'  Epoch Time: {train_metrics["epoch_time"]:.2f}s')
            
            # Save best model
            if save_path and val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(self.model.state_dict(), save_path)
                print(f'  Saved best model with val acc: {best_val_acc:.2f}%')
            
            print('-' * 60)
        
        return self.history

def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> optim.Optimizer:
    """Create optimizer"""
    
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    epochs: int = 10,
    **kwargs
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler"""
    
    if scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            **kwargs
        )
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name.lower() == 'multistep':
        milestones = kwargs.get('milestones', [epochs // 2, epochs * 3 // 4])
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
    elif scheduler_name.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """Computes the accuracy over the k top predictions"""
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res
