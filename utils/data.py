import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from typing import Tuple, Optional
import numpy as np

class DatasetFactory:
    """Factory for creating datasets"""
    
    @staticmethod
    def get_cifar10(
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224
    ) -> Tuple[DataLoader, DataLoader, int]:
        """Get CIFAR-10 dataset"""
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=val_transform
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, 10  # 10 classes
    
    @staticmethod
    def get_cifar100(
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224
    ) -> Tuple[DataLoader, DataLoader, int]:
        """Get CIFAR-100 dataset"""
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Datasets
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        val_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=val_transform
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, 100  # 100 classes
    
    @staticmethod
    def get_imagenet_subset(
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        subset_ratio: float = 0.1
    ) -> Tuple[DataLoader, DataLoader, int]:
        """Get ImageNet subset using HuggingFace datasets"""
        
        # Load ImageNet dataset
        dataset = load_dataset("imagenet-1k", split="train", streaming=True)
        val_dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
        
        # Create subset
        train_size = int(1281167 * subset_ratio)  # ImageNet train size
        val_size = int(50000 * subset_ratio)  # ImageNet val size
        
        train_subset = dataset.take(train_size)
        val_subset = val_dataset.take(val_size)
        
        # Custom dataset class for HuggingFace datasets
        class HFImageDataset(Dataset):
            def __init__(self, hf_dataset, transform=None):
                self.data = list(hf_dataset)
                self.transform = transform
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                image = item['image'].convert('RGB')
                label = item['label']
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = HFImageDataset(train_subset, train_transform)
        val_dataset = HFImageDataset(val_subset, val_transform)
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, 1000  # 1000 classes

def get_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    **kwargs
) -> Tuple[DataLoader, DataLoader, int]:
    """Get dataset by name"""
    
    factory = DatasetFactory()
    
    if dataset_name.lower() == "cifar10":
        return factory.get_cifar10(data_dir, batch_size, num_workers, image_size)
    elif dataset_name.lower() == "cifar100":
        return factory.get_cifar100(data_dir, batch_size, num_workers, image_size)
    elif dataset_name.lower() == "imagenet":
        subset_ratio = kwargs.get('subset_ratio', 0.1)
        return factory.get_imagenet_subset(
            data_dir, batch_size, num_workers, image_size, subset_ratio
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def calculate_dataset_stats(dataloader: DataLoader) -> dict:
    """Calculate dataset statistics"""
    
    total_samples = 0
    total_batches = len(dataloader)
    
    # Calculate mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    print("Calculating dataset statistics...")
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_samples': total_samples,
        'total_batches': total_batches
    }
