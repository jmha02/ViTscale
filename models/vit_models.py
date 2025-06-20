import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional
import math

class SimpleLoRALayer(nn.Module):
    """Simple LoRA layer implementation"""
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA matrices
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.dropout(self.lora_A(x))) * (self.alpha / self.rank)
        return original_output + lora_output

class ViTWithSimpleLoRA(nn.Module):
    """ViT with Simple LoRA implementation"""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 1000,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None
    ):
        super().__init__()
        
        # Load base model
        self.base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
        
        # Apply LoRA to target modules
        if target_modules is None:
            target_modules = ["qkv", "proj"]
            
        self.apply_lora(target_modules, lora_rank, lora_alpha, lora_dropout)
        
    def apply_lora(self, target_modules, rank, alpha, dropout):
        """Apply LoRA to specified modules"""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this module should be replaced
                for target in target_modules:
                    if target in name:
                        # Replace with LoRA layer
                        parent = self.base_model
                        module_path = name.split('.')
                        
                        # Navigate to parent
                        for path_part in module_path[:-1]:
                            parent = getattr(parent, path_part)
                        
                        # Replace the module
                        lora_layer = SimpleLoRALayer(module, rank, alpha, dropout)
                        setattr(parent, module_path[-1], lora_layer)
                        break
                        
    def forward(self, x):
        return self.base_model(x)
        
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for param in self.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        return {
            'trainable_params': trainable_params,
            'all_params': all_params,
            'trainable_percentage': 100.0 * trainable_params / all_params if all_params > 0 else 0.0
        }

class BaselineViT(nn.Module):
    """Baseline ViT model without LoRA"""
    
    def __init__(self, model_name: str, num_classes: int = 1000):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.model(x)
        
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable_params': trainable_params,
            'all_params': all_params,
            'trainable_percentage': 100.0 * trainable_params / all_params if all_params > 0 else 0.0
        }

def create_model(
    model_name: str,
    num_classes: int,
    mode: str = 'baseline',
    lora_config: Optional[Dict[str, Any]] = None
):
    """Create model based on specified mode"""
    
    if mode == 'baseline':
        return BaselineViT(model_name, num_classes)
    
    elif mode == 'lora':
        if lora_config is None:
            lora_config = {
                'lora_rank': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.1,
                'target_modules': ['qkv', 'proj']
            }
        
        return ViTWithSimpleLoRA(
            model_name,
            num_classes,
            lora_rank=lora_config['lora_rank'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules']
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
