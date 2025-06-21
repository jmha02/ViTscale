import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional
from peft import LoraConfig, get_peft_model, TaskType
import re

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

class ViTWithPEFTLoRA(nn.Module):
    """ViT with PEFT LoRA implementation"""
    
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
        
        # Convert target modules to PEFT format
        if target_modules is None:
            target_modules = ["qkv", "proj"]
        
        # Get actual module names from the model
        peft_target_modules = self._get_peft_target_modules(target_modules)
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # For vision tasks
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=peft_target_modules,
            bias="none",  # No bias training
        )
        
        # Apply PEFT LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        
    def _get_peft_target_modules(self, target_modules):
        """Convert target module names to actual module names in the model"""
        peft_target_modules = []
        
        # Get all linear layer names
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                for target in target_modules:
                    if target in name:
                        peft_target_modules.append(name)
                        
        # If no matches found, use common ViT module names
        if not peft_target_modules:
            print("Warning: No target modules found with specified names. Using default ViT modules.")
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Linear):
                    # Common ViT attention and MLP modules
                    if any(x in name for x in ['qkv', 'proj', 'fc1', 'fc2']):
                        peft_target_modules.append(name)
        
        print(f"PEFT target modules: {peft_target_modules}")
        return peft_target_modules
        
    def forward(self, x):
        return self.model(x)
        
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for param in self.model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        return {
            'trainable_params': trainable_params,
            'all_params': all_params,
            'trainable_percentage': 100.0 * trainable_params / all_params if all_params > 0 else 0.0
        }
    
    def merge_and_unload(self):
        """Merge LoRA weights back to base model"""
        return self.model.merge_and_unload()
    
    def save_pretrained(self, save_directory):
        """Save only LoRA adapters"""
        self.model.save_pretrained(save_directory)
    
    def load_adapter(self, adapter_path):
        """Load LoRA adapters"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)

def create_model(
    model_name: str,
    num_classes: int,
    mode: str = 'baseline',
    lora_config: Optional[Dict[str, Any]] = None,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None
):
    """Create model based on specified mode"""
    
    if mode == 'baseline':
        return BaselineViT(model_name, num_classes)
    
    elif mode == 'lora':
        # Use provided parameters or lora_config
        if lora_config is not None:
            lora_rank = lora_config.get('lora_rank', lora_rank)
            lora_alpha = lora_config.get('lora_alpha', lora_alpha) 
            lora_dropout = lora_config.get('lora_dropout', lora_dropout)
            target_modules = lora_config.get('target_modules', target_modules)
        
        if target_modules is None:
            target_modules = ['qkv', 'proj']
        
        return ViTWithPEFTLoRA(
            model_name,
            num_classes,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def print_model_info(model, model_name="Model"):
    """Print detailed model information"""
    param_info = model.get_trainable_parameters()
    
    print(f"\n{model_name} Information:")
    print(f"  Total parameters: {param_info['all_params']:,}")
    print(f"  Trainable parameters: {param_info['trainable_params']:,}")
    print(f"  Trainable percentage: {param_info['trainable_percentage']:.2f}%")
    
    if hasattr(model, 'model') and hasattr(model.model, 'peft_config'):
        print(f"  LoRA rank: {model.model.peft_config['default'].r}")
        print(f"  LoRA alpha: {model.model.peft_config['default'].lora_alpha}")
        print(f"  LoRA dropout: {model.model.peft_config['default'].lora_dropout}")
        print(f"  Target modules: {model.model.peft_config['default'].target_modules}")

def get_model_size_mb(model):
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb
