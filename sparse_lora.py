import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

class SparseLoRALayer(nn.Module):
    """
    Sparse Low-Rank Adaptation (SparseLoRA) layer.
    Combines LoRA with dynamic sparsity for enhanced parameter efficiency.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        sparse_threshold: float = 0.01,
        sparse_decay: float = 0.99,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.sparse_threshold = sparse_threshold
        self.sparse_decay = sparse_decay
        self.training_step = 0
        
        # Get dimensions from original layer
        if hasattr(original_layer, 'weight'):
            if len(original_layer.weight.shape) == 2:  # Linear layer
                self.in_features = original_layer.weight.shape[1]
                self.out_features = original_layer.weight.shape[0]
            else:
                raise ValueError(f"Unsupported layer shape: {original_layer.weight.shape}")
        else:
            raise ValueError("Original layer must have weight attribute")
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Sparsity tracking
        self.register_buffer('active_mask', torch.ones_like(self.lora_A, dtype=torch.bool))
        self.register_buffer('importance_scores', torch.zeros_like(self.lora_A))
        
        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def update_sparsity(self):
        """Update sparsity mask based on importance scores"""
        if not self.training:
            return
        
        self.training_step += 1
        
        # Update importance scores based on gradient magnitude
        if self.lora_A.grad is not None:
            grad_magnitude = torch.abs(self.lora_A.grad)
            # Exponential moving average of importance
            self.importance_scores = 0.9 * self.importance_scores + 0.1 * grad_magnitude
        
        # Dynamic threshold decay
        current_threshold = self.sparse_threshold * (self.sparse_decay ** (self.training_step // 100))
        current_threshold = max(current_threshold, 0.001)  # Minimum threshold
        
        # Update active mask based on importance
        self.active_mask = self.importance_scores > current_threshold
        
        # Ensure minimum sparsity (at least 10% active)
        if self.active_mask.sum().item() < 0.1 * self.active_mask.numel():
            # Keep top 10% most important parameters
            flat_importance = self.importance_scores.flatten()
            k = int(0.1 * len(flat_importance))
            _, top_indices = torch.topk(flat_importance, k)
            
            new_mask = torch.zeros_like(flat_importance, dtype=torch.bool)
            new_mask[top_indices] = True
            self.active_mask = new_mask.reshape(self.importance_scores.shape)
    
    def forward(self, x):
        # Original layer forward pass
        result = self.original_layer(x)
        
        # Apply dropout if specified
        if self.dropout_layer is not None:
            x_lora = self.dropout_layer(x)
        else:
            x_lora = x
        
        # Apply sparsity mask to LoRA weights
        sparse_lora_A = self.lora_A * self.active_mask.float()
        
        # LoRA forward pass: x @ A^T @ B^T
        lora_out = F.linear(x_lora, sparse_lora_A)    # x @ A^T (A has shape [rank, in_features])
        lora_out = F.linear(lora_out, self.lora_B)    # @ B^T (B has shape [out_features, rank])
        
        # Scale by alpha/rank and add to original output
        scaling = self.alpha / self.rank
        result = result + scaling * lora_out
        
        return result
    
    def get_sparsity_info(self) -> Dict[str, float]:
        """Get current sparsity information"""
        total_params = self.active_mask.numel()
        active_params = self.active_mask.sum().item()
        sparsity = 1.0 - (active_params / total_params)
        
        return {
            'sparsity': sparsity,
            'active_params': active_params,
            'total_params': total_params,
            'threshold': self.sparse_threshold * (self.sparse_decay ** (self.training_step // 100))
        }


class SparseLoRAWrapper:
    """
    Wrapper to apply SparseLoRA to specific modules in a model
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str] = None,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        sparse_threshold: float = 0.01,
        sparse_decay: float = 0.99,
    ):
        self.model = model
        self.target_modules = target_modules or ["qkv", "proj", "fc1", "fc2"]
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.sparse_threshold = sparse_threshold
        self.sparse_decay = sparse_decay
        self.sparse_layers = []
        
        self._apply_sparse_lora()
    
    def _apply_sparse_lora(self):
        """Apply SparseLoRA to target modules"""
        def apply_to_module(module, name=""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                # Check if this module should be replaced
                if any(target in child_name for target in self.target_modules):
                    if isinstance(child_module, nn.Linear):
                        # Replace with SparseLoRA layer
                        sparse_layer = SparseLoRALayer(
                            child_module,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout,
                            sparse_threshold=self.sparse_threshold,
                            sparse_decay=self.sparse_decay
                        )
                        setattr(module, child_name, sparse_layer)
                        self.sparse_layers.append((full_name, sparse_layer))
                        print(f"Applied SparseLoRA to {full_name}")
                else:
                    # Recursively apply to children
                    apply_to_module(child_module, full_name)
        
        apply_to_module(self.model)
        print(f"Applied SparseLoRA to {len(self.sparse_layers)} layers")
    
    def update_sparsity(self):
        """Update sparsity for all SparseLoRA layers"""
        for _, layer in self.sparse_layers:
            layer.update_sparsity()
    
    def get_total_sparsity_info(self) -> Dict[str, float]:
        """Get aggregated sparsity information"""
        total_params = 0
        total_active = 0
        avg_sparsity = 0
        
        for _, layer in self.sparse_layers:
            info = layer.get_sparsity_info()
            total_params += info['total_params']
            total_active += info['active_params']
            avg_sparsity += info['sparsity']
        
        if len(self.sparse_layers) > 0:
            avg_sparsity /= len(self.sparse_layers)
            overall_sparsity = 1.0 - (total_active / total_params) if total_params > 0 else 0
        else:
            avg_sparsity = 0
            overall_sparsity = 0
        
        return {
            'overall_sparsity': overall_sparsity,
            'average_sparsity': avg_sparsity,
            'total_lora_params': total_params,
            'active_lora_params': total_active,
            'num_sparse_layers': len(self.sparse_layers)
        }
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get total and trainable parameter counts"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def print_sparsity_summary(self):
        """Print detailed sparsity summary"""
        info = self.get_total_sparsity_info()
        total_params, trainable_params = self.get_trainable_parameters()
        
        print(f"\n{'='*50}")
        print("SPARSELORA SPARSITY SUMMARY")
        print(f"{'='*50}")
        print(f"Total model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Parameter efficiency: {trainable_params/total_params:.2%}")
        print(f"Number of SparseLoRA layers: {info['num_sparse_layers']}")
        print(f"Total LoRA parameters: {info['total_lora_params']:,}")
        print(f"Active LoRA parameters: {info['active_lora_params']:,}")
        print(f"Overall LoRA sparsity: {info['overall_sparsity']:.2%}")
        print(f"Average layer sparsity: {info['average_sparsity']:.2%}")
        print(f"{'='*50}")


def apply_sparse_lora_to_vit(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    sparse_threshold: float = 0.01,
    sparse_decay: float = 0.99,
) -> SparseLoRAWrapper:
    """
    Convenience function to apply SparseLoRA to a Vision Transformer model
    """
    # Common ViT attention module names
    target_modules = ["qkv", "proj", "fc1", "fc2"]
    
    wrapper = SparseLoRAWrapper(
        model=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        sparse_threshold=sparse_threshold,
        sparse_decay=sparse_decay
    )
    
    return wrapper