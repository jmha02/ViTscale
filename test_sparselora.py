#!/usr/bin/env python3
"""
Test script for SparseLoRA implementation
"""

import torch
import torch.nn as nn
import timm
from sparse_lora import apply_sparse_lora_to_vit, SparseLoRALayer

def test_sparse_lora_layer():
    """Test basic SparseLoRA layer functionality"""
    print("Testing SparseLoRA Layer...")
    
    # Create a simple linear layer
    original_layer = nn.Linear(128, 64)
    
    # Create SparseLoRA layer
    sparse_layer = SparseLoRALayer(
        original_layer=original_layer,
        rank=4,
        alpha=1.0,
        sparse_threshold=0.01,
        sparse_decay=0.99
    )
    
    # Test forward pass
    x = torch.randn(8, 128)
    output = sparse_layer(x)
    
    assert output.shape == (8, 64), f"Expected shape (8, 64), got {output.shape}"
    print("✓ Forward pass works correctly")
    
    # Test sparsity info
    sparsity_info = sparse_layer.get_sparsity_info()
    required_keys = ['sparsity', 'active_params', 'total_params', 'threshold']
    for key in required_keys:
        assert key in sparsity_info, f"Missing key {key} in sparsity info"
    
    print("✓ Sparsity info works correctly")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    assert sparse_layer.lora_A.grad is not None, "Gradient should not be None"
    assert sparse_layer.lora_B.grad is not None, "Gradient should not be None"
    
    print("✓ Backward pass works correctly")
    
    # Test sparsity update
    sparse_layer.update_sparsity()
    print("✓ Sparsity update works correctly")
    
    print("SparseLoRA Layer test passed!\n")

def test_vit_integration():
    """Test SparseLoRA integration with ViT model"""
    print("Testing ViT Integration...")
    
    # Create a small ViT model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    
    # Get original parameter count
    original_params = sum(p.numel() for p in model.parameters())
    original_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Original model - Total: {original_params:,}, Trainable: {original_trainable:,}")
    
    # Apply SparseLoRA
    sparse_wrapper = apply_sparse_lora_to_vit(
        model=model,
        rank=4,
        alpha=1.0,
        sparse_threshold=0.01,
        sparse_decay=0.99
    )
    
    # Check parameter counts
    total_params, trainable_params = sparse_wrapper.get_trainable_parameters()
    
    print(f"After SparseLoRA - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    assert total_params > original_params, "Should have more total parameters due to LoRA additions"
    assert trainable_params < original_trainable, "Should have fewer trainable parameters"
    
    print("✓ Parameter counts are correct")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    print("✓ Forward pass works correctly")
    
    # Test sparsity info
    sparsity_info = sparse_wrapper.get_total_sparsity_info()
    required_keys = ['overall_sparsity', 'average_sparsity', 'total_lora_params', 'active_lora_params', 'num_sparse_layers']
    for key in required_keys:
        assert key in sparsity_info, f"Missing key {key} in sparsity info"
    
    print("✓ Sparsity info works correctly")
    
    # Test backward pass and sparsity update
    loss = output.sum()
    loss.backward()
    sparse_wrapper.update_sparsity()
    
    print("✓ Backward pass and sparsity update work correctly")
    
    # Print sparsity summary
    sparse_wrapper.print_sparsity_summary()
    
    print("ViT Integration test passed!\n")

def test_memory_efficiency():
    """Test memory efficiency of SparseLoRA"""
    print("Testing Memory Efficiency...")
    
    # Test with different model sizes
    model_configs = [
        ('vit_tiny_patch16_224', 10),
        ('vit_small_patch16_224', 100)
    ]
    
    for model_name, num_classes in model_configs:
        print(f"\nTesting {model_name}...")
        
        # Create model
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # Apply SparseLoRA
        sparse_wrapper = apply_sparse_lora_to_vit(model, rank=8, alpha=16.0)
        
        # Get parameter info
        total_params, trainable_params = sparse_wrapper.get_trainable_parameters()
        sparsity_info = sparse_wrapper.get_total_sparsity_info()
        
        efficiency = trainable_params / total_params
        print(f"  Parameter efficiency: {efficiency:.4%}")
        print(f"  LoRA sparsity: {sparsity_info['overall_sparsity']:.2%}")
        print(f"  Number of SparseLoRA layers: {sparsity_info['num_sparse_layers']}")
        
        # Verify efficiency
        assert efficiency < 0.1, f"Parameter efficiency should be < 10%, got {efficiency:.2%}"
        print(f"  ✓ Efficient parameter usage")
    
    print("Memory Efficiency test passed!\n")

def test_training_simulation():
    """Simulate a few training steps to test the complete pipeline"""
    print("Testing Training Simulation...")
    
    # Create model and data
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    sparse_wrapper = apply_sparse_lora_to_vit(model, rank=4, alpha=1.0)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Simulate training steps
    model.train()
    for step in range(5):
        # Create dummy batch
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 10, (4,))
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Update sparsity
        sparse_wrapper.update_sparsity()
        
        # Optimizer step
        optimizer.step()
        
        # Get sparsity info
        sparsity_info = sparse_wrapper.get_total_sparsity_info()
        
        print(f"  Step {step+1}: Loss = {loss.item():.4f}, Sparsity = {sparsity_info['overall_sparsity']:.2%}")
    
    print("✓ Training simulation completed successfully")
    print("Training Simulation test passed!\n")

def main():
    """Run all tests"""
    print("="*60)
    print("SPARSELORA IMPLEMENTATION TESTS")
    print("="*60)
    
    try:
        test_sparse_lora_layer()
        test_vit_integration()
        test_memory_efficiency()
        test_training_simulation()
        
        print("="*60)
        print("ALL TESTS PASSED! ✅")
        print("SparseLoRA implementation is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()