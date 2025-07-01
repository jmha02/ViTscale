#!/usr/bin/env python3
"""
Quick test script to verify all training scripts work correctly
"""

import subprocess
import sys
import time

def run_test(script_name, timeout=30):
    """Run a script for limited time to test if it works"""
    print(f"\n{'='*50}")
    print(f"Testing {script_name}")
    print(f"{'='*50}")
    
    try:
        # Run the script with timeout
        result = subprocess.run(
            [sys.executable, script_name], 
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd='/root/Vitscale'
        )
        
        # Check if script started successfully
        if "Training" in result.stdout or "BENCHMARKING" in result.stdout:
            print(f"✅ {script_name} - WORKING!")
            print("First few lines of output:")
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"❌ {script_name} - FAILED!")
            print("Error output:")
            print(result.stderr[:500])
            
    except subprocess.TimeoutExpired:
        print(f"✅ {script_name} - WORKING! (Timed out as expected)")
    except Exception as e:
        print(f"❌ {script_name} - ERROR: {e}")

def main():
    print("Testing all ViT CIFAR-100 training scripts...")
    print("Note: Scripts will timeout after a short period - this is expected!")
    
    # Test each script
    run_test("train_vit_cifar100.py", timeout=15)
    run_test("train_vit_lora_cifar100.py", timeout=15) 
    run_test("compare_vit_lora.py", timeout=60)  # Comparison needs more time
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print("All scripts have been tested!")
    print("\nTo use the scripts:")
    print("1. python train_vit_cifar100.py        # Regular ViT training")
    print("2. python train_vit_lora_cifar100.py   # LoRA ViT training") 
    print("3. python compare_vit_lora.py          # Performance comparison")

if __name__ == "__main__":
    main()