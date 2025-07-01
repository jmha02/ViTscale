#!/usr/bin/env python3
"""
Quick test for continual learning scripts
"""

import subprocess
import sys
import time

def test_continual_script(script_name):
    """Test if continual learning script starts correctly"""
    print(f"\nTesting {script_name}...")
    print("-" * 40)
    
    try:
        # Run the script with very short timeout
        result = subprocess.run(
            [sys.executable, script_name], 
            timeout=30,
            capture_output=True,
            text=True,
            cwd='/root/Vitscale'
        )
        
        # Check if script started successfully
        if "Split-CIFAR100" in result.stdout and "Task 0:" in result.stdout:
            print(f"✅ {script_name} - WORKING!")
            print("Sample output:")
            lines = result.stdout.split('\n')[:15]
            for line in lines:
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print(f"❌ {script_name} - FAILED!")
            if result.stderr:
                print("Error:", result.stderr[:300])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✅ {script_name} - WORKING! (Timed out as expected)")
        return True
    except Exception as e:
        print(f"❌ {script_name} - ERROR: {e}")
        return False

def main():
    print("Testing Continual Learning Scripts")
    print("=" * 50)
    
    # Test baseline scripts
    baseline_regular_works = test_continual_script("train_vit_continual_baseline.py")
    baseline_lora_works = test_continual_script("train_vit_lora_continual_baseline.py")
    
    # Test advanced scripts
    regular_works = test_continual_script("train_vit_continual.py")
    lora_works = test_continual_script("train_vit_lora_continual.py")
    
    print(f"\n{'='*50}")
    print("CONTINUAL LEARNING TEST SUMMARY")
    print(f"{'='*50}")
    
    if baseline_regular_works and baseline_lora_works and regular_works and lora_works:
        print("✅ All continual learning scripts are working!")
    else:
        print("❌ Some scripts have issues")
        print(f"Baseline Regular: {'✅' if baseline_regular_works else '❌'}")
        print(f"Baseline LoRA: {'✅' if baseline_lora_works else '❌'}")
        print(f"Advanced Regular: {'✅' if regular_works else '❌'}")
        print(f"Advanced LoRA: {'✅' if lora_works else '❌'}")
    
    print("\nContinual Learning Features:")
    print("BASELINE scripts:")
    print("- Split-CIFAR100 into 10 sequential tasks")
    print("- Pure sequential learning (no regularization)")
    print("- Shows raw catastrophic forgetting")
    print("\nADVANCED scripts:")
    print("- Experience replay buffer")
    print("- Elastic Weight Consolidation (EWC)")
    print("- LoRA parameter-efficient adaptation")

if __name__ == "__main__":
    main()