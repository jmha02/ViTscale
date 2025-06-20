#!/usr/bin/env python3
"""
Convenience script to run training benchmark from project root
"""

import subprocess
import sys
import os

def main():
    """Run training benchmark"""
    script_path = os.path.join('benchmarks', 'training.py')
    
    # Pass through all command line arguments
    cmd = [sys.executable, script_path] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running training benchmark: {e}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())
