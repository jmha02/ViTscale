#!/usr/bin/env python3
"""
Convenience script to run inference benchmark from project root
"""

import subprocess
import sys
import os

def main():
    """Run inference benchmark"""
    script_path = os.path.join('benchmarks', 'inference.py')
    
    # Pass through all command line arguments
    cmd = [sys.executable, script_path] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running inference benchmark: {e}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())
