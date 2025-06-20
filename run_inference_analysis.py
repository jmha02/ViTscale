#!/usr/bin/env python3
"""
Convenience script to run inference analysis from project root
"""

import subprocess
import sys
import os

def main():
    """Run inference analysis"""
    script_path = os.path.join('analysis', 'inference.py')
    
    # Pass through all command line arguments
    cmd = [sys.executable, script_path] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running inference analysis: {e}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())
