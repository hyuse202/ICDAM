"""
Setup Verification Script
Checks that all dependencies and data are properly configured.
"""

import os
import numpy as np
import ortools  # <--- Sửa: Import gói chính để check version
from ortools.sat.python import cp_model

def main():
    print("=" * 60)
    print("ENVIRONMENT HEALTH CHECK")
    print("=" * 60)
    
    # 1. Print versions of numpy and ortools
    print("\n[1] Checking installed packages...")
    print(f"    numpy version: {np.__version__}")
    # Sửa dòng dưới này để gọi đúng version của ortools
    print(f"    ortools version: {ortools.__version__}") 
    
    # 2. Check if the directory exists
    print("\n[2] Checking data directory...")
    data_dir = os.path.join("data", "raw", "rcpsp", "j30")
    
    if os.path.exists(data_dir):
        print(f" Directory exists: {data_dir}")
    else:
        print(f" Directory NOT found: {data_dir}")
        return
    
    # 3. Count and print the number of files
    print("\n[3] Counting files in directory...")
    # Thêm check os.path.isfile để tránh đếm nhầm folder con (nếu có)
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        print(f"    Found {len(files)} files in {data_dir}")
        
        if len(files) == 0:
            print("    ✗ No files found in directory")
            return
    else:
        return # Đã báo lỗi ở trên rồi
    
    # 4. Read the first file and print its first 10 lines
    print("\n[4] Reading first file to verify format...")
    first_file = files[0]
    first_file_path = os.path.join(data_dir, first_file)
    print(f"    File: {first_file}")
    print(f"    Path: {first_file_path}")
    print("\n    First 10 lines:")
    print("    " + "-" * 56)
    
    try:
        with open(first_file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"    {i+1:2d} | {line.rstrip()}")
    except Exception as e:
        print(f" Error reading file: {e}")
        return
    
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()