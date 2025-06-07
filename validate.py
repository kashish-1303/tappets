import os
import numpy as np
from collections import Counter

def check_dataset_balance(normal_dir, abnormal_dir):
    """Check class balance in the dataset"""
    
    normal_files = len([f for f in os.listdir(normal_dir) if f.endswith('.npy')])
    abnormal_files = len([f for f in os.listdir(abnormal_dir) if f.endswith('.npy')])
    
    total = normal_files + abnormal_files
    normal_ratio = normal_files / total
    abnormal_ratio = abnormal_files / total
    
    print(f"Dataset Balance:")
    print(f"Normal samples: {normal_files} ({normal_ratio:.2%})")
    print(f"Abnormal samples: {abnormal_files} ({abnormal_ratio:.2%})")
    print(f"Imbalance ratio: {max(normal_files, abnormal_files) / min(normal_files, abnormal_files):.2f}:1")
    
    if normal_ratio > 0.9 or abnormal_ratio > 0.9:
        print("WARNING: Severe class imbalance detected!")
    
    return normal_files, abnormal_files

if __name__ == "__main__":
    check_dataset_balance("processed_data/normal", "processed_data/abnormal")