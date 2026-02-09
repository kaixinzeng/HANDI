#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重排轨迹数据脚本 (Remove Traj 6)
输入: processed_trajectories_4D_8traj.npy (8条)
输出: reordered_trajectories_6traj.npy (6条)

重排规则:
删除: 原 Traj 4 (index 3), 原 Traj 6 (index 5)
保留: 1, 2, 3, 5, 7, 8
新顺序:
New 1 <- Old 2 (idx 1) [Train]
New 2 <- Old 3 (idx 2) [Train]
New 3 <- Old 5 (idx 4) [Train]
New 4 <- Old 8 (idx 7) [Train]
New 5 <- Old 1 (idx 0) [Test]
New 6 <- Old 7 (idx 6) [Test]
"""

import numpy as np
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming input is in parent dir or same dir? The original read from ".", let's try reading from parent if not found locally, 
    # but based on previous logs, the big data file might be in h:\handi_vi. 
    # Let's use absolute path to be safe or relative to script.
    # The original file `processed_trajectories_4D_8traj.npy` is likely in h:\handi_vi.
    
    # Try looking in parent directory first since we are in a subdir
    input_path = os.path.join(script_dir, "..", "processed_trajectories_4D_8traj.npy")
    if not os.path.exists(input_path):
        input_path = os.path.join(script_dir, "processed_trajectories_4D_8traj.npy")
        
    output_path = os.path.join(script_dir, "reordered_trajectories_6traj.npy")
    
    if not os.path.exists(input_path):
        # Fallback to absolute path if typical
        input_path = r"h:\handi_vi\processed_trajectories_4D_8traj.npy"
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found.")
            return
        
    data = np.load(input_path)
    print(f"原始数据形状: {data.shape}")
    
    # 映射索引 (0-based)
    # Old indices:
    # 0=T1, 1=T2, 2=T3, 3=T4, 4=T5, 5=T6, 6=T7, 7=T8
    
    # Target:
    # New 0 (T1') <- Old 1 (T2)
    # New 1 (T2') <- Old 2 (T3)
    # New 2 (T3') <- Old 4 (T5)
    # New 3 (T4') <- Old 7 (T8)
    # New 4 (T5') <- Old 0 (T1)
    # New 5 (T6') <- Old 6 (T7)  <-- Old 6 (T7) is index 6
    
    # Note: Old Traj 6 (Index 5) is REMOVED.
    
    new_indices = [1, 2, 4, 7, 0, 6]
    
    new_data = data[new_indices]
    print(f"新数据形状: {new_data.shape}")
    
    np.save(output_path, new_data)
    print(f"已保存重排后的数据: {output_path}")
    
    # 验证一下
    print("\n映射验证:")
    names = ["Old T2", "Old T3", "Old T5", "Old T8", "Old T1", "Old T7"]
    for i, idx in enumerate(new_indices):
        print(f"New Traj {i+1} corresponds to {names[i]} (Old Index {idx})")

if __name__ == "__main__":
    main()
