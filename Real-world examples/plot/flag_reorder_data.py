#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
    new_indices = [1, 2, 4, 7, 0, 6]
    
    new_data = data[new_indices]
    np.save(output_path, new_data)
    names = ["Old T2", "Old T3", "Old T5", "Old T8", "Old T1", "Old T7"]
    for i, idx in enumerate(new_indices):
        print(f"New Traj {i+1} corresponds to {names[i]} (Old Index {idx})")

if __name__ == "__main__":
    main()
