#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to plot trajectory
- Opening to the left
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# ------------------ Plot Style ------------------
class PlotStyle:
    line_width: float = 10
    truth_color: str = 'lightgray'
    truth_linewidth_factor: float = 2.0

STYLE = PlotStyle()

# ------------------ Core Functions ------------------
def load_equations_from_file(filepath: str):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Equation file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = r"dx\((\d+)\)/dt\s*=\s*(.*)"
    matches = re.findall(pattern, content)
    
    if not matches:
        raise ValueError(f"Could not find any equations in format 'dx(i)/dt = ...' in file '{filepath}'.")
        
    matches.sort(key=lambda m: int(m[0]))
    eqs = []
    for i_str, eq_str in matches:
        processed_eq = eq_str.strip()
        for j in range(1, len(matches) + 2):
            processed_eq = processed_eq.replace(f"x({j})", f"x[{j-1}]")
        eqs.append(processed_eq)
    print(f"Successfully loaded {len(eqs)} equations.")
    return eqs

def make_vector_field(eqs: list):
    codes = [compile(eq, f"<eq{i+1}>", "eval") for i, eq in enumerate(eqs)]
    def f(t, x):
        local_env = {"t": t, "x": x, "np": np}
        return np.array([eval(code, {}, local_env) for code in codes], dtype=float)
    return f

def load_npy_safely(path: str):
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception:
        arr = np.load(path)
    if arr.dtype == object:
        arr = np.array([np.asarray(e, dtype=float) for e in arr])
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr

def perform_peak_analysis(sol, t_dense, truth_data=None, truth_time_axis=None):
    """Perform peak analysis to find representative peak points"""

    print("\nStarting peak analysis...")
    m = 10.0

    # Predicted data analysis

    x_pred, vx_pred, y_pred, vy_pred = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    total_ke_pred = 0.5 * m * (vx_pred**2 + vy_pred**2)
    ke_max_indices_pred, _ = find_peaks(total_ke_pred, distance=50, prominence=0.15)

    print(f"Found {len(ke_max_indices_pred)} peaks in predicted data.")

    # True data analysis

    x_truth, y_truth = None, None
    if truth_data is not None:
        x_truth, vx_truth, y_truth, vy_truth = truth_data[:, 0], truth_data[:, 1], truth_data[:, 2], truth_data[:, 3]

    # Find representative peak points

    representative_points_pred_coords = []
    
    if ke_max_indices_pred.size > 1:
        group1_indices_pred = ke_max_indices_pred[::2]
        group2_indices_pred = ke_max_indices_pred[1::2]
        
        if group1_indices_pred.size > 0:
            avg_x1 = np.mean(x_pred[group1_indices_pred])
            closest_idx_in_group = np.argmin(np.abs(x_pred[group1_indices_pred] - avg_x1))
            final_idx1 = group1_indices_pred[closest_idx_in_group]
            coords1 = (x_pred[final_idx1], y_pred[final_idx1])
            representative_points_pred_coords.append(coords1)
        
        if group2_indices_pred.size > 0:
            avg_x2 = np.mean(x_pred[group2_indices_pred])
            closest_idx_in_group = np.argmin(np.abs(x_pred[group2_indices_pred] - avg_x2))
            final_idx2 = group2_indices_pred[closest_idx_in_group]
            coords2 = (x_pred[final_idx2], y_pred[final_idx2])
            representative_points_pred_coords.append(coords2)

    analysis_results = {
        "x_pred": x_pred, "y_pred": y_pred,
        "representative_points_pred_coords": representative_points_pred_coords,
        "x_truth": x_truth, "y_truth": y_truth
    }
    
    return analysis_results

def plot_tip_trajectory(results, output_path):
    """
    Plot trajectory
    - Opening left: rotate 90 degrees plot(-y, x)
    """

    fig, ax = plt.subplots(figsize=(5, 12))
    
    # 
    if results["x_truth"] is not None and results["y_truth"] is not None:
        # 90plot(-y, x)
        ax.plot(-results["y_truth"], results["x_truth"], 
                color=STYLE.truth_color,
                lw=STYLE.line_width * STYLE.truth_linewidth_factor, 
                zorder=1)
    
    # 
    # 90plot(-y, x)
    ax.plot(-results["y_pred"], results["x_pred"], 
            color='#d06569', 
            lw=STYLE.line_width, 
            zorder=2)
    
    # Plot predicted representative points (commented out)
    # if results["representative_points_pred_coords"]:
    #     pred_rep_x = [p[0] for p in results["representative_points_pred_coords"]]
    #     pred_rep_y = [p[1] for p in results["representative_points_pred_coords"]]
    #     ax.scatter(pred_rep_x, pred_rep_y, 
    #               s=600, color='#4FA8D5', zorder=5, marker='o')

    
    # Add basic axis labels
    ax.set_xlabel("y", fontsize=14)
    ax.set_ylabel("x", fontsize=14)
    
    # Optional: Keep spines but remove right/top 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Invert x-axis as requested (Right-to-Left)
    ax.invert_xaxis()

    
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Tip trajectory plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot tip trajectory only")
    parser.add_argument("--dt", type=float, default=0.2, help="Time step (0.1, 0.1666, 0.2)")
    parser.add_argument("--trajectory-index", type=int, default=1, 
                       help="Trajectory index to plot (default=1 for trajectory 2)")
    parser.add_argument("--output-dir", type=str, default="../result",
                       help="Output directory for SVG files")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Determine NPY path based on dt
    # Using the same logic as before to map dt to downsample suffix
    dt_map = {0.1: "ds3", 0.1666: "ds5", 0.2: "ds6", 0.1667: "ds5"}
    ds_suffix = dt_map.get(args.dt, "ds6")
    
    # Try different possible paths for the data file
    possible_npy_paths = [
        rf"../data/flag_traj_4_dt{args.dt}.npy",
        rf"../data/4d_dt{args.dt}.npy",

        rf"../data/4d_{ds_suffix}.npy"
    ]
    
    npy_path = None
    for path in possible_npy_paths:
        if os.path.exists(path):
            npy_path = path
            break
            
    if npy_path is None:
        raise FileNotFoundError(f"Could not find NPY data file for dt={args.dt}")
    
    print(f"Loading data from: {npy_path}")
    X_raw = load_npy_safely(npy_path)

    # 2. Setup simulation parameters
    stride = 6
    raw_dt = 0.033333
    model_dt = raw_dt * stride
    num_timesteps = X_raw.shape[1] # X_raw is likely already downsampled in the new files, so let's adjust logic
    
    t_span = [0, (num_timesteps - 1) * args.dt]
    dense_factor = 30
    num_dense = (num_timesteps - 1) * dense_factor + 1
    t_eval = np.linspace(t_span[0], t_span[1], num_dense)
    truth_time_axis = np.arange(num_timesteps) * args.dt

    # 3. Process specified trajectory
    i = args.trajectory_index
    if i >= X_raw.shape[0]:
        raise ValueError(f"Trajectory index {i} is out of range. Available trajectories: 0-{X_raw.shape[0]-1}")

    print(f"\n=== Processing Trajectory {i+1} for dt={args.dt} ===")
    
    initial_condition = X_raw[i, 0, :].astype(float)
    current_truth_data = X_raw[i]

    # 4. Loop through methods
    dt_str = "0.1666" if abs(args.dt - 0.1666) < 1e-4 else str(args.dt)
    # Only process HANDI method as requested
    methods = ["HANDI"]
    
    for method in methods:
        if method == "HANDI":
            eq_file = rf"../result/{method}_flag_dt{dt_str}.txt"
        else:
            eq_file = rf"../data/{method}_flag_dt{dt_str}.txt"
        if not os.path.exists(eq_file):
            print(f"Warning: Equation file not found for {method}: {eq_file}")
            continue
            
        print(f"\n--- Testing {method} ---")
        try:
            eqs = load_equations_from_file(eq_file)
            vf = make_vector_field(eqs)
            
            print(f"Simulating for {t_span[1]:.2f} seconds...")
            sol = solve_ivp(
                vf, t_span, initial_condition, t_eval=t_eval,
                method="RK45", rtol=1e-8, atol=1e-10
            )
            
            if not sol.success:
                print(f"Warning: ODE solver failed for {method}: {sol.message}")
                continue
                
            # Perform analysis and plot
            analysis_results = perform_peak_analysis(
                sol, t_eval,
                truth_data=current_truth_data,
                truth_time_axis=truth_time_axis
            )
            
            output_path = os.path.join(args.output_dir, f"figure5_flag_traj_{i+1}_{method}_dt{dt_str}.pdf")

            plot_tip_trajectory(analysis_results, output_path)
            
        except Exception as e:
            print(f"Error processing {method}: {e}")

    print("\nAll done!")

if __name__ == "__main__":
    main()
