#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot Test Set Time Series Comparison for Reordered Trajectories (New 5, 6)

Data source: reordered_trajectories_6traj.npy

This script compares predictions from multiple methods (HANDI, SINDy, SR3, PSE)
against ground truth for test trajectories. The ODE systems are integrated
using the Radau method for numerical stability.
"""

from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
from scipy.integrate import solve_ivp
import os


class Style:
    """Style configuration for time series comparison plots."""
    
    # Figure dimensions
    fig_w = 12                    # Figure width
    fig_h_per_subplot = 4         # Figure height per subplot
    
    # Line widths
    pred_lw = 5                   # Line width for predictions
    true_lw = 7                   # Line width for ground truth (thicker for visibility)
    
    # Font sizes
    tick_fontsize = 46
    label_fontsize = 32
    title_fontsize = 28
    legend_fontsize = 24
    
    # Axis spine settings
    spine_width = 3.0
    spine_color = "black"
    show_spines = ("left", "bottom")  # Only show left and bottom spines

    # Color scheme
    true_color = "#BBBBBB"        # Light gray for ground truth
    method_colors = {
        "HANDI": "#50AAD8",       # Blue for HANDI predictions
        "SINDy": "#d06569",       # Red for SINDy predictions
        "SR3": '#C9A1CB',         # Light purple for SR3 predictions
        "PSE": "#D8A0A7"          # Pink for PSE predictions
    }


STYLE = Style()


def parse_equations(eq_file):
    """
    Parse differential equations from a text file and create an ODE function.
    
    Reads equations in the format "dx(i)/dt = expression" and compiles them
    into callable Python code for use with scipy's solve_ivp.
    
    Args:
        eq_file: Path to the equation file
        
    Returns:
        system_func: A function f(t, x) returning derivatives for solve_ivp
    """
    with open(eq_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    rhs_list = {}
    for line in content.split('\n'):
        line = line.strip()
        # Skip empty lines, comments, and lines without 'dx'
        if not line or line.startswith('#') or 'dx' not in line:
            continue
        
        # Parse equation format: dx(i)/dt = expression
        m = re.match(r"dx\(?(\d+)\)?/dt\s*=\s*(.*)", line)
        if m:
            idx = int(m.group(1)) - 1  # Convert to 0-based index
            rhs = m.group(2)
            # Convert x(i) notation to x[i-1] for Python indexing
            rhs = re.sub(r"x\((\d+)\)", lambda m: f"x[{int(m.group(1))-1}]", rhs)
            rhs_list[idx] = rhs
    
    # Compile each equation into executable code
    dim = len(rhs_list)
    funcs = []
    for i in range(dim):
        rhs = rhs_list.get(i, "0")  # Default to 0 if equation not found
        code = compile(rhs, f"eq_{i}", "eval")
        funcs.append(code)
    
    def system_func(t, x):
        """
        ODE system function for solve_ivp.
        
        Args:
            t: Current time (unused but required by solve_ivp)
            x: Current state vector
            
        Returns:
            List of derivatives [dx0/dt, dx1/dt, ...]
        """
        return [
            eval(code, {}, {"x": x, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt}) 
            for code in funcs
        ]
    
    return system_func


def integrate_trajectory(func, x0, t_eval):
    """
    Integrate an ODE system from initial condition x0 over specified time points.
    
    Uses the Radau method (implicit Runge-Kutta) for numerical stability with
    potentially stiff systems.
    
    Args:
        func: ODE function f(t, x) returning derivatives
        x0: Initial state vector
        t_eval: Array of time points for evaluation
        
    Returns:
        tuple: (trajectory array of shape (len(t_eval), dim), time array)
               Returns (None, None) on integration failure
    """
    t_span = (t_eval[0], t_eval[-1])
    try:
        sol = solve_ivp(
            func, t_span, x0, 
            t_eval=t_eval, 
            method='Radau',  # Implicit method for stiff systems
            rtol=1e-3,       # Relative tolerance
            atol=1e-6        # Absolute tolerance
        )
        if sol.success:
            return sol.y.T, sol.t
        else:
            # Return partial solution if available
            if hasattr(sol, 'y'):
                return sol.y.T, sol.t
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def apply_spines(ax):
    """
    Configure axis spines visibility and appearance.
    
    Hides all spines except those specified in STYLE.show_spines,
    then applies consistent styling to visible spines.
    
    Args:
        ax: Matplotlib axes object
    """
    # First hide all spines
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    
    # Show and style only the desired spines
    for side in STYLE.show_spines:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(STYLE.spine_width)
        ax.spines[side].set_color(STYLE.spine_color)
    
    # Configure tick positions
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def apply_ticks(ax):
    """
    Configure axis tick appearance and formatting.
    
    Sets tick font size, limits number of ticks to 3 per axis,
    and formats tick labels with one decimal place.
    
    Args:
        ax: Matplotlib axes object
    """
    ax.tick_params(
        labelsize=STYLE.tick_fontsize, 
        width=STYLE.spine_width, 
        direction="out", 
        top=False, 
        right=False
    )
    
    # Limit number of ticks on each axis to 3
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    
    # Format tick labels with one decimal place
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))


def main():
    """
    Main function to generate time series comparison plots.
    
    For each test trajectory (5 and 6) and each state dimension (x1-x4):
    1. Loads the ground truth trajectory data
    2. Parses discovered equations from each method
    3. Integrates the ODE systems from the same initial condition
    4. Plots ground truth vs. predictions from all methods
    5. Saves the figure as an SVG file
    """
    parser = argparse.ArgumentParser()
    
    # Set default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "reordered_trajectories_6traj.npy")
    
    # Equation files for each method (discovered ODE coefficients)
    HANDI_eq_file = os.path.join(script_dir, "HANDI_best_equations_dt0.200.txt")
    SINDy_eq_file = os.path.join(script_dir, "SINDy_dt0.2_equations.txt")
    SR3_eq_file = os.path.join(script_dir, "SR3_dt0.2_equations.txt")
    PSE_eq_file = os.path.join(script_dir, "PSE_dt0.2_equations.txt")
    default_output = os.path.join(script_dir, "final_test_compare")
    
    # Command line arguments
    parser.add_argument("--data_path", type=str, default=default_data_path,
                        help="Path to the reordered trajectory data file")
    parser.add_argument("--HANDI_eq_file", type=str, default=HANDI_eq_file,
                        help="Path to HANDI discovered equations")
    parser.add_argument("--SINDy_eq_file", type=str, default=SINDy_eq_file,
                        help="Path to SINDy discovered equations")
    parser.add_argument("--SR3_eq_file", type=str, default=SR3_eq_file,
                        help="Path to SR3 discovered equations")
    parser.add_argument("--PSE_eq_file", type=str, default=PSE_eq_file,
                        help="Path to PSE discovered equations")
    parser.add_argument("--output", type=str, default=default_output,
                        help="Prefix for output file names")
    args = parser.parse_args()
    
    # Validate all required input files exist
    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} not found. Please run reorder_data.py first.")
        return

    if not os.path.exists(args.HANDI_eq_file):
        print(f"Error: {args.HANDI_eq_file} not found.")
        return

    if not os.path.exists(args.SINDy_eq_file):
        print(f"Error: {args.SINDy_eq_file} not found.")
        return

    if not os.path.exists(args.SR3_eq_file):
        print(f"Error: {args.SR3_eq_file} not found.")
        return

    if not os.path.exists(args.PSE_eq_file):
        print(f"Error: {args.PSE_eq_file} not found.")
        return

    # Load trajectory data
    # Expected shape: (6, T, 4) where 6=trajectories, T=timesteps, 4=state dimensions
    data = np.load(args.data_path)
    
    # Test trajectory indices (0-based, corresponding to trajectories 5 and 6)
    test_indices = [4, 5]
    labels = ["Traj 5", "Traj 6"]
    
    # Time parameters
    raw_dt = 0.03333333  # Time step (30 Hz sampling rate)
    t_raw = np.arange(data.shape[1]) * raw_dt  # Raw time array
    
    # Parse equation files and create ODE functions for each method
    ode_func_handi = parse_equations(args.HANDI_eq_file)
    ode_func_sindy = parse_equations(args.SINDy_eq_file)
    ode_func_sr3 = parse_equations(args.SR3_eq_file)
    ode_func_pse = parse_equations(args.PSE_eq_file)
    
    # Default integration start time (currently not used as we start from 0)
    start_time_default = 3.0
    
    # Calculate valid length for trajectory 5
    # Trajectory 5 has been preprocessed: starts from 1.5s, with zeros padded at the end
    traj5_valid_length = data.shape[1] - int(1.5 / raw_dt)
    
    # State dimensions to plot
    # x1=position_x, x2=velocity_x, x3=position_y, x4=velocity_y
    dims = [(0, "x1"), (1, "x2"), (2, "x3"), (3, "x4")]
    
    # Generate plots for each dimension and each test trajectory
    for dim_idx, dim_name in dims:
        for idx, name in zip(test_indices, labels):
            fig, ax = plt.subplots(1, 1, figsize=(STYLE.fig_w, STYLE.fig_h_per_subplot))
            
            # Handle special case for trajectory 5 (idx=4)
            # Data has been preprocessed to start from 1.5s, zeros padded at end
            if idx == 4:
                # Trajectory 5: data already starts from 1.5s offset, exclude zero padding
                traj_true = data[idx][:traj5_valid_length]
            else:
                # Trajectory 6: use complete data
                traj_true = data[idx]
            
            # Both trajectories now start from time 0 for plotting
            t_plot = np.arange(len(traj_true)) * raw_dt
            plot_start_idx = 0      # Start plotting from beginning
            integrate_start_idx = 0  # Start integration from beginning
            
            # Extract ground truth for current dimension
            x_true_full = traj_true[:, dim_idx]
            
            # Time and data arrays for plotting
            t_plot_true = t_plot[plot_start_idx:]
            x_plot_true = x_true_full[plot_start_idx:]
            
            # Time array and initial conditions for integration
            t_eval_full = t_plot[integrate_start_idx:]
            x0 = traj_true[integrate_start_idx, :]  # Initial state (all dimensions)
            
            # Plot ground truth trajectory (gray, thicker line)
            ax.plot(t_plot_true, x_plot_true, color=STYLE.true_color, 
                    linewidth=STYLE.true_lw, alpha=0.8, zorder=1)

            # Plot predictions from each method
            # Order: PSE, SR3, SINDy, HANDI (HANDI plotted last to appear on top)
            for method_name, ode_func in (
                ("PSE", ode_func_pse), 
                ("SR3", ode_func_sr3), 
                ("SINDy", ode_func_sindy), 
                ("HANDI", ode_func_handi)
            ):
                # Integrate ODE from initial condition
                pred_y, pred_t = integrate_trajectory(ode_func, x0, t_eval_full)
                color = STYLE.method_colors.get(method_name, "blue")
                
                if pred_y is None:
                    continue  # Skip if integration failed
                
                # Extract prediction for current dimension
                x_pred = pred_y[:, dim_idx]
                
                # Plot prediction
                ax.plot(pred_t, x_pred, color=color, linewidth=STYLE.pred_lw, 
                        alpha=0.95, zorder=10)

            # Special y-axis limit for x2 dimension of trajectory 5
            # Prevents extreme values from distorting the plot
            if dim_name == "x2" and idx == 4:
                ax.set_ylim(-1.5, 4.5)

            # Apply styling
            apply_spines(ax)
            apply_ticks(ax)
            ax.grid(False)
            
            # Set x-axis range from 0 to 12 seconds
            ax.set_xlim(0, 12)

            # Save figure
            plt.tight_layout()
            output_path = f"{args.output}_{name.replace(' ', '').lower()}_{dim_name}.svg"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {output_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
