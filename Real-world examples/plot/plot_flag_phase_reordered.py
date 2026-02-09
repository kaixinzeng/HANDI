#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot Phase Portraits for Reordered Trajectories (New 1-6)

Data source: reordered_trajectories_6traj.npy
Train set: Index 0-3 (New 1-4) -> Gray color
Test set: Index 4-5 (New 5-6) -> Blue/Purple color

Plot style follows jiahao.py conventions.
"""

from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.ticker as ticker


class Style:
    """Style configuration for phase portrait plots."""
    
    # Figure dimensions
    fig_size = (9, 9)
    
    # Line styles
    method_lw = 4.0               # Line width for trajectory curves
    method_alpha = 1              # Line transparency
    marker_size = 36              # Size for initial point markers
    
    # Font sizes
    tick_fontsize = 56
    label_fontsize = 32
    title_fontsize = 28
    
    # Axis spine settings
    spine_width = 4
    spine_color = "black"
    show_spines = ("left", "bottom")  # Only show left and bottom spines
    
    # Color scheme
    train_color = "#B0B0B0"       # Gray for training trajectories
    test_color = "#414592"        # Purple/Blue for test trajectories


STYLE = Style()


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
    
    Sets tick font size, locator to limit number of ticks,
    and formats y-axis values to one decimal place.
    
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
    
    # Format y-axis tick labels with one decimal place
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))


def main():
    """
    Main function to generate phase portrait plots.
    
    Creates two types of phase portraits:
    1. Position space: y vs x
    2. Velocity space: dy/dt vs dx/dt
    
    Each plot shows all 6 trajectories with:
    - Training trajectories (1-4) in gray
    - Test trajectories (5-6) in purple/blue
    - Initial points marked with large colored circles
    """
    parser = argparse.ArgumentParser()
    
    # Set default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "reordered_trajectories_6traj.npy")
    
    # Command line arguments
    parser.add_argument("--data_path", type=str, default=default_data_path,
                        help="Path to the reordered trajectory data file")
    parser.add_argument("--output_prefix", type=str, default=os.path.join(script_dir, "final_phase"),
                        help="Prefix for output file names")
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} not found.")
        return
    
    # Load trajectory data
    # Expected shape: (6, T, 4) where 6=trajectories, T=timesteps, 4=state dimensions
    data = np.load(args.data_path)
    
    # Time step of raw data (30 Hz sampling rate)
    raw_dt = 0.03333333
    
    # Calculate valid length for trajectory 5
    # Trajectory 5 has been preprocessed: starts from 1.5s, with zeros padded at the end
    # Original total length is 451, starting from 1.5s (idx=45), valid length = 451-45 = 406
    traj5_valid_length = data.shape[1] - int(1.5 / raw_dt)
    
    # Define trajectory indices for training and test sets
    train_indices = [0, 1, 2, 3]  # New trajectories 1-4
    test_indices = [4, 5]          # New trajectories 5-6
    indices = train_indices + test_indices
    
    # Plot configuration for two phase space types
    titles = ["(y vs x)", "(dy/dt vs dx/dt)"]
    file_suffixes = ["xy", "v_xy"]
    col_pairs = [(2, 0), (3, 1)]  # Column indices: (y, x) for position, (dy, dx) for velocity
    labels = [r'$y$', r'$x$', r'$\dot{y}$', r'$\dot{x}$']
    
    # Generate plots for each phase space type
    for k in range(2):
        fig, ax = plt.subplots(figsize=STYLE.fig_size)
        idx_h, idx_v = col_pairs[k]  # Horizontal and vertical axis column indices
        
        # Plot each trajectory
        for idx in indices:
            traj = data[idx]
            
            # Special handling for trajectory 5 (idx=4): exclude zero-padded tail
            if idx == 4:
                traj = traj[:traj5_valid_length]
            
            # Extract values for horizontal and vertical axes
            h_vals = traj[:, idx_h]
            v_vals = traj[:, idx_v]
            
            # Set color and z-order based on train/test split
            if idx in train_indices:
                color = STYLE.train_color
                zorder = 1  # Training trajectories drawn behind
            else:
                color = STYLE.test_color
                zorder = 2  # Test trajectories drawn on top
            
            # Plot trajectory curve
            ax.plot(h_vals, v_vals, color=color, linewidth=STYLE.method_lw, 
                    alpha=STYLE.method_alpha, zorder=zorder, solid_capstyle="round")
            
            # Mark initial point with colored circle
            h0 = h_vals[0]  # Initial horizontal value
            v0 = v_vals[0]  # Initial vertical value
            
            if idx in train_indices:
                pt_color = STYLE.train_color
                pt_zorder = 10
            else:
                pt_color = STYLE.test_color
                pt_zorder = 11
            
            # Draw initial point marker with white edge
            ax.plot(h0, v0, 'o', color=pt_color, markeredgecolor='white', 
                    markeredgewidth=1.5, markersize=STYLE.marker_size, zorder=pt_zorder)
            
            # Note: Text labels (trajectory numbers 1-6) can be added here if needed
            # Currently disabled for cleaner visualization

        # Apply styling
        ax.grid(False)
        apply_spines(ax)
        apply_ticks(ax)
        
        # Save figure
        plt.tight_layout()
        out_name = f"{args.output_prefix}_{file_suffixes[k]}.svg"
        fig.savefig(out_name, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_name}")
        plt.close(fig)


if __name__ == "__main__":
    main()
