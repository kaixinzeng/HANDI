#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot reordered test set x1 time series comparison (New 5, 6)
and Phase Portraits (x1-x2, x3-x1) for all 6 trajectories.
Refined based on plot_results.py and mofang.PY styles.
"""

from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
from scipy.integrate import solve_ivp
import os

class Style:
    fig_w = 12
    fig_h_per_subplot = 4
    pred_lw = 5
    true_lw = 7
    tick_fontsize = 46
    label_fontsize = 32
    title_fontsize = 28
    legend_fontsize = 24
    spine_width = 3.0
    spine_color = "black"
    show_spines = ("left", "bottom")

    true_color = "#BBBBBB"
    method_colors = {
        "HANDI": "#50AAD8",
        "SINDy": "#d06569",
        "SR3": '#C9A1CB',
        "PSE": "#D8A0A7"
    }

STYLE = Style()

# Phase Portrait specific style (cloned from plot_results.py)
P_STYLE = {
    "fig_size": (10, 10),
    "lw": 4.0,
    "marker_size": 36,
    "train_color": "#B0B0B0", # Grey for 1-4
    "test_color": "#414592",  # Dark Blue for 5-6
    "tick_fontsize": 56,
    "spine_width": 4
}

def parse_equations(eq_file):
    with open(eq_file, 'r', encoding='utf-8') as f:
        content = f.read()
    rhs_list = {}
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or 'dx' not in line: continue
        m = re.match(r"dx\(?(\d+)\)?/dt\s*=\s*(.*)", line)
        if m:
            idx = int(m.group(1)) - 1
            rhs = m.group(2)
            rhs = re.sub(r"x\((\d+)\)", lambda m: f"x[{int(m.group(1))-1}]", rhs)
            rhs_list[idx] = rhs
    dim = len(rhs_list)
    funcs = []
    for i in range(dim):
        rhs = rhs_list.get(i, "0")
        code = compile(rhs, f"eq_{i}", "eval")
        funcs.append(code)
    def system_func(t, x):
        return [eval(code, {}, {"x": x, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt}) for code in funcs]
    return system_func

def integrate_trajectory(func, x0, t_eval):
    t_span = (t_eval[0], t_eval[-1])
    try:
        sol = solve_ivp(func, t_span, x0, t_eval=t_eval, method='Radau', rtol=1e-3, atol=1e-6)
        if sol.success: return sol.y.T, sol.t
        else:
            if hasattr(sol, 'y'): return sol.y.T, sol.t
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def apply_spines(ax, spine_w=STYLE.spine_width):
    for side in ("left", "right", "top", "bottom"): ax.spines[side].set_visible(False)
    for side in STYLE.show_spines:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(spine_w)
        ax.spines[side].set_color(STYLE.spine_color)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

def apply_ticks(ax, fontsize=STYLE.tick_fontsize, spine_w=STYLE.spine_width):
    ax.tick_params(labelsize=fontsize, width=spine_w, direction="out", top=False, right=False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

def plot_phase_portrait(data_raw, ode_func_handi, out_dir):
    """
    Plot Phase Portraits: x1-x3 and x2-x4 for all trajectories (1-6)
    Matched to plot_results.py dimensions and styling.
    """
    raw_dt = 0.03333333
    num_trajs = data_raw.shape[0]
    traj5_valid_length = data_raw.shape[1] - int(1.5 / raw_dt)
    
    # Swapped axes per user request:
    # xy   -> (2, 0) -> x=x3, y=x1
    # v_xy -> (3, 1) -> x=x4, y=x2
    plot_configs = [
        (2, 0, "x3", "x1", "phase_xy"),
        (3, 1, "x4", "x2", "phase_v_xy")
    ]
    
    for dim_x, dim_y, name_x, name_y, file_suffix in plot_configs:
        fig, ax = plt.subplots(figsize=P_STYLE["fig_size"])
        
        for i in range(num_trajs):
            traj_true = data_raw[i]
            if i == 4: traj_true = traj_true[:traj5_valid_length]
            
            x_true = traj_true[:, dim_x]
            y_true = traj_true[:, dim_y]
            
            # Determine Color per plot_results.py
            color = P_STYLE["train_color"] if i < 4 else P_STYLE["test_color"]
            zorder = 1 if i < 4 else 2
            
            # 1. Plot Ground Truth
            ax.plot(x_true, y_true, color=color, linewidth=P_STYLE["lw"], 
                    alpha=1.0, zorder=zorder, solid_capstyle="round")
            
            # 2. For test trajectories (5, 6), plot HANDI prediction (Same color, slightly thicker)
            if i >= 4:
                t_eval = np.arange(len(traj_true)) * raw_dt
                x0 = traj_true[0, :]
                pred_y, _ = integrate_trajectory(ode_func_handi, x0, t_eval)
                if pred_y is not None:
                    ax.plot(pred_y[:, dim_x], pred_y[:, dim_y], 
                            color=color, 
                            linewidth=P_STYLE["lw"] + 1, alpha=0.9, zorder=zorder+1)
            
            # 3. Mark Initial Point (Filled circle with border, centered number)
            ax.plot(x_true[0], y_true[0], 'o', color=color, 
                    markeredgecolor='white', markeredgewidth=1.5, 
                    markersize=P_STYLE["marker_size"], zorder=zorder+10)
            
            # Number centered with path effects
            from matplotlib import patheffects
            txt = ax.text(x_true[0], y_true[0], f"{i+1}", fontsize=28, fontweight='bold', 
                          ha='center', va='center', color='black', zorder=zorder+11)
            txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])

        ax.set_xlabel(name_x, fontsize=STYLE.label_fontsize)
        ax.set_ylabel(name_y, fontsize=STYLE.label_fontsize)
        apply_spines(ax, spine_w=P_STYLE["spine_width"])
        apply_ticks(ax, fontsize=P_STYLE["tick_fontsize"], spine_w=P_STYLE["spine_width"])
        ax.grid(False)
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"figure5_flag_{file_suffix}.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved Phase Portrait: {out_path}")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--dt", type=float, default=0.2, help="Sampling period")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--HANDI_eq_file", type=str, default="")
    parser.add_argument("--SINDy_eq_file", type=str, default="")
    parser.add_argument("--SR3_eq_file", type=str, default="")
    parser.add_argument("--PSE_eq_file", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    dt_str = f"{args.dt:g}"
    if not args.data_path: args.data_path = os.path.join(script_dir, "..", "data", "flag_trajectories_6traj.npy")
    if not args.HANDI_eq_file: args.HANDI_eq_file = os.path.join(script_dir, "..", "results", f"HANDI_flag_dt{dt_str}.txt")
    if not args.SINDy_eq_file: args.SINDy_eq_file = os.path.join(script_dir, "..", "data", f"SINDy_flag_dt{dt_str}.txt")
    if not args.SR3_eq_file: args.SR3_eq_file = os.path.join(script_dir, "..", "data", f"SR3_flag_dt{dt_str}.txt")
    if not args.PSE_eq_file: args.PSE_eq_file = os.path.join(script_dir, "..", "data", f"PSE_flag_dt{dt_str}.txt")
    if not args.output: args.output = os.path.join(script_dir, "..", "results", f"final_test_compare_flag_dt{dt_str}")
    
    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} not found.")
        return

    data = np.load(args.data_path)
    test_indices = [4, 5]
    labels = ["Traj 5", "Traj 6"]
    raw_dt = 0.03333333
    
    ode_func_handi = parse_equations(args.HANDI_eq_file)
    ode_func_sindy = parse_equations(args.SINDy_eq_file)
    ode_func_sr3 = parse_equations(args.SR3_eq_file)
    ode_func_pse = parse_equations(args.PSE_eq_file)
    
    # 1. Generate Refined Phase Portraits
    print("\nGenerating Refined Phase Portraits...")
    out_dir_final = os.path.dirname(args.output) if args.output else os.path.join(script_dir, "..", "results")
    plot_phase_portrait(data, ode_func_handi, out_dir_final)

    # 2. Time Series Comparison
    traj5_valid_length = data.shape[1] - int(1.5 / raw_dt)
    dims = [(0, "x1"), (1, "x2")]
    for dim_idx, dim_name in dims:
        for idx, name in zip(test_indices, labels):
            fig, ax = plt.subplots(1, 1, figsize=(STYLE.fig_w, STYLE.fig_h_per_subplot))
            
            if idx == 4:
                traj_true_target = data[idx][:traj5_valid_length]
            else:
                traj_true_target = data[idx]
            
            t_eval_full = np.arange(len(traj_true_target)) * raw_dt
            x0 = traj_true_target[0, :]
            
            ax.plot(t_eval_full, traj_true_target[:, dim_idx], color=STYLE.true_color, linewidth=STYLE.true_lw, alpha=0.8, zorder=1)

            for method_name, ode_func in (("PSE", ode_func_pse), ("SR3", ode_func_sr3), ("SINDy", ode_func_sindy), ("HANDI", ode_func_handi)):
                pred_y, pred_t = integrate_trajectory(ode_func, x0, t_eval_full)
                if pred_y is not None:
                    ax.plot(pred_t, pred_y[:, dim_idx], color=STYLE.method_colors[method_name], 
                            linewidth=STYLE.pred_lw, alpha=0.95, zorder=10)
        
            if dim_name == "x2" and idx == 4:
                ax.set_ylim(-1.5, 4.5)

            apply_spines(ax)
            apply_ticks(ax)
            ax.grid(False)
            ax.set_xlim(0, 12)
            ax.set_xlabel('Time (s)', fontsize=STYLE.label_fontsize)
            ax.set_ylabel(dim_name, fontsize=STYLE.label_fontsize)

            plt.tight_layout()
            traj_label = 'traj5' if idx == 4 else 'traj6'
            output_path = os.path.join(out_dir_final, f"figure5_flag_{traj_label}_{dim_name}.pdf")
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {output_path}")
            plt.close(fig)

if __name__ == "__main__":
    main()
