#!/usr/bin/env python3
import os
import re
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator, LogFormatterMathtext

METHOD_ORDER = ['HANDI', 'SINDy', 'SR3', 'PSE'] 
fill_colors = [ '#4FA8D5', '#d06569', '#C9A1CB','#D8A0A7']
edge_colors = ['#2C6DA1', '#8E2F34', '#7D4F87', '#8C4B54']

def load_data_multi_dt(root_dir: str) -> Dict[str, Dict[float, Dict[str, pd.Series]]]:

    all_data: Dict[str, Dict[float, Dict[str, pd.Series]]] = {}

    file_pattern = re.compile(r'(.+)_mse_all_dt(\d+\.?\d*)_xyrange(\d+\.?\d*)_grid(\d+)\.csv$', re.IGNORECASE)

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            match = file_pattern.match(fname)
            if match:
                full_path = os.path.join(dirpath, fname)
                try:

                    system_name_raw = match.group(1)
                    system_name = system_name_raw.replace('_', ' ')
                    dt_value = float(match.group(2))
                    
                    df = pd.read_csv(full_path)
                    df_cleaned = df.replace(np.inf, np.nan)
                    
                    if system_name not in all_data:
                        all_data[system_name] = {}
                    
                    if dt_value not in all_data[system_name]:
                        all_data[system_name][dt_value] = {}

                    for method_name in METHOD_ORDER:
                        if method_name in df_cleaned.columns:
                            valid_data = df_cleaned[method_name].dropna()
                            if not valid_data.empty:
                                all_data[system_name][dt_value][method_name] = valid_data
                            
                except Exception as e:
                    print(f"[WARN] Failed to load {full_path}: {e}")
                    continue

    return all_data


def plot_mse_boxplot_per_dt(data: Dict[str, Dict[float, Dict[str, pd.Series]]], output_dir: str):

    output_mse_dir = os.path.join(output_dir, 'MSE_Boxplots_Per_Dt')
    if not os.path.exists(output_mse_dir):
        os.makedirs(output_mse_dir)

    system_names = sorted(data.keys())

    for system_name in system_names:
        dt_data = data[system_name]
        if not dt_data:
            continue

        sorted_dts = sorted(dt_data.keys())

        for dt_value in sorted_dts:
            method_results = dt_data[dt_value]

            plot_methods = [m for m in METHOD_ORDER if m in method_results]
            plot_data = [method_results[m] for m in plot_methods]

            if not plot_data:
                print(f"[WARN] No valid MSE data to plot for {system_name} at dt={dt_value:.2f}.")
                continue
                
            fig, ax = plt.subplots(figsize=(4, 4))

            box_plot = ax.boxplot(
                plot_data,
                widths=0.6,
                patch_artist=True,
                showfliers=False,
            )
            
            LINE_WIDTH = 1.5
            for patch, fillcolor, edgecolor in zip(box_plot['boxes'], fill_colors, edge_colors):
                patch.set_facecolor(fillcolor)
                patch.set_edgecolor(edgecolor)
                patch.set_linewidth(LINE_WIDTH)
                # patch.set_alpha(0.8)
            for whisker, color in zip(box_plot['whiskers'], np.repeat(edge_colors, 2)):
                whisker.set(color=color, linewidth=LINE_WIDTH)
            for cap, color in zip(box_plot['caps'], np.repeat(edge_colors, 2)):
                cap.set(color=color,linewidth=LINE_WIDTH)
            for median, color in zip(box_plot['medians'], edge_colors):
                median.set(color=color, linewidth=LINE_WIDTH + 0.5)
            for flier, color in zip(box_plot['fliers'], edge_colors):
                flier.set_marker('o')
                flier.set_markersize(4.5)
                flier.set_markerfacecolor(color)
                flier.set_markeredgecolor('#444444')
                flier.set_markeredgewidth(1.1)


            ax.set_xticklabels(plot_methods, rotation=45, ha='right', fontsize=20)
            ax.tick_params(axis='x', which='major', length=8, width=1.5)

            ax.set_yscale('log')

            # duffing
            ax.set_ylim(1e-8, 50)
            ax.set_yticks([1e-7,1e-3,1e+1])

            # vdp
            # ax.set_ylim(1e-6, 0.5*1e+4)
            # ax.set_yticks([1e-5,1e-1,1e+3])

            # fixed       
            # ax.set_ylim(1e-9, 10)
            # ax.set_yticks([1e-8, 1e-4, 1e-0])

            # cyc
            # ax.set_ylim(1e-8, 5)
            # ax.set_yticks([1e-7,1e-4,1e-1])  
          
            # ax.set_ylim(1e-5, 5)
            # ax.set_yticks([1e-4,1e-2,1e-0])

            ax.tick_params(axis='y', which='minor', length=0)
            ax.tick_params(axis='y', which='major', labelsize=20, length=8, width=1.5)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)

            plt.tight_layout()
            safe_system_name = system_name.replace(' ', '_')
            plot_fname = f'{safe_system_name}_dt{dt_value:.2f}_MSE_Boxplot'
            plot_path_svg = os.path.join(output_mse_dir, f'{plot_fname}.svg')
            plt.savefig(plot_path_svg, bbox_inches='tight', dpi=300)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate individual MSE Boxplots per sampling interval (dt).")
    parser.add_argument("--root", type=str, default="./nrmse_mse_r2/duffing", help="Root directory to search for MSE result .csv files (e.g., system_mse_all_dt0.1_xyrange1.0_grid10.csv).")
    parser.add_argument("--output", type=str, default="./nrmse_mse_r2/plots_boxplot", help="Directory to save the generated boxplots.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    output_dir = args.output

    if not os.path.isdir(root):
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    print(f"Searching for MSE files in: {root}")
    data = load_data_multi_dt(root)

    if not data:
        print("[WARN] No complete MSE data found for visualization.")
        return

    print(f"Generating Boxplots and saving to: {os.path.join(output_dir, 'MSE_Boxplots_Per_Dt')}")
    plot_mse_boxplot_per_dt(data, output_dir)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()