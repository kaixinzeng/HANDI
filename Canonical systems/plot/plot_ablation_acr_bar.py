#!/usr/bin/env python3
import os
import re
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
import pandas as pd
from matplotlib.transforms import blended_transform_factory

size = 15
arial_font = FontProperties(fname="ablation_study/fonts/arial.ttf", size=size)
symbol_font = FontProperties(family='DejaVu Sans', size=size) 
rcParams['font.family'] = arial_font.get_name()
METHOD_ORDER = ['HANDI','ablationA', 'ortho']
COLORS = [ '#F1C77E', '#A4D2F0', '#89D3B2']
# METHOD_COLOR_MAP = {method: COLORS[i % len(COLORS)] for i, method in enumerate(METHOD_ORDER)}


def load_data_multi_dt(root_dir: str) -> Dict[str, Dict[float, Dict[str, float]]]:

    all_data: Dict[str, Dict[float, Dict[str, float]]] = {}
    
    METRIC_COL = 'ACR_Percentage' 
    METHOD_COL = 'Method'

    file_pattern = re.compile(r'(.+)_acr_dt(\d+\.?\d*)\.csv$', re.IGNORECASE)

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            match = file_pattern.match(fname)
            if match:
                full_path = os.path.join(dirpath, fname)
                try:
                    system_name = match.group(1).replace('_', ' ')
                    dt_value = float(match.group(2))
                    
                    df = pd.read_csv(full_path)
                    
                    if METHOD_COL not in df.columns or METRIC_COL not in df.columns:
                        continue
                        
                    if system_name not in all_data:
                        all_data[system_name] = {}
                    
                    if dt_value not in all_data[system_name]:
                        all_data[system_name][dt_value] = {}

                    for index, row in df.iterrows():
                        method_name = row[METHOD_COL]
                        acr_percentage = row[METRIC_COL]
                        
                        if pd.isna(acr_percentage):
                            continue
                            
                        all_data[system_name][dt_value][method_name] = acr_percentage
                        
                except Exception as e:
                    print(f"[WARN] Failed to load {full_path}: {e}")
                    continue

    return all_data

def plot_acr_bar(data: Dict[str, Dict[float, Dict[str, float]]], output_dir: str):

    output_acr_dir = os.path.join(output_dir, 'ACR_Charts_Per_Dt_0118')
    if not os.path.exists(output_acr_dir):
        os.makedirs(output_acr_dir)

    system_names = sorted(data.keys())

    for system_name in system_names:
        dt_data = data[system_name]
        if not dt_data:
            continue

        sorted_dts = sorted(dt_data.keys())
        for dt_value in sorted_dts:
            method_results = dt_data[dt_value]

            plot_data = []
            for method_name in METHOD_ORDER:
                if method_name in method_results:
                    plot_data.append({
                        'method': method_name,
                        'acr': method_results[method_name]
                    })
            
            if not plot_data:
                print(f"[WARN] No method data to plot for {system_name} at dt={dt_value:.2f}.")
                continue
            

            labels = [d['method'] for d in plot_data]
            acr_values = [d['acr'] for d in plot_data]
            colors = COLORS
            num_methods = len(labels)

            fig, ax = plt.subplots(figsize=(4, 4))

            x_positions = np.arange(num_methods)
            bars = ax.bar(x_positions, acr_values, width=0.6, color=colors)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
            ax.tick_params(axis='x', which='major', length=8, width=1.5)
            ax.set_xticks([])

            ax.set_ylim(0.0, 1.10)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            ax.tick_params(axis='y', which='major', labelsize=22, length=8, width=1.5)
            
            data_dict = {'★': 'HANDI','①':'A', '②': 'O'}
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            xtick_labels = list(data_dict.keys())
            for i, label in enumerate(xtick_labels):
                ax.text(
                    x_positions[i], -0.09,
                    label,
                    transform=trans,
                    ha='center', va='center',
                    fontsize=30,
                    fontproperties=symbol_font,
                    color='black' if i == 0 else 'black',
                    clip_on=False
                )

            if acr_values:
                max_acr = max(acr_values)
                for bar in bars:
                    height = bar.get_height()
                    font_weight = 'bold' if height == max_acr else 'normal'
                    
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.2f}',
                            ha='center', va='bottom', fontsize=20, 
                            # fontweight=font_weight
                            )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)
            
            plt.tight_layout()
            
            safe_system_name = system_name.replace(' ', '_')
            plot_fname = f'{safe_system_name}_dt{dt_value:.2f}_ACR'
            plot_path_svg = os.path.join(output_acr_dir, f'{plot_fname}.svg')
            
            plt.savefig(plot_path_svg, bbox_inches='tight', dpi=300)
            
            plt.close(fig)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./ablation_study", help="Root directory to search for ACR result .csv files (e.g., fixed_acr_dt0.1.csv).",)
    parser.add_argument("--output", type=str, default="./ablation_study/plots_bar", help="Directory to save the generated plots",)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    output_dir = args.output

    if not os.path.isdir(root):
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    data = load_data_multi_dt(root)

    if not data:
        print("[WARN] No complete ACR data found for visualization.")
        return

    plot_acr_bar(data, output_dir)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()