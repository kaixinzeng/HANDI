#!/usr/bin/env python3
import os
import re
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def unsanitize_name(filename: str) -> str:
    name = re.sub(r'_(sr|ausc)\.npy$', '', filename)
    return name.replace('_', ' ')

def load_data(root_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Recursively find and load all SR curve and AUSC data.
    
    Returns:
        {system_name: {method_name: {'sr': np.ndarray, 'ausc': float}}}
    """
    all_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for dirpath, _, filenames in os.walk(root_dir):

        system_name = os.path.basename(dirpath)

        if system_name == os.path.basename(root_dir) and dirpath == root_dir:
            continue
            
        for fname in filenames:
            if fname.lower().endswith(("_sr.npy", "_ausc.npy")):
                full_path = os.path.join(dirpath, fname)
                try:
                    data = np.load(full_path, allow_pickle=False)
                except Exception as e:
                    print(f"[WARN] Failed to load {full_path}: {e}")
                    continue

                if fname.endswith("_sr.npy"):
                    file_type = 'sr'
                elif fname.endswith("_ausc.npy"):
                    file_type = 'ausc'
                else:
                    continue 

                method_name = unsanitize_name(fname)

                if system_name not in all_data:
                    all_data[system_name] = {}
                if method_name not in all_data[system_name]:
                    all_data[system_name][method_name] = {'sr': None, 'ausc': None}

                if file_type == 'ausc':
                     data = data.item()
                
                all_data[system_name][method_name][file_type] = data

    systems_to_remove: List[str] = []
    for sys_name, methods in all_data.items():
        methods_to_remove: List[str] = []
        for meth_name, res in methods.items():
            if res['sr'] is None or res['ausc'] is None:
                if not isinstance(res['ausc'], (float, int)):
                     print(f"[WARN] Skipping {sys_name}/{meth_name}: AUSC data invalid.")
                     methods_to_remove.append(meth_name)
                     continue
                
                print(f"[WARN] Skipping {sys_name}/{meth_name}: Missing SR or AUSC data.")
                methods_to_remove.append(meth_name)
        
        for meth_name in methods_to_remove:
            del methods[meth_name]
        
        if not methods:
            systems_to_remove.append(sys_name)
            
    for sys_name in systems_to_remove:
        del all_data[sys_name]

    return all_data

def plot_ausc_bar(data: Dict[str, Dict[str, Dict[str, Any]]], output_dir: str):

    output_ausc_dir = os.path.join(output_dir, 'new_AUSC_noxlabel_Charts')
    if not os.path.exists(output_ausc_dir):
        os.makedirs(output_ausc_dir)

    system_names = sorted(data.keys())

    colors = [ '#50AAD8', '#d06569', '#C9A1CB','#D8A0A7','#F4B36B','#528FBF','#4AB4B2']

    for system_name in system_names:
        methods = data[system_name]
        if not methods:
            continue
        
        bar_data = []
        for method_name, results in methods.items():
            bar_data.append({
                'method': method_name,
                'ausc': results['ausc']
            })

        METHOD_ORDER = ['EDMD', 'gEDMD', 'wSINDy','PSE', 'SR3','SINDy','HANDI' ]
        bar_data.sort(key=lambda x: METHOD_ORDER.index(x['method']))

        
        labels = [d['method'] for d in bar_data]
        ausc_values = [d['ausc'] for d in bar_data]
        num_methods = len(labels)

        fig, ax = plt.subplots(figsize=(4, 6))
        y_positions = np.arange(num_methods) * 0.7
        bars = ax.barh(y_positions, ausc_values, height=0.4, color=list(reversed(colors)))
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.tick_params(axis='x', which='major', labelsize=16, length=8, width=2)
        ax.set_xlim(0.0, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])

        best_ausc = max(ausc_values)
        for j, bar in enumerate(bars):
            width = bar.get_width()
            font_weight = 'bold' if width == best_ausc else 'normal'
            
            ax.text(
                width + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                ha='left', va='center', 
                fontsize=16, 
                # fontweight=font_weight
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        plt.tight_layout() 
        safe_system_name = system_name.replace(' ', '_')
        plot_path = os.path.join(output_ausc_dir, f'{safe_system_name}_AUSC.svg')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"  -> Saved individual AUSC chart for {system_name} to: {plot_path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./mse_ausc_sr", help="Root directory to search for .npy result files (default: current directory).",)
    parser.add_argument("--output", type=str, default="./mse_ausc_sr/plots_bar", help="Directory to save the generated plots",)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    output_dir = args.output

    if not os.path.isdir(root):
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    print(f"[INFO] Searching for .npy files under: {root}")
    data = load_data(root)

    if not data:
        print("[WARN] No complete data (SR and AUSC) found for visualization.")
        return

    plot_ausc_bar(data, output_dir)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()