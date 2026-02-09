from matplotlib import ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from matplotlib.lines import Line2D

CONFIG = {
    'base_path': r"test_benchmark/test",
    'output_path': r"test_benchmark/test/plots_boxplot/id_mse",
    'system_ids': ['id6', 'id9', 'id31', 'id39'],
    # 'system_ids': ['id6'],
    'noise_levels': [1, 2, 3, 4, 5],
    'colors': {
        'EDMD': '#4AB4B2', 'gEDMD': '#528FBF','wSINDy': '#F4B36B','PSE': '#D8A0A7',
        'SR3': '#C9A1CB','SINDy': '#d06569', 'HANDI': '#50AAD8',
    },
    'font_config': {'family': 'sans-serif', 'size': 10},
}

if not os.path.exists(CONFIG['output_path']):
    os.makedirs(CONFIG['output_path'])

method_order = list(CONFIG['colors'].keys())

def process_system_data(sys_id):
    mse_data = {m: [] for m in method_order}
    success_rate_matrix = pd.DataFrame(np.nan, index=method_order, columns=CONFIG['noise_levels'])
    valid_methods = set()

    for noise in CONFIG['noise_levels']:
        pattern = os.path.join(CONFIG['base_path'], sys_id, f"{noise}%", "*mse_all*.csv")
        search = glob.glob(pattern)
        file_path = search[0] if search else None

        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                df.replace([np.inf, -np.inf], np.nan, inplace=True)

                for method in method_order:
                    if method in df.columns:
                        col_data = df[method]
                        total_runs = len(col_data)
                        valid_runs = col_data.count()

                        rate = valid_runs / total_runs if total_runs > 0 else 0
                        success_rate_matrix.loc[method, noise] = rate

                        if valid_runs > 0:
                            mse_data[method].append(col_data.mean())
                            valid_methods.add(method)
                        else:
                            mse_data[method].append(np.nan)
                    else:
                        mse_data[method].append(np.nan)
                        success_rate_matrix.loc[method, noise] = 0.0
            except Exception:
                for m in mse_data: mse_data[m].append(np.nan)
        else:
            for m in mse_data: mse_data[m].append(np.nan)
            success_rate_matrix[noise] = 0.0

    return mse_data, success_rate_matrix, valid_methods

def plot_integrated_mse(sys_id, mse_data, success_matrix, valid_methods):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    for method in method_order:
        if method not in valid_methods:
            continue

        y_values = np.array(mse_data[method])
        color = CONFIG['colors'][method]
        ax.plot(CONFIG['noise_levels'], y_values, color=color, alpha=1.0, linewidth=3, zorder=1)

        for i, noise in enumerate(CONFIG['noise_levels']):
            val = y_values[i]
            rate = success_matrix.loc[method, noise]

            if np.isnan(val):
                continue

            if rate >= 0.99:
                ax.plot(noise, val, marker='o', markersize=10,
                        color=color, fillstyle='full',
                        markeredgecolor=color, zorder=2)
            elif rate >= 0.4:
                ax.plot(noise, val, marker='o', markersize=10,
                        color=color, fillstyle='left',
                        markeredgecolor=color,
                        markerfacecoloralt='white',
                        zorder=2)

    ax.tick_params(axis='x', which='major', labelsize=22, length=8, width=1.5)
    ax.set_yscale('log')
    
    if sys_id == 'id6':
        ax.set_ylim(3*1e-6, 4*1e+3)
        ax.set_yticks([1e-5,1e-1,1e+3])

    if sys_id == 'id9':
        ax.set_ylim(3*1e-6, 4*1e+3)
        ax.set_yticks([1e-5,1e-1,1e+3]) 
    
    if sys_id == 'id31':
        ax.set_ylim(2*1e-2, 4*1e+3)
        ax.set_yticks([1e-1,1e+1,1e+3])

    if sys_id == 'id39':
        ax.set_ylim(1e-6, 4*1e+3)
        ax.set_yticks([1e-5,1e-1,1e+3])

    ax.tick_params(axis='y', which='minor', length=0)
    ax.tick_params(axis='y', which='major', labelsize=22, length=8, width=1.5)


    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

    plt.tight_layout()
    save_path = os.path.join(CONFIG['output_path'], f"{sys_id}.svg")
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")

if __name__ == "__main__":
    for sys_id in CONFIG['system_ids']:
        print(f"Processing {sys_id}...")
        mse, success_rate, valid = process_system_data(sys_id)
        plot_integrated_mse(sys_id, mse, success_rate, valid)