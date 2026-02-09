#!/usr/bin/env python3
import os
import argparse
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = ['HANDI', 'SINDy', 'SR3', 'PSE', 'WSINDy', 'gEDMD', 'EDMD']
COLORS = ['#4FA8D5', '#d06569', '#C9A1CB', '#D8A0A7', '#F4B36B', '#528FBF', '#4AB4B2']
MISSING_COLOR = '#E0E0E0'
METHOD_COLOR_MAP = {method: COLORS[idx % len(COLORS)] for idx, method in enumerate(METHOD_ORDER)}


def normalize_method(name: str) -> str:
    """Normalize method naming so csv variations map into a fixed order."""
    key = name.strip().lower()
    mapping = {
        'sindy': 'SINDy',
        'sr3': 'SR3',
        'wsindy': 'WSINDy',
        'pse': 'PSE',
        'handi': 'HANDI',
        'gedmd': 'gEDMD',
        'edmd': 'EDMD',
    }
    return mapping.get(key, name.strip())


def load_variable_errors(root_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Read every metrics.csv and collect error percentages per variable and method."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower() != 'metrics.csv':
                continue

            full_path = os.path.join(dirpath, fname)
            system_name = os.path.basename(dirpath)

            try:
                df = pd.read_csv(full_path)
            except Exception as exc:  # pragma: no cover - defensive read
                print(f"[WARN] Failed to read {full_path}: {exc}")
                continue

            if 'Method' not in df.columns:
                print(f"[WARN] Skip {full_path}: missing 'Method' column")
                continue

            error_columns = [c for c in df.columns if c.endswith('_Err%')]
            if not error_columns:
                print(f"[WARN] Skip {full_path}: no *_Err% columns found")
                continue

            for _, row in df.iterrows():
                method = normalize_method(str(row['Method']))

                for err_col in error_columns:
                    variable = err_col[:-5]  # strip trailing '_Err%'
                    value = row[err_col]
                    if pd.isna(value):
                        continue

                    results.setdefault(system_name, {}).setdefault(variable, {})[method] = float(value)

    return results


def _format_value(value: float) -> str:
    if value == 0:
        return '0.00'
    abs_val = abs(value)
    if abs_val >= 1e2 or abs_val < 1e-2:
        s = f"{value:.1e}"      # e+04
        s = s.replace('e+0', 'e').replace('e+', 'e')
        s = s.replace('e-0', 'e-')
        # return f"{value:.1e}"
        return s
    return f"{value:.2f}"


def plot_error_bars(
    data: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    yscale: str = "linear",
    symlog_linthresh: float = 1e-3,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for system_name in sorted(data.keys()):
        system_dir = os.path.join(output_dir, system_name)
        os.makedirs(system_dir, exist_ok=True)

        variables = data[system_name]
        for variable_name in sorted(variables.keys()):
            method_values = variables[variable_name]

            values = []
            colors = []
            for method in METHOD_ORDER:
                val = method_values.get(method)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    values.append(np.nan)
                    colors.append(MISSING_COLOR)
                else:
                    values.append(val)
                    colors.append(METHOD_COLOR_MAP.get(method, COLORS[0]))

            if all(np.isnan(values)):
                print(f"[WARN] No numeric data for {system_name}-{variable_name}, skip plot")
                continue
            sanitized_values = []

            for v in values:
                if np.isnan(v):
                    sanitized_values.append(np.nan)
                else:
                    sanitized_values.append(v)

            x_positions = np.arange(len(METHOD_ORDER))
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(
                x_positions,
                sanitized_values,
                width=0.60,
                color=colors,
                # edgecolor='#444444',
                # linewidth=0.8,
            )

            if yscale == "log":
                ax.set_yscale('log')
            elif yscale == "symlog":
                ax.set_yscale('symlog', linthresh=symlog_linthresh)

            ax.set_xticks(x_positions)
            ax.tick_params(axis='x', which='major', labelsize=18, length=5, width=1.5)
            ax.set_xticklabels(METHOD_ORDER, rotation=35, ha='right')

            valid_values = [v for v in sanitized_values if not np.isnan(v)]
            if not valid_values:
                print(f"[WARN] No valid numeric data for {system_name}-{variable_name}, skip plot")
                plt.close(fig)
                continue


            if variable_name == 'alpha': # id6
                ax.set_ylim(min(v for v in valid_values if v > 0)*0.8, 1e+5)
                ax.set_yticks([1e+0,1e+2,1e+4])
            elif variable_name == 'beta': # id6
                ax.set_ylim(min(v for v in valid_values if v > 0)*0.8, 1e+6)
                ax.set_yticks([1e+1,1e+3,1e+5])

            elif variable_name == 'c0': # id9
                ax.set_ylim(min(v for v in valid_values if v > 0)*0.8, 1e+8)
                ax.set_yticks([1e-1,1e+3,1e+7])
            elif variable_name == 'c1': # id9
                ax.set_ylim(min(v for v in valid_values if v > 0)*0.8, 1e+10)
                ax.set_yticks([1e-1,1e+4,1e+9])

            elif variable_name == 'k': # id31
                ax.set_ylim(5, 0.5*1e+4)
                ax.set_yticks([1e+1,1e+2,1e+3])
            elif variable_name == 'gamma': # id31
                ax.set_ylim(0.01, 1e+4)
                ax.set_yticks([1e-1,1e+1,1e+3])

            elif variable_name == 'a': # id39
                ax.set_ylim(min(v for v in valid_values if v > 0)*0.8, 0.3*1e+4)
                ax.set_yticks([1e-1,1e+1,1e+3])
            elif variable_name == 'b': # id39
                ax.set_ylim(0.005, 500)
                ax.set_yticks([1e-2,1e+0,1e+2])

            for idx, bar in enumerate(bars):
                height = bar.get_height()
                display_val = sanitized_values[idx]
                if np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width() / 2, ax.get_ylim()[1], 'N/A',
                            ha='center', va='bottom', fontsize=9, color='#666666')
                    bar.set_edgecolor('#BBBBBB')
                    continue

                label_y = height * 1.02 if height > 0 else ax.get_ylim()[0] * 1.05
                ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                        _format_value(display_val), ha='center', va='bottom', fontsize=16)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.tick_params(axis='y', which='major', labelsize=18, length=5, width=1.5)
            ax.tick_params(axis='y', which='minor', length=0)

            plt.tight_layout()
            plot_name = f"{system_name}_{variable_name}_error_bar.svg"
            plt.savefig(os.path.join(system_dir, plot_name), dpi=300, bbox_inches='tight')
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="./id_true_error/id31",
        help="Root directory containing subfolders with metrics.csv files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./id_true_error/plots_bar",
        help="Directory to save generated bar charts (one per variable).",
    )
    parser.add_argument(
        "--yscale",
        choices=["linear", "log", "symlog"],
        default="log",
        help="Y-axis scale for error bars.",
    )
    parser.add_argument(
        "--symlog-linthresh",
        type=float,
        default=1e-3,
        help="Linear range threshold when using symlog scale.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    data = load_variable_errors(root)
    if not data:
        print("[WARN] No metrics.csv files found or no usable data.")
        return

    plot_error_bars(data, args.output, yscale=args.yscale, symlog_linthresh=args.symlog_linthresh)
    print("Bar charts generated in", os.path.abspath(args.output))


if __name__ == "__main__":
    main()