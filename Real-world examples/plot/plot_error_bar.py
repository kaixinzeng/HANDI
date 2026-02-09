#!/usr/bin/env python3
import os
import re
import argparse
from typing import Dict, Iterable, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = ['HANDI', 'SINDy', 'SR3', 'PSE']
COLORS = ['#4FA8D5', '#d06569', '#C9A1CB', '#D8A0A7']
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


def _parse_dt_from_name(fname: str) -> Optional[str]:
    """Extract the `dt` value from a filename like results_dt0.01.csv."""
    # match = re.search(r'dt([0-9\.]+)', fname, flags=re.IGNORECASE)
    match = re.search(r'dt(\d+(?:\.\d+)?)', fname, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _iter_err_columns(columns: Iterable[str]) -> List[str]:
    """Return all columns whose names end with 'Err%' (case sensitive)."""
    return [c for c in columns if str(c).strip().endswith('Err%')]


def load_dt_errors(root_dir: str, systems: Iterable[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Collect averaged error (mean of all *_Err% columns) per system/dt/method."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for system_name in systems:
        system_dir = os.path.join(root_dir, system_name)
        if not os.path.isdir(system_dir):
            print(f"[WARN] Skip system '{system_name}': {system_dir} not found")
            continue

        for fname in os.listdir(system_dir):
            if not fname.lower().endswith('.csv') or 'dt' not in fname.lower():
                continue

            dt_label = _parse_dt_from_name(fname)
            if not dt_label:
                print(f"[WARN] Skip {fname}: cannot parse dt value")
                continue

            full_path = os.path.join(system_dir, fname)
            try:
                df = pd.read_csv(full_path)
            except Exception as exc:  # pragma: no cover - defensive read
                print(f"[WARN] Failed to read {full_path}: {exc}")
                continue

            if 'Method' not in df.columns:
                print(f"[WARN] Skip {full_path}: missing 'Method' column")
                continue

            err_cols = _iter_err_columns(df.columns)
            if not err_cols:
                print(f"[WARN] Skip {full_path}: no columns ending with 'Err%'")
                continue

            for _, row in df.iterrows():
                method = normalize_method(str(row['Method']))
                if method not in METHOD_ORDER:
                    continue

                numeric_vals = [pd.to_numeric(row[c], errors='coerce') for c in err_cols]
                finite_vals = [v for v in numeric_vals if pd.notna(v) and np.isfinite(v)]
                if not finite_vals:
                    continue

                mean_err = float(np.mean(finite_vals))
                results.setdefault(system_name, {}).setdefault(dt_label, {})[method] = mean_err

    return results


def _format_value(value: float) -> str:
    if value == 0:
        return '0.00'
    abs_val = abs(value)
    if abs_val >= 1e2 or abs_val <= 1e-2:
        s = f"{value:.1e}"
        s = s.replace('e+0', 'e').replace('e+', 'e')
        s = s.replace('e-0', 'e-')
        # return f"{value:.1e}"
        return s
    if abs_val >= 10 and abs_val <= 100:
        s = f"{value:.1f}"
        # s = s.replace('e+0', 'e').replace('e+', 'e')
        # s = s.replace('e-0', 'e-')
        # return f"{value:.1e}"
        return s
    return f"{value:.2f}"


def _sort_dt_labels(dt_labels: Iterable[str]) -> List[str]:
    """Sort dt labels numerically when possible, otherwise lexicographically."""
    def key(label: str) -> float:
        try:
            return float(label)
        except ValueError:
            return float('inf')

    return sorted(dt_labels, key=key)


def plot_error_bars(
    data: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    yscale: str = "linear",
    symlog_linthresh: float = 1e-3,
) -> None:
    """Plot one figure per system: x-axis=dt, grouped bars for four methods."""
    os.makedirs(output_dir, exist_ok=True)

    for system_name in sorted(data.keys()):
        system_data = data[system_name]
        if not system_data:
            print(f"[WARN] No data for {system_name}, skip plot")
            continue

        dt_labels = _sort_dt_labels(system_data.keys())
        x_positions = np.arange(len(dt_labels))*4
        bar_width = 0.8

        fig, ax = plt.subplots(figsize=(10, 3))
        if yscale == "log":
            ax.set_yscale('log')
        elif yscale == "symlog":
            ax.set_yscale('symlog', linthresh=symlog_linthresh)

        if system_name == "NI":
            ax.set_ylim(0.05, 15)
            ax.set_yticks([1e-1, 1e-0, 1e+1])
        elif system_name == "4d":
            ax.set_ylim(0.3, 150)
            ax.set_yticks([1e-0, 1e+1, 1e+2])
        elif system_name == "pendulum":
            ax.set_ylim(0.3, 1e+4*3)
            ax.set_yticks([1e-0, 1e+2, 1e+4])
        elif system_name == "wheel":
            ax.set_ylim(0.5*1e-2, 3*1e-0)
            ax.set_yticks([1e-2, 1e-1, 1e+0])
        elif system_name == "new_4d":
            ax.set_ylim(0.05, 30)
            ax.set_yticks([1e-1, 1e-0, 1e+1])
        elif system_name == "new_pendulum":
            ax.set_ylim(0.3, 1e+2*3)
            ax.set_yticks([1e-0, 1e+1, 1e+2])

        ymin, ymax = ax.get_ylim()

        for idx, method in enumerate(METHOD_ORDER):
            offsets = x_positions + (idx - (len(METHOD_ORDER)-1) / 2) * bar_width
            values = []
            colors = []
            for dt_label in dt_labels:
                val = system_data.get(dt_label, {}).get(method)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    values.append(np.nan)
                    colors.append(METHOD_COLOR_MAP.get(method, COLORS[0]))
                else:
                    values.append(val)
                    colors.append(METHOD_COLOR_MAP.get(method, COLORS[0]))

            bars = ax.bar(offsets, values, width=bar_width, color=colors, edgecolor='white', linewidth=2, label=method)

            na_bars = []
            value_bars = []
            for bar, val in zip(bars, values):
                if np.isnan(val):
                    na_bars.append((bar, METHOD_COLOR_MAP.get(method, COLORS[0])))
                else:
                    value_bars.append((bar, val))

            for bar, val in value_bars:
                label_y = val
                ax.text(bar.get_x() + bar.get_width() / 2, label_y, _format_value(val), ha='center', va='bottom', fontsize=18)

            na_y = ymin + 0.000 * (ymax - ymin)
            for bar, color in na_bars:
                ax.text(bar.get_x() + bar.get_width() / 2, na_y, 'N/A', ha='center', va='bottom', fontsize=18, color=color,fontweight='bold')

        ax.set_xticks(x_positions)
        ax.set_xticklabels(dt_labels, rotation=0, ha='center')
        ax.set_xlim(x_positions[0] - 2, x_positions[-1] + 2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(axis='x', which='major', labelsize=22, length=4, width=1.5)
        ax.tick_params(axis='y', which='major', labelsize=22, length=4, width=1.5)
        ax.tick_params(axis='y', which='minor', length=0)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plot_name = f"{system_name}_error_by_dt.svg"
        plt.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="./id_true_error",
        help="Root directory containing subfolders with metrics.csv files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./id_true_error/plots_bar",
        help="Directory to save generated bar charts (one per system).",
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
    parser.add_argument(
        "--systems",
        nargs="*",
        default=["NI",'new_4d','new_pendulum','wheel'],
        help="System subfolders to include.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    data = load_dt_errors(root, args.systems)
    if not data:
        print("[WARN] No dt csv files found or no usable data.")
        return

    plot_error_bars(data, args.output, yscale=args.yscale, symlog_linthresh=args.symlog_linthresh)
    print("Bar charts generated in", os.path.abspath(args.output))


if __name__ == "__main__":
    main()