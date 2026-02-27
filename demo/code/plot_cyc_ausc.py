#!/usr/bin/env python3
import os
import csv
import math
import argparse
import re
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DELTA_MAX = 1.0

def sanitize_name(name: str) -> str:
    """Return a file-safe name from a method name."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name.strip())


def read_mse_csv(path: str) -> Tuple[List[str], Dict[str, List[float]]]:
    """
    Read a CSV whose first row is method names and subsequent rows are MSE values.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    header : list of str
        Method names in the order they appear as columns.
    data : dict
        Mapping from method name to list of MSE values (floats).
        Non-finite entries (NaN) are converted to +inf so that they are always failures.
    """
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file appears to be empty: {path}")

        header = [h.strip() for h in header if h.strip() != ""]
        data: Dict[str, List[float]] = {h: [] for h in header}

        for row_idx, row in enumerate(reader, start=2):
            # Skip completely empty lines
            if not row or all(cell.strip() == "" for cell in row):
                continue

            # Only use as many cells as there are headers
            for h, cell in zip(header, row):
                cell = cell.strip()
                if cell == "":
                    val = float("nan")
                else:
                    try:
                        val = float(cell)
                    except ValueError:
                        # Common textual representations of infinities
                        lc = cell.lower()
                        if lc in ("inf", "+inf", "infinity", "+infinity"):
                            val = float("inf")
                        elif lc in ("-inf", "-infinity"):
                            val = float("inf")  # treat -inf as "very bad"
                        else:
                            # Fallback: treat as missing
                            val = float("nan")

                # Treat NaN as +inf so they are always failures but keep sample count
                if math.isnan(val):
                    val = float("inf")

                data[h].append(val)

    return header, data


def compute_ausc(mse: np.ndarray, delta_max: float) -> float:
    """
    Compute AUSC for one method using the exact step-function integral.

    Parameters
    ----------
    mse : np.ndarray
        1D array of trajectory-level MSE values for this method.
        Entries may be finite or +inf. Negative values, if any, are clipped to 0.
    delta_max : float
        Upper limit of delta for the integral (here fixed to 1.0 in the caller).

    Returns
    -------
    ausc : float
        Area under the success-rate curve over [0, delta_max], normalized by delta_max.
        Always in [0, 1]. Returns NaN if mse is empty.
    """
    mse = np.asarray(mse, dtype=float)
    n = mse.size
    if n == 0:
        return float("nan")

    # MSE should be non-negative; negative values, if any, are clipped.
    mse = np.clip(mse, 0.0, np.inf)

    # Sort MSE values; +inf naturally go to the end.
    m_sorted = np.sort(mse)

    # Extend with 0 at the left and delta_max at the right,
    # and clip values above delta_max for integration.
    extended = np.concatenate(([0.0], np.minimum(m_sorted, delta_max), [delta_max]))

    area = 0.0
    for k in range(0, n + 1):
        left = extended[k]
        right = extended[k + 1]
        # Length of this interval on the delta axis
        length = max(0.0, right - left)
        # Success rate is k / n on [m_(k), m_(k+1))
        sr = k / n
        area += sr * length

    if delta_max > 0.0:
        ausc = area / delta_max
    else:
        # Degenerate case: all MSE = 0 so delta_max = 0; define AUSC = 1.
        ausc = 1.0

    # Numerical clipping to [0, 1]
    ausc = max(0.0, min(1.0, ausc))
    return ausc


def process_csv(csv_path: str) -> Dict[str, float]:
    """Return AUSC per method for one CSV (no files written)."""
    header, data = read_mse_csv(csv_path)

    print(f"[INFO] Processing CSV: {csv_path}")
    print(f"       Methods: {', '.join(header)}")
    print(f"       delta_max is FIXED to {DELTA_MAX:.6g} for all methods")

    ausc_by_method: Dict[str, float] = {}
    for method_name in header:
        mse_values = np.asarray(data[method_name], dtype=float)
        ausc_val = compute_ausc(mse_values, DELTA_MAX)
        ausc_by_method[method_name] = ausc_val
        print(f"       AUSC for '{method_name}': {ausc_val:.6f}")

    return ausc_by_method


def plot_ausc_bar(data: Dict[str, Dict[str, float]], output_dir: str, dt_label: str) -> None:
    """Create one AUSC bar chart per system without saving intermediate npy files."""
    output_ausc_dir = os.path.join(output_dir)
    os.makedirs(output_ausc_dir, exist_ok=True)

    system_names = sorted(data.keys())
    colors = ["#4AB4B2", "#528FBF", "#F4B36B", "#D8A0A7", "#C9A1CB","#d06569","#50AAD8"]
    method_order = ["EDMD", "gEDMD", "wSINDy", "PSE", "SR3", "SINDy", "HANDI"]
    safe_dt = sanitize_name(dt_label) if dt_label else "all"

    for system_name in system_names:
        methods = data[system_name]
        if not methods:
            continue

        # Keep only known methods, maintain a deterministic order, then append unknowns
        def sort_key(item: Tuple[str, float]) -> Tuple[int, str]:
            name, _ = item
            return (method_order.index(name) if name in method_order else len(method_order), name)

        bar_data = sorted(methods.items(), key=sort_key)

        labels = [name for name, _ in bar_data]
        ausc_values = [val for _, val in bar_data]
        num_methods = len(labels)

        fig, ax = plt.subplots(figsize=(4, 6))
        y_positions = np.arange(num_methods) * 0.7
        bar_colors = [colors[i % len(colors)] for i in range(num_methods)]
        bars = ax.barh(y_positions, ausc_values, height=0.4, color=bar_colors)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)

        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.tick_params(axis="x", which="major", labelsize=16, length=8, width=2)
        ax.set_xlim(0.0, 1.05)
        ax.set_yticks([])

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                ha="left",
                va="center",
                fontsize=16,
            )

        ax.legend(bars, labels, title="Method", fontsize=10, title_fontsize=11, loc="lower right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        plt.tight_layout()
        safe_system_name = sanitize_name(system_name)
        plot_path = os.path.join(output_ausc_dir, f"figure6_{safe_system_name}_AUSC.pdf")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  -> Saved AUSC chart for {system_name} to: {plot_path}")


def find_csv_files(root: str, dt_label: str = ""):
    """Recursively find CSV files that match the expected mse_dt naming."""
    suffix = f"_mse_dt{dt_label}.csv".lower() if dt_label else None
    pattern = re.compile(r"_mse_dt[^/\\]*\.csv$", re.IGNORECASE)

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            name_l = fname.lower()
            if suffix:
                if name_l.endswith(suffix):
                    yield os.path.join(dirpath, fname)
            else:
                if pattern.search(name_l):
                    yield os.path.join(dirpath, fname)


def main():
    filepath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(filepath)

    parser = argparse.ArgumentParser(
        description="Compute success-rate curves and AUSC from MSE CSV files "
                    "(delta_max fixed to 1.0)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="../data",
        help="Root directory to search for MSE CSV files (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results",
        help="Directory to save generated AUSC bar charts.",
    )
    parser.add_argument(
        "--dt_label",
        type=str,
        default="1.0",
        help="If set, only process CSV files ending with _mse_dt{dt_label}.csv.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    print(f"[INFO] Searching for CSV files under: {root}")
    csv_files = list(find_csv_files(root, args.dt_label))
    if not csv_files:
        print("[WARN] No CSV files found.")
        return

    system_data: Dict[str, Dict[str, float]] = {}
    for csv_path in csv_files:
        system_name = os.path.splitext(os.path.basename(csv_path))[0]
        ausc_values = process_csv(csv_path)
        system_data[system_name] = ausc_values

    plot_ausc_bar(system_data, args.output, args.dt_label)


if __name__ == "__main__":
    main()
