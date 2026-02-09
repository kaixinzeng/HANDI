#!/usr/bin/env python3
"""
Compute success-rate curves and AUSC for each method in MSE CSV files.

Directory layout assumption
---------------------------
The root directory contains subfolders for different systems. If you walk
down each system folder, you eventually reach a CSV file whose columns are
different methods and whose rows are MSE values for different trajectories.

Example:
    mse/
      system1/
        xxx.csv
      system2/
        yyy.csv
      ...

For each CSV file, this script will:
  * Read the header row as method names.
  * For each method (each column), collect the trajectory-level MSE values.
  * Fix delta_max = 1.0 for all methods and all CSV files.
  * For each method:
        - compute SR curve on [0, 1]
        - compute AUSC on [0, 1]
        - save in the SAME directory as the CSV:
              <method>_sr.npy   : shape (K, 2), columns [delta, SR(delta)]
              <method>_ausc.npy : scalar AUSC (0D array)
"""

import os
import csv
import math
import argparse
import re
from typing import Dict, List, Tuple

import numpy as np

DELTA_MAX = 1.0

def sanitize_name(name: str) -> str:
    """
    Make a safe file name from a method name.

    Keeps letters, digits, dot, dash and underscore; replaces other chars with '_'.
    """
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


def compute_success_rate_curve(mse: np.ndarray, delta_max: float) -> np.ndarray:
    """
    Compute success-rate curve data SR(delta) for one method.

    The curve is evaluated at all distinct MSE values (clipped to [0, delta_max]),
    plus delta = 0 and delta = delta_max. This is sufficient to reconstruct the
    step-function success curve exactly.

    Parameters
    ----------
    mse : np.ndarray
        1D array of trajectory-level MSE values for this method.
    delta_max : float
        Maximum delta value of interest (here fixed to 1.0 in the caller).

    Returns
    -------
    sr_data : np.ndarray
        2D array of shape (K, 2), with columns:
            sr_data[:, 0] = delta values (ascending, starting at 0)
            sr_data[:, 1] = SR(delta) = fraction of trajectories with MSE <= delta
    """
    mse = np.asarray(mse, dtype=float)
    n = mse.size
    if n == 0:
        # No data: SR(delta) is 0 everywhere
        deltas = np.array([0.0, delta_max], dtype=float)
        sr = np.zeros_like(deltas)
        return np.stack([deltas, sr], axis=1)

    mse = np.clip(mse, 0.0, np.inf)

    # Use all finite MSEs, clipped to delta_max, as potential breakpoints
    finite_mask = np.isfinite(mse)
    finite_clipped = np.minimum(mse[finite_mask], delta_max)

    if finite_clipped.size == 0:
        # All values are +inf: success rate is always 0
        deltas = np.array([0.0, delta_max], dtype=float)
        sr = np.zeros_like(deltas)
        return np.stack([deltas, sr], axis=1)

    deltas = np.unique(np.concatenate(([0.0], finite_clipped, [delta_max])))
    sr = np.empty_like(deltas)

    # For each delta, compute fraction of MSE <= delta (inf never counted as success)
    for i, d in enumerate(deltas):
        sr[i] = np.mean(mse <= d)

    return np.stack([deltas, sr], axis=1)


def process_csv(csv_path: str) -> None:
    """
    Process one CSV file:
      * Read MSE per method
      * Use fixed delta_max = DELTA_MAX (here 1.0) for all methods
      * For each method:
            - compute SR curve on [0, 1]
            - compute AUSC on [0, 1]
            - save as .npy files in the same directory

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    """
    header, data = read_mse_csv(csv_path)

    out_dir = os.path.dirname(csv_path)
    print(f"[INFO] Processing CSV: {csv_path}")
    print(f"       Methods: {', '.join(header)}")
    print(f"       delta_max is FIXED to {DELTA_MAX:.6g} for all methods")

    for method_name in header:
        mse_values = np.asarray(data[method_name], dtype=float)

        # Success-rate curve on [0, DELTA_MAX]
        sr_data = compute_success_rate_curve(mse_values, DELTA_MAX)
        # AUSC on [0, DELTA_MAX]
        ausc_val = compute_ausc(mse_values, DELTA_MAX)

        safe_name = sanitize_name(method_name)
        sr_path = os.path.join(out_dir, f"{safe_name}_sr.npy")
        ausc_path = os.path.join(out_dir, f"{safe_name}_ausc.npy")

        np.save(sr_path, sr_data)
        np.save(ausc_path, np.array(ausc_val, dtype=float))

        print(f"       Saved for method '{method_name}':")
        print(f"         SR curve -> {sr_path}")
        print(f"         AUSC     -> {ausc_path} (value = {ausc_val:.6f})")


def find_csv_files(root: str):
    """
    Recursively find all .csv files under the given root directory.
    """
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                yield os.path.join(dirpath, fname)


def main():
    parser = argparse.ArgumentParser(
        description="Compute success-rate curves and AUSC from MSE CSV files "
                    "(delta_max fixed to 1.0)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./mse_ausc_sr",
        help="Root directory to search for MSE CSV files (default: current directory).",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    print(f"[INFO] Searching for CSV files under: {root}")
    csv_files = list(find_csv_files(root))
    if not csv_files:
        print("[WARN] No CSV files found.")
        return

    for csv_path in csv_files:
        process_csv(csv_path)


if __name__ == "__main__":
    main()
