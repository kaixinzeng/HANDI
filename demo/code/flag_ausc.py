#!/usr/bin/env python3
"""
Compute AUSC (Area Under Success-rate Curve) for 7 system identification methods.

Workflow  (matching the original compute_mse pipeline):
  1. Load ground truth from traj_4_dt{label}.npy (already downsampled to dt)
  2. Parse discovered ODE equations for each method
  3. Integrate each ODE at original_dt = 1/30,  then  downsample  to dt
  4. Compute per-trajectory MSE between downsampled prediction and ground truth
  5. Compute AUSC from MSE distributions
  6. Plot a horizontal bar chart comparing all methods
"""

import os
import re
import math
import argparse
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.integrate import solve_ivp

# ─────────────────────────── Constants ───────────────────────────
DELTA_MAX = 1.0           # Upper integration limit for AUSC
ORIGINAL_DT = 1.0 / 30.0  # Physical sampling period (30 Hz)

# Ordered list of methods and their colors
METHOD_ORDER = ["EDMD", "gEDMD", "WSINDy", "PSE", "SR3", "SINDy", "HANDI"]
METHOD_COLORS = [
    "#4AB4B2",  # EDMD
    "#528FBF",  # gEDMD
    "#F4B36B",  # WSINDy
    "#D8A0A7",  # PSE
    "#C9A1CB",  # SR3
    "#d06569",  # SINDy
    "#50AAD8",  # HANDI
]


# ─────────────────── Equation Parsing ────────────────────────────
# Follows the same logic as compute_nrmse_names.parse_equations()

def parse_equations_standard(eq_file: str) -> Optional[List[str]]:
    """
    Parse equations in the format used by EDMD, gEDMD, WSINDy, SR3, SINDy, HANDI:
        dx(1)/dt = coeff*x(1)*x(2) + ...
    Supports both `*` and `·` as multiplication operators.

    Returns a list of RHS strings ready for eval(), or None on failure.
    """
    try:
        with open(eq_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[WARN] Equation file not found: {eq_file}")
        return None

    # Replace mid-dot with *
    content = content.replace("·", "*")

    pattern = r"dx\((\d+)\)/dt\s*=\s*(.*?)(?=\ndx\(\d|$)"
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        print(f"[WARN] No equations parsed from {eq_file}")
        return None

    equations = {}
    for var_idx, eq in matches:
        # Convert x(i) → x[i-1]
        eq = re.sub(r"x\((\d+)\)", lambda m: f"x[{int(m.group(1))-1}]", eq)
        eq = eq.strip().rstrip("\r")
        equations[int(var_idx) - 1] = eq

    return [equations[i] for i in sorted(equations.keys())]


def parse_equations_pse(eq_file: str) -> Optional[List[str]]:
    """
    Parse equations in the PSE format (0-based, ** power, xN notation):
        dx(0)/dt = expression using x0, x1, x2, x3 and ** for power
    First expands with sympy and converts to the standard x(i) format,
    then parses as standard equations.
    """
    try:
        from sympy import Symbol, sympify, expand
    except ImportError:
        print("[WARN] sympy required for PSE equation parsing")
        return None

    try:
        with open(eq_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[WARN] Equation file not found: {eq_file}")
        return None

    # Match both dx(0)/dt and dx0/dt formats
    pattern = r"dx\(?(\d+)\)?/dt\s*=\s*(.*?)(?=\r?\ndx\(?\d|$)"
    matches = re.findall(pattern, content, flags=re.DOTALL)
    if not matches:
        print(f"[WARN] No equations parsed from {eq_file}")
        return None

    max_index = max(int(m[0]) for m in matches)
    equations = {}

    for var_id, eq in matches:
        eq = eq.strip().rstrip("\r")
        expr = expand(sympify(eq))
        eq_str = str(expr)

        # Replace x0**2 → x0*x0, etc.
        def replace_power(m):
            v = m.group(1)
            p = int(m.group(2))
            return "*".join([v] * p) if p > 1 else v
        eq_str = re.sub(r"([a-zA-Z_]\w*)\*\*(\d+)", replace_power, eq_str)

        # Replace x0 → x[0], x1 → x[1], etc.  (descending to avoid x1 matching in x10)
        for i in range(max_index, -1, -1):
            eq_str = eq_str.replace(f"x{i}", f"x[{i}]")

        idx = int(var_id)
        equations[idx] = eq_str

    return [equations[i] for i in sorted(equations.keys())]


def detect_and_parse(eq_file: str) -> Optional[List[str]]:
    """Auto-detect equation format (PSE vs standard) and parse."""
    with open(eq_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if re.match(r"dx\(?0\)?/dt", first_line):
        return parse_equations_pse(eq_file)
    return parse_equations_standard(eq_file)


def create_learned_model(equations: List[str]) -> Callable:
    """
    Build an ODE model function from parsed equation strings.
    Matches the original create_learned_model() in compute_nrmse_names.py.
    """
    def model(x, t):
        dxdt = np.zeros(len(equations), dtype=float)
        for i, eq in enumerate(equations):
            try:
                val = eval(eq, {"np": np, "x": x})
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                dxdt[i] = float(val)
            except Exception:
                dxdt[i] = 0.0
        return dxdt
    return model


# ──────────────────── MSE Computation ────────────────────────────
# Matches the logic in compute_mse_nrmse_r2_click.py:
#   integrate at original_dt → downsample → compare with ground truth

def compute_mse_per_trajectory(
    data: np.ndarray,               # (N_traj, T, dim) at DT
    ode_func: Callable,
    dt: float,                      # Training dt (e.g. 0.2)
    original_dt: float = ORIGINAL_DT,
) -> np.ndarray:
    """
    For every trajectory in *data*, integrate the ODE at original_dt,
    downsample to dt, and compute MSE vs. ground truth.

    Returns a 1-D array of length N_traj (inf for diverged trajectories).
    """
    n_traj, T, dim = data.shape
    T_total = (T - 1) * dt                # total time span
    t_eval_original = np.linspace(0, T_total, int(T_total / original_dt) + 1)
    downsample_factor = int(round(dt / original_dt))

    mse_values = np.full(n_traj, np.inf)

    for i in range(n_traj):
        x0 = data[i, 0, :]
        try:
            sol = solve_ivp(
                lambda t, x: ode_func(x, t),
                [t_eval_original[0], t_eval_original[-1]],
                x0,
                t_eval=t_eval_original,
                method="Radau",
                rtol=1e-4,
                atol=1e-6,
            )
            pred = sol.y.T  # (len(t_eval_original), dim)
        except Exception as e:
            print(f"    [WARN] Integration error traj {i}: {e}")
            continue

        # Downsample prediction to match ground-truth dt
        downsample_indices = np.arange(0, pred.shape[0], downsample_factor)
        downsampled_pred = pred[downsample_indices]

        # Check shape compatibility
        if downsampled_pred.shape[0] != T:
            print(f"    [WARN] Shape mismatch traj {i}: "
                  f"pred={downsampled_pred.shape[0]} vs truth={T}")
            continue

        mse = float(np.mean((data[i] - downsampled_pred) ** 2))
        mse_values[i] = mse

    return mse_values


# -------------------- AUSC Computation ---------------------------

def compute_ausc(mse: np.ndarray, delta_max: float) -> float:
    """
    Compute AUSC via exact step-function integration over [0, delta_max].
    Returns a value in [0, 1] (NaN if mse is empty).
    """
    mse = np.asarray(mse, dtype=float)
    n = mse.size
    if n == 0:
        return float("nan")

    mse = np.clip(mse, 0.0, np.inf)
    m_sorted = np.sort(mse)
    extended = np.concatenate(([0.0], np.minimum(m_sorted, delta_max), [delta_max]))

    area = 0.0
    for k in range(n + 1):
        length = max(0.0, extended[k + 1] - extended[k])
        sr = k / n
        area += sr * length

    ausc = area / delta_max if delta_max > 0 else 1.0
    return max(0.0, min(1.0, ausc))


# -------------------- Plotting -----------------------------------

def plot_ausc_bar(ausc_by_method: Dict[str, float], output_path: str) -> None:
    """Create a horizontal AUSC bar chart for all methods."""
    ordered = []
    for m in METHOD_ORDER:
        if m in ausc_by_method:
            ordered.append((m, ausc_by_method[m]))
    for m, v in ausc_by_method.items():
        if m not in METHOD_ORDER:
            ordered.append((m, v))

    labels = [name for name, _ in ordered]
    values = [val for _, val in ordered]
    num = len(labels)

    fig, ax = plt.subplots(figsize=(4, 6))
    y_pos = np.arange(num) * 0.7
    colors = [METHOD_COLORS[i % len(METHOD_COLORS)] for i in range(num)]

    bars = ax.barh(y_pos, values, height=0.4, color=colors)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.tick_params(axis="x", which="major", labelsize=16, length=8, width=2)
    ax.set_xlim(0.0, 1.05)
    ax.set_yticks([])
    
    ax.set_xlabel("AUSC", fontsize=16)
    ax.set_ylabel("Method", fontsize=16)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", ha="left", va="center", fontsize=16)

    ax.legend(bars, labels, title="Method", fontsize=10,
              title_fontsize=11, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> Saved AUSC chart to: {output_path}")


# ──────────────────── Main ─────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(
        description="Compute AUSC from ground-truth NPY + 7 method equation files."
    )
    parser.add_argument(
        "--data", type=str,
        default=os.path.join(script_dir, "..", "data", "flag_traj_4_dt0.2.npy"),
        help="Path to the ground-truth .npy at the training dt.",
    )
    parser.add_argument(
        "--result_dir", type=str,
        default=os.path.join(script_dir, "..", "results"),
        help="Directory containing method equation .txt files.",
    )
    parser.add_argument(
        "--dt", type=float, default=0.2,
        help="Training time step (default: 0.2).",
    )
    parser.add_argument(
        "--original_dt", type=float, default=ORIGINAL_DT,
        help="Original sampling dt (default: 1/30).",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(script_dir, "..", "results",
                             "figure6_flag_AUSC.pdf"),
        help="Output path for the AUSC bar chart.",
    )
    args = parser.parse_args()

    # ── Load ground truth ──
    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        raise SystemExit(f"Data file not found: {data_path}")
    data = np.load(data_path)
    if data.ndim == 2:
        data = data[None, ...]
    print(f"[INFO] Loaded ground-truth data: {data_path}  shape={data.shape}")
    print(f"[INFO] Training dt={args.dt},  original_dt={args.original_dt:.8f}")

    result_dir = os.path.abspath(args.result_dir)
    dt_label = str(args.dt)

    # ── Build equation file paths for each method ──
    method_files: Dict[str, str] = {}
    for method in METHOD_ORDER:
        fname = f"{method}_flag_dt{dt_label}.txt"
        
        if method == "HANDI":
            fpath = os.path.join(result_dir, fname)
        else:
            data_dir = os.path.join(os.path.dirname(result_dir), "data")
            fpath = os.path.join(data_dir, fname)

        if os.path.isfile(fpath):
            method_files[method] = fpath
        else:
            print(f"[WARN] Equation file not found for {method}: {fpath}")

    if not method_files:
        raise SystemExit("No equation files found. Check --result_dir / --dt.")

    # ── Compute MSE and AUSC per method ──
    ausc_results: Dict[str, float] = {}

    for method, eq_path in method_files.items():
        print(f"\n[INFO] Processing method: {method}")
        print(f"       Equation file: {eq_path}")

        eq_list = detect_and_parse(eq_path)
        if eq_list is None:
            print(f"       [SKIP] Failed to parse equations.")
            continue

        ode_func = create_learned_model(eq_list)
        mse_arr = compute_mse_per_trajectory(
            data, ode_func, dt=args.dt, original_dt=args.original_dt
        )

        print(f"       Per-trajectory MSE: {mse_arr}")
        ausc_val = compute_ausc(mse_arr, DELTA_MAX)
        ausc_results[method] = ausc_val
        print(f"       AUSC = {ausc_val:.6f}")

    # ── Summary ──
    print("\n" + "=" * 50)
    print(f"AUSC Summary  (delta_max = {DELTA_MAX})")
    print("=" * 50)
    for method in METHOD_ORDER:
        if method in ausc_results:
            print(f"  {method:>8s} : {ausc_results[method]:.6f}")
    print("=" * 50)

    # ── Plot ──
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_ausc_bar(ausc_results, output_path)


if __name__ == "__main__":
    main()
