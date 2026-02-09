#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_ni_styled_upsample.py

3D NI System ODE overlay plotter (plotting first 2 dims):
- Dimension: 3D
- Plot: First 2 dimensions only
- Constraint: dx(3)/dt is forced to 0 during simulation for ALL methods.
- Correct variable mapping: x(1)->x1, x(2)->x2...
- Robust Bounds Calculation (Span-based, from TRUTH)
- Independent Dimension Truncation: If dim 3 diverges, dim 1 and 2 continue plotting if stable.

NEW UPDATES:
- Denormalization: Supports loading mean/std npy files to restore physical units.
  Formula: X_real = X_norm * std + mean
- Formatter: Width=5, always 1 decimal place.
- Y-Limits: Controlled by TRUTH range * scale, or manual overrides.

Example Usage:
python plot_ni_styled_upsample.py \
  --truth-npy NI/data/ni_truth.npy \
  --mean-npy NWB_traj2_states=3_t=1.6_zscore_mean.npy \
  --std-npy NWB_traj2_states=3_t=1.6_zscore_std.npy \
  --out-dir plots_NI \
  --tmax 20
"""

import os
import re
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

# -----------------------------
# Style Configuration
# -----------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Style:
    fig_w: float = 8.96
    fig_h_unit: float = 3.05  # height per subplot

    method_lw: float = 4.0
    method_alpha: float = 0.95
    smooth_dense_factor: int = 12

    truth_color: str = "#000000"
    truth_lw: float = 2.6
    truth_alpha: float = 1.0
    truth_dash: Tuple[int, int] = (1, 1)
    truth_scatter_s: float = 55.0
    truth_scatter_edge_lw: float = 1.8

    tick_fontsize: int = 46
    spine_width: float = 3.0
    spine_color: str = "black"
    show_spines: Tuple[str, ...] = ("left", "bottom")

    colors: Dict[str, str] = None


STYLE = Style(colors={
    "PSE":   "#D8A0A7",
    "HANDI": "#50AAD8",
    "SINDy": "#d06569",
    "SR3":   "#C9A1CB",
})


def _pretty_1dec_formatter(x, pos=None) -> str:

    if not np.isfinite(x):
        return ""

    if abs(x - round(x)) < 1e-8:
        return f"{round(x):>5.1f}"
    else:
        return f"{x:>5.1f}"


def apply_spines(ax) -> None:
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    for side in STYLE.show_spines:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(STYLE.spine_width)
        ax.spines[side].set_color(STYLE.spine_color)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def apply_ticks_adaptive(ax, xspan: float, yspan: float) -> None:
    nbins_x = 3 if xspan <= 1.2 else 4
    ax.xaxis.set_major_locator(
        ticker.MaxNLocator(nbins=nbins_x, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2)
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: f"{x:>5.1f}"
    ))

    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
    )

    def y_formatter(x, pos=None):
        if not np.isfinite(x):
            return ""

        if yspan < 0.05:
            return f"{x:>5.2f}"
        elif yspan < 0.5:
            return f"{x:>5.2f}"
        else:
            if abs(x - round(x)) < 1e-8:
                return f"{round(x):>5.1f}"
            return f"{x:>5.1f}"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))

    ax.tick_params(
        labelsize=STYLE.tick_fontsize,
        width=STYLE.spine_width,
        direction="out",
        top=False,
        right=False,
    )


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_time_grid(T: int, dt: float):
    return np.arange(T, dtype=float) * float(dt)


def mse_euclid(true_traj: np.ndarray, pred_traj: np.ndarray) -> float:
    true = np.asarray(true_traj, float)
    pred = np.asarray(pred_traj, float)
    mask = np.isfinite(pred).all(axis=1)
    if not np.any(mask):
        return float("inf")
    d = pred[mask] - true[mask]
    return float(np.mean(np.sum(d * d, axis=1)))


def compress_spaces(s: str) -> str:
    return " ".join(s.replace("\r", "\n").split())


def map_x_paren_to_xi(expr_str: str) -> str:
    return re.sub(r"x\s*\(\s*(\d+)\s*\)", lambda m: f"x{int(m.group(1))}", expr_str)


def map_xi_to_x_paren(expr_str: str) -> str:
    return re.sub(r"\bx(\d+)\b", lambda m: f"x({int(m.group(1))})", expr_str)


def extract_rhs_blocks(text: str, dim: int):
    rhs_list = [None] * dim
    for k in range(1, dim + 1):
        m = re.search(
            rf"dx\s*\(\s*{k}\s*\)\s*/\s*dt\s*=\s*(.*?)(?=\n\s*dx\s*\(|\Z)",
            text,
            flags=re.DOTALL,
        )
        if m is None:
            m2 = re.search(
                rf"dx\s*{k}\s*/\s*dt\s*=\s*(.*?)(?=\n\s*dx\s*\d\s*/\s*dt|\Z)",
                text,
                flags=re.DOTALL,
            )
            rhs_list[k - 1] = compress_spaces(m2.group(1)) if m2 else None
        else:
            rhs_list[k - 1] = compress_spaces(m.group(1))
    return rhs_list


_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse_equations_file(eq_path: str, dim: int):
    with open(eq_path, "r", encoding="utf-8") as f:
        text = f.read()

    rhs_blocks = extract_rhs_blocks(text, dim)

    xs = sp.symbols(" ".join([f"x{i}" for i in range(1, dim + 1)]))
    locals_map = {f"x{i}": xs[i - 1] for i in range(1, dim + 1)}
    locals_map.update({"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt})

    exprs = []
    unified_lines = []

    for k in range(1, dim + 1):
        rhs = rhs_blocks[k - 1]
        if rhs is None:
            e = sp.Integer(0)
            exprs.append(e)
            unified_lines.append(f"dx({k})/dt = 0")
            continue

        rhs = rhs.replace("^", "**")
        rhs = map_x_paren_to_xi(rhs)
        rhs = compress_spaces(rhs)

        try:
            e = parse_expr(rhs, local_dict=locals_map, transformations=_TRANS, evaluate=True)
        except Exception as ex:
            raise RuntimeError(
                f"[PARSE ERROR] {eq_path} dim {k} failed.\n"
                f"RHS(after map) = {rhs}\n"
                f"Error: {ex}"
            )

        exprs.append(sp.simplify(e))
        unified_lines.append(f"dx({k})/dt = {map_xi_to_x_paren(str(exprs[-1]))}")

    return exprs, unified_lines


def build_rhs_func(exprs, dim: int, force_zero_dim_idx: int = -1):
    xs = sp.symbols(" ".join([f"x{i}" for i in range(1, dim + 1)]))
    f_lambdas = [sp.lambdify(xs, expr, "numpy") for expr in exprs]

    def rhs(t, y):
        x = tuple(float(v) for v in y)
        out = np.empty(dim, dtype=float)
        for i, fi in enumerate(f_lambdas):
            if i == force_zero_dim_idx:
                out[i] = 0.0
            else:
                try:
                    out[i] = float(fi(*x))
                except Exception:
                    out[i] = np.nan
        return out

    return rhs


def compute_bounds_from_truth_extrema(pred: np.ndarray, scale: float, abs_floor: float = 100.0):
    P = np.asarray(pred, float)
    mask = np.isfinite(P).all(axis=1)
    if not np.any(mask):
        return None, None
    A = P[mask]

    minv = np.nanmin(A, axis=0)
    maxv = np.nanmax(A, axis=0)

    s = float(scale)
    if s <= 0:
        raise ValueError("bounds-scale must be > 0")

    low = np.empty_like(minv)
    high = np.empty_like(maxv)

    for i in range(len(minv)):
        mn = float(minv[i])
        mx = float(maxv[i])
        low[i] = mn * s if mn < 0 else mn / s
        high[i] = mx * s if mx > 0 else mx / s

    af = float(abs_floor)
    if af > 0:
        low = np.minimum(low, -abs(af))
        high = np.maximum(high, abs(af))

    for i in range(len(low)):
        if low[i] > high[i]:
            m = max(abs(low[i]), abs(high[i]), af, 1e-6)
            low[i], high[i] = -m, m

    return low, high


def apply_bounds_mask(Y: np.ndarray, low: Optional[np.ndarray], high: Optional[np.ndarray]) -> np.ndarray:
    if Y is None or low is None or high is None:
        return Y

    Y_out = Y.copy()
    D = Y.shape[1]

    for d in range(D):
        with np.errstate(invalid="ignore"):
            mask = (Y_out[:, d] < low[d]) | (Y_out[:, d] > high[d])
        Y_out[mask, d] = np.nan

    return Y_out


def integrate_ode(rhs, x0, t_eval, method="RK45", rtol=1e-7, atol=1e-9, max_step=None):
    t_span = (float(t_eval[0]), float(t_eval[-1]))
    y0 = np.asarray(x0, float)

    try:
        sol = solve_ivp(
            rhs,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
    except ValueError:
        Y = np.full((len(t_eval), len(y0)), np.nan, dtype=float)

        class Dummy:
            pass

        sol = Dummy()
        sol.success = False
        sol.message = "Integration crashed."
        sol.y = None
        return sol, Y

    if sol.y is None or sol.y.size == 0:
        Y = np.full((len(t_eval), len(y0)), np.nan, dtype=float)
        return sol, Y

    Y_part = sol.y.T
    Y = np.full((len(t_eval), Y_part.shape[1]), np.nan, dtype=float)
    n = min(len(t_eval), Y_part.shape[0])
    Y[:n] = Y_part[:n]

    return sol, Y


# -----------------------------
# Plot Helpers
# -----------------------------
def upsample_series(t: np.ndarray, y: np.ndarray, dense_factor: int):
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    mask = np.isfinite(y)
    if not np.any(mask):
        return None, None

    if np.isnan(y).any():
        return t, y

    if len(t) < 4 or dense_factor <= 1:
        return t, y

    t_dense = np.linspace(t[0], t[-1], (len(t) - 1) * dense_factor + 1)
    try:
        cs = CubicSpline(t, y)
        y_dense = cs(t_dense)
    except Exception:
        y_dense = np.interp(t_dense, t, y)

    return t_dense, y_dense


def plot_smooth_method(ax, t: np.ndarray, y: np.ndarray, *, color: str):
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    mask = np.isfinite(y)
    if not np.any(mask):
        return

    if np.isnan(y).any():
        ax.plot(
            t, y,
            color=color, lw=STYLE.method_lw, alpha=STYLE.method_alpha,
            solid_capstyle="round", zorder=2
        )
        return

    t_dense, y_dense = upsample_series(t, y, dense_factor=STYLE.smooth_dense_factor)
    if t_dense is None:
        return

    ax.plot(
        t_dense, y_dense,
        color=color, lw=STYLE.method_lw, alpha=STYLE.method_alpha,
        solid_capstyle="round", zorder=2
    )


def plot_truth_upsampled_plus_scatter(
        ax, t_train: np.ndarray, y_train: np.ndarray, *, dense_factor: int, marker_step: int
):
    t_hi, y_hi = upsample_series(t_train, y_train, dense_factor=dense_factor)
    if t_hi is not None and y_hi is not None:
        ax.plot(
            t_hi, y_hi,
            color=STYLE.truth_color, linewidth=STYLE.truth_lw, alpha=STYLE.truth_alpha,
            linestyle=(0, STYLE.truth_dash), zorder=6
        )

    step = int(marker_step) if int(marker_step) >= 1 else 1
    tt = np.asarray(t_train, float)[::step]
    yy = np.asarray(y_train, float)[::step]
    mask = np.isfinite(yy)
    if np.any(mask):
        last_idx = np.where(mask)[0][-1]
        ax.scatter(
            tt[:last_idx + 1], yy[:last_idx + 1],
            s=STYLE.truth_scatter_s, facecolors="#FFFFFF", edgecolors=STYLE.truth_color,
            linewidths=STYLE.truth_scatter_edge_lw, zorder=7
        )


def maybe_add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    good = [(h, l) for (h, l) in zip(handles, labels) if l and not l.startswith("_")]
    if len(good) == 0:
        return
    h2, l2 = zip(*good)
    ax.legend(h2, l2, loc="best", frameon=False)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # UPDATED DEFAULTS FOR NI SYSTEM
    ap.add_argument("--truth-npy", type=str, default="neural1/NI_dt0.03+poly3.npy", help="(N,T,d) truth npy")

    ap.add_argument("--mean-npy", type=str, default="neural1/NWB_traj2_states=3_t=1.6_zscore_mean.npy")
    ap.add_argument("--std-npy", type=str, default="neural1/NWB_traj2_states=3_t=1.6_zscore_std.npy")

    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--traj-idx", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=0.4, help="truncate to t<=tmax")
    ap.add_argument("--dim", type=int, default=3)

    # VISUAL OFFSET
    ap.add_argument("--t-offset", type=float, default=1.0, help="Visual time-axis offset")

    # Y-AXIS LIMITS CONTROLS
    ap.add_argument("--ylim-scale", type=float, default=1.5,
                    help="Expand Y-limits relative to Truth range (default 1.1)")
    ap.add_argument("--y1-min", type=float, default=None, help="Override Y min for dim 1")
    ap.add_argument("--y1-max", type=float, default=None, help="Override Y max for dim 1")
    ap.add_argument("--y2-min", type=float, default=None, help="Override Y min for dim 2")
    ap.add_argument("--y2-max", type=float, default=None, help="Override Y max for dim 2")

    # EQ PATHS
    ap.add_argument("--handi-eq", type=str, default="NI/HANDI/dt0.030/best_equations_dt0.030.txt")
    ap.add_argument("--pse-eq", type=str, default="NI/PSE/pse_equation_dt0.03_new.txt")
    ap.add_argument("--sindy-eq", type=str, default="NI/sindy/sindy_equations_dt0.03.txt")
    ap.add_argument("--sr3-eq", type=str, default="NI/sr3/NI_dt0.030/sindy_sr3_equations_mul.txt")

    ap.add_argument("--out-dir", type=str, default="plots_NI")

    ap.add_argument("--ivp-method", type=str, default="RK45")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--max-step", type=float, default=0.0002)

    # TRUNCATION
    ap.add_argument("--bounds-from", type=str, default="TRUTH")
    ap.add_argument("--bounds-scale", type=float, default=2.0)
    ap.add_argument("--bounds-min-abs", type=float, default=10.0)

    # PLOTTING
    ap.add_argument("--plot-dims", type=int, default=2)
    ap.add_argument("--truth-upsample-factor", type=int, default=12)
    ap.add_argument("--marker-step", type=int, default=3)
    ap.add_argument("--save-format", type=str, default="svg", choices=["png", "svg"])
    ap.add_argument("--no-legend", action="store_true", default=True)

    # STYLE OVERRIDES
    ap.add_argument("--method-lw", type=float, default=None)
    ap.add_argument("--truth-lw", type=float, default=None)
    ap.add_argument("--dash-on", type=int, default=None)
    ap.add_argument("--dash-off", type=int, default=None)
    ap.add_argument("--tick-fontsize", type=int, default=None)
    ap.add_argument("--spine-width", type=float, default=None)
    ap.add_argument("--smooth-factor", type=int, default=None)
    ap.add_argument("--hspace", type=float, default=0.5)

    args = ap.parse_args()

    # apply style
    if args.method_lw is not None:
        STYLE.method_lw = float(args.method_lw)
    if args.truth_lw is not None:
        STYLE.truth_lw = float(args.truth_lw)
    if args.dash_on is not None and args.dash_off is not None:
        STYLE.truth_dash = (int(args.dash_on), int(args.dash_off))
    if args.tick_fontsize is not None:
        STYLE.tick_fontsize = int(args.tick_fontsize)
    if args.spine_width is not None:
        STYLE.spine_width = float(args.spine_width)
    if args.smooth_factor is not None:
        STYLE.smooth_dense_factor = int(args.smooth_factor)

    ensure_dir(args.out_dir)

    # ---- load truth ----
    print(f"[LOAD] {args.truth_npy}")
    try:
        data = np.load(args.truth_npy, allow_pickle=True).astype(float)
    except FileNotFoundError:
        print(f"!!! Error: Truth file not found at {args.truth_npy}")
        return

    if data.ndim != 3 or data.shape[-1] != args.dim:
        raise ValueError(f"Expected shape (N,T,{args.dim}), got {data.shape}")

    # ---- Load Mean/Std for Denormalization ----
    # Default is identity (no change) if files not provided
    mean_vec = np.zeros(args.dim, dtype=float)
    std_vec = np.ones(args.dim, dtype=float)

    if args.mean_npy and args.std_npy:
        print(f"[LOAD] Mean: {args.mean_npy}")
        print(f"[LOAD] Std:  {args.std_npy}")
        try:
            m_tmp = np.load(args.mean_npy).reshape(-1)
            s_tmp = np.load(args.std_npy).reshape(-1)
            if len(m_tmp) != args.dim or len(s_tmp) != args.dim:
                print(f"[WARN] Mean/Std shape mismatch! Expected {args.dim}, got {len(m_tmp)}/{len(s_tmp)}. Ignoring.")
            else:
                mean_vec = m_tmp
                std_vec = s_tmp
                print(f"       Loaded Mean: {mean_vec}")
                print(f"       Loaded Std:  {std_vec}")
        except Exception as e:
            print(f"[WARN] Failed to load mean/std: {e}. Using raw normalized data.")
    else:
        print("[INFO] No mean/std provided. Plotting normalized data.")

    N, T, d = data.shape
    if args.tmax is not None:
        T_use = int(np.floor(float(args.tmax) / float(args.dt))) + 1
        T_use = max(2, min(T, T_use))
    else:
        T_use = T

    # TRUTH (Normalized) - needed for x0 and integration
    truth_norm = data[args.traj_idx, :T_use, :]
    t_eval = to_time_grid(T_use, args.dt)

    # ---- methods ----
    methods: List[Tuple[str, str]] = []
    if args.handi_eq.strip():
        methods.append(("HANDI", args.handi_eq))
    if args.pse_eq.strip():
        methods.append(("PSE", args.pse_eq))
    if args.sindy_eq.strip():
        methods.append(("SINDy", args.sindy_eq))
    if args.sr3_eq.strip():
        methods.append(("SR3", args.sr3_eq))

    eq_path_map = {n: p for (n, p) in methods}

    # ---- save unified equations ----
    unified_path = os.path.join(args.out_dir, f"unified_equations_{d}d.txt")
    with open(unified_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"# unified equations (dim={d})\n")
        f_out.write(f"# CONSTRAINT: dx({d})/dt is forced to 0.0 during simulation.\n")
        for name, eq_path in methods:
            try:
                exprs, unified_lines = parse_equations_file(eq_path, d)
                f_out.write(f"\n[{name}] {eq_path}\n")
                for line in unified_lines:
                    f_out.write(line + "\n")
            except Exception as e:
                print(f"[WARN] Failed to parse {name} for log: {e}")

    # ---- Phase A: compute bounds (IN NORMALIZED SPACE) ----
    # Bounds must be computed on normalized data because integration happens in normalized space
    bounds_low = None
    bounds_high = None
    if args.bounds_from:
        if args.bounds_from == "TRUTH":
            bounds_low, bounds_high = compute_bounds_from_truth_extrema(
                truth_norm, scale=float(args.bounds_scale), abs_floor=float(args.bounds_min_abs)
            )
        elif args.bounds_from in eq_path_map:
            exprs_b, _ = parse_equations_file(eq_path_map[args.bounds_from], d)
            rhs_b = build_rhs_func(exprs_b, d, force_zero_dim_idx=2)
            max_step_b = args.max_step if args.max_step is not None else args.dt
            _sol_b, _Y_b = integrate_ode(
                rhs_b, truth_norm[0, :], t_eval,
                method=args.ivp_method, rtol=args.rtol, atol=args.atol,
                max_step=max_step_b
            )
            bounds_low, bounds_high = compute_bounds_from_truth_extrema(
                truth_norm, scale=float(args.bounds_scale), abs_floor=float(args.bounds_min_abs)
            )

    # ---- Phase B: integrate all (IN NORMALIZED SPACE) ----
    preds_norm: Dict[str, np.ndarray] = {}
    x0_norm = truth_norm[0, :]

    for name, eq_path in methods:
        print(f"\n[RUN] {name} (Norm)")
        exprs, _ = parse_equations_file(eq_path, d)
        rhs = build_rhs_func(exprs, d, force_zero_dim_idx=2)
        max_step = args.max_step if args.max_step is not None else args.dt
        sol, Y = integrate_ode(
            rhs, x0_norm, t_eval,
            method=args.ivp_method, rtol=args.rtol, atol=args.atol,
            max_step=max_step
        )
        if bounds_low is not None and bounds_high is not None:
            Y = apply_bounds_mask(Y, bounds_low, bounds_high)
        preds_norm[name] = Y

    # ---- Phase C: Denormalize Everything (Convert to Physical Units) ----
    # Formula: Real = Norm * std + mean
    # Note: NaN values in Y will remain NaN, which is correct.

    truth_real = truth_norm * std_vec + mean_vec
    preds_real = {}

    print("\n[TRANSFORM] Denormalizing data to physical units...")
    for name, Y_norm in preds_norm.items():
        # Broadcasting: (T, D) * (D,) + (D,)
        preds_real[name] = Y_norm * std_vec + mean_vec

        # Compute MSE on REAL physical values
        m = mse_euclid(truth_real, preds_real[name])
        print(f"[METRIC] {name} (Physical): MSE = {m:.6e}")

    # ---- Plotting (Using Physical Data) ----
    t_offset = float(args.t_offset)
    t_plot = t_eval + t_offset

    plot_dims = min(int(args.plot_dims), d)
    fig_h = STYLE.fig_h_unit * plot_dims
    fig, axes = plt.subplots(plot_dims, 1, figsize=(STYLE.fig_w, fig_h), sharex=True)
    if plot_dims == 1:
        axes = [axes]

    fig.subplots_adjust(hspace=float(args.hspace))

    x_left = float(t_plot[0])
    x_end = float(t_plot[-1])
    span = max(x_end - x_left, 1e-12)
    right_pad = 0.02 * span
    right_pad = max(right_pad, 0.5 * float(args.dt))
    x_right = x_end + right_pad
    xspan = x_right - x_left

    for i in range(plot_dims):
        ax = axes[i]

        # 1. Determine Y-Limits based on TRUTH (Physical) + SCALE + OVERRIDES
        y_truth = truth_real[:, i]
        y_min_t, y_max_t = np.nanmin(y_truth), np.nanmax(y_truth)

        # Expand by scale
        y_mid = (y_min_t + y_max_t) / 2.0
        y_half_span = (max(y_max_t - y_min_t, 1e-9) / 2.0) * float(args.ylim_scale)
        cur_ymin = y_mid - y_half_span
        cur_ymax = y_mid + y_half_span

        # Overrides
        if i == 0:
            if args.y1_min is not None: cur_ymin = float(args.y1_min)
            if args.y1_max is not None: cur_ymax = float(args.y1_max)
        elif i == 1:
            if args.y2_min is not None: cur_ymin = float(args.y2_min)
            if args.y2_max is not None: cur_ymax = float(args.y2_max)

        ax.set_ylim(cur_ymin, cur_ymax)

        # 2. Plot Methods (Physical)
        for name, _ in methods:
            color = STYLE.colors.get(name, "black")
            if not args.no_legend:
                ax.plot([], [], color=color, lw=STYLE.method_lw, label=name)
            plot_smooth_method(ax, t_plot, preds_real[name][:, i], color=color)

        # 3. Plot Truth (Physical)
        plot_truth_upsampled_plus_scatter(
            ax, t_plot, truth_real[:, i],
            dense_factor=int(args.truth_upsample_factor),
            marker_step=int(args.marker_step),
        )

        ax.margins(x=0)
        ax.set_xlim(x_left, x_right)

        # 4. Apply Ticks & Spines
        yspan_current = cur_ymax - cur_ymin

        apply_ticks_adaptive(ax, xspan=xspan, yspan=yspan_current)

        apply_spines(ax)
        ax.grid(False)

    if not args.no_legend:
        maybe_add_legend(axes[0])

    out_fig = os.path.join(args.out_dir, f"overlay_NI_{plot_dims}dims_real.{args.save_format}")
    plt.savefig(out_fig, bbox_inches="tight", pad_inches=0.03, dpi=200 if args.save_format == "png" else None)
    plt.close(fig)

    print(f"[OK] saved figure: {out_fig}")


if __name__ == "__main__":
    main()