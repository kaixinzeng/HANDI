#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_pendulum_styled_upsample.py

2D Pendulum ODE overlay plotter:
- Correct variable mapping: x(1)->x1, x(2)->x2
- True dim-D integration via solve_ivp
- Robust Bounds Calculation (Span-based)
- Robust Integration (NaN detection)

Example Usage:
python plot_pendulum_styled_upsample.py \
  --truth-npy dataV1/pendulum/pendulum_ds3.npy \
  --dt 0.1 --dim 2 \
  --bounds-from TRUTH --bounds-scale 1.5 \
  --out-dir plots_pendulum
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
# Style
# -----------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Style:
    fig_w: float = 8.96
    fig_h_unit: float = 3.05

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


def apply_ticks_adaptive(ax, xspan: float) -> None:
    nbins_x = 3 if xspan <= 1.2 else 4
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins_x, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2))

    fmt = ticker.FuncFormatter(_pretty_1dec_formatter)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ax.tick_params(
        labelsize=STYLE.tick_fontsize,
        width=STYLE.spine_width,
        direction="out",
        top=False,
        right=False,
    )


# -----------------------------
# helpers
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
        # Match dx(k)/dt = ... or dxk/dt = ...
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


def build_rhs_func(exprs, dim: int):
    xs = sp.symbols(" ".join([f"x{i}" for i in range(1, dim + 1)]))
    f_lambdas = [sp.lambdify(xs, expr, "numpy") for expr in exprs]

    def rhs(t, y):
        x = tuple(float(v) for v in y)
        out = np.empty(dim, dtype=float)
        for i, fi in enumerate(f_lambdas):
            try:
                out[i] = float(fi(*x))
            except Exception:
                out[i] = np.nan
        return out

    return rhs


def compute_bounds_from_pred(pred: np.ndarray, scale: float, min_abs: float = 0.0):
    """
    FIXED: Span-based expansion to avoid incorrect truncation for offset data.
    """
    P = np.asarray(pred, float)
    mask = np.isfinite(P).all(axis=1)
    if not np.any(mask):
        return None, None
    A = P[mask]

    # 1. Min/Max
    low = np.nanmin(A, axis=0)
    high = np.nanmax(A, axis=0)

    # 2. Span & Center
    span = high - low
    center = (high + low) / 2.0

    # 3. Minimum span guard
    span = np.maximum(span, 1e-6)

    # 4. Expand
    half_new_span = (span * scale) / 2.0

    bounds_low = center - half_new_span
    bounds_high = center + half_new_span

    # 5. Optional abs guard
    if min_abs > 0:
        bounds_low = np.minimum(bounds_low, -abs(min_abs))
        bounds_high = np.maximum(bounds_high, abs(min_abs))

    return bounds_low, bounds_high




def _parse_csv_floats(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    parts = [x.strip() for x in s.split(',')]
    out = []
    for x in parts:
        if x == "":
            continue
        out.append(float(x))
    return out if out else None


def compute_ylims_from_truth(truth: "np.ndarray", plot_dims: int, scale: float):
    """Per-dimension y-lims from truth, expanded by `scale` around center."""
    truth = np.asarray(truth, float)
    ymins, ymaxs = [], []
    for i in range(plot_dims):
        y = truth[:, i]
        m = np.isfinite(y)
        if not np.any(m):
            ymins.append(None)
            ymaxs.append(None)
            continue
        lo = float(np.nanmin(y[m]))
        hi = float(np.nanmax(y[m]))
        span = max(hi - lo, 1e-9)
        center = 0.5 * (hi + lo)
        half = 0.5 * span * float(scale)
        # guard: if span is tiny, still give a visible range
        if half < 1e-6:
            half = 1e-6 * float(scale)
        ymins.append(center - half)
        ymaxs.append(center + half)
    return ymins, ymaxs


def compute_manual_ylims(args, plot_dims: int):
    """Manual y-lims priority:
    1) --ymin/--ymax CSV lists
    2) per-dimension --y1-min/--y1-max ...
    """
    ymin_list = _parse_csv_floats(getattr(args, 'ymin', None))
    ymax_list = _parse_csv_floats(getattr(args, 'ymax', None))

    ymins, ymaxs = [None] * plot_dims, [None] * plot_dims

    if ymin_list is not None:
        for i, v in enumerate(ymin_list[:plot_dims]):
            ymins[i] = float(v)
    if ymax_list is not None:
        for i, v in enumerate(ymax_list[:plot_dims]):
            ymaxs[i] = float(v)

    # per-dim overrides if provided
    for i in range(plot_dims):
        k = i + 1
        vmin = getattr(args, f'y{k}_min', None)
        vmax = getattr(args, f'y{k}_max', None)
        if vmin is not None:
            ymins[i] = float(vmin)
        if vmax is not None:
            ymaxs[i] = float(vmax)

    return ymins, ymaxs


def integrate_ode(
        rhs, x0, t_eval,
        method="RK45", rtol=1e-7, atol=1e-9, max_step=None,
        bounds_low: Optional[np.ndarray] = None,
        bounds_high: Optional[np.ndarray] = None,
):
    """
    FIXED: Explicit NaN check in event to trigger termination on explosion.
    """
    t_span = (float(t_eval[0]), float(t_eval[-1]))
    y0 = np.asarray(x0, float)

    events = None
    if bounds_low is not None and bounds_high is not None:
        low = np.asarray(bounds_low, float)
        high = np.asarray(bounds_high, float)

        if np.any(y0 < low) or np.any(y0 > high):
            Y = np.full((len(t_eval), len(y0)), np.nan, dtype=float)

            class Dummy: pass

            sol = Dummy()
            sol.success = False
            sol.message = "Initial condition outside bounds."
            sol.y = None
            return sol, Y

        def diverge_event(t, y):
            y = np.asarray(y, float)
            # FIX: Immediate panic trigger if NaN/Inf
            if not np.all(np.isfinite(y)):
                return -1.0
            return float(np.min(np.concatenate([y - low, high - y])))

        diverge_event.terminal = True
        diverge_event.direction = -1
        events = [diverge_event]

    sol = solve_ivp(
        rhs, t_span=t_span, y0=y0, t_eval=t_eval,
        method=method, rtol=rtol, atol=atol, max_step=max_step,
        events=events,
    )

    if sol.y is None or sol.y.size == 0:
        Y = np.full((len(t_eval), len(y0)), np.nan, dtype=float)
        return sol, Y

    Y_part = sol.y.T
    Y = np.full((len(t_eval), Y_part.shape[1]), np.nan, dtype=float)
    n = min(len(t_eval), Y_part.shape[0])
    Y[:n] = Y_part[:n]
    return sol, Y


# -----------------------------
# plot helpers
# -----------------------------
def upsample_series(t: np.ndarray, y: np.ndarray, dense_factor: int):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    mask = np.isfinite(y)
    if not np.any(mask):
        return None, None
    last = np.where(mask)[0][-1]
    t0, y0 = t[:last + 1], y[:last + 1]

    if len(t0) < 4 or dense_factor <= 1:
        return t0, y0

    t_dense = np.linspace(t0[0], t0[-1], (len(t0) - 1) * dense_factor + 1)
    try:
        cs = CubicSpline(t0, y0)
        y_dense = cs(t_dense)
    except Exception:
        y_dense = np.interp(t_dense, t0, y0)
    return t_dense, y_dense


def plot_smooth_method(ax, t: np.ndarray, y: np.ndarray, *, color: str):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    mask = np.isfinite(y)
    if not np.any(mask):
        return
    last = np.where(mask)[0][-1]
    t0, y0 = t[:last + 1], y[:last + 1]

    if len(t0) < 4 or STYLE.smooth_dense_factor <= 1:
        ax.plot(t0, y0, color=color, lw=STYLE.method_lw, alpha=STYLE.method_alpha,
                solid_capstyle="round", zorder=2)
        return

    t_dense = np.linspace(t0[0], t0[-1], (len(t0) - 1) * STYLE.smooth_dense_factor + 1)
    try:
        cs = CubicSpline(t0, y0)
        y_dense = cs(t_dense)
    except Exception:
        y_dense = np.interp(t_dense, t0, y0)

    ax.plot(t_dense, y_dense, color=color, lw=STYLE.method_lw, alpha=STYLE.method_alpha,
            solid_capstyle="round", zorder=2)


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
        last = np.where(mask)[0][-1]
        ax.scatter(
            tt[:last + 1], yy[:last + 1],
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
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # UPDATED DEFAULTS FOR PENDULUM
    ap.add_argument("--truth-npy", type=str, default="dataV1/pendulum/pendulum_ds12.npy", help="(N,T,d) truth npy")
    ap.add_argument("--dt", type=float, default=0.1)  # Assuming dt=0.1 based on HANDI/PSE equations
    ap.add_argument("--traj-idx", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=5, help="truncate to t<=tmax")
    ap.add_argument("--dim", type=int, default=2)  # Changed to 2D

    # UPDATED EQ PATHS
    ap.add_argument("--handi-eq", type=str, default="true_systemV1/true_systemV1/true_system/pendulum/HANDI/dt0.1_equations.txt")
    ap.add_argument("--pse-eq", type=str, default="true_systemV1/true_systemV1/true_system/pendulum/PSE/dt0.1_equations.txt")
    ap.add_argument("--sindy-eq", type=str, default="true_systemV1/true_systemV1/true_system/pendulum/SINDy/dt0.1_equations.txt")
    ap.add_argument("--sr3-eq", type=str, default="true_systemV1/true_systemV1/true_system/pendulum/SR3/dt0.1_equations.txt")

    ap.add_argument("--out-dir", type=str, default="plots_pendulum")

    ap.add_argument("--ivp-method", type=str, default="RK45")
    ap.add_argument("--rtol", type=float, default=1e-7)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--max-step", type=float, default=None)

    # TRUNCATION: Recommend using TRUTH
    ap.add_argument("--bounds-from", type=str, default="TRUTH",
                    help="Use 'TRUTH' or a method name (HANDI) to compute bounds.")
    ap.add_argument("--bounds-scale", type=float, default=2.0,
                    help="Expand bounds by this factor.")
    ap.add_argument("--bounds-min-abs", type=float, default=0.0)

    # Y-limits (minimal-invasive): unify y-range by TRUTH per subplot (consistent across different dt runs)
    ap.add_argument("--ylim-mode", type=str, default="truth", choices=["truth", "manual", "none"],
                    help="truth: auto from truth; manual: use provided y-lims; none: leave matplotlib default")
    ap.add_argument("--ylim-scale", type=float, default=1.5,
                    help="Expand truth-derived y-range by this factor (span-based).")
    # manual lists (comma-separated), length >= dim
    ap.add_argument("--ymin", type=str, default="", help="CSV list of ymin for each dim, e.g. '-2,-1'")
    ap.add_argument("--ymax", type=str, default="", help="CSV list of ymax for each dim, e.g. '3,1'")
    # manual per-dim (kept for convenience)
    ap.add_argument("--y1-min", dest="y1_min", type=float, default=None)
    ap.add_argument("--y1-max", dest="y1_max", type=float, default=None)
    ap.add_argument("--y2-min", dest="y2_min", type=float, default=None)
    ap.add_argument("--y2-max", dest="y2_max", type=float, default=None)
    ap.add_argument("--y3-min", dest="y3_min", type=float, default=None)
    ap.add_argument("--y3-max", dest="y3_max", type=float, default=None)
    ap.add_argument("--y4-min", dest="y4_min", type=float, default=None)
    ap.add_argument("--y4-max", dest="y4_max", type=float, default=None)

    ap.add_argument("--plot-dims", type=int, default=2, help="Plot dim 1 to plot-dims")
    ap.add_argument("--truth-upsample-factor", type=int, default=12)
    ap.add_argument("--marker-step", type=int, default=1)
    ap.add_argument("--save-format", type=str, default="svg", choices=["png", "svg"])
    ap.add_argument("--no-legend",default=True)

    # optional style
    ap.add_argument("--method-lw", type=float, default=None)
    ap.add_argument("--truth-lw", type=float, default=None)
    ap.add_argument("--dash-on", type=int, default=None)
    ap.add_argument("--dash-off", type=int, default=None)
    ap.add_argument("--tick-fontsize", type=int, default=None)
    ap.add_argument("--spine-width", type=float, default=None)
    ap.add_argument("--smooth-factor", type=int, default=None)
    ap.add_argument("--hspace", type=float, default=0.5,
                    help="Vertical space between subplots (figure-relative). Increase to separate panels.")

    args = ap.parse_args()

    # apply style
    if args.method_lw is not None: STYLE.method_lw = float(args.method_lw)
    if args.truth_lw is not None: STYLE.truth_lw = float(args.truth_lw)
    if args.dash_on is not None and args.dash_off is not None: STYLE.truth_dash = (
    int(args.dash_on), int(args.dash_off))
    if args.tick_fontsize is not None: STYLE.tick_fontsize = int(args.tick_fontsize)
    if args.spine_width is not None: STYLE.spine_width = float(args.spine_width)
    if args.smooth_factor is not None: STYLE.smooth_dense_factor = int(args.smooth_factor)

    ensure_dir(args.out_dir)

    # ---- load truth ----
    print(f"[LOAD] {args.truth_npy}")
    data = np.load(args.truth_npy, allow_pickle=True).astype(float)
    if data.ndim != 3 or data.shape[-1] != args.dim:
        raise ValueError(f"Expected shape (N,T,{args.dim}), got {data.shape}")

    N, T, d = data.shape
    if not (0 <= args.traj_idx < N):
        raise ValueError(f"traj-idx out of range: {args.traj_idx}, N={N}")

    if args.tmax is not None:
        T_use = int(np.floor(float(args.tmax) / float(args.dt))) + 1
        T_use = max(2, min(T, T_use))
    else:
        T_use = T

    truth = data[args.traj_idx, :T_use, :]
    t_eval = to_time_grid(T_use, args.dt)

    # ---- methods ----
    methods: List[Tuple[str, str]] = []
    if args.handi_eq.strip(): methods.append(("HANDI", args.handi_eq))
    if args.pse_eq.strip():   methods.append(("PSE", args.pse_eq))
    if args.sindy_eq.strip(): methods.append(("SINDy", args.sindy_eq))
    if args.sr3_eq.strip():   methods.append(("SR3", args.sr3_eq))

    eq_path_map = {n: p for (n, p) in methods}

    # ---- save unified equations ----
    unified_path = os.path.join(args.out_dir, f"unified_equations_{d}d.txt")
    with open(unified_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"# unified equations (dim={d})\n")
        for name, eq_path in methods:
            try:
                exprs, unified_lines = parse_equations_file(eq_path, d)
                f_out.write(f"\n[{name}] {eq_path}\n")
                for line in unified_lines:
                    f_out.write(line + "\n")
            except Exception as e:
                print(f"[WARN] Failed to parse {name} for log: {e}")

    # ---- Phase A: compute bounds ----
    bounds_low = None
    bounds_high = None

    bounds_from = args.bounds_from.strip() if args.bounds_from else ""
    if bounds_from:
        # FIXED: Allow using TRUTH directly
        if bounds_from == "TRUTH":
            print(f"\n[BOUNDS] computing bounds from GROUND TRUTH (robust).")
            bounds_low, bounds_high = compute_bounds_from_pred(
                truth, scale=float(args.bounds_scale), min_abs=float(args.bounds_min_abs)
            )
        elif bounds_from in eq_path_map:
            print(f"\n[BOUNDS] computing bounds from {bounds_from}")
            exprs_b, _ = parse_equations_file(eq_path_map[bounds_from], d)
            rhs_b = build_rhs_func(exprs_b, d)
            max_step_b = args.max_step if args.max_step is not None else args.dt

            sol_b, Y_b = integrate_ode(
                rhs_b, truth[0, :], t_eval,
                method=args.ivp_method, rtol=args.rtol, atol=args.atol,
                max_step=max_step_b, bounds_low=None, bounds_high=None,
            )
            bounds_low, bounds_high = compute_bounds_from_pred(
                Y_b, scale=float(args.bounds_scale), min_abs=float(args.bounds_min_abs)
            )
        else:
            print(f"[WARN] bounds-from '{bounds_from}' not valid (TRUTH or Method). No bounds.")

        if bounds_low is not None:
            print("[BOUNDS] low =", bounds_low)
            print("[BOUNDS] high=", bounds_high)
        else:
            print("[BOUNDS] Truncation DISABLED (Could not compute bounds).")

    # ---- Phase B: integrate all ----
    preds: Dict[str, np.ndarray] = {}
    x0 = truth[0, :]

    for name, eq_path in methods:
        print(f"\n[RUN] {name} from x0={x0}")
        exprs, _ = parse_equations_file(eq_path, d)
        rhs = build_rhs_func(exprs, d)
        max_step = args.max_step if args.max_step is not None else args.dt

        sol, Y = integrate_ode(
            rhs, x0, t_eval,
            method=args.ivp_method, rtol=args.rtol, atol=args.atol,
            max_step=max_step, bounds_low=bounds_low, bounds_high=bounds_high,
        )

        preds[name] = Y
        m = mse_euclid(truth, Y)
        print(f"[METRIC] {name}: MSE = {m:.6e}")

    # ---- Plotting ----
    plot_dims = min(int(args.plot_dims), d)

    # --- compute y-lims (per subplot) ---
    ylim_mode = (args.ylim_mode or "truth").lower()
    if ylim_mode == "truth":
        ymins, ymaxs = compute_ylims_from_truth(truth, plot_dims=plot_dims, scale=float(args.ylim_scale))
    elif ylim_mode == "manual":
        ymins, ymaxs = compute_manual_ylims(args, plot_dims=plot_dims)
    else:
        ymins, ymaxs = [None]*plot_dims, [None]*plot_dims
    fig_h = STYLE.fig_h_unit * plot_dims
    fig, axes = plt.subplots(plot_dims, 1, figsize=(STYLE.fig_w, fig_h), sharex=True)
    if plot_dims == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=float(args.hspace))

    t_end = float(t_eval[-1])
    right_pad = 0.02 * max(t_end, 1e-12)
    right_pad = max(right_pad, 0.5 * float(args.dt))
    x_right = t_end + right_pad
    xspan = x_right - 0.0

    for i in range(plot_dims):
        ax = axes[i]

        # Methods
        for name, _ in methods:
            color = STYLE.colors.get(name, "black")
            if not args.no_legend:
                ax.plot([], [], color=color, lw=STYLE.method_lw, label=name)
            plot_smooth_method(ax, t_eval, preds[name][:, i], color=color)

        # Truth
        plot_truth_upsampled_plus_scatter(
            ax, t_eval, truth[:, i],
            dense_factor=int(args.truth_upsample_factor),
            marker_step=int(args.marker_step),
        )

        # Y-limits (do not touch ticks/spines/style)
        if ylim_mode != "none" and ymins[i] is not None and ymaxs[i] is not None:
            ax.set_ylim(float(ymins[i]), float(ymaxs[i]))

        ax.margins(x=0)
        ax.set_xlim(0.0, x_right)
        apply_ticks_adaptive(ax, xspan=xspan)
        apply_spines(ax)
        ax.grid(False)

    if not args.no_legend:
        maybe_add_legend(axes[0])

    out_fig = os.path.join(args.out_dir, f"overlay_pendulum_{plot_dims}dims.{args.save_format}")
    plt.savefig(out_fig, bbox_inches="tight", pad_inches=0.03, dpi=200 if args.save_format == "png" else None)
    plt.close(fig)

    print(f"[OK] saved figure: {out_fig}")


if __name__ == "__main__":
    main()