#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import os.path as osp
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Callable

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

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# -------------------- Style --------------------

@dataclass
class Style:
    fig_w: float = 8.96
    fig_h: float = 6.1

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
    "PSE":   "#d06569",
    "HANDI": "#4FA8D5",
    "SINDy": "#ECBC91",
    "SR3":   "#C9A1CB",
})


# -------------------- I/O --------------------

def load_npy_safely(path: str) -> np.ndarray:
    try:
        arr = np.load(path)
    except ValueError:
        arr = np.load(path, allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.array([np.asarray(a) for a in arr], dtype=float)

    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError(f"Invalid npy shape {arr.shape}, expected (T,>=2) or (N,T,>=2).")
    return arr.astype(float)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# -------------------- Ticks / Spines --------------------

def _pretty_1dec_formatter(x, pos=None) -> str:
    """
    Fixed-width formatter to stabilize tight bbox:
    - always show 1 decimal
    - right-align to width=4 (e.g. " 2.5", "12.5", "-2.5")
    """
    if not np.isfinite(x):
        return ""
    return f"{x:>4.1f}"


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
    """
    x ticks: adaptive 3 or 4
    y ticks: 3
    labels: fixed-width (see _pretty_1dec_formatter)
    """
    nbins_x = 3 if xspan <= 1.2 else 4

    ax.xaxis.set_major_locator(
        ticker.MaxNLocator(nbins=nbins_x, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2)
    )
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2)
    )

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


# -------------------- Interpolation (for exact tmax) --------------------

def _resample_2d_traj_to_times(
    t_old: np.ndarray,
    y_old: np.ndarray,
    t_new: np.ndarray,
) -> np.ndarray:
    """
    Resample a (T,2) trajectory from t_old to t_new using CubicSpline; fallback to linear.
    Assumes y_old finite (truth usually is). If there are NaNs, it will do a safer fallback.
    """
    t_old = np.asarray(t_old, float)
    y_old = np.asarray(y_old, float)
    t_new = np.asarray(t_new, float)

    out = np.full((len(t_new), 2), np.nan, dtype=float)

    for d in range(2):
        yd = y_old[:, d]
        mask = np.isfinite(yd) & np.isfinite(t_old)
        if np.sum(mask) < 2:
            continue

        tt = t_old[mask]
        vv = yd[mask]

        # If not strictly increasing, sort
        order = np.argsort(tt)
        tt = tt[order]
        vv = vv[order]

        # If duplicates exist, keep first occurrence
        uniq, idx = np.unique(tt, return_index=True)
        tt = uniq
        vv = vv[idx]

        if len(tt) < 2:
            continue

        try:
            if len(tt) >= 4:
                cs = CubicSpline(tt, vv, extrapolate=False)
                out[:, d] = cs(t_new)
            else:
                out[:, d] = np.interp(t_new, tt, vv, left=np.nan, right=np.nan)
        except Exception:
            out[:, d] = np.interp(t_new, tt, vv, left=np.nan, right=np.nan)

    return out


def build_time_grid_with_tmax(dt: float, tmax: float, mode: str = "exact") -> np.ndarray:
    """
    Build t_eval.
    - floor: end = floor(tmax/dt)*dt
    - round: end = round(tmax/dt)*dt
    - ceil : end = ceil(tmax/dt)*dt
    - exact: dt-grid up to floor + append exact tmax if not already on grid
    """
    dt = float(dt)
    tmax = float(tmax)
    mode = str(mode).lower()

    if dt <= 0:
        raise ValueError("dt must be > 0")
    if tmax <= 0:
        raise ValueError("tmax must be > 0")

    if mode == "floor":
        n = int(np.floor(tmax / dt))
        n = max(1, n)
        return np.arange(n + 1, dtype=float) * dt

    if mode == "round":
        n = int(np.round(tmax / dt))
        n = max(1, n)
        return np.arange(n + 1, dtype=float) * dt

    if mode == "ceil":
        n = int(np.ceil(tmax / dt))
        n = max(1, n)
        return np.arange(n + 1, dtype=float) * dt

    if mode == "exact":
        n = int(np.floor(tmax / dt))
        n = max(1, n)
        grid = np.arange(n + 1, dtype=float) * dt
        if abs(grid[-1] - tmax) < 1e-12:
            return grid
        return np.concatenate([grid, np.array([tmax], dtype=float)])

    raise ValueError(f"Unknown tmax-mode: {mode}")


# -------------------- Plot helpers --------------------

def plot_smooth_curve(
    ax,
    t: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    lw: float,
    alpha: float,
    dense_factor: int,
    zorder: int = 2,
) -> None:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if not np.any(mask):
        return

    last = np.where(mask)[0][-1]
    t0 = t[: last + 1]
    y0 = y[: last + 1]

    # IMPORTANT: t0 might be non-uniform (exact tmax). CubicSpline requires strictly increasing.
    # We'll do spline only if t0 is strictly increasing and has enough points, else fallback.
    if len(t0) < 4 or dense_factor <= 1:
        ax.plot(t0, y0, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=zorder)
        return

    if np.any(np.diff(t0) <= 0):
        ax.plot(t0, y0, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=zorder)
        return

    t_dense = np.linspace(t0[0], t0[-1], (len(t0) - 1) * dense_factor + 1)
    try:
        cs = CubicSpline(t0, y0)
        y_dense = cs(t_dense)
    except Exception:
        y_dense = np.interp(t_dense, t0, y0)

    ax.plot(t_dense, y_dense, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=zorder)


def plot_truth_line_plus_scatter(
    ax,
    *,
    # scatter uses ORIGINAL sampled points (dt-grid)
    t_scatter: np.ndarray,
    y_scatter: np.ndarray,
    marker_step: int,
    # line uses (possibly) resampled truth that reaches exact tmax
    t_line: np.ndarray,
    y_line: np.ndarray,
    color: str,
    lw: float,
    alpha: float,
    dash: Tuple[int, int],
    s: float,
    edge_lw: float,
    zorder_line: int = 6,
    zorder_scatter: int = 7,
) -> None:
    t_line = np.asarray(t_line, float)
    y_line = np.asarray(y_line, float)
    mask = np.isfinite(y_line)
    if np.any(mask):
        last = np.where(mask)[0][-1]
        ax.plot(
            t_line[: last + 1], y_line[: last + 1],
            color=color, linewidth=lw, alpha=alpha,
            linestyle=(0, dash),
            zorder=zorder_line
        )

    step = int(marker_step) if int(marker_step) >= 1 else 1
    tt = np.asarray(t_scatter, float)[::step]
    yy = np.asarray(y_scatter, float)[::step]
    mask2 = np.isfinite(yy)
    if np.any(mask2):
        last2 = np.where(mask2)[0][-1]
        ax.scatter(
            tt[: last2 + 1], yy[: last2 + 1],
            s=s, facecolors="#FFFFFF", edgecolors=color,
            linewidths=edge_lw, zorder=zorder_scatter
        )


# -------------------- Equation parsing (SymPy) --------------------

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

X1 = sp.Symbol("x1")
X2 = sp.Symbol("x2")


def _parse_sympy(expr_str: str, local_dict: Dict[str, sp.Symbol]) -> sp.Expr:
    return parse_expr(
        expr_str.strip(),
        local_dict=local_dict,
        transformations=_TRANSFORMS,
        evaluate=True,
    )


def parse_equations_pse(text: str) -> Tuple[sp.Expr, sp.Expr]:
    m0 = re.search(r"dx\s*\(\s*0\s*\)\s*/\s*dt\s*=\s*(.+)", text)
    m1 = re.search(r"dx\s*\(\s*1\s*\)\s*/\s*dt\s*=\s*(.+)", text)
    if not (m0 and m1):
        raise ValueError("PSE file must contain lines: dx(0)/dt = ... and dx(1)/dt = ...")

    rhs0 = m0.group(1).strip()
    rhs1 = m1.group(1).strip()

    x0 = sp.Symbol("x0")
    x1 = sp.Symbol("x1")
    local = {"x0": x0, "x1": x1, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}

    e0 = sp.simplify(sp.expand(_parse_sympy(rhs0, local)))
    e1 = sp.simplify(sp.expand(_parse_sympy(rhs1, local)))
    e0 = sp.simplify(sp.expand(e0.subs({x0: X1, x1: X2})))
    e1 = sp.simplify(sp.expand(e1.subs({x0: X1, x1: X2})))
    return e0, e1


def parse_equations_handi_style(text: str) -> Tuple[sp.Expr, sp.Expr]:
    m1 = re.search(r"dx\s*\(\s*1\s*\)\s*/\s*dt\s*=\s*(.+)", text)
    m2 = re.search(r"dx\s*\(\s*2\s*\)\s*/\s*dt\s*=\s*(.+)", text)
    if not (m1 and m2):
        raise ValueError("Equation file must contain lines: dx(1)/dt = ... and dx(2)/dt = ...")

    rhs1 = m1.group(1).strip()
    rhs2 = m2.group(1).strip()

    rhs1 = re.sub(r"x\s*\(\s*1\s*\)", "x1", rhs1)
    rhs1 = re.sub(r"x\s*\(\s*2\s*\)", "x2", rhs1)
    rhs2 = re.sub(r"x\s*\(\s*1\s*\)", "x1", rhs2)
    rhs2 = re.sub(r"x\s*\(\s*2\s*\)", "x2", rhs2)

    local = {"x1": X1, "x2": X2, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}
    e1 = sp.simplify(_parse_sympy(rhs1, local))
    e2 = sp.simplify(_parse_sympy(rhs2, local))
    return e1, e2


def sympy_to_handi_str(expr: sp.Expr) -> str:
    expr = sp.expand(expr)

    def _pow_to_mul(e):
        if isinstance(e, sp.Pow):
            base, exp = e.args
            if exp.is_Integer:
                p = int(exp)
                if 2 <= p <= 8:
                    out = base
                    for _ in range(p - 1):
                        out = out * base
                    return out
        return e

    expr2 = expr.replace(lambda e: isinstance(e, sp.Pow), _pow_to_mul)
    s = sp.sstr(expr2)
    s = re.sub(r"\bx1\b", "x(1)", s)
    s = re.sub(r"\bx2\b", "x(2)", s)
    return s


# -------------------- Bounds & Integration --------------------

def compute_bounds_from_handi_single(
    handi_pred: Optional[np.ndarray],
    fallback_truth_train: np.ndarray,
    fallback_truth_hires: Optional[np.ndarray],
    scale: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    vals: List[np.ndarray] = []
    if handi_pred is not None:
        P = np.asarray(handi_pred)
        if P.ndim == 2 and P.shape[1] >= 2:
            mask = np.isfinite(P[:, 0]) & np.isfinite(P[:, 1])
            if np.any(mask):
                vals.append(P[mask, :2])

    if len(vals) > 0:
        A = np.concatenate(vals, axis=0)
    else:
        A = (fallback_truth_hires.reshape(-1, 2) if fallback_truth_hires is not None
             else fallback_truth_train.reshape(-1, 2))

    low = np.nanmin(A, axis=0)
    high = np.nanmax(A, axis=0)
    return (scale * low).astype(float), (scale * high).astype(float)


def make_rhs_func(expr1: sp.Expr, expr2: sp.Expr) -> Callable[[float, np.ndarray], List[float]]:
    f1 = sp.lambdify((X1, X2), expr1, modules=["numpy"])
    f2 = sp.lambdify((X1, X2), expr2, modules=["numpy"])

    def f(t: float, y: np.ndarray) -> List[float]:
        x1v, x2v = float(y[0]), float(y[1])
        try:
            dx1 = f1(x1v, x2v)
            dx2 = f2(x1v, x2v)
            if not (np.isfinite(dx1) and np.isfinite(dx2)):
                return [np.nan, np.nan]
            return [dx1, dx2]
        except Exception:
            return [np.nan, np.nan]

    return f


def integrate_traj(
    f: Callable[[float, np.ndarray], List[float]],
    x0: List[float],
    t_eval: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: Optional[float] = None,
    bounds_low: Optional[np.ndarray] = None,
    bounds_high: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    T = len(t_eval)
    t0, t1 = float(t_eval[0]), float(t_eval[-1])

    if bounds_low is not None and bounds_high is not None:
        y0 = np.asarray(x0, dtype=float)
        if np.any(y0 < bounds_low) or np.any(y0 > bounds_high):
            return np.full((T, 2), np.nan, dtype=float)

    events = None
    if bounds_low is not None and bounds_high is not None:
        low = np.asarray(bounds_low, dtype=float)
        high = np.asarray(bounds_high, dtype=float)

        def diverge_event(t, y):
            g1 = y[0] - low[0]
            g2 = high[0] - y[0]
            g3 = y[1] - low[1]
            g4 = high[1] - y[1]
            return min(g1, g2, g3, g4)

        diverge_event.terminal = True
        diverge_event.direction = -1
        events = [diverge_event]

    solve_kwargs = dict(
        fun=f,
        t_span=(t0, t1),
        y0=np.asarray(x0, dtype=float),
        t_eval=np.asarray(t_eval, float),
        method="RK45",
        rtol=rtol,
        atol=atol,
    )

    if (max_step is not None) and (float(max_step) > 0.0):
        solve_kwargs["max_step"] = float(max_step)
    if events is not None:
        solve_kwargs["events"] = events

    sol = solve_ivp(**solve_kwargs)
    if sol.y is None or sol.y.size == 0:
        return None

    Y = sol.y.T
    if np.any(~np.isfinite(Y)):
        return None

    if Y.shape[0] == T:
        return Y

    out = np.full((T, 2), np.nan, dtype=float)
    out[:Y.shape[0], :] = Y[:, :2]
    return out


# -------------------- Plotting --------------------

def _make_plot_indices(T: int, plot_step: int, force_last: bool) -> np.ndarray:
    """
    Plot every plot_step points.
    If force_last=True, ensure last index is included.
    """
    ps = int(plot_step) if int(plot_step) >= 1 else 1
    idx = np.arange(0, T, ps, dtype=int)
    if idx.size == 0:
        return np.array([0], dtype=int)
    if force_last and idx[-1] != T - 1:
        idx = np.concatenate([idx, np.array([T - 1], dtype=int)])
    return idx


def plot_overlay_timeseries_single(
    truth_scatter: np.ndarray,   # (Tg,2) original sampled truth (dt grid) for dots
    t_scatter: np.ndarray,       # (Tg,)
    truth_line: np.ndarray,      # (Te,2) truth resampled to exact t_eval for dashed line
    t_eval: np.ndarray,          # (Te,)
    preds: Dict[str, Optional[np.ndarray]],  # method -> (Te,2)
    out_path: str,
    *,
    marker_step: int = 1,
    plot_step: int = 1,
    force_last_plot: bool = False,
    hspace: float = 0.35,
    ylim_mode: str = "truth",    # truth / auto
    ylim_scale: float = 1.05,
) -> None:
    Te = len(t_eval)
    t_eval = np.asarray(t_eval, float)

    idx_plot = _make_plot_indices(Te, plot_step, force_last_plot)
    t_plot = t_eval[idx_plot]

    # x-axis end MUST be exact last time
    t_end = float(t_eval[-1])
    x_left = float(t_eval[0])
    xspan = t_end - x_left

    fig, axes = plt.subplots(2, 1, figsize=(STYLE.fig_w, STYLE.fig_h), sharex=True)
    fig.subplots_adjust(hspace=float(hspace))

    # methods
    for method, P in preds.items():
        if P is None:
            continue
        color = STYLE.colors.get(method, "black")
        plot_smooth_curve(
            axes[0], t_plot, P[idx_plot, 0],
            color=color, lw=STYLE.method_lw, alpha=STYLE.method_alpha,
            dense_factor=STYLE.smooth_dense_factor, zorder=2
        )
        plot_smooth_curve(
            axes[1], t_plot, P[idx_plot, 1],
            color=color, lw=STYLE.method_lw, alpha=STYLE.method_alpha,
            dense_factor=STYLE.smooth_dense_factor, zorder=2
        )

    # truth: line reaches exact tmax, dots are original samples only
    plot_truth_line_plus_scatter(
        axes[0],
        t_scatter=t_scatter, y_scatter=truth_scatter[:, 0], marker_step=marker_step,
        t_line=t_eval, y_line=truth_line[:, 0],
        color=STYLE.truth_color, lw=STYLE.truth_lw, alpha=STYLE.truth_alpha,
        dash=STYLE.truth_dash, s=STYLE.truth_scatter_s, edge_lw=STYLE.truth_scatter_edge_lw
    )
    plot_truth_line_plus_scatter(
        axes[1],
        t_scatter=t_scatter, y_scatter=truth_scatter[:, 1], marker_step=marker_step,
        t_line=t_eval, y_line=truth_line[:, 1],
        color=STYLE.truth_color, lw=STYLE.truth_lw, alpha=STYLE.truth_alpha,
        dash=STYLE.truth_dash, s=STYLE.truth_scatter_s, edge_lw=STYLE.truth_scatter_edge_lw
    )

    # y-lims (optional): fix based on truth_line for layout stability
    if str(ylim_mode).lower() == "truth":
        for di, ax in enumerate(axes):
            yy = np.asarray(truth_line[:, di], float)
            m = np.isfinite(yy)
            if not np.any(m):
                continue
            lo = float(np.min(yy[m]))
            hi = float(np.max(yy[m]))
            c = 0.5 * (lo + hi)
            span = max(hi - lo, 1e-12)
            half = 0.5 * span * float(ylim_scale)
            ax.set_ylim(c - half, c + half)

    # right padding
    right_pad = 0.02 * max(xspan, 1e-12)
    # dt-like minimum padding: infer typical step from t_eval
    if len(t_eval) >= 2:
        dt_typ = float(np.median(np.diff(t_eval[:-1])) if len(t_eval) > 2 else (t_eval[1] - t_eval[0]))
    else:
        dt_typ = 0.0
    right_pad = max(right_pad, 0.5 * max(dt_typ, 1e-12))
    x_right = t_end + right_pad

    for ax in axes:
        ax.margins(x=0)
        ax.set_xlim(x_left, x_right)
        apply_ticks_adaptive(ax, xspan=(x_right - x_left))
        apply_spines(ax)

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser("Overlay truth + PSE/HANDI/SINDy/SR3 (time series only, SINGLE trajectory)")

    ap.add_argument("--truth-npy", default="wheel_shimmy_train_ds_22.npy")
    ap.add_argument("--dt", type=float, default=0.11)

    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=1.2,
                    help="Max time to use (same unit as dt). Overrides --T if set.")
    ap.add_argument("--tmax-mode", type=str, default="exact",
                    choices=["exact", "floor", "round", "ceil"],
                    help="How to map tmax onto the time grid. Default: exact (append tmax).")

    ap.add_argument("--traj-idx", type=int, default=0)

    ap.add_argument("--truth-npy-hires", default="wheel_shimmy_train.npy")
    ap.add_argument("--dt-hires", type=float, default=0.005)

    ap.add_argument("--out-dir", default="plots")

    ap.add_argument("--pse-eq", default=r".\PSE\0.08+4\final_ode_system_from_psrn.txt")
    ap.add_argument("--handi-eq", default=r".\HANDI\find_results\run_ds_16_dt_0.0800\best_equations_by_mse_dt0.080.txt")
    ap.add_argument("--sindy-eq", default=r".\Sindy\sindy_wheel\sindy_equation_best_wheel_shimmy_train_stride16.txt")
    ap.add_argument("--sr3-eq", default=r".\SR3\sindy_sr3_wheel\wheel_dt0.08\sindy_sr3_equations.txt")

    ap.add_argument("--rtol", type=float, default=1e-7)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--max-step", type=float, default=0.0)

    ap.add_argument("--bounds-scale", type=float, default=3.0)

    ap.add_argument("--smooth-factor", type=int, default=12)
    ap.add_argument("--marker-step", type=int, default=1)
    ap.add_argument("--plot-step", type=int, default=1)
    ap.add_argument("--force-last-plot", action="store_true", default=False,
                    help="If set, force the last t_eval point to be included in plot subsampling.")

    ap.add_argument("--truth-lw", type=float, default=2.6)
    ap.add_argument("--truth-dash-on", type=int, default=1)
    ap.add_argument("--truth-dash-off", type=int, default=1)
    ap.add_argument("--truth-dot-s", type=float, default=55.0)
    ap.add_argument("--truth-dot-edge-lw", type=float, default=1.8)

    ap.add_argument("--hspace", type=float, default=0.35,
                    help="Vertical space between the two subplots.")

    ap.add_argument("--ylim-mode", type=str, default="truth", choices=["truth", "auto"])
    ap.add_argument("--ylim-scale", type=float, default=1.05)

    args = ap.parse_args()

    STYLE.smooth_dense_factor = int(args.smooth_factor)
    STYLE.truth_lw = float(args.truth_lw)
    STYLE.truth_dash = (int(args.truth_dash_on), int(args.truth_dash_off))
    STYLE.truth_scatter_s = float(args.truth_dot_s)
    STYLE.truth_scatter_edge_lw = float(args.truth_dot_edge_lw)

    truth_all = load_npy_safely(args.truth_npy)[:, :, :2]
    N, T_full, _ = truth_all.shape

    traj_idx = int(args.traj_idx)
    if traj_idx < 0 or traj_idx >= N:
        raise ValueError(f"--traj-idx out of range: got {traj_idx}, but N={N}")

    dt = float(args.dt)

    # ----- build t_eval (default exact) -----
    if args.tmax is not None and float(args.tmax) > 0:
        t_eval = build_time_grid_with_tmax(dt=dt, tmax=float(args.tmax), mode=str(args.tmax_mode))
    else:
        T = int(args.T) if args.T and int(args.T) > 0 else T_full
        T = max(2, min(T, T_full))
        t_eval = np.arange(T, dtype=float) * dt

    # truth samples available only on dt-grid:
    # use grid length up to floor(tmax/dt)+1 (or T_full if smaller)
    tmax_effective = float(t_eval[-1])
    n_grid = int(np.floor(tmax_effective / dt)) + 1
    n_grid = max(2, min(n_grid, T_full))

    truth_grid = truth_all[traj_idx, :n_grid, :]
    t_grid = np.arange(n_grid, dtype=float) * dt

    # build truth_line on t_eval (interpolate for exact tmax)
    truth_line = _resample_2d_traj_to_times(t_old=t_grid, y_old=truth_grid, t_new=t_eval)

    # optional hires truth
    truth_hires_line = None
    if args.truth_npy_hires and args.truth_npy_hires.strip():
        hi_all = load_npy_safely(args.truth_npy_hires)[:, :, :2]
        Nh, Th, _ = hi_all.shape
        if Nh == N:
            hi = hi_all[traj_idx, :, :]
        elif Nh == 1:
            hi = hi_all[0, :, :]
        else:
            raise ValueError(f"hires truth N={Nh} mismatch train N={N} and is not 1")

        dt_h = float(args.dt_hires) if float(args.dt_hires) > 0 else dt
        t_hi = np.arange(hi.shape[0], dtype=float) * dt_h

        # Keep hires samples within [0, tmax_effective], then interpolate to t_eval for exact alignment
        keep = t_hi <= (tmax_effective + 1e-12)
        if np.any(keep):
            hi = hi[:np.sum(keep), :]
            t_hi = t_hi[:np.sum(keep)]
            truth_hires_line = _resample_2d_traj_to_times(t_old=t_hi, y_old=hi, t_new=t_eval)

    # prefer hires line if available
    if truth_hires_line is not None and np.isfinite(truth_hires_line).any():
        truth_line_to_use = truth_hires_line
    else:
        truth_line_to_use = truth_line

    # parse equations
    pse_e1, pse_e2 = parse_equations_pse(_read_text(args.pse_eq))
    handi_e1, handi_e2 = parse_equations_handi_style(_read_text(args.handi_eq))
    sindy_e1, sindy_e2 = parse_equations_handi_style(_read_text(args.sindy_eq))
    sr3_e1, sr3_e2 = parse_equations_handi_style(_read_text(args.sr3_eq))

    os.makedirs(args.out_dir, exist_ok=True)
    uni_path = osp.join(args.out_dir, "unified_equations.txt")
    with open(uni_path, "w", encoding="utf-8") as f:
        f.write("=== Unified equations (canonical vars x(1), x(2)) ===\n\n")
        f.write("[PSE]\n")
        f.write("dx(1)/dt = " + sympy_to_handi_str(pse_e1) + "\n")
        f.write("dx(2)/dt = " + sympy_to_handi_str(pse_e2) + "\n\n")
        f.write("[HANDI]\n")
        f.write("dx(1)/dt = " + sympy_to_handi_str(handi_e1) + "\n")
        f.write("dx(2)/dt = " + sympy_to_handi_str(handi_e2) + "\n\n")
        f.write("[SINDy]\n")
        f.write("dx(1)/dt = " + sympy_to_handi_str(sindy_e1) + "\n")
        f.write("dx(2)/dt = " + sympy_to_handi_str(sindy_e2) + "\n\n")
        f.write("[SR3]\n")
        f.write("dx(1)/dt = " + sympy_to_handi_str(sr3_e1) + "\n")
        f.write("dx(2)/dt = " + sympy_to_handi_str(sr3_e2) + "\n")

    rhs = {
        "PSE":   make_rhs_func(pse_e1, pse_e2),
        "HANDI": make_rhs_func(handi_e1, handi_e2),
        "SINDy": make_rhs_func(sindy_e1, sindy_e2),
        "SR3":   make_rhs_func(sr3_e1, sr3_e2),
    }

    max_step = None if float(args.max_step) <= 0.0 else float(args.max_step)

    x0 = truth_grid[0, :].tolist()

    # bounds from HANDI (no bounds) on the SAME t_eval
    handi_pred_nobounds = integrate_traj(
        rhs["HANDI"], x0=x0, t_eval=t_eval,
        rtol=args.rtol, atol=args.atol, max_step=max_step,
        bounds_low=None, bounds_high=None
    )

    bounds_low, bounds_high = compute_bounds_from_handi_single(
        handi_pred=handi_pred_nobounds,
        fallback_truth_train=truth_grid,
        fallback_truth_hires=(truth_line_to_use if truth_line_to_use is not None else None),
        scale=float(args.bounds_scale),
    )

    preds: Dict[str, Optional[np.ndarray]] = {}
    for method, f in rhs.items():
        preds[method] = integrate_traj(
            f, x0=x0, t_eval=t_eval,
            rtol=args.rtol, atol=args.atol, max_step=max_step,
            bounds_low=bounds_low, bounds_high=bounds_high
        )

    out_svg = osp.join(args.out_dir, "overlay_timeseries_5curves.svg")
    plot_overlay_timeseries_single(
        truth_scatter=truth_grid,
        t_scatter=t_grid,
        truth_line=truth_line_to_use,
        t_eval=t_eval,
        preds=preds,
        out_path=out_svg,
        marker_step=int(args.marker_step),
        plot_step=int(args.plot_step),
        force_last_plot=bool(args.force_last_plot),
        hspace=float(args.hspace),
        ylim_mode=str(args.ylim_mode),
        ylim_scale=float(args.ylim_scale),
    )

    print("Saved:", out_svg)
    print("Saved:", uni_path)
    print(f"[t_eval] last = {t_eval[-1]:.10g}  (tmax-mode={args.tmax_mode})")


if __name__ == "__main__":
    main()
