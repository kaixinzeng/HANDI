#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
id39 (2D Phase Plane System) - Smart Filtering Version
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# =============================
# 0) Style
# =============================
def set_ncs_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 46,
        "axes.labelsize": 18,
        "axes.titlesize": 14,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "axes.grid": False,
        "svg.fonttype": "none",
    })


def beautify_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3.0)
    ax.spines["bottom"].set_linewidth(3.0)
    ax.tick_params(axis="x", which="major", labelsize=27, length=6, width=1.5)
    ax.tick_params(axis="y", which="major", labelsize=27, length=6, width=1.5)


# =============================
# 1) System Definitions
# =============================
def f_true(t, x):
    x0, x1 = x
    dx0 = 2.4 * x1 - 1.0 * x0 + 1.0 * (x0 * x0 * x1)
    dx1 = 0.07 - 2.4 * x0 - 1.0 * (x0 * x0 * x1)
    return np.array([dx0, dx1], dtype=float)


def f_pred(t, x):
    x0, x1 = x
    dx0 = (2.3999 * x1 - 1.00101 * x0 + 0.98595 * (x0 * x0 * x1)
           + 0.0257401 * (x0 * x1 * x1) + 0.00333737 * (x1 * x1 * x1)
           + 0.00240551 * (x0 * x1) + 0.00237569 * (x0 * x0)
           + 0.00120466 * (x0 * x0 * x0) - 0.000111829 * (x1 * x1)
           - 4.57825e-05 * 1.0)
    dx1 = (-2.39948 * x0 - 0.983099 * (x0 * x0 * x1) + 0.0700334 * 1.0
           - 0.0145857 * (x0 * x1 * x1) - 0.00597264 * (x1 * x1 * x1)
           - 0.00293468 * (x0 * x0 * x0) - 0.00278489 * (x0 * x1)
           - 0.00177572 * (x0 * x0) + 0.000297233 * (x1 * x1)
           - 0.000217666 * x1)
    return np.array([dx0, dx1], dtype=float)


# =============================
# 2) Integration & Sampling
# =============================
def integrate_traj(f, x0, t_end=60.0, n_points=3000):
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(lambda t, x: f(t, x), (0.0, t_end), np.array(x0), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return t_eval, sol.y.T


def generate_samples(f, x0, t_end=60.0, dt=0.3):
    t_eval = np.arange(0.0, t_end + 1e-9, dt)
    sol = solve_ivp(lambda t, x: f(t, x), (0.0, t_end), np.array(x0), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol.y.T


# ============================================================
#  (Smart Filter)
# ============================================================
def filter_points_smart(points, min_dist=0.08, center_radius=0.15):

    if len(points) == 0:
        return np.array([])

    center_point = points[-1]

    keep = []
    keep.append(points[0])
    last_p = points[0]

    for i in range(1, len(points)):
        curr_p = points[i]

        dist_prev = np.linalg.norm(curr_p - last_p)

        dist_center = np.linalg.norm(curr_p - center_point)

        if dist_prev >= min_dist and dist_center >= center_radius:
            keep.append(curr_p)
            last_p = curr_p

    return np.array(keep)


# =============================
# 3) Helper Functions
# =============================
def _nice_step(span, target_divs=2):
    raw = span / max(target_divs, 1)
    if raw <= 0: return 1.0
    exp = np.floor(np.log10(raw))
    base = raw / (10 ** exp)
    if base <= 1:
        nice = 1
    elif base <= 2:
        nice = 2
    elif base <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exp)


def auto_phase_range_nice(inits, t_end=60.0, n_points=6000, pad=0.06, extra=1.02):
    xs, ys = [], []
    for x0 in inits:
        _, Xt = integrate_traj(f_true, x0, t_end, n_points)
        _, Xp = integrate_traj(f_pred, x0, t_end, n_points)
        X = np.vstack([Xt, Xp])
        xs.append(X[:, 0]);
        ys.append(X[:, 1])
    xs = np.concatenate(xs);
    ys = np.concatenate(ys)
    xlo, xhi = float(xs.min()), float(xs.max())
    ylo, yhi = float(ys.min()), float(ys.max())
    dx = xhi - xlo if xhi > xlo else 1.0
    dy = yhi - ylo if yhi > ylo else 1.0
    xlo -= pad * dx;
    xhi += pad * dx
    ylo -= pad * dy;
    yhi += pad * dy
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    h = max(0.5 * (xhi - xlo), 0.5 * (yhi - ylo)) * extra
    step = _nice_step(2.0 * h, 2)
    xlo_n = step * np.floor((cx - h) / step)
    xhi_n = step * np.ceil((cx + h) / step)
    ylo_n = step * np.floor((cy - h) / step)
    yhi_n = step * np.ceil((cy + h) / step)
    return (xlo_n, xhi_n), (ylo_n, yhi_n), step


def set_three_ticks(ax, x_range, y_range):
    ax.set_xticks([x_range[0], 0.5 * (x_range[0] + x_range[1]), x_range[1]])
    ax.set_yticks([y_range[0], 0.5 * (y_range[0] + y_range[1]), y_range[1]])


def vector_field_grid(f, x0_range, x1_range, grid_n=50):
    X0, X1 = np.meshgrid(np.linspace(*x0_range, grid_n), np.linspace(*x1_range, grid_n))
    U, V = np.zeros_like(X0), np.zeros_like(X1)
    for i in range(grid_n):
        for j in range(grid_n):
            u, v = f(0.0, [X0[i, j], X1[i, j]])
            U[i, j], V[i, j] = u, v
    return X0, X1, U, V, np.sqrt(U ** 2 + V ** 2)


# =============================
# 4) Plotting
# =============================
def plot_phase_panel(f_field, inits, x0_range, x1_range, draw_recovered=True,
                     samples_list=None, t_end=60.0, n_points=3000, filename=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Vector field
    X0, X1, U, V, speed = vector_field_grid(f_field, x0_range, x1_range)
    ax.streamplot(X0, X1, U / (speed + 1e-6), V / (speed + 1e-6), density=1.35, color="0.80",
                  linewidth=1.8, arrowsize=0.8, zorder=1)

    for i, x0 in enumerate(inits):
        _, X_true = integrate_traj(f_true, x0, t_end, n_points)
        _, X_pred = integrate_traj(f_pred, x0, t_end, n_points)

        # True Trajectory (Black Line)
        ax.plot(X_true[:, 0], X_true[:, 1], "-", lw=5.2, color="black", zorder=3)

        if draw_recovered:
            ax.plot(X_pred[:, 0], X_pred[:, 1], "--", lw=5.2, color="#4FA8D5", zorder=3)

        # Samples (Smart filtered)
        if samples_list:
            S = samples_list[i]
            # min_dist=0.08
            # center_radius=0.15
            S_plot = filter_points_smart(S, min_dist=0.08, center_radius=0.15)

            ax.scatter(S_plot[:, 0], S_plot[:, 1], s=150, marker='o',
                       facecolors='white', edgecolors='black', linewidths=2.5, zorder=10)

    ax.set_xlim(*x0_range)
    ax.set_ylim(*x1_range)
    set_three_ticks(ax, x0_range, x1_range)
    ax.set_xlabel("");
    ax.set_ylabel("");
    ax.set_title("")
    beautify_axes(ax)
    plt.tight_layout()
    if filename: plt.savefig(filename, format="svg", bbox_inches="tight"); print(f"Saved: {filename}")
    plt.show()


def plot_streamline_family(f_field, seeds, x0_range, x1_range, color, filename=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (x, y) in seeds:
        t = np.linspace(0, 6.0, 160)
        sol_f = solve_ivp(lambda t, x: f_field(0, x), (0, 6), [x, y], t_eval=t)
        sol_b = solve_ivp(lambda t, x: -f_field(0, x), (0, 6), [x, y], t_eval=t)
        ax.plot(sol_f.y[0], sol_f.y[1], color=color, lw=1.1, alpha=0.25)
        ax.plot(sol_b.y[0], sol_b.y[1], color=color, lw=1.1, alpha=0.25)
    ax.set_xlim(*x0_range);
    ax.set_ylim(*x1_range)
    set_three_ticks(ax, x0_range, x1_range)
    ax.set_xlabel("");
    ax.set_ylabel("");
    ax.set_title("")
    beautify_axes(ax)
    plt.tight_layout()
    if filename: plt.savefig(filename, format="svg", bbox_inches="tight"); print(f"Saved: {filename}")
    plt.show()


# =============================
# 5) Main
# =============================
def main():
    set_ncs_style()
    inits = [[0.35, 0.31], [-0.42, 0.28]]
    t_end = 60.0
    DT_SAMPLE = 0.3

    x0_range, x1_range, step = auto_phase_range_nice(inits, t_end, 6000)

    samples_list = [generate_samples(f_true, x0, t_end, DT_SAMPLE) for x0 in inits]

    # Figure 1: True + Traj + Samples (Smart Filtered)
    plot_phase_panel(f_true, inits, x0_range, x1_range, draw_recovered=False,
                     samples_list=samples_list, filename="1_id39_phase_true.svg")

    # Figure 2: Rec + Traj
    plot_phase_panel(f_pred, inits, x0_range, x1_range, draw_recovered=True,
                     filename="2_id39_phase_recovered.svg")

    seeds = []
    nx, ny = 10, 8
    margin = 0.08
    dx = (x0_range[1] - x0_range[0]) * margin
    dy = (x1_range[1] - x1_range[0]) * margin
    for x in np.linspace(x0_range[0] + dx, x0_range[1] - dx, nx):
        for y in np.linspace(x1_range[0] + dy, x1_range[1] - dy, ny):
            seeds.append((x, y))

    plot_streamline_family(f_true, seeds, x0_range, x1_range, "0.20", "3_id39_streamlines_true.svg")
    plot_streamline_family(f_pred, seeds, x0_range, x1_range, "#D55E00", "4_id39_streamlines_pred.svg")


if __name__ == "__main__":
    main()