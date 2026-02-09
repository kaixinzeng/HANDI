#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
id31 (2D Phase Plane System)

Features:
1) True vector field + Trajectories + SAMPLES (dt = 0.3)
2) Recovered vector field + Trajectories
3) Streamlines (True)
4) Streamlines (Predicted)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator


# ============================================================
# 0) Global plotting style
# ============================================================
def set_style_like_ref():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 46,

        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.grid": False,
        "svg.fonttype": "none",
    })


def beautify_axes(ax):
    # Hide top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3.0)
    ax.spines["bottom"].set_linewidth(3.0)

    # Tick style
    ax.tick_params(axis="x", which="major", labelsize=30, length=6, width=1.5)
    ax.tick_params(axis="y", which="major", labelsize=30, length=6, width=1.5)

    # X axis: use 3 ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

    # Y axis: custom tick logic (remove zero, keep top/bottom)
    ymin, ymax = ax.get_ylim()
    cand = MaxNLocator(nbins=3).tick_values(ymin, ymax)
    cand = [v for v in cand if ymin <= v <= ymax and abs(v) > 1e-12]

    if len(cand) >= 2:
        yticks = [cand[0], cand[-1]]
    else:
        yticks = [
            ymin + 0.35 * (ymax - ymin),
            ymin + 0.75 * (ymax - ymin),
        ]

    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))


# ============================================================
# 1) id31 system (true & recovered dynamics)
# ============================================================
def f_true(t, x):
    x0, x1 = x
    return np.array([
        -0.4 * x0 * x1,
        0.4 * x0 * x1 - 0.314 * x1
    ], dtype=float)


def f_pred(t, x):
    x0, x1 = x
    return np.array([
        -0.336855 * x0 * x1
        - 0.233229 * x0
        - 0.0184153 * x0 * x0
        + 0.00168879 * x1
        - 0.000225622 * x1 * x1,

        0.34006 * x0 * x1
        - 0.313728 * x1
        + 0.196023 * x0
        + 0.0212828 * x0 * x0
        + 8.1152e-05 * x1 * x1
    ], dtype=float)


# ============================================================
# 2) Trajectory integration & sample generation
# ============================================================
def integrate_traj(f, x0, t_end=60.0, n_points=9000):
    """
    Generate a high-resolution continuous trajectory for smooth plotting.
    """
    t_eval = np.linspace(0.0, t_end, n_points)

    def f_wrap(t, x):
        # Enforce non-negativity as in the original model
        x = np.maximum(x, 0.0)
        return f(t, x)

    sol = solve_ivp(
        f_wrap, (0.0, t_end), np.asarray(x0, float),
        t_eval=t_eval, rtol=1e-9, atol=1e-12
    )
    return np.maximum(sol.y.T, 0.0)


def generate_samples(f, x0, t_end=60.0, dt=0.3):
    """
    Generate discrete samples with step dt.

    Returns:
        samples: array of shape (N, 2)
    """
    # Generate time grid from 0 to t_end with step dt
    t_eval = np.arange(0.0, t_end + 1e-9, dt)

    def f_wrap(t, x):
        x = np.maximum(x, 0.0)
        return f(t, x)

    sol = solve_ivp(
        f_wrap, (0.0, t_end), np.asarray(x0, float),
        t_eval=t_eval, rtol=1e-9, atol=1e-12
    )

    # Optionally add small noise to mimic observations
    # samples = sol.y.T + np.random.normal(0, 0.05, size=sol.y.T.shape)

    # Here we keep clean samples
    samples = np.maximum(sol.y.T, 0.0)
    return samples


# ============================================================
# 3) Vector field grid construction
# ============================================================
def vector_field_grid(f, x_range, y_range, n=60):
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(n):
        for j in range(n):
            u, v = f(0.0, [X[i, j], Y[i, j]])
            U[i, j] = u
            V[i, j] = v

    speed = np.sqrt(U ** 2 + V ** 2)
    return X, Y, U, V, speed


# ============================================================
# 4) Automatic plotting bounds from trajectories
# ============================================================
def bounds_from_trajs(trajs, margin=0.08):
    XY = np.vstack(trajs)
    xmin, ymin = XY.min(axis=0)
    xmax, ymax = XY.max(axis=0)
    dx = max(xmax - xmin, 1e-12)
    dy = max(ymax - ymin, 1e-12)

    return (
        max(0.0, xmin - margin * dx),
        xmax + margin * dx
    ), (
        max(0.0, ymin - margin * dy),
        ymax + margin * dy
    )


# ============================================================
# 5) Vector-field error metrics
# ============================================================
def vector_field_errors(f_ref, f_hat, x_range, y_range, n=80):
    X, Y, U, V, _ = vector_field_grid(f_ref, x_range, y_range, n)

    err2 = 0.0
    ref2 = 0.0

    for i in range(n):
        for j in range(n):
            ft = np.array(f_ref(0.0, [X[i, j], Y[i, j]]))
            fh = np.array(f_hat(0.0, [X[i, j], Y[i, j]]))
            diff = fh - ft
            err2 += np.dot(diff, diff)
            ref2 += np.dot(ft, ft)

    E_L2 = np.sqrt(err2 / (n * n))
    E_rel = np.sqrt(err2 / ref2)
    return E_L2, E_rel


# ============================================================
# 6) Figure A: True field + true trajectories + samples
# ============================================================
def plot_true_field_with_samples(x_range, y_range, trajs_true, samples_list=None, filename=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    # 1. Background vector field (normalized streamplot)
    X, Y, U, V, speed = vector_field_grid(f_true, x_range, y_range)
    eps = 1e-6
    ax.streamplot(
        X, Y, U / (speed + eps), V / (speed + eps),
        color="0.75", linewidth=1.8, density=1.3
    )

    # 2. Continuous true trajectories (solid black)
    for T in trajs_true:
        ax.plot(T[:, 0], T[:, 1], color="black", lw=5.2, zorder=5)

    # 3. Discrete samples (dt = 0.3), white-filled circles with black edges
    if samples_list:
        for S in samples_list:
            ax.scatter(
                S[:, 0], S[:, 1],
                s=150,
                marker='o',
                facecolors='white',
                edgecolors='black',
                linewidths=2.5,
                zorder=10
            )

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    beautify_axes(ax)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format="svg", bbox_inches="tight")
        print(f"Saved: {filename}")

    plt.show()


# ============================================================
# 7) Figure B: Recovered field + true / predicted trajectories
# ============================================================
def plot_recovered_field(x_range, y_range, trajs_true, trajs_pred, filename=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    X, Y, U, V, speed = vector_field_grid(f_pred, x_range, y_range)
    eps = 1e-6
    ax.streamplot(
        X, Y, U / (speed + eps), V / (speed + eps),
        color="0.80", linewidth=1.8, density=1.3
    )

    # True trajectories (solid black)
    for T in trajs_true:
        ax.plot(T[:, 0], T[:, 1], color="black", lw=5.2)

    # Predicted trajectories (dashed blue)
    for T in trajs_pred:
        ax.plot(T[:, 0], T[:, 1], color="#4FA8D5", lw=5.2, ls="--")

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    beautify_axes(ax)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format="svg", bbox_inches="tight")
        print(f"Saved: {filename}")

    plt.show()


# ============================================================
# 8) Figure D: Streamline family
# ============================================================
def plot_streamline_family(f, x_range, y_range, color, filename=None):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)

    # Seed points for forward/backward integration
    seeds_x = np.linspace(x_range[0] * 1.05, x_range[1] * 0.95, 10)
    seeds_y = np.linspace(y_range[0] * 1.05, y_range[1] * 0.95, 8)

    for x in seeds_x:
        for y in seeds_y:
            for sign in [+1, -1]:
                sol = solve_ivp(
                    lambda t, z: sign * f(0.0, z),
                    (0, 6), [x, y],
                    rtol=1e-6, atol=1e-9
                )
                Z = np.maximum(sol.y.T, 0.0)
                ax.plot(Z[:, 0], Z[:, 1], color=color, lw=1.1, alpha=0.25)

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    beautify_axes(ax)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format="svg", bbox_inches="tight")
        print(f"Saved: {filename}")

    plt.show()


# ============================================================
# 9) Main program
# ============================================================
def main():
    set_style_like_ref()

    # Initial conditions
    inits = [[7.2, 0.98], [20.0, 12.4]]
    T_END = 60.0
    DT_SAMPLE = 0.3

    # 1. High-resolution trajectories for smooth curves
    trajs_true = [integrate_traj(f_true, x0, t_end=T_END) for x0 in inits]
    trajs_pred = [integrate_traj(f_pred, x0, t_end=T_END) for x0 in inits]

    # 2. Low-resolution samples (dt = 0.3) representing observations
    samples_list = [
        generate_samples(f_true, x0, t_end=T_END, dt=DT_SAMPLE)
        for x0 in inits
    ]

    # 3. Plotting bounds
    xA, yA = bounds_from_trajs(trajs_true)
    xB, yB = bounds_from_trajs(trajs_true + trajs_pred)

    # 4. Compute and save vector-field errors
    E_L2, E_rel = vector_field_errors(f_true, f_pred, xB, yB)

    with open("id31_vectorfield_metrics.txt", "w") as f:
        f.write(f"x_range = {xB}\n")
        f.write(f"y_range = {yB}\n")
        f.write(f"E_L2  = {E_L2:.6e}\n")
        f.write(f"E_rel = {E_rel:.6e}\n")

    # 5. Generate and save figures

    # Figure 1: True field + trajectories + samples
    plot_true_field_with_samples(
        xA, yA, trajs_true, samples_list,
        filename="1_true_vector_field.svg"
    )

    # Figure 2: Recovered field + overlay
    plot_recovered_field(
        xB, yB, trajs_true, trajs_pred,
        filename="2_recovered_vector_field.svg"
    )

    # Figure 3: True streamlines
    plot_streamline_family(
        f_true, xB, yB, color="0.2",
        filename="3_true_streamlines.svg"
    )

    # Figure 4: Predicted streamlines
    plot_streamline_family(
        f_pred, xB, yB, color="#D55E00",
        filename="4_pred_streamlines.svg"
    )

    print("[OK] Done.")


if __name__ == "__main__":
    main()
