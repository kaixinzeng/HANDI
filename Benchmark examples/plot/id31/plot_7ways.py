#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
8 vector-field figures (Truth + 7 methods).
Specific override: SINDy and SR3 now use HANDI's bounds (xB, yB).

For each figure:
- streamplot of the target field (normalized by speed)
- overlay TRUE trajectories (black solid)
- overlay PRED trajectories for that method (colored dashed), except the truth figure

Bounds policy:
- Truth figure uses bounds from TRUE trajs only: (xA, yA)
- HANDI, EDMD, GEDMD, PSE, WSINDY: bounds from TRUE + Method trajs
- SINDy, SR3: bounds FORCED to match HANDI's bounds
"""

import os
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator


# ============================================================
# 0) Global style
# ============================================================
def set_style_like_ref():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,

        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.grid": False,
        "svg.fonttype": "none",
    })


def beautify_axes(ax):
    # spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(axis="x", which="major", labelsize=20, length=6, width=1.5)
    ax.tick_params(axis="y", which="major", labelsize=20, length=6, width=1.5)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

    ymin, ymax = ax.get_ylim()
    cand = MaxNLocator(nbins=3).tick_values(ymin, ymax)
    cand = [v for v in cand if ymin <= v <= ymax and abs(v) > 1e-12]

    if len(cand) >= 2:
        yticks = [cand[0], cand[-1]]
    else:
        yticks = [ymin + 0.35 * (ymax - ymin), ymin + 0.75 * (ymax - ymin)]

    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))


# ============================================================
# 1) Truth RHS
# ============================================================
def f_true(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -0.4 * x1 * x2,
        0.4 * x1 * x2 - 0.314 * x2
    ], dtype=float)


# ============================================================
# 2) 7 recovered RHS
# ============================================================
def f_handi(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -0.336855 * x1 * x2 - 0.233229 * x1 - 0.0184153 * x1 * x1 + 0.00168879 * x2 - 0.000225622 * x2 * x2,
        0.34006 * x1 * x2 - 0.313728 * x2 + 0.196023 * x1 + 0.0212828 * x1 * x1 + 8.1152e-05 * x2 * x2
    ], dtype=float)


def f_edmd(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -118.5820 * x1 + 2.6781 * x2 - 16.0676 * x1 * x1 - 12.9253 * x1 * x2 - 5.2153 * x2 * x2,
        0.1028 * x1 - 7.4762 * x2 - 0.0146 * x1 * x1 - 0.1350 * x1 * x2 + 12.7551 * x2 * x2
    ], dtype=float)


def f_gedmd(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -0.00004 * x1 - 0.00018 * x2 - 0.00000 * x1 * x1 - 0.00000 * x1 * x2 - 0.00027 * x2 * x2,
        -0.00104 * x1 - 0.04316 * x2 - 0.00000 * x1 * x1 + 0.00001 * x1 * x2 - 0.06529 * x2 * x2
    ], dtype=float)


def f_pse(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -3.0 * x1,
        3.0 * x1 - 2.0
    ], dtype=float)


def f_sindy(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -2.25643 * x1 + 0.26458 * x1 * x1 - 0.05809 * x1 * x2,
        4.97752 * x1 - 0.30083 * x2 - 0.59010 * x1 * x1 - 0.31973 * x1 * x2 - 0.00887 * x2 * x2
    ], dtype=float)


def f_sr3(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -0.88751 * x1 + 0.17159 * x2 + 0.23671 * x1 * x1 - 0.68507 * x1 * x2 - 0.01068 * x2 * x2,
        0.98093 * x1 - 0.49494 * x2 - 0.25497 * x1 * x1 + 0.69619 * x1 * x2 + 0.01123 * x2 * x2
    ], dtype=float)


def f_wsindy(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -0.0 * x1 - 0.0 * x2 - 0.0 * x1 * x1 - 11.99998 * x1 * x2 - 0.0 * x2 * x2,
        0.0 * x1 - 9.41998 * x2 + 0.0 * x1 * x1 + 11.99944 * x1 * x2 + 0.0 * x2 * x2
    ], dtype=float)


# ============================================================
# 3) Trajectory integration
# ============================================================
def integrate_traj(f, x0, t_end=60.0, n_points=10000):
    t_eval = np.linspace(0.0, t_end, n_points)

    def f_wrap(t, x):
        x = np.maximum(x, 0.0)
        return f(t, x)

    sol = solve_ivp(
        f_wrap, (0.0, t_end), np.asarray(x0, float),
        t_eval=t_eval, rtol=1e-9, atol=1e-12
    )
    return np.maximum(sol.y.T, 0.0)


# ============================================================
# 4) Vector-field grid
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
# 5) Auto bounds
# ============================================================
def bounds_from_trajs(trajs, margin=0.08):
    XY = np.vstack(trajs)
    xmin, ymin = XY.min(axis=0)
    xmax, ymax = XY.max(axis=0)
    dx = max(xmax - xmin, 1e-12)
    dy = max(ymax - ymin, 1e-12)
    return (max(0.0, xmin - margin * dx), xmax + margin * dx), \
        (max(0.0, ymin - margin * dy), ymax + margin * dy)


# ============================================================
# 6) Vector-field error
# ============================================================
def vector_field_errors(f_ref, f_hat, x_range, y_range, n=80):
    X, Y, _, _, _ = vector_field_grid(f_ref, x_range, y_range, n)
    err2 = 0.0
    ref2 = 0.0
    for i in range(n):
        for j in range(n):
            ft = np.array(f_ref(0.0, [X[i, j], Y[i, j]]), dtype=float)
            fh = np.array(f_hat(0.0, [X[i, j], Y[i, j]]), dtype=float)
            d = fh - ft
            err2 += float(np.dot(d, d))
            ref2 += float(np.dot(ft, ft))
    return np.sqrt(err2 / (n * n)), np.sqrt(err2 / max(ref2, 1e-30))


# ============================================================
# 7) Plot one vector-field figure
# ============================================================
def plot_vector_field(f_field, x_range, y_range, trajs_true, trajs_pred=None,
                      pred_color="#4FA8D5", filename=None):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)

    X, Y, U, V, speed = vector_field_grid(f_field, x_range, y_range, n=60)
    eps = 1e-6
    ax.streamplot(
        X, Y, U / (speed + eps), V / (speed + eps),
        color="0.72",
        linewidth=2.0,
        density=1.2,
        arrowsize=1.4,
        arrowstyle='-|>',
        minlength=0.05,
        zorder=1
    )

    for T in trajs_true:
        ax.plot(T[:, 0], T[:, 1], color="black", lw=5.2)

    if trajs_pred is not None:
        for T in trajs_pred:
            ax.plot(T[:, 0], T[:, 1], color=pred_color, lw=5.2, ls="--")

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    beautify_axes(ax)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Saved: {filename}")

    plt.show()


# ============================================================
# 8) Main
# ============================================================
def main():
    set_style_like_ref()

    # output dir
    out_dir = "vf_8plots"
    os.makedirs(out_dir, exist_ok=True)

    inits = [[7.2, 0.98], [20.0, 12.4]]

    # methods: (rhs, color)
    methods = {
        "handi": (f_handi, "#50AAD8"),
        "edmd": (f_edmd, "#4AB4B2"),
        "gedmd": (f_gedmd, "#528FBF"),
        "pse": (f_pse, "#D8A0A7"),
        "sindy": (f_sindy, "#d06569"),
        "sr3": (f_sr3, "#C9A1CB"),
        "wsindy": (f_wsindy, "#F4B36B"),
    }

    # ---- integrate truth ----
    trajs_true = [integrate_traj(f_true, x0) for x0 in inits]

    # ---- truth bounds (A) ----
    xA, yA = bounds_from_trajs(trajs_true)

    # ---- integrate each method, compute bounds, and plot ----
    metrics_rows = []

    # 1) Truth plot (only truth trajs, truth bounds)
    plot_vector_field(
        f_true, xA, yA,
        trajs_true=trajs_true,
        trajs_pred=None,
        filename=os.path.join(out_dir, "1_true_vector_field.pdf")
    )

    # Placeholder for HANDI bounds (to apply to SINDy/SR3)
    handi_bounds = None

    idx = 2
    for name, (f_hat, color) in methods.items():
        trajs_pred = [integrate_traj(f_hat, x0) for x0 in inits]

        # Calculate natural bounds based on this method's trajectory
        xB, yB = bounds_from_trajs(trajs_true + trajs_pred)

        # --------------------------------------------------------
        # SPECIAL LOGIC:
        # 1. Capture HANDI bounds
        # 2. Force SINDy and SR3 to use HANDI bounds
        # --------------------------------------------------------
        if name == "handi":
            handi_bounds = (xB, yB)

        if name in ["sindy", "sr3"]:
            if handi_bounds is not None:
                xB, yB = handi_bounds
            else:
                print(f"Warning: Processing {name} before 'handi'. Cannot apply HANDI bounds.")

        # Compute metrics on the specific (xB, yB) chosen above
        E_L2, E_rel = vector_field_errors(f_true, f_hat, xB, yB, n=80)
        metrics_rows.append([name, xB, yB, E_L2, E_rel])

        plot_vector_field(
            f_hat, xB, yB,
            trajs_true=trajs_true,
            trajs_pred=trajs_pred,
            pred_color=color,
            filename=os.path.join(out_dir, f"{idx}_{name}_vector_field.pdf")
        )
        idx += 1

    # ---- save metrics ----
    csv_path = os.path.join(out_dir, "vectorfield_errors_7methods.csv")
    with open(csv_path, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["method", "x_range", "y_range", "E_L2", "E_rel"])
        for name, xB, yB, E_L2, E_rel in metrics_rows:
            w.writerow([name, f"{xB}", f"{yB}", f"{E_L2:.16e}", f"{E_rel:.16e}"])
    print(f"Saved: {csv_path}")

    txt_path = os.path.join(out_dir, "id31_vectorfield_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as g:
        g.write(f"[Truth bounds]\n")
        g.write(f"xA = {xA}\n")
        g.write(f"yA = {yA}\n\n")
        g.write(f"[Per-method bounds + metrics]\n")
        g.write(f"(Note: sindy and sr3 bounds forced to match handi)\n")
        for name, xB, yB, E_L2, E_rel in metrics_rows:
            g.write(f"{name:8s}  xB={xB}  yB={yB}  E_L2={E_L2:.6e}  E_rel={E_rel:.6e}\n")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()