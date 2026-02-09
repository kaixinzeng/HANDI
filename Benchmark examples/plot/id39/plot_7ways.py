#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
id39-like (2D) — ONLY vector-field panels (Truth + 7 methods) following your reference program:

What it does:
- Uses your "nice range" logic: auto_phase_range_nice(inits, ...) BUT now generalized:
    * For truth panel: range from (truth only)
    * For each method panel: range from (truth + that method), like your reference (truth+f_pred)
- Plots ONLY phase vector-field panels (streamplot + trajectories):
    1) Truth vector field + truth trajectories
    2..8) Method vector field + truth trajectories (black) + method trajectories (colored dashed)
- Saves 8 SVGs + metrics:
    - vectorfield_errors_7methods.csv  (E_L2, E_rel on each method's own nice range)
    - id39_vectorfield_metrics.txt     (records ranges, nice_step, settings)

No streamline-family plots. No extra figures.

NOTE:
- We do NOT clamp states to >=0 here (your reference id39 uses unconstrained integration).
- PSE RHS is built via restricted eval from expression strings (as you provided).
"""

import os
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# =============================
# 0) Style (same as your reference)
# =============================
def set_ncs_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 14,
        "legend.fontsize": 12,

        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,

        # keep SVG text editable (no paths)
        "svg.fonttype": "none",
    })


def beautify_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(axis="x", which="major", labelsize=20, length=6, width=1.5)
    ax.tick_params(axis="y", which="major", labelsize=20, length=6, width=1.5)


# =========================
# 1) Truth RHS  (KEEP YOUR SYSTEM HERE)
# =========================
def f_true(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        2.4 * x2 - 1.0 * x1 + 1.0 * (x1 * x1 * x2),
        0.07 - 2.4 * x1 - 1.0 * (x1 * x1 * x2),
    ], dtype=float)


# =========================
# 2) Helper: safe expr -> RHS (for PSE)
# =========================
def make_rhs_from_expr_2d(expr_dx0: str, expr_dx1: str, var0="x0", var1="x1"):
    """
    Build 2D RHS f(t, x)->(2,) from python expression strings using restricted eval.
    - var0/var1 correspond to x[0]/x[1]
    """
    expr_dx0 = expr_dx0.replace("·", "*")
    expr_dx1 = expr_dx1.replace("·", "*")

    allowed = {
        "np": np,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
    }

    def f(t, x):
        local = dict(allowed)
        local[var0] = float(x[0])
        local[var1] = float(x[1])
        v0 = eval(expr_dx0, {"__builtins__": {}}, local)
        v1 = eval(expr_dx1, {"__builtins__": {}}, local)
        return np.array([float(v0), float(v1)], dtype=float)

    return f


# =========================
# 3) 7 recovered RHS
# =========================
def f_handi(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        2.3999 * x2 - 1.00101 * x1 + 0.98595 * x1 * x1 * x2
        + 0.0257401 * x1 * x2 * x2 + 0.00333737 * x2 * x2 * x2
        + 0.00240551 * x1 * x2 + 0.00237569 * x1 * x1 + 0.00120466 * x1 * x1 * x1
        - 0.000111829 * x2 * x2 - 4.57825e-05 * 1.0,

        -2.39948 * x1 - 0.983099 * x1 * x1 * x2 + 0.0700334 * 1.0
        - 0.0145857 * x1 * x2 * x2 - 0.00597264 * x2 * x2 * x2
        - 0.00293468 * x1 * x1 * x1 - 0.00278489 * x1 * x2 - 0.00177572 * x1 * x1
        + 0.000297233 * x2 * x2 - 0.000217666 * x2
    ], dtype=float)


def f_edmd(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -0.9993 * x1 - 2.4024 * x2 + 0.0001 * x1 * x1 + 0.0699 * x1 * x2 - 0.0 * x2 * x2,
        2.3977 * x1 + 0.0018 * x2 - 0.0002 * x1 * x1 + 0.1400 * x2 * x2
    ], dtype=float)


def f_gedmd(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -0.00262 * 1.0 - 0.87444 * x1 + 2.40188 * x2
        + 4.24600 * x1 * x1 - 22.56273 * x1 * x2 - 3.25212 * x2 * x2
        - 133.15863 * x1 * x1 * x1 + 1193.14243 * x1 * x1 * x2 + 389.39694 * x1 * x2 * x2
        - 262.79033 * x2 * x2 * x2,

        0.06412 * 1.0 - 2.19347 * x1 - 0.23552 * x2
        - 4.41880 * x1 * x1 + 41.97822 * x1 * x2 + 7.95677 * x2 * x2
        + 194.07448 * x1 * x1 * x1 - 1993.43352 * x1 * x1 * x2 - 914.40153 * x1 * x2 * x2
        + 478.48663 * x2 * x2 * x2
    ], dtype=float)


# PSE (x0,x1 form)  -> here x0:=x[0], x1:=x[1]
pse_dx0 = (
    "-0.785442*x0 + 1.99054*x0**2.0*x1 - 181.249988*x0**5.0*x1*(0.991343*x0 + 0.520801)"
    " - 3.337857*x1*(-x0)**3.0 + 2.256492*x1 + 0.447445*(-x1)**3.0 - 0.003751"
)
pse_dx1 = (
    "0.112678*x0*x1 + 0.944319*x0*x1**2.0 + 1.06588*x0*x1**3.0*(-24.273381*x0 - 4.11022)"
    " - 2.249288*x0 - 2.078647*x0**2.0*x1**2.0 + 0.111316*x1 + 0.06498"
)
f_pse = make_rhs_from_expr_2d(pse_dx0, pse_dx1, var0="x0", var1="x1")


def f_sindy(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -0.87368 * x1 + 2.27152 * x2 - 0.17653 * x1 * x1 + 0.12903 * x1 * x2 - 0.23984 * x2 * x2
        + 0.84150 * x1 * x1 * x1 - 1.01104 * x1 * x1 * x2 - 1.22742 * x2 * x2 * x2,

        0.06333 * 1.0 - 2.20840 * x1 + 0.15026 * x2 + 0.05288 * x1 * x1 - 0.20995 * x1 * x2 - 0.19477 * x2 * x2
        - 0.22795 * x1 * x1 * x1 - 2.14889 * x1 * x1 * x2 - 2.78174 * x1 * x2 * x2 - 1.94819 * x2 * x2 * x2
    ], dtype=float)


def f_sr3(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        -0.0 * 1.0 - 0.80480 * x1 + 2.21658 * x2
        - 1.27832 * x1 * x2 * x2 + 0.59068 * x2 * x2 * x2,

        0.06421 * 1.0 - 2.23697 * x1 + 0.14209 * x2
        - 0.68110 * x1 * x2
        - 2.45551 * x1 * x1 * x2
        - 0.28015 * x2 * x2 * x2
    ], dtype=float)


def f_wsindy(t, x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        0.0 * 1.0 - 0.99378 * x1 + 2.38726 * x2,
        0.07003 * 1.0 - 2.40168 * x1 + 0.0 * x2
    ], dtype=float)


# =============================
# 2) Trajectory integration
# =============================
def integrate_traj(f, x0, t_end=60.0, n_points=3000, rtol=1e-9, atol=1e-12):
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        fun=lambda t, x: f(t, x),
        t_span=(0.0, t_end),
        y0=np.array(x0, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    return t_eval, sol.y.T


# =============================
# 2.5) Nice range
# =============================
def _nice_step(span, target_divs=2):
    raw = span / max(target_divs, 1)
    if raw <= 0:
        return 1.0
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


def auto_phase_range_nice_two_fields(inits, fA, fB=None, t_end=60.0, n_points=6000, pad=0.06, extra=1.02):
    """
    Generalized from your auto_phase_range_nice:
    - if fB is provided: range from trajectories of fA (truth) and fB (method)
    - if fB is None: range from trajectories of fA only (truth only)
    """
    xs, ys = [], []
    for x0 in inits:
        _, XA = integrate_traj(fA, x0, t_end=t_end, n_points=n_points)
        if fB is not None:
            _, XB = integrate_traj(fB, x0, t_end=t_end, n_points=n_points)
            X = np.vstack([XA, XB])
        else:
            X = XA
        xs.append(X[:, 0])
        ys.append(X[:, 1])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    xlo, xhi = float(xs.min()), float(xs.max())
    ylo, yhi = float(ys.min()), float(ys.max())

    dx = (xhi - xlo) if xhi > xlo else 1.0
    dy = (yhi - ylo) if yhi > ylo else 1.0
    xlo -= pad * dx
    xhi += pad * dx
    ylo -= pad * dy
    yhi += pad * dy

    cx = 0.5 * (xlo + xhi)
    cy = 0.5 * (ylo + yhi)
    hx = 0.5 * (xhi - xlo)
    hy = 0.5 * (yhi - ylo)
    h = max(hx, hy) * extra

    span = 2.0 * h
    step = _nice_step(span, target_divs=2)

    xlo_n = step * np.floor((cx - h) / step)
    xhi_n = step * np.ceil((cx + h) / step)
    ylo_n = step * np.floor((cy - h) / step)
    yhi_n = step * np.ceil((cy + h) / step)

    return (float(xlo_n), float(xhi_n)), (float(ylo_n), float(yhi_n)), float(step)


def set_three_ticks(ax, x_range, y_range):
    xt = np.array([x_range[0], 0.5 * (x_range[0] + x_range[1]), x_range[1]], dtype=float)
    yt = np.array([y_range[0], 0.5 * (y_range[0] + y_range[1]), y_range[1]], dtype=float)
    ax.set_xticks(xt)
    ax.set_yticks(yt)


# =============================
# 3) Background streamlines grid
# =============================
def vector_field_grid(f, x0_range, x1_range, grid_n=50):
    x0_min, x0_max = x0_range
    x1_min, x1_max = x1_range
    X0, X1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_n),
        np.linspace(x1_min, x1_max, grid_n),
        indexing="xy"
    )
    U = np.zeros_like(X0, dtype=float)
    V = np.zeros_like(X1, dtype=float)
    for i in range(grid_n):
        for j in range(grid_n):
            u, v = f(0.0, [X0[i, j], X1[i, j]])
            U[i, j] = u
            V[i, j] = v
    speed = np.sqrt(U ** 2 + V ** 2)
    return X0, X1, U, V, speed


# =============================
# 4) Metrics (same as your ref)
# =============================
def vector_field_L2_errors(f_ref, f_hat, x0_range, x1_range, grid_n=80):
    x0 = np.linspace(*x0_range, grid_n)
    x1 = np.linspace(*x1_range, grid_n)
    X0, X1 = np.meshgrid(x0, x1, indexing="xy")

    err2_sum = 0.0
    ref2_sum = 0.0

    for i in range(grid_n):
        for j in range(grid_n):
            ft = np.asarray(f_ref(0.0, [X0[i, j], X1[i, j]]), dtype=float)
            fh = np.asarray(f_hat(0.0, [X0[i, j], X1[i, j]]), dtype=float)
            diff = fh - ft
            err2_sum += float(np.dot(diff, diff))
            ref2_sum += float(np.dot(ft, ft))

    N = grid_n * grid_n
    E_L2 = np.sqrt(err2_sum / max(N, 1))
    E_rel = np.sqrt(err2_sum / max(ref2_sum, 1e-30))
    return E_L2, E_rel


# =============================
# 5) Phase panel (ONLY this plot)
# =============================
def plot_phase_panel(
    f_field,
    f_method_for_traj,
    inits,
    x0_range, x1_range,
    t_end=60.0, n_points=3000,
    stream_density=1.35,
    stream_color="0.80",
    pred_color="#4FA8D5",
    filename=None
):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)

    X0, X1, U, V, speed = vector_field_grid(f_field, x0_range, x1_range, grid_n=50)

    eps = 1e-6
    U_plot = U / (speed + eps)
    V_plot = V / (speed + eps)

    ax.streamplot(
        X0, X1, U_plot, V_plot,
        color="0.72",
        linewidth=2.0,
        density=1.2,
        arrowsize=1.4,
        arrowstyle='-|>',
        minlength=0.05,
        zorder=1
    )

    for x0 in inits:
        _, X_true = integrate_traj(f_true, x0, t_end=t_end, n_points=n_points)
        ax.plot(X_true[:, 0], X_true[:, 1],
                linestyle="-", linewidth=5.2, color="black", zorder=3)

        if f_method_for_traj is not None:
            _, X_pred = integrate_traj(f_method_for_traj, x0, t_end=t_end, n_points=n_points)
            ax.plot(X_pred[:, 0], X_pred[:, 1],
                    linestyle="--", linewidth=5.2, color=pred_color, zorder=3)

    ax.set_xlim(*x0_range)
    ax.set_ylim(*x1_range)

    set_three_ticks(ax, x0_range, x1_range)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    beautify_axes(ax)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Saved: {filename}")

    plt.show()


# =============================
# 6) Main: ONLY 8 vector-field panels
# =============================
def main():
    set_ncs_style()

    out_dir = "vf_8plots_id39"
    os.makedirs(out_dir, exist_ok=True)

    inits = [
        [0.35, 0.31],
        [-0.42, 0.28],
    ]

    t_end = 60.0
    n_points_traj = 3000

    # ---- UPDATED COLORS (as you requested) ----
    COLOR = {
        "HANDI":  "#50AAD8",
        "edmd":   "#4AB4B2",
        "gedmd":  "#528FBF",
        "PSE":    "#D8A0A7",
        "sindy":  "#d06569",
        "SR3":    "#C9A1CB",
        "wsindy": "#F4B36B",
    }
    # normalize to internal (lowercase) method keys
    COLOR_MAP = {
        "handi":  COLOR["HANDI"],
        "edmd":   COLOR["edmd"],
        "gedmd":  COLOR["gedmd"],
        "pse":    COLOR["PSE"],
        "sindy":  COLOR["sindy"],
        "sr3":    COLOR["SR3"],
        "wsindy": COLOR["wsindy"],
    }

    methods = {
        "handi":  (f_handi,  COLOR_MAP["handi"]),
        "edmd":   (f_edmd,   COLOR_MAP["edmd"]),
        "gedmd":  (f_gedmd,  COLOR_MAP["gedmd"]),
        "pse":    (f_pse,    COLOR_MAP["pse"]),
        "sindy":  (f_sindy,  COLOR_MAP["sindy"]),
        "sr3":    (f_sr3,    COLOR_MAP["sr3"]),
        "wsindy": (f_wsindy, COLOR_MAP["wsindy"]),
    }

    # ---- 1) Truth panel range (truth only) ----
    x_range_true, y_range_true, step_true = auto_phase_range_nice_two_fields(
        inits, fA=f_true, fB=None, t_end=t_end, n_points=6000, pad=0.06, extra=1.02
    )

    # ---- Truth panel (only truth trajs) ----
    plot_phase_panel(
        f_field=f_true,
        f_method_for_traj=None,
        inits=inits,
        x0_range=x_range_true,
        x1_range=y_range_true,
        t_end=t_end,
        n_points=n_points_traj,
        stream_density=1.35,
        stream_color="0.80",
        pred_color=COLOR_MAP["handi"],
        filename=os.path.join(out_dir, "1_true_vector_field.pdf")
    )

    # ---- metrics outputs ----
    rows = []
    metrics_txt = os.path.join(out_dir, "id39_vectorfield_metrics.txt")
    with open(metrics_txt, "w", encoding="utf-8") as ftxt:
        ftxt.write("Vector-field metrics (per-method nice range, like reference)\n\n")
        ftxt.write("[Truth-only range]\n")
        ftxt.write(f"x_range_true = [{x_range_true[0]:.8g}, {x_range_true[1]:.8g}]\n")
        ftxt.write(f"y_range_true = [{y_range_true[0]:.8g}, {y_range_true[1]:.8g}]\n")
        ftxt.write(f"nice_step_true = {step_true:.8g}\n")
        ftxt.write(f"t_end = {t_end}\n")
        ftxt.write(f"n_points(traj) = {n_points_traj}\n")
        ftxt.write(f"inits = {inits}\n\n")

        # ---- 2..8) Each method: its own range from (truth + method) ----
        idx = 2
        for name, (f_hat, color) in methods.items():
            x_range, y_range, step = auto_phase_range_nice_two_fields(
                inits, fA=f_true, fB=f_hat, t_end=t_end, n_points=6000, pad=0.06, extra=1.02
            )

            E_L2, E_rel = vector_field_L2_errors(f_true, f_hat, x_range, y_range, grid_n=80)
            rows.append([name, x_range, y_range, step, E_L2, E_rel])

            ftxt.write(f"[{name}]\n")
            ftxt.write(f"x_range = [{x_range[0]:.8g}, {x_range[1]:.8g}]\n")
            ftxt.write(f"y_range = [{y_range[0]:.8g}, {y_range[1]:.8g}]\n")
            ftxt.write(f"nice_step = {step:.8g}\n")
            ftxt.write(f"E_L2  = {E_L2:.12e}\n")
            ftxt.write(f"E_rel = {E_rel:.12e}\n\n")

            plot_phase_panel(
                f_field=f_hat,
                f_method_for_traj=f_hat,
                inits=inits,
                x0_range=x_range,
                x1_range=y_range,
                t_end=t_end,
                n_points=n_points_traj,
                stream_density=1.35,
                stream_color="0.80",
                pred_color=color,
                filename=os.path.join(out_dir, f"{idx}_{name}_vector_field.pdf")
            )
            idx += 1

    print(f"Saved: {metrics_txt}")

    csv_path = os.path.join(out_dir, "vectorfield_errors_7methods.csv")
    with open(csv_path, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["method", "x_range", "y_range", "nice_step", "E_L2", "E_rel"])
        for name, xr, yr, step, E_L2, E_rel in rows:
            w.writerow([name, f"{xr}", f"{yr}", f"{step:.16e}", f"{E_L2:.16e}", f"{E_rel:.16e}"])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
