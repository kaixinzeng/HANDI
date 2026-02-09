#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1D system (id9-like) recovery visualization with 7 methods.
Truth: dx/dt = -0.60*x + 0.32

Outputs (NCS-like style, no titles/legends/labels, L-shaped axes, 3 ticks):
Vector fields:
  - id9b_fig_true_vectorfield.svg
  - id9b_fig_vf_HANDI.svg
  - id9b_fig_vf_EDMD.svg
  - id9b_fig_vf_gedmd.svg
  - id9b_fig_vf_PSE.svg
  - id9b_fig_vf_sindy.svg
  - id9b_fig_vf_SR3.svg
  - id9b_fig_vf_wsindy.svg

Trajectories:
  - id9b_fig_true_trajectories.svg
  - id9b_fig_pred_HANDI.svg
  - id9b_fig_pred_EDMD.svg
  - id9b_fig_pred_gedmd.svg
  - id9b_fig_pred_PSE.svg
  - id9b_fig_pred_sindy.svg
  - id9b_fig_pred_SR3.svg
  - id9b_fig_pred_wsindy.svg

Metrics:
  - id9b_vectorfield_errors_7methods.csv
  - id9b_trajectory_errors_7methods.csv
  - id9b_system_recovery_analysis.txt
"""

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter


# =========================================================
# Style
# =========================================================
def set_style_like_ref():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,

        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.grid": False,
    })


set_style_like_ref()


# =========================================================
# Axis style: L-shape, 3 ticks, no labels
# =========================================================
def apply_axis_style(ax, tick_labelsize=20, tick_length=6, y_ndecimals=1):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)

    ax.xaxis.set_major_locator(MaxNLocator(3))

    ymin, ymax = ax.get_ylim()
    cand = MaxNLocator(nbins=3).tick_values(ymin, ymax)
    cand = [v for v in cand if (ymin - 1e-12) <= v <= (ymax + 1e-12) and abs(v) > 1e-12]

    if len(cand) >= 2:
        yticks = [cand[0], cand[-1]]
    else:
        yticks = [ymin + 0.35 * (ymax - ymin), ymin + 0.75 * (ymax - ymin)]
        yticks = [v for v in yticks if abs(v) > 1e-12]

    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.{y_ndecimals}f}"))

    ax.tick_params(direction="out", length=tick_length, width=1.5, labelsize=tick_labelsize)
    ax.set_xlabel("")
    ax.set_ylabel("")


def finalize_and_save(fig, path, left=0.11, right=0.99, bottom=0.12, top=0.99,
                      dpi=300, pad_inches=0.02):
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


# =========================================================
# 1) Shared truth dynamics
# =========================================================
def f_true(t, x):
    return [-0.60 * x[0] + 0.32]


# =========================================================
# 2) Safe expression -> RHS (for PSE)
# =========================================================
def make_rhs_from_expr(expr: str, var_name: str = "x0"):
    expr = expr.replace("·", "*")

    allowed_names = {
        "np": np,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
    }

    def f(t, x):
        local = dict(allowed_names)
        local[var_name] = float(x[0])
        val = eval(expr, {"__builtins__": {}}, local)
        return [float(val)]

    return f


# =========================================================
# 3) 7 recovered methods RHS
# =========================================================
def make_methods():
    methods = {}

    def f_handi(t, x):
        return [-0.600124 * x[0] + 0.320036]
    methods["HANDI"] = f_handi

    def f_edmd(t, x):
        return [-0.0120 + 6893.3243 * x[0]]
    methods["EDMD"] = f_edmd

    def f_gedmd(t, x):
        return [-20935.79346 + 353379.20733 * x[0]]
    methods["gedmd"] = f_gedmd

    pse_expr = (
        "(0.406904 - 6.956742*x0**3.0)**2.0"
        "*(3.935375*x0 + (13.194191*x0**3.0 - 2.858364)**2.0 - 1.257415)"
        " + (6.027893*x0 - 8.281227*x0**2.0 - 0.030986)**4.0"
        " - 1.133355"
    )
    methods["PSE"] = make_rhs_from_expr(pse_expr, var_name="x0")

    def f_sindy(t, x):
        return [0.25917 - 0.47121 * x[0]]
    methods["sindy"] = f_sindy

    def f_sr3(t, x):
        return [0.26309 - 0.48762 * x[0]]
    methods["SR3"] = f_sr3

    def f_wsindy(t, x):
        return [0.32232 - 0.60435 * x[0]]
    methods["wsindy"] = f_wsindy

    return methods


METHODS = make_methods()


# =========================================================
# Integration
# =========================================================
def integrate(fun, x0, T=10.0, n_t=800):
    t_eval = np.linspace(0.0, T, int(n_t))
    sol = solve_ivp(
        fun, (0.0, T), [float(x0)],
        t_eval=t_eval,
        rtol=1e-9, atol=1e-12
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[0]


# =========================================================
# Vector-field errors
#   E_L2  = sqrt(mean |fh-f|^2)
#   E_rel = sqrt(int |fh-f|^2 dx) / sqrt(int |f|^2 dx)
# =========================================================
def compute_vectorfield_errors_1d(f_true_scalar, f_hat_scalar, Omega, N=10001):
    x = np.linspace(Omega[0], Omega[1], int(N))
    f  = f_true_scalar(x)
    fh = f_hat_scalar(x)

    diff2 = (fh - f) ** 2
    E_L2 = float(np.sqrt(np.mean(diff2)))

    num = float(np.sqrt(np.trapz(diff2, x)))
    den = float(np.sqrt(np.trapz(f**2, x)))
    E_rel = float(num / (den + 1e-15))
    return E_L2, E_rel


# =========================================================
# Trajectory errors (averaged over ICs)
#   E_L2  = sqrt(mean_t (xh-xt)^2)
#   E_rel = ||xh-xt||2 / ||xt||2
# =========================================================
def compute_trajectory_errors(fun_true, fun_hat, x0_list, T, n_t):
    e_l2_list, e_rel_list = [], []
    n_fail = 0

    for x0 in x0_list:
        try:
            _, xt = integrate(fun_true, x0, T, n_t)
            _, xh = integrate(fun_hat,  x0, T, n_t)
        except Exception:
            n_fail += 1
            continue

        diff = xh - xt
        e_l2 = float(np.sqrt(np.mean(diff ** 2)))

        norm_diff = float(np.linalg.norm(diff))
        norm_true = float(np.linalg.norm(xt))
        e_rel = 0.0 if norm_true < 1e-15 else float(norm_diff / norm_true)

        e_l2_list.append(e_l2)
        e_rel_list.append(e_rel)

    if len(e_l2_list) == 0:
        return np.nan, np.nan, n_fail

    return float(np.mean(e_l2_list)), float(np.mean(e_rel_list)), n_fail


# =========================================================
# Config
# =========================================================
T = 10.0
n_t = 800

x0_many = np.linspace(0.05, 1.00, 80)
x0_hi = [0.14, 0.55]

xg = np.linspace(-0.2, 1.2, 600)
Omega = (0.0, 1.0)

TRUE_BLACK = "black"
# per-method highlight colors
COLOR = {
    "HANDI":  "#50AAD8",
    "edmd":   "#4AB4B2",
    "gedmd":  "#528FBF",
    "PSE":    "#D8A0A7",
    "sindy":  "#d06569",
    "SR3":    "#C9A1CB",
    "wsindy": "#F4B36B",
}


# =========================================================
# Build scalar versions for vector-field plots/errors
# =========================================================
def f_true_scalar(x):
    x = np.asarray(x, dtype=float)
    return -0.60 * x + 0.32


def make_hat_scalar(fun_hat):
    def fh(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        out = np.empty_like(x, dtype=float)
        for i, xi in enumerate(x):
            out[i] = float(fun_hat(0.0, [float(xi)])[0])
        return out
    return fh


# =========================================================
# Shared y-limits for trajectory plots (from TRUTH only)
# =========================================================
def compute_traj_ylim_from_truth():
    ymin, ymax = np.inf, -np.inf
    for x0 in x0_many[::4]:
        _, xt = integrate(f_true, x0, T, 400)
        ymin = min(ymin, float(np.min(xt)))
        ymax = max(ymax, float(np.max(xt)))
    # small padding
    pad = 0.02 * (ymax - ymin + 1e-12)
    return ymin - pad, ymax + pad


def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# =========================================================
# Main plotting + metrics
# =========================================================
def main():
    print("[INFO] Methods:", list(METHODS.keys()))
    print("[INFO] Colors (highlighted, no legend in figures):")
    for k in METHODS.keys():
        print(f"  {k:6s} -> {COLOR.get(k, 'gray')}")

    # ---------- shared y-lims for trajectories ----------
    ymin_traj, ymax_traj = compute_traj_ylim_from_truth()
    print(f"[INFO] Trajectory y-lims (from TRUTH only): [{ymin_traj:.6f}, {ymax_traj:.6f}]")

    # =========================================================
    # 1) TRUE vector field
    # =========================================================
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    vf_true = f_true_scalar(xg)
    ax.plot(xg, vf_true, color=TRUE_BLACK, lw=2.0)
    ax.axhline(0.0, color="0.6", lw=1.2)

    ax.set_xlim(float(xg.min()), float(xg.max()))
    apply_axis_style(ax)
    finalize_and_save(fig, "id9b_fig_true_vectorfield.svg")
    print("[OK] Saved: id9b_fig_true_vectorfield.svg")

    # =========================================================
    # 2) EACH method vector field (7 figs)
    # =========================================================
    for name, fhat in METHODS.items():
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()

        fh_scalar = make_hat_scalar(fhat)
        vf_hat = fh_scalar(xg)

        ax.plot(xg, vf_hat, color=COLOR.get(name, "#4FA8D5"), lw=2.0)
        ax.axhline(0.0, color="0.6", lw=1.2)

        ax.set_xlim(float(xg.min()), float(xg.max()))
        apply_axis_style(ax)
        out = f"id9b_fig_vf_{name}.svg"
        finalize_and_save(fig, out)
        print(f"[OK] Saved: {out}")

    # =========================================================
    # 3) TRUE trajectories (1 fig)
    # =========================================================
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for x0 in x0_many:
        t, xt = integrate(f_true, x0, T, n_t)
        ax.plot(t, xt, color="0.80", lw=1.8)

    for x0 in x0_hi:
        t, xt = integrate(f_true, x0, T, n_t)
        ax.plot(t, xt, color=TRUE_BLACK, lw=5.2)

    ax.set_xlim(0.0, T)
    ax.set_ylim(ymin_traj, ymax_traj)
    apply_axis_style(ax)
    finalize_and_save(fig, "id9b_fig_true_trajectories.pdf")
    print("[OK] Saved: id9b_fig_true_trajectories.pdf")

    # =========================================================
    # 4) Predicted trajectories: split into 7 figs (one per method)
    # =========================================================
    for name, fhat in METHODS.items():
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()

        # background (gray) = this method family
        for x0 in x0_many:
            try:
                t, xh = integrate(fhat, x0, T, n_t)
            except Exception:
                continue
            ax.plot(t, xh, color="0.80", lw=1.8)

        # highlighted (method color)
        for x0 in x0_hi:
            try:
                t, xh = integrate(fhat, x0, T, n_t)
            except Exception:
                continue
            ax.plot(t, xh, color=COLOR.get(name, "#4FA8D5"), lw=5.2)

        ax.set_xlim(0.0, T)
        ax.set_ylim(ymin_traj, ymax_traj)
        apply_axis_style(ax)

        out = f"id9b_fig_pred_{name}.pdf"
        finalize_and_save(fig, out)
        print(f"[OK] Saved: {out}")

    # =========================================================
    # 5) Metrics: vector-field + trajectories (per method)
    # =========================================================
    vf_rows = []
    traj_rows = []

    for name, fhat in METHODS.items():
        fh_scalar = make_hat_scalar(fhat)
        E_L2_vf, E_rel_vf = compute_vectorfield_errors_1d(f_true_scalar, fh_scalar, Omega, N=10001)
        vf_rows.append([name, E_L2_vf, E_rel_vf, Omega[0], Omega[1]])

        E_L2_tr, E_rel_tr, n_fail = compute_trajectory_errors(f_true, fhat, x0_many, T, n_t)
        traj_rows.append([name, E_L2_tr, E_rel_tr, n_fail, len(x0_many)])

    save_csv(
        "id9b_vectorfield_errors_7methods.csv",
        ["method", "E_L2", "E_rel", "Omega0", "Omega1"],
        vf_rows
    )
    save_csv(
        "id9b_trajectory_errors_7methods.csv",
        ["method", "E_L2", "E_rel", "n_fail", "n_IC"],
        traj_rows
    )
    print("[OK] Saved: id9b_vectorfield_errors_7methods.csv")
    print("[OK] Saved: id9b_trajectory_errors_7methods.csv")

    # =========================================================
    # 6) Write a single analysis txt
    # =========================================================
    out_txt = "id9b_system_recovery_analysis.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("id9b system recovery analysis\n\n")
        f.write("Truth ODE: dx/dt = -0.60*x + 0.32\n\n")
        f.write(f"Config:\n  T={T}, n_t={n_t}\n  x0_many in [{x0_many.min()}, {x0_many.max()}], n={len(x0_many)}\n")
        f.write(f"  x0_hi = {x0_hi}\n  Omega = [{Omega[0]}, {Omega[1]}]\n")
        f.write(f"  traj_ylim(from truth) = [{ymin_traj}, {ymax_traj}]\n\n")

        f.write("Vector-field errors per method:\n")
        for r in vf_rows:
            f.write(f"  {r[0]:6s}  E_L2={r[1]:.12e}  E_rel={r[2]:.12e}\n")
        f.write("\nTrajectory errors per method (avg over x0_many):\n")
        for r in traj_rows:
            e_l2 = r[1]
            e_rel = r[2]
            f.write(f"  {r[0]:6s}  E_L2={e_l2:.12e}  E_rel={e_rel:.12e}  n_fail={r[3]}\n")

    print(f"[OK] Saved: {out_txt}")

    # Console summary
    print("=" * 72)
    print("Trajectory Errors (avg over ICs):")
    for r in traj_rows:
        name, e_l2, e_rel, n_fail, n_ic = r
        print(f"  {name:6s}  E_L2={e_l2:.6e}  E_rel={e_rel:.6e}  n_fail={n_fail}/{n_ic}")
    print("=" * 72)


if __name__ == "__main__":
    main()
