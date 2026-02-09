#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split predicted trajectories into SEVEN figures (one per method).

Outputs:
  1) fig_truth_trajectories.svg
  2) fig_pred_HANDI.svg
  3) fig_pred_edmd.svg
  4) fig_pred_gedmd.svg
  5) fig_pred_PSE.svg
  6) fig_pred_sindy.svg
  7) fig_pred_SR3.svg
  8) fig_pred_wsindy.svg

Also:
  - Compute trajectory errors for each method averaged over many ICs
  - Print a table
  - Save: trajectory_errors_7methods.csv
"""

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter


# =========================================================
# 0) Style
# =========================================================
def set_style_like_ref():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,

        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.grid": False,
    })


set_style_like_ref()


# =========================================================
# 1) Axis style: L-shape, 3 x ticks, y ticks like screenshot (no y=0)
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
# 2) Truth + 7 methods
# =========================================================
a_true, b_true = 2.1, -0.5

def f_true(t, x):
    return [a_true * x[0] + b_true * x[0] ** 2]

METHODS = {
    "HANDI": {"a": 2.10172,   "b": -0.495336},
    "edmd":  {"a": 59.7432,   "b": 208.3850},
    "gedmd": {"a": -51.13703, "b": 441.35281},
    "PSE":   {"a": 0.0,       "b": 0.0, "const": 1.0},
    "sindy": {"a": 1.68343,   "b": -0.40079},
    "SR3":   {"a": 1.36485,   "b": -0.32314},
    "wsindy":{"a": 2.38334,   "b": -0.567},
}

def make_method_rhs(name: str):
    cfg = METHODS[name]
    if "const" in cfg:
        c = float(cfg["const"])
        def f(t, x):
            return [c]
        return f
    a = float(cfg["a"])
    b = float(cfg["b"])
    def f(t, x):
        return [a * x[0] + b * x[0] ** 2]
    return f


# =========================================================
# 3) Integration
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
# 4) Plot configs
# =========================================================
T = 10.0
n_t = 800
x0_many = np.linspace(0.05, 4.5, 80)
x0_hi = [0.13, 2.24]

COLOR = {
    "HANDI":  "#50AAD8",
    "edmd":   "#4AB4B2",
    "gedmd":  "#528FBF",
    "PSE":    "#D8A0A7",
    "sindy":  "#d06569",
    "SR3":    "#C9A1CB",
    "wsindy": "#F4B36B",
}
TRUE_BLACK = "black"


# =========================================================
# 5) Trajectory error evaluation
# =========================================================
def compute_errors_for_method(fun_hat, fun_true, x0_list, T, n_t):
    e_l2_list = []
    e_rel_list = []
    n_fail = 0

    for x0 in x0_list:
        try:
            _, x_true_seq = integrate(fun_true, x0, T, n_t)
            _, x_hat_seq  = integrate(fun_hat,  x0, T, n_t)
        except Exception:
            n_fail += 1
            continue

        diff = x_hat_seq - x_true_seq
        e_l2 = float(np.sqrt(np.mean(diff ** 2)))
        norm_diff = float(np.linalg.norm(diff))
        norm_true = float(np.linalg.norm(x_true_seq))
        e_rel = 0.0 if norm_true < 1e-15 else float(norm_diff / norm_true)

        e_l2_list.append(e_l2)
        e_rel_list.append(e_rel)

    if len(e_l2_list) == 0:
        return np.nan, np.nan, n_fail

    return float(np.mean(e_l2_list)), float(np.mean(e_rel_list)), n_fail


def save_errors_csv(rows, path="trajectory_errors_7methods.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "E_L2", "E_rel", "n_fail", "n_IC"])
        for r in rows:
            w.writerow(r)


# =========================================================
# 6) Main
# =========================================================
def main():
    print("[INFO] Method colors:")
    for k in METHODS.keys():
        print(f"  {k:6s} -> {COLOR.get(k)}")

    # ---------- errors ----------
    print(f"\n[INFO] Computing trajectory errors over {len(x0_many)} ICs ...")
    err_rows = []
    for name in METHODS.keys():
        f_hat = make_method_rhs(name)
        e_l2, e_rel, n_fail = compute_errors_for_method(f_hat, f_true, x0_many, T, n_t)
        err_rows.append([name, e_l2, e_rel, n_fail, len(x0_many)])

    print("=" * 72)
    print(f"{'method':8s} {'E_L2':>14s} {'E_rel':>14s} {'n_fail':>8s} {'n_IC':>6s}")
    for name, e_l2, e_rel, n_fail, n_ic in err_rows:
        e_l2_str = "nan" if (e_l2 != e_l2) else f"{e_l2:.6e}"
        e_rel_str = "nan" if (e_rel != e_rel) else f"{e_rel:.6e}"
        print(f"{name:8s} {e_l2_str:>14s} {e_rel_str:>14s} {n_fail:8d} {n_ic:6d}")
    print("=" * 72)

    save_errors_csv(err_rows)
    print("[OK] Saved: trajectory_errors_7methods.csv")

    # ---------- FIG 1: Truth ----------
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for x0 in x0_many:
        t, xt = integrate(f_true, x0, T, n_t)
        ax.plot(t, xt, color="0.80", lw=1.8)

    for x0 in x0_hi:
        t, xt = integrate(f_true, x0, T, n_t)
        ax.plot(t, xt, color=TRUE_BLACK, lw=5.2)

    ax.set_xlim(0.0, T)
    apply_axis_style(ax)
    finalize_and_save(fig, "fig_truth_trajectories.pdf")
    print("[OK] Saved: fig_truth_trajectories.pdf")

    # ---------- FIGS 2~8: Each method ----------
    for name in METHODS.keys():
        print(f"[PLOT] {name}")
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()

        f_hat = make_method_rhs(name)
        c = COLOR.get(name, "0.4")

        for x0 in x0_many:
            try:
                t, xh = integrate(f_hat, x0, T, n_t)
            except Exception:
                continue
            ax.plot(t, xh, color="0.80", lw=1.8)      # gray background

        for x0 in x0_hi:
            try:
                t, xh = integrate(f_hat, x0, T, n_t)
            except Exception:
                continue
            ax.plot(t, xh, color=c, lw=5.2)           # highlighted in method color

        ax.set_xlim(0.0, T)
        apply_axis_style(ax)

        out_path = f"fig_pred_{name}.pdf"
        finalize_and_save(fig, out_path)
        print(f"[OK] Saved: {out_path}")

    print("\n[DONE] 1 truth + 7 predicted trajectory figures saved.")


if __name__ == "__main__":
    main()
