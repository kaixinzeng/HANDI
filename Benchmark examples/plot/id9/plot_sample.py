#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
id9 (split into 4 separate figures, NCS style):

1) True vector field  (black)
2) Recovered vector field (blue)
3) True trajectories from many ICs (highlighted = black) + SAMPLES (dt=2.0)
4) Recovered trajectories from many ICs (highlighted = blue)

Plus:
- Compute vector-field errors (E_L2, E_rel)
- Write them (and parameter errors + trajectory errors) into a txt file

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter
import os


# =========================================================
# Style (exactly following your reference)
# =========================================================
def set_style_like_ref():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 46,
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
# Helper: consistent margins (stable left whitespace)
# =========================================================
def finalize_and_save(fig, path,
                      left=0.11, right=0.99, bottom=0.12, top=0.99,
                      dpi=300, pad_inches=0.02):
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


# =========================================================
# Axis style: L-shape + x 3 ticks + y custom ticks (remove 0)
# =========================================================
def apply_axis_style(ax, tick_labelsize=20, tick_length=6, y_ndecimals=1, y_width=5):
    # L-shaped axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(4.0)
    ax.spines["bottom"].set_linewidth(4.0)

    # x: 3 ticks
    ax.xaxis.set_major_locator(MaxNLocator(3))

    # y: mimic screenshot: remove 0, keep 2 ticks (bottom/top)
    ymin, ymax = ax.get_ylim()
    cand = MaxNLocator(nbins=3).tick_values(ymin, ymax)
    cand = [v for v in cand if (ymin - 1e-12) <= v <= (ymax + 1e-12) and abs(v) > 1e-12]

    if len(cand) >= 2:
        yticks = [cand[0], cand[-1]]
    else:
        yticks = [ymin + 0.35 * (ymax - ymin), ymin + 0.75 * (ymax - ymin)]
        yticks = [v for v in yticks if abs(v) > 1e-12]

    ax.yaxis.set_major_locator(FixedLocator(yticks))

    # Scheme B: fixed-width (reserve sign space via leading spaces, no '+')
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: f"{v:{y_width}.{y_ndecimals}f}")
    )

    # tick appearance
    ax.tick_params(axis="x", direction="out", length=tick_length, width=1.5, labelsize=tick_labelsize)
    ax.tick_params(axis="y", direction="out", length=tick_length, width=1.5, labelsize=tick_labelsize)

    # y tick font = Arial
    for lab in ax.get_yticklabels():
        lab.set_fontfamily("Arial")

    # remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")


# =========================================================
# Colors
# =========================================================
REC_BLUE = "#4FA8D5"
TRUE_BLACK = "black"

# =========================================================
# System definitions (id9)
# True: dx/dt = -0.60*x + 0.32
# Rec : dx/dt = -0.600124*x + 0.320036
# =========================================================
a_true, b_true = -0.60, 0.32
a_rec, b_rec = -0.600124, 0.320036


def f_true(t, x):
    return [a_true * x[0] + b_true]


def f_rec(t, x):
    return [a_rec * x[0] + b_rec]


def integrate(fun, x0, T=10.0, n_t=800):
    t_eval = np.linspace(0.0, T, int(n_t))
    sol = solve_ivp(fun, (0.0, T), [float(x0)],
                    t_eval=t_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[0]


# =========================================================
# Vector-field errors (1D)
# =========================================================
def compute_vectorfield_errors_1d(f_true_scalar, f_rec_scalar, Omega, N=10001):
    x = np.linspace(Omega[0], Omega[1], int(N))
    f = f_true_scalar(x)
    fh = f_rec_scalar(x)

    diff2 = (fh - f) ** 2
    E_L2 = float(np.sqrt(np.mean(diff2)))

    num = float(np.sqrt(np.trapz(diff2, x)))
    den = float(np.sqrt(np.trapz(f ** 2, x)))
    E_rel = float(num / (den + 1e-15))
    return E_L2, E_rel


# =========================================================
# Config
# =========================================================
T = 10.0
n_t = 800

x0_many = np.linspace(0.05, 1.00, 80)  # background family
x0_hi = [0.14, 0.55]  # highlighted

xg = np.linspace(-0.2, 1.2, 600)  # vector field plotting domain
Omega = (0.0, 1.0)  # error domain

# =========================================================
# Load Sample Data (Logic Update for dt=2.0)
# =========================================================
dt_sample = 2.0
npy_filename = "id9_downsample_40.npy"
sample_data = None
t_all_samples = None

if os.path.exists(npy_filename):
    print(f"Loading samples from {npy_filename}...")
    # Expected shape: (n_trajectories, n_steps, d)
    sample_data = np.load(npy_filename)
else:
    print(f"'{npy_filename}' not found. Generating dummy data for visualization...")
    # Generate dummy data (longer than T=10 to test clipping)
    dummy_T_total = 14.0
    dummy_steps = int(dummy_T_total / dt_sample) + 1
    t_dummy_eval = np.arange(dummy_steps) * dt_sample

    temp_list = []
    for x0 in x0_hi:
        sol = solve_ivp(f_true, (0.0, dummy_T_total), [x0], t_eval=t_dummy_eval)
        y_samp = sol.y[0]
        y_samp += np.random.normal(0, 0.015, size=y_samp.shape)  # Add noise
        temp_list.append(y_samp)

    # Shape: (2, steps, 1)
    sample_data = np.array(temp_list)[:, :, np.newaxis]

# Calculate physical time for samples: t = step_index * dt
num_sample_steps = sample_data.shape[1]
t_all_samples = np.arange(num_sample_steps) * dt_sample

# =========================================================
# 1) TRUE vector field (black)
# =========================================================
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

vf_true = a_true * xg + b_true
ax.plot(xg, vf_true, color=TRUE_BLACK, lw=2.0)
ax.axhline(0.0, color="0.6", lw=1.2)

ax.set_xlim(xg.min(), xg.max())
apply_axis_style(ax, tick_labelsize=46, tick_length=6, y_ndecimals=1, y_width=5)

finalize_and_save(fig, "id9_fig_true_vectorfield.svg")

# =========================================================
# 2) REC vector field (blue)
# =========================================================
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

vf_rec = a_rec * xg + b_rec
ax.plot(xg, vf_rec, color=REC_BLUE, lw=2.0)
ax.axhline(0.0, color="0.6", lw=1.2)

ax.set_xlim(xg.min(), xg.max())
apply_axis_style(ax, tick_labelsize=46, tick_length=6, y_ndecimals=1, y_width=5)

finalize_and_save(fig, "id9_fig_rec_vectorfield.svg")

# =========================================================
# Shared y-limits for trajectory plots (so true/rec comparable)
# =========================================================
ymin_traj, ymax_traj = np.inf, -np.inf
for x0 in x0_many[::4]:
    _, xt = integrate(f_true, x0, T, 400)
    _, xr = integrate(f_rec, x0, T, 400)
    ymin_traj = min(ymin_traj, float(np.min(xt)), float(np.min(xr)))
    ymax_traj = max(ymax_traj, float(np.max(xt)), float(np.max(xr)))

ymin_traj -= 0.02
ymax_traj += 0.02

# =========================================================
# 3) TRUE trajectories (highlighted = black) + SAMPLES
# =========================================================
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

# A. Background
for x0 in x0_many:
    t, xt = integrate(f_true, x0, T, n_t)
    ax.plot(t, xt, color="0.80", lw=1.8)

# B. Highlighted + Samples
for i, x0 in enumerate(x0_hi):
    # 1. Line
    t, xt = integrate(f_true, x0, T, n_t)
    ax.plot(t, xt, color=TRUE_BLACK, lw=5.2, zorder=5)

    # 2. Samples (Clipped to T)
    if sample_data is not None and i < sample_data.shape[0]:
        y_all = sample_data[i, :, 0]

        # Mask: keep only t <= T
        valid_mask = t_all_samples <= T
        t_plot = t_all_samples[valid_mask]
        y_plot = y_all[valid_mask]

        ax.scatter(t_plot, y_plot,
                   s=150,
                   marker='o',
                   facecolors='white',
                   edgecolors=TRUE_BLACK,
                   linewidths=2.5,
                   zorder=10)

ax.set_xlim(0.0, T)
ax.set_ylim(ymin_traj, ymax_traj)
apply_axis_style(ax, tick_labelsize=46, tick_length=6, y_ndecimals=1, y_width=5)

finalize_and_save(fig, "id9_fig_true_trajectories.svg")

# =========================================================
# 4) REC trajectories (highlighted overlay: TRUE black solid, REC blue dashed)
# =========================================================
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

# background: recovered family in gray (keep original intent)
for x0 in x0_many:
    t, xr = integrate(f_rec, x0, T, n_t)
    ax.plot(t, xr, color="0.80", lw=1.8)

# highlighted: overlay TRUE (black solid) + REC (blue dashed)
for x0 in x0_hi:
    t, xt = integrate(f_true, x0, T, n_t)
    _, xr = integrate(f_rec, x0, T, n_t)

    ax.plot(t, xt, color=TRUE_BLACK, lw=5.2)  # TRUE: black solid
    ax.plot(t, xr, color=REC_BLUE, lw=5.2, ls="--")  # REC : blue dashed

ax.set_xlim(0.0, T)
ax.set_ylim(ymin_traj, ymax_traj)
apply_axis_style(ax, tick_labelsize=46, tick_length=6, y_ndecimals=1, y_width=5)

finalize_and_save(fig, "id9_fig_rec_trajectories_overlay.svg")


# =========================================================
# Compute + write metrics to txt
# =========================================================
def f_true_scalar(x):
    return a_true * x + b_true


def f_rec_scalar(x):
    return a_rec * x + b_rec


E_L2_vf, E_rel_vf = compute_vectorfield_errors_1d(f_true_scalar, f_rec_scalar, Omega, N=10001)
print(f"Calculating trajectory errors over {len(x0_many)} ICs...")

traj_L2_list = []
traj_rel_list = []

for x0 in x0_many:
    _, yt = integrate(f_true, x0, T, n_t)
    _, yr = integrate(f_rec, x0, T, n_t)
    diff = yr - yt

    e_l2 = float(np.sqrt(np.mean(diff ** 2)))
    traj_L2_list.append(e_l2)

    norm_diff = float(np.linalg.norm(diff))
    norm_true = float(np.linalg.norm(yt))
    e_rel = 0.0 if norm_true < 1e-15 else (norm_diff / norm_true)
    traj_rel_list.append(float(e_rel))

avg_traj_L2 = float(np.mean(traj_L2_list))
avg_traj_rel = float(np.mean(traj_rel_list))

out_txt = "id9_system_recovery_analysis.txt"
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("id9 system recovery analysis\n\n")
    f.write("True ODE: dx(1)/dt = -0.60*x(1) + 0.32*1\n")
    f.write("Rec  ODE: dx(1)/dt = -0.600124*x(1) + 0.320036*1\n\n")

    f.write("Vector-field Errors (Global):\n")
    f.write(f"  Omega = [{Omega[0]}, {Omega[1]}]\n")
    f.write(f"  E_L2  = {E_L2_vf:.12e}\n")
    f.write(f"  E_rel = {E_rel_vf:.12e}\n\n")

    f.write("Trajectory Errors (Averaged over background ICs):\n")
    f.write(f"  Avg E_L2 (RMSE)    = {avg_traj_L2:.12e}\n")
    f.write(f"  Avg E_rel (RelErr) = {avg_traj_rel:.12e}\n\n")

    f.write("Parameter Error:\n")
    f.write(f"  da = {a_rec - a_true:.9f}\n")
    f.write(f"  db = {b_rec - b_true:.9f}\n")

print("[OK] Saved 4 figures:")
print("  id9_fig_true_vectorfield.svg")
print("  id9_fig_rec_vectorfield.svg")
print("  id9_fig_true_trajectories.svg")
print("  id9_fig_rec_trajectories_overlay.svg")
print("[OK] Saved metrics:")
print(f"  {out_txt}")