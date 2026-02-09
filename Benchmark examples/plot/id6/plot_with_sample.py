#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final (split into separate figures, NCS style):

1) True vector field (removed)
2) Recovered vector field (removed)

3) TRUE trajectories from many ICs + SAMPLE POINTS
   - background: gray
   - highlighted: black solid
   - samples: white circles with black edges (loaded from npy)

4) REC trajectories from many ICs:
   - for highlighted ICs: TRUE = black solid, REC = blue dashed

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter
import os


# =========================================================
# Style (NCS / Science Style)
# =========================================================
def set_style_like_ref():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 46,
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
# Axis style logic (L-shape, fixed-width y-labels)
# =========================================================
def apply_axis_style(ax, tick_labelsize=46, tick_length=6, y_ndecimals=1):
    # L-shaped axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(4.0)
    ax.spines["bottom"].set_linewidth(4.0)

    # x: 3 ticks
    ax.xaxis.set_major_locator(MaxNLocator(3))

    # y: Custom tick logic
    ymin, ymax = ax.get_ylim()
    cand = MaxNLocator(nbins=3).tick_values(ymin, ymax)
    cand = [v for v in cand if (ymin - 1e-12) <= v <= (ymax + 1e-12) and abs(v) > 1e-12]

    if len(cand) >= 2:
        yticks = [cand[0], cand[-1]]
    else:
        yticks = [ymin + 0.35 * (ymax - ymin), ymin + 0.75 * (ymax - ymin)]
        yticks = [v for v in yticks if abs(v) > 1e-12]

    ax.yaxis.set_major_locator(FixedLocator(yticks))

    # Scheme B: fixed-width (5 chars) WITHOUT '+'; monospace font
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:5.{y_ndecimals}f}"))

    # Tick params
    ax.tick_params(axis="x", direction="out", length=tick_length, width=1.5, labelsize=tick_labelsize)
    ax.tick_params(axis="y", direction="out", length=tick_length, width=1.5, labelsize=tick_labelsize)

    # Monospace only for y tick labels
    for lab in ax.get_yticklabels():
        lab.set_fontfamily("Arial")  # Or 'DejaVu Sans Mono' if strictly needed, but Arial usually works well

    # No axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")


def finalize_and_save(fig, path, left=0.11, right=0.99, bottom=0.12, top=0.99, dpi=300):
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# =========================================================
# System & Config
# =========================================================
REC_BLUE = "#4FA8D5"
TRUE_BLACK = "black"

a_true, b_true = 2.1, -0.5
a_rec, b_rec = 2.10172, -0.495336


def f_true(t, x):
    return [a_true * x[0] + b_true * x[0] ** 2]


def f_rec(t, x):
    return [a_rec * x[0] + b_rec * x[0] ** 2]


def integrate(fun, x0, T=10.0, n_t=800):
    t_eval = np.linspace(0.0, T, int(n_t))
    sol = solve_ivp(fun, (0.0, T), [float(x0)], t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0]


T = 10.0
n_t = 800

x0_many = np.linspace(0.05, 4.5, 80)
x0_hi = [0.13, 2.24]  # 2 highlighted trajectories

# =========================================================
# Load Sample Data (or generate dummy if missing)
# =========================================================
npy_filename = "id6_downsample_45.npy"
sample_data = None

if os.path.exists(npy_filename):
    print(f"Loading samples from {npy_filename}...")
    # Expected shape: (2, n_steps, d)
    sample_data = np.load(npy_filename)
else:
    print(f"'{npy_filename}' not found. Generating dummy data for visualization...")
    # Generate dummy data matching shape (2, 15, 1)
    dummy_steps = 15
    temp_list = []
    for x0 in x0_hi:
        # Sample form true trajectory
        _, y_samp = integrate(f_true, x0, T, dummy_steps)
        # Add some noise to make it look like "data"
        y_samp += np.random.normal(0, 0.05, size=y_samp.shape)
        temp_list.append(y_samp)

    # Stack to get (2, 15) then expand dims to (2, 15, 1)
    sample_data = np.array(temp_list)[:, :, np.newaxis]

# Determine time vector for samples (Assumes uniform sampling over [0, T])
num_sample_steps = 10
t_samples = np.linspace(0.0, T, num_sample_steps)

# =========================================================
# 3) TRUE trajectories + SAMPLES
# =========================================================
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

# A. Background (Gray)
for x0 in x0_many:
    t, xt = integrate(f_true, x0, T, n_t)
    ax.plot(t, xt, color="0.80", lw=1.8)

# B. Highlighted (Black) + Samples
# sample_data shape is (Trajectory, Time, Dim)
for i, x0 in enumerate(x0_hi):
    # 1. Continuous True Line
    t, xt = integrate(f_true, x0, T, n_t)
    ax.plot(t, xt, color=TRUE_BLACK, lw=5.2, zorder=5)

    # 2. Sample Points
    if i < sample_data.shape[0]:
        # Extract data: i-th trajectory, all times, 0-th dimension
        y_samp = sample_data[i, :, 0]

        ax.scatter(t_samples, y_samp,
                   s=150,  # Size of dots
                   marker='o',  # Circle
                   facecolors='white',  # Hollow-look (white fill)
                   edgecolors=TRUE_BLACK,  # Black edge
                   linewidths=2.5,  # Edge thickness
                   zorder=10,  # On top of lines
                   label="Samples" if i == 0 else "")

ax.set_xlim(0.0, T)
apply_axis_style(ax)

finalize_and_save(fig, "fig_true_trajectories_with_samples.svg")

# =========================================================
# 4) REC trajectories (Overlay) - Unchanged
# =========================================================
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

for x0 in x0_many:
    t, xr = integrate(f_rec, x0, T, n_t)
    ax.plot(t, xr, color="0.80", lw=1.8)

for x0 in x0_hi:
    t, xt = integrate(f_true, x0, T, n_t)
    _, xr = integrate(f_rec, x0, T, n_t)
    ax.plot(t, xt, color=TRUE_BLACK, lw=5.2)  # True
    ax.plot(t, xr, color=REC_BLUE, lw=5.2, ls="--")  # Rec

ax.set_xlim(0.0, T)
apply_axis_style(ax)

finalize_and_save(fig, "fig_rec_trajectories_overlay.svg")

print("=" * 40)
print("[OK] Done.")
print("Generated: fig_true_trajectories_with_samples.svg")
print("Generated: fig_rec_trajectories_overlay.svg")
print("=" * 40)