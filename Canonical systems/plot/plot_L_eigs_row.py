import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sympy as sp
import re
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

def minus_formatter(x, pos):
    return f"{x:.1f}".replace("-", "−")

mpl.rcParams['axes.unicode_minus'] = True
path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)

system = 'fix'
dt = 0.4
G = np.load(f'eigen/{system}_best_L_hat_dt{dt:.3f}.npy')
eigvals, eigvecs = eig(G)
idx = np.argsort(np.abs(np.real(eigvals)))

main_eigvals = eigvals[idx]
main_eigvecs = eigvecs[:, idx]

fig, ax = plt.subplots(figsize=(9, 6))


# plt.scatter(main_eigvals.real, main_eigvals.imag, marker='o', s=260,
#             edgecolors="white", facecolors="#F8C5CC", linewidths=1.5, zorder=6)


# closest_idx = np.argsort(np.abs(main_eigvals.real))[:4]
# plt.scatter(main_eigvals.real[closest_idx], main_eigvals.imag[closest_idx],
#             marker='o', s=260, edgecolors="white", facecolors="#FA2727",
#             linewidths=1.5, zorder=6)

plt.scatter(main_eigvals.real, main_eigvals.imag, marker='o', s=260,
            edgecolors="white", facecolors="#BDC1C4", linewidths=1.5, zorder=6)

deep_blue = "#0075B0"
light_blue = "#8297A8"

magnitude = np.abs(main_eigvals)
top5_idx = np.argsort(magnitude)[:10]

# tol_imag = 5
# mask_near_3j = np.isclose(np.imag(main_eigvals[top5_idx]), 0.0, atol=tol_imag) | \
#                np.isclose(np.imag(main_eigvals[top5_idx]), -0.0, atol=tol_imag)
# indices_3j = top5_idx[mask_near_3j]
tol_imag = 1.5
mask_near_3j = np.isclose(np.real(main_eigvals[top5_idx]), 0.0, atol=tol_imag) | \
               np.isclose(np.real(main_eigvals[top5_idx]), -0.0, atol=tol_imag)
indices_3j = top5_idx[mask_near_3j]

if len(indices_3j) > 0:
    plt.scatter(main_eigvals.real[indices_3j], main_eigvals.imag[indices_3j],
                marker='o', s=260, edgecolors="white", facecolors=deep_blue,
                linewidths=1.5, zorder=6, label=r'$\pm 3j$ modes')
# plt.scatter(main_eigvals.real[0], main_eigvals.imag[0],
#                 marker='o', s=260, edgecolors="white", facecolors=deep_blue,
#                 linewidths=1.5, zorder=6, label=r'$\pm 3j$ modes')

tol_real_axis = 1e-1
real_axis_mask = np.isclose(main_eigvals.imag, 0.0, atol=tol_real_axis)
real_axis_indices = np.where(real_axis_mask)[0]

real_vals = main_eigvals.real[real_axis_indices]
sorted_real_idx = real_axis_indices[np.argsort(-real_vals)]

distances = np.abs(main_eigvals.real[sorted_real_idx])
max_dist = distances.max() if len(distances) > 0 else 1.0

norm_distances = distances / max_dist if max_dist > 0 else distances

from matplotlib.colors import to_rgb
def interpolate_color(c1, c2, t):
    c1_rgb = np.array(to_rgb(c1))
    c2_rgb = np.array(to_rgb(c2))
    return tuple((1 - t) * c1_rgb + t * c2_rgb)

blue_dark = deep_blue     # "#00008B"
blue_light = light_blue   # "#CCCCFF"

for idx in sorted_real_idx:
    d = abs(main_eigvals.real[idx])
    t = d / max_dist
    color = interpolate_color(blue_dark, blue_light, t)
    plt.scatter(main_eigvals.real[idx], main_eigvals.imag[idx],
                marker='o', s=260, edgecolors="white", facecolors=color,
                linewidths=1.5, zorder=6)

plt.axhline(0, color="black", linewidth=2, zorder=2)
plt.axvline(0, color="black", linewidth=2, zorder=2)
plt.grid(True, linestyle="--", alpha=0.6, zorder=0)

ax.xaxis.set_major_formatter(FuncFormatter(minus_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(minus_formatter))

plt.yticks(np.arange(-8, 8.1, 4))
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(False)

for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color("black")
    ax.spines[side].set_linewidth(3)

ax.axhline(0, color="black", linewidth=2, zorder=2)
ax.axvline(0, color="black", linewidth=2, zorder=2)

plt.tight_layout()
plt.savefig(f"spectrum_{system}_dt{dt:.3f}_L_new.svg", format="svg", bbox_inches="tight")
plt.show()