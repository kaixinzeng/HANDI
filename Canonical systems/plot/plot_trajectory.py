import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import re, os
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import matplotlib as mpl
# path_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(path_dir)

def true_model(x, t):
    x1, x2 = x[0], x[1]
    r_squared = x1**2 + x2**2
    
    dx1dt = 3 * x2 - x1 * (r_squared - 1)
    dx2dt = -3 * x1 - x2 * (r_squared - 1)
    return [dx1dt, dx2dt]

def load_and_simulate(filename, x0, t):
    with open(filename, 'r') as f:
        content = f.read()

    pattern = r"dx\(1\)/dt = (.*?)\ndx\(2\)/dt = (.*?)$"
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        raise ValueError(f"{filename} format does not meet requirements")

    eq1 = match.group(1).strip().replace('x(1)', 'x[0]').replace('x(2)', 'x[1]')
    eq2 = match.group(2).strip().replace('x(1)', 'x[0]').replace('x(2)', 'x[1]')

    def learned_model(x, t):
        dx1dt = eval(eq1)
        dx2dt = eval(eq2)
        return [dx1dt, dx2dt]

    sol = solve_ivp(
        lambda t, x: learned_model(x, t),
        [t[0], t[-1]], x0,
        t_eval=t, method="RK45",
        rtol=1e-6, atol=1e-8
    )
    return sol.y[0], sol.y[1]

x0=[1.2572631998551747, -1.0950035546168009]
dt=1.0
ht=0.1
scale=int(dt/ht)
num_traj=100
t_plot=3*scale
t = np.linspace(0, 30, 3000)
t_start = 0
t_end = 3000

true_solution = odeint(true_model, x0, t)
x1_true, x2_true = true_solution[:, 0], true_solution[:, 1]

fig, ax = plt.subplots(figsize=(6, 4))

plt.plot(t[t_start:t_end], x1_true[t_start:t_end], color='#000000', linestyle=(0, (3, 3)), linewidth=6, label='Truth',zorder=5)
plt.scatter(t[t_start:t_end:int(dt*100)], x1_true[t_start:t_end:int(dt*100)],
           facecolors='#ffffff',edgecolors='#000000', s=100, linewidths=2.5, marker='o', zorder=5)

methods = [
    {"name": "PSE", "file": f"cyc/results/PSE/equation_{dt:.1f}_new.txt", "color": "#D8A0A7", 'linewidth':6},
    {"name": "SR3", "file": f"cyc/results/sr3/cyc{dt*10:.0f}/tune_best_equations_mul.txt", "color": "#C9A1CB", 'linewidth':6},
    {"name": "SINDy", "file": f"cyc/results/sindy/SINDY_equation_poly_5_dt_{dt:.2f}.txt", "color": "#d06569", 'linewidth':6},
    {"name": "HANDI", "file": f"cyc/results/HANDI/edmd_runs_cyc{dt:.3f}/best_equations_dt{dt:.3f}.txt", "color": "#4FA8D5", 'linewidth':8},
]

for m in methods:
    try:
        x1_m, x2_m = load_and_simulate(m["file"], x0, t)
        plt.plot(t[t_start:t_end], x1_m[t_start:t_end], '-', linewidth=m['linewidth'], color=m["color"], label=m["name"], zorder=3)
    except Exception as e:
        print(f"{m['name']} error:", e)

plt.rcParams['mathtext.fontset'] = 'cm'
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
# plt.xlim(-2, 32)
ax.tick_params(axis='x', which='major', labelsize=20, length=4, width=1.5)
ax.tick_params(axis='y', which='major', labelsize=20, length=4, width=1.5)
for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color("black")
    ax.spines[side].set_linewidth(2)
ax.set_xlabel("$t$", fontsize=24)
ax.set_ylabel("$x_1$", fontsize=24)
# plt.xticks([])
# plt.yticks([])

plt.tight_layout()
plt.savefig('P3_1127/plot_cyc/1212/trajectory_cyc_dt1.0_x1_label.pdf', format="pdf", bbox_inches="tight")

