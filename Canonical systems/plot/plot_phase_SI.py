import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import re, os
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['axes.unicode_minus'] = True
path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)

def true_model(x, t):
    x1, x2 = x[0], x[1]
    dx1dt = -3 * x2 - x1 ** 3 - x1 * x2 ** 2
    dx2dt = 3 * x1 - x2 ** 3 - x1 ** 2 * x2
    return [dx1dt, dx2dt]

def load_learned_model(filename):
    try:
        with open(filename, "r") as f:
            content = f.read()

        pattern = r"dx\(1\)/dt = (.*?)\ndx\(2\)/dt = (.*?)(?=\n\s*\n|\Z)"
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

        if match:
            eq1_raw = match.group(1).strip()
            eq2_raw = match.group(2).strip()
            
            eq1_code = eq1_raw.replace("x(1)", "x[0]").replace("x(2)", "x[1]")
            eq2_code = eq2_raw.replace("x(1)", "x[0]").replace("x(2)", "x[1]")

            allowed_globals = {"np": np, "__builtins__": {}}
            return lambda x, t: [
                eval(eq1_code, allowed_globals, {"x": x, "t": t}),
                eval(eq2_code, allowed_globals, {"x": x, "t": t})
            ]
        else:
            print(f"Warning: could not extract equations from {filename}")
            return None
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

# X = np.random.uniform(-1, 1, size=(100, 2))
# np.save('random_data100.npy', X)
X0 = np.load('random_data100.npy')
dt = 1.0
num_step = int(dt*100)
t = np.linspace(0, 5, 500)

file_best = f"./ours_equations/best_equations_dt{dt:.3f}.txt"
file_sindy = f"./sindy_equations/SINDY_equation_poly_7_dt_{dt:.2f}.txt"
output_filename_ours = f'SI_1029/phase_ours_dt{dt:.3f}.svg'
output_filename_sindy = f'SI_1029/phase_sindy_dt{dt:.3f}.svg'
os.makedirs(os.path.dirname(output_filename_sindy), exist_ok=True)

model_best = load_learned_model(file_best)
model_sindy = load_learned_model(file_sindy)

fig, ax = plt.subplots(figsize=(9, 6))

for idx, x0 in tqdm(enumerate(X0), total=len(X0)):

    true_solution = odeint(true_model, x0, t)
    x1_true, x2_true = true_solution[:, 0], true_solution[:, 1]
    plt.plot(x1_true, x2_true, color="#403E3E", linewidth=5, label='Truth' if idx == 0 else "",zorder=1)

plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color("black")
    ax.spines[side].set_linewidth(3)
    
ax.set_xlabel("$x_1$", fontsize=35)
ax.set_ylabel("$x_2$", fontsize=35)

plt.tight_layout()
plt.savefig('SI_1029/phase_true.svg', format="svg", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(9, 6))
for idx, x0 in tqdm(enumerate(X0), total=len(X0)):
    if model_best:
        sol_best = solve_ivp(lambda t_, y_: model_best(y_, t_), [t[0], t[-1]], x0, t_eval=t,
                             method='RK45', rtol=1e-8, atol=1e-10)
        if sol_best.success:
            plt.plot(sol_best.y[0], sol_best.y[1], '-', linewidth=5,
                     color="#29477D", label='Best Eqn' if idx == 0 else "",zorder=4)

plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color("black")
    ax.spines[side].set_linewidth(3)

plt.tight_layout()
ax.set_xticks([])
ax.set_yticks([])

for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(False)
plt.savefig(output_filename_ours, format="svg", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(9, 6))
for idx, x0 in tqdm(enumerate(X0), total=len(X0)):
    #SINDY_equations
    if model_sindy:
        sol_sindy = solve_ivp(lambda t_, y_: model_sindy(y_, t_), [t[0], t[-1]], x0, t_eval=t,
                              method='RK45', rtol=1e-8, atol=1e-10)
        if sol_sindy.success:
            plt.plot(sol_sindy.y[0], sol_sindy.y[1], '-', linewidth=5,
                     color="#992121", label='SINDY' if idx == 0 else "", zorder=3)   
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color("black")
    ax.spines[side].set_linewidth(3)

plt.tight_layout()
ax.set_xticks([])
ax.set_yticks([])

for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(False)

plt.savefig(output_filename_sindy, format="svg", bbox_inches="tight")