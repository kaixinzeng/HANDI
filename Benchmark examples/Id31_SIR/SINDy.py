import numpy as np
import pandas as pd
import pysindy as ps
from tqdm import tqdm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import re

path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)

def format_sindy_equations_to_txt(model, save_path, coef_tol=1e-12, sort_by_abs=True):
    coeffs = np.array(model.coefficients())
    n_outputs, n_features = coeffs.shape
    feature = model.get_feature_names()

    def expand_term(term):
        s = term.replace(" ", "*")
        s = re.sub(r'\bx(\d+)\b', lambda m: f"x({m.group(1)})", s)
        pattern = re.compile(r'(x\(\d+\))\^(\d+)')
        while True:
            m = pattern.search(s)
            if not m:
                break
            var, p = m.group(1), int(m.group(2))
            s = s[:m.start()] + "*".join([var] * p) + s[m.end():]
        return s

    with open(save_path, 'w', encoding='utf-8') as f_eq:
        for out_idx in range(n_outputs):
            row = coeffs[out_idx, :]
            terms = []
            for coef, fname in zip(row, feature):
                if abs(coef) < coef_tol:
                    continue
                terms.append((coef, expand_term(fname)))

            if sort_by_abs:
                terms.sort(key=lambda x: -abs(x[0]))

            lhs = f"dx({out_idx+1})/dt = "
            if not terms:
                eq_line = lhs
            else:
                first_coef, first_term = terms[0]
                parts = [f"{first_coef:.6g}*{first_term}"]
                for coef, term in terms[1:]:
                    if coef < 0:
                        parts.append(f" - {abs(coef):.6g}*{term}")
                    else:
                        parts.append(f" + {coef:.6g}*{term}")
                eq_line = lhs + "".join(parts)

            f_eq.write(eq_line + "\n")


def convert_sindy_coeffs_to_L_hat(model):
    return model.coefficients()

def fixed_point_system(x):
    """
    Defines the ground truth dynamics for the fixed-point system.
    This is used to generate the test data to evaluate the model's accuracy.
    """
    x1, x2 = x[:, 0], x[:, 1]
    r2 = x1 ** 2 + x2 ** 2
    return np.stack([-3.0 * x2 - x1 * r2, 3.0 * x1 - x2 * r2], axis=1)

def make_uniform_test_set(num_points, dim, low=-1.0, high=1.0):
    """Creates a test set of points uniformly distributed on a grid."""
    n_per_axis = int(round(num_points ** (1.0 / dim)))
    axes = [np.linspace(low, high, n_per_axis) for _ in range(dim)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)
    if grid.shape[0] >= num_points:
        idx = np.random.choice(grid.shape[0], num_points, replace=False)
        return grid[idx]
    raise ValueError("Grid does not have enough points to sample from.")

def compute_nrmse(w_true, w_pred):
    w_pred = w_pred.T
    rmse = np.sqrt(np.mean((w_true - w_pred) ** 2))
    nonzero_mask = w_true != 0
    if not np.any(nonzero_mask):
        return rmse
    avg_abs_w = np.mean(np.abs(w_true[nonzero_mask]))
    return rmse / avg_abs_w

data_filename_pattern="./data_num30_traj1000_[-1,1]/fixed_point_{}.npy"
# --- General experiment parameters ---
DELTA_T_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]  # down sampled deltaT
STLSQ_THRESHOLD = 0.0001
NUM_TEST_PTS = 100
test_low, test_high = -1, 1
POLYNOMIAL_DEGREE = 7
dt = 0.01 #original deltaT 
dim = 2 # dimension of the system

RESULTS_DIR = f'./sindy/result_threshod{STLSQ_THRESHOLD}/poly{POLYNOMIAL_DEGREE}'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

TRAJECTORY_DIR = f'./sindy/trajectory_threshod{STLSQ_THRESHOLD}/poly{POLYNOMIAL_DEGREE}'
if not os.path.exists(TRAJECTORY_DIR):
    os.makedirs(TRAJECTORY_DIR)

OUTPUT_CSV_BASE = f"./sindy/mse_threshod{STLSQ_THRESHOLD}/poly{POLYNOMIAL_DEGREE}"
if not os.path.exists(OUTPUT_CSV_BASE):
    os.makedirs(OUTPUT_CSV_BASE)

X_test = make_uniform_test_set(NUM_TEST_PTS, dim, test_low, test_high)
f_true_test = fixed_point_system(X_test)

library = ps.PolynomialLibrary(degree=POLYNOMIAL_DEGREE, include_bias=False)
optimizer = ps.STLSQ(threshold=STLSQ_THRESHOLD)
feature_names = [f"x{i + 1}" for i in range(dim)]

all_results = []
for delta_t in tqdm(DELTA_T_VALUES):
    data_filename = data_filename_pattern.format(int(delta_t*100))
    data = np.load(data_filename)
    N_traj, Nsim, _ = data.shape
    X_list = [data[i] for i in range(N_traj)]
    t_list = [np.arange(data[i].shape[0]) * delta_t for i in range(N_traj)]

    try:
        model = ps.SINDy(
            feature_library=library,
            optimizer=optimizer,
            discrete_time=False,
            feature_names=feature_names
        )
        model.fit(X_list, t=t_list)

        equation_filename = f"SINDY_equation_poly_{POLYNOMIAL_DEGREE}_dt_{delta_t:.2f}.txt"
        equation_path = os.path.join(RESULTS_DIR, equation_filename)
        format_sindy_equations_to_txt(model, equation_path)

        f_coeff = model.coefficients()

        names = library.get_feature_names(feature_names)
        num_features = len(names)
        coeff1 = np.zeros(num_features)
        coeff2 = np.zeros(num_features)

        for i, name in enumerate(names):
            if name == "x2":
                coeff1[i] = -3.0
            elif name == "x1^3":
                coeff1[i] = -1.0
            elif name == "x1 x2^2":
                coeff1[i] = -1.0
            elif name == "x1":
                coeff2[i] = 3.0
            elif name == "x1^2 x2":
                coeff2[i] = -1.0
            elif name == "x2^3":
                coeff2[i] = -1.0

        true_coeff = np.stack([coeff1, coeff2], axis=0).T
        nrmse = compute_nrmse(true_coeff, f_coeff)

        f_pred_test = model.predict(X_test)
        diff = f_true_test - f_pred_test
        mse = np.mean(np.sum(diff ** 2, axis=1))

    except Exception as e:
        print(f"An exception occurred for dt={delta_t}: {e}")
        mse, nrmse = np.nan, np.nan

    all_results.append(dict(delta_t=delta_t, nrmse=nrmse, mse=mse))

    if not np.isnan(mse):
        x0 = [0.42275418, -0.33623535]
        t_plot = np.linspace(0, 5, 500)

        def true_model_ode(t, y):
            return fixed_point_system(np.array(y).reshape(1, -1)).flatten()

        def sindy_ode_model(t, y):
            y_reshaped = y.reshape(1, -1)
            dydt = model.predict(y_reshaped)
            return dydt.flatten()

        true_solution = solve_ivp(true_model_ode, [t_plot[0], t_plot[-1]], x0, t_eval=t_plot, rtol=1e-8, atol=1e-8)
        learned_solution = solve_ivp(sindy_ode_model, [t_plot[0], t_plot[-1]], x0, t_eval=t_plot, rtol=1e-8, atol=1e-8)

        plt.figure(figsize=(8, 6))
        plt.plot(true_solution.y[0], true_solution.y[1], color='#3E68A6', linewidth=2, label='True Trajectory')
        plt.plot(learned_solution.y[0], learned_solution.y[1], color='#963644', linestyle='--', linewidth=2,
                 label=f'Learned (Δt={delta_t:.2f})')

        training_traj_to_plot = data[4]
        indices = min(30, int(10/delta_t))
        plt.scatter(training_traj_to_plot[:indices, 0], training_traj_to_plot[:indices, 1],
                    color='#1E78B5', s=30, alpha=0.8, marker='s', label='Training Samples')

        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.title('Sindy: True vs Learned Models', fontsize=16, weight='bold')
        plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(False)
        plt.axis('equal')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()

        plot_path = os.path.join(TRAJECTORY_DIR, f'dt{delta_t:.2f}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

if all_results:
    out_csv = f"{OUTPUT_CSV_BASE}/mse and nrmse.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)

    mse_csv_path = f"{OUTPUT_CSV_BASE}/mse.csv"
    pd.DataFrame([r['mse'] for r in all_results]).to_csv(mse_csv_path, index=False, header=False)

    nrmse_csv_path = f"{OUTPUT_CSV_BASE}/nrmse.csv"
    pd.DataFrame([r['nrmse'] for r in all_results]).to_csv(nrmse_csv_path, index=False, header=False)
else:
    print("No results were generated.")