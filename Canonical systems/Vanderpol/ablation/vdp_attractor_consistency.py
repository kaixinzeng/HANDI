import os
import re
import numpy as np
import pandas as pd
import argparse
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from compute_nrmse_names import *
from tqdm import tqdm
from pysindy.feature_library import PolynomialLibrary
import matplotlib.pyplot as plt


def get_analysis_segment(x1, x2, t_full, t_start_ratio=0.5):
    N = len(t_full)
    start_idx = int(N * t_start_ratio)

    if start_idx >= N:
        return np.array([]), np.array([]), np.array([])

    return x1[start_idx:], x2[start_idx:], t_full[start_idx:]


def compute_R(x1_seg, x2_seg):
    return np.sqrt(x1_seg**2 + x2_seg**2)


def compute_velocity_magnitude(x1_seg, x2_seg, t_seg, system_func):

    v_list = []
    for i in range(len(t_seg)):
        state = np.array([x1_seg[i], x2_seg[i]])
        t_val = t_seg[i]
        dxdt = np.array(system_func(t_val, state))
        v = np.linalg.norm(dxdt)
        v_list.append(v)
    return np.array(v_list)

def compute_velocity_magnitude_true(x1_seg, x2_seg, t_seg, system_func):

    v_list = []
    for i in range(len(t_seg)):
        state = np.array([x1_seg[i], x2_seg[i]])
        t_val = t_seg[i]
        dxdt = np.array(system_func(state,t_val))
        v = np.linalg.norm(dxdt)
        v_list.append(v)
    return np.array(v_list)


def is_vdp_attractor_consistent(
        x1_pred, x2_pred, x1_true, x2_true, t_full,
        true_system_func, pred_system_func,
        delta_amp=0.2, delta_period=0.5, Q_ratio_threshold=0.5,eps_low_factor=0.5,eps_high_factor=2
):

    x1_p, x2_p, t_seg = get_analysis_segment(x1_pred, x2_pred, t_full)
    x1_t, x2_t, _ = get_analysis_segment(x1_true, x2_true, t_full)

    if not(len(x1_p) == len(x1_t)): return False

    zc_p = np.where(np.diff(np.sign(x1_p)))[0]
    zc_t = np.where(np.diff(np.sign(x1_t)))[0]

    if len(zc_t) < 2:
        print('Warning: True system data too short.')
        return False
    if len(zc_p) < 2:
        return False

    T_pred = 2 * np.mean(np.diff(t_seg[zc_p]))
    T_true = 2 * np.mean(np.diff(t_seg[zc_t]))

    if not (1 - delta_period < T_pred / T_true < 1 + delta_period):
        return False

    R_p = np.sqrt(x1_p ** 2 + x2_p ** 2)
    R_t = np.sqrt(x1_t ** 2 + x2_t ** 2)

    R_t_mean = np.mean(R_t)
    R_p_mean = np.mean(R_p)

    if abs(R_p_mean - R_t_mean) / R_t_mean > delta_amp:
        return False
    if np.max(R_p) > eps_high_factor * R_t_mean or np.min(R_p) < eps_low_factor * R_t_mean:
        return False

    try:
        v_p = compute_velocity_magnitude(x1_p, x2_p, t_seg, pred_system_func)
        Q_p = np.quantile(v_p, 0.9) / (np.quantile(v_p, 0.1) + 1e-9)

        v_t = compute_velocity_magnitude_true(x1_t, x2_t, t_seg, true_system_func)
        Q_t = np.quantile(v_t, 0.9) / (np.quantile(v_t, 0.1) + 1e-9)

        if Q_p < Q_ratio_threshold * Q_t:
            return False
    except:
        return False

    return True


def main():
    args = parse_arguments()

    SYSTEMS = {
        "fixed": fixed_point_system,
        "linear": linear_system,
        "vdp": vdp,
        'cyc': limit_cycle_system,
        'duffing': duffing_system
    }

    system = args.system
    true_equation = args.true_equation
    dt = args.dt
    grid_size = args.grid_size
    dim = args.dim
    degree = args.degree
    t_eval = args.t_eval
    output_dir = args.output_dir
    xy_range_min = args.xy_range_min
    xy_range_max = args.xy_range_max
    mse_slice_steps = args.mse_slices_steps
    include_bias = args.include_bias

    t = np.linspace(0, t_eval, t_eval * 100)

    x_range = np.linspace(xy_range_min, xy_range_max, grid_size)
    y_range = np.linspace(xy_range_min, xy_range_max, grid_size)
    X1, X2 = np.meshgrid(x_range, y_range)
    initial_conditions = np.vstack([X1.ravel(), X2.ravel()]).T
    num_traj = len(initial_conditions)

    filename_dt = dt
    filename_nrmse = [
        args.path_handi.format(dt=f"{filename_dt:.3f}"),
        args.path_ablationA.format(dt=f"{filename_dt:.3f}"),
        args.path_ortho.format(dt=f"{filename_dt:.3f}"),
    ]

    filename_mse = list(filename_nrmse)
    method_names_mse = ['HANDI','ablationA','ortho']
    method_names_nrmse = ['HANDI','ablationA','ortho']

    acr_counts = {name: 0 for name in method_names_mse}
    all_invalid_counts = {name: 0 for name in method_names_mse}

    library = PolynomialLibrary(degree=degree, include_bias=include_bias)
    library.fit(np.random.rand(10, dim))
    names = library.get_feature_names([f"x{i + 1}" for i in range(dim)])

    true_system_func = SYSTEMS[args.system]

    try:
        true_coeff = build_true_coeff_from_file(true_equation, names, dim)
        nrmse_total_list = {}
        for idx0, file in enumerate(filename_nrmse):
            method_name = method_names_nrmse[idx0]
            current_file = file
            # if method_name == 'PSE':
            #     new_pse_file = file.replace('.txt', f'_new_{dt:.2f}.txt')
            #     change_equations(file, new_pse_file)
            #     filename_mse[idx0] = new_pse_file
            #     current_file = new_pse_file
            if method_name == 'wSINDy':
                new_wsindy_file = replace_dot_to_mul(file)
                filename_mse[idx0] = new_wsindy_file
                current_file = new_wsindy_file
            if method_name == 'SR3':
                new_sr3_file = replace_dot_to_mul(file)
                filename_mse[idx0] = new_sr3_file
                current_file = new_sr3_file
            if method_name == 'gEDMD':
                new_gedmd_file = replace_dot_to_mul(file)
                filename_mse[idx0] = new_gedmd_file
                current_file = new_gedmd_file

            pred_coeff = build_true_coeff_from_file(current_file, names, dim)
            nrmse = compute_nrmse(true_coeff, pred_coeff)
            nrmse_total_list[method_name] = nrmse
    except Exception as e:
        print(f"\n[WARNING] NRMSE calculation skipped: {e}")

    for n, x0 in enumerate(tqdm(initial_conditions)):

        true_solution = odeint(true_system_func, x0, t)
        x1_true = true_solution[:, 0]
        x2_true = true_solution[:, 1]

        learned_models = []
        learned_traj = []

        for idx, file in enumerate(filename_mse):
            method_name = method_names_mse[idx]
            equations_list = parse_and_prepare_equations(file, dim)
            if equations_list and len(equations_list) == dim:
                learned_model = create_learned_model(equations_list)
                learned_models.append(learned_model)
                sol = solve_ivp(learned_model, [t[0], t[-1]], x0, t_eval=t, method='RK45', rtol=1e-6, atol=1e-8)
                x1_pred = sol.y[0]
                x2_pred = sol.y[1]
                learned_traj.append((x1_pred, x2_pred))
            else:
                learned_traj.append((np.array([]), np.array([])))
                learned_models.append(None)
                all_invalid_counts[method_name] += 1

        for idx, (x1_pred, x2_pred) in enumerate(learned_traj):
            method_name = method_names_mse[idx]
            if len(x1_pred) == 0:
                continue

            pred_model = learned_models[idx]
            if pred_model is None:
                continue
            if (method_name == 'gEDMD' and n == 45) or (method_name == 'gEDMD' and n == 91):
                print('stop')

            consistent = is_vdp_attractor_consistent(
                x1_pred=x1_pred,
                x2_pred=x2_pred,
                x1_true=x1_true,
                x2_true=x2_true,
                t_full=t,
                true_system_func=true_system_func,
                pred_system_func=pred_model,
                delta_amp=0.2, delta_period=0.5, Q_ratio_threshold=0.65,eps_low_factor=0.5,eps_high_factor=2,
            )
            if consistent:
                acr_counts[method_name] += 1

            if (consistent == True and method_name == 'gEDMD'):
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(x1_true, x2_true, color="#403E3E", linewidth=2, label='True')
                ax.plot(x1_pred, x2_pred, color="#D62728", linewidth=2, label='Predicted')
                ax.set_title(f'{method_name} - Inconsistent Attractor (IC {n})')
                ax.legend()
                plt.savefig(f"debug_vdp_inconsistent_{n}.png")
                plt.close()

    total_trajs = initial_conditions.shape[0]
    acr_percentages = {name: acr_counts[name] / total_trajs for name in method_names_mse}

    os.makedirs(output_dir, exist_ok=True)
    acr_df = pd.DataFrame({
        'Method': method_names_mse,
        'ACR_Count': [acr_counts[name] for name in method_names_mse],
        'Total_Trajectories': [total_trajs] * len(method_names_mse),
        'ACR_Percentage': [acr_percentages[name] for name in method_names_mse]
    })

    acr_csv_path = f'{output_dir}/{system}_acr_dt{dt}_xyrange{xy_range_max}_grid{grid_size}.csv'
    acr_df.to_csv(acr_csv_path, index=False, float_format='%.4f')
    print(f"\n✅ ACR results saved to {acr_csv_path}")
    for method, pct in acr_percentages.items():
        print(f"  {method}: {pct:.2%} ({acr_counts[method]}/{total_trajs})")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='vdp', choices=['fixed', 'linear', 'vdp', 'cyc','duff'],
                        help='Select system dynamics for ground truth simulation.')
    parser.add_argument("--true_equation", type=str, default="ablation_study/vdp/results/true_equation.txt",help="path to true system equation file")
    parser.add_argument("--path_handi", type=str,default="ablation_study/vdp/results/residual/best_equations_dt{dt}.txt")
    parser.add_argument("--path_ablationA", type=str,default="ablation_study/vdp/results/ablationA/best_equations_dt{dt}.txt")
    parser.add_argument("--path_ortho", type=str,default="ablation_study/vdp/results/ortho/best_equations_dt{dt}.txt")

    parser.add_argument('--dt', type=float, default=0.4, help='Time step for downsampling/evaluation (dt).')
    parser.add_argument('--t_eval', type=int, default=24,
                        help='Time horizon for evaluation of attractor consistency. [0,t_eval]')

    parser.add_argument('--grid_size', type=int, default=10,
                        help='Number of points in each dimension for initial condition grid (grid_size x grid_size).')
    parser.add_argument('--xy_range_min', type=float, default=-3, help='Minimum value for initial condition range.')
    parser.add_argument('--xy_range_max', type=float, default=3, help='Maximum value for initial condition range.')

    parser.add_argument('--degree', type=int, default=5, help='Polynomial degree for feature library.')
    parser.add_argument('--dim', type=int, default=2, help='Dimension of the system (dim).')
    parser.add_argument('--include_bias', type=bool, default=False,
                        help='Whether to include bias in the feature library.')

    parser.add_argument('--output_dir', type=str, default='ablation_study/acr_results/vdp',
                        help='Base directory for saving output CSV files.')
    parser.add_argument('--mse_slices_steps', nargs='*', default=[10, 20, None], help="Steps for MSE slicing")

    return parser.parse_args()


if __name__ == '__main__':
    main()