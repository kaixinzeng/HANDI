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



def calculate_long_term_mean(x_series, T_eval_start, T_eval_end):

    start_idx = int(len(x_series) * T_eval_start/T_eval_end)
    end_idx = int(len(x_series) * 1)
    x_segment = x_series[start_idx:end_idx]
    x_bar = np.mean(x_segment)
    return x_bar


def calculate_attractor_consistency_rate(x_true_x1, x_pred_x1, T_eval_start=0.5, T_eval_end=1.0):

    x_true_bar = calculate_long_term_mean(x_true_x1, T_eval_start, T_eval_end)

    x_pred_bar = calculate_long_term_mean(x_pred_x1, T_eval_start, T_eval_end)

    consistent = np.sign(x_true_bar) == np.sign(x_pred_bar)

    return consistent, x_true_bar, x_pred_bar


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

    # method_names_nrmse = ['EDMD', 'SINDy', 'HANDI', 'PSE', 'gEDMD', 'wSINDy', 'SR3']
    # method_names_mse = ['EDMD', 'SINDy', 'HANDI', 'PSE', 'gEDMD', 'wSINDy', 'SR3']
    # method_names_nrmse = ['SINDy', 'HANDI', 'PSE', 'wSINDy', 'SR3']
    # method_names_mse = ['SINDy', 'HANDI', 'PSE', 'wSINDy', 'SR3']
    method_names_mse = ['HANDI','ablationA','ortho']
    method_names_nrmse = ['HANDI','ablationA','ortho']
    mse_slices_total = {steps: {name: [] for name in method_names_mse} for steps in mse_slice_steps}
    nrmse_total_list = {name: np.nan for name in method_names_nrmse}

    # ACR
    acr_results = {name: [] for name in method_names_mse}
    acr_counts = {name: 0 for name in method_names_mse}

    library = PolynomialLibrary(degree=degree, include_bias=include_bias)
    library.fit(np.random.rand(10, dim))
    names = library.get_feature_names([f"x{i + 1}" for i in range(dim)])

    try:
        true_coeff = build_true_coeff_from_file(true_equation, names, dim)

        for idx0, file in enumerate(filename_nrmse):
            method_name = method_names_nrmse[idx0]
            current_file = file

            if method_name == 'PSE':
                new_pse_file = file.replace('.txt', f'_new_{dt:.2f}.txt')
                change_equations(file, new_pse_file)
                filename_mse[idx0] = new_pse_file
                current_file = new_pse_file
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
        print(f"\n[WARNING] NRMSE calculation skipped or failed: {e}")


    all_invalid_counts = {name: 0 for name in method_names_mse}

    for n, x0 in enumerate(tqdm(initial_conditions)):

        true_solution = odeint(SYSTEMS[args.system], x0, t)
        x1_true = true_solution[:, 0]
        x2_true = true_solution[:, 1]
        x_true = np.vstack([x1_true, x2_true]).T

        learned_traj = []
        valid_for_acr = True

        for idx, file in enumerate(filename_mse):
            method_name = method_names_mse[idx]
            equations_list = parse_and_prepare_equations(file, dim)

            if equations_list and len(equations_list) == dim:
                learned_model = create_learned_model(equations_list)

                sol = solve_ivp(learned_model, [t[0], t[-1]], x0, t_eval=t, method='RK45', rtol=1e-6, atol=1e-8)

                x1_pred = sol.y[0]
                x2_pred = sol.y[1]
                learned_traj.append((x1_pred, x2_pred))

            else:
                all_invalid_counts[method_name] += 1
                valid_for_acr = False
                learned_traj.append((np.array([]), np.array([])))

        for idx, (x1_pred, x2_pred) in enumerate(learned_traj):
            name = method_names_mse[idx]
            if len(x1_pred) > 0 and len(x1_pred) != len(x1_true):
                all_invalid_counts[name] += 1
                valid_for_acr = False

        # ACR
        for idx, (x1_pred, x2_pred) in enumerate(learned_traj):
            method_name = method_names_mse[idx]
            error = (x1_pred[-1] - x1_true[-1]) ** 2 + (x2_pred[-1] - x2_true[-1]) ** 2
            if len(x1_pred) > 0 and len(x1_pred) == len(x1_true) and error<8:
                consistent, x_true_bar, x_pred_bar = calculate_attractor_consistency_rate(
                    x1_true, x1_pred, T_eval_start=t_eval/2, T_eval_end=t_eval)
            else:
                consistent = False
            if consistent == False and method_name == 'HANDI':
                fig, ax = plt.subplots(figsize=(9, 6))
                plt.plot(x1_true, x2_true, color="#403E3E", linewidth=5, label='Truth' if idx == 0 else "",
                                zorder=1)
                plt.plot(x1_pred, x2_pred, color="#29477D", linewidth=5, label='Truth' if idx == 0 else "",
                                zorder=1)
                print(n)
            acr_results[method_name].append(consistent)
            if consistent:
                acr_counts[method_name] += 1


    acr_percentages = {}
    for method_name in method_names_mse:
        acr_percentages[method_name] = acr_counts[method_name] / initial_conditions.shape[0]

    os.makedirs(output_dir, exist_ok=True)
    output_dt_label = dt

    acr_df = pd.DataFrame({
        'Method': method_names_mse,
        'ACR_Count': [acr_counts[name] for name in method_names_mse],
        'Total_Trajectories': [initial_conditions.shape[0]] * len(method_names_mse),
        'ACR_Percentage': [acr_percentages[name] for name in method_names_mse]
    })

    acr_csv_path = f'{output_dir}/{system}_acr_dt{output_dt_label}_xyrange{xy_range_max}_grid{grid_size}.csv'
    acr_df.to_csv(acr_csv_path, index=False,float_format='%.2f')
    print(f"ACR results saved to {acr_csv_path}")
    print(f"ACR Results:")
    for method, percentage in acr_percentages.items():
        print(f"  {method}: {percentage:.2f} ({acr_counts[method]}/{initial_conditions.shape[0]})")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='duffing', choices=['fixed', 'linear', 'vdp', 'cyc'],
                        help='Select system dynamics for ground truth simulation.')
    parser.add_argument("--true_equation", type=str, default="ablation_study/duffing/results/true_equation.txt",help="path to true system equation file")
    parser.add_argument("--path_handi", type=str,default="ablation_study/duffing/results/residual/dt{dt}/best_equations_dt{dt}.txt")
    parser.add_argument("--path_ablationA", type=str,default="ablation_study/duffing/results/ablationA/dt{dt}/best_equations_dt{dt}.txt")
    parser.add_argument("--path_ortho", type=str,default="ablation_study/duffing/results/ortho/dt{dt}/best_equations_dt{dt}.txt")

    parser.add_argument('--dt', type=float, default=0.4, help='Time step for downsampling/evaluation (dt).')
    parser.add_argument('--t_eval', type=int, default=10,
                        help='Time horizon for evaluation of attractor consistency. [0,t_eval]')

    parser.add_argument('--grid_size', type=int, default=10,help='Number of points in each dimension for initial condition grid (grid_size x grid_size).')
    parser.add_argument('--xy_range_min', type=float, default=-4, help='Minimum value for initial condition range.')
    parser.add_argument('--xy_range_max', type=float, default=4, help='Maximum value for initial condition range.')

    parser.add_argument('--degree', type=int, default=3, help='Polynomial degree for feature library.')
    parser.add_argument('--dim', type=int, default=2, help='Dimension of the system (dim).')
    parser.add_argument('--include_bias', type=bool, default=False, help='Whether to include bias in the feature library.')

    parser.add_argument('--output_dir', type=str, default='ablation_study/acr_results/duffing',help='Base directory for saving output CSV files.')
    parser.add_argument('--mse_slices_steps', nargs='*', default=[10, 20, None], help="Steps for MSE slicing")

    return parser.parse_args()


if __name__ == '__main__':
    main()



