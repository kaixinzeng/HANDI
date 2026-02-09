import os
import re
import numpy as np
import pandas as pd
import argparse
from compute_nrmse_names import *
from tqdm import tqdm
from pysindy.feature_library import PolynomialLibrary
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

def main():
    args = parse_arguments()
    system = args.system
    noise = args.noise
    initial_conditions_file = args.init_conditions_file
    true_equation_file = args.true_equation
    dt = args.dt
    grid_size = args.grid_size
    dim = args.dim
    degree = args.degree
    t_plot = args.t_plot
    output_dir = args.output_dir
    x_range_min = args.x_range_min
    x_range_max = args.x_range_max
    include_bias = args.include_bias
    
    t = np.linspace(0, t_plot, t_plot * 100)

    x_traj = np.load(initial_conditions_file)
    initial_conditions = x_traj[:,0,:]
    num_traj = len(initial_conditions)

    filename_dt = dt 
    filename_nrmse = [
        args.path_edmd.format(system=system, noise=noise),
        args.path_sindy.format(system=system, noise=noise),
        args.path_handi.format(system=system, noise=noise),
        args.path_pse.format(system=system, noise=noise),
        args.path_gedmd.format(system=system, noise=noise),
        args.path_wsindy.format(system=system, noise=noise),
        args.path_sr3.format(system=system, noise=noise)
    ]

    filename_mse = list(filename_nrmse)

    method_names_nrmse = ['EDMD', 'SINDy', 'HANDI','PSE','gEDMD', 'wSINDy', 'SR3']
    method_names_mse = ['EDMD', 'SINDy', 'HANDI','PSE','gEDMD', 'wSINDy', 'SR3']
    # method_names_nrmse = ['EDMD','SINDy', 'HANDI', 'PSE','gEDMD', 'SR3']
    # method_names_mse = ['EDMD','SINDy', 'HANDI', 'PSE','gEDMD', 'SR3']
    # mse_slices_total = {steps: {name: [] for name in method_names_mse} for steps in mse_slice_steps}
    nrmse_total_list = {name: np.nan for name in method_names_nrmse}
    mse_all_points = {name: [] for name in method_names_mse}
    all_invalid_counts = {name: 0 for name in method_names_mse}

    library = PolynomialLibrary(degree=degree, include_bias=include_bias)
    library.fit(np.random.rand(10, dim))
    names = library.get_feature_names([f"x{i+1}" for i in range(dim)])

    try:
        true_coeff = build_true_coeff_from_file(true_equation_file, names, dim)

        for idx0, file in enumerate(filename_nrmse):
            method_name = method_names_nrmse[idx0]
            current_file = file
            
            if method_name == 'PSE':
                new_pse_file = file.replace('.txt', f'_new.txt')
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
        nrmse_total_list = {name: np.nan for name in method_names_nrmse}


    for n, x0 in enumerate(tqdm(initial_conditions)):
        true_equation = parse_equations(true_equation_file)
        true_model = create_learned_model(true_equation)

        true_sol = rk4_fixed_step(true_model, t, x0)
        x1_true = true_sol[:, 0]
        x_true = true_sol

        learned_traj = []
        
        for idx, file in enumerate(filename_mse):
            method_name = method_names_mse[idx]
            equations_list = parse_equations(file)
            if equations_list and len(equations_list) == dim:
                learned_model = create_learned_model(equations_list)

                pred_sol = rk4_fixed_step(learned_model, t, x0)
                x1_pred = pred_sol[:, 0]

                learned_traj.append((x1_pred))

            else:
                learned_traj.append((np.array([])))

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(t[:len(x1_true)], x1_true, color="#403E3E", linewidth=2, label='True')
            ax.plot(t[:len(x1_pred)], x1_pred, color="#D62728", linewidth=2, label='Predicted')
            ax.set_title(f'{method_name} - {noise} - (IC {n})')
            ax.legend()
            filepath = os.path.join(output_dir, f'{noise}', f"{method_name}")
            os.makedirs(filepath, exist_ok=True)
            plt.savefig(os.path.join(filepath, f'cyc_{n}.png'))
            plt.close()

        for idx,(x1_pred) in enumerate(learned_traj):
            name = method_names_mse[idx]

            if not np.all(np.isfinite(x1_pred)):
                mse = np.inf
                mse_all_points[name].append(min(mse, 1e3))
                all_invalid_counts[name] += 1
                continue

            if len(x1_pred) < len(x1_true):
                pad_val = x1_pred[-1]
                pad_len = len(x1_true) - len(x1_pred)
                x1_pred = np.concatenate([x1_pred, np.full(pad_len, pad_val)])
                all_invalid_counts[name] += 1

            x_pred = np.vstack([x1_pred]).T
            mse = np.mean((x_true - x_pred)**2)
            mse_all_points[name].append(min(mse, 1e3))
  
    output_dir = f'{output_dir}/{noise}'
    os.makedirs(output_dir, exist_ok=True)

    output_dt_label = dt
    summary_df_nrmse = pd.DataFrame({
        'Method': method_names_nrmse,
        "NRMSE": [nrmse_total_list[name] for name in method_names_nrmse]
    })
    summary_csv_nrmse = f'{output_dir}/{system}_nrmse_dt{output_dt_label}.csv'
    summary_df_nrmse.to_csv(summary_csv_nrmse, index=False, float_format='%.8e')
    print(f"Summary NRMSE saved to {summary_csv_nrmse}")

    summary_df_valid = pd.DataFrame({
        'Method': method_names_mse,
        'valid_rate': [1 - all_invalid_counts[name] / num_traj for name in method_names_mse]
    })
    summary_csv_valid = f'{output_dir}/{system}_valid_rate_dt{output_dt_label}.csv'
    summary_df_valid.to_csv(summary_csv_valid, index=False, float_format='%.8e')
    print(f"Summary Valid Rate saved to {summary_csv_valid}")

    mean_mse_no_inf = compute_mean_mse_no_inf(mse_all_points)
    summary_df_mse_mean = pd.DataFrame({
        'Method': method_names_mse,
        'mean_MSE': [mean_mse_no_inf[name] for name in method_names_mse]
    })
    summary_csv_mse_mean = f"{output_dir}/{system}_mse_mean_dt{output_dt_label}.csv"
    summary_df_mse_mean.to_csv(summary_csv_mse_mean, index=False, float_format='%.8e')
    print(f"Mean MSE (finite only) saved to {summary_csv_mse_mean}")

    mse_all_points = pd.DataFrame({name: mse_all_points[name] for name in method_names_mse})
    csv_mse_all_points = f"{output_dir}/{system}_mse_all_dt{output_dt_label}.csv"
    mse_all_points.to_csv(csv_mse_all_points, index=False, float_format='%.8e')
    print(f"MSE including inf saved to {csv_mse_all_points}")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--system", type=str, default="id6", help="System name for output csv, e.g., NI")
    parser.add_argument("--noise", type=str, default="0%", help="Noise level directory name, e.g., 0%, 0.125%")
    parser.add_argument("--init_conditions_file", type=str, default="test_benchmark/results/true/id6.npy",help="path to true system equation file")
    parser.add_argument("--true_equation", type=str, default="test_benchmark/results/true/ground_truth_equations_id6.txt",help="path to true system equation file")
    parser.add_argument("--path_edmd", type=str, default="test_benchmark/results/{system}/edmd/{noise}/2.25+2/equations.txt")
    parser.add_argument("--path_sindy", type=str,default="test_benchmark/results/{system}/SINDy/{noise}/2.25+2/sindy_equation_best_2.25+2_stride1.txt")
    parser.add_argument("--path_handi", type=str,default="test_benchmark/results/{system}/HANDI/{noise}/best_equations_dt2.250.txt")
    parser.add_argument("--path_pse", type=str,default="test_benchmark/results/{system}/pse/{noise}/2.25+2/final_ode_system_from_psrn.txt")
    parser.add_argument("--path_gedmd", type=str,default="test_benchmark/results/{system}/gedmd/{noise}/2.25+2/equations.txt")
    parser.add_argument("--path_wsindy", type=str,default="test_benchmark/results/{system}/wsindy/{noise}/tune_best_equations.txt")
    parser.add_argument("--path_sr3", type=str,default="test_benchmark/results/{system}/SR3/{noise}/tune_best_equations.txt")
    parser.add_argument('--dt', type=float, default=2.25, help='Time step for downsampling/evaluation (dt).')
    parser.add_argument('--t_plot', type=int, default=20, help='Time factor for total integration time (T_max = t_plot * scale).')

    parser.add_argument('--grid_size', type=int, default=100, help='Number of points in each dimension for initial condition grid (grid_size x grid_size).')
    parser.add_argument('--x_range_min', type=float, default=0.05, help='Minimum value for initial condition range.')
    parser.add_argument('--x_range_max', type=float, default=4.5, help='Maximum value for initial condition range.')
    
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree for feature library.')
    parser.add_argument('--dim', type=int, default=1, help='Dimension of the system (dim).')
    parser.add_argument('--include_bias', type=bool, default=False, help='Whether to include bias in the feature library.')

    parser.add_argument('--output_dir', type=str, default='test_benchmark/test/id6', help='Base directory for saving output CSV files.')
    
    return parser.parse_args()

if __name__ == '__main__':
    main()