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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def criterion_cyc_mean_norm_with_stat(x_traj_pred, x_traj_true, tail_start_idx, eps_cyc):
    x_tail_pred = x_traj_pred[tail_start_idx:]
    x_tail_true = x_traj_true[tail_start_idx:]

    norms_pred = np.linalg.norm(x_tail_pred, axis=1)
    norms_true = np.linalg.norm(x_tail_true, axis=1)

    mean_pred = np.mean(norms_pred)
    mean_true = np.mean(norms_true)

    if mean_true == 0:
        return False, np.nan

    rel_err = np.abs(mean_pred - mean_true) / mean_true
    is_valid = rel_err < eps_cyc

    return is_valid, rel_err


def criterion_cyc_period_with_stat(x_traj_pred, x_traj_true, t, tail_start_idx, delta_period):

    x_tail_pred = x_traj_pred[tail_start_idx:, 0]
    x_tail_true = x_traj_true[tail_start_idx:, 0]
    t_tail = t[tail_start_idx:]

    zc_pred = np.where(np.diff(np.sign(x_tail_pred)))[0]
    zc_true = np.where(np.diff(np.sign(x_tail_true)))[0]

    if len(zc_true) < 2 or len(zc_pred) < 2:
        return False, np.inf

    T_pred = 2.0 * np.mean(np.diff(t_tail[zc_pred]))
    T_true = 2.0 * np.mean(np.diff(t_tail[zc_true]))

    if T_true <= 0:
        return False, np.inf

    period_ratio = T_pred / T_true

    is_valid = (1 - delta_period) <= period_ratio <= (1 + delta_period)

    return is_valid, period_ratio


def main():
    args = parse_arguments()
    
    SYSTEMS = {
        "fixed": fixed_point_system,
        "linear": linear_system,
        "vdp": vdp,
        "cyc": limit_cycle_system,
        "duffing": duffing_system,
        'm1k16c8': m1k16c8_system,
        'm1k17c2': m1k17c2_system,
        'm1k25c20': m1k25c20_system
    }

    system = args.system
    true_equation_file = args.true_equation
    dt = args.dt
    grid_size = args.grid_size
    dim = args.dim
    degree = args.degree
    t_plot = args.t_plot
    output_dir = args.output_dir
    xy_range_min = args.xy_range_min
    xy_range_max = args.xy_range_max
    mse_slice_steps = args.mse_slices_steps
    include_bias = args.include_bias
    
    t = np.linspace(0, t_plot , t_plot * 100)             
    tail_start_idx = int(len(t) * args.tail_ratio)

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

    # method_names_nrmse = ['EDMD', 'SINDy', 'HANDI','PSE','gEDMD', 'wSINDy', 'SR3']
    # method_names_mse = ['EDMD', 'SINDy', 'HANDI','PSE','gEDMD', 'wSINDy', 'SR3']
    # method_names_nrmse = ['SINDy', 'HANDI', 'PSE', 'SR3']
    # method_names_mse = ['SINDy', 'HANDI', 'PSE', 'SR3']
    method_names_mse = ['HANDI','ablationA','ortho']
    method_names_nrmse = ['HANDI','ablationA','ortho']
    c1_valid = {name: [] for name in method_names_mse}
    c1_relative_error = {name: [] for name in method_names_mse}
    c1_valid_count = {name: 0 for name in method_names_mse}
    c2_valid = {name: [] for name in method_names_mse}
    c2_period_ratio = {name: [] for name in method_names_mse}
    c2_valid_count = {name: 0 for name in method_names_mse}
    structure_valid_count = {name: 0 for name in method_names_mse}

    library = PolynomialLibrary(degree=degree, include_bias=include_bias)
    library.fit(np.random.rand(10, dim))
    names = library.get_feature_names([f"x{i+1}" for i in range(dim)])

    try:
        true_coeff = build_true_coeff_from_file(true_equation_file, names, dim)

        for idx0, file in enumerate(filename_nrmse):
            method_name = method_names_nrmse[idx0]
            
            # if method_name == 'PSE':
            #     new_pse_file = file.replace('.txt', f'_new_{dt:.2f}.txt')
            #     change_equations(file, new_pse_file)
            #     filename_mse[idx0] = new_pse_file
            if method_name == 'wSINDy':
                new_wsindy_file = replace_dot_to_mul(file)
                filename_mse[idx0] = new_wsindy_file
            if method_name == 'SR3':
                new_sr3_file = replace_dot_to_mul(file)
                filename_mse[idx0] = new_sr3_file
            if method_name == 'gEDMD':
                new_gedmd_file = replace_dot_to_mul(file)
                filename_mse[idx0] = new_gedmd_file
    except Exception as e:
        print(f"failed: {e}")

    for n, x0 in enumerate(tqdm(initial_conditions)):

        true_solution = odeint(SYSTEMS[args.system], x0, t)

        for idx, file in enumerate(filename_mse):
            c1 = False
            rel_err = np.inf
            c2 = False
            period_ratio = np.inf
            method_name = method_names_mse[idx]

            equations_list = parse_and_prepare_equations(file, dim)
            if not equations_list or len(equations_list) != dim:
                continue
            learned_model = create_learned_model(equations_list)

            try:
                sol = solve_ivp(learned_model,[t[0], t[-1]],x0,t_eval=t,method='RK45',rtol=1e-6,atol=1e-8)
            except Exception:
                print(f"Error details: {Exception}")
                continue

            if sol.success:
                x_traj = sol.y.T
                if x_traj.shape[0] == len(t):
                    c1, rel_err = criterion_cyc_mean_norm_with_stat(x_traj_pred=x_traj, x_traj_true=true_solution, tail_start_idx=tail_start_idx,eps_cyc=args.eps_cyc)
                    c2, period_ratio = criterion_cyc_period_with_stat(x_traj_pred=x_traj, x_traj_true=true_solution, t=t, tail_start_idx=tail_start_idx, delta_period=args.delta_period)
            
            c1_valid[method_name].append(c1)
            c1_relative_error[method_name].append(rel_err)
            c2_valid[method_name].append(c2)
            c2_period_ratio[method_name].append(period_ratio)

            if c1:
                c1_valid_count[method_name] += 1
            if c2:
                c2_valid_count[method_name] += 1
            if c1 and c2:
                structure_valid_count[method_name] += 1

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(true_solution[:, 0], true_solution[:, 1], color="#403E3E", linewidth=2, label='True')
            ax.plot(x_traj[:, 0], x_traj[:, 1], color="#D62728", linewidth=2, label='Predicted')
            ax.set_title(f'{method_name} - Inconsistent Attractor (IC {n})')
            ax.legend()
            filepath = os.path.join(output_dir, f"{method_name}")
            os.makedirs(filepath, exist_ok=True)
            plt.savefig(os.path.join(filepath, f'cyc_{n}.png'))
            plt.close()
            
    os.makedirs(output_dir, exist_ok=True)
    

    df_structure_valid_count = pd.DataFrame({
        'Method': method_names_mse,
        'C1_Count':[c1_valid_count[name] for name in method_names_mse],
        'C1_Percentage':[c1_valid_count[name]/num_traj for name in method_names_mse],
        'C2_Count':[c2_valid_count[name] for name in method_names_mse],
        'C2_Percentage':[c2_valid_count[name]/num_traj for name in method_names_mse],
        'ACR_Count': [structure_valid_count[name] for name in method_names_mse],
        'ACR_Percentage': [structure_valid_count[name]/num_traj for name in method_names_mse]
    }) 
    structure_csv_path = f"{output_dir}/{system}_acr_dt{dt}.csv"
    df_structure_valid_count.to_csv(structure_csv_path, index=False, float_format="%.8f")

    df_c1_valid = pd.DataFrame({name: c1_valid[name] for name in method_names_mse})
    c1_valid_csv_path = f"{output_dir}/{system}_c1_valid_dt{dt}.csv"
    df_c1_valid.to_csv(c1_valid_csv_path, index=False, float_format="%.8f")

    df_c1_norms = pd.DataFrame({name: c1_relative_error[name] for name in method_names_mse})
    c1_norms_csv_path = f"{output_dir}/{system}_c1_norms_dt{dt}.csv"
    df_c1_norms.to_csv(c1_norms_csv_path, index=False, float_format="%.8f")

    df_c2_valid = pd.DataFrame({name: c2_valid[name] for name in method_names_mse})
    c2_valid_csv_path = f"{output_dir}/{system}_c2_valid_dt{dt}.csv"
    df_c2_valid.to_csv(c2_valid_csv_path, index=False, float_format="%.8f")

    df_c2_period = pd.DataFrame({name: c2_period_ratio[name] for name in method_names_mse})
    c2_period_csv_path = f"{output_dir}/{system}_c2_period_dt{dt}.csv"
    df_c2_period.to_csv(c2_period_csv_path, index=False, float_format="%.8f")

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--system', type=str, default='cyc',choices=['fixed', 'linear', 'vdp','cyc','duffing','m1k16c8'],help='Select system dynamics for ground truth simulation.')
    parser.add_argument("--true_equation", type=str, default="ablation_study/cyc/results/true_equation.txt",help="path to true system equation file")
    parser.add_argument("--path_handi", type=str,default="ablation_study/cyc/results/residual/best_equations_dt{dt}.txt")
    parser.add_argument("--path_ablationA", type=str,default="ablation_study/cyc/results/ablationA/best_equations_dt{dt}.txt")
    parser.add_argument("--path_ortho", type=str,default="ablation_study/cyc/results/ortho/best_equations_dt{dt}.txt")

    parser.add_argument('--dt', type=float, default=0.5, help='Time step for downsampling/evaluation (dt).')
    parser.add_argument('--t_plot', type=int, default=5, help='Time factor for total integration time (T_max = t_plot * scale).')
    parser.add_argument('--tail_ratio', type=float, default=0.5, help='Ratio of the tail segment of the trajectory to consider for analysis.')
    parser.add_argument('--eps_cyc', type=float, default=0.05, help='Tolerance for cycle consistency.')
    parser.add_argument('--delta_period', type=float, default=0.5, help='Relative tolerance for cycle period consistency')


    parser.add_argument('--grid_size', type=int, default=10, help='Number of points in each dimension for initial condition grid (grid_size x grid_size).')
    parser.add_argument('--xy_range_min', type=float, default=-1.5, help='Minimum value for initial condition range.')
    parser.add_argument('--xy_range_max', type=float, default=1.5, help='Maximum value for initial condition range.')
    
    parser.add_argument('--degree', type=int, default=5, help='Polynomial degree for feature library.')
    parser.add_argument('--dim', type=int, default=2, help='Dimension of the system (dim).')
    parser.add_argument('--include_bias', type=bool, default=False, help='Whether to include bias in the feature library.')

    parser.add_argument('--output_dir', type=str, default='ablation_study/acr_results/cyc', help='Base directory for saving output CSV files.')
    parser.add_argument('--mse_slices_steps', nargs='*', default=[5, 10, 20, None], help="Steps for MSE slicing")
    
    return parser.parse_args()

if __name__ == '__main__':
    main()