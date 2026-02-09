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
    ht = args.ht
    grid_size = args.grid_size
    dim = args.dim
    degree = args.degree
    t_plot = args.t_plot
    output_dir = args.output_dir
    xy_range_min = args.xy_range_min
    xy_range_max = args.xy_range_max
    mse_slice_steps = args.mse_slices_steps
    include_bias = args.include_bias
    
    scale = int(round(dt / ht))
    t = np.linspace(0, t_plot * scale, t_plot * scale * 100)

    x_range = np.linspace(xy_range_min, xy_range_max, grid_size)
    y_range = np.linspace(xy_range_min, xy_range_max, grid_size)
    X1, X2 = np.meshgrid(x_range, y_range)
    initial_conditions = np.vstack([X1.ravel(), X2.ravel()]).T
    num_traj = len(initial_conditions)

    filename_dt = dt 
    filename_nrmse = [
        args.path_edmd.format(dt=f"{filename_dt:.2f}"),
        args.path_sindy.format(dt=f"{filename_dt:.2f}"), 
        args.path_handi.format(dt=f"{filename_dt:.3f}"),
        args.path_pse.format(dt=f"{filename_dt:.2f}"),
        args.path_gedmd.format(dt=f"{filename_dt:.1f}"),
        # args.path_wsindy.format(dt=f"{filename_dt * 10 :.0f}"),
        args.path_sr3.format(dt=f"{filename_dt * 10 :.0f}")
    ]

    filename_mse = list(filename_nrmse)

    # method_names_nrmse = ['EDMD', 'SINDy', 'HANDI','PSE','gEDMD', 'wSINDy', 'SR3']
    # method_names_mse = ['EDMD', 'SINDy', 'HANDI','PSE','gEDMD', 'wSINDy', 'SR3']
    method_names_nrmse = ['EDMD','SINDy', 'HANDI', 'PSE','gEDMD', 'SR3']
    method_names_mse = ['EDMD','SINDy', 'HANDI', 'PSE','gEDMD', 'SR3']
    mse_slices_total = {steps: {name: [] for name in method_names_mse} for steps in mse_slice_steps}
    nrmse_total_list = {name: np.nan for name in method_names_nrmse}
    final_r2_results = {m:[] for m in method_names_mse}

    library = PolynomialLibrary(degree=degree, include_bias=include_bias)
    library.fit(np.random.rand(10, dim))
    names = library.get_feature_names([f"x{i+1}" for i in range(dim)])

    try:
        true_coeff = build_true_coeff_from_file(true_equation_file, names, dim)

        for idx0, file in enumerate(filename_nrmse):
            method_name = method_names_nrmse[idx0]
            current_file = file
            
            if method_name == 'PSE':
                new_pse_file = file.replace('.txt', f'_new_{dt:.2f}.txt')
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

    all_true_list = []
    mse_all_points = {name: [] for name in method_names_mse}
    nppi_normal = {name: [] for name in method_names_mse}
    mse_pool_for_global_max = []
    all_preds_dict = {name: [] for name in method_names_mse}
    all_invalid_counts = {name: 0 for name in method_names_mse}

    for n, x0 in enumerate(tqdm(initial_conditions)):

        true_solution = odeint(SYSTEMS[args.system], x0, t)
        x1_true = true_solution[:, 0]
        x2_true = true_solution[:, 1]
        x1_true_dp = downsample(dt, x1_true)
        x2_true_dp = downsample(dt, x2_true)
        x_true_dp = np.vstack([x1_true_dp, x2_true_dp]).T

        # --- 2. 预测轨迹 ---
        learned_traj = []
        valid_for_r2 = True
        
        for idx, file in enumerate(filename_mse):
            method_name = method_names_mse[idx]
            equations_list = parse_and_prepare_equations(file, dim)

            if equations_list and len(equations_list) == dim:
                learned_model = create_learned_model(equations_list)
                
                sol = solve_ivp(learned_model, [t[0], t[-1]], x0, t_eval=t, method='RK45', rtol=1e-6, atol=1e-8)
                
                x1_pred = sol.y[0]
                x2_pred = sol.y[1]
                x1_pred_dp = downsample(dt, x1_pred)
                x2_pred_dp = downsample(dt, x2_pred)
                
                learned_traj.append((x1_pred_dp, x2_pred_dp))

            else:
                all_invalid_counts[method_name] += 1
                valid_for_r2 = False
                learned_traj.append((np.array([]), np.array([]))) 

        for idx, (x1_pred_dp, x2_pred_dp) in enumerate(learned_traj):
            name = method_names_mse[idx]
            if len(x1_pred_dp) > 0 and len(x1_pred_dp) != len(x1_true_dp):
                all_invalid_counts[name] += 1
                valid_for_r2 = False

        # --- MSE ---      
        if valid_for_r2:
            all_true_list.append(x_true_dp)

            for steps in mse_slice_steps:
                if steps is None: 
                    N = len(x_true_dp)
                else: 
                    N = min(steps,len(x_true_dp))

                x_true_slice = x_true_dp[:N,:]

                for idx,(x1_pred,x2_pred) in enumerate(learned_traj):
                    method_name = method_names_mse[idx]
                    x_pred_dp = np.vstack([x1_pred, x2_pred]).T  # (N_full, 2)
                    x_pred_slice = x_pred_dp[:N, :]
                    
                    mse = np.mean((x_true_slice-x_pred_slice)**2)
                    mse_slices_total[steps][method_name].append(mse)

                    if steps==mse_slice_steps[0]:
                        all_preds_dict[method_name].append(x_pred_dp)

        # ----NPPI----
        for idx,(x1_pred,x2_pred) in enumerate(learned_traj):
            name = method_names_mse[idx]
            if not valid_for_r2 and (len(x1_pred)!=len(x1_true_dp)):
                mse= np.inf
                mse_all_points[name].append(mse)
            else:
                x_pred_dp = np.vstack([x1_pred,x2_pred]).T
                x_true_dp_s = x_true_dp
                mse= np.mean((x_true_dp_s - x_pred_dp)**2)
                mse_pool_for_global_max.append(mse)
                mse_all_points[name].append(mse)

    global_mse_max = max(mse_pool_for_global_max)
    for name in method_names_mse:
        for mse in mse_all_points[name]:
            if mse == np.inf:
                nppi_normal[name].append(0)
            else:
                nppi_normal[name].append(max(0, 1 - mse/global_mse_max))

    if len(all_true_list) != 0:
        all_true = np.vstack(all_true_list)
        for method_name in method_names_mse:
            pred_list = all_preds_dict[method_name]
            r2 = np.nan
            if pred_list:
                all_pred = np.vstack(pred_list)
                if all_true.shape == all_pred.shape:
                    r2 = compute_r2(all_true, all_pred)

            final_r2_results[method_name] = r2

    os.makedirs(output_dir, exist_ok=True)
    output_dt_label = dt
    
    for steps, mse_data in mse_slices_total.items():
        steps_label = f'{steps}' if steps is not None else 'full'
        avg_mse_data = {
            'Method': method_names_mse,
            'MSE_Average': [np.mean(mse_data[name]) for name in method_names_mse]
        }
        summary_df_mse_slice = pd.DataFrame(avg_mse_data)
        summary_csv_mse_slice = f'{output_dir}/{system}_average_mse_dt{output_dt_label}_steps_{steps_label}_xyrange{xy_range_max}_grid{grid_size}.csv'
        summary_df_mse_slice.to_csv(summary_csv_mse_slice, index=False, float_format='%.8e')
        print(f"Summary MSE for {steps_label} steps saved to {summary_csv_mse_slice}")
        
        df_for_boxplot = pd.DataFrame({name: mse_data[name] for name in method_names_mse})
        boxplot_data_csv = f'{output_dir}/{system}_mse_dt{output_dt_label}_steps_{steps_label}_xyrange{xy_range_max}_grid{grid_size}.csv'
        df_for_boxplot.to_csv(boxplot_data_csv, index=False, float_format='%.8e')
        print(f"Boxplot MSE data for {steps_label} steps saved to {boxplot_data_csv}")

    summary_df_nrmse = pd.DataFrame({
        'Method': method_names_nrmse,
        "NRMSE": [nrmse_total_list[name] for name in method_names_nrmse]
    })
    summary_csv_nrmse = f'{output_dir}/{system}_nrmse_dt{output_dt_label}_xyrange{xy_range_max}_grid{grid_size}.csv'
    summary_df_nrmse.to_csv(summary_csv_nrmse, index=False, float_format='%.8e')
    print(f"Summary NRMSE saved to {summary_csv_nrmse}")

    summary_df_valid = pd.DataFrame({
        'Method': method_names_mse,
        'valid_rate': [1 - all_invalid_counts[name] / num_traj for name in method_names_mse]
    })
    summary_csv_valid = f'{output_dir}/{system}_valid_rate_dt{output_dt_label}_xyrange{xy_range_max}_grid{grid_size}.csv'
    summary_df_valid.to_csv(summary_csv_valid, index=False, float_format='%.8e')
    print(f"Summary Valid Rate saved to {summary_csv_valid}")

    summary_df_r2 = pd.DataFrame({
        'Method': method_names_mse,
        'r2_Average': [final_r2_results[name] for name in method_names_mse]
    })
    summary_csv_r2 = f'{output_dir}/{system}_r2_dt{output_dt_label}_xyrange{xy_range_max}_grid{grid_size}.csv'
    summary_df_r2.to_csv(summary_csv_r2, index=False, float_format='%.8e')
    print(f"Summary r2 saved to {summary_csv_r2}")
    
    average_df_nppi = pd.DataFrame({
        "Method": method_names_mse,
        "NPPI_Average":[np.mean(nppi_normal[name]) for name in method_names_mse],
    })
    average_csv_nppi = f"{output_dir}/{system}_average_nppi_dt{output_dt_label}_xyrange{xy_range_max}_grid{grid_size}.csv"
    average_df_nppi.to_csv(average_csv_nppi, index=False, float_format='%.8e')
    print(f"Average NPPI saved to {average_csv_nppi}")

    df_nppi = pd.DataFrame({name: nppi_normal[name] for name in method_names_mse})
    csv_nppi = f"{output_dir}/{system}_nppi_dt{output_dt_label}_xyrange{xy_range_max}_grid{grid_size}.csv"
    df_nppi.to_csv(csv_nppi, index=False, float_format='%.8e')
    print(f"NPPI saved to {csv_nppi}")

    mse_all_points = pd.DataFrame({name: mse_all_points[name] for name in method_names_mse})
    csv_mse_all_points = f"{output_dir}/{system}_mse_all_dt{output_dt_label}_xyrange{xy_range_max}_grid{grid_size}.csv"
    mse_all_points.to_csv(csv_mse_all_points, index=False, float_format='%.8e')
    print(f"MSE including inf saved to {csv_mse_all_points}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--system', type=str, default='fixed',choices=['fixed', 'linear', 'vdp','cyc','duffing','m1k16c8'],help='Select system dynamics for ground truth simulation.')
    parser.add_argument("--true_equation", type=str, default="fixed_point/results/true_equation.txt",help="path to true system equation file")
    parser.add_argument("--path_edmd", type=str, default="fixed_point/results/EDMD/EDMD_equation_poly_10_polysysid_7_dt_{dt}.txt")
    parser.add_argument("--path_sindy", type=str,default="fixed_point/results/sindy/best_equation_for_threshod/SINDY_equation_poly_7_dt_{dt}_threshod0.1.txt")
    parser.add_argument("--path_handi", type=str,default="fixed_point/results/HANDI/edmd_runs_0.9_poly7/best_equations_dt{dt}.txt")
    parser.add_argument("--path_pse", type=str,default="fixed_point/results/PSE/PSE_equation_poly_7_dt_{dt}.txt")
    parser.add_argument("--path_gedmd", type=str,default="fixed_point/results/gedmd/gedmd_dt{dt}/best_equations.txt")
    parser.add_argument("--path_wsindy", type=str,default="fixed_point/results/wsindy/fix{dt}/tune_best_equations.txt")
    parser.add_argument("--path_sr3", type=str,default="fixed_point/results/sr3/fix{dt}/tune_best_equations.txt")

    parser.add_argument('--dt', type=float, default=0.9, help='Time step for downsampling/evaluation (dt).')
    parser.add_argument('--ht', type=float, default=0.1, help='Time step for dense integration (ht).')
    parser.add_argument('--t_plot', type=int, default=3, help='Time factor for total integration time (T_max = t_plot * scale).')

    parser.add_argument('--grid_size', type=int, default=10, help='Number of points in each dimension for initial condition grid (grid_size x grid_size).')
    parser.add_argument('--xy_range_min', type=float, default=-1.0, help='Minimum value for initial condition range.')
    parser.add_argument('--xy_range_max', type=float, default=1.0, help='Maximum value for initial condition range.')
    
    parser.add_argument('--degree', type=int, default=7, help='Polynomial degree for feature library.')
    parser.add_argument('--dim', type=int, default=2, help='Dimension of the system (dim).')
    parser.add_argument('--include_bias', type=bool, default=False, help='Whether to include bias in the feature library.')

    parser.add_argument('--output_dir', type=str, default='nrmse_mse_r2/fixed_point_five_methods/fixed0.9', help='Base directory for saving output CSV files.')
    parser.add_argument('--mse_slices_steps', nargs='*', default=[5, 10, 20, None], help="Steps for MSE slicing")
    
    return parser.parse_args()

if __name__ == '__main__':
    main()