import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import solve_ivp
from compute_nrmse_names import *

def main():

    args = parse_arguments()
    DT = args.dt
    Original_DT = args.original_dt
    TRUE_DATA_FILE = args.true_data_file
    filename_mse = [
        args.path_edmd.format(dt=f"{DT}"),
        args.path_sindy.format(dt=f"{DT}"), 
        args.path_handi.format(dt=f"{DT}"),
        args.path_pse.format(dt=f"{DT}"),
        args.path_gedmd.format(dt=f"{DT}"),
        args.path_wsindy.format(dt=f"{DT}"),
        args.path_sr3.format(dt=f"{DT}")
    ]
    mse_slice_steps = args.mse_steps

    try:
        all_true_data = np.load(TRUE_DATA_FILE, allow_pickle=True)
    except FileNotFoundError:
        print(f"[Error] Ground truth file not found: {TRUE_DATA_FILE}")
        return

    if all_true_data.ndim == 2:
        all_true_data = all_true_data[None,...]

    num_traj, T, dim = all_true_data.shape
    print(f"\nLoaded {num_traj} trajectories, T={T}, dim={dim}, dt={DT}")

    T_total = (T-1) * DT 
    t_eval_original = np.linspace(0, T_total, int(T_total/Original_DT) + 1)
    Original_T = len(t_eval_original)

    downsample_factor = int(round(DT/ Original_DT))

    initial_conditions = all_true_data[:,0,:]

    method_names = ['EDMD', 'SINDy', 'HANDI','PSE', 'gEDMD', 'wSINDy', 'SR3']
    mse_slices_total = {s:{m:[] for m in method_names} for s in mse_slice_steps}
    r2_total_list = {m:[] for m in method_names}

    # for idx0,file in enumerate(filename_mse):
    #     method_name = method_names[idx0]
    #     if method_name=='PSE':
    #         new_filename = file.replace('.txt', '_new.txt')
    #         change_equations(file, new_filename)
    #         filename_mse[idx0] = new_filename
    #     if method_name == 'wSINDy':
    #         new_wsindy_file = replace_dot_to_mul(file)
    #         filename_mse[idx0] = new_wsindy_file
    #     if method_name == 'SR3':
    #         new_sr3_file = replace_dot_to_mul(file)
    #         filename_mse[idx0] = new_sr3_file
    #     if method_name == 'gEDMD':
    #         new_gedmd_file = replace_dot_to_mul(file)
    #         filename_mse[idx0] = new_gedmd_file

    all_true = []
    mse_all_points = {name: [] for name in method_names}
    nppi_normal = {name: [] for name in method_names}
    mse_pool_for_global_max = []
    all_preds_dict = {m:[] for m in method_names}
    all_invalid_counts = {name: 0 for name in method_names}

    for n in tqdm(range(num_traj)):

        x_true = all_true_data[n]
        x0 = initial_conditions[n]
        traj_preds=[]
        valid=True

        for i,f in enumerate(filename_mse):
            eq=parse_equations(f)
            if not eq:
                valid=False
                traj_preds.append(None)
                continue

            model=create_learned_model(eq)

            sol = solve_ivp(lambda t, x: model(x, t),[t_eval_original[0], t_eval_original[-1]],x0,t_eval=t_eval_original,method='Radau',rtol=1e-4,atol=1e-6)  #LSODA RK45 Radau
            
            pred = sol.y.T
            downsample_indices = np.arange(0, pred.shape[0], downsample_factor)
            downsampled_pred = pred[downsample_indices]
            traj_preds.append(downsampled_pred)

            if downsampled_pred.shape[0]!=T:
                valid=False
                all_invalid_counts[method_names[i]] += 1

        if valid:
            all_true.append(x_true)

            for steps in mse_slice_steps:
                N=T if steps is None else min(steps,T)
                true_slice=x_true[:N]

                for idx,x_pred in enumerate(traj_preds):
                    name=method_names[idx]
                    pred_slice=x_pred[:N]
                    mse=np.mean((true_slice-pred_slice)**2)
                    mse_slices_total[steps][name].append(mse)

                    if steps==mse_slice_steps[0]:
                        all_preds_dict[name].append(x_pred)
        
        for idx,x_pred in enumerate(traj_preds):
            name = method_names[idx]
            if not valid and (x_pred.shape[0]!=T):
                mse= np.inf
                mse_all_points[name].append(mse)
            else:
                x_pred_dp = x_pred
                x_true_dp = x_true
                mse= np.mean((x_true_dp - x_pred_dp)**2)
                mse_pool_for_global_max.append(mse)
                mse_all_points[name].append(mse)

    global_mse_max = max(mse_pool_for_global_max)
    for name in method_names:
        for mse in mse_all_points[name]:
            if mse == np.inf:
                nppi_normal[name].append(0)
            else:
                nppi_normal[name].append(max(0, 1 - mse/global_mse_max))

    if len(all_true) != 0:
        all_true_cat=np.vstack(all_true)
        for name,preds in all_preds_dict.items():
            p=np.vstack(preds)
            r2 = compute_r2(all_true_cat,p) if p.shape==all_true_cat.shape else np.nan
            r2_total_list[name]=r2

    os.makedirs(args.output_dir,exist_ok=True)

    for steps,mse_data in mse_slices_total.items():
        steps_label = f'{steps}' if steps is not None else 'full'

        df_avg = pd.DataFrame({
            "Method":method_names,
            "MSE_avg":[np.mean(mse_data[m]) for m in method_names]
        })
        df_avg.to_csv(f"{args.output_dir}/{args.system}_average_mse_dt{DT}_steps_{steps_label}.csv",index=False, float_format='%.8e')

        df_for_boxplot = pd.DataFrame({name: mse_data[name] for name in method_names})
        df_for_boxplot.to_csv(f"{args.output_dir}/{args.system}_mse_dt{DT}_steps_{steps_label}.csv",index=False, float_format='%.8e')
    
    summary_df_valid = pd.DataFrame({
        'Method': method_names,
        'Valid_rate': [(1-all_invalid_counts[name]/num_traj) for name in method_names]
    })
    summary_df_r2 = pd.DataFrame({
        'Method': method_names,
        'r2_Average': [r2_total_list[name] for name in method_names]
    })
    summary_csv_r2 = f'{args.output_dir}/{args.system}_r2_dt{DT}.csv'
    summary_df_r2.to_csv(summary_csv_r2, index=False, float_format='%.8e')
    # print(f"Summary r2 saved to {summary_csv_r2}")

    summary_csv_valid = f'{args.output_dir}/{args.system}_valid_rate_dt{DT}.csv'
    summary_df_valid.to_csv(summary_csv_valid, index=False, float_format='%.8e')
    # print(f"Summary Valid Rate saved to {summary_csv_valid}")

    average_df_nppi = pd.DataFrame({
        "Method": method_names,
        "NPPI_Average":[np.mean(nppi_normal[name]) for name in method_names],
    })
    average_csv_nppi = f"{args.output_dir}/{args.system}_average_nppi_dt{DT}.csv"
    average_df_nppi.to_csv(average_csv_nppi, index=False, float_format='%.8e')
    # print(f"Average NPPI saved to {average_csv_nppi}")

    df_nppi = pd.DataFrame({name: nppi_normal[name] for name in method_names})
    csv_nppi = f"{args.output_dir}/{args.system}_nppi_dt{DT}.csv"
    df_nppi.to_csv(csv_nppi, index=False, float_format='%.8e')
    # print(f"NPPI saved to {csv_nppi}")

    mse_all_points = pd.DataFrame({name: mse_all_points[name] for name in method_names})
    csv_mse_all_points = f"{args.output_dir}/{args.system}_allmse_dt{DT}.csv"
    mse_all_points.to_csv(csv_mse_all_points, index=False, float_format='%.8e')
    # print(f"MSE including inf saved to {csv_mse_all_points}")

def parse_arguments():
    parser=argparse.ArgumentParser()

    parser.add_argument("--true_data_file",type=str, default="flag/4d_dt0.1.npy")
    parser.add_argument("--system", type=str, default="flag", help="System name for output csv, e.g., NI")
    parser.add_argument("--dt", type=float,default=0.1, help="Time step of the trained data")
    parser.add_argument("--original_dt", type=float, default=1/30, help="Time step of the original data")
    parser.add_argument("--path_edmd", type=str, default="flag/results/EDMD/dt{dt}_equations.txt")
    parser.add_argument("--path_sindy", type=str,default="flag/results/SINDy/dt{dt}_equations.txt")
    parser.add_argument("--path_handi", type=str,default="flag/results/HANDI/dt{dt}_equations.txt")
    parser.add_argument("--path_pse", type=str,default="flag/results/PSE/dt{dt}_equations.txt")
    parser.add_argument("--path_gedmd", type=str,default="flag/results/gEDMD/dt{dt}_equations.txt")
    parser.add_argument("--path_wsindy", type=str,default="flag/results/WSINDy/dt{dt}_equations.txt")
    parser.add_argument("--path_sr3", type=str,default="flag/results/SR3/dt{dt}_equations.txt")
    parser.add_argument("--mse_steps", nargs='*',default=[None], help="Steps for MSE slicing, None means full length")
    parser.add_argument("--output_dir", type=str,default="nrmse_mse_r2/4d/results/dt0.1")

    return parser.parse_args()


if __name__=="__main__":
    main()
