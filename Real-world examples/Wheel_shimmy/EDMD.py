import numpy as np
import scipy.linalg
import time
import os
import re
import math

path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)

def comb_with_rep(N, K):
    if K == 0:
        return [()]
    if N == 0:
        return []
    
    combs = []
    for i in range(N):
        sub_combs = comb_with_rep(N - i, K - 1)
        for sub_comb in sub_combs:
            adjusted_comb = tuple([i] + [c + i for c in sub_comb])
            combs.append(adjusted_comb)
    return combs

def build_basis(data, dataname, polyorder):
    num_samples, num_features = data.shape
    total_basis_count = math.comb(polyorder + num_features, num_features) - 1
    
    if total_basis_count <= 0:
        basis_matrix = np.empty((num_samples, 0), dtype=data.dtype)
        basis_name = []
        return basis_matrix, basis_name

    basis_matrix = np.empty((num_samples, total_basis_count), dtype=data.dtype)
    basis_name = [None] * total_basis_count # Pre-allocate list for names

    current_index = 0 # Index to track where to place the next basis function

    for degree in range(1, polyorder + 1):
        combinations_for_degree = comb_with_rep(num_features, degree)
        
        for comb in combinations_for_degree:
            if current_index >= total_basis_count:
                 print(f"Warning: Generated more basis functions than calculated. Stopping at index {current_index}.")
                 break
                 
            term_data = np.ones(num_samples, dtype=data.dtype) # Initialize as ones
            term_name_parts = []
            for idx in comb:
                term_data *= data[:, idx]
                term_name_parts.append(dataname[idx])
            term_name_parts.sort() # Sort names for consistent naming
            term_name = '*'.join(term_name_parts)
            
            # Assign to pre-allocated arrays
            basis_matrix[:, current_index] = term_data
            basis_name[current_index] = term_name
            
            current_index += 1
            
    if current_index < total_basis_count:
        print(f"Warning: Expected {total_basis_count} basis functions, generated {current_index}. Truncating arrays.")
        basis_matrix = basis_matrix[:, :current_index]
        basis_name = basis_name[:current_index]

    return basis_matrix, basis_name

def identify_system(X, weights_true, deltaT, h, polyorder=10, Nbasis_for_sysid = 9, save_dir="./"):
    print(f'Starting sampling, period = {h * deltaT:.2f} s')
    
    n_traj, n_time_steps, n_features = X.shape
    
    print(f"Data shape: {X.shape}")

    # Prepare data matrices Xtmp (states) and Ytmp (next states)
    Xtmp = X[:, :-1, :]  # Shape: (n_traj, n_time_steps-1, n_features)
    Ytmp = X[:, 1:, :]   # Shape: (n_traj, n_time_steps-1, n_features)

    # Reshape into 2D matrices (samples, features)
    Xtmp_reshaped = Xtmp.reshape(-1, n_features) # Shape: (n_samples, n_features)
    Ytmp_reshaped = Ytmp.reshape(-1, n_features) # Shape: (n_samples, n_features)
    
    state_name = [f'x({j})' for j in range(1, n_features + 1)]
    print(f"State names: {state_name}")

    print("Building basis functions...")
    start_time = time.time()
    Xlift, xbasis_name = build_basis(Xtmp_reshaped, state_name, polyorder)
    Ylift, _ = build_basis(Ytmp_reshaped, state_name, polyorder)
    print(f"Basis built in {time.time() - start_time:.2f} seconds.")
    
    N_basis = Xlift.shape[1] 
    print(f"Number of basis functions: {N_basis}")

    print("Starting regression...")
    start_time = time.time()
    
    try:
        U = np.dot(np.linalg.pinv(Xlift), Ylift)
    except np.linalg.LinAlgError:
        print(f"SVD did not converge for h={h}.")
        return None, None, None, None, None
        
    try:
        log_U = scipy.linalg.logm(U) 
        L = log_U / (h * deltaT)
        L = L.real
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Matrix logarithm/logm failed for h={h}: {e}")
        return None, None, None, None, None
        
    print(f"Regression done, time = {time.time() - start_time:.2f} s")
    
    os.makedirs(save_dir, exist_ok=True)
    L_filename = os.path.join(save_dir, f"EDMD_Lhat_poly_{polyorder}_dt_{deltaT*h:.2f}.npy")
    np.save(L_filename, L)
    print(f"Saved L matrix to {L_filename}")

    delta = 1e-8
    w_estimate = L[:Nbasis_for_sysid,:n_features] # L is the estimated weights matrix (N_basis, n_features)
    
    equations = [""] * n_features
    for i in range(n_features):
        y_name = f'dx({i+1})/dt' 
        equations[i] += f'{y_name} = '
        terms_printed = 0
        for j in range(Nbasis_for_sysid):
            coeff_magnitude_sq = w_estimate[j, i] ** 2
            coeff_norm_sq = np.linalg.norm(w_estimate[:, i]) ** 2
            
            if coeff_norm_sq > 0 and (coeff_magnitude_sq / coeff_norm_sq) >= delta:
                coeff = w_estimate[j, i]
                basis_term = xbasis_name[j]
                if coeff >= 0 and terms_printed > 0:
                     equations[i] += '+'
                equations[i] += f'{coeff:.4f}*{basis_term}'
                terms_printed += 1
        if terms_printed == 0:
            equations[i] += "0"
    
    eq_filename = os.path.join(save_dir, f"EDMD_equation_poly_{polyorder}_polysysid_{polyorder_sysid}_dt_{deltaT*h:.2f}.txt")
    with open(eq_filename, 'w') as f:
        f.write("\n".join(equations))
    print(f"Saved equations to {eq_filename}")

    print("\nEstimated dynamics:")
    for eq in equations:
        print(eq)

    if weights_true.shape != (Nbasis_for_sysid, n_features):
         print(f"Warning: weights_true shape {weights_true.shape} does not match expected shape ({N_basis}, {n_features}). Skipping error calculation.")
         mea = np.nan
         nrmse = np.nan
    else:
        # Count non-zero elements in weights_true for normalization
        Nw_nonzero = np.count_nonzero(weights_true)
        if Nw_nonzero == 0:
            print("Warning: weights_true is all zeros. Skipping error calculation.")
            mea = np.nan
            nrmse = np.nan
        else:
            err = np.abs(w_estimate - weights_true)
            mea = np.mean(err)
            
            sum_sq_err = np.sum(err**2)
            sum_abs_true = np.sum(np.abs(weights_true))
            
            if sum_abs_true > 0:
                nrmse = np.sqrt(sum_sq_err / (N_basis * n_features)) * Nw_nonzero / sum_abs_true
            else:
                nrmse = np.nan
            
            print(f'\nCalculated error for polyorder = {polyorder} : mea = {mea:.4f}, NRMSE = {nrmse:.4f}')
    
    return L, mea, nrmse, eq_filename, L_filename


def run_edmd(weights_true, deltaT=0.01, 
             T_sample=np.array([10, 30, 50, 70, 90, 110, 200]), 
             polyorder=10, Nbasis_for_sysid = 9, save_dir="./edmd_results_classic",
             data_filename_pattern="duff_train{}_Nsim2.npy"):
    # Ensure save_dir is relative to the new working directory or absolute
    os.makedirs(save_dir, exist_ok=True) 

    mea_list = []
    nrmse_list = []
    l_filenames = []
    eq_filenames = []
    successful_h_values = [] # Stores the 'h' values (elements from T_sample) that were successful

    for h_idx, h in enumerate(T_sample):
        print(f"\n--- Processing T_sample index {h_idx}, h = {h} ---")
        
        # Construct the data filename using the pattern
        data_filename = data_filename_pattern.format(h)
        try:
            X = np.load(data_filename) 
            print(f"Loaded data from {data_filename} with shape {X.shape}")
        except FileNotFoundError:
            print(f"Data file '{data_filename}' not found. Skipping h={h}.")
            continue 
            
        # Pass weights_true to the identification function
        L, mea, nrmse, eq_file, L_file = identify_system(
            X, weights_true, deltaT, h, polyorder, Nbasis_for_sysid, save_dir
        )
        
        if L is not None or not (np.isnan(mea) and np.isnan(nrmse) and eq_file is None and L_file is None):
             mea_list.append(mea)
             nrmse_list.append(nrmse)
             l_filenames.append(L_file)
             eq_filenames.append(eq_file)
             successful_h_values.append(h*deltaT) # Store the T_sample index 'h' that succeeded

    if successful_h_values:
        t_sample_array = np.array(successful_h_values)
        nrmse_array = np.array(nrmse_list)
        
        # Stack them as columns: [T_sample_values, NRMSE_values]
        nrmse_data_to_save = np.column_stack((t_sample_array, nrmse_array))
        
        nrmse_filename = os.path.join(save_dir, f"EDMD_NRMSE_poly_{polyorder}_polysysid_{polyorder_sysid}.npy")
        np.save(nrmse_filename, nrmse_data_to_save) # Save the 2-column array
        print(f"\nSaved aggregated NRMSE data (for {len(successful_h_values)} runs) to {nrmse_filename}")
        print(f"  Data shape: {nrmse_data_to_save.shape} (Columns: T_sample_index, NRMSE)")
    else:
        print("\nNo successful runs completed to save NRMSE data.")

    print("\n--- EDMD Process Completed ---")
    return mea_list, nrmse_list, eq_filenames, l_filenames, successful_h_values 

if __name__ == "__main__":
    lambda_val = 0.5
    alpha = -5
    beta = 1
    n_features_example = 2 # Dimension of the state space
    polyorder_example = 10  # Polynomial order for basis
    polyorder_sysid = 7 

    dummy_data_for_basis = np.zeros((1, n_features_example)) # Shape doesn't matter much, just need n_features
    dummy_names = [f'x({j})' for j in range(1, n_features_example + 1)]
    _, xbasis_name_example = build_basis(dummy_data_for_basis, dummy_names, polyorder_sysid)
    N_basis_example = len(xbasis_name_example)
    print(f"\n[Main Setup] Calculated N_basis for example system: {N_basis_example}")
    print(f"[Main Setup] Basis names (first 10): {xbasis_name_example[:10]}")

    # Initialize the true weights vector
    weights_true_example = np.zeros((N_basis_example, n_features_example))
    
    # Populate weights_true based on the known Duffing dynamics and basis names
    try:

        idx_x2_for_dx1 = xbasis_name_example.index('x(2)')
        idx_x1_cubed_for_dx1 = xbasis_name_example.index('x(1)*x(1)*x(1)')
        idx_x1x2_for_dx1 = xbasis_name_example.index('x(1)*x(2)*x(2)')

        weights_true_example[idx_x2_for_dx1, 0] = -3.0
        weights_true_example[idx_x1_cubed_for_dx1, 0] = -1
        weights_true_example[idx_x1x2_for_dx1, 0] = -1

        idx_x1_for_dx2 = xbasis_name_example.index('x(1)')
        idx_x2_cubed_for_dx2 = xbasis_name_example.index('x(2)*x(2)*x(2)')
        idx_x1x2_for_dx2 = xbasis_name_example.index('x(1)*x(1)*x(2)')

        weights_true_example[idx_x1_for_dx2, 1] = 3
        weights_true_example[idx_x2_cubed_for_dx2, 1] = -1
        weights_true_example[idx_x1x2_for_dx2, 1] = -1
        
        print(f"[Main Setup] Populated weights_true_example for Duffing system.")
    except ValueError as e:
        print(f"[Main Setup] Error populating weights_true_example: {e}")
        print("[Main Setup] Setting weights_true_example to zeros.")
        weights_true_example = np.zeros((N_basis_example, n_features_example)) # Fallback

    # --- Define other parameters ---
    default_params = {
        'weights_true': weights_true_example, # Pass the true weights vector
        'deltaT': 0.01,
        'T_sample': np.array([10,20,30,40,50,60,70,80,90,100,110]),
        'polyorder': polyorder_example,
        'Nbasis_for_sysid': N_basis_example,
        'save_dir': './edmd_results_classic_10_7_num30_traj1000', # Updated save directory
        'data_filename_pattern': 'data_num30_traj1000_[-1,1]/fixed_point_{}.npy' # NEW: Data filename pattern
    }

    print(f"\n[Main] Starting EDMD with parameters: deltaT={default_params['deltaT']}, polyorder={default_params['polyorder']}")
    print(f"[Main] True weights vector shape: {default_params['weights_true'].shape}")
    print(f"[Main] Data filename pattern: '{default_params['data_filename_pattern']}'")

    # Run the EDMD process
    mea_results, nrmse_results, equation_files, L_files, h_vals_success = run_edmd(**default_params)
    print('done')