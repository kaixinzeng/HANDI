# pip install allensdk h5py numpy
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.core.nwb_data_set import NwbDataSet
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import decimate
from scipy.signal import find_peaks
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import re
import os
path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)

ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
specimen_id = 324257146 
data_set = ctc.get_ephys_data(specimen_id)
sweeps = ctc.get_ephys_sweeps(specimen_id)

ls_numbers = [s['sweep_number'] for s in sweeps if s['stimulus_name'] == 'Long Square']
assert len(ls_numbers) > 0, "This cell has no Long Square sweeps, try another specimen_id"

X_list = []

for sw in ls_numbers[:2]:
    sweep = data_set.get_sweep(sw)
    stim = sweep['stimulus']
    resp = sweep['response']
    sr   = sweep['sampling_rate']

    t     = np.arange(len(stim)) / sr
    t0, t1 = 1.02, 1.62
    i0, i1 = int(t0*sr), int(t1*sr)

    I_const = np.mean(stim[i0:i1])
    V_seg   = resp[i0:i1]
    dt      = 1.0 / sr

    window_length = int(0.005 / dt)
    if window_length % 2 == 0:
        window_length += 1
    polyorder = 3

    V_s   = savgol_filter(V_seg, window_length, polyorder, deriv=0, delta=dt, mode='interp')
    Vdot  = savgol_filter(V_seg, window_length, polyorder, deriv=1, delta=dt, mode='interp')

    W_rec = V_s - (V_s**3)/3.0 - Vdot + I_const
    W_s   = savgol_filter(W_rec, 51, 3, deriv=0, delta=dt, mode='interp')

    I_s = np.full_like(V_s, I_const)
    X_traj = np.column_stack([V_s, W_s, I_s])
    X_list.append(X_traj)

X_all = np.vstack(X_list)
mean_all = X_all.mean(axis=0)
std_all  = X_all.std(axis=0) + 1e-12

X_norm_list = []
for X_traj in X_list:
    X_norm = (X_traj - mean_all) / std_all
    X_norm_list.append(X_norm)

target_fs = 100
q = int(sr / target_fs)  # sr = 20K hz

X_down_list = []
for X_norm in X_norm_list:
    X_down = X_norm[::q]
    X_down_list.append(X_down)

X_process_save = np.stack(X_down_list, axis=0)
X_true_save = np.stack(X_list, axis=0)
np.save("NWB_true_traj2_process.npy", X_true_save)
np.save("NWB_traj2_states=3_t=1.6_zscore_num60.npy", X_process_save)
np.save("NWB_traj2_states=3_t=1.6_zscore_mean", mean_all)
np.save("NWB_traj2_states=3_t=1.6_zscore_std", std_all)

