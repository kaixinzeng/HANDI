from scipy.stats import mannwhitneyu, ttest_ind
import numpy as np
import pandas as pd
import os
data=pd.read_csv('ablation_study/mse_results/fixed/fixed_ablation_mse_all_dt0.5_xyrange1_grid10.csv')
# data=pd.read_csv('ablation_study/mse_results/cyc/cyc_ablation_mse_all_dt0.5_xyrange1.5_grid10.csv')
# data=pd.read_csv('ablation_study/mse_results/vdp/vdp_ablation_mse_all_dt0.4_xyrange3_grid10.csv')
# data=pd.read_csv('ablation_study/mse_results/duffing/duffing_ablation_mse_all_dt0.3_xyrange4_grid10.csv')
# data = data.replace(np.inf, np.nan).dropna(how='any')

full = data['HANDI'].values  # ★
A = data['ablationA'].values            # ①
O = data['ortho'].values            # ②

def compute_pvalue(group1, group2, method='mannwhitneyu'):
    if method == 'mannwhitneyu':
        stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
    elif method == 'ttest':
        stat, p = ttest_ind(group1, group2)
    return p

p_full_A = compute_pvalue(full, A)
p_full_O = compute_pvalue(full, O)
p_A_O = compute_pvalue(A, O)

print(f"Full vs A: p = {p_full_A:.2e}")
print(f"Full vs O: p = {p_full_O:.2e}")
print(f"A vs O: p = {p_A_O:.2e}")