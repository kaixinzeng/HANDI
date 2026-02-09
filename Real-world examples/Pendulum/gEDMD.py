#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for gEDMD-OLS over multiple dt values.

Assumes data files are named like:
  ./data/duff_train1_Nsim10.npy    # for dt=0.01
  ./data/duff_train2_Nsim10.npy    # for dt=0.02
  ./data/duff_train5_Nsim10.npy    # for dt=0.05
  ./data/duff_train10_Nsim10.npy   # for dt=0.10

Usage:
  python run_gedmd_batch.py --dts 0.01 0.02 0.05 0.1 --polyorder 10 --polyorder_for_sysid 3
"""

import os
import json
import argparse
import numpy as np

from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference


# ----------------------------- I/O helpers -----------------------------
def _ensure_dir(p: str) -> None:
    if p and (not os.path.isdir(p)):
        os.makedirs(p, exist_ok=True)


# ------------------------- Feature name utilities ----------------------
def _names_from_powers(powers: np.ndarray) -> list:
    Dp, d = powers.shape
    names = []
    for k in range(Dp):
        exps = powers[k]
        terms = []
        for j in range(d):
            p = int(exps[j])
            if p <= 0:
                continue
            for _ in range(p):
                terms.append(f"x({j+1})")
        names.append("1" if len(terms) == 0 else "*".join(terms))
    return names


def _find_linear_feature_indices(powers: np.ndarray) -> list:
    Dp, d = powers.shape
    idx_x = [-1] * d
    for k in range(Dp):
        exp = powers[k]
        if int(exp.sum()) != 1:
            continue
        ones = np.where(exp == 1)[0]
        if ones.size == 1:
            i = int(ones[0])
            idx_x[i] = k
    if any(i < 0 for i in idx_x):
        missing = [str(i + 1) for i, v in enumerate(idx_x) if v < 0]
        raise RuntimeError(
            f"Polynomial library is missing linear terms for x({', '.join(missing)}). "
            f"Please ensure degree>=1."
        )
    return idx_x


# --------------------------- Output formatting --------------------------
def format_equations(C: np.ndarray, names: list, state_dim: int, topk=None, thresh: float = 0.0) -> list:
    lines = []
    Dp = min(len(names), C.shape[0])
    for j in range(state_dim):
        coef = C[:Dp, j]
        idx = np.arange(Dp)
        if thresh is not None and thresh > 0:
            idx = idx[np.abs(coef[idx]) >= float(thresh)]
        if topk is not None and topk > 0 and len(idx) > topk:
            take = np.argsort(-np.abs(coef[idx]))[:topk]
            idx = idx[take]
        parts = [f"{coef[i]:+.5f}*{names[i]}" for i in idx]
        rhs = " ".join(parts) if parts else "0"
        lines.append(f"dx({j+1})/dt = {rhs}")
    return lines


def compute_nrmse_padded(C_hat: np.ndarray, C_true: np.ndarray, eps: float = 1e-12) -> float:
    rh, ch = C_hat.shape
    rt, ct = C_true.shape
    R = max(rh, rt)
    Cc = max(ch, ct)
    A = np.zeros((R, Cc), dtype=float)
    B = np.zeros((R, Cc), dtype=float)
    A[:rh, :ch] = C_hat
    B[:rt, :ct] = C_true
    rmse = np.sqrt(np.mean((A - B) ** 2))
    mask = (np.abs(B) > 0)
    if not np.any(mask):
        return float('inf')
    denom = np.mean(np.abs(B[mask]))
    return float(rmse / max(eps, denom))


# ------------------------------ Core gEDMD ------------------------------
def fit_gedmd_ols(X_list, dt: float, polyorder: int, polyorder_for_sysid: int) -> tuple:
    if len(X_list) == 0:
        raise ValueError("X_list is empty")
    if polyorder_for_sysid > polyorder:
        raise ValueError("polyorder_for_sysid must be <= polyorder")

    X_all = np.vstack([np.asarray(Xi, dtype=float) for Xi in X_list])
    poly_lib = PolynomialLibrary(degree=int(polyorder), include_bias=False)
    poly_lib.fit(X_all)

    powers = np.asarray(poly_lib.powers_, dtype=int)
    full_names = _names_from_powers(powers)
    idx_x = _find_linear_feature_indices(powers)

    diff = FiniteDifference(axis=-2)

    Phi_list = []
    Phi_dot_list = []

    for Xi in X_list:
        Xi = np.asarray(Xi, dtype=float)
        T = Xi.shape[0]
        t = np.arange(T, dtype=float) * float(dt)

        Phi = np.asarray(poly_lib.transform(Xi), dtype=float)
        Phi_dot = np.asarray(diff(Phi, t=t), dtype=float)

        Phi_list.append(Phi)
        Phi_dot_list.append(Phi_dot)

    Phi_all = np.asarray(np.vstack(Phi_list), dtype=float)
    Phi_dot_all = np.asarray(np.vstack(Phi_dot_list), dtype=float)

    L = np.linalg.lstsq(Phi_all, Phi_dot_all, rcond=None)[0]
    L = np.asarray(L, dtype=float)

    d = X_list[0].shape[1]
    Dp_full = powers.shape[0]
    C_full = np.zeros((Dp_full, d), dtype=float)
    for i in range(d):
        C_full[:, i] = L[:, idx_x[i]]

    total_degrees = powers.sum(axis=1)
    keep_mask = total_degrees <= polyorder_for_sysid
    keep_indices = np.where(keep_mask)[0]

    C_base = C_full[keep_indices, :]
    names = [full_names[i] for i in keep_indices]

    return C_base, names


# ------------------------------ Main driver -----------------------------
def run_single_dt(dt: float, data_pattern: str, out_base: str, polyorder: int, polyorder_for_sysid: int, true_coeff_json: str):
    # Build data path
    file_id = int(round(dt * 100))  # e.g., dt=0.01 → 1, dt=0.1 → 10
    data_path = data_pattern.format(file_id)
    if not os.path.exists(data_path):
        print(f"[SKIP] Data file not found: {data_path}")
        return

    outdir = os.path.join(out_base, f"gedmd_dt{dt:.3f}".rstrip('0').rstrip('.'))
    _ensure_dir(outdir)

    # Load data
    X = np.load(data_path)
    if X.ndim == 2:
        X_list = [X]
    elif X.ndim == 3:
        X_list = [X[i] for i in range(X.shape[0])]
    else:
        raise ValueError(f"Unsupported data shape {X.shape}")

    # Parse true coefficients (optional)
    C_true = None
    if true_coeff_json:
        try:
            C_true = np.asarray(json.loads(true_coeff_json), dtype=float)
        except Exception as e:
            print(f"[WARN] load true_coeff_json error: {e}")

    # Fit model
    C_hat, names = fit_gedmd_ols(X_list, dt=dt, polyorder=polyorder, polyorder_for_sysid=polyorder_for_sysid)

    coeff_nrmse = None
    if C_true is not None:
        coeff_nrmse = compute_nrmse_padded(C_hat, C_true)

    # Save outputs
    np.save(os.path.join(outdir, "coeff_hat.npy"), C_hat)

    with open(os.path.join(outdir, "feature_names.txt"), "w", encoding="utf-8") as f:
        for nm in names:
            f.write(nm + "\n")

    if coeff_nrmse is not None:
        with open(os.path.join(outdir, "nrmse.txt"), "w", encoding="utf-8") as f:
            f.write(f"{coeff_nrmse:.6g}\n")

    final_eq_lines = format_equations(C_hat, names, state_dim=C_hat.shape[1])
    with open(os.path.join(outdir, "equations.txt"), "w", encoding="utf-8") as f:
        for line in final_eq_lines:
            f.write(line + "\n")

    params_to_dump = {
        "dt": float(dt),
        "polyorder": int(polyorder),
        "polyorder_for_sysid": int(polyorder_for_sysid),
        "tuned": False,
        "best_coeff_nrmse": coeff_nrmse,
        "best_rollout_mse": None,
    }
    with open(os.path.join(outdir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params_to_dump, f, indent=2, ensure_ascii=False)

    # Also save "best_*" files for compatibility
    np.save(os.path.join(outdir, "best_coeff.npy"), C_hat)
    with open(os.path.join(outdir, "best_feature_names.txt"), "w", encoding="utf-8") as f:
        for nm in names:
            f.write(nm + "\n")
    with open(os.path.join(outdir, "best_equations.txt"), "w", encoding="utf-8") as f:
        for line in final_eq_lines:
            f.write(line + "\n")
    with open(os.path.join(outdir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({"best": {"coeff_nrmse": coeff_nrmse}}, f, indent=2, ensure_ascii=False)

    print(f"[DONE] dt={dt} → {outdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], type=float, help="List of dt values to run, e.g., --dt 0.01 0.02 0.05 0.1")
    parser.add_argument("--data_pattern", type=str, default="fixed_point/data_num30_traj1000_[-1,1]/fixed_point_{}.npy", help="Data filename pattern with {} for dt*100")
    parser.add_argument("--out_base", type=str, default="fixed_point/results/gedmd", help="Base output directory")
    parser.add_argument("--polyorder", type=int, default=10)
    parser.add_argument("--polyorder_for_sysid", type=int, default=7)
    parser.add_argument("--true_coeff_json", type=str, default="[[ 0, 3], [ -3, 0], [ 0, 0], [ 0, 0], [ 0, 0], [ -1, 0], [ 0, -1], [ -1, 0], [ 0, -1]]")

    args = parser.parse_args()

    print(f"Running gEDMD-OLS for dt values: {args.dt}")
    for dt in args.dt:
        run_single_dt(
            dt=dt,
            data_pattern=args.data_pattern,
            out_base=args.out_base,
            polyorder=args.polyorder,
            polyorder_for_sysid=args.polyorder_for_sysid,
            true_coeff_json=args.true_coeff_json,
        )

    print("\n✅ All dt runs completed.")


if __name__ == "__main__":
    main()