import os, json, argparse, warnings
import numpy as np

import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import SR3

try:
    import optuna
except Exception:
    optuna = None
    warnings.warn("Please pip install optuna")

def _ensure_dir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _names_from_powers(powers):
    names = []
    for exp in powers:
        terms = []
        for i, e in enumerate(exp):
            if e > 0:
                terms += [f"x({i+1})"] * int(e)
        if len(terms) == 0:
            names.append("1")
        else:
            names.append("*".join(terms))
    return names

def format_equations(C, names, state_dim, topk=None, thresh=0.0):
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
        parts = []
        for i in idx:
            val_str = f"{coef[i]:+.5f}"
            if coef[i] >= 0 and i == 0 and len(parts)==0:
                 val_str = f"{coef[i]:.5f}" 
            parts.append(f"{val_str}·{names[i]}")
        rhs = " ".join(parts) if parts else "0"
        lines.append(f"dx({j+1})/dt = {rhs}")
    return lines

def compute_nrmse_padded(C_hat, C_true, eps=1e-12):
    rh, ch = C_hat.shape
    rt, ct = C_true.shape
    R = max(rh, rt)
    Cc = max(ch, ct)
    A = np.zeros((R, Cc), dtype=float)
    B = np.zeros((R, Cc), dtype=float)
    A[:rh, :ch] = C_hat
    B[:rt, :ct] = C_true
    num = np.linalg.norm(A - B)
    denom = np.linalg.norm(B - B.mean())
    return num / (denom + eps)

def _split_train_val(X_list, val_ratio=0.25, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X_list))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if len(tr_idx) == 0:
        tr_idx = val_idx
    X_tr = [X_list[i] for i in tr_idx]
    X_val = [X_list[i] for i in val_idx]
    return X_tr, X_val

def _downsample_for_tuning(X_list, stride=5, max_T=200):
    X_small = []
    for Xi in X_list:
        Xi_ds = Xi[::stride]
        if Xi_ds.shape[0] > max_T:
            Xi_ds = Xi_ds[:max_T]
        X_small.append(Xi_ds)
    return X_small

def build_wsindy_model(
    t_grid,             
    polyorder,
    K,
    sr3_threshold,       
    sr3_nu,              
    sr3_max_iter,
):
    poly_lib = PolynomialLibrary(
        degree=polyorder,
        include_bias=False  
    )
    diff_method=ps.FiniteDifference(axis=-2)

    if sr3_threshold <= 0:
        sr3_threshold = 1e-12 
        
    lam = SR3.calculate_l0_weight(sr3_threshold, sr3_nu)
    optimizer = SR3(
        reg_weight_lam=lam,
        regularizer="L0",
        relax_coeff_nu=sr3_nu,
        max_iter=sr3_max_iter,
        tol=1e-10,
        normalize_columns=True,
        verbose=False,
    )
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=poly_lib,
        differentiation_method=diff_method
        # discrete_time=False,
    )
    return model

def _coeff_and_names_from_model(model, state_dim):
    C_raw = np.asarray(model.coefficients(), dtype=float)
    weak_lib = model.feature_library
    inner_lib = getattr(weak_lib, "function_library", None)
    if inner_lib is not None and hasattr(inner_lib, "powers_"):
        powers = inner_lib.powers_
        names = _names_from_powers(powers)
        n_terms = len(names)
    else:
        raw_names = model.get_feature_names()
        names = []
        for nm in raw_names:
            parts = []
            tokens = nm.split(" ")
            for tok in tokens:
                tok = tok.strip()
                if tok == "":
                    continue
                if tok.startswith("x"):
                    if "^" in tok:
                        base, p_str = tok.split("^")
                        idx = int(base[1:]) + 1
                        p = int(p_str)
                        parts += [f"x({idx})"] * p
                    else:
                        idx = int(tok[1:]) + 1
                        parts.append(f"x({idx})")
                else:
                    parts.append(tok)
            names.append("*".join(parts) if parts else "1")
        n_terms = len(names)
    

    if C_raw.shape == (n_terms, state_dim):
        C_hat = C_raw
    elif C_raw.shape == (state_dim, n_terms):
        C_hat = C_raw.T
    else:
        
        if C_raw.shape[0] == state_dim and C_raw.shape[1] == n_terms:
             C_hat = C_raw.T
        elif C_raw.shape[1] == state_dim and C_raw.shape[0] == n_terms:
             C_hat = C_raw
        else:
            raise RuntimeError(
                f"Unexpected coefficient shape {C_raw.shape}, "
                f"cannot match (n_terms={n_terms}, state_dim={state_dim})"
            )
    return C_hat, names

def fit_and_eval_wsindy(
    X_list,
    dt,
    polyorder,
    K,
    sr3_threshold,
    sr3_nu,
    sr3_max_iter,
    true_coeff=None,
    verbose=True,
):
    if len(X_list) == 0:
        raise ValueError("X_list is empty")
        
    T = X_list[0].shape[0]
    t_real = np.linspace(0.0, dt * (T - 1), T)
    
    model = build_wsindy_model(
        t_grid=t_real,
        polyorder=polyorder,
        K=K,
        sr3_threshold=sr3_threshold,
        sr3_nu=sr3_nu,
        sr3_max_iter=sr3_max_iter,
    )
    

    model.fit(
        X_list,
        t=[t_real] * len(X_list),   
    )
    
    d = X_list[0].shape[1]
    C_hat, names = _coeff_and_names_from_model(model, state_dim=d)
    
    nrmse_coeff = None
    if true_coeff is not None:
        C_true = np.asarray(true_coeff, dtype=float)
        nrmse_coeff = compute_nrmse_padded(C_hat, C_true)
    
    if verbose:
        print("--- Discovered Equations ---")
        eq_lines = format_equations(C_hat, names, state_dim=d, topk=None, thresh=0.0)
        for line in eq_lines:
            print(line)
        if nrmse_coeff is not None:
            print(f"[final] coeff-NRMSE = {nrmse_coeff:.6g}")
        print("--------------------------")

    return C_hat, names, nrmse_coeff

def _monomial_exponents_from_names(names, d):
    exps = np.zeros((len(names), d), dtype=np.int32)
    for k, nm in enumerate(names):
        nm = nm.strip()
        if nm == "" or nm == "1":
            continue
        parts = nm.split("*")
        for p in parts:
            p = p.strip()
            if p.startswith("x(") and p.endswith(")"):
                j = int(p[2:-1]) - 1
                if 0 <= j < d:
                    exps[k, j] += 1
    return exps

def _eval_rhs_from_coeff(x, exps, C_hat):
    n_terms, d = exps.shape
    phi = np.ones(n_terms, dtype=float)
    for k in range(n_terms):
        for j in range(d):
            e = exps[k, j]
            if e > 0:
                phi[k] *= (x[j] ** e)
    dxdt = phi @ C_hat
    return dxdt

def rollout_mse_autonomous_euler(
    C_hat,
    names,
    X_val_list,
    dt,
    substeps=1,
    clip_x=1e6,
):
    if len(X_val_list) == 0:
        return np.inf

    d = X_val_list[0].shape[1]
    exps = _monomial_exponents_from_names(names, d)
    all_err2 = []

    for Xi in X_val_list:
        T_i = Xi.shape[0]
        if T_i < 2:
            continue
        x_pred = Xi[0].copy()
        for t_idx in range(T_i - 1):
            h = dt / float(substeps)
            for _ in range(substeps):
                dxdt = _eval_rhs_from_coeff(x_pred, exps, C_hat)
                x_pred = x_pred + h * dxdt
                if clip_x is not None:
                    x_pred = np.clip(x_pred, -clip_x, clip_x)
            diff = x_pred - Xi[t_idx + 1]
            all_err2.append(np.mean(diff * diff))

    if len(all_err2) == 0:
        return np.inf
    return float(np.mean(all_err2))

def tune_wsindy(
    X_list, dt, polyorder, true_coeff, n_trials=50, timeout=None, seed=0,
    K_range=(100, 1000), thr_range=(1e-5, 1e-1), nu_range=(1e-6, 1e0),
    it_range=(500, 3000), ds_stride=5, ds_max_T=200, outdir="./wsindy_out"
):

    raise NotImplementedError("Tuning logic omitted for brevity in this snippet. Please use original function if tuning is needed.")


def main():
    filepath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(filepath)
    parser = argparse.ArgumentParser(description="PySINDy with parameter loading support")
    parser.add_argument("--data", type=str, default=None, help="Path to numpy .npy file, shape (N,T,d)")
    parser.add_argument("--dt", type=float, default=0.1, required=True)
    parser.add_argument("--polyorder", type=int, default=5, help="Highest polynomial order")
    parser.add_argument("--true_coeff_json", type=str, default="[[1,-3],[3,1],[0,0],[0,0],[0,0],[-1,0],[0,-1],[-1,0],[0,-1]]", help="JSON string or file path for true coefficients")
    parser.add_argument("--outdir", type=str, default="../results")
    parser.add_argument("--seed", type=int, default=3407)
    

    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    parser.add_argument("--load_params", type=str, default=None, 
                        help="Path to JSON file containing hyperparameters (K, threshold, nu, max_iter). Skips tuning.")
    

    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--tune_timeout", type=int, default=0)
    parser.add_argument("--ds_stride", type=int, default=1)
    parser.add_argument("--ds_max_T", type=int, default=200)
    

    parser.add_argument("--fixed_K", type=int, default=200)
    parser.add_argument("--fixed_threshold", type=float, default=1e-3)
    parser.add_argument("--fixed_nu", type=float, default=1e-2)
    parser.add_argument("--fixed_max_iter", type=int, default=2000)

    args = parser.parse_args()
    dt_str = f"{args.dt:.3f}"

    if args.data is None:
        args.data = f"../data/cyc_train{int(args.dt*100)}.npy"
    if args.load_params is None:
        args.load_params = f'../data/cyc_SR3_dt{dt_str}_tune_best_params.json'
    
    
    _ensure_dir(args.outdir)

    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
        return
    Xall = np.load(args.data)
    assert Xall.ndim == 3, "Data must be shape (N, T, d)"
    N, T, d = Xall.shape
    X_list = [Xall[i] for i in range(N)]


    C_true = None
    if args.true_coeff_json:
        try:
            if os.path.exists(args.true_coeff_json):
                with open(args.true_coeff_json, 'r') as f:
                    C_true = np.array(json.load(f), dtype=float)
            else:
               
                C_true = np.array(json.loads(args.true_coeff_json), dtype=float)
        except Exception as e:
            print(f"[WARN] Could not load true coefficients: {e}")
            C_true = None


    params_used = {}
    best_result = None

    if args.load_params:
      
        print(f"[Mode] Loading parameters from: {args.load_params}")
        if not os.path.exists(args.load_params):
            raise FileNotFoundError(f"Parameter file {args.load_params} not found.")
        
        with open(args.load_params, 'r', encoding='utf-8') as f:
            loaded_cfg = json.load(f)
   
        K = loaded_cfg.get('K', loaded_cfg.get('k', args.fixed_K))
        threshold = loaded_cfg.get('threshold', loaded_cfg.get('sr3_threshold', loaded_cfg.get('thr', args.fixed_threshold)))
        nu = loaded_cfg.get('nu', loaded_cfg.get('sr3_nu', loaded_cfg.get('relax_coeff', args.fixed_nu)))
        max_iter = loaded_cfg.get('max_iter', loaded_cfg.get('sr3_max_iter', loaded_cfg.get('iters', args.fixed_max_iter)))
        
        params_used = {
            "K": K,
            "threshold": threshold,
            "nu": nu,
            "max_iter": max_iter,
            "source": "loaded_json"
        }
        print(f"[Params] K={K}, threshold={threshold}, nu={nu}, max_iter={max_iter}")

        C_hat, names, coeff_nrmse = fit_and_eval_wsindy(
            X_list, args.dt, args.polyorder,
            K=K,
            sr3_threshold=threshold,
            sr3_nu=nu,
            sr3_max_iter=max_iter,
            true_coeff=C_true,
            verbose=True
        )
        

        rollout_mse = None 
        
        best_result = {
            "best_C_hat": C_hat,
            "best_names": names,
            "best_rollout_mse": rollout_mse,
            "best_coeff_nrmse": coeff_nrmse,
            "best_equations": format_equations(C_hat, names, state_dim=d),
            "params": params_used
        }

    elif args.tune:

        print("[Mode] Running Hyperparameter Tuning...")

        try:
            best_result = tune_wsindy(
                X_list, dt=args.dt, polyorder=args.polyorder, true_coeff=C_true,
                n_trials=args.n_trials, timeout=args.tune_timeout or None,
                seed=args.seed, ds_stride=args.ds_stride, ds_max_T=args.ds_max_T,
                outdir=args.outdir
            )
            params_used = {k: best_result[k] for k in ["K", "threshold", "nu", "max_iter"]}
            params_used["source"] = "optuna_tuning"
        except NameError:
            print("[ERROR] tune_wsindy function is missing. Please include the full function code.")
            return

    else:
    
        print("[Mode] Using fixed/default parameters")
        params_used = {
            "K": args.fixed_K,
            "threshold": args.fixed_threshold,
            "nu": args.fixed_nu,
            "max_iter": args.fixed_max_iter,
            "source": "cli_defaults"
        }
        
        C_hat, names, coeff_nrmse = fit_and_eval_wsindy(
            X_list, args.dt, args.polyorder,
            K=params_used["K"],
            sr3_threshold=params_used["threshold"],
            sr3_nu=params_used["nu"],
            sr3_max_iter=params_used["max_iter"],
            true_coeff=C_true,
            verbose=True
        )
        
        rollout_mse = None
        best_result = {
            "best_C_hat": C_hat,
            "best_names": names,
            "best_rollout_mse": rollout_mse,
            "best_coeff_nrmse": coeff_nrmse,
            "best_equations": format_equations(C_hat, names, state_dim=d),
            "params": params_used
        }


    if best_result:
        C_hat = best_result["best_C_hat"]
        names = best_result["best_names"]
        equations = best_result["best_equations"]
        
        
        eq_file = os.path.join(args.outdir, f"cyc_SR3_dt{args.dt:.3f}_equations.txt")
        with open(eq_file, "w", encoding="utf-8") as f:
            for line in equations:
                f.write(line + "\n")
        print(f"[Save] Equations saved to: {eq_file}")

        
            
        print("\n=== Final Discovered Equations ===")
        for line in equations:
            print(line)
        print("==================================")

if __name__ == "__main__":
    main()