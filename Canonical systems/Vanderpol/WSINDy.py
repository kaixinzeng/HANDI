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
    for exp in powers:  # exp: array of length d, each entry is exponent
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
        coef = C[:Dp, j]  # shape (Dp,)
        idx = np.arange(Dp)
        if thresh is not None and thresh > 0:
            idx = idx[np.abs(coef[idx]) >= float(thresh)]
        if topk is not None and topk > 0 and len(idx) > topk:
            take = np.argsort(-np.abs(coef[idx]))[:topk]
            idx = idx[take]
        parts = []
        for i in idx:
            parts.append(f"{coef[i]:+.5f}·{names[i]}")
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

    weak_lib = ps.WeakPDELibrary(
        function_library=poly_lib,
        spatiotemporal_grid=t_grid,  
        include_bias=False,
        K=K,                        
        H_xt=0.4 * float(t_grid[-1] - t_grid[0])
    )
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
        feature_library=weak_lib,
        discrete_time=False,
    )
    return model

def _coeff_and_names_from_model(model, state_dim):
    C_raw = np.asarray(model.coefficients(), dtype=float)
    weak_lib = model.feature_library
    inner_lib = getattr(weak_lib, "function_library", None)
    if inner_lib is not None and hasattr(inner_lib, "powers_"):
        powers = inner_lib.powers_
        names = _names_from_powers(powers)  # ['x(1)', 'x(2)', 'x(1)*x(2)', ...]
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
        raise RuntimeError(
            f"Unexpected coefficient shape {C_raw.shape}, "
            f"cannot match (n_terms={n_terms}, state_dim={state_dim})"
        )
    return C_hat, names

def fit_and_eval_wsindy(
    X_list,          # list of arrays, each shape (T,d)
    dt,
    polyorder,
    K,
    sr3_threshold,
    sr3_nu,
    sr3_max_iter,
    true_coeff=None,
    verbose=True,
):
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
        for out_idx in range(d):
            terms_str = []
            for term_idx, coef in enumerate(C_hat[:, out_idx]):
                if abs(coef) > 1e-6:
                    terms_str.append(f"{coef:+.5f}·{names[term_idx]}")
            rhs = " ".join(terms_str) if terms_str else "0"
            print(f"dx({out_idx+1})/dt = {rhs}")
        if nrmse_coeff is not None:
            print(f"[final] coeff-NRMSE = {nrmse_coeff:.6g}")

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
    dxdt = phi @ C_hat  # -> shape (d,)
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
    X_list,
    dt,
    polyorder,
    true_coeff,
    n_trials=50,
    timeout=None,
    seed=0,
    K_range=(100, 1000),
    thr_range=(1e-5, 1e-1),
    nu_range=(1e-6, 1e0),
    it_range=(500, 3000),
    ds_stride=5,
    ds_max_T=200,
    outdir="./wsindy_out"
):
    _ensure_dir(outdir)
    X_tr, X_val = _split_train_val(X_list, val_ratio=0.3, seed=seed)
    X_tr_small = _downsample_for_tuning(X_tr, stride=ds_stride, max_T=ds_max_T)
    dt_small = dt * ds_stride
    if optuna is None:
        print("[tune] Skip auto-tuning")
        fallback = {"K": 200, "threshold": 1e-3, "nu": 1e-2, "max_iter": 2000}
        C_hat_full, names_full, coeff_nrmse_full = fit_and_eval_wsindy(
            X_list, dt, polyorder,
            fallback["K"], fallback["threshold"], fallback["nu"], fallback["max_iter"],
            true_coeff=true_coeff, verbose=False
        )
        rollout_mse_full = rollout_mse_autonomous_euler(
            C_hat_full, names_full, X_val, dt,
            substeps=5, clip_x=1e6
        )
        equations_full = format_equations(
            C_hat_full, names_full,
            state_dim=C_hat_full.shape[1],
            topk=None, thresh=0.0
        )
        np.save(os.path.join(outdir, "tune_best_coeff.npy"), C_hat_full)
        with open(os.path.join(outdir, "tune_best_feature_names.txt"), "w", encoding="utf-8") as f:
            for nm in names_full:
                f.write(nm + "\n")
        with open(os.path.join(outdir, "tune_best_equations.txt"), "w", encoding="utf-8") as f:
            for line in equations_full:
                f.write(line + "\n")
        with open(os.path.join(outdir, "tune_best_params.json"), "w", encoding="utf-8") as f:
            json.dump({
                "K": fallback["K"],
                "threshold": fallback["threshold"],
                "nu": fallback["nu"],
                "max_iter": fallback["max_iter"],
                "rollout_mse": rollout_mse_full,
                "coeff_nrmse": coeff_nrmse_full,
            }, f, indent=2, ensure_ascii=False)
        return {
            "K": fallback["K"],
            "threshold": fallback["threshold"],
            "nu": fallback["nu"],
            "max_iter": fallback["max_iter"],
            "best_rollout_mse": rollout_mse_full,
            "best_coeff_nrmse": coeff_nrmse_full,
            "best_C_hat": C_hat_full,
            "best_names": names_full,
            "best_equations": equations_full,
            "best_rollout_mse_small": None,
            "best_coeff_nrmse_small": None,
            "best_C_hat_small": None,
            "best_names_small": None,
        }

    def objective(trial):
        K   = trial.suggest_int("K", K_range[0], K_range[1], step=100)
        thr = trial.suggest_float("threshold", thr_range[0], thr_range[1], log=True)
        nu  = trial.suggest_float("nu", nu_range[0], nu_range[1], log=True)
        mit = 300  
        C_hat_small, names_small, coeff_nrmse_small = fit_and_eval_wsindy(
            X_tr_small, dt_small, polyorder,
            K, thr, nu, mit,
            true_coeff=true_coeff,  
            verbose=False
        )
        trial_rollout_mse = rollout_mse_autonomous_euler(
            C_hat_small, names_small, X_val, dt,
            substeps=5, clip_x=1e6
        )
        trial.set_user_attr("C_hat_small", C_hat_small)
        trial.set_user_attr("names_small", names_small)
        trial.set_user_attr("rollout_mse_small", trial_rollout_mse)
        trial.set_user_attr("coeff_nrmse_small", coeff_nrmse_small)
        return float(trial_rollout_mse)

    def _callback(study, trial):
        best_trial = study.best_trial
        best_params = dict(best_trial.params)

        C_hat_full, names_full, coeff_nrmse_full = fit_and_eval_wsindy(
            X_list, dt, polyorder,
            best_params["K"], best_params["threshold"],
            best_params["nu"], best_params.get("max_iter", 300),
            true_coeff=true_coeff, verbose=False
        )

        rollout_mse_full = rollout_mse_autonomous_euler(
            C_hat_full, names_full, X_val, dt,
            substeps=5, clip_x=1e6
        )

        equations_full = format_equations(
            C_hat_full, names_full,
            state_dim=C_hat_full.shape[1],
            topk=None, thresh=0.0
        )

        np.save(os.path.join(outdir, "tune_best_coeff.npy"), C_hat_full)
        with open(os.path.join(outdir, "tune_best_feature_names.txt"), "w", encoding="utf-8") as f:
            for nm in names_full:
                f.write(nm + "\n")
        with open(os.path.join(outdir, "tune_best_equations.txt"), "w", encoding="utf-8") as f:
            for line in equations_full:
                f.write(line + "\n")
        with open(os.path.join(outdir, "tune_best_params.json"), "w", encoding="utf-8") as f:
            json.dump({
                "K": best_params["K"],
                "threshold": best_params["threshold"],
                "nu": best_params["nu"],
                "max_iter": best_params.get("max_iter", 300),
                "rollout_mse": rollout_mse_full,
                "coeff_nrmse": coeff_nrmse_full,
            }, f, indent=2, ensure_ascii=False)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout or None,
        callbacks=[_callback]
    )
    
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    print(f"[tune] best value(rollout_mse)={best_trial.value:.6g}, params={best_params}")

    best_C_hat_small            = best_trial.user_attrs["C_hat_small"]
    best_names_small            = best_trial.user_attrs["names_small"]
    best_rollout_mse_small      = best_trial.user_attrs["rollout_mse_small"]
    best_coeff_nrmse_small      = best_trial.user_attrs["coeff_nrmse_small"]

    C_hat_full, names_full, coeff_nrmse_full = fit_and_eval_wsindy(
        X_list, dt, polyorder,
        best_params["K"], best_params["threshold"],
        best_params["nu"], best_params.get("max_iter", 300),
        true_coeff=true_coeff, verbose=False
    )

    rollout_mse_full = rollout_mse_autonomous_euler(
        C_hat_full, names_full, X_val, dt,
        substeps=5, clip_x=1e6
    )

    equations_full = format_equations(
        C_hat_full, names_full,
        state_dim=C_hat_full.shape[1],
        topk=None, thresh=0.0
    )

    return {
        "K": best_params["K"],
        "threshold": best_params["threshold"],
        "nu": best_params["nu"],
        "max_iter": best_params.get("max_iter", 300),
        "best_rollout_mse": rollout_mse_full,
        "best_coeff_nrmse": coeff_nrmse_full,
        "best_C_hat": C_hat_full,
        "best_names": names_full,
        "best_equations": equations_full,
        "best_rollout_mse_small": best_rollout_mse_small,
        "best_coeff_nrmse_small": best_coeff_nrmse_small,
        "best_C_hat_small": best_C_hat_small,
        "best_names_small": best_names_small,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='lorenz.npy',
                        help="Path: numpy .npy, shape (N,T,d)")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--polyorder", type=int, default=3,
                        help="Highest order")
    parser.add_argument(
        "--true_coeff_json",
        type=str,
        default="[[ -10, 28, 0], [ 10, -1, 0], [ 0, 0, -2.666666666666667], [ 0, 0, 0], [ 0, 0, 1], [ 0, -1, 0], [ 0, 0, 0], [ 0, 0, 0], [ 0, 0, 0]]",
        help="Coef matrix JSON (shape (Dp,d), Dp=total number of basis functions)"
    )
    parser.add_argument("--outdir", type=str, default="./wsindy_out")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--tune", action="store_true", default=True)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--tune_timeout", type=int, default=0)
    parser.add_argument("--ds_stride", type=int, default=1,
                        help="Downsampling steps（>=1）")
    parser.add_argument("--ds_max_T", type=int, default=200,
                        help="Downsampling length")
    args = parser.parse_args()
    _ensure_dir(args.outdir)
    Xall = np.load(args.data)  
    assert Xall.ndim == 3
    N, T, d = Xall.shape
    X_list = [Xall[i] for i in range(N)]
    C_true = None
    if args.true_coeff_json:
        try:
            C_true = np.array(json.loads(args.true_coeff_json), dtype=float)
        except Exception as e:
            print(f"[WARN] load true_coeff_json error: {e} (no coeff NRMSE)")
            C_true = None

    if args.tune:
        best = tune_wsindy(
            X_list,
            dt=args.dt,
            polyorder=args.polyorder,
            true_coeff=C_true,
            n_trials=args.n_trials,
            timeout=args.tune_timeout or None,
            seed=args.seed,
            ds_stride=args.ds_stride,
            ds_max_T=args.ds_max_T,
            outdir=args.outdir
        )

        print("[cfg] best params: ",
              {k: best[k] for k in ["K", "threshold", "nu", "max_iter"]})
        print(f"[cfg] best rollout MSE (val) = {best['best_rollout_mse']}")
        print(f"[cfg] best coeff NRMSE      = {best['best_coeff_nrmse']}")
        print("[best equations]")
        for line in best["best_equations"]:
            print(line)
        C_hat          = best["best_C_hat"]
        names          = best["best_names"]
        rollout_mse    = best["best_rollout_mse"]
        coeff_nrmse    = best["best_coeff_nrmse"]

    else:
        fixed_params = {"K": 200, "threshold": 1e-3, "nu": 1e-2, "max_iter": 2000}
        print(f"[cfg] fixed params: {fixed_params}")

        C_hat, names, coeff_nrmse = fit_and_eval_wsindy(
            X_list,
            dt=args.dt,
            polyorder=args.polyorder,
            K=fixed_params["K"],
            sr3_threshold=fixed_params["threshold"],
            sr3_nu=fixed_params["nu"],
            sr3_max_iter=fixed_params["max_iter"],
            true_coeff=C_true,
            verbose=True
        )
        rollout_mse = None
    np.save(os.path.join(args.outdir, "coeff_hat.npy"), C_hat)
    with open(os.path.join(args.outdir, "feature_names.txt"),
              "w", encoding="utf-8") as f:
        for nm in names:
            f.write(nm + "\n")
    if coeff_nrmse is not None:
        with open(os.path.join(args.outdir, "nrmse.txt"),
                  "w", encoding="utf-8") as f:
            f.write(f"{coeff_nrmse:.6g}\n")
    final_eq_lines = format_equations(
        C_hat, names,
        state_dim=C_hat.shape[1],
        topk=None,
        thresh=0.0
    )
    with open(os.path.join(args.outdir, "equations.txt"),
              "w", encoding="utf-8") as f:
        for line in final_eq_lines:
            f.write(line + "\n")
    params_to_dump = {
        "dt": args.dt,
        "polyorder": args.polyorder,
        "tuned": bool(args.tune),
        "best_coeff_nrmse": coeff_nrmse,
        "best_rollout_mse": rollout_mse,
    }
    if args.tune:
        params_to_dump.update({
            "K": best["K"],
            "threshold": best["threshold"],
            "nu": best["nu"],
            "max_iter": best["max_iter"],
        })
    else:
        params_to_dump.update({
            "K": 200,
            "threshold": 1e-3,
            "nu": 1e-2,
            "max_iter": 2000,
        })
    with open(os.path.join(args.outdir, "params.json"),
              "w", encoding="utf-8") as f:
        json.dump(params_to_dump, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
