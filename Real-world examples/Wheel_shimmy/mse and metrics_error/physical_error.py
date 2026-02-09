#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outputs:
- ts_physics_stride{stride}.csv
- ts_relative_error_stride{stride}.csv
"""

import os
import re
import csv
import json
import math
import argparse
from typing import Dict, Tuple, Optional, Callable, List, Any

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


# =========================
# 0) dt / stride
# =========================
DT0 = 0.005

def dt_from_stride(stride: int) -> float:
    return DT0 * float(stride)


# =========================
# 1) Equation parsing -> unify all into (x1, x2)
# =========================
def _normalize_equation_text(text: str) -> str:
    text = text.replace("·", "*").replace("−", "-").replace("×", "*")
    text = re.sub(r"x\((\d+)\)", r"x\1", text)
    text = re.sub(r"dx\((\d+)\)/dt", r"dx\1/dt", text)
    return text


def parse_2d_equations_from_file(path: str) -> Tuple[sp.Expr, sp.Expr, Tuple[sp.Symbol, sp.Symbol]]:
    """
    Returns f1(x1,x2), f2(x1,x2), (x1,x2)

    Supported:
      dx(1)/dt = ...   dx(2)/dt = ...
      dx(0)/dt = ...   dx(1)/dt = ...  (PSE-like, x0/x1) -> mapped to (x1,x2)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = _normalize_equation_text(f.read())

    eq_map: Dict[int, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^dx(\d+)/dt\s*=\s*(.+)$", line)
        if m:
            idx = int(m.group(1))
            eq_map[idx] = m.group(2).strip()

    if len(eq_map) < 2:
        raise ValueError(f"Failed to parse 2 equations from: {path}")

    x1, x2 = sp.symbols("x1 x2", real=True)

    # PSE-like: dx0/dt & dx1/dt with x0,x1 -> map x0->x1, x1->x2
    if (0 in eq_map) and (1 in eq_map) and (2 not in eq_map):
        locals_dict = {"x0": x1, "x1": x2, "x2": x2}
        f1 = sp.sympify(eq_map[0], locals=locals_dict)
        f2 = sp.sympify(eq_map[1], locals=locals_dict)
        return f1, f2, (x1, x2)

    locals_dict = {"x1": x1, "x2": x2, "x0": x1}
    if (1 in eq_map) and (2 in eq_map):
        f1 = sp.sympify(eq_map[1], locals=locals_dict)
        f2 = sp.sympify(eq_map[2], locals=locals_dict)
    else:
        keys = sorted(eq_map.keys())[:2]
        f1 = sp.sympify(eq_map[keys[0]], locals=locals_dict)
        f2 = sp.sympify(eq_map[keys[1]], locals=locals_dict)

    return f1, f2, (x1, x2)


def make_rhs(f1: sp.Expr, f2: sp.Expr, xs: Tuple[sp.Symbol, sp.Symbol]) -> Callable:
    x1, x2 = xs
    return sp.lambdify((x1, x2), (f1, f2), "numpy")


# =========================
# 2) Simulation (dense t_eval)
# =========================
def simulate_on_grid(rhs_func: Callable, x0: np.ndarray, t_eval: np.ndarray,
                     ode_method: str = "RK45", rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Simulate and return x(t_eval) with shape (len(t_eval), 2)."""
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be 1D with >=2 points")

    def f(t, y):
        a, b = rhs_func(y[0], y[1])
        return [float(a), float(b)]

    sol = solve_ivp(
        f,
        (float(t_eval[0]), float(t_eval[-1])),
        y0=np.asarray(x0, dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method=ode_method,
    )
    if not sol.success:
        raise RuntimeError("solve_ivp failed")
    return sol.y.T


# =========================
# 3) De-standardization + truncation
# =========================
def destandardize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """x can be (T,2) or (N,T,2). Returns x_phys = x*sigma + mu."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if x.ndim == 2:
        return x * sigma.reshape(1, 2) + mu.reshape(1, 2)
    if x.ndim == 3:
        return x * sigma.reshape(1, 1, 2) + mu.reshape(1, 1, 2)
    raise ValueError("destandardize expects (T,2) or (N,T,2)")


def truncate_by_time(x: np.ndarray, dt: float, t_end: float) -> np.ndarray:
    """x: (T,2) sampled with dt. Return first samples such that time <= t_end."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("truncate_by_time expects (T,2)")
    if t_end <= 0:
        return x[:2].copy() if x.shape[0] >= 2 else x[:1].copy()
    n = int(math.floor(t_end / float(dt))) + 1
    n = max(2, min(n, x.shape[0]))
    return x[:n].copy()


# =========================
# 4) TS metrics (FFT + robust peaks for delta)
# =========================
def _dominant_freq_fft(sig: np.ndarray, dt: float) -> Optional[float]:
    sig = np.asarray(sig, dtype=float)
    if sig.size < 16:
        return None
    sig = sig - np.mean(sig)
    n = sig.size
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.abs(np.fft.rfft(sig))
    if spec.size <= 1:
        return None
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if k <= 0 or freqs[k] <= 0:
        return None
    return float(freqs[k])


def _log_decrement_from_peaks(
    sig: np.ndarray,
    dt: float,
    fd_hz: float,
    prominence: Optional[float],
    peak_distance_frac: float,
    max_pos_peaks: int,
) -> Tuple[Optional[float], int, str]:
    sig = np.asarray(sig, dtype=float)
    sig = sig - np.mean(sig)

    if (fd_hz is None) or (not np.isfinite(fd_hz)) or (fd_hz <= 0):
        return None, 0, "no_fd_for_distance"

    period_s = 1.0 / float(fd_hz)
    dist_samp = max(1, int(round(float(peak_distance_frac) * period_s / float(dt))))

    kwargs = {"distance": dist_samp}
    if prominence is not None and float(prominence) > 0:
        kwargs["prominence"] = float(prominence)

    peaks, _ = find_peaks(sig, **kwargs)
    if peaks.size < 3:
        return None, int(peaks.size), f"few_peaks(dist={dist_samp})"

    A = sig[peaks]
    A = A[A > 0]
    if A.size < 3:
        return None, int(A.size), "few_pos_peaks"

    K = int(max_pos_peaks)
    if K > 0:
        A = A[:K]
    if A.size < 3:
        return None, int(A.size), "few_pos_peaks_after_K"

    ratios = A[:-1] / A[1:]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < 2:
        return None, int(A.size), "bad_ratios"

    delta = float(np.median(np.log(ratios)))
    return delta, int(A.size), "ok"


def ts_metrics_from_traj(
    x_phys: np.ndarray,
    dt_used: float,
    signal: str,
    prominence: Optional[float],
    peak_distance_frac: float,
    max_pos_peaks: int,
) -> Dict[str, Any]:
    nan = float("nan")
    x = np.asarray(x_phys, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        return {"ts_success": 0, "ts_note": "bad_shape",
                "ts_fd_hz": nan, "ts_omega_d": nan, "ts_omega_n": nan, "ts_fn_hz": nan,
                "ts_zeta": nan, "ts_Q": nan, "ts_delta": nan, "ts_n_peaks": 0}

    if signal == "radius":
        sig = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        note_base = "radius"
    elif signal == "theta":
        sig = x[:, 0]
        note_base = "theta"
    else:
        return {"ts_success": 0, "ts_note": f"bad_signal:{signal}",
                "ts_fd_hz": nan, "ts_omega_d": nan, "ts_omega_n": nan, "ts_fn_hz": nan,
                "ts_zeta": nan, "ts_Q": nan, "ts_delta": nan, "ts_n_peaks": 0}

    fd = _dominant_freq_fft(sig, dt_used)
    if fd is None:
        return {"ts_success": 0, "ts_note": f"{note_base}:no_fd",
                "ts_fd_hz": nan, "ts_omega_d": nan, "ts_omega_n": nan, "ts_fn_hz": nan,
                "ts_zeta": nan, "ts_Q": nan, "ts_delta": nan, "ts_n_peaks": 0}

    omega_d = 2.0 * math.pi * float(fd)

    delta, n_pos_peaks, peak_note = _log_decrement_from_peaks(
        sig, dt=dt_used, fd_hz=float(fd), prominence=prominence,
        peak_distance_frac=float(peak_distance_frac),
        max_pos_peaks=int(max_pos_peaks),
    )

    if (delta is None) or (not np.isfinite(delta)) or (delta <= 0):
        return {"ts_success": 0, "ts_note": f"{note_base}:no_delta:{peak_note}",
                "ts_fd_hz": float(fd), "ts_omega_d": float(omega_d),
                "ts_omega_n": nan, "ts_fn_hz": nan,
                "ts_zeta": nan, "ts_Q": nan,
                "ts_delta": float(delta) if delta is not None else nan,
                "ts_n_peaks": int(n_pos_peaks)}

    zeta = float(delta / math.sqrt((2.0 * math.pi) ** 2 + delta ** 2))
    Q = float(math.pi / delta)

    if (not np.isfinite(zeta)) or (zeta <= 0) or (zeta >= 1.0):
        return {"ts_success": 0, "ts_note": f"{note_base}:bad_zeta",
                "ts_fd_hz": float(fd), "ts_omega_d": float(omega_d),
                "ts_omega_n": nan, "ts_fn_hz": nan,
                "ts_zeta": float(zeta) if np.isfinite(zeta) else nan,
                "ts_Q": float(Q) if np.isfinite(Q) else nan,
                "ts_delta": float(delta), "ts_n_peaks": int(n_pos_peaks)}

    omega_n = float(omega_d / math.sqrt(1.0 - zeta ** 2))
    fn = float(omega_n / (2.0 * math.pi))

    return {"ts_success": 1, "ts_note": f"{note_base}",
            "ts_fd_hz": float(fd), "ts_omega_d": float(omega_d),
            "ts_omega_n": float(omega_n), "ts_fn_hz": float(fn),
            "ts_zeta": float(zeta), "ts_Q": float(Q),
            "ts_delta": float(delta), "ts_n_peaks": int(n_pos_peaks)}


# =========================
# 5) Data loading
# =========================
def load_truth_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError("Expected truth shape (T,2)")
        return arr
    if arr.ndim == 3:
        if arr.shape[2] != 2:
            raise ValueError("Expected truth shape (N,T,2)")
        return arr
    raise ValueError("Unsupported truth npy shape")


# =========================
# 6) Defaults: model paths
# =========================
def build_default_models(stride: int) -> List[Tuple[str, str]]:
    if stride == 16:
        return [
            ("EDMD",  "edmd/stride_16/equations.txt"),
            ("gEDMD", "gedmd/stride_16/gedmd_dt0.08/best_equations.txt"),
            ("HANDI", "HANDI/find_results/run_ds_16_dt_0.0800/best_equations_by_mse_dt0.080.txt"),
            ("PSE",   "PSE/0.08+4/final_ode_system_from_psrn.txt"),
            ("SINDy", "Sindy/sindy_wheel/sindy_equation_best_wheel_shimmy_train_stride16.txt"),
            ("SR3",   "SR3/sindy_sr3_wheel/wheel_dt0.08/sindy_sr3_equations.txt"),
            ("wSINDy", "wsindy_wheel/dt0.08/tune_best_equations.txt"),
        ]
    return []


# =========================
# 7) FAIL-aware MEAN and relative errors
# =========================
def add_mean_rows_fail_aware(rows: List[Dict[str, Any]], base_names: List[str], expected_n: int) -> None:
    nan = float("nan")
    keys = ["ts_omega_n", "ts_zeta", "ts_omega_d", "ts_fd_hz", "ts_fn_hz", "ts_Q", "ts_delta"]
    for b in base_names:
        sub = [r for r in rows if r.get("name", "").startswith(f"{b}(traj")]
        if len(sub) < expected_n:
            any_fail = True
        else:
            any_fail = any(int(r.get("ts_success", 0)) == 0 for r in sub)

        mean = {"name": f"{b}(MEAN)", "traj_idx": "MEAN",
                "equation_path": sub[0].get("equation_path", None) if sub else None,
                "dt_used": sub[0].get("dt_used", None) if sub else None,
                "t_end": sub[0].get("t_end", None) if sub else None}

        if any_fail or len(sub) == 0:
            mean["ts_success"] = 0
            for k in keys:
                mean[k] = nan
            mean["ts_note"] = "partial_fail"
        else:
            mean["ts_success"] = 1
            for k in keys:
                vals = [r.get(k, nan) for r in sub]
                vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
                mean[k] = float(np.mean(vals)) if vals else nan
            mean["ts_note"] = "ok"
        rows.append(mean)


def add_relative_errors_vs_truth_mean(rows: List[Dict[str, Any]]) -> None:
    nan = float("nan")
    truth = None
    for r in rows:
        if r.get("name") == "TRUTH(MEAN)":
            truth = r
            break
    if truth is None:
        return

    if int(truth.get("ts_success", 0)) == 0:
        for r in rows:
            if r.get("name", "").endswith("(MEAN)") and not r.get("name", "").startswith("TRUTH"):
                r["ts_relerr_wn"] = nan
                r["ts_relerr_zeta"] = nan
        return

    wn_ref = float(truth.get("ts_omega_n", nan))
    z_ref = float(truth.get("ts_zeta", nan))

    def rel_err(a, b):
        try:
            a = float(a); b = float(b)
        except Exception:
            return nan
        if (not np.isfinite(a)) or (not np.isfinite(b)) or b == 0:
            return nan
        return abs(a - b) / abs(b)

    for r in rows:
        name = r.get("name", "")
        if not name.endswith("(MEAN)") or name.startswith("TRUTH"):
            continue
        if int(r.get("ts_success", 0)) == 0:
            r["ts_relerr_wn"] = nan
            r["ts_relerr_zeta"] = nan
        else:
            r["ts_relerr_wn"] = rel_err(r.get("ts_omega_n", nan), wn_ref)
            r["ts_relerr_zeta"] = rel_err(r.get("ts_zeta", nan), z_ref)


def export_relative_error_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    nan = float("nan")
    truth = None
    for r in rows:
        if r.get("name") == "TRUTH(MEAN)":
            truth = r
            break
    if truth is None:
        raise RuntimeError("TRUTH(MEAN) not found.")

    truth_wn = truth.get("ts_omega_n", nan)
    truth_zeta = truth.get("ts_zeta", nan)

    recs = []
    for r in rows:
        name = r.get("name", "")
        if not name.endswith("(MEAN)") or name.startswith("TRUTH"):
            continue
        recs.append({
            "method": name.replace("(MEAN)", ""),
            "ts_omega_n": r.get("ts_omega_n", nan),
            "ts_zeta": r.get("ts_zeta", nan),
            "truth_ts_omega_n": truth_wn,
            "truth_ts_zeta": truth_zeta,
            "ts_relerr_wn": r.get("ts_relerr_wn", nan),
            "ts_relerr_zeta": r.get("ts_relerr_zeta", nan),
            "ts_success": r.get("ts_success", 0),
        })

    fields = ["method","ts_omega_n","ts_zeta","truth_ts_omega_n","truth_ts_zeta",
              "ts_relerr_wn","ts_relerr_zeta","ts_success"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in recs:
            w.writerow(rec)


def save_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in keys})


# =========================
# 8) Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=16)

    ap.add_argument("--truth_stride", type=str, default="wheel_shimmy_train_ds_16.npy",
                    help="stride truth npy path (N,T,2) or (T,2); defines horizon")
    ap.add_argument("--truth_hires", type=str, default="wheel_shimmy_train.npy",
                    help="optional hires truth npy (dt=dt_hires). If set, TRUTH metrics use hires truncated to horizon.")
    ap.add_argument("--dt_hires", type=float, default=0.005, help="dt for truth_hires (default 0.005)")
    ap.add_argument("--dt_eval", type=float, default=0.005, help="dense dt for model evaluation + TS metrics")

    # de-standardization (defaults = your paper)
    ap.add_argument("--mu1", type=float, default=0.0009544)
    ap.add_argument("--mu2", type=float, default=-0.0475395)
    ap.add_argument("--sigma1", type=float, default=0.0664808)
    ap.add_argument("--sigma2", type=float, default=0.9665666)

    # DEFAULT TRUE
    ap.add_argument("--truth_is_standardized", action="store_true", default=True)
    ap.add_argument("--model_state_is_standardized", action="store_true", default=True)
    ap.add_argument("--truth_is_physical", dest="truth_is_standardized", action="store_false")
    ap.add_argument("--model_state_is_physical", dest="model_state_is_standardized", action="store_false")

    ap.add_argument("--models_json", type=str, default="", help="Optional JSON list of {name,path}")
    ap.add_argument("--out_csv", type=str, default="", help="output csv path")
    ap.add_argument("--out_rel_csv", type=str, default="", help="relative error csv path")

    ap.add_argument("--ts_signal", type=str, default="theta", choices=["radius","theta"])
    ap.add_argument("--prominence", type=float, default=0.0)
    ap.add_argument("--peak_distance_frac", type=float, default=0.6)
    ap.add_argument("--max_pos_peaks", type=int, default=12)

    # ODE solver options
    ap.add_argument("--ode_method", type=str, default="RK45", choices=["RK45", "DOP853", "Radau", "BDF", "LSODA"])
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-10)

    # extra IC (physical units)
    ap.add_argument("--extra_ic", type=str, default="-2.368421052631579,1.3157894736842106")
    ap.add_argument("--extra_T_end", type=float, default=-1.0)

    args = ap.parse_args()

    stride = int(args.stride)
    dt_stride = dt_from_stride(stride)
    dt_eval = float(args.dt_eval)
    dt_hires = float(args.dt_hires)

    mu = np.array([args.mu1, args.mu2], dtype=float)
    sigma = np.array([args.sigma1, args.sigma2], dtype=float)
    prominence = None if float(args.prominence) <= 0 else float(args.prominence)

    # load stride truth -> horizon
    truth_stride = load_truth_npy(args.truth_stride)
    if truth_stride.ndim == 2:
        truth_stride = truth_stride[None, ...]
    truth_stride_phys = destandardize(truth_stride, mu=mu, sigma=sigma) if bool(args.truth_is_standardized) else truth_stride

    N, T_stride, D = truth_stride_phys.shape
    if D != 2:
        raise ValueError("truth_stride must be 2D state")
    t_end = (T_stride - 1) * dt_stride

    # optional hires truth
    truth_hires_phys = None
    if args.truth_hires.strip():
        truth_hires = load_truth_npy(args.truth_hires)
        if truth_hires.ndim == 2:
            truth_hires = truth_hires[None, ...]
        if truth_hires.shape[0] != N:
            raise ValueError(f"truth_hires N={truth_hires.shape[0]} != truth_stride N={N}")
        truth_hires_phys = destandardize(truth_hires, mu=mu, sigma=sigma) if bool(args.truth_is_standardized) else truth_hires

    # models
    if args.models_json.strip():
        with open(args.models_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        models = [(d["name"], d["path"]) for d in cfg]
    else:
        models = build_default_models(stride)
    if len(models) == 0:
        raise RuntimeError("No models specified.")

    # dense evaluation grid for models
    n_eval = int(math.floor(t_end / dt_eval)) + 1
    n_eval = max(2, n_eval)
    t_eval = np.arange(n_eval, dtype=float) * dt_eval
    t_eval[-1] = t_end

    rows: List[Dict[str, Any]] = []
    nan = float("nan")

    for traj_idx in range(N):
        # TRUTH metrics: hires (truncated) if available, else stride
        if truth_hires_phys is not None:
            x_truth = truncate_by_time(truth_hires_phys[traj_idx], dt=dt_hires, t_end=t_end)
            dt_truth_used = dt_hires
        else:
            x_truth = truth_stride_phys[traj_idx]
            dt_truth_used = dt_stride

        ts_t = ts_metrics_from_traj(
            x_truth, dt_used=dt_truth_used, signal=args.ts_signal, prominence=prominence,
            peak_distance_frac=args.peak_distance_frac, max_pos_peaks=args.max_pos_peaks
        )
        rows.append({
            "name": f"TRUTH(traj{traj_idx})",
            "equation_path": None,
            "traj_idx": traj_idx,
            "dt_used": dt_truth_used,
            "t_end": t_end,
            **ts_t,
        })

        # model IC in physical units: prefer hires IC if provided
        x0_phys = (truth_hires_phys[traj_idx][0] if truth_hires_phys is not None else truth_stride_phys[traj_idx][0]).copy()

        for mname, mpath in models:
            label = f"{mname}(traj{traj_idx})"
            if not os.path.exists(mpath):
                rows.append({
                    "name": label, "equation_path": mpath, "traj_idx": traj_idx,
                    "dt_used": dt_eval, "t_end": t_end,
                    "ts_success": 0, "ts_note": "missing_file",
                    "ts_fd_hz": nan, "ts_omega_d": nan, "ts_omega_n": nan, "ts_fn_hz": nan,
                    "ts_zeta": nan, "ts_Q": nan, "ts_delta": nan, "ts_n_peaks": 0,
                })
                continue

            try:
                f1, f2, xs = parse_2d_equations_from_file(mpath)
                rhs = make_rhs(f1, f2, xs)

                if bool(args.model_state_is_standardized):
                    z0 = (x0_phys - mu) / sigma
                    zsim = simulate_on_grid(rhs, x0=z0, t_eval=t_eval,
                                            ode_method=args.ode_method, rtol=args.rtol, atol=args.atol)
                    xsim = destandardize(zsim, mu=mu, sigma=sigma)
                else:
                    xsim = simulate_on_grid(rhs, x0=x0_phys, t_eval=t_eval,
                                            ode_method=args.ode_method, rtol=args.rtol, atol=args.atol)

                ts_m = ts_metrics_from_traj(
                    xsim, dt_used=dt_eval, signal=args.ts_signal, prominence=prominence,
                    peak_distance_frac=args.peak_distance_frac, max_pos_peaks=args.max_pos_peaks
                )
                rows.append({
                    "name": label, "equation_path": mpath, "traj_idx": traj_idx,
                    "dt_used": dt_eval, "t_end": t_end,
                    **ts_m,
                })
            except Exception as e:
                rows.append({
                    "name": label, "equation_path": mpath, "traj_idx": traj_idx,
                    "dt_used": dt_eval, "t_end": t_end,
                    "ts_success": 0, "ts_note": f"sim_fail:{type(e).__name__}:{e}",
                    "ts_fd_hz": nan, "ts_omega_d": nan, "ts_omega_n": nan, "ts_fn_hz": nan,
                    "ts_zeta": nan, "ts_Q": nan, "ts_delta": nan, "ts_n_peaks": 0,
                })

    # MEAN + relative errors
    base_names = ["TRUTH"] + [m[0] for m in models]
    add_mean_rows_fail_aware(rows, base_names=base_names, expected_n=N)
    add_relative_errors_vs_truth_mean(rows)

    # extra IC (limit cycle-ish)
    if args.extra_ic.strip():
        parts = [p.strip() for p in args.extra_ic.split(",")]
        if len(parts) == 2:
            extra_x0 = np.array([float(parts[0]), float(parts[1])], dtype=float)
            extra_t_end = float(args.extra_T_end)
            if extra_t_end < 0:
                extra_t_end = t_end
            n_extra = int(math.floor(extra_t_end / dt_eval)) + 1
            n_extra = max(2, n_extra)
            t_eval_extra = np.arange(n_extra, dtype=float) * dt_eval
            t_eval_extra[-1] = extra_t_end
            for mname, mpath in models:
                if not os.path.exists(mpath):
                    continue
                try:
                    f1, f2, xs = parse_2d_equations_from_file(mpath)
                    rhs = make_rhs(f1, f2, xs)
                    if bool(args.model_state_is_standardized):
                        z0 = (extra_x0 - mu) / sigma
                        zsim = simulate_on_grid(rhs, x0=z0, t_eval=t_eval_extra,
                                                ode_method=args.ode_method, rtol=args.rtol, atol=args.atol)
                        xsim = destandardize(zsim, mu=mu, sigma=sigma)
                    else:
                        xsim = simulate_on_grid(rhs, x0=extra_x0, t_eval=t_eval_extra,
                                                ode_method=args.ode_method, rtol=args.rtol, atol=args.atol)
                    ts_m = ts_metrics_from_traj(
                        xsim, dt_used=dt_eval, signal=args.ts_signal, prominence=prominence,
                        peak_distance_frac=args.peak_distance_frac, max_pos_peaks=args.max_pos_peaks
                    )
                    rows.append({
                        "name": f"{mname}(extraIC)", "equation_path": mpath, "traj_idx": "extraIC",
                        "dt_used": dt_eval, "t_end": extra_t_end, **ts_m
                    })
                except Exception:
                    pass

    out_csv = args.out_csv.strip() or f"ts_physics_stride{stride}.csv"
    out_rel = args.out_rel_csv.strip() or f"ts_relative_error_stride{stride}.csv"
    save_csv(rows, out_csv)
    export_relative_error_csv(rows, out_rel)

    print(f"[OK] saved: {out_csv}")
    print(f"[OK] saved: {out_rel}")
    print(f"[INFO] horizon t_end={t_end:.6f}s | dt_stride={dt_stride:.6f} | dt_eval={dt_eval:.6f} | truth_used={'hires' if truth_hires_phys is not None else 'stride'}")


if __name__ == "__main__":
    main()
