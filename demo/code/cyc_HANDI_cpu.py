import os
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations_with_replacement
from scipy.linalg import logm
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def _safe_pinv(a, rcond=1e-10):
    return np.linalg.pinv(a, rcond=rcond)

def _nan_guard_np(x, clip=1e6):
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)

def _nan_guard_torch(t, clip=1e6):
    return torch.clamp(torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)

def build_poly(X, polyorder=3):
   
    N, d = X.shape
    basis, names = [], []
    for i in range(d):
        basis.append(X[:, i:i+1]); names.append(f'x({i+1})')
    for deg in range(2, polyorder+1):
        for comb in combinations_with_replacement(range(d), deg):
            term = np.ones((N,1)); name = ''
            for idx in comb:
                term *= X[:, idx:idx+1]
                name = f'{name}*x({idx+1})' if name else f'x({idx+1})'
            basis.append(term); names.append(name)
    return np.hstack(basis), names

def ridge(A, b, lam=1e-8):
    AtA = A.T @ A; Atb = A.T @ b
    return np.linalg.solve(AtA + lam*np.eye(AtA.shape[0]), Atb)

def format_equations(L, names, topk=8, thresh=1e-6):
  
    Dp = len(names); d = sum(1 for nm in names if nm.startswith('x(') and nm.count('*')==0)
    eqs=[]
    for i in range(d):
        target = names[i]
        coeffs = []
        for j, term in enumerate(names):
            c = float(L[j, i])
            if abs(c) >= thresh:
                coeffs.append((abs(c), c, term))
        coeffs.sort(reverse=True, key=lambda t: t[0])
        coeffs = coeffs[:topk]
        if not coeffs:
            eqs.append(f"d{target}/dt = 0"); continue
        parts=[]
        for _, c, term in coeffs:
            sign = " + " if c>=0 else " - "
            mag = abs(c)
            parts.append(f"{sign}{mag:.6g}*{term}")
        rhs = "".join(parts)
        rhs = rhs[3:] if rhs.startswith(" + ") else ("-" + rhs[3:])
        eqs.append(f"d{target}/dt = {rhs}")
    return eqs

class DictNN(nn.Module):

    def __init__(self, in_dim, out_dim, width=128, depth=2, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class LightningResidualKoopman(nn.Module):
   
   
    def __init__(self,
                 Xall_np,
                 dt=0.1,
                 polyorder=3,
                 g_dim=32,
                 width=128,
                 depth=2,
                 dropout=0.0,
                 lam_phi=3e-10,
                 lam_g=3e-6,
                 lam_full=3e-6,
                 rollout_B=8):
        super().__init__()
     
        self.dt = dt
        self.polyorder = polyorder
        self.g_dim = g_dim
        self.lam_phi = lam_phi
        self.lam_g = lam_g
        self.lam_full = lam_full
        self.rollout_B = rollout_B
        
        Xall = Xall_np
        N, T, d = Xall.shape
        self.d = d
        self.win_S = max(2, self.rollout_B + 1)

  
        X0 = Xall[:, :-1, :].reshape(-1, d)
        X1 = Xall[:,  1:, :].reshape(-1, d)
        
        Phi_x_mono, names = build_poly(X0, polyorder=polyorder)
        Phi_y_mono, _    = build_poly(X1, polyorder=polyorder)
        
        self.names = names
        self.Dp = Phi_x_mono.shape[1]

      
        Qx, Rx = np.linalg.qr(Phi_x_mono)
        Rinv = _safe_pinv(Rx, rcond=1e-10)
        Qy = Phi_y_mono @ Rinv
        
    
        K_phi_Q = ridge(Qx, Qy, lam=self.lam_phi)
        RQ = Qy - Qx @ K_phi_Q

    
        self.register_buffer("X_all_cpu", torch.from_numpy(Xall.astype(np.float32)), persistent=False)
        self.register_buffer("X0_cpu", torch.from_numpy(X0.astype(np.float32)), persistent=False)
        self.register_buffer("X1_cpu", torch.from_numpy(X1.astype(np.float32)), persistent=False)
        self.register_buffer("Qx_cpu", torch.from_numpy(Qx.astype(np.float32)), persistent=False)
        self.register_buffer("Qy_cpu", torch.from_numpy(Qy.astype(np.float32)), persistent=False)
        self.register_buffer("RQ_cpu", torch.from_numpy(RQ.astype(np.float32)), persistent=False)

        self.register_buffer("R_mono2Q", torch.from_numpy(Rx.astype(np.float32)), persistent=False)
        self.register_buffer("Rinv_Q2mono", torch.from_numpy(Rinv.astype(np.float32)), persistent=False)

    
        self.model = DictNN(d, g_dim, width=width, depth=depth, dropout=dropout)
        
    
        self.Kg = nn.Parameter(torch.zeros(g_dim, self.Dp))
        self.register_buffer("Kg_closed", torch.zeros(g_dim, self.Dp), persistent=False)
        self.register_buffer("K_aug_Q", torch.eye(self.Dp + self.g_dim), persistent=False)

    @torch.no_grad()
    def fwd_all(self, X_cpu, bs=65536):
      
        outs=[]
       
        dev = torch.device('cpu')
        for i in range(0, X_cpu.shape[0], bs):
            xb = X_cpu[i:i+bs].to(dev, non_blocking=False)
            outs.append(self.model(xb))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def compute_Kaug_full(self):

        dev = torch.device('cpu')
  
        Qx = self.Qx_cpu.to(dev, dtype=torch.float64)
        Qy = self.Qy_cpu.to(dev, dtype=torch.float64)
        
    
        Gx_raw = self.fwd_all(self.X0_cpu).to(dtype=torch.float64)
        Gy_raw = self.fwd_all(self.X1_cpu).to(dtype=torch.float64)

      
        Gx_perp = Gx_raw - Qx @ (Qx.T @ Gx_raw)
        Gy_perp = Gy_raw - Qy @ (Qy.T @ Gy_raw)

 
        Z_x = torch.cat([Qx, Gx_perp], dim=1)
        Z_y = torch.cat([Qy, Gy_perp], dim=1)
        
        D = Z_x.shape[1]
        I = torch.eye(D, device=dev, dtype=torch.float64)
        lam = float(max(self.lam_full, 1e-4))
        
     
        ZtZ = Z_x.T @ Z_x + lam * I
        ZtY = Z_x.T @ Z_y
        
        try:
            K = torch.linalg.solve(ZtZ, ZtY)
        except RuntimeError:
        
            diag_mean = torch.mean(torch.diag(ZtZ)).clamp(min=1e-12)
            lam2 = max(lam, float(1e-3 * diag_mean))
            ZtZ2 = (Z_x.T @ Z_x) + lam2 * I
            try:
                K = torch.linalg.solve(ZtZ2, ZtY)
            except RuntimeError:
                s = math.sqrt(lam2)
                Zxa = torch.cat([Z_x, s * I], dim=0)
                Zya = torch.cat([Z_y, torch.zeros(D, D, device=dev, dtype=torch.float64)], dim=0)
                K = torch.linalg.lstsq(Zxa, Zya, rcond=None).solution
        
        K = K.to(torch.float32)
        return _nan_guard_torch(K)

@torch.no_grad()
def extract_L_C_eqs(module):

    K_aug_Q = module.K_aug_Q.detach().cpu().numpy()
    Dp = module.Dp
    gdim = module.g_dim
    
    R = module.R_mono2Q.detach().cpu().numpy()
    Rinv = module.Rinv_Q2mono.detach().cpu().numpy()

    K_aug_Q = _nan_guard_np(K_aug_Q)
    R = _nan_guard_np(R)
    Rinv = _nan_guard_np(Rinv)

    #  T: Mono -> Q+G -> Mono
    # K_aug_mono = T_inv @ K_aug_Q @ T
    T = np.block([[R, np.zeros((Dp, gdim))],[np.zeros((gdim, Dp)), np.eye(gdim)]])
    Tinv = np.block([[Rinv, np.zeros((Dp, gdim))],[np.zeros((gdim, Dp)), np.eye(gdim)]])
    
    K_aug_mono = Tinv @ K_aug_Q @ T


    try:
        L = (logm(K_aug_mono) / module.dt).real
    except Exception:
       
        jitter = 1e-6
        L = (logm(K_aug_mono + np.eye(K_aug_mono.shape[0]) * jitter) / module.dt).real
        
    L = _nan_guard_np(L)
    

    C_hat = L[:Dp, :module.d]
    
    eqs = format_equations(L, module.names, topk=12, thresh=1e-10)
    return L, C_hat, eqs

def main():
    filepath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(filepath)
    parser = argparse.ArgumentParser(description="CPU Inference for Residual Koopman")

    parser.add_argument("--dt", type=float, default=0.1, required=True)
    parser.add_argument("--rollout_B", type=int, default=30)
    parser.add_argument("--num_point", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--low_threshold", type=float, default=-4) 
    parser.add_argument("--high_threshold", type=float, default=4) 
    parser.add_argument("--out_dir_final", type=str, default="../results", help="path to save final files")

    parser.add_argument("--model_weights_path", type=str, default=None, help="path to model weights (auto-generated if not provided based on dt)")
    parser.add_argument("--config", type=str, default=None, help="path to config file (auto-generated if not provided based on dt)")
    parser.add_argument("--L", type=str, default=None, help="path to L_hat file (auto-generated if not provided based on dt)")
    parser.add_argument("--data", type=str, default=None, help="path to npy array of shape (N,T,d)")
    
    args, unknown = parser.parse_known_args()
    args = parser.parse_args()

    dt_str = f"{args.dt:.3f}"

    base_dir = "../data"  
    prefix = "cyc_HANDI_best"

    if args.config is None:
        args.config = os.path.join(base_dir, f"{prefix}_config_dt{dt_str}.json")
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {args.config}")
    else:
        config = {}

    for key, value in config.items():
        if hasattr(args, key):  
            continue  
        parser.add_argument(f'--{key}', type=type(value), default=value)

    for key, value in config['best_cfg'].items():
        if hasattr(args, key):  
            continue  
        parser.add_argument(f'--{key}', type=type(value), default=value)
    args = parser.parse_args()
    if args.config is None:
        args.config = os.path.join(base_dir, f"{prefix}_config_dt{dt_str}.json")

    if args.model_weights_path is None:
        args.model_weights_path = os.path.join(base_dir, f"{prefix}_model_dt{dt_str}.pt")

    if args.L is None:
        args.L = os.path.join(base_dir, f"{prefix}_L_hat_dt{dt_str}.npy")

    if args.data is None: 
        args.data = f"../data/cyc_train{int(args.dt*100)}.npy"  #..

    print(f"[INFO] Running on CPU only.")
    print(f"[INFO] Loading data from: {args.data}")
    print(f"[INFO] Loading model weights from: {args.model_weights_path}")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    Xall = np.load(args.data)
    print(f"[INFO] Data shape: {Xall.shape}")


    checkpoint = torch.load(args.model_weights_path, map_location=torch.device('cpu'))
    

    if "hyper_parameters" in checkpoint:
        hp = checkpoint["hyper_parameters"]
  
        for k, v in hp.items():
            if k == 'Xall_np': continue
            if hasattr(args, k):
                setattr(args, k, v)
                # print(f"[INFO] Loaded param {k}={v} from checkpoint")
    

    model = LightningResidualKoopman(
        Xall_np=Xall,
        dt=args.dt,
        polyorder=args.polyorder,
        g_dim=args.g_dim,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        lam_phi=args.lam_phi,
        lam_g=args.lam_g,
        lam_full=args.lam_full,
        rollout_B=args.rollout_B
    )



    state_dict = checkpoint.get("state_dict", checkpoint)
    
    model_state = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("model."): 
             name = name[6:]
   
        if name in model_state:
            new_state_dict[name] = v
        elif k in model_state:
            new_state_dict[k] = v
            
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"[WARN] Missing keys in loaded state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"[WARN] Unexpected keys in loaded state_dict: {unexpected_keys}")
    
    model.eval()
    print("[INFO] Model loaded successfully.")


    print("[INFO] Recomputing K_aug_Q on CPU...")
    K_new = model.compute_Kaug_full()
    model.K_aug_Q.copy_(K_new)
    print("[INFO] K_aug_Q updated.")


    print("[INFO] Extracting equations...")
    L, C_hat, eqs = extract_L_C_eqs(model)


    os.makedirs(args.out_dir_final, exist_ok=True)
    
    eq_file = os.path.join(args.out_dir_final, f"cyc_HANDI_best_equations_dt{args.dt:.3f}.txt")
    with open(eq_file, "w") as f:
        for line in eqs:
            f.write(line + "\n")
    
  

    print("\n===== Learned Equations =====")
    for line in eqs:
        print(line)
    print(f"\n[SUCCESS] Results saved to {args.out_dir_final}")

if __name__ == "__main__":
    main()