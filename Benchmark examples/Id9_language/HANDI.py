import os, math, json, argparse, gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from itertools import combinations_with_replacement
from scipy.linalg import logm

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

import torch.distributed as dist
import shutil

os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

try:
    import optuna
except Exception:
    optuna = None

def _safe_pinv(a, rcond=1e-10):
    return np.linalg.pinv(a, rcond=rcond)

def _nan_guard_np(x, clip=1e6):
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)

def _nan_guard_torch(t, clip=1e6):
    return torch.clamp(torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)

def seed_everything_cpu_only(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)  

class SeedInChildren(pl.Callback):
    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)
    def setup(self, trainer, pl_module, stage=None):
        import random
        s = self.base_seed + int(getattr(trainer, "global_rank", 0))
        random.seed(s); np.random.seed(s % (2**32 - 1)); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

def _is_lightning_child() -> bool:
    return os.environ.get("PL_TRAINER_LAUNCHED", "0") == "1"

def _visible_cuda_count():
    s = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if not s or s.strip() in ("", "-1"):
        return 0
    ids = [t for t in s.split(",") if t.strip() and t.strip() != "-1"]
    return len(ids)

def _pick_accel_and_devices(args):
    if getattr(args, "devices", 0) and args.devices > 0:
        cnt = _visible_cuda_count()
        use = min(args.devices, cnt) if cnt > 0 else args.devices
        return "gpu", max(1, use)
    return "cpu", 1

def build_poly(X, polyorder=3):
    N, d = X.shape
    basis, names = [], []
    for i in range(d):
        basis.append(X[:, i:i+1]); names.append(f'x({i+1})')
    basis.append(np.ones((N, 1))); names.append('1')
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

def compute_nrmse(C_hat: np.ndarray, C_true: np.ndarray) -> float:
    C_hat = np.asarray(C_hat, dtype=float)
    C_true = np.asarray(C_true, dtype=float)
    Dp = min(C_hat.shape[0], C_true.shape[0])
    d  = min(C_hat.shape[1], C_true.shape[1])
    C_hat = C_hat[:Dp, :d]; C_true = C_true[:Dp, :d]
    rmse = np.sqrt(np.mean((C_hat - C_true) ** 2))
    mask = (np.abs(C_true) > 0)
    if not np.any(mask):
        return float('inf')
    denom = np.mean(np.abs(C_true[mask]))
    return float(rmse / max(1e-12, denom))

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

def make_dataloaders(X_all_tensor, polyorder, R_mono2Q_tensor, win_S=5, batch_size=512,
                     num_workers=0, val_split=0.2, seed=0):
    X = X_all_tensor.numpy() if isinstance(X_all_tensor, torch.Tensor) else X_all_tensor
    N, T, d = X.shape
    S = max(2, min(int(win_S), T-1))

    X_flat = X.reshape(-1, d)
    Phi_all_mono, _ = build_poly(X_flat, polyorder=polyorder)
    R = R_mono2Q_tensor.numpy() if isinstance(R_mono2Q_tensor, torch.Tensor) else R_mono2Q_tensor
    Rinv = _safe_pinv(R, rcond=1e-10)
    Phi_all_Q = Phi_all_mono @ Rinv
    Dp = Phi_all_Q.shape[1]
    Phi_all_Q = Phi_all_Q.reshape(N, T, Dp)

    X0 = X[:, :-1, :].reshape(-1, d)
    X1 = X[:,  1:, :].reshape(-1, d)
    Phi_x_mono, _ = build_poly(X0, polyorder=polyorder)
    Phi_y_mono, _ = build_poly(X1, polyorder=polyorder)
    Qx = Phi_x_mono @ Rinv
    Qy = Phi_y_mono @ Rinv
    K_phi_Q = ridge(Qx, Qy, lam=3e-10)
    RQ_flat = Qy - Qx @ K_phi_Q
    RQ_all = RQ_flat.reshape(N, T-1, Dp)

    X_seq_list, Q_seq_list, RQ_first_list = [], [], []
    for n in range(N):
        for t in range(0, T - S + 1):
            X_seq_list.append(X[n, t:t+S, :])
            Q_seq_list.append(Phi_all_Q[n, t:t+S, :])
            RQ_first_list.append(RQ_all[n, t, :])

    X_seq = torch.from_numpy(np.stack(X_seq_list).astype(np.float32))
    Q_seq = torch.from_numpy(np.stack(Q_seq_list).astype(np.float32))
    RQ_first = torch.from_numpy(np.stack(RQ_first_list).astype(np.float32))

    ds = TensorDataset(X_seq, Q_seq, RQ_first)
    if val_split > 0.0:
        val_len = int(len(ds) * val_split)
        train_len = len(ds) - val_len
        g = torch.Generator().manual_seed(seed)
        ds_train, ds_val = random_split(ds, [train_len, val_len], generator=g)
    else:
        ds_train, ds_val = ds, None

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers>0))
    val_loader = None if ds_val is None else DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers>0))
    return train_loader, val_loader

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

class LightningResidualKoopman(pl.LightningModule):
    def __init__(self,
                 Xall_np,
                 dt=0.1,
                 polyorder=3,
                 g_dim=32,
                 width=128,
                 depth=2,
                 dropout=0.0,
                 epochs_A=150,
                 epochs_B=300,
                 lrA=2e-3,
                 lrB_start=3e-3,
                 lrB_end=3e-4,
                 decayA_last_frac=0.3,
                 decayA_min_ratio=0.2,
                 lam_phi=3e-10,
                 lam_g=3e-6,
                 lam_full=3e-6,
                 rollout_B=8,
                 beta_align=1e-2,
                 batch_size=512,
                 grad_clip=1.0):
        super().__init__()
        self.save_hyperparameters(ignore=['Xall_np'])
        self.dt = dt; self.polyorder = polyorder
        self.g_dim = g_dim; self.batch_size = batch_size
        self.epochs_A=epochs_A; self.epochs_B=epochs_B
        self.lrA=lrA; self.lrB_start=lrB_start; self.lrB_end=lrB_end
        self.decayA_last_frac=decayA_last_frac; self.decayA_min_ratio=decayA_min_ratio
        self.lam_phi=lam_phi; self.lam_g=lam_g; self.lam_full=lam_full
        self.rollout_B=rollout_B; self.beta_align=beta_align
        self.grad_clip = grad_clip

        Xall = Xall_np; N, T, d = Xall.shape; self.d=d
        self.win_S = max(2, self.rollout_B + 1)

        X0 = Xall[:, :-1, :].reshape(-1, d)
        X1 = Xall[:,  1:, :].reshape(-1, d)
        Phi_x_mono, names = build_poly(X0, polyorder=polyorder)
        Phi_y_mono, _    = build_poly(X1, polyorder=polyorder)
        self.names = names; self.Dp = Phi_x_mono.shape[1]

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
        self.Kg = nn.Parameter(1e-2 * torch.randn(g_dim, self.Dp))
        self.register_buffer("Kg_closed", torch.zeros(g_dim, self.Dp), persistent=False)

        self.register_buffer("K_aug_Q", torch.eye(self.Dp + self.g_dim), persistent=False)

    @property
    def device_(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def fwd_all(self, X_cpu, bs=65536):
        outs=[]
        dev = self.device_
        for i in range(0, X_cpu.shape[0], bs):
            xb = X_cpu[i:i+bs].to(dev, non_blocking=True)
            outs.append(self.model(xb))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def _world_info(self):
        is_dist = dist.is_available() and dist.is_initialized()
        if not is_dist:
            return False, 1, 0
        return True, dist.get_world_size(), dist.get_rank()

    @torch.no_grad()
    def _split_range(self, total):
        is_dist, world_size, rank = self._world_info()
        if not is_dist:
            return 0, total
        per = (total + world_size - 1) // world_size
        s = rank * per
        e = min(total, (rank + 1) * per)
        return s, e

    @torch.no_grad()
    def compute_Kg_full_closed(self, alpha=1.0):
        dev = self.device_
        Gx_raw = self.fwd_all(self.X0_cpu)
        Qx = self.Qx_cpu.to(dev, non_blocking=True)
        RQ = self.RQ_cpu.to(dev, non_blocking=True)
        GT = Qx.T @ Gx_raw
        Gx_perp = Gx_raw - alpha*(Qx @ GT)
        GTG = Gx_perp.T @ Gx_perp + self.lam_g * torch.eye(self.g_dim, device=dev)
        GTR = Gx_perp.T @ RQ
        return _nan_guard_torch(torch.linalg.solve(GTG, GTR))

    @torch.no_grad()
    def compute_Kg_full_closed_dist(self, alpha=1.0):
        dev = self.device_
        is_dist, _, _ = self._world_info()
        total = self.X0_cpu.shape[0]
        s, e = self._split_range(total)

        Qx_local = self.Qx_cpu[s:e].to(dev, non_blocking=True)          # [m, Dp]
        RQ_local = self.RQ_cpu[s:e].to(dev, non_blocking=True)          # [m, Dp]
        Gx_local = self.fwd_all(self.X0_cpu[s:e])                       # [m, g]

        GT_local = Qx_local.T @ Gx_local                                 # [Dp, g]
        if is_dist:
            dist.all_reduce(GT_local, op=dist.ReduceOp.SUM)
        GT_global = GT_local

        Gx_perp_local = Gx_local - alpha * (Qx_local @ GT_global)        # [m, g]
        GTG_local = Gx_perp_local.T @ Gx_perp_local                       # [g, g]
        GTR_local = Gx_perp_local.T @ RQ_local                            # [g, Dp]
        if is_dist:
            dist.all_reduce(GTG_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(GTR_local, op=dist.ReduceOp.SUM)

        GTG = GTG_local + self.lam_g * torch.eye(self.g_dim, device=dev)
        Kg = torch.linalg.solve(GTG, GTR_local)
        return _nan_guard_torch(Kg)

    @torch.no_grad()
    def compute_Kaug_full(self):
        dev = self.device_
        Qx = self.Qx_cpu.to(dev, non_blocking=True).to(torch.float64)
        Qy = self.Qy_cpu.to(dev, non_blocking=True).to(torch.float64)
        Gx_raw = self.fwd_all(self.X0_cpu).to(torch.float64)
        Gy_raw = self.fwd_all(self.X1_cpu).to(torch.float64)

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
    def compute_Kaug_full_dist(self):
        dev = self.device_
        is_dist, _, _ = self._world_info()
        total = self.X0_cpu.shape[0]
        s, e = self._split_range(total)

        Qx = self.Qx_cpu[s:e].to(dev, non_blocking=True, dtype=torch.float64)
        Qy = self.Qy_cpu[s:e].to(dev, non_blocking=True, dtype=torch.float64)
        Gx_raw = self.fwd_all(self.X0_cpu[s:e]).to(torch.float64)
        Gy_raw = self.fwd_all(self.X1_cpu[s:e]).to(torch.float64)

        GTx_local = Qx.T @ Gx_raw
        GTy_local = Qy.T @ Gy_raw
        if is_dist:
            dist.all_reduce(GTx_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(GTy_local, op=dist.ReduceOp.SUM)

        Gx_perp = Gx_raw - Qx @ GTx_local
        Gy_perp = Gy_raw - Qy @ GTy_local

        Zx = torch.cat([Qx, Gx_perp], dim=1)
        Zy = torch.cat([Qy, Gy_perp], dim=1)
        D = Zx.shape[1]

        Sxx_local = Zx.T @ Zx
        Sxy_local = Zx.T @ Zy
        if is_dist:
            dist.all_reduce(Sxx_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(Sxy_local, op=dist.ReduceOp.SUM)

        lam = float(max(self.lam_full, 1e-4))
        I = torch.eye(D, device=dev, dtype=torch.float64)
        Sxx = Sxx_local + lam * I

        try:
            K = torch.linalg.solve(Sxx, Sxy_local)
        except RuntimeError:
            diag_mean = torch.mean(torch.diag(Sxx)).clamp(min=1e-12)
            lam2 = max(lam, float(1e-3 * diag_mean))
            Sxx2 = Sxx_local + lam2 * I
            try:
                K = torch.linalg.solve(Sxx2, Sxy_local)
            except RuntimeError:
                s = math.sqrt(lam2)
                Zxa = torch.cat([Zx, s * I], dim=0)
                Zya = torch.cat([Zy, torch.zeros(D, D, device=dev, dtype=torch.float64)], dim=0)
                K = torch.linalg.lstsq(Zxa, Zya, rcond=None).solution

        K = K.to(torch.float32)
        return _nan_guard_torch(K)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(list(self.model.parameters()) + [self.Kg],
                                lr=self.lrA, weight_decay=1e-5)
        lastN = int(self.decayA_last_frac * self.epochs_A)
        start_decay = self.epochs_A - max(1, lastN)
        def lr_lambda(epoch):
            if epoch < self.epochs_A:
                if epoch >= start_decay:
                    progress = (epoch - start_decay) / max(1, lastN)
                    scale = self.decayA_min_ratio + (1 - self.decayA_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
                    return scale
                else:
                    return 1.0
            else:
                t = epoch - self.epochs_A
                T = max(1, self.epochs_B)
                cosw = 0.5 * (1 + math.cos(math.pi * t / T))
                return (self.lrB_end + (self.lrB_start - self.lrB_end) * cosw) / max(self.lrA, 1e-12)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        inA = epoch < self.epochs_A
        was_train = self.training
        self.model.eval()
        try:
            is_dist = dist.is_available() and dist.is_initialized()
            if inA:
                alpha = min(1.0, epoch / max(1, self.epochs_A // 3))
                Kg_new = self.compute_Kg_full_closed_dist(alpha=alpha) if is_dist else self.compute_Kg_full_closed(alpha=alpha)
                self.Kg_closed.copy_(Kg_new)
            else:
                K_new = self.compute_Kaug_full_dist() if is_dist else self.compute_Kaug_full()
                self.K_aug_Q.copy_(K_new)
        finally:
            if was_train:
                self.model.train()

    def training_step(self, batch, batch_idx):
        device = self.device_
        X_seq, Q_seq, RQ_first = batch
        X_seq = X_seq.to(device); Q_seq = Q_seq.to(device); RQ_first = RQ_first.to(device)
        B, S, d = X_seq.shape
        Dp = Q_seq.shape[-1]
        I_Dp = torch.eye(Dp, device=device); eps = 1e-6

        epoch = self.current_epoch
        inA = epoch < self.epochs_A

        if inA:
            X0 = X_seq[:, 0, :]
            Q0 = Q_seq[:, 0, :]
            G_raw = self.model(X0)

            alpha = min(1.0, epoch / max(1, self.epochs_A // 3))
            XtX = Q0.T @ Q0; XtY = Q0.T @ G_raw
            C = torch.linalg.solve(XtX + eps * I_Dp, XtY)
            proj = Q0 @ C
            G_perp = G_raw - alpha * proj
            pred = G_perp @ self.Kg

            loss_main = torch.mean((pred - RQ_first) ** 2)
            loss_align = self.beta_align * torch.mean((self.Kg - self.Kg_closed) ** 2)
            loss = loss_main + loss_align

            self.log_dict({"train/loss": loss, "train/main": loss_main, "train/align": loss_align},
                          prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        else:
            X_flat = X_seq.reshape(-1, d)
            G_flat = self.model(X_flat)
            G_seq = G_flat.reshape(B, S, -1)

            G_perp_list=[]
            for tstep in range(S):
                Qt = Q_seq[:, tstep, :]
                Gt = G_seq[:, tstep, :]
                XtX = Qt.T @ Qt; XtY = Qt.T @ Gt
                C = torch.linalg.solve(XtX + eps * I_Dp, XtY)
                proj = Qt @ C
                G_perp_list.append(Gt - proj)
            G_perp_seq = torch.stack(G_perp_list, dim=1)

            Z_seq = torch.cat([Q_seq, G_perp_seq], dim=-1)
            Z0 = Z_seq[:, 0, :]
            K = self.K_aug_Q
            pred1 = Z0 @ K
            loss1 = torch.mean((pred1 - Z_seq[:, 1, :]) ** 2)

            steps = min(max(1, self.rollout_B), S - 1)
            Kt = torch.eye(self.Dp + self.g_dim, device=device)
            loss_roll = 0.0
            for k in range(1, steps + 1):
                Kt = Kt @ K
                pred_k = Z0 @ Kt
                loss_roll = loss_roll + torch.mean((pred_k - Z_seq[:, k, :]) ** 2)
            loss_roll_avg = loss_roll / steps
            loss = loss1 + 0.3 * loss_roll_avg   ##

            self.log_dict({"train/loss": loss, "train/step1": loss1, "train/rollout_avg": loss_roll_avg},
                          prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        device = self.device_
        X_seq, Q_seq, _ = batch
        X_seq = X_seq.to(device); Q_seq = Q_seq.to(device)
        B, S, d = X_seq.shape
        Dp = Q_seq.shape[-1]
        I_Dp = torch.eye(Dp, device=device); eps = 1e-6

        X_flat = X_seq.reshape(-1, d)
        G_flat = self.model(X_flat)
        G_seq = G_flat.reshape(B, S, -1)

        G_perp_list=[]
        for tstep in range(S):
            Qt = Q_seq[:, tstep, :]
            Gt = G_seq[:, tstep, :]
            XtX = Qt.T @ Qt; XtY = Qt.T @ Gt
            C = torch.linalg.solve(XtX + eps * I_Dp, XtY)
            proj = Qt @ C
            G_perp_list.append(Gt - proj)
        G_perp_seq = torch.stack(G_perp_list, dim=1)

        Z_seq = torch.cat([Q_seq, G_perp_seq], dim=-1)
        Z0 = Z_seq[:, 0, :]
        K = self.K_aug_Q
        pred1 = Z0 @ K
        loss1 = torch.mean((pred1 - Z_seq[:, 1, :]) ** 2)

        steps = min(max(1, self.rollout_B), S - 1)
        Kt = torch.eye(self.Dp + self.g_dim, device=device)
        loss_roll = 0.0
        for k in range(1, steps + 1):
            Kt = Kt @ K
            pred_k = Z0 @ Kt
            loss_roll = loss_roll + torch.mean((pred_k - Z_seq[:, k, :]) ** 2)
        loss_roll_avg = loss_roll / steps

        loss = loss1 + 0.3 * loss_roll_avg
        self.log_dict({"val/loss": loss, "val/step1": loss1, "val/rollout_avg": loss_roll_avg},
                      prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

@torch.no_grad()
def extract_L_C_eqs(module: LightningResidualKoopman):
    K_aug_Q = module.K_aug_Q.detach().cpu().numpy()
    Dp = module.Dp; gdim = module.g_dim
    R = module.R_mono2Q.detach().cpu().numpy()
    Rinv = module.Rinv_Q2mono.detach().cpu().numpy()

    K_aug_Q = _nan_guard_np(K_aug_Q)
    R = _nan_guard_np(R); Rinv = _nan_guard_np(Rinv)

    T = np.block([[R, np.zeros((Dp, gdim))],[np.zeros((gdim, Dp)), np.eye(gdim)]])
    Tinv = np.block([[Rinv, np.zeros((Dp, gdim))],[np.zeros((gdim, Dp)), np.eye(gdim)]])
    K_aug_mono = Tinv @ K_aug_Q @ T

    L = (logm(K_aug_mono)/module.dt).real
    C_hat = L[:Dp, :module.d]
    eqs = format_equations(L, module.names, topk=12, thresh=1e-10)
    return L, C_hat, eqs

@torch.no_grad()
def evaluate_equation_rollout_mse_ivp(module, val_loader, fine_h: float = 1e-3,
                                      eig_eps: float = 1e-8, verbose: bool = False,
                                      top_r: int = 0):
    device = torch.device("cpu")
    K_aug_Q = _nan_guard_np(module.K_aug_Q.detach().cpu().numpy())
    Dp, gdim = module.Dp, module.g_dim
    R = _nan_guard_np(module.R_mono2Q.detach().cpu().numpy())
    Rinv = _nan_guard_np(module.Rinv_Q2mono.detach().cpu().numpy())

    T = np.block([[R, np.zeros((Dp, gdim))],[np.zeros((gdim, Dp)), np.eye(gdim)]])
    Tinv = np.block([[Rinv, np.zeros((Dp, gdim))],[np.zeros((gdim, Dp)), np.eye(gdim)]])
    K_aug_mono = _nan_guard_np(Tinv @ K_aug_Q @ T)

    if not np.isfinite(K_aug_mono).all():
        return 1e9

    w, V = np.linalg.eig(K_aug_mono)
    w = np.where(np.abs(w) < eig_eps, eig_eps, w)
    K_aug_mono = (V @ np.diag(w) @ np.linalg.inv(V)).real

    try:
        L = (logm(K_aug_mono) / float(module.dt)).real
    except Exception:
        try:
            jitter = 1e-6
            L = (logm(K_aug_mono + np.eye(K_aug_mono.shape[0]) * jitter) / float(module.dt)).real
        except Exception:
            return 1e9

    C_np = L[:Dp, :module.d]
    C = torch.from_numpy(C_np.astype(np.float64)).to(device)   # [Dp, d]
    if isinstance(top_r, (int, np.integer)) and top_r > 0:
        flat_abs = torch.abs(C).flatten()
        r = int(min(top_r, flat_abs.numel()))
        if r > 0:
            top_idx = torch.topk(flat_abs, r, largest=True).indices
            mask = torch.zeros_like(flat_abs, dtype=torch.bool)
            mask[top_idx] = True
            mask = mask.view_as(C)
            C = C * mask  

    names = module.names; d = module.d
    mono_idx = []
    for j, nm in enumerate(names):
        if j < d:
            mono_idx.append([j])
        else:
            if nm == '1':
                mono_idx.append([])
            else:
                parts = nm.split('*')
                mono_idx.append([int(p[2:-1]) - 1 for p in parts])

    def build_phi_torch(x):
        B = x.shape[0]
        feats = [x]
        high = []
        for j in range(d, len(mono_idx)):
            idxs = mono_idx[j]
            v = torch.ones(B, 1, device=device, dtype=x.dtype)
            for ii in idxs:
                v = v * x[:, ii:ii+1]
            high.append(v)
        if high:
            high = torch.cat(high, dim=1)
            Phi = torch.cat([feats[0], high], 1)
        else:
            Phi = feats[0]
        return Phi

    dt = float(module.dt); h = float(fine_h)
    steps_per_dt = int(round(dt / h))

    def f_cont(x):
        return build_phi_torch(x) @ C

    def rk4_step(x, hstep):
        k1 = f_cont(x)
        k2 = f_cont(x + 0.5*hstep*k1)
        k3 = f_cont(x + 0.5*hstep*k2)
        k4 = f_cont(x + hstep*k3)
        return x + (hstep/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    mse_sum, n_used = 0.0, 0
    for batch in val_loader:
        X_seq, _, _ = batch
        X_seq = X_seq.to(device=device, dtype=torch.float64, non_blocking=False)
        B, S, _ = X_seq.shape
        steps = S-1

        x_pred = X_seq[:, 0, :]
        mse_b = torch.zeros([], device=device, dtype=torch.float64)

        fine_steps_total = steps * steps_per_dt
        boundary_stride = steps_per_dt
        k_next = 1
        for t in range(1, fine_steps_total + 1):
            x_pred = rk4_step(x_pred, h)
            if t % boundary_stride == 0:
                x_true = X_seq[:, k_next, :]
                diff = x_pred - x_true
                mse_b = mse_b + torch.mean(diff*diff)
                k_next += 1
                if k_next > steps:
                    break

        mse_b = mse_b / steps
        mse_sum += float(mse_b.detach().cpu().item())
        n_used += 1

    if n_used == 0:
        return float("nan")
    return mse_sum / n_used

@torch.no_grad()
def evaluate_koopman_rollout_mse_disc(module, val_loader, eig_eps: float = 1e-8, verbose: bool = False):
    device = torch.device("cpu")
    Dp, gdim = module.Dp, module.g_dim

    KQ   = _nan_guard_np(module.K_aug_Q.detach().cpu().numpy())
    R    = _nan_guard_np(module.R_mono2Q.detach().cpu().numpy())
    Rinv = _nan_guard_np(module.Rinv_Q2mono.detach().cpu().numpy())
    T    = np.block([[R, np.zeros((Dp, gdim))],
                     [np.zeros((gdim, Dp)), np.eye(gdim)]])
    Tinv = np.block([[Rinv, np.zeros((Dp, gdim))],
                     [np.zeros((gdim, Dp)), np.eye(gdim)]])
    Kmono = _nan_guard_np(Tinv @ KQ @ T)
    if not np.isfinite(Kmono).all():
        return 1e9
    Kmono_t = torch.from_numpy(Kmono).to(device=device, dtype=torch.float64)

    names = module.names
    d = module.d
    mono_idx = []
    for j, nm in enumerate(names):
        if j < d:
            mono_idx.append([j])
        else:
            if nm == '1':
                mono_idx.append([])
            else:
                parts = nm.split('*')
                mono_idx.append([int(p[2:-1]) - 1 for p in parts])

    def build_phi_torch(x):  # x: [B,d] -> Phi_mono: [B,Dp]
        B = x.shape[0]
        feats = [x]
        high = []
        for j in range(d, len(mono_idx)):
            idxs = mono_idx[j]
            v = torch.ones(B, 1, device=device, dtype=x.dtype)
            for ii in idxs:
                v = v * x[:, ii:ii+1]
            high.append(v)
        return torch.cat([feats[0]] + high, dim=1) if high else feats[0]

    eps = 1e-6
    I_Dp = torch.eye(Dp, device=device, dtype=torch.float64)

    mse_sum, n_batches = 0.0, 0
    for X_seq, Q_seq, _ in val_loader:
        X_seq = X_seq.to(device=device, dtype=torch.float64)
        Q_seq = Q_seq.to(device=device, dtype=torch.float64)
        B, S, _ = X_seq.shape
        if S <= 1:
            continue  
        X0 = X_seq[:, 0, :]
        Phi0 = build_phi_torch(X0)
        dev_model = module.device_
        G_raw0 = module.model(X0.to(dev_model).float()).to(device=device, dtype=torch.float64)
        Qt = Q_seq[:, 0, :]
        XtX = Qt.T @ Qt
        XtY = Qt.T @ G_raw0
        try:
            C = torch.linalg.solve(XtX + eps*I_Dp, XtY)
        except RuntimeError:
            C = torch.linalg.lstsq(XtX + eps*I_Dp, XtY, rcond=None).solution
        G_perp0 = G_raw0 - Qt @ C
        Zk = torch.cat([Phi0, G_perp0], dim=1).to(torch.float64)

        mse_b = 0.0
        for k in range(1, S):
            Zk = Zk @ Kmono_t
            x_pred = Zk[:, :d].to(torch.float32)
            x_true = X_seq[:, k, :].to(torch.float32)
            mse_b += torch.mean((x_pred - x_true) ** 2).item()
        mse_b /= max(1, S-1)

        mse_sum += float(mse_b)
        n_batches += 1

    if n_batches == 0:
        return float("nan")
    return mse_sum / n_batches

def export_artifacts(module, args, val_loader=None, outdir=None):
    outdir = outdir or args.outdir
    os.makedirs(outdir, exist_ok=True)

    metric_mse = None
    if val_loader is not None:
        try:
            metric_mse = evaluate_equation_rollout_mse_ivp(module, val_loader, verbose=False, top_r=getattr(args, "top_r", 0))
        except Exception as e:
            print(f"[eval WARN] {e}")

    L, C_hat, eqs = extract_L_C_eqs(module)

    nrmse_val = None
    if getattr(args, "true_coeff_json", ""):
        try:
            C_true = np.array(json.loads(args.true_coeff_json), dtype=float)
            nrmse_val = compute_nrmse(C_hat, C_true)
        except Exception as e:
            print(f"[NRMSE WARN] true_coeff_json：{e}")

    with open(os.path.join(outdir, "learned_equations.txt"), "w") as f:
        for line in eqs:
            f.write(line + "\n")

    best_pack = {"best_cfg": vars(args), "dt": float(args.dt)}
    if nrmse_val is not None and np.isfinite(nrmse_val):
        best_pack["nrmse_at_best"] = float(nrmse_val)
    if metric_mse is not None and np.isfinite(metric_mse):
        best_pack["val_rollout_mse"] = float(metric_mse)
    with open(os.path.join(outdir, f"best_config_dt{args.dt:.3f}.json"), "w") as f:
        json.dump(best_pack, f, indent=2)

    np.save(os.path.join(outdir, f"best_C_hat_dt{args.dt:.3f}.npy"), C_hat)
    with open(os.path.join(outdir, f"best_equations_dt{args.dt:.3f}.txt"), "w") as f:
        for line in eqs:
            f.write(line + "\n")

    torch.save({
        "state_dict": module.state_dict(),
        "config": vars(args),
        "dt": float(module.dt),
        "Dp": int(module.Dp),
        "d": int(module.d),
        "names": module.names,
    }, os.path.join(outdir, f"best_model_dt{args.dt:.3f}.pt"))

    if nrmse_val is not None and np.isfinite(nrmse_val):
        with open(os.path.join(outdir, f"nrmse_dt{args.dt:.3f}.txt"), "w") as f:
            f.write(str(float(nrmse_val)) + "\n")
    if metric_mse is not None and np.isfinite(metric_mse):
        with open(os.path.join(outdir, f"metric_mse_dt{args.dt:.3f}.txt"), "w") as f:
            f.write(str(float(metric_mse)) + "\n")

    print("\n===== Learned equations (top terms) =====")
    for line in eqs:
        print(line)
    if nrmse_val is not None:
        print(f"\n[final] NRMSE = {nrmse_val:.6g}")
    if metric_mse is not None:
        print(f"[final] Val rollout MSE = {metric_mse:.6g}")
    print(f"\nSaved to {outdir}")

def fit_once(args, Xall):
    model = LightningResidualKoopman(
        Xall_np=Xall,
        dt=args.dt, polyorder=args.polyorder,
        g_dim=args.g_dim, width=args.width, depth=args.depth, dropout=args.dropout,
        epochs_A=args.epochs_A, epochs_B=args.epochs_B,
        lrA=args.lrA, lrB_start=args.lrB_start, lrB_end=args.lrB_end,
        decayA_last_frac=args.decayA_last_frac, decayA_min_ratio=args.decayA_min_ratio,
        lam_phi=args.lam_phi, lam_g=args.lam_g, lam_full=args.lam_full,
        rollout_B=args.rollout_B, beta_align=args.beta_align,
        batch_size=args.batch_size, grad_clip=args.grad_clip
    )

    train_loader, val_loader = make_dataloaders(
        model.X_all_cpu,
        polyorder=args.polyorder,
        R_mono2Q_tensor=model.R_mono2Q,
        win_S=max(2, args.rollout_B+1),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed
    )

    total_epochs = args.epochs_A + args.epochs_B
    max_epochs = total_epochs if args.max_epochs <= 0 else args.max_epochs

    accelerator, devices_use = _pick_accel_and_devices(args)
    strategy = DDPStrategy(find_unused_parameters=True,
                           process_group_backend="nccl") if devices_use > 1 else "auto"
    logger = CSVLogger(save_dir=args.outdir, name="logs")

    trainer = pl.Trainer(
        accelerator=accelerator, devices=devices_use, strategy=strategy,
        max_epochs=max_epochs, precision=args.precision,
        gradient_clip_val=args.grad_clip,
        logger=logger,
        callbacks=[SeedInChildren(args.seed)],
        enable_model_summary=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0
    )

    print(f"[FINAL HPs] dt={args.dt:g}")
    print(f"  rollout_B={args.rollout_B}  epochs_A={args.epochs_A}  epochs_B={args.epochs_B}")
    print(f"  lrA={args.lrA:g}  lrB_start={args.lrB_start:g}  lrB_end={args.lrB_end:g}")
    print(f"  lam_phi={args.lam_phi:g}  lam_g={args.lam_g:g}  lam_full={args.lam_full:g}  beta_align={args.beta_align:g}  dropout={args.dropout:g}")
    print(f"  devices={devices_use}  accelerator={accelerator}")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer, model, val_loader

def train_and_export(args):
    seed_everything_cpu_only(args.seed)
    Xall = np.load(args.data)
    
    trainer, model, val_loader = fit_once(args, Xall)
    if getattr(trainer, "is_global_zero", True):
        export_artifacts(model, args, val_loader=val_loader, outdir=args.outdir)
    return model


def run_tuning(args):
    if optuna is None:
        raise SystemExit("Optuna： pip install optuna")
    if _is_lightning_child():
        return

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception as _e:
            pass

    seed_everything_cpu_only(args.seed)
    Xall = np.load(args.data)

    base = argparse.Namespace(**vars(args))
    save_trials = bool(args.tune_save_trials)
    tuning_root = os.path.join(args.outdir, f"tuning_dt{args.dt:.3f}")
    if save_trials:
        os.makedirs(tuning_root, exist_ok=True)

    best_metric = float("inf")
    best_params = None

    pruner = optuna.pruners.NopPruner()
    storage = None  
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=8, seed=args.seed)
    study = optuna.create_study(direction="minimize",
                                study_name=f"edmd_dt{args.dt:.3f}",
                                storage=storage, sampler=sampler, pruner=pruner)

    def objective(trial: "optuna.trial.Trial"):
        cfg = argparse.Namespace(**vars(base))

        cfg.rollout_B = base.rollout_B
        cfg.width     = trial.suggest_int("width", max(64, base.width-64), base.width+64, step=32)
        cfg.epochs_A     = trial.suggest_int("epochs_A", max(50, base.epochs_A-50), base.epochs_A+50, step=10)
        cfg.epochs_B     = trial.suggest_int("epochs_B", max(100, base.epochs_B-100), base.epochs_B+100, step=20)
        cfg.depth     = trial.suggest_int("depth", max(1, base.depth-1), base.depth+1, step=1)
        cfg.dropout   = trial.suggest_float("dropout", max(0.0, base.dropout-0.25), min(0.5, base.dropout+0.25), step=0.05)
        cfg.lrA       = base.lrA
        cfg.lrB_start = base.lrB_start
        cfg.lrB_end   = base.lrB_end
        cfg.decayA_last_frac = base.decayA_last_frac
        cfg.decayA_min_ratio = base.decayA_min_ratio
        cfg.lam_phi   = base.lam_phi
        cfg.lam_g     = trial.suggest_float("lam_g",   max(1e-12, base.lam_g/10),     base.lam_g*10,   log=True)
        cfg.lam_full  = trial.suggest_float("lam_full",max(1e-12, base.lam_full/10),  base.lam_full*10,log=True)
        cfg.beta_align= trial.suggest_float("beta_align",max(1e-12, base.beta_align/10),  base.beta_align*10,log=True)

        trial_dir = os.path.join(tuning_root, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        cfg.outdir = trial_dir
        cfg.tune = False
        cfg.devices = base.devices

        try:
            trainer, module, val_loader = fit_once(cfg, Xall)  
            metric = evaluate_equation_rollout_mse_ivp(module, val_loader, verbose=False, top_r=getattr(args, "top_r", 0))
            if not np.isfinite(metric):
                metric = 1e9

            if getattr(trainer, "is_global_zero", True):
                export_artifacts(module, cfg, val_loader=val_loader, outdir=trial_dir)

            nrmse_val = None
            if getattr(base, "true_coeff_json", ""):
                try:
                    C_true = np.array(json.loads(base.true_coeff_json), dtype=float)
                    C_hat = np.load(os.path.join(trial_dir, f"best_C_hat_dt{base.dt:.3f}.npy"))
                    nrmse_val = compute_nrmse(C_hat, C_true)
                except Exception as e:
                    print(f"[trial {trial.number}] NRMSE compute error: {e}")

            nonlocal best_metric, best_params
            if metric < best_metric:
                best_metric = float(metric)
                best_params = dict(vars(cfg))
                pack = {"best_cfg": best_params, "dt": base.dt, "score": best_metric}
                if nrmse_val is not None and np.isfinite(nrmse_val):
                    pack["nrmse_at_best"] = float(nrmse_val)
                with open(os.path.join(args.outdir, f"best_config_dt{base.dt:.3f}.json"), "w") as f:
                    json.dump(pack, f, indent=2)

                def _cp(name):
                    src = os.path.join(trial_dir, name)
                    dst = os.path.join(args.outdir, name)
                    if os.path.exists(src):
                        shutil.copyfile(src, dst)
                _cp(f"best_C_hat_dt{base.dt:.3f}.npy")
                _cp(f"best_equations_dt{base.dt:.3f}.txt")
                _cp(f"best_model_dt{base.dt:.3f}.pt")
                _cp(f"metric_mse_dt{base.dt:.3f}.txt")

                print(f"[best] improved to {best_metric:.6g}; best_* updated.")

            print(f"[trial {trial.number}] metric={metric}")
            del trainer, module, val_loader
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return float(metric)

        except Exception as e:
            print(f"[trial {trial.number}] ERROR: {e}")
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return 1e9

    study.optimize(objective,
                   n_trials=args.n_trials,
                   timeout=args.tune_timeout or None,
                   n_jobs=1,
                   gc_after_trial=True)

    if len(study.trials) == 0 or study.best_trial is None:
        print("[optuna] No trials completed, fall back to base training.")
        return train_and_export(args)

    print("\n[optuna] Best value=%.6g" % study.best_value)
    print("[optuna] Best params:", study.best_params)
    for k, v in study.best_params.items():
        setattr(args, k, v)

    return train_and_export(args)

def main():
    p = argparse.ArgumentParser()

    # data & dictionary
    p.add_argument("--data", type=str, default="id9_downsample_6.npy", help="path to npy array of shape (N,T,d)")
    p.add_argument("--dt", type=float, default=0.3)
    p.add_argument("--polyorder", type=int, default=1)  # 4-14

    # model
    p.add_argument("--g_dim", type=int, default=63)  #4-51
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)

    # training
    p.add_argument("--epochs_A", type=int, default=100)
    p.add_argument("--epochs_B", type=int, default=200)
    p.add_argument("--lrA", type=float, default=2e-3)
    p.add_argument("--lrB_start", type=float, default=3e-3)
    p.add_argument("--lrB_end", type=float, default=3e-4)
    p.add_argument("--decayA_last_frac", type=float, default=0.3)
    p.add_argument("--decayA_min_ratio", type=float, default=0.2)
    p.add_argument("--lam_phi", type=float, default=3e-10)
    p.add_argument("--lam_g", type=float, default=3e-6)
    p.add_argument("--lam_full", type=float, default=3e-6)
    p.add_argument("--rollout_B", type=int, default=5)
    p.add_argument("--beta_align", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=15)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_split", type=float, default=0.2)

    # misc
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--outdir", type=str, default="./ploy_1_run1")
    p.add_argument("--true_coeff_json", type=str,
                   default="",
                   help="True monomial coefficient matrix JSON string; leave empty to skip NRMSE calculation")

    # lightning runtime
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs; >1 enables DDP")
    p.add_argument("--precision", type=str, default="32-true", help="Precision: 32-true/16-mixed/bf16-mixed etc.")
    p.add_argument("--max_epochs", type=int, default=-1, help="Override epochs_A+epochs_B; -1 means use A+B")

    # tuning
    p.add_argument("--tune", dest="tune", action="store_true", default=True)
    p.add_argument("--no-tune", dest="tune", action="store_false")
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--tune_timeout", type=int, default=0)
    p.add_argument("--tune_save_trials", type=int, default=0, help="Whether to save each trial to tuning_dt.../trial_xxxx (1=yes, 0=no)")
    p.add_argument("--hp_json", type=str, default="", help="JSON dict to override hyperparameters for this training run")
    p.add_argument("--top_r", type=int, default=4,
               help="Number of coefficients to keep during IVP evaluation (across all states), <=0 means no clipping")

    args = p.parse_args()

    if args.hp_json:
        try:
            hp = json.loads(args.hp_json)
            for k, v in hp.items():
                setattr(args, k, v)
        except Exception as e:
            print(f"[hp_json WARN] Parse failed: {e}")

    if args.max_epochs > 0:
        args.epochs_B = max(0, args.max_epochs - args.epochs_A)

    if args.tune:
        run_tuning(args)
    else:
        train_and_export(args)

if __name__ == "__main__":
    if not _is_lightning_child():
        main()
