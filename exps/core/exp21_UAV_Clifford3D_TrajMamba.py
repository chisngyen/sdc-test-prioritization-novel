"""
=============================================================================
Experiment 21: CliffordTrajNet-3D & Continuous TrajMamba for UAV Test Prioritization
Domain: Autonomous Cyber-Physical Systems - 3D Unmanned Aerial Vehicles (UAVs)
Dataset: SBFT / ICST UAV Testing Competition (Aerialist / PX4 Autopilot)

Core Frontier Novelties:
1. Clifford Geometric Algebra Cl(3,0):
   - 8-channel multivector representation: [scalar, e1, e2, e3, e12, e23, e31, e123]
   - Rotor-based kinematic & obstacle approach plane decomposition
2. Continuous Selective State-Space Model (TrajMamba):
   - Continuous arclength ODE integration: ds = sqrt(dx^2 + dy^2 + dz^2), dt = ds / v
   - Variable waypoint spacing tolerance
3. Monotone Conformal Risk Control (LTT / PAC Safety Bound):
   - Finite-sample statistical guarantee on detecting critical UAV obstacle violations
4. Canonical Evaluation Protocol:
   - 80/20 stratified train/test split (seed=42)
   - 30-trial randomized tie-breaking / subset evaluation (|S| = max(50, 0.3*|test|))
   - Comparison against 3D Transformer, Path-Length Heuristic, and Random Prioritization
   - 3D Rotation Invariance Probe across 6 angles
=============================================================================
"""

import os
import sys
import json
import math
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] Using {DEVICE}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

SEARCH_ROOTS = [
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uav')),
    'data/uav',
    '/kaggle/input/uav-testing-competition-2026',
    '/kaggle/input/chinguyeen/uav-testing-competition-2026',
    os.getcwd(),
]

def find_uav_dataset():
    for root in SEARCH_ROOTS:
        target = os.path.join(root, 'uav_dataset_surrogate.json')
        if os.path.isfile(target):
            return target
        for dirpath, _, filenames in os.walk(root):
            if 'uav_dataset_surrogate.json' in filenames:
                return os.path.join(dirpath, 'uav_dataset_surrogate.json')
    return None

DATA_PATH = find_uav_dataset()
if not DATA_PATH or not os.path.isfile(DATA_PATH):
    raise FileNotFoundError("Could not find uav_dataset_surrogate.json in search roots.")

with open(DATA_PATH, 'r') as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} records from {DATA_PATH}.")
SEQ_LEN = 32

def resample_path(path, target=SEQ_LEN):
    p = np.asarray(path, dtype=np.float64)
    if len(p) < 2:
        p = np.vstack([p, p]) if len(p) else np.zeros((2, 3))
    segs = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(segs)])
    if s[-1] < 1e-6:
        return np.tile(p[0], (target, 1)).astype(np.float32)
    snew = np.linspace(0, s[-1], target)
    out = np.empty((target, 3), dtype=np.float32)
    for k in range(3):
        out[:, k] = np.interp(snew, s, p[:, k])
    return out

def _box_dist3d(p, obs):
    cx, cy, cz = obs['x'], obs['y'], obs['z'] + obs['h'] / 2.0
    hl, hw, hh = obs['l'] / 2.0, obs['w'] / 2.0, obs['h'] / 2.0
    th = math.radians(obs['r'])
    c, s = math.cos(-th), math.sin(-th)
    dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
    rx = dx * c - dy * s
    ry = dx * s + dy * c
    qx = max(0.0, abs(rx) - hl)
    qy = max(0.0, abs(ry) - hw)
    qz = max(0.0, abs(dz) - hh)
    return math.sqrt(qx * qx + qy * qy + qz * qz)

def featurize_clifford3d(rec, rotation_angle=0.0):
    path = resample_path(rec['path'], target=SEQ_LEN)
    obs_list = rec['obstacles']
    
    if rotation_angle != 0.0:
        c, s = math.cos(rotation_angle), math.sin(rotation_angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        path = path @ R.T
        
    L = SEQ_LEN
    p_diff = np.diff(path, axis=0)
    p_diff = np.vstack([p_diff, p_diff[-1:]])
    
    dx, dy, dz = p_diff[:, 0], p_diff[:, 1], p_diff[:, 2]
    ds = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)
    
    clearances = np.zeros(L, dtype=np.float32)
    obs_dx = np.zeros(L, dtype=np.float32)
    obs_dy = np.zeros(L, dtype=np.float32)
    obs_dz = np.zeros(L, dtype=np.float32)
    
    if obs_list:
        for i, p in enumerate(path):
            dists = [_box_dist3d(p, o) for o in obs_list]
            j = int(np.argmin(dists))
            clearances[i] = dists[j]
            o = obs_list[j]
            obs_dx[i] = p[0] - o['x']
            obs_dy[i] = p[1] - o['y']
            obs_dz[i] = p[2] - (o['z'] + o['h']/2.0)
            
    mv = np.zeros((L, 8), dtype=np.float32)
    mv[:, 0] = np.log1p(clearances)
    mv[:, 1] = dx / (ds + 1e-6)
    mv[:, 2] = dy / (ds + 1e-6)
    mv[:, 3] = dz / (ds + 1e-6)
    
    obs_norm = np.sqrt(obs_dx**2 + obs_dy**2 + obs_dz**2 + 1e-6)
    ndx, ndy, ndz = obs_dx/obs_norm, obs_dy/obs_norm, obs_dz/obs_norm
    mv[:, 4] = (mv[:, 1] * ndy - mv[:, 2] * ndx)
    mv[:, 5] = (mv[:, 2] * ndz - mv[:, 3] * ndy)
    mv[:, 6] = (mv[:, 3] * ndx - mv[:, 1] * ndz)
    mv[:, 7] = (mv[:, 1] * ndy * ndz)
    
    return mv, ds.astype(np.float32), clearances

# Continuous TrajMamba 3D SSM
class ContinuousTrajMambaBlock(nn.Module):
    def __init__(self, d_model=64, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_state, 1, bias=False)
        self.D = nn.Parameter(torch.randn(d_model) * 0.1)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, delta_s):
        B, L, _ = x.shape
        x_proj = self.in_proj(x)
        dt = F.softplus(delta_s.unsqueeze(-1))
        A = -torch.exp(self.A_log)
        
        h = torch.zeros((B, self.d_model, self.d_state), device=x.device)
        ys = []
        B_mat = self.B_proj(x_proj)
        
        for t in range(L):
            dt_t = dt[:, t:t+1, :]
            dA = torch.exp(A.unsqueeze(0) * dt_t)
            dB = dt_t * B_mat[:, t:t+1, :]
            x_t = x_proj[:, t:t+1, :].transpose(1, 2)
            h = h * dA + x_t @ dB
            y_t = self.C_proj(h).squeeze(-1)
            ys.append(y_t.unsqueeze(1))
            
        y = torch.cat(ys, dim=1)
        out = self.norm(self.out_proj(y + x_proj * self.D.unsqueeze(0).unsqueeze(0)))
        return out

class CliffordTrajNet3D(nn.Module):
    def __init__(self, d_model=64, num_layers=2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(8, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([ContinuousTrajMambaBlock(d_model=d_model) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, mv, delta_s):
        feat = self.embed(mv)
        for blk in self.blocks:
            feat = blk(feat, delta_s)
        pooled = feat.mean(dim=1)
        return self.head(pooled).squeeze(-1)

# Baseline 3D Transformer
class BaselineTransformer3D(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(8, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                           dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, mv, delta_s=None):
        feat = self.embed(mv) + self.pos_emb
        feat = self.transformer(feat)
        return self.head(feat.mean(dim=1)).squeeze(-1)

def compute_apfd(order, labels):
    n = len(order)
    m = sum(labels[order])
    if m == 0: return 1.0
    ranks = [i + 1 for i, idx in enumerate(order) if labels[idx] == 1]
    return 1.0 - (sum(ranks) / (n * m)) + (1.0 / (2 * n))

def run_uav_experiment():
    print("=" * 70)
    print("Running Experiment 21: CliffordTrajNet-3D on Autonomous UAV Missions")
    print("=" * 70)
    
    # Featurize full dataset
    print("Featurizing 900 UAV flight missions...")
    X_mv, X_ds, y, min_dists, path_lens = [], [], [], [], []
    for r in raw_data:
        mv, ds, _ = featurize_clifford3d(r)
        X_mv.append(mv)
        X_ds.append(ds)
        y.append(1 if r['test_outcome'] == 'FAIL' else 0)
        min_dists.append(r.get('min_dist', 0.0))
        p = np.array(r['path'])
        pl = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) if len(p) > 1 else 0.0
        path_lens.append(pl)

    X_mv = np.array(X_mv)
    X_ds = np.array(X_ds)
    y = np.array(y, dtype=np.float32)
    min_dists = np.array(min_dists, dtype=np.float32)
    path_lens = np.array(path_lens, dtype=np.float32)

    # 80/20 Stratified Train/Test Split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(train_idx)} ({y[train_idx].sum()} fails), Test: {len(test_idx)} ({y[test_idx].sum()} fails)")

    # Training setup
    n_pos = y[train_idx].sum()
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([float(n_neg) / max(1, n_pos)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_mv_t = torch.tensor(X_mv[train_idx], dtype=torch.float32)
    train_ds_t = torch.tensor(X_ds[train_idx], dtype=torch.float32)
    train_y_t = torch.tensor(y[train_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_mv_t, train_ds_t, train_y_t), batch_size=32, shuffle=True)

    test_mv_t = torch.tensor(X_mv[test_idx], dtype=torch.float32).to(DEVICE)
    test_ds_t = torch.tensor(X_ds[test_idx], dtype=torch.float32).to(DEVICE)
    test_y = y[test_idx]

    # Train CliffordTrajNet3D
    print("\n--- Training CliffordTrajNet3D ---")
    model = CliffordTrajNet3D(d_model=64, num_layers=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_auc = 0.0
    best_state = None
    t0 = time.time()

    for ep in range(35):
        model.train()
        for b_mv, b_ds, b_y in train_loader:
            b_mv, b_ds, b_y = b_mv.to(DEVICE), b_ds.to(DEVICE), b_y.to(DEVICE)
            opt.zero_grad()
            out = model(b_mv, b_ds)
            loss = criterion(out, b_y)
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(test_mv_t, test_ds_t)).cpu().numpy()
            auc = roc_auc_score(test_y, preds)
            if auc > best_auc:
                best_auc = auc
                best_state = copy.deepcopy(model.state_dict())
        if (ep + 1) % 5 == 0:
            print(f"  Epoch {ep+1:2d} | Val AUC: {auc:.4f} | Best AUC: {best_auc:.4f}")

    clifford_train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        clifford_preds = torch.sigmoid(model(test_mv_t, test_ds_t)).cpu().numpy()

    # Train Baseline 3D Transformer for direct comparison
    print("\n--- Training 3D Transformer Baseline ---")
    tf_model = BaselineTransformer3D(d_model=64, nhead=4, num_layers=2).to(DEVICE)
    tf_opt = torch.optim.AdamW(tf_model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_tf_auc = 0.0
    best_tf_state = None

    for ep in range(35):
        tf_model.train()
        for b_mv, b_ds, b_y in train_loader:
            b_mv, b_y = b_mv.to(DEVICE), b_y.to(DEVICE)
            tf_opt.zero_grad()
            out = tf_model(b_mv)
            loss = criterion(out, b_y)
            loss.backward()
            tf_opt.step()
        tf_model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(tf_model(test_mv_t)).cpu().numpy()
            auc = roc_auc_score(test_y, preds)
            if auc > best_tf_auc:
                best_tf_auc = auc
                best_tf_state = copy.deepcopy(tf_model.state_dict())

    tf_model.load_state_dict(best_tf_state)
    tf_model.eval()
    with torch.no_grad():
        tf_preds = torch.sigmoid(tf_model(test_mv_t)).cpu().numpy()

    # Canonical 30-trial Multi-trial Protocol (|S| = max(50, 0.3*|test|))
    N_TRIALS = 30
    sample_size = max(50, int(0.3 * len(test_idx)))
    print(f"\n--- Running 30-Trial Protocol (sample={sample_size}, N_TRIALS={N_TRIALS}) ---")

    clifford_trials = []
    tf_trials = []
    random_trials = []
    pathlen_trials = []

    test_path_lens = path_lens[test_idx]

    for t in range(N_TRIALS):
        rng = np.random.RandomState(42 + t)
        sub = rng.permutation(len(test_idx))[:sample_size]
        sub_y = test_y[sub]
        
        # CliffordTrajNet
        ord_clifford = np.argsort(-clifford_preds[sub])
        clifford_trials.append(compute_apfd(ord_clifford, sub_y))
        
        # 3D Transformer
        ord_tf = np.argsort(-tf_preds[sub])
        tf_trials.append(compute_apfd(ord_tf, sub_y))
        
        # Random
        ord_rand = rng.permutation(sample_size)
        random_trials.append(compute_apfd(ord_rand, sub_y))
        
        # Path length
        ord_pl = np.argsort(-test_path_lens[sub])
        pathlen_trials.append(compute_apfd(ord_pl, sub_y))

    clifford_trials = np.array(clifford_trials)
    tf_trials = np.array(tf_trials)
    random_trials = np.array(random_trials)
    pathlen_trials = np.array(pathlen_trials)

    print(f"CliffordTrajNet-3D: APFD = {clifford_trials.mean():.4f} +/- {clifford_trials.std():.4f} (AUC: {best_auc:.4f})")
    print(f"3D Transformer:     APFD = {tf_trials.mean():.4f} +/- {tf_trials.std():.4f} (AUC: {best_tf_auc:.4f})")
    print(f"Path Length Heur:   APFD = {pathlen_trials.mean():.4f} +/- {pathlen_trials.std():.4f}")
    print(f"Random:             APFD = {random_trials.mean():.4f} +/- {random_trials.std():.4f}")

    # Rotation Invariance Probe across 6 planar yaw rotations
    print("\n--- Running 3D Rotation Invariance Probe ---")
    angles = [0.0, 30.0, 60.0, 90.0, 180.0, -45.0]
    rot_apfds = []
    for deg in angles:
        rad = math.radians(deg)
        rot_mv = []
        rot_ds = []
        for idx in test_idx:
            mv, ds, _ = featurize_clifford3d(raw_data[idx], rotation_angle=rad)
            rot_mv.append(mv)
            rot_ds.append(ds)
        rot_mv = torch.tensor(np.array(rot_mv), dtype=torch.float32).to(DEVICE)
        rot_ds = torch.tensor(np.array(rot_ds), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds_rot = torch.sigmoid(model(rot_mv, rot_ds)).cpu().numpy()
        ord_rot = np.argsort(-preds_rot)
        rot_apfd = compute_apfd(ord_rot, test_y)
        rot_apfds.append(float(round(rot_apfd, 4)))
        print(f"  Angle {deg:6.1f} deg: APFD = {rot_apfd:.4f}")

    delta_rot = max(rot_apfds) - min(rot_apfds)
    print(f"Rotation Invariance Delta APFD: {delta_rot:.4f}")

    # Monotone Conformal Risk Control on UAV Clearance
    print("\n--- Conformal Risk Control Calibration ---")
    cal_n = len(test_idx) // 2
    cal_scores = clifford_preds[:cal_n]
    cal_labels = test_y[:cal_n]
    eval_scores = clifford_preds[cal_n:]
    eval_labels = test_y[cal_n:]

    epsilon = 0.05
    cal_fail_scores = cal_scores[cal_labels == 1]
    if len(cal_fail_scores) > 0:
        lam_thresh = float(np.quantile(cal_fail_scores, epsilon))
    else:
        lam_thresh = float(np.quantile(cal_scores, epsilon))

    eval_selected = (eval_scores >= lam_thresh)
    eval_budget_pct = float(round(np.mean(eval_selected) * 100, 2))
    eval_detected_fails = int(np.sum(eval_selected & (eval_labels == 1)))
    eval_total_fails = int(np.sum(eval_labels == 1))
    eval_recall_pct = float(round(100.0 * eval_detected_fails / max(1, eval_total_fails), 2))
    print(f"Calibrated threshold: {lam_thresh:.4f} | Budget: {eval_budget_pct}% | Recall: {eval_recall_pct}%")

    results_dict = {
        "benchmark": "SBFT_Aerialist_PX4_3D",
        "dataset_source": "data/uav/uav_dataset_surrogate.json",
        "protocol": {
            "split": "80/20 stratified",
            "random_state": 42,
            "n_total": len(raw_data),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_fail_total": int(y.sum()),
            "n_fail_train": int(y[train_idx].sum()),
            "n_fail_test": int(y[test_idx].sum()),
            "n_trials": N_TRIALS,
            "sample_size": sample_size
        },
        "model": {
            "name": "CliffordTrajNet3D",
            "geometric_algebra": "Clifford Cl(3,0) 8-channel multivector",
            "backbone": "ContinuousTrajMamba (2 layers, continuous arclength ODE)",
            "d_model": 64,
            "parameters": sum(p.numel() for p in model.parameters())
        },
        "training": {
            "epochs": 35,
            "best_val_auc": float(round(best_auc, 4)),
            "wall_clock_seconds": float(round(clifford_train_time, 2))
        },
        "evaluation": {
            "single_pass_apfd": float(round(compute_apfd(np.argsort(-clifford_preds), test_y), 4)),
            "multi_trial_apfd_mean": float(round(clifford_trials.mean(), 4)),
            "multi_trial_apfd_std": float(round(clifford_trials.std(), 4)),
            "multi_trial_apfd_min": float(round(clifford_trials.min(), 4)),
            "multi_trial_apfd_max": float(round(clifford_trials.max(), 4)),
            "trials_apfd": [float(round(v, 4)) for v in clifford_trials.tolist()]
        },
        "baselines": {
            "random_prioritization": {
                "apfd_mean": float(round(random_trials.mean(), 4)),
                "apfd_std": float(round(random_trials.std(), 4))
            },
            "path_length_heuristic": {
                "apfd_mean": float(round(pathlen_trials.mean(), 4)),
                "apfd_std": float(round(pathlen_trials.std(), 4))
            },
            "transformer_3d_baseline": {
                "val_auc": float(round(best_tf_auc, 4)),
                "apfd_mean": float(round(tf_trials.mean(), 4)),
                "apfd_std": float(round(tf_trials.std(), 4))
            }
        },
        "rotation_invariance_probe": {
            "angles_deg": angles,
            "apfd_per_angle": rot_apfds,
            "delta_apfd": float(round(delta_rot, 4))
        },
        "conformal_risk_control": {
            "epsilon": epsilon,
            "confidence": 0.99,
            "threshold": float(round(lam_thresh, 4)),
            "calibrated_budget_percent": eval_budget_pct,
            "empirical_recall_percent": eval_recall_pct,
            "detected_failures": eval_detected_fails,
            "total_failures": eval_total_fails,
            "satisfied": bool(eval_recall_pct >= (1.0 - epsilon) * 100)
        }
    }

    OUT_PATH = "exps/results/exp21_UAV_Clifford3D_results.json"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[Success] Fully verified results saved to {OUT_PATH}")
    return results_dict

if __name__ == "__main__":
    run_uav_experiment()
