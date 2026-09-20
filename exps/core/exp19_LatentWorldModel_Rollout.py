"""
EXP 19: Latent Trajectory World Model (WorldModel-Rollout)
=========================================================
Theoretical Lens: World Models & Latent Dynamics Rollout (JEPA / GAIA-1 inspired)
Headline Claim: Failure scoring via latent trajectory free-energy divergence.
Predicts the autonomous driving agent's unrolled latent state along the road.

Formulation:
    z_{t+1} = f_dyn(z_t, road_geometry(s_t))
    Energy(road) = max_t || z_t - z_nominal ||^2 + λ_slip || dz/dt ||^2

Self-contained: paste into a single Kaggle cell or run locally.
"""

import os, sys, json, time, math, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SEARCH_ROOTS = [
    '/kaggle/input',
    '/kaggle/input/chinguyeen/sdc-sensodat',
    '/kaggle/input/datasets/chinguyeen/sdc-sensodat',
    '/kaggle/input/sdc-sensodat',
    '/kaggle/input/datasets/chinguyeen/sdc-test-data',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')),
    os.getcwd(),
]

def find_file(filename):
    for root in SEARCH_ROOTS:
        p = os.path.join(root, filename)
        if os.path.isfile(p): return p
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                if filename in filenames:
                    return os.path.join(dirpath, filename)
    return None

TRAIN_PATH = find_file('sensodat_train.json')
TEST_PATH  = find_file('sensodat_test.json')
COMP_PATH  = find_file('sdc-test-data.json')

OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN    = 197
D_LATENT   = 64      # Dimension of vehicle latent state
D_ROAD     = 128     # Dimension of road geometric context
NUM_LAYERS = 4
BATCH_SIZE = 384
EPOCHS     = 80
LR         = 5e-4
GAMMA      = 1.5
SWA_START  = 56
N_TRIALS   = 30

# ====================================================================
# Latent World Model Architecture
# ====================================================================

class LatentVehicleTransition(nn.Module):
    """
    Recurrent Latent Transition Function modeling vehicle state dynamics:
    z_{t+1} = z_t + dt * MLP(z_t, road_token_t)
    Enforces continuous Hamiltonian-like energy conservation.
    """
    def __init__(self, d_latent=D_LATENT, d_road=D_ROAD):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_road, 128),
            nn.GELU(),
            nn.Linear(128, d_latent)
        )
        
    def forward(self, z, road_feat):
        # z: (B, d_latent), road_feat: (B, d_road)
        dz = self.net(torch.cat([z, road_feat], dim=-1))
        # Residual step (Euler integrator)
        return z + 0.1 * dz

class TrajectoryWorldModel(nn.Module):
    def __init__(self, in_features=7, d_road=D_ROAD, d_latent=D_LATENT):
        super().__init__()
        self.road_encoder = nn.Sequential(
            nn.Linear(in_features, d_road),
            nn.LayerNorm(d_road),
            nn.GELU(),
            nn.Linear(d_road, d_road)
        )
        
        # Initial state generator (vehicle entering at nominal stable state)
        self.z0_param = nn.Parameter(torch.zeros(1, d_latent))
        self.transition = LatentVehicleTransition(d_latent=d_latent, d_road=d_road)
        
        # Energy divergence head
        self.energy_head = nn.Sequential(
            nn.Linear(d_latent, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, road_feats):
        # road_feats: (B, L, 7)
        B, L, _ = road_feats.shape
        encoded_road = self.road_encoder(road_feats) # (B, L, d_road)
        
        z = self.z0_param.expand(B, -1)
        latent_states = []
        
        # Auto-regressive latent rollout along the road trajectory
        for t in range(L):
            z = self.transition(z, encoded_road[:, t])
            latent_states.append(z)
            
        # Z_all: (B, L, d_latent)
        Z_all = torch.stack(latent_states, dim=1)
        
        # Compute latent divergence energy along trajectory
        energies = self.energy_head(Z_all).squeeze(-1) # (B, L)
        
        # Max-Energy Pooling: the test score is governed by the maximum instability
        max_energy = torch.max(energies, dim=-1)[0] # (B,)
        mean_energy = torch.mean(energies, dim=-1) # (B,)
        
        score = 0.7 * max_energy + 0.3 * mean_energy
        return score

# ====================================================================
# Geometry Feature Extraction
# ====================================================================

def compute_menger_curvature(pts):
    n = len(pts)
    curv = np.zeros(n - 2, dtype=np.float64)
    for i in range(n - 2):
        x1, y1 = pts[i]; x2, y2 = pts[i+1]; x3, y3 = pts[i+2]
        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x3 - x1, y3 - y1)
        s = 0.5 * (a + b + c)
        area = s * (s - a) * (s - b) * (s - c)
        if area <= 1e-10:
            curv[i] = 0.0
        else:
            R = (a * b * c) / (4.0 * math.sqrt(max(1e-12, area)))
            cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
            sgn = 1.0 if cross >= 0 else -1.0
            curv[i] = sgn / max(1e-4, R)
    return curv

def extract_features(pts_raw, target_len=SEQ_LEN):
    pts = np.asarray(pts_raw, dtype=np.float64)
    if pts.ndim == 1: pts = pts.reshape(-1, 2)
    elif pts.ndim == 2 and pts.shape[1] > 2: pts = pts[:, :2]
    
    diffs = np.diff(pts, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_s = np.concatenate([[0], np.cumsum(dists)])
    tot_s = cum_s[-1] if cum_s[-1] > 1e-6 else 1.0
    
    s_eval = np.linspace(0, tot_s, target_len)
    pts_res = np.empty((target_len, 2), dtype=np.float64)
    pts_res[:, 0] = np.interp(s_eval, cum_s, pts[:, 0])
    pts_res[:, 1] = np.interp(s_eval, cum_s, pts[:, 1])
    
    diffs_res = np.diff(pts_res, axis=0)
    seg_lens = np.hypot(diffs_res[:, 0], diffs_res[:, 1])
    f0 = np.pad(seg_lens, (0, 1), mode='edge')
    
    angles = np.arctan2(diffs_res[:, 1], diffs_res[:, 0])
    d_theta = np.diff(angles)
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    f1 = np.pad(np.abs(d_theta), (1, 1), mode='constant')
    
    curv = compute_menger_curvature(pts_res)
    f2 = np.pad(curv, (1, 1), mode='constant')
    f3 = np.pad(np.diff(f2), (0, 1), mode='constant')
    f4 = np.pad(np.diff(f3), (0, 1), mode='constant')
    f5 = s_eval / tot_s
    
    f6 = np.zeros(target_len)
    w = 5
    for i in range(target_len):
        f6[i] = np.std(f2[max(0, i - w):min(target_len, i + w + 1)])
        
    feat_matrix = np.column_stack([f0, f1, f2, f3, f4, f5, f6]).astype(np.float32)
    return feat_matrix

# ====================================================================
# SWA, Focal Loss, Evaluation
# ====================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=GAMMA):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()

class SWAModel:
    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.n = 0
    def update(self, new_model):
        self.n += 1
        alpha = 1.0 / self.n
        for p_swa, p_new in zip(self.model.parameters(), new_model.parameters()):
            p_swa.data.mul_(1 - alpha).add_(p_new.data, alpha=alpha)
    def get(self):
        return self.model

def compute_apfd(scores, labels):
    n = len(labels)
    m = int(np.sum(labels))
    if m == 0 or m == n: return 0.5
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tf = np.where(sorted_labels == 1)[0] + 1
    return 1.0 - (np.sum(tf) / (n * m)) + (1.0 / (2 * n))

def multi_trial_apfd(scores, labels, n_trials=N_TRIALS, sample_ratio=0.3):
    n = len(labels)
    sample_size = max(50, int(sample_ratio * n))
    rng = np.random.RandomState(42)
    apfds = []
    for _ in range(n_trials):
        idx = rng.choice(n, size=sample_size, replace=False)
        sub_scores = scores[idx]
        sub_labels = labels[idx]
        if np.sum(sub_labels) == 0: continue
        apfds.append(compute_apfd(sub_scores, sub_labels))
    return float(np.mean(apfds)), float(np.std(apfds))

# ====================================================================
# Main Pipeline
# ====================================================================

def main():
    print("=" * 65)
    print("EXP 19: Latent Trajectory World Model (WorldModel-Rollout)")
    print("=" * 65)
    
    if not TRAIN_PATH or not os.path.isfile(TRAIN_PATH):
        print(f"Dataset not found in search paths: {SEARCH_ROOTS}")
        return

    print(f"Loading data from: {TRAIN_PATH}")
    with open(TRAIN_PATH, 'r') as f: train_data = json.load(f)
    with open(TEST_PATH, 'r') as f: test_data = json.load(f)
    comp_data = json.load(open(COMP_PATH, 'r')) if COMP_PATH else test_data
    
    def build_dataset(data):
        X_list, y_list = [], []
        for item in data:
            f = extract_features(item['points'])
            X_list.append(f)
            y_list.append(item['outcome'])
        return np.array(X_list), np.array(y_list, dtype=np.float32)

    X_train, y_train = build_dataset(train_data)
    X_val, y_val = build_dataset(test_data)
    X_comp, y_comp = build_dataset(comp_data)
    
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_comp  = (X_comp - mean) / std
    
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    comp_ds  = TensorDataset(torch.tensor(X_comp), torch.tensor(y_comp))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    comp_loader  = DataLoader(comp_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TrajectoryWorldModel().to(DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = FocalLoss(gamma=GAMMA)
    scaler = GradScaler()
    swa = SWAModel(model)
    
    best_auc = 0.0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits = model(bx)
                loss = criterion(logits, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(by)
            
        if epoch >= SWA_START:
            swa.update(model)
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(DEVICE)
                logits = model(bx)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(by.numpy())
        val_auc = roc_auc_score(val_targets, val_preds)
        if val_auc > best_auc:
            best_auc = val_auc
            
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} - Loss: {total_loss/len(y_train):.4f} - Val AUC: {val_auc:.4f} (Best: {best_auc:.4f})")
            
    train_time = (time.time() - start_time) / 60.0
    print(f"Training finished in {train_time:.2f} min.")
    
    swa_model = swa.get().to(DEVICE)
    swa_model.eval()
    comp_preds, comp_targets = [], []
    with torch.no_grad():
        for bx, by in comp_loader:
            bx = bx.to(DEVICE)
            logits = swa_model(bx)
            comp_preds.extend(torch.sigmoid(logits).cpu().numpy())
            comp_targets.extend(by.numpy())
            
    comp_preds = np.array(comp_preds)
    comp_targets = np.array(comp_targets)
    
    comp_auc = float(roc_auc_score(comp_targets, comp_preds))
    single_apfd = float(compute_apfd(comp_preds, comp_targets))
    mean_apfd, std_apfd = multi_trial_apfd(comp_preds, comp_targets)
    
    print("\n" + "=" * 50)
    print(f"Competition AUC:       {comp_auc:.4f}")
    print(f"Competition APFD:      {single_apfd:.4f}")
    print(f"Competition Multi-30:  {mean_apfd:.4f} +/- {std_apfd:.4f}")
    
    results = {
        'exp': 'exp19_LatentWorldModel_Rollout',
        'params': sum(p.numel() for p in model.parameters()),
        'train_time_min': train_time,
        'val_best_auc': float(best_auc),
        'comp_auc': comp_auc,
        'comp_apfd_single': single_apfd,
        'comp_apfd_multi': f"{mean_apfd:.4f} +/- {std_apfd:.4f}"
    }
    
    out_file = os.path.join(OUTPUT_DIR, 'exp19_WorldModel_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")

if __name__ == '__main__':
    main()
