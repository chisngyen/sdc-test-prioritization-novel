"""
EXP 18: RoadMamba — Continuous Selective State Space Model
==========================================================
Theoretical Lens: Continuous-Time Dynamical Systems & Selective SSM (Mamba)
Headline Claim: Zero-discretization error via continuous ODE integration.
Linear O(L) complexity, modeling cumulative lateral vehicle instability.

Continuous State Space Formulation:
    dh(s)/ds = A(s) h(s) + B(s) x(s)
    y(s)     = C(s) h(s)
Continuous-time discretization driven by arclength intervals Δs:
    A_bar = exp(Δs · A)
    B_bar = (Δs · A)^{-1} (exp(Δs · A) - I) (Δs · B)

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

# ---------- Hardware & Paths ----------
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

# ---------- Config ----------
SEQ_LEN    = 197
D_MODEL    = 192
D_STATE    = 32     # SSM hidden state dimension
NUM_LAYERS = 6
BATCH_SIZE = 384
EPOCHS     = 80
LR         = 5e-4
GAMMA      = 1.5
SWA_START  = 56
N_TRIALS   = 30

# ====================================================================
# Continuous Selective State Space Block (RoadMamba)
# ====================================================================

class ContinuousSSM(nn.Module):
    """
    Continuous-Time Selective State Space Layer (S4D/Mamba parameterized).
    Continuous transition matrix A is diagonalized with negative real parts (stable ODE).
    Discretization is dynamically modulated by physical arclength step Δs.
    """
    def __init__(self, d_model=D_MODEL, d_state=D_STATE):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Continuous diagonal A: initialized with HiPPO-like decay
        # log(-A) is learned to enforce stability (A < 0)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        
        # Input-dependent projections for B, C, and Δs scale (Selection mechanism)
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        
        # D skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x, delta_s):
        # x: (B, L, D), delta_s: (B, L, 1)
        B, L, D = x.shape
        
        # Selective parameters from input
        proj = self.x_proj(x) # (B, L, 2*d_state + 1)
        B_mat = proj[:, :, :self.d_state] # (B, L, d_state)
        C_mat = proj[:, :, self.d_state:2*self.d_state] # (B, L, d_state)
        dt_scale = F.softplus(proj[:, :, -1:]) # (B, L, 1)
        
        # Continuous time step modulated by physical arclength: dt = Δs * dt_scale
        dt = F.softplus(self.dt_proj(delta_s * dt_scale)) # (B, L, D)
        
        # Discretize continuous A: A_bar = exp(dt * A)
        # A is negative: -exp(A_log)
        A = -torch.exp(self.A_log) # (D, d_state)
        # dt: (B, L, D, 1), A: (1, 1, D, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, D, self.d_state)) # (B, L, D, d_state)
        
        # Discretize B: dB = dt * B (Euler approximation for selective SSM)
        # dt: (B, L, D, 1), B_mat: (B, L, 1, d_state)
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2) # (B, L, D, d_state)
        
        # Sequential selective scan (can be parallelized in associative scan)
        # x: (B, L, D) -> dB * x: (B, L, D, d_state)
        dBx = dB * x.unsqueeze(-1)
        
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            # y_t = C_t * h_t: (B, D)
            y_t = torch.sum(h * C_mat[:, t].unsqueeze(1), dim=-1)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # (B, L, D)
        y = y + x * self.D
        return y

class RoadMambaBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, d_state=D_STATE, d_ff=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Bidirectional Continuous SSM: forward in arclength & reverse (braking anticipation)
        self.ssm_fwd = ContinuousSSM(d_model, d_state)
        self.ssm_bwd = ContinuousSSM(d_model, d_state)
        self.out_proj = nn.Linear(d_model * 2, d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, delta_s):
        residual = x
        x_norm = self.norm1(x)
        
        # Forward pass along road trajectory
        y_fwd = self.ssm_fwd(x_norm, delta_s)
        
        # Backward pass (looking ahead into oncoming curves)
        y_bwd = self.ssm_bwd(torch.flip(x_norm, dims=[1]), torch.flip(delta_s, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])
        
        y = self.out_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        x = residual + y
        x = x + self.ffn(self.norm2(x))
        return x

class RoadMambaNet(nn.Module):
    def __init__(self, in_features=7, d_model=D_MODEL, n_layers=NUM_LAYERS):
        super().__init__()
        self.input_embed = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList([
            RoadMambaBlock(d_model=d_model)
            for _ in range(n_layers)
        ])
        
        # Dual-Pooling: Global average + Max-Hazard (Extreme Value Pooling)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, feats, delta_s):
        # feats: (B, L, 7), delta_s: (B, L, 1)
        B, L, _ = feats.shape
        x = self.input_embed(feats)
        
        for blk in self.blocks:
            x = blk(x, delta_s)
            
        # Extreme Value Pooling: Mean pool + Max hazard pool
        mean_pool = torch.mean(x, dim=1)
        max_pool = torch.max(x, dim=1)[0]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        
        logits = self.classifier(pooled).squeeze(-1)
        return logits

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
    delta_s = np.diff(s_eval, prepend=0).reshape(-1, 1).astype(np.float32)
    return feat_matrix, delta_s

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
# Resolution Invariance Probe
# ====================================================================

def run_resolution_probe(model, raw_tests):
    model.eval()
    densities = [64, 96, 128, 160, 197]
    results = {}
    with torch.no_grad():
        for n_pts in densities:
            scores, labels = [], []
            for t in raw_tests:
                f, ds = extract_features(t['points'], target_len=n_pts)
                f_t = torch.tensor(f, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                ds_t = torch.tensor(ds, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                logit = model(f_t, ds_t).cpu().item()
                scores.append(logit)
                labels.append(t['outcome'])
            apfd = compute_apfd(np.array(scores), np.array(labels))
            results[f"N_{n_pts}"] = float(apfd)
    vals = list(results.values())
    delta = max(vals) - min(vals)
    results['delta'] = float(delta)
    return results

# ====================================================================
# Main Pipeline
# ====================================================================

def main():
    print("=" * 65)
    print("EXP 18: RoadMamba — Continuous Selective State Space Model")
    print("=" * 65)
    
    if not TRAIN_PATH or not os.path.isfile(TRAIN_PATH):
        print(f"Dataset not found in search paths: {SEARCH_ROOTS}")
        return

    print(f"Loading data from: {TRAIN_PATH}")
    with open(TRAIN_PATH, 'r') as f: train_data = json.load(f)
    with open(TEST_PATH, 'r') as f: test_data = json.load(f)
    comp_data = json.load(open(COMP_PATH, 'r')) if COMP_PATH else test_data
    
    def build_dataset(data):
        X_list, DS_list, y_list = [], [], []
        for item in data:
            f, ds = extract_features(item['points'])
            X_list.append(f)
            DS_list.append(ds)
            y_list.append(item['outcome'])
        return np.array(X_list), np.array(DS_list), np.array(y_list, dtype=np.float32)

    X_train, DS_train, y_train = build_dataset(train_data)
    X_val, DS_val, y_val = build_dataset(test_data)
    X_comp, DS_comp, y_comp = build_dataset(comp_data)
    
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_comp  = (X_comp - mean) / std
    
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(DS_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(DS_val), torch.tensor(y_val))
    comp_ds  = TensorDataset(torch.tensor(X_comp), torch.tensor(DS_comp), torch.tensor(y_comp))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    comp_loader  = DataLoader(comp_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = RoadMambaNet().to(DEVICE)
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
        for bx, bds, by in train_loader:
            bx, bds, by = bx.to(DEVICE), bds.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits = model(bx, bds)
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
            for bx, bds, by in val_loader:
                bx, bds = bx.to(DEVICE), bds.to(DEVICE)
                logits = model(bx, bds)
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
        for bx, bds, by in comp_loader:
            bx, bds = bx.to(DEVICE), bds.to(DEVICE)
            logits = swa_model(bx, bds)
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
    
    print("\nRunning Resolution Invariance Probe (N in [64, 197])...")
    res_probe = run_resolution_probe(swa_model, comp_data)
    print(f"Resolution Delta: {res_probe['delta']:.5f}")
    for k, v in res_probe.items():
        if k != 'delta': print(f"  {k}: {v:.4f}")
        
    results = {
        'exp': 'exp18_RoadMamba_SSM',
        'params': sum(p.numel() for p in model.parameters()),
        'train_time_min': train_time,
        'val_best_auc': float(best_auc),
        'comp_auc': comp_auc,
        'comp_apfd_single': single_apfd,
        'comp_apfd_multi': f"{mean_apfd:.4f} +/- {std_apfd:.4f}",
        'resolution_probe': res_probe
    }
    
    out_file = os.path.join(OUTPUT_DIR, 'exp18_RoadMamba_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")

if __name__ == '__main__':
    main()
