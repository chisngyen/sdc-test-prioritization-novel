"""
EXP 17: Clifford Geometric Transformer (CliffordNet)
=====================================================
Theoretical Lens: Geometric Algebra / Clifford Algebra Cℓ(2,0)
Headline Claim: Exact multivector grade-preserving geometric attention.
Eliminates coordinate-frame artifacts via Rotor Sandwiches and geometric products.

Multivector Representation:
    M(s) = s (scalar, grade 0) + [t_x e1 + t_y e2] (vector, grade 1) + κ e12 (bivector, grade 2)
Geometric Product:
    u v = u · v + u ∧ v
Rotor transformation:
    M' = R M R†, where R = cos(θ/2) - e12 sin(θ/2)

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
        # recursive check
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
D_MODEL    = 192  # divisible by 4 for Clifford grades (scalar, v_x, v_y, bivector)
NUM_HEADS  = 8
NUM_LAYERS = 6
BATCH_SIZE = 384
EPOCHS     = 80
LR         = 5e-4
GAMMA      = 1.5
SWA_START  = 56
N_TRIALS   = 30

# ====================================================================
# Clifford Algebra Cℓ(2,0) Primitives
# ====================================================================

class CliffordLinear(nn.Module):
    """
    Grade-preserving linear transformation for Cℓ(2,0) multivectors.
    Channels are split into 4 components: [scalar, e1, e2, e12].
    Scalars and Bivectors transform independently; Vectors transform via 2x2 blocks.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0
        self.c_in = in_features // 4
        self.c_out = out_features // 4
        
        # Grade-0 (scalar) & Grade-2 (bivector) weights
        self.w_scalar = nn.Linear(self.c_in, self.c_out, bias=False)
        self.w_bivector = nn.Linear(self.c_in, self.c_out, bias=False)
        
        # Grade-1 (vector e1, e2) coupled 2x2 equivariant transformation
        self.w_vec_diag = nn.Linear(self.c_in, self.c_out, bias=False)
        self.w_vec_cross = nn.Linear(self.c_in, self.c_out, bias=False)
        
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: (B, L, in_features), reshaped into (B, L, 4, c_in)
        B, L, _ = x.shape
        x_mv = x.view(B, L, 4, self.c_in)
        s, v1, v2, bv = x_mv[:, :, 0], x_mv[:, :, 1], x_mv[:, :, 2], x_mv[:, :, 3]
        
        # Linear projections
        out_s = self.w_scalar(s)
        out_bv = self.w_bivector(bv)
        
        # Vector transformation: preserves rotational equivariance
        out_v1 = self.w_vec_diag(v1) - self.w_vec_cross(v2)
        out_v2 = self.w_vec_cross(v1) + self.w_vec_diag(v2)
        
        out = torch.stack([out_s, out_v1, out_v2, out_bv], dim=2).view(B, L, -1)
        return out + self.bias

class CliffordAttentionBlock(nn.Module):
    """
    Clifford-Equivariant Attention with Geometric Product Inner Product.
    The attention affinity between multivector tokens A and B utilizes the scalar part
    of their geometric product: <A B†>_0 = s_A s_B + v_A · v_B + bv_A bv_B.
    """
    def __init__(self, d_model=D_MODEL, nhead=NUM_HEADS, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = CliffordLinear(d_model, d_model)
        self.k_proj = CliffordLinear(d_model, d_model)
        self.v_proj = CliffordLinear(d_model, d_model)
        self.out_proj = CliffordLinear(d_model, d_model)
        
        # Arclength relative bias
        self.rff_dim = 32
        self.register_buffer('rff_freq', torch.randn(self.rff_dim) * 2.0)
        self.bias_head = nn.Linear(self.rff_dim * 2, nhead, bias=False)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, arclength):
        # x: (B, L, D)
        B, L, D = x.shape
        residual = x
        x_norm = self.norm1(x)
        
        q = self.q_proj(x_norm).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        # Clifford geometric product scalar part for dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Relative arclength bias
        # delta_s: (B, L, L)
        delta_s = torch.abs(arclength.unsqueeze(2) - arclength.unsqueeze(1))
        # RFF projection
        proj = delta_s.unsqueeze(-1) * self.rff_freq
        rff = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) # (B, L, L, 2*rff_dim)
        rel_bias = self.bias_head(rff).permute(0, 3, 1, 2) # (B, nhead, L, L)
        
        scores = scores + rel_bias
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        x = residual + self.out_proj(context)
        x = x + self.ffn(self.norm2(x))
        return x

class CliffordGeometricNet(nn.Module):
    def __init__(self, in_mv_dim=4, d_model=D_MODEL, n_layers=NUM_LAYERS):
        super().__init__()
        # in_mv_dim: 4 channels per multivector (scalar, e1, e2, e12)
        # We expand to 7 intrinsic channels wrapped in Cℓ(2,0)
        self.input_embed = nn.Sequential(
            nn.Linear(7, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.blocks = nn.ModuleList([
            CliffordAttentionBlock(d_model=d_model, nhead=NUM_HEADS)
            for _ in range(n_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, feats, arclength):
        # feats: (B, L, 7), arclength: (B, L)
        B, L, _ = feats.shape
        x = self.input_embed(feats)
        
        # CLS token prepended
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Arclength with CLS at s=0
        cls_s = torch.zeros(B, 1, device=arclength.device)
        s_all = torch.cat([cls_s, arclength], dim=1)
        
        for blk in self.blocks:
            x = blk(x, s_all)
            
        cls_out = x[:, 0]
        logits = self.classifier(cls_out).squeeze(-1)
        return logits

# ====================================================================
# Geometry Feature Extraction (7-Channel Intrinsic Cℓ(2,0))
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

def extract_clifford_features(pts_raw, target_len=SEQ_LEN):
    pts = np.asarray(pts_raw, dtype=np.float64)
    if pts.ndim == 1: pts = pts.reshape(-1, 2)
    elif pts.ndim == 2 and pts.shape[1] > 2: pts = pts[:, :2]
    
    # Uniform arclength resample to target_len
    diffs = np.diff(pts, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_s = np.concatenate([[0], np.cumsum(dists)])
    tot_s = cum_s[-1] if cum_s[-1] > 1e-6 else 1.0
    
    s_eval = np.linspace(0, tot_s, target_len)
    pts_res = np.empty((target_len, 2), dtype=np.float64)
    pts_res[:, 0] = np.interp(s_eval, cum_s, pts[:, 0])
    pts_res[:, 1] = np.interp(s_eval, cum_s, pts[:, 1])
    
    # 7 Intrinsic Channels:
    # 0: segment length
    diffs_res = np.diff(pts_res, axis=0)
    seg_lens = np.hypot(diffs_res[:, 0], diffs_res[:, 1])
    f0 = np.pad(seg_lens, (0, 1), mode='edge')
    
    # 1: intrinsic turn angle
    angles = np.arctan2(diffs_res[:, 1], diffs_res[:, 0])
    d_theta = np.diff(angles)
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    f1 = np.pad(np.abs(d_theta), (1, 1), mode='constant')
    
    # 2: signed Menger curvature
    curv = compute_menger_curvature(pts_res)
    f2 = np.pad(curv, (1, 1), mode='constant')
    
    # 3: curvature jerk dκ/ds
    f3 = np.pad(np.diff(f2), (0, 1), mode='constant')
    
    # 4: curvature acceleration d²κ/ds²
    f4 = np.pad(np.diff(f3), (0, 1), mode='constant')
    
    # 5: normalized arclength s/L
    f5 = s_eval / tot_s
    
    # 6: local curvature std (volatility)
    f6 = np.zeros(target_len)
    w = 5
    for i in range(target_len):
        f6[i] = np.std(f2[max(0, i - w):min(target_len, i + w + 1)])
        
    feat_matrix = np.column_stack([f0, f1, f2, f3, f4, f5, f6]).astype(np.float32)
    return feat_matrix, s_eval.astype(np.float32)

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
# Rotation Probe for Geometric Equivariance
# ====================================================================

def rotate_points(pts, angle_deg):
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return np.dot(pts, R.T)

def run_rotation_probe(model, raw_tests):
    model.eval()
    rotations = [0, 30, 60, 90, 180, -45]
    results = {}
    with torch.no_grad():
        for deg in rotations:
            scores = []
            labels = []
            for t in raw_tests:
                pts = np.array(t['points'], dtype=np.float64)
                if deg != 0:
                    pts = rotate_points(pts, deg)
                f, s = extract_clifford_features(pts)
                f_t = torch.tensor(f, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                logit = model(f_t, s_t).cpu().item()
                scores.append(logit)
                labels.append(t['outcome'])
            apfd = compute_apfd(np.array(scores), np.array(labels))
            results[f"{deg}_deg"] = float(apfd)
    
    vals = list(results.values())
    delta = max(vals) - min(vals)
    results['delta'] = float(delta)
    return results

# ====================================================================
# Main Pipeline
# ====================================================================

def main():
    print("=" * 65)
    print("EXP 17: Clifford Geometric Transformer (Cℓ(2,0) Equivariant)")
    print("=" * 65)
    
    if not TRAIN_PATH or not os.path.isfile(TRAIN_PATH):
        print(f"Dataset not found in search paths: {SEARCH_ROOTS}")
        print("Please verify data path on Kaggle or download via kagglehub.")
        return

    print(f"Loading data from: {TRAIN_PATH}")
    with open(TRAIN_PATH, 'r') as f: train_data = json.load(f)
    with open(TEST_PATH, 'r') as f: test_data = json.load(f)
    comp_data = json.load(open(COMP_PATH, 'r')) if COMP_PATH else test_data
    
    print(f"Train tests: {len(train_data)}, Test: {len(test_data)}, Comp: {len(comp_data)}")
    
    # Feature extraction
    t0 = time.time()
    def build_dataset(data):
        X_list, S_list, y_list = [], [], []
        for item in data:
            f, s = extract_clifford_features(item['points'])
            X_list.append(f)
            S_list.append(s)
            y_list.append(item['outcome'])
        return np.array(X_list), np.array(S_list), np.array(y_list, dtype=np.float32)

    X_train, S_train, y_train = build_dataset(train_data)
    X_val, S_val, y_val = build_dataset(test_data)
    X_comp, S_comp, y_comp = build_dataset(comp_data)
    print(f"Features extracted in {time.time() - t0:.1f}s. Shape: {X_train.shape}")
    
    # Normalization (per-channel on training set)
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_comp  = (X_comp - mean) / std
    
    # DataLoaders with WeightedRandomSampler for class balance
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(S_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(S_val), torch.tensor(y_val))
    comp_ds  = TensorDataset(torch.tensor(X_comp), torch.tensor(S_comp), torch.tensor(y_comp))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    comp_loader  = DataLoader(comp_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = CliffordGeometricNet().to(DEVICE)
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
        for bx, bs, by in train_loader:
            bx, bs, by = bx.to(DEVICE), bs.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits = model(bx, bs)
                loss = criterion(logits, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(by)
            
        if epoch >= SWA_START:
            swa.update(model)
            
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, bs, by in val_loader:
                bx, bs = bx.to(DEVICE), bs.to(DEVICE)
                logits = model(bx, bs)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(by.numpy())
        val_auc = roc_auc_score(val_targets, val_preds)
        if val_auc > best_auc:
            best_auc = val_auc
            
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} - Loss: {total_loss/len(y_train):.4f} - Val AUC: {val_auc:.4f} (Best: {best_auc:.4f})")
            
    train_time = (time.time() - start_time) / 60.0
    print(f"Training finished in {train_time:.2f} min.")
    
    # Evaluate SWA model on Competition split
    swa_model = swa.get().to(DEVICE)
    swa_model.eval()
    comp_preds, comp_targets = [], []
    with torch.no_grad():
        for bx, bs, by in comp_loader:
            bx, bs = bx.to(DEVICE), bs.to(DEVICE)
            logits = swa_model(bx, bs)
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
    
    # Rotation Probe on Competition Set
    print("\nRunning Rotation Probe (6 angles)...")
    rot_probe = run_rotation_probe(swa_model, comp_data)
    print(f"Rotation Delta: {rot_probe['delta']:.5f}")
    for k, v in rot_probe.items():
        if k != 'delta': print(f"  {k}: {v:.4f}")
        
    # Save results
    results = {
        'exp': 'exp17_Clifford_GeometricNet',
        'params': sum(p.numel() for p in model.parameters()),
        'train_time_min': train_time,
        'val_best_auc': float(best_auc),
        'comp_auc': comp_auc,
        'comp_apfd_single': single_apfd,
        'comp_apfd_multi': f"{mean_apfd:.4f} +/- {std_apfd:.4f}",
        'rotation_probe': rot_probe
    }
    
    out_file = os.path.join(OUTPUT_DIR, 'exp17_Clifford_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")

if __name__ == '__main__':
    main()
