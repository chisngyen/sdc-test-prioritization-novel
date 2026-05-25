"""
FULL-RUN on ALL benchmarks (one file, every dataset under input/datasets)
=========================================================================
Companion to `exps/best_all/exp_best_all.py`. Same recipe, but:

  * Hardcoded paths to /kaggle/input/datasets/{chiboiz,chinguyeen}/...
    No walking, no fallback. If a dataset is missing the script prints
    [SKIP] and moves on, but the intended layout is fixed.
  * its4sdc is a FIRST-CLASS benchmark (not the OOB-Regression fallback).
  * "Full" = no subsampling. Training uses the entire train split, and
    APFD is reported on the *entire* held-out test set in addition to
    the 30%-trial protocol (kept so numbers stay comparable to the
    SensoDat leaderboard in exps/tracker.md).
  * Per-benchmark report includes: N, FAIL%, n_train, n_test, AUC,
    APFD-full (single pass), APFD-trial (30 trials, sample=30%).

Geometry pipeline (sensodat / scissor / its4sdc / travel):
  Transformer (10ch, d=128, 4L) + SWA + Focal(gamma=2.5), 75ep, batch=256.

Tabular pipeline (sdc-pririotizer-rp):
  LightGBM (fallback HistGradientBoosting), 5-fold CV.

Saves `full_all_results.json` after each benchmark so partial runs are
not lost.
"""
import os, sys, json, time, math, copy, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import pandas as pd
    HAVE_PD = True
except Exception:
    HAVE_PD = False

# ---------- Paths ----------
DATA_ROOT = '/kaggle/input/datasets'
PATHS = {
    'sensodat':  os.path.join(DATA_ROOT, 'chinguyeen', 'sdc-sensodat'),
    'scissor':   os.path.join(DATA_ROOT, 'chinguyeen', 'sdc-scissor',
                              'christianbirchler-org-sdc-scissor-faf11b2',
                              'sample_tests'),
    'its4sdc':   os.path.join(DATA_ROOT, 'chiboiz', 'its4sdc', 'executed-10000'),
    'travel':    os.path.join(DATA_ROOT, 'chiboiz', 'sdc-travel', 'competition'),
    'rp_base':   os.path.join(DATA_ROOT, 'chiboiz', 'sdc-pririotizer-rp',
                              'SDC-Pririotizer-RP'),
}
OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")

# ---------- Config ----------
SEQ_LEN     = 197
GAMMA       = 2.5
EPOCHS      = 75
BATCH       = 256
LR          = 5e-4
SWA_START   = 50
N_TRIALS    = 30
SEED        = 42

# ====================================================================
# Geometry feature extractor (identical to exp_best_all.py)
# ====================================================================

def _compute_curvature(pts):
    n = len(pts); curv = np.zeros(n - 2)
    for i in range(n - 2):
        x1, y1 = pts[i]; x2, y2 = pts[i + 1]; x3, y3 = pts[i + 2]
        a = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        s = 0.5 * (a + b + c); at = s * (s - a) * (s - b) * (s - c)
        if at <= 1e-10:
            curv[i] = 0.0
        else:
            R = a * b * c / (4 * math.sqrt(at))
            curv[i] = 1.0 / R if R > 0 else 0.0
    return curv

def _normalize_points(pts_raw):
    if not pts_raw: return np.zeros((0, 2), dtype=np.float64)
    first = pts_raw[0]
    if isinstance(first, dict):
        return np.array([[p['x'], p['y']] for p in pts_raw], dtype=np.float64)
    arr = np.asarray(pts_raw, dtype=np.float64)
    if arr.ndim == 1: arr = arr.reshape(-1, 2)
    elif arr.ndim == 2 and arr.shape[1] >= 2: arr = arr[:, :2]
    return arr

def extract_seq_10ch(pts_raw):
    pts = _normalize_points(pts_raw); n = len(pts)
    if n < 3:
        pts = np.vstack([pts] * 3)[:max(3, n)] if n else np.zeros((3, 2)); n = len(pts)
    diffs = np.diff(pts, axis=0); seg_lens = np.linalg.norm(diffs, axis=1)
    seg_full = np.pad(seg_lens, (0, 1), mode='edge')
    angles = np.arctan2(diffs[:, 1], diffs[:, 0]); ac = np.diff(angles)
    ac = (ac + np.pi) % (2 * np.pi) - np.pi
    abs_ac_full = np.pad(np.abs(ac), (1, 1), mode='constant')
    curv = np.abs(_compute_curvature(pts))
    curv_full = np.pad(curv, (1, 1), mode='constant')
    curv_deriv_full = np.pad(np.diff(curv_full), (0, 1), mode='constant')
    cum_dist = np.cumsum(seg_full)
    cum_dist_norm = cum_dist / (cum_dist[-1] + 1e-8)
    heading_full = np.pad(angles, (0, 1), mode='edge')
    heading_sin = np.sin(heading_full); heading_cos = np.cos(heading_full)
    rel_pos = np.linspace(0, 1, n)
    w = 11; local_std = np.zeros(n); hw = w // 2
    for i in range(n):
        s, e = max(0, i - hw), min(n, i + hw + 1)
        local_std[i] = np.std(curv_full[s:e])
    curv_accel_full = np.pad(np.diff(curv_deriv_full), (0, 1), mode='constant')
    return np.column_stack([seg_full, abs_ac_full, curv_full, curv_deriv_full, cum_dist_norm,
                            heading_sin, heading_cos, rel_pos, local_std, curv_accel_full]).astype(np.float32)

def resample(seq, target_len=SEQ_LEN):
    n, c = seq.shape
    if n == target_len: return seq
    x_old = np.linspace(0, 1, n); x_new = np.linspace(0, 1, target_len)
    out = np.empty((target_len, c), dtype=np.float32)
    for ch in range(c):
        out[:, ch] = np.interp(x_new, x_old, seq[:, ch])
    return out

# ====================================================================
# Model / loss / SWA / training
# ====================================================================

class RoadTransformer(nn.Module):
    def __init__(self, in_channels=10, seq_len=SEQ_LEN, d_model=128,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model),
                                         nn.LayerNorm(d_model), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        x = x.permute(0, 2, 1); B, L, C = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :L + 1, :]
        x = self.transformer(x)
        return self.classifier(x[:, 0, :]).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.alpha, self.gamma, self.pos_weight = alpha, gamma, pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weight = torch.where(targets == 1, self.pos_weight, 1.0)
        bce = bce * weight
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

class SWAModel:
    def __init__(self, model):
        self.model = copy.deepcopy(model); self.n = 0
    def update(self, new_model):
        self.n += 1; alpha = 1.0 / self.n
        for p_swa, p_new in zip(self.model.parameters(), new_model.parameters()):
            p_swa.data.mul_(1 - alpha).add_(p_new.data, alpha=alpha)
    def get_model(self): return self.model

def train_geometry(X_train, y_train, X_val, y_val,
                   epochs=EPOCHS, batch=BATCH, lr=LR,
                   gamma=GAMMA, swa_start=SWA_START, name=''):
    print(f"\n--- Train {name} | gamma={gamma} | SWA@{swa_start} | ep={epochs} bs={batch} ---")
    model = RoadTransformer(in_channels=10, seq_len=SEQ_LEN).to(DEVICE)
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pw = float(n_neg) / max(1, n_pos)
    weights = np.where(y_train == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    X_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch,
                          sampler=sampler, num_workers=2, pin_memory=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = FocalLoss(alpha=1.0, gamma=gamma, pos_weight=pw)
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    best_auc = 0; best_state = None; swa_model = None

    for epoch in range(epochs):
        model.train(); total_loss = 0; nb = 0
        for xb, yb in train_dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): loss = criterion(model(xb), yb)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); nb += 1
        scheduler.step()
        if epoch >= swa_start:
            if swa_model is None: swa_model = SWAModel(model)
            else: swa_model.update(model)
        model.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp): vl = model(X_v)
            try:
                val_auc = roc_auc_score(y_val, torch.sigmoid(vl).cpu().numpy())
            except Exception:
                val_auc = 0.5
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 15 == 0:
            print(f"  Ep {epoch+1:3d} | Loss:{total_loss/nb:.4f} | AUC:{val_auc:.4f} | Best:{best_auc:.4f}")

    model.load_state_dict(best_state)
    swa = swa_model.get_model().to(DEVICE) if swa_model else None
    return model, swa, best_auc

def compute_apfd_from_pids(pids, td):
    n = len(pids)
    fp = [i + 1 for i, t in enumerate(pids) if td[t]['test_outcome'] == 'FAIL']
    m = len(fp); return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0

def predict_probs(model, X_norm):
    Xt = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    model.eval().to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(Xt)).cpu().numpy()

def full_apfd_geom(eval_data, model, feats):
    """Single-pass APFD on the ENTIRE held-out test set (no subsampling)."""
    td = {tc['_id']: tc for tc in eval_data}
    ids = [tc['_id'] for tc in eval_data]
    probs = predict_probs(model, feats)
    pids = [t for _, t in sorted(zip(probs, ids), key=lambda x: -x[0])]
    return compute_apfd_from_pids(pids, td), probs

def multi_trial_apfd_geom(eval_data, feats, probs, n_trials=N_TRIALS, frac=0.3, min_size=50):
    """Multi-trial APFD on |S| = max(min_size, frac * |test|) random subsets;
    reuses pre-computed probs so it is essentially free."""
    sample_size = max(min_size, int(frac * len(eval_data)))
    sample_size = min(sample_size, len(eval_data))
    apfds = []
    for t in range(n_trials):
        rng = np.random.RandomState(SEED + t)
        idx = rng.permutation(len(eval_data))[:sample_size]
        ed = [eval_data[i] for i in idx]
        td = {tc['_id']: tc for tc in ed}
        ids = [tc['_id'] for tc in ed]
        sub_probs = probs[idx]
        pids = [t for _, t in sorted(zip(sub_probs, ids), key=lambda x: -x[0])]
        apfds.append(compute_apfd_from_pids(pids, td))
    return float(np.mean(apfds)), float(np.std(apfds)), sample_size

def prepare_geom(data):
    X = np.array([resample(extract_seq_10ch(tc['road_points'])) for tc in data], dtype=np.float32)
    y = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data], dtype=np.int64)
    return X, y

def run_geom_split(train_data, test_data, name=''):
    print(f"  [{name}] Train: {len(train_data)} | Test: {len(test_data)}")
    X_tr, y_tr = prepare_geom(train_data)
    X_te, y_te = prepare_geom(test_data)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1))
    stds[stds < 1e-8] = 1.0
    X_tr_n = (X_tr - means) / stds; X_te_n = (X_te - means) / stds
    model, swa, auc = train_geometry(X_tr_n, y_tr, X_te_n, y_te, name=name)
    eval_model = swa if swa is not None else model
    apfd_full, probs = full_apfd_geom(test_data, eval_model, X_te_n)
    apfd_mean, apfd_std, sample_size = multi_trial_apfd_geom(test_data, X_te_n, probs)
    print(f"  {name:30s} AUC={auc:.4f} | APFD_full={apfd_full:.4f} "
          f"| APFD_trial={apfd_mean:.4f}+/-{apfd_std:.4f} (|S|={sample_size})")
    return {'auc': float(auc),
            'apfd_full': float(apfd_full),
            'apfd_trial_mean': apfd_mean,
            'apfd_trial_std': apfd_std,
            'sample_size': sample_size,
            'n_train': len(train_data),
            'n_test': len(test_data),
            'n_fail_train': int(y_tr.sum()),
            'n_fail_test': int(y_te.sum())}

# ====================================================================
# Loaders -- explicit paths under /kaggle/input/datasets/
# ====================================================================

def load_sensodat(root):
    if not root or not os.path.isdir(root): return []
    full = os.path.join(root, 'sensodat_full.json')
    candidates = [full] if os.path.isfile(full) else [
        p for p in (os.path.join(root, n) for n in ('sensodat_train.json', 'sensodat_test.json'))
        if os.path.isfile(p)
    ]
    if not candidates:
        print(f"  [WARN] no sensodat_*.json under {root}"); return []
    data = []
    for fp in candidates:
        try:
            with open(fp) as f: items = json.load(f)
        except Exception as e:
            print(f"  [WARN] {fp}: {type(e).__name__}: {e}"); continue
        if not isinstance(items, list): continue
        kept_before = len(data)
        for tc in items:
            md = tc.get('meta_data') or {}
            ti = md.get('test_info') or {}
            if isinstance(ti, str):
                try: import ast; ti = ast.literal_eval(ti)
                except Exception: continue
            if ti.get('is_valid') is False: continue
            out = ti.get('test_outcome')
            if out not in ('FAIL', 'PASS'): continue
            pts = tc.get('road_points')
            if not pts: continue
            _id = tc.get('_id')
            if isinstance(_id, dict): _id = _id.get('$oid') or str(_id)
            data.append({'_id': _id, 'road_points': pts, 'test_outcome': out})
        print(f"    {os.path.basename(fp)}: kept {len(data) - kept_before} / total {len(items)}")
    return data

def load_flat_json_dir(path, pattern='*.json'):
    """Loader for scissor/its4sdc/OOB-style flat dirs of per-test JSONs."""
    files = sorted(glob.glob(os.path.join(path, pattern)))
    data = []
    for fp in files:
        try:
            with open(fp) as f: tc = json.load(f)
        except Exception:
            continue
        if tc.get('is_valid', True) is False: continue
        pts = tc.get('road_points') or tc.get('interpolated_road_points')
        out = tc.get('test_outcome')
        if not pts or out not in ('FAIL', 'PASS'): continue
        data.append({'_id': os.path.basename(fp), 'road_points': pts, 'test_outcome': out})
    return data

def load_travel(root):
    if not root or not os.path.isdir(root): return []
    data = []
    for camp in sorted(os.listdir(root)):
        cp = os.path.join(root, camp)
        if not os.path.isdir(cp): continue
        for fp in glob.glob(os.path.join(cp, 'test.*.json')):
            try:
                with open(fp) as f: tc = json.load(f)
            except Exception:
                continue
            if not tc.get('is_valid', True): continue
            pts = tc.get('interpolated_points') or tc.get('road_points')
            out = tc.get('test_outcome')
            if not pts or out not in ('FAIL', 'PASS'): continue
            data.append({'_id': f'{camp}/{os.path.basename(fp)}',
                         'campaign': camp, 'road_points': pts, 'test_outcome': out})
    return data

# ====================================================================
# Driver helpers
# ====================================================================

def stratified_split(data, test_size=0.2, seed=SEED):
    y = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data])
    idx_tr, idx_te = train_test_split(np.arange(len(data)), test_size=test_size,
                                       stratify=y, random_state=seed)
    return [data[i] for i in idx_tr], [data[i] for i in idx_te]

def bench_summary(data, label):
    nf = sum(tc['test_outcome'] == 'FAIL' for tc in data)
    print(f"  [{label}] N={len(data)} | FAIL={nf} ({100*nf/max(1,len(data)):.2f}%)")
    return nf

# ====================================================================
# Per-benchmark drivers
# ====================================================================

def bench_sensodat():
    print(f"\n{'='*70}\nSensoDat (full data, 80/20)\n{'='*70}")
    root = PATHS['sensodat']
    if not os.path.isdir(root):
        print(f"  [SKIP] {root}"); return {'status': 'missing'}
    print(f"  root: {root}")
    data = load_sensodat(root)
    nf = bench_summary(data, 'SensoDat')
    if len(data) < 100 or nf < 20:
        return {'status': 'too_small', 'n': len(data), 'n_fail': nf}
    tr, te = stratified_split(data)
    return run_geom_split(tr, te, name='SensoDat')

def bench_scissor():
    print(f"\n{'='*70}\nSDC-Scissor (5-fold CV, full data)\n{'='*70}")
    root = PATHS['scissor']
    if not os.path.isdir(root):
        print(f"  [SKIP] {root}"); return {'status': 'missing'}
    print(f"  root: {root}")
    data = load_flat_json_dir(root, pattern='*-test.json')
    nf = bench_summary(data, 'Scissor')
    if len(data) < 20 or nf < 5 or len(data) - nf < 5:
        return {'status': 'too_small', 'n': len(data), 'n_fail': nf}
    y_all = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for fk, (tr_idx, te_idx) in enumerate(skf.split(np.arange(len(data)), y_all)):
        tr = [data[i] for i in tr_idx]; te = [data[i] for i in te_idx]
        folds.append(run_geom_split(tr, te, name=f'Scissor f{fk+1}'))
    return {'folds': folds,
            'auc_mean':         float(np.mean([f['auc']             for f in folds])),
            'apfd_full_mean':   float(np.mean([f['apfd_full']       for f in folds])),
            'apfd_full_std':    float(np.std ([f['apfd_full']       for f in folds])),
            'apfd_trial_mean':  float(np.mean([f['apfd_trial_mean'] for f in folds])),
            'apfd_trial_std':   float(np.mean([f['apfd_trial_std']  for f in folds]))}

def bench_its4sdc():
    print(f"\n{'='*70}\nits4sdc (executed-10000, full data, 80/20)\n{'='*70}")
    root = PATHS['its4sdc']
    if not os.path.isdir(root):
        print(f"  [SKIP] {root}"); return {'status': 'missing'}
    print(f"  root: {root}")
    data = load_flat_json_dir(root, pattern='*.json')
    nf = bench_summary(data, 'its4sdc')
    if len(data) < 100 or nf < 20:
        return {'status': 'too_small', 'n': len(data), 'n_fail': nf}
    tr, te = stratified_split(data)
    return run_geom_split(tr, te, name='its4sdc')

def bench_travel():
    print(f"\n{'='*70}\nsdc-travel (66 campaigns pooled, 80/20)\n{'='*70}")
    root = PATHS['travel']
    if not os.path.isdir(root):
        print(f"  [SKIP] {root}"); return {'status': 'missing'}
    print(f"  root: {root}")
    data = load_travel(root)
    nf = bench_summary(data, 'Travel')
    if len(data) < 100 or nf < 20:
        return {'status': 'too_small', 'n': len(data), 'n_fail': nf}
    tr, te = stratified_split(data)
    return run_geom_split(tr, te, name='Travel')

def bench_rp():
    print(f"\n{'='*70}\nSDC-Pririotizer-RP (LightGBM 5-fold)\n{'='*70}")
    if not HAVE_PD:
        print("  [SKIP] pandas not available"); return {'status': 'no_pandas'}
    base = PATHS['rp_base']
    if not os.path.isdir(base):
        print(f"  [SKIP] {base}"); return {'status': 'missing'}
    print(f"  base: {base}")
    sets = {
        'BeamNG_RF_1':   'datasets/fullroad/BeamNG_AI/BeamNG_RF_1/BeamNG_RF_1_Complete.csv',
        'BeamNG_RF_1_5': 'datasets/fullroad/BeamNG_AI/BeamNG_RF_1_5/BeamNG_RF_1_5_Complete.csv',
        'BeamNG_RF_2':   'datasets/fullroad/BeamNG_AI/BeamNG_RF_2/BeamNG_RF_2_Complete.csv',
        'DriverAI':      'datasets/fullroad/Driver_AI/DriverAI_Complete.csv',
    }
    out = {}
    LABEL_COL = 'safety'
    DROP_COLS = {'start_time', 'end_time', LABEL_COL}
    for name, rel in sets.items():
        path = os.path.join(base, rel)
        if not os.path.isfile(path):
            print(f"  [SKIP] {name}: {rel}"); out[name] = {'status': 'missing'}; continue
        df = pd.read_csv(path)
        y = (df[LABEL_COL].astype(str).str.lower() == 'unsafe').astype(int).values
        feat_cols = [c for c in df.columns if c not in DROP_COLS]
        X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
        n_pos = int(y.sum()); n = len(y)
        print(f"  {name}: N={n} FAIL={n_pos} ({100*n_pos/n:.1f}%) feats={len(feat_cols)}")
        if n_pos < 5 or (n - n_pos) < 5:
            out[name] = {'status': 'too_imbalanced', 'n': n, 'n_fail': n_pos}; continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        apfds, aucs = [], []
        for fk, (tr, te) in enumerate(skf.split(X, y)):
            if HAVE_LGB:
                clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                         min_data_in_leaf=10, subsample=0.9, colsample_bytree=0.9,
                                         class_weight='balanced', random_state=SEED, verbosity=-1)
            else:
                clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                      max_leaf_nodes=63, random_state=SEED)
            clf.fit(X[tr], y[tr])
            probs = clf.predict_proba(X[te])[:, 1]
            y_te = y[te]
            try: aucs.append(roc_auc_score(y_te, probs))
            except Exception: aucs.append(float('nan'))
            order = np.argsort(-probs)
            n_te = len(order); fp = [pos + 1 for pos, idx in enumerate(order) if y_te[idx] == 1]
            m_te = len(fp)
            apfd = 1 - sum(fp) / (n_te * m_te) + 1 / (2 * n_te) if n_te and m_te else 1.0
            apfds.append(apfd)
        out[name] = {'apfd_full_mean': float(np.mean(apfds)),
                     'apfd_full_std':  float(np.std(apfds)),
                     'auc_mean': float(np.nanmean(aucs)),
                     'n': n, 'n_fail': n_pos,
                     'classifier': 'LightGBM' if HAVE_LGB else 'HistGradientBoosting'}
        print(f"    * {name}: APFD={out[name]['apfd_full_mean']:.4f}+/-{out[name]['apfd_full_std']:.4f}")
    return out

# ====================================================================
# Main driver
# ====================================================================

def main():
    t0 = time.time()
    results = {
        'recipe_geom':    'Transformer + SWA + Focal(gamma=2.5), 75 ep, batch=256',
        'recipe_tabular': 'LightGBM' if HAVE_LGB else 'HistGradientBoosting',
        'epochs': EPOCHS, 'gamma': GAMMA, 'seq_len': SEQ_LEN,
        'data_root': DATA_ROOT, 'seed': SEED,
    }

    # cheap -> expensive ordering for partial-run safety
    benches = [
        ('scissor',  bench_scissor),
        ('rp',       bench_rp),
        ('its4sdc',  bench_its4sdc),
        ('sensodat', bench_sensodat),
        ('travel',   bench_travel),
    ]

    for tag, fn in benches:
        try:
            results[tag] = fn()
        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] stopping after {tag}"); break
        except Exception as e:
            print(f"  [ERR] {tag}: {type(e).__name__}: {e}")
            results[tag] = {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
        out = os.path.join(OUTPUT_DIR, 'full_all_results.json')
        with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
        print(f"  [save] {out}")

    print(f"\n{'='*70}\nFULL-RUN SUMMARY\n{'='*70}")
    def _print_row(name, blob):
        if not isinstance(blob, dict): return
        if 'apfd_full' in blob:
            print(f"  {name:>24s}: AUC={blob['auc']:.4f}  APFD_full={blob['apfd_full']:.4f}  "
                  f"APFD_trial={blob['apfd_trial_mean']:.4f}+/-{blob['apfd_trial_std']:.4f}  "
                  f"(n_te={blob['n_test']})")
        elif 'apfd_full_mean' in blob:
            tail = ''
            if 'auc_mean' in blob: tail = f"  AUC={blob['auc_mean']:.4f}"
            print(f"  {name:>24s}:{tail}  APFD={blob['apfd_full_mean']:.4f}+/-{blob['apfd_full_std']:.4f}")
        elif 'status' in blob:
            print(f"  {name:>24s}: {blob['status']}")

    for tag, _ in benches:
        blob = results.get(tag)
        if isinstance(blob, dict):
            if any(k in blob for k in ('apfd_full', 'apfd_full_mean')):
                _print_row(tag, blob)
            elif 'folds' in blob:
                _print_row(tag, blob)
            else:
                for sub, sb in blob.items():
                    _print_row(f'{tag}/{sub}', sb)

    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == '__main__':
    main()
