"""
BEST-OF-RECIPE on ALL benchmarks (self-contained, single file)
==============================================================
Story for ICSE 2027 oral: ONE recipe, MANY benchmarks. The SensoDat-tuned
configuration (Transformer + SWA + Focal gamma=2.5) is re-run on every
geometry benchmark we have access to, plus a LightGBM equivalent for the
pre-tabulated SDC-Pririotizer-RP datasets.

What's covered
--------------
GEOMETRY (Transformer + SWA + Focal gamma=2.5, 75 epochs, batch=256):
  - SensoDat                                (8 raw .json corpora pooled)
  - Dataset-OOB-0-1, OOB-0-3, OOB-0-5       (Zenodo 16939865)
  - SDC-Scissor sample_tests                (Zenodo 5914130, 5-fold CV)
  - sdc-travel competition                  (66 generator campaigns pooled)

TABULAR (LightGBM, fallback HistGradientBoosting, 5-fold CV):
  - SDC-Pririotizer-RP / BeamNG_RF_1 / RF_1_5 / RF_2 / DriverAI

Each benchmark uses its native split protocol (80/20, k-fold, or transfer)
and reports APFD with a comparable multi-trial protocol (30 trials, |S| =
30% of held-out, min 50) for geometry benches. RP benches use mean fold
APFD as in Birchler et al.

Hardware notes
--------------
On Kaggle T4: ~60-90 min end-to-end. On RTX PRO 6000 (Blackwell, 96 GB):
~25-35 min. The script will auto-skip any benchmark whose data folder is
not found, so partial runs are fine.

Saves: `best_all_results.json` summarising APFD per benchmark.
"""
import os, sys, json, time, math, copy, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

# ---------- Optional tabular dep ----------
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
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

SEARCH_ROOTS = [
    '/kaggle/input',
    os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', '..', 'data')),
    os.path.normpath(os.path.join(HERE, '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', 'data')),
    os.getcwd(),
]
OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.normpath(os.path.join(HERE, '..', '..', 'models'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")

# ---------- Common config ----------
SEQ_LEN     = 197
GAMMA       = 2.5      # SensoDat winning focal gamma
EPOCHS      = 75
BATCH       = 256
LR          = 5e-4
SWA_START   = 50       # epoch index (0-based)
N_TRIALS    = 30

# ====================================================================
# Geometry pipeline (shared by SensoDat / OOB / Scissor / Travel)
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

def extract_seq_10ch(pts_raw):
    pts = np.array(pts_raw, dtype=np.float64).reshape(-1, 2); n = len(pts)
    if n < 3:
        pts = np.vstack([pts] * 3)[:max(3, n)]; n = len(pts)
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
    print(f"\n--- Train {name} | gamma={gamma} | SWA@{swa_start} ---")
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

def compute_apfd(pids, td):
    n = len(pids)
    fp = [i + 1 for i, t in enumerate(pids) if td[t]['test_outcome'] == 'FAIL']
    m = len(fp); return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0

def predict_probs(model, X_norm):
    Xt = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    model.eval().to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(Xt)).cpu().numpy()

def multi_trial_apfd_geom(eval_data, model, means, stds, n_trials=N_TRIALS, name=''):
    sample_size = max(50, int(0.3 * len(eval_data)))
    sample_size = min(sample_size, len(eval_data))
    feats = np.array([resample(extract_seq_10ch(tc['road_points'])) for tc in eval_data],
                     dtype=np.float32)
    feats = (feats - means) / stds
    apfds = []
    for t in range(n_trials):
        rng = np.random.RandomState(42 + t)
        idx = rng.permutation(len(eval_data))[:sample_size]
        ed = [eval_data[i] for i in idx]
        td = {tc['_id']: tc for tc in ed}
        ids = [tc['_id'] for tc in ed]
        probs = predict_probs(model, feats[idx])
        pids = [t for _, t in sorted(zip(probs, ids), key=lambda x: -x[0])]
        apfds.append(compute_apfd(pids, td))
    m_, s_ = float(np.mean(apfds)), float(np.std(apfds))
    print(f"  {name:40s} APFD={m_:.4f}+/-{s_:.4f} ({n_trials} trials, |S|={sample_size})")
    return m_, s_

def prepare_geom(data):
    X = np.array([resample(extract_seq_10ch(tc['road_points'])) for tc in data], dtype=np.float32)
    y = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data], dtype=np.int64)
    return X, y

def run_geom_split(train_data, test_data, name='', n_trials=N_TRIALS):
    """Train on train_data, evaluate (multi-trial APFD) on test_data."""
    print(f"  [{name}] Train: {len(train_data)} | Test: {len(test_data)}")
    X_tr, y_tr = prepare_geom(train_data)
    X_te, y_te = prepare_geom(test_data)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1))
    stds[stds < 1e-8] = 1.0
    X_tr_n = (X_tr - means) / stds; X_te_n = (X_te - means) / stds
    model, swa, auc = train_geometry(X_tr_n, y_tr, X_te_n, y_te, name=name)
    eval_model = swa if swa is not None else model
    apfd_mean, apfd_std = multi_trial_apfd_geom(test_data, eval_model, means, stds,
                                                 n_trials=n_trials, name=name + ' SWA')
    return {'auc': float(auc), 'apfd_mean': apfd_mean, 'apfd_std': apfd_std,
            'n_train': len(train_data), 'n_test': len(test_data)}

# ====================================================================
# Loaders (one per benchmark family)
# ====================================================================

def _walk_for(target_name, want_files=True, ext='.json'):
    """Find the first folder whose basename == target_name and that contains
    files with given ext (if want_files). Tolerates nested mounts."""
    seen = set()
    for root in SEARCH_ROOTS:
        if not root or not os.path.isdir(root) or root in seen: continue
        seen.add(root)
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == target_name:
                if not want_files or any(fn.endswith(ext) for fn in filenames):
                    return dirpath
                for d in dirnames:
                    inner = os.path.join(dirpath, d)
                    try:
                        if any(fn.endswith(ext) for fn in os.listdir(inner)):
                            return inner
                    except OSError:
                        continue
    return None

# --- SensoDat ---
def find_sensodat_root():
    """Sensodat: 8 corpora as sibling subfolders, each contains .json tests."""
    seen = set()
    for root in SEARCH_ROOTS:
        if not root or not os.path.isdir(root) or root in seen: continue
        seen.add(root)
        for dirpath, dirnames, _ in os.walk(root):
            base = os.path.basename(dirpath).lower()
            if 'sensodat' in base:
                # Heuristic: a sensodat root contains many .json or many subfolders.
                if any(d.endswith('.json') for d in os.listdir(dirpath)) or len(dirnames) >= 3:
                    return dirpath
    return None

def load_sensodat(root, log_every=2000):
    if not root or not os.path.isdir(root): return []
    data = []
    seen = 0
    for fp in glob.iglob(os.path.join(root, '**', '*.json'), recursive=True):
        seen += 1
        try:
            with open(fp) as f: tc = json.load(f)
        except Exception:
            continue
        if not tc.get('is_valid', True): continue
        pts = tc.get('road_points') or tc.get('interpolated_road_points') or tc.get('interpolated_points')
        out = tc.get('test_outcome')
        if not pts or out not in ('FAIL', 'PASS'): continue
        data.append({'_id': fp[len(root):].lstrip(os.sep), 'road_points': pts, 'test_outcome': out})
        if seen % log_every == 0:
            print(f"    sensodat parsed {seen}, kept {len(data)}")
    return data

# --- OOB ---
def load_oob_dir(path):
    files = sorted(glob.glob(os.path.join(path, '*.json')))
    data = []
    for fp in files:
        try:
            with open(fp) as f: tc = json.load(f)
        except Exception:
            continue
        if not tc.get('is_valid', True): continue
        pts = tc.get('road_points'); out = tc.get('test_outcome')
        if not pts or out not in ('FAIL', 'PASS'): continue
        data.append({'_id': os.path.basename(fp), 'road_points': pts, 'test_outcome': out})
    return data

# --- Scissor ---
def find_scissor_root():
    seen = set()
    for root in SEARCH_ROOTS:
        if not root or not os.path.isdir(root) or root in seen: continue
        seen.add(root)
        for dirpath, _, filenames in os.walk(root):
            jsons = [fn for fn in filenames if fn.endswith('-test.json')]
            if len(jsons) >= 50:
                return dirpath
    return None

def load_scissor(root):
    if not root: return []
    files = sorted(glob.glob(os.path.join(root, '*-test.json')))
    data = []
    for fp in files:
        try:
            with open(fp) as f: tc = json.load(f)
        except Exception:
            continue
        if tc.get('is_valid', True) is False: continue
        pts = tc.get('interpolated_road_points') or tc.get('road_points')
        out = tc.get('test_outcome')
        if not pts or out not in ('FAIL', 'PASS'): continue
        data.append({'_id': os.path.basename(fp), 'road_points': pts, 'test_outcome': out})
    return data

# --- Travel ---
def find_travel_root():
    seen = set()
    for root in SEARCH_ROOTS:
        if not root or not os.path.isdir(root) or root in seen: continue
        seen.add(root)
        for dirpath, dirnames, _ in os.walk(root):
            if os.path.basename(dirpath) == 'competition':
                for d in dirnames[:3]:
                    if 'generator' in d.lower():
                        return dirpath
    return None

def load_travel(root):
    if not root: return []
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

# --- RP (tabular) ---
def find_rp_base():
    seen = set()
    for root in SEARCH_ROOTS:
        if not root or not os.path.isdir(root) or root in seen: continue
        seen.add(root)
        for dirpath, dirnames, _ in os.walk(root):
            if 'datasets' in dirnames and os.path.basename(dirpath) == 'SDC-Pririotizer-RP':
                return dirpath
    return None

def find_rp_csv(base, rel):
    if not base: return None
    direct = os.path.join(base, rel)
    if os.path.isfile(direct): return direct
    target = os.path.basename(rel)
    for dirpath, _, filenames in os.walk(base):
        if target in filenames:
            return os.path.join(dirpath, target)
    return None

# ====================================================================
# Per-benchmark drivers
# ====================================================================

def stratified_split(data, test_size=0.2, seed=42):
    y = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data])
    idx_tr, idx_te = train_test_split(np.arange(len(data)), test_size=test_size,
                                       stratify=y, random_state=seed)
    return [data[i] for i in idx_tr], [data[i] for i in idx_te]

def bench_sensodat():
    print(f"\n{'='*70}\nSensoDat\n{'='*70}")
    root = find_sensodat_root()
    if not root:
        print("  [SKIP] SensoDat root not found"); return {'status': 'missing'}
    print(f"  root: {root}")
    data = load_sensodat(root)
    nf = sum(tc['test_outcome'] == 'FAIL' for tc in data)
    print(f"  N={len(data)} | FAIL={nf} ({100*nf/max(1,len(data)):.1f}%)")
    if len(data) < 100 or nf < 20:
        return {'status': 'too_small', 'n': len(data), 'n_fail': nf}
    tr, te = stratified_split(data)
    return run_geom_split(tr, te, name='SensoDat')

def bench_oob_within():
    print(f"\n{'='*70}\nOOB-Regression (within-threshold)\n{'='*70}")
    out = {}
    for tag in ('0-1', '0-3', '0-5'):
        path = _walk_for(f'Dataset-OOB-{tag}')
        if not path:
            print(f"  [SKIP] OOB-{tag} not found"); out[tag] = {'status': 'missing'}; continue
        print(f"  OOB-{tag}: {path}")
        data = load_oob_dir(path)
        nf = sum(tc['test_outcome'] == 'FAIL' for tc in data)
        print(f"    N={len(data)} | FAIL={nf}")
        if len(data) < 100 or nf < 20:
            out[tag] = {'status': 'too_small', 'n': len(data), 'n_fail': nf}; continue
        tr, te = stratified_split(data)
        out[tag] = run_geom_split(tr, te, name=f'OOB-{tag}')
    return out

def bench_scissor():
    print(f"\n{'='*70}\nSDC-Scissor (5-fold CV)\n{'='*70}")
    root = find_scissor_root()
    if not root:
        print("  [SKIP] scissor root not found"); return {'status': 'missing'}
    print(f"  root: {root}")
    data = load_scissor(root)
    nf = sum(tc['test_outcome'] == 'FAIL' for tc in data)
    print(f"  N={len(data)} | FAIL={nf}")
    if len(data) < 20 or nf < 5 or len(data) - nf < 5:
        return {'status': 'too_small', 'n': len(data), 'n_fail': nf}
    y_all = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for fk, (tr_idx, te_idx) in enumerate(skf.split(np.arange(len(data)), y_all)):
        tr = [data[i] for i in tr_idx]; te = [data[i] for i in te_idx]
        fr = run_geom_split(tr, te, name=f'Scissor fold{fk+1}', n_trials=15)
        folds.append(fr)
    apfds = [fr['apfd_mean'] for fr in folds]
    return {'folds': folds, 'apfd_mean': float(np.mean(apfds)),
            'apfd_std': float(np.std(apfds))}

def bench_travel():
    print(f"\n{'='*70}\nsdc-travel (pooled 80/20)\n{'='*70}")
    root = find_travel_root()
    if not root:
        print("  [SKIP] travel root not found"); return {'status': 'missing'}
    print(f"  root: {root}")
    data = load_travel(root)
    nf = sum(tc['test_outcome'] == 'FAIL' for tc in data)
    print(f"  N={len(data)} | FAIL={nf} ({100*nf/max(1,len(data)):.2f}%)")
    if len(data) < 100 or nf < 20:
        return {'status': 'too_small', 'n': len(data), 'n_fail': nf}
    tr, te = stratified_split(data)
    return run_geom_split(tr, te, name='Travel')

def bench_rp():
    print(f"\n{'='*70}\nSDC-Pririotizer-RP (LightGBM 5-fold)\n{'='*70}")
    if not HAVE_PD:
        print("  [SKIP] pandas not available"); return {'status': 'no_pandas'}
    base = find_rp_base()
    if not base:
        print("  [SKIP] RP base not found"); return {'status': 'missing'}
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
        path = find_rp_csv(base, rel)
        if not path or not os.path.isfile(path):
            print(f"  [SKIP] {name}: {rel}"); out[name] = {'status': 'missing'}; continue
        df = pd.read_csv(path)
        y = (df[LABEL_COL].astype(str).str.lower() == 'unsafe').astype(int).values
        feat_cols = [c for c in df.columns if c not in DROP_COLS]
        X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
        n_pos = int(y.sum()); n = len(y)
        print(f"  {name}: N={n} FAIL={n_pos} ({100*n_pos/n:.1f}%) feats={len(feat_cols)}")
        if n_pos < 5 or (n - n_pos) < 5:
            out[name] = {'status': 'too_imbalanced', 'n': n, 'n_fail': n_pos}; continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        apfds, aucs = [], []
        for fk, (tr, te) in enumerate(skf.split(X, y)):
            if HAVE_LGB:
                clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                         min_data_in_leaf=10, subsample=0.9, colsample_bytree=0.9,
                                         class_weight='balanced', random_state=42, verbosity=-1)
            else:
                clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                      max_leaf_nodes=63, random_state=42)
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
        out[name] = {'apfd_mean': float(np.mean(apfds)), 'apfd_std': float(np.std(apfds)),
                     'auc_mean': float(np.nanmean(aucs)), 'n': n, 'n_fail': n_pos,
                     'classifier': 'LightGBM' if HAVE_LGB else 'HistGradientBoosting'}
        print(f"    ★ {name}: APFD={out[name]['apfd_mean']:.4f}+/-{out[name]['apfd_std']:.4f}")
    return out

# ====================================================================
# Driver
# ====================================================================

def main():
    t0 = time.time()
    results = {'recipe': 'Transformer + SWA + Focal(gamma=2.5), 75 ep, batch=256',
               'tabular_recipe': 'LightGBM' if HAVE_LGB else 'HistGradientBoosting',
               'epochs': EPOCHS, 'gamma': GAMMA}

    # Order: cheapest -> most expensive so partial runs still produce something.
    benches = [
        ('scissor',       bench_scissor),
        ('rp_external',   bench_rp),
        ('oob_within',    bench_oob_within),
        ('sensodat',      bench_sensodat),
        ('travel',        bench_travel),
    ]

    for tag, fn in benches:
        try:
            results[tag] = fn()
        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] stopping after {tag}"); break
        except Exception as e:
            print(f"  [ERR] {tag}: {type(e).__name__}: {e}")
            results[tag] = {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
        # Save after each bench so partial runs are not lost.
        out = os.path.join(OUTPUT_DIR, 'best_all_results.json')
        with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
        print(f"  [save] {out}")

    print(f"\n{'='*70}\nBEST-OF-RECIPE SUMMARY\n{'='*70}")
    def _print_apfd(name, blob):
        if isinstance(blob, dict) and 'apfd_mean' in blob:
            print(f"  {name:>20s}: APFD = {blob['apfd_mean']:.4f}+/-{blob['apfd_std']:.4f}")
        elif isinstance(blob, dict) and 'status' in blob:
            print(f"  {name:>20s}: {blob['status']}")
    for tag, _ in benches:
        blob = results.get(tag)
        if isinstance(blob, dict):
            if 'apfd_mean' in blob:
                _print_apfd(tag, blob)
            else:
                for sub, sb in blob.items():
                    _print_apfd(f'{tag}/{sub}', sb)
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == '__main__':
    main()
