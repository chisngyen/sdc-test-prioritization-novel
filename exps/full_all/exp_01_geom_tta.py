"""
Exp 01 -- Geometric Test-Time Augmentation across ALL 5 benchmarks
==================================================================
Hypothesis: the SensoDat-winner Transformer is *approximately* (but not
exactly) SE(2)-equivariant. Exp 02 in `exps/` showed Delta=0.0000 with
an explicitly E2-equivariant model. The practical Transformer used in
the winner recipe is NOT E2-equivariant -- it sees `heading_sin`,
`heading_cos`, and absolute curvature signs. So averaging its
predictions across K geometric views of the same road *should* capture
the residual non-invariance and lift APFD for free at inference time.

TTA family used:
  - K_R rotations in {0, 60, 120, 180, 240, 300} degrees     (6)
  - Horizontal flip across the y axis (chirality swap)        (2)
  - Traversal reversal (drive the road backwards)             (2)
  -> 6 x 2 x 2 = 24 views per test road.

All three transforms are *label preserving* for SDC test failures:
  - Rotation: failure depends on intrinsic road geometry, not heading.
  - Reflection: the simulator's lane logic is symmetric across the y axis
    (we evaluate the same intrinsic curvature signal).
  - Reversal: failure happens at the same physical waypoints; only the
    order of traversal flips.

Each view is fed through the model independently; sigmoids are averaged
to produce the final ranking score. We report:
  - APFD with no TTA (1 view)              -- baseline
  - APFD with rotation TTA only            -- ablation 1
  - APFD with rot + flip + reverse (full)  -- headline

Distinct from Exp C (`exps/best_all/exp_C_multi_resolution_tta.py`),
which varies *resolution* N, not pose. Both can stack (future work).

Saves `exp_01_geom_tta_results.json`.
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

# ---------- Paths / config (mirror exp_full_all.py) ----------
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

SEQ_LEN, GAMMA, EPOCHS, BATCH, LR, SWA_START, N_TRIALS, SEED = 197, 2.5, 75, 256, 5e-4, 50, 30, 42

# TTA family
TTA_ROT_DEG  = [0, 60, 120, 180, 240, 300]   # 6
TTA_FLIPS    = [False, True]                 # 2 (mirror across y)
TTA_REVERSE  = [False, True]                 # 2 (reverse traversal)
# 6*2*2 = 24 views

# ====================================================================
# Geometric primitives + 10-channel feature extractor
# ====================================================================

def _compute_curvature(pts):
    n = len(pts); curv = np.zeros(n - 2)
    for i in range(n - 2):
        x1, y1 = pts[i]; x2, y2 = pts[i + 1]; x3, y3 = pts[i + 2]
        a = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        s = 0.5 * (a + b + c); at = s * (s - a) * (s - b) * (s - c)
        if at <= 1e-10: curv[i] = 0.0
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

def apply_tta(pts, rot_deg=0, flip=False, reverse=False):
    """Apply rotation + flip + reversal to Nx2 raw points (label preserving)."""
    if pts.shape[0] == 0: return pts
    p = pts.copy()
    if reverse: p = p[::-1].copy()
    if flip:    p[:, 0] = -p[:, 0]
    if rot_deg != 0:
        th = math.radians(rot_deg); c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        p = p @ R.T
    return p

def extract_seq_10ch(pts):
    n = len(pts)
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

def road_to_feat(pts_raw, rot_deg=0, flip=False, reverse=False):
    pts = _normalize_points(pts_raw)
    pts = apply_tta(pts, rot_deg=rot_deg, flip=flip, reverse=reverse)
    return resample(extract_seq_10ch(pts))

# ====================================================================
# Model / Focal / SWA / Training (identical to exp_full_all.py)
# ====================================================================

class RoadTransformer(nn.Module):
    def __init__(self, in_channels=10, seq_len=SEQ_LEN, d_model=128,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model),
                                         nn.LayerNorm(d_model), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout, activation='gelu',
                                         batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
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
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class SWAModel:
    def __init__(self, model):
        self.model = copy.deepcopy(model); self.n = 0
    def update(self, new_model):
        self.n += 1; alpha = 1.0 / self.n
        for p_swa, p_new in zip(self.model.parameters(), new_model.parameters()):
            p_swa.data.mul_(1 - alpha).add_(p_new.data, alpha=alpha)
    def get_model(self): return self.model

def train_geometry(X_train, y_train, X_val, y_val, name=''):
    print(f"\n--- Train {name} | gamma={GAMMA} | SWA@{SWA_START} | ep={EPOCHS} bs={BATCH} ---")
    model = RoadTransformer(in_channels=10, seq_len=SEQ_LEN).to(DEVICE)
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pw = float(n_neg) / max(1, n_pos)
    weights = np.where(y_train == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    X_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH,
                    sampler=sampler, num_workers=2, pin_memory=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warm = 5
    def lr_lambda(ep):
        if ep < warm: return (ep + 1) / warm
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, EPOCHS - warm))))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = FocalLoss(alpha=1.0, gamma=GAMMA, pos_weight=pw)
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    best_auc, best_state, swa = 0, None, None

    for ep in range(EPOCHS):
        model.train(); tl = 0; nb = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item(); nb += 1
        sched.step()
        if ep >= SWA_START:
            if swa is None: swa = SWAModel(model)
            else: swa.update(model)
        model.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp): vl = model(X_v)
            try: val_auc = roc_auc_score(y_val, torch.sigmoid(vl).cpu().numpy())
            except Exception: val_auc = 0.5
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 15 == 0:
            print(f"  Ep {ep+1:3d} | Loss:{tl/nb:.4f} | AUC:{val_auc:.4f} | Best:{best_auc:.4f}")
    model.load_state_dict(best_state)
    return model, swa.get_model().to(DEVICE) if swa else None, best_auc

# ====================================================================
# TTA inference + APFD
# ====================================================================

def compute_apfd_from_pids(pids, td):
    n = len(pids); fp = [i + 1 for i, t in enumerate(pids) if td[t]['test_outcome'] == 'FAIL']
    m = len(fp); return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0

def predict_probs(model, X_norm):
    Xt = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    model.eval().to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(Xt)).cpu().numpy()

def tta_probs(model, test_data, means, stds, tta_views, batch_size=512):
    """Average sigmoids across `tta_views` = list of (rot, flip, reverse)."""
    n = len(test_data)
    probs_sum = np.zeros(n, dtype=np.float64)
    for k, (rot, flp, rev) in enumerate(tta_views):
        feats = np.array([road_to_feat(tc['road_points'], rot_deg=rot, flip=flp, reverse=rev)
                          for tc in test_data], dtype=np.float32)
        feats = (feats - means) / stds
        view_probs = predict_probs(model, feats)
        probs_sum += view_probs
        if (k + 1) % 6 == 0:
            print(f"    TTA view {k+1}/{len(tta_views)} done")
    return probs_sum / len(tta_views)

def full_apfd_from_probs(test_data, probs):
    td = {tc['_id']: tc for tc in test_data}
    ids = [tc['_id'] for tc in test_data]
    pids = [t for _, t in sorted(zip(probs, ids), key=lambda x: -x[0])]
    return compute_apfd_from_pids(pids, td)

def trial_apfd_from_probs(test_data, probs, n_trials=N_TRIALS, frac=0.3, min_size=50):
    sample_size = max(min_size, int(frac * len(test_data)))
    sample_size = min(sample_size, len(test_data))
    apfds = []
    for t in range(n_trials):
        rng = np.random.RandomState(SEED + t)
        idx = rng.permutation(len(test_data))[:sample_size]
        td = {test_data[i]['_id']: test_data[i] for i in idx}
        ids = [test_data[i]['_id'] for i in idx]
        sub = probs[idx]
        pids = [t for _, t in sorted(zip(sub, ids), key=lambda x: -x[0])]
        apfds.append(compute_apfd_from_pids(pids, td))
    return float(np.mean(apfds)), float(np.std(apfds)), sample_size

# ====================================================================
# Geometry split + ablation driver
# ====================================================================

def prepare_train(data):
    X = np.array([road_to_feat(tc['road_points']) for tc in data], dtype=np.float32)
    y = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data], dtype=np.int64)
    return X, y

def run_geom_split_tta(train_data, test_data, name=''):
    print(f"  [{name}] Train: {len(train_data)} | Test: {len(test_data)}")
    X_tr, y_tr = prepare_train(train_data)
    X_te_plain = np.array([road_to_feat(tc['road_points']) for tc in test_data], dtype=np.float32)
    y_te = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in test_data], dtype=np.int64)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1))
    stds[stds < 1e-8] = 1.0
    X_tr_n = (X_tr - means) / stds; X_te_plain_n = (X_te_plain - means) / stds
    model, swa, auc = train_geometry(X_tr_n, y_tr, X_te_plain_n, y_te, name=name)
    eval_model = swa if swa is not None else model

    # Three TTA modes
    views_none = [(0, False, False)]
    views_rot  = [(r, False, False) for r in TTA_ROT_DEG]                 # 6
    views_full = [(r, f, v) for r in TTA_ROT_DEG
                            for f in TTA_FLIPS
                            for v in TTA_REVERSE]                        # 24

    print(f"  --- TTA inference ({name}) ---")
    out = {'auc': float(auc), 'n_train': len(train_data),
           'n_test': len(test_data), 'n_fail_test': int(y_te.sum())}
    for tag, views in [('no_tta', views_none), ('rot6', views_rot), ('full24', views_full)]:
        t0 = time.time()
        probs = tta_probs(eval_model, test_data, means, stds, views)
        apfd_full = full_apfd_from_probs(test_data, probs)
        apfd_tm, apfd_ts, sz = trial_apfd_from_probs(test_data, probs)
        dt = time.time() - t0
        out[tag] = {'n_views': len(views), 'apfd_full': float(apfd_full),
                    'apfd_trial_mean': apfd_tm, 'apfd_trial_std': apfd_ts,
                    'sample_size': sz, 'sec': round(dt, 1)}
        print(f"  {name:>15s}/{tag:<8s} views={len(views):3d} | "
              f"APFD_full={apfd_full:.4f} APFD_trial={apfd_tm:.4f}+/-{apfd_ts:.4f} ({dt:.1f}s)")

    # Delta-APFD vs no-TTA baseline (the headline number)
    base = out['no_tta']['apfd_full']
    out['delta_full24_vs_no_tta'] = float(out['full24']['apfd_full'] - base)
    out['delta_rot6_vs_no_tta']   = float(out['rot6']['apfd_full']   - base)
    return out

# ====================================================================
# Loaders -- identical to exp_full_all.py
# ====================================================================

def load_sensodat(root):
    if not root or not os.path.isdir(root): return []
    full = os.path.join(root, 'sensodat_full.json')
    candidates = [full] if os.path.isfile(full) else [
        p for p in (os.path.join(root, n) for n in ('sensodat_train.json', 'sensodat_test.json'))
        if os.path.isfile(p)
    ]
    if not candidates: return []
    data = []
    for fp in candidates:
        try:
            with open(fp) as f: items = json.load(f)
        except Exception: continue
        if not isinstance(items, list): continue
        kb = len(data)
        for tc in items:
            md = tc.get('meta_data') or {}; ti = md.get('test_info') or {}
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
        print(f"    {os.path.basename(fp)}: kept {len(data)-kb}")
    return data

def load_flat_json_dir(path, pattern='*.json'):
    files = sorted(glob.glob(os.path.join(path, pattern)))
    data = []
    for fp in files:
        try:
            with open(fp) as f: tc = json.load(f)
        except Exception: continue
        if tc.get('is_valid', True) is False: continue
        pts = tc.get('road_points') or tc.get('interpolated_road_points')
        out = tc.get('test_outcome')
        if not pts or out not in ('FAIL', 'PASS'): continue
        data.append({'_id': os.path.basename(fp), 'road_points': pts, 'test_outcome': out})
    return data

def load_travel(root):
    if not os.path.isdir(root): return []
    data = []
    for camp in sorted(os.listdir(root)):
        cp = os.path.join(root, camp)
        if not os.path.isdir(cp): continue
        for fp in glob.glob(os.path.join(cp, 'test.*.json')):
            try:
                with open(fp) as f: tc = json.load(f)
            except Exception: continue
            if not tc.get('is_valid', True): continue
            pts = tc.get('interpolated_points') or tc.get('road_points')
            out = tc.get('test_outcome')
            if not pts or out not in ('FAIL', 'PASS'): continue
            data.append({'_id': f'{camp}/{os.path.basename(fp)}', 'campaign': camp,
                         'road_points': pts, 'test_outcome': out})
    return data

def stratified_split(data, test_size=0.2, seed=SEED):
    y = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data])
    a, b = train_test_split(np.arange(len(data)), test_size=test_size, stratify=y, random_state=seed)
    return [data[i] for i in a], [data[i] for i in b]

# ====================================================================
# Per-benchmark drivers
# ====================================================================

def bench_geom(tag, loader, kfold=False):
    print(f"\n{'='*70}\n{tag} (Geom + TTA)\n{'='*70}")
    data = loader()
    if not data:
        print(f"  [SKIP] {tag}: no data"); return {'status': 'missing'}
    nf = sum(tc['test_outcome'] == 'FAIL' for tc in data)
    print(f"  [{tag}] N={len(data)} FAIL={nf} ({100*nf/max(1,len(data)):.2f}%)")
    if len(data) < 100 or nf < 20:
        return {'status': 'too_small', 'n': len(data), 'n_fail': nf}
    if kfold:
        y_all = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = []
        for fk, (ti, ei) in enumerate(skf.split(np.arange(len(data)), y_all)):
            tr = [data[i] for i in ti]; te = [data[i] for i in ei]
            folds.append(run_geom_split_tta(tr, te, name=f'{tag} f{fk+1}'))
        # aggregate
        agg = {'folds': folds}
        for k in ('no_tta', 'rot6', 'full24'):
            agg[k] = {
                'apfd_full_mean': float(np.mean([f[k]['apfd_full'] for f in folds])),
                'apfd_full_std':  float(np.std ([f[k]['apfd_full'] for f in folds])),
            }
        agg['delta_full24_vs_no_tta'] = float(np.mean([f['delta_full24_vs_no_tta'] for f in folds]))
        return agg
    tr, te = stratified_split(data)
    return run_geom_split_tta(tr, te, name=tag)

def bench_rp():
    """RP is tabular -- TTA does not apply. We re-run LightGBM baseline
    so the output JSON has all 5 datasets in one place. apfd_no_tta IS
    the LightGBM number; rot6/full24 set to None."""
    print(f"\n{'='*70}\nSDC-Pririotizer-RP (LightGBM 5-fold, no TTA -- tabular)\n{'='*70}")
    if not HAVE_PD: return {'status': 'no_pandas'}
    base = PATHS['rp_base']
    if not os.path.isdir(base): return {'status': 'missing'}
    sets = {
        'BeamNG_RF_1':   'datasets/fullroad/BeamNG_AI/BeamNG_RF_1/BeamNG_RF_1_Complete.csv',
        'BeamNG_RF_1_5': 'datasets/fullroad/BeamNG_AI/BeamNG_RF_1_5/BeamNG_RF_1_5_Complete.csv',
        'BeamNG_RF_2':   'datasets/fullroad/BeamNG_AI/BeamNG_RF_2/BeamNG_RF_2_Complete.csv',
        'DriverAI':      'datasets/fullroad/Driver_AI/DriverAI_Complete.csv',
    }
    out = {}
    LABEL_COL = 'safety'; DROP_COLS = {'start_time', 'end_time', LABEL_COL}
    for name, rel in sets.items():
        path = os.path.join(base, rel)
        if not os.path.isfile(path):
            out[name] = {'status': 'missing'}; continue
        df = pd.read_csv(path)
        y = (df[LABEL_COL].astype(str).str.lower() == 'unsafe').astype(int).values
        feat_cols = [c for c in df.columns if c not in DROP_COLS]
        X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
        n_pos = int(y.sum()); n = len(y)
        print(f"  {name}: N={n} FAIL={n_pos} ({100*n_pos/n:.1f}%)")
        if n_pos < 5 or (n - n_pos) < 5:
            out[name] = {'status': 'too_imbalanced'}; continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        apfds = []
        for fk, (tr, te) in enumerate(skf.split(X, y)):
            if HAVE_LGB:
                clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                         min_data_in_leaf=10, subsample=0.9, colsample_bytree=0.9,
                                         class_weight='balanced', random_state=SEED, verbosity=-1)
            else:
                clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                      max_leaf_nodes=63, random_state=SEED)
            clf.fit(X[tr], y[tr]); probs = clf.predict_proba(X[te])[:, 1]
            order = np.argsort(-probs); y_te = y[te]
            fp = [i + 1 for i, idx in enumerate(order) if y_te[idx] == 1]
            n_te = len(order); m_te = len(fp)
            apfds.append(1 - sum(fp) / (n_te * m_te) + 1 / (2 * n_te) if n_te and m_te else 1.0)
        out[name] = {'no_tta': {'apfd_full': float(np.mean(apfds)),
                                'apfd_full_std': float(np.std(apfds))},
                     'rot6': None, 'full24': None,
                     'n': n, 'n_fail': n_pos}
        print(f"    * {name}: APFD={out[name]['no_tta']['apfd_full']:.4f}")
    return out

# ====================================================================
# Main
# ====================================================================

def main():
    t0 = time.time()
    results = {
        'exp': '01_geom_tta',
        'tta_family': {'rot_deg': TTA_ROT_DEG, 'flips': TTA_FLIPS, 'reverse': TTA_REVERSE,
                       'n_views_full': len(TTA_ROT_DEG) * len(TTA_FLIPS) * len(TTA_REVERSE)},
        'recipe': 'Transformer + SWA + Focal(gamma=2.5), 75 ep, batch=256',
        'epochs': EPOCHS, 'gamma': GAMMA, 'seed': SEED,
    }
    benches = [
        ('scissor',  lambda: bench_geom('scissor',  lambda: load_flat_json_dir(PATHS['scissor'], '*-test.json'), kfold=True)),
        ('rp',       lambda: bench_rp()),
        ('its4sdc',  lambda: bench_geom('its4sdc',  lambda: load_flat_json_dir(PATHS['its4sdc'],  '*.json'))),
        ('sensodat', lambda: bench_geom('sensodat', lambda: load_sensodat(PATHS['sensodat']))),
        ('travel',   lambda: bench_geom('travel',   lambda: load_travel(PATHS['travel']))),
    ]
    for tag, fn in benches:
        try:
            results[tag] = fn()
        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] stopped after {tag}"); break
        except Exception as e:
            print(f"  [ERR] {tag}: {type(e).__name__}: {e}")
            results[tag] = {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
        op = os.path.join(OUTPUT_DIR, 'exp_01_geom_tta_results.json')
        with open(op, 'w') as f: json.dump(results, f, indent=2, default=str)
        print(f"  [save] {op}")

    print(f"\n{'='*70}\nEXP 01 -- TTA HEADLINE (Delta-APFD = full24 - no_tta)\n{'='*70}")
    for tag, _ in benches:
        b = results.get(tag, {})
        if not isinstance(b, dict): continue
        if 'delta_full24_vs_no_tta' in b:
            base = b['no_tta']['apfd_full']; full = b['full24']['apfd_full']
            d = b['delta_full24_vs_no_tta']
            print(f"  {tag:>10s}:  no_tta={base:.4f}  full24={full:.4f}  Delta={d:+.4f}")
        elif tag == 'rp':
            for sub, sb in b.items():
                if isinstance(sb, dict) and isinstance(sb.get('no_tta'), dict):
                    print(f"  {('rp/'+sub):>16s}:  no_tta={sb['no_tta']['apfd_full']:.4f}  (TTA n/a)")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == '__main__':
    main()
