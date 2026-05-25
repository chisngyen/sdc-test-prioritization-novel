"""
Exp L -- APFD-Direct Loss via Differentiable Soft-Rank
=======================================================
Hypothesis: Exp B in `best_all/` proved the identity
    APFD = (1 - p) * AUC + p / 2     (no ties, same split)
which means that for a fixed FAIL rate p, maximizing AUC is *equivalent*
to maximizing APFD. But Focal/BCE optimizes a per-instance proxy, not
AUC directly; and pairwise listwise losses (Exp 03 PL, Exp G prefix-PL)
optimize ranking quality, but not the APFD prefix structure.

This experiment optimizes APFD **literally** via a differentiable
surrogate built from soft-rank:

   rank_soft(s_i) = sum_j sigmoid((s_j - s_i) / tau)         (~rank, in [0,n-1])

   APFD_soft = 1 - (1 / (n * m)) * sum_{i: y_i=1} (rank_soft(s_i) + 1)
             + 1 / (2 n)

   loss = 1 - APFD_soft         (so grad steps maximize APFD)

We anneal temperature `tau` (5.0 -> 0.5) and start from a Focal warmup
(15 epochs), then transition to a 70/30 blend that gradually shifts to
pure APFD-direct.

Why this is novel for SDC:
  - PL/listwise losses optimize permutation likelihood, not APFD.
  - SoftRank is well-known in information retrieval (Taylor 2008,
    Cuturi 2019 SinkhornSort, Grover 2019 NeuralSort) but has NOT been
    applied to test prioritization metrics.
  - Plausibly closes the AUC<->APFD gap from the inside (see Exp B):
    raises APFD without trading off AUC.

Eval reports the SAME APFD-on-full-test plus 30-trial sigma protocol
used by `exp_full_all.py`. Distinct from Exp 03 (PL) and Exp G (prefix-
weighted PL) -- both kept BCE/Focal and added a ranking term; we
*replace* the loss with the APFD surrogate itself.

Saves `exp_L_apfd_direct_results.json`.
"""
import os, sys, json, time, math, copy, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# ---------- Paths / config ----------
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

# APFD-direct schedule
APFD_WARMUP_EPOCHS = 15        # pure Focal for the first 15 ep
APFD_BLEND_START   = 15        # after warmup, mix Focal + APFD
APFD_PURE_FROM     = 45        # from ep 45 onwards, pure APFD-direct
TAU_INIT           = 5.0       # soft-rank temperature (anneals -> TAU_FINAL)
TAU_FINAL          = 0.5

# ====================================================================
# 10-channel features (verbatim)
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
    cum_dist = np.cumsum(seg_full); cum_dist_norm = cum_dist / (cum_dist[-1] + 1e-8)
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
# Model
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

class SWAModel:
    def __init__(self, m): self.model = copy.deepcopy(m); self.n = 0
    def update(self, m):
        self.n += 1; a = 1.0 / self.n
        for p, q in zip(self.model.parameters(), m.parameters()):
            p.data.mul_(1 - a).add_(q.data, alpha=a)
    def get_model(self): return self.model

# ====================================================================
# Losses: Focal + APFD-direct via SoftRank
# ====================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=1.0):
        super().__init__(); self.a, self.g, self.pw = alpha, gamma, pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        w = torch.where(targets == 1, self.pw, 1.0); bce = bce * w
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return (self.a * (1 - pt) ** self.g * bce).mean()

def soft_rank(scores, tau):
    """SoftRank via sigmoid-pairwise. scores: (B,) -> rank in [0, B-1]
    rank_i = sum_{j != i} sigmoid((s_j - s_i) / tau)
    Lower temperature -> closer to true rank but harsher gradient."""
    s = scores.unsqueeze(0) - scores.unsqueeze(1)            # (B, B): s_j - s_i at [i,j]
    P = torch.sigmoid(s / tau)
    P = P - torch.diag_embed(P.diagonal(dim1=-2, dim2=-1))   # zero out i==j
    return P.sum(dim=1)                                      # rank_i

def soft_apfd_loss(logits, targets, tau):
    """1 - APFD_soft on the current batch. Returns NaN-safe scalar.
    Assumes targets in {0, 1} float. Sigmoid on logits = score (higher
    = more failure-prone -> ranked first)."""
    s = logits.float()
    y = targets.float()
    n = s.shape[0]
    m = y.sum()
    if m.item() == 0:
        return torch.zeros((), device=s.device, requires_grad=False)
    # rank: 0 = top-most (highest score). soft_rank gives 0 for max.
    # APFD wants position_i (1-indexed) of each FAIL after sorting by
    # descending score. position_i = rank_i + 1.
    r = soft_rank(s, tau)
    pos = r + 1.0
    apfd_soft = 1.0 - (pos * y).sum() / (n * m) + 1.0 / (2.0 * n)
    return 1.0 - apfd_soft

def schedule_blend(epoch):
    """Returns (w_focal, w_apfd, tau) for the given epoch."""
    if epoch < APFD_WARMUP_EPOCHS:
        return 1.0, 0.0, TAU_INIT
    if epoch >= APFD_PURE_FROM:
        # cosine anneal tau to final value
        frac = min(1.0, (epoch - APFD_PURE_FROM) / max(1, EPOCHS - APFD_PURE_FROM))
        tau = TAU_INIT * (1 - frac) + TAU_FINAL * frac
        return 0.0, 1.0, tau
    # blend region: linearly shift focal -> apfd
    frac = (epoch - APFD_BLEND_START) / max(1, APFD_PURE_FROM - APFD_BLEND_START)
    w_apfd = frac; w_focal = 1.0 - frac
    tau = TAU_INIT * (1 - 0.5 * frac)
    return w_focal, w_apfd, tau

# ====================================================================
# Training
# ====================================================================

def train_apfd_direct(X_train, y_train, X_val, y_val, name=''):
    print(f"\n--- Train {name} | APFD-Direct (Focal warmup -> SoftRank APFD) ---")
    model = RoadTransformer(in_channels=10, seq_len=SEQ_LEN).to(DEVICE)
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pw = float(n_neg) / max(1, n_pos)

    # IMPORTANT: APFD loss is *batchwise*. We use a STANDARD (non-
    # weighted) shuffled loader so each batch contains its natural mix
    # of FAIL/PASS; weighted sampling would distort the batch ranking
    # statistics. The class imbalance is instead carried by the focal
    # warmup with pos_weight.
    X_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH,
                    shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warm = 5
    def lr_lambda(ep):
        if ep < warm: return (ep + 1) / warm
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, EPOCHS - warm))))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    focal = FocalLoss(alpha=1.0, gamma=GAMMA, pos_weight=pw)
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    best_auc, best_state, swa = 0.0, None, None

    for ep in range(EPOCHS):
        w_f, w_a, tau = schedule_blend(ep)
        model.train(); tl_f, tl_a, nb = 0.0, 0.0, 0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            # APFD-direct must be done in fp32 because tiny diffs in
            # the pairwise matrix matter. Focal stays in autocast.
            if w_f > 0:
                with autocast(enabled=use_amp):
                    logits = model(xb)
                    f_loss = focal(logits, yb)
            else:
                logits = model(xb).float()
                f_loss = torch.zeros((), device=DEVICE)
            if w_a > 0:
                # recompute logits in fp32 for the APFD pathway
                logits_fp32 = model(xb).float() if w_f > 0 else logits
                a_loss = soft_apfd_loss(logits_fp32, yb.float(), tau)
            else:
                a_loss = torch.zeros((), device=DEVICE)
            loss = w_f * f_loss + w_a * a_loss
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl_f += float(f_loss); tl_a += float(a_loss); nb += 1
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
            print(f"  Ep {ep+1:3d} | w_f={w_f:.2f} w_a={w_a:.2f} tau={tau:.2f} "
                  f"| L_f:{tl_f/nb:.4f} L_a:{tl_a/nb:.4f} | AUC:{val_auc:.4f} Best:{best_auc:.4f}")

    model.load_state_dict(best_state)
    swa_m = swa.get_model().to(DEVICE) if swa else None
    return model, swa_m, best_auc

# ====================================================================
# Eval helpers
# ====================================================================

def compute_apfd_from_pids(pids, td):
    n = len(pids); fp = [i + 1 for i, t in enumerate(pids) if td[t]['test_outcome'] == 'FAIL']
    m = len(fp); return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0

def predict_probs(model, X_norm):
    Xt = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    model.eval().to(DEVICE)
    with torch.no_grad(): return torch.sigmoid(model(Xt)).cpu().numpy()

def full_apfd(test_data, probs):
    td = {tc['_id']: tc for tc in test_data}; ids = [tc['_id'] for tc in test_data]
    pids = [t for _, t in sorted(zip(probs, ids), key=lambda x: -x[0])]
    return compute_apfd_from_pids(pids, td)

def trial_apfd(test_data, probs, n_trials=N_TRIALS, frac=0.3, min_size=50):
    sample_size = min(len(test_data), max(min_size, int(frac * len(test_data))))
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

def prepare(data):
    X = np.array([resample(extract_seq_10ch(tc['road_points'])) for tc in data], dtype=np.float32)
    y = np.array([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in data], dtype=np.int64)
    return X, y

def run_split(train_data, test_data, name=''):
    print(f"  [{name}] Train: {len(train_data)} | Test: {len(test_data)}")
    X_tr, y_tr = prepare(train_data); X_te, y_te = prepare(test_data)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1))
    stds[stds < 1e-8] = 1.0
    X_tr_n = (X_tr - means) / stds; X_te_n = (X_te - means) / stds
    model, swa, auc = train_apfd_direct(X_tr_n, y_tr, X_te_n, y_te, name=name)
    eval_model = swa if swa is not None else model
    probs = predict_probs(eval_model, X_te_n)
    apfd_full = full_apfd(test_data, probs)
    apfd_tm, apfd_ts, sz = trial_apfd(test_data, probs)
    print(f"  {name:30s} AUC={auc:.4f} | APFD_full={apfd_full:.4f} "
          f"| APFD_trial={apfd_tm:.4f}+/-{apfd_ts:.4f}")
    return {'auc': float(auc), 'apfd_full': float(apfd_full),
            'apfd_trial_mean': apfd_tm, 'apfd_trial_std': apfd_ts,
            'sample_size': sz, 'n_train': len(train_data), 'n_test': len(test_data),
            'n_fail_test': int(y_te.sum())}

# ====================================================================
# Loaders (same as exp_full_all.py)
# ====================================================================

def load_sensodat(root):
    if not os.path.isdir(root): return []
    full = os.path.join(root, 'sensodat_full.json')
    candidates = [full] if os.path.isfile(full) else [
        p for p in (os.path.join(root, n) for n in ('sensodat_train.json', 'sensodat_test.json'))
        if os.path.isfile(p)]
    data = []
    for fp in candidates:
        try:
            with open(fp) as f: items = json.load(f)
        except Exception: continue
        if not isinstance(items, list): continue
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
# RP -- APFD-direct doesn't apply to LightGBM, but we re-fit LightGBM
# AND also run a per-row LambdaRank objective (the closest tabular
# analogue to APFD-direct). Reported side-by-side.
# ====================================================================

def bench_rp():
    print(f"\n{'='*70}\nSDC-Pririotizer-RP (LightGBM binary vs LambdaRank)\n{'='*70}")
    if not HAVE_PD: return {'status': 'no_pandas'}
    base = PATHS['rp_base']
    if not os.path.isdir(base): return {'status': 'missing'}
    sets = {
        'BeamNG_RF_1':   'datasets/fullroad/BeamNG_AI/BeamNG_RF_1/BeamNG_RF_1_Complete.csv',
        'BeamNG_RF_1_5': 'datasets/fullroad/BeamNG_AI/BeamNG_RF_1_5/BeamNG_RF_1_5_Complete.csv',
        'BeamNG_RF_2':   'datasets/fullroad/BeamNG_AI/BeamNG_RF_2/BeamNG_RF_2_Complete.csv',
        'DriverAI':      'datasets/fullroad/Driver_AI/DriverAI_Complete.csv',
    }
    LABEL_COL = 'safety'; DROP_COLS = {'start_time', 'end_time', LABEL_COL}
    out = {}
    for name, rel in sets.items():
        path = os.path.join(base, rel)
        if not os.path.isfile(path):
            out[name] = {'status': 'missing'}; continue
        df = pd.read_csv(path)
        y = (df[LABEL_COL].astype(str).str.lower() == 'unsafe').astype(int).values
        feat_cols = [c for c in df.columns if c not in DROP_COLS]
        X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
        n_pos = int(y.sum()); n = len(y)
        if n_pos < 5 or (n - n_pos) < 5:
            out[name] = {'status': 'too_imbalanced'}; continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        apfd_bin, apfd_rank = [], []
        for fk, (tr, te) in enumerate(skf.split(X, y)):
            # binary baseline
            if HAVE_LGB:
                clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                         min_data_in_leaf=10, subsample=0.9, colsample_bytree=0.9,
                                         class_weight='balanced', random_state=SEED, verbosity=-1)
            else:
                clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                      max_leaf_nodes=63, random_state=SEED)
            clf.fit(X[tr], y[tr]); pb = clf.predict_proba(X[te])[:, 1]
            order = np.argsort(-pb); y_te = y[te]
            fp = [i + 1 for i, idx in enumerate(order) if y_te[idx] == 1]
            n_te = len(order); m_te = len(fp)
            apfd_bin.append(1 - sum(fp) / (n_te * m_te) + 1 / (2 * n_te) if n_te and m_te else 1.0)
            # LambdaRank objective (rank-aware LightGBM proxy for APFD)
            if HAVE_LGB:
                rk = lgb.LGBMRanker(objective='lambdarank', n_estimators=400,
                                    learning_rate=0.05, num_leaves=63,
                                    min_data_in_leaf=10, label_gain=[0, 1],
                                    random_state=SEED, verbosity=-1)
                rk.fit(X[tr], y[tr], group=[len(tr)])
                pr = rk.predict(X[te])
                order = np.argsort(-pr)
                fp = [i + 1 for i, idx in enumerate(order) if y_te[idx] == 1]
                apfd_rank.append(1 - sum(fp) / (n_te * m_te) + 1 / (2 * n_te) if n_te and m_te else 1.0)
            else:
                apfd_rank.append(float('nan'))
        out[name] = {
            'apfd_binary_mean': float(np.mean(apfd_bin)),
            'apfd_binary_std':  float(np.std(apfd_bin)),
            'apfd_lambdarank_mean': float(np.nanmean(apfd_rank)),
            'apfd_lambdarank_std':  float(np.nanstd(apfd_rank)),
            'delta_rank_vs_binary': float(np.nanmean(apfd_rank) - np.mean(apfd_bin)),
            'n': n, 'n_fail': n_pos,
        }
        print(f"  {name}: binary={out[name]['apfd_binary_mean']:.4f} "
              f"| lambdarank={out[name]['apfd_lambdarank_mean']:.4f} "
              f"| Delta={out[name]['delta_rank_vs_binary']:+.4f}")
    return out

# ====================================================================
# Geometry bench drivers
# ====================================================================

def bench_geom(tag, loader, kfold=False):
    print(f"\n{'='*70}\n{tag} (APFD-Direct)\n{'='*70}")
    data = loader()
    if not data: return {'status': 'missing'}
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
            folds.append(run_split(tr, te, name=f'{tag} f{fk+1}'))
        return {'folds': folds,
                'auc_mean':            float(np.mean([f['auc']              for f in folds])),
                'apfd_full_mean':      float(np.mean([f['apfd_full']        for f in folds])),
                'apfd_full_std':       float(np.std ([f['apfd_full']        for f in folds])),
                'apfd_trial_mean_avg': float(np.mean([f['apfd_trial_mean']  for f in folds])),
                'apfd_trial_std_avg':  float(np.mean([f['apfd_trial_std']   for f in folds]))}
    tr, te = stratified_split(data)
    return run_split(tr, te, name=tag)

def main():
    t0 = time.time()
    results = {
        'exp': 'L_apfd_direct',
        'recipe': 'Transformer + SWA + Focal warmup -> APFD-Direct (SoftRank, tau 5->0.5)',
        'schedule': {'warmup_focal_epochs': APFD_WARMUP_EPOCHS,
                     'blend_start': APFD_BLEND_START, 'pure_from': APFD_PURE_FROM,
                     'tau_init': TAU_INIT, 'tau_final': TAU_FINAL},
        'epochs': EPOCHS, 'gamma': GAMMA, 'seed': SEED,
    }
    benches = [
        ('scissor',  lambda: bench_geom('scissor', lambda: load_flat_json_dir(PATHS['scissor'], '*-test.json'), kfold=True)),
        ('rp',       lambda: bench_rp()),
        ('its4sdc',  lambda: bench_geom('its4sdc', lambda: load_flat_json_dir(PATHS['its4sdc'], '*.json'))),
        ('sensodat', lambda: bench_geom('sensodat', lambda: load_sensodat(PATHS['sensodat']))),
        ('travel',   lambda: bench_geom('travel',  lambda: load_travel(PATHS['travel']))),
    ]
    for tag, fn in benches:
        try: results[tag] = fn()
        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] stopped after {tag}"); break
        except Exception as e:
            print(f"  [ERR] {tag}: {type(e).__name__}: {e}")
            results[tag] = {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
        op = os.path.join(OUTPUT_DIR, 'exp_L_apfd_direct_results.json')
        with open(op, 'w') as f: json.dump(results, f, indent=2, default=str)
        print(f"  [save] {op}")

    print(f"\n{'='*70}\nEXP L SUMMARY\n{'='*70}")
    for tag, _ in benches:
        b = results.get(tag, {})
        if not isinstance(b, dict): continue
        if 'apfd_full' in b:
            print(f"  {tag:>10s}: AUC={b['auc']:.4f}  APFD_full={b['apfd_full']:.4f}  "
                  f"APFD_trial={b['apfd_trial_mean']:.4f}+/-{b['apfd_trial_std']:.4f}")
        elif 'apfd_full_mean' in b:
            print(f"  {tag:>10s}: APFD_full={b['apfd_full_mean']:.4f}+/-{b['apfd_full_std']:.4f} (5-fold)")
        elif tag == 'rp':
            for sub, sb in b.items():
                if isinstance(sb, dict) and 'apfd_binary_mean' in sb:
                    print(f"  {'rp/'+sub:>16s}: binary={sb['apfd_binary_mean']:.4f}  "
                          f"lambdarank={sb['apfd_lambdarank_mean']:.4f}  "
                          f"Delta={sb['delta_rank_vs_binary']:+.4f}")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == '__main__':
    main()
