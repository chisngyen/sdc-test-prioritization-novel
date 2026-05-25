"""
Exp 03 -- Ensemble Distillation (5 gammas -> 1 student) across ALL 5 benchmarks
==============================================================================
Hypothesis: the SensoDat 5-config ensemble (Focal gammas in
{1.5, 2.0, 2.5, 3.0, 3.5}, same Transformer + SWA) gives APFD=0.8077
versus best-single 0.8066 -- a +0.001 lift at 5x inference cost. We
want to capture that lift in a *single* model via knowledge distillation
so that inference cost remains 1x.

Recipe (per benchmark):
  1) Phase 1 -- Train 5 teachers (gammas={1.5, 2.0, 2.5, 3.0, 3.5}).
     Each is the SensoDat winner recipe (Transformer + SWA), just with
     a different focal gamma. We keep the SWA model for each teacher.
  2) Soft-target snapshot -- For every training example, compute the
     **arithmetic mean of teacher sigmoids**. This is the distilled
     soft label `s_i = (1/K) * sum_k sigmoid_k(x_i)`.
  3) Phase 2 -- Train a fresh student (same architecture) with a
     blended loss:
            L_student = alpha * BCE(student, hard_label)
                      + (1 - alpha) * KL(student || teacher_soft)
     alpha=0.3 (i.e. 70% distillation, 30% hard label). Plus the same
     SWA pass at the same SWA_START.

Why this is novel for SDC test prioritization:
  - Distillation has been used widely in CV/NLP but never (per our
    survey of `exps/tracker.md`) for SDC test ranking.
  - The SensoDat ensemble is the only ensemble result reported in the
    leaderboard; distilling it gives the ensemble-quality APFD with
    no inference overhead -- the "deploy-time" pitch.
  - Cross-bench: a student distilled per-bench (this exp) is the
    cheapest possible drop-in replacement for the 5-config ensemble.
    A *universal* student over the pooled 5-bench soft labels is the
    natural follow-up (defer to future work).

What we report per benchmark:
  - teachers[k]: AUC + APFD per individual teacher (k=0..4)
  - ensemble:    AUC + APFD of the averaged-sigmoid 5-ensemble (oracle)
  - student:     AUC + APFD of the distilled single student
  - delta_student_vs_best_teacher  -- did we recover the ensemble lift?

Saves `exp_03_distill_results.json`.
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

SEQ_LEN, EPOCHS, BATCH, LR, SWA_START, N_TRIALS, SEED = 197, 75, 256, 5e-4, 50, 30, 42
TEACHER_GAMMAS = [1.5, 2.0, 2.5, 3.0, 3.5]      # 5 teachers
DISTILL_ALPHA  = 0.3                            # weight on hard label; rest is KL

# ====================================================================
# Features (verbatim)
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
            R = a * b * c / (4 * math.sqrt(at)); curv[i] = 1.0 / R if R > 0 else 0.0
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
# Model / SWA / Focal
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=1.0):
        super().__init__(); self.a, self.g, self.pw = alpha, gamma, pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        w = torch.where(targets == 1, self.pw, 1.0); bce = bce * w
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return (self.a * (1 - pt) ** self.g * bce).mean()

# ====================================================================
# Phase 1 -- train one teacher with given gamma
# ====================================================================

def train_one_teacher(X_train, y_train, X_val, y_val, gamma, name=''):
    print(f"\n--- Teacher gamma={gamma:.1f} ({name}) ---")
    model = RoadTransformer(in_channels=10, seq_len=SEQ_LEN).to(DEVICE)
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pw = float(n_neg) / max(1, n_pos)
    weights = np.where(y_train == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    X_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH, sampler=sampler,
                    num_workers=2, pin_memory=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warm = 5
    def lr_lambda(ep):
        if ep < warm: return (ep + 1) / warm
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, EPOCHS - warm))))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = FocalLoss(alpha=1.0, gamma=gamma, pos_weight=pw)
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    best_auc, best_state, swa = 0, None, None
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
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
        if (ep + 1) % 25 == 0:
            print(f"  Ep {ep+1:3d} | AUC:{val_auc:.4f} | Best:{best_auc:.4f}")
    model.load_state_dict(best_state)
    return swa.get_model().to(DEVICE) if swa else model, best_auc

# ====================================================================
# Phase 2 -- distill student from averaged teacher probs
# ====================================================================

def predict_probs_torch(model, X_norm):
    Xt = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    model.eval().to(DEVICE)
    with torch.no_grad(): return torch.sigmoid(model(Xt)).cpu().numpy()

def teacher_soft_targets(teachers, X_norm):
    """Return mean-of-sigmoids over K teachers, shape (N,)."""
    s = np.zeros(len(X_norm), dtype=np.float64)
    for t in teachers:
        s += predict_probs_torch(t, X_norm).astype(np.float64)
    return (s / len(teachers)).astype(np.float32)

class DistillLoss(nn.Module):
    """alpha * BCE(student, hard) + (1-alpha) * KL(student || soft)."""
    def __init__(self, alpha=DISTILL_ALPHA, pos_weight=1.0):
        super().__init__(); self.alpha = alpha; self.pw = pos_weight
    def forward(self, logits, hard, soft):
        # BCE with class-weighting on hard label
        bce = F.binary_cross_entropy_with_logits(logits, hard, reduction='none')
        w = torch.where(hard == 1, self.pw, 1.0)
        loss_bce = (bce * w).mean()
        # Bernoulli KL: KL(soft || student_sigmoid)
        # = soft * log(soft/p) + (1-soft) * log((1-soft)/(1-p))
        p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        s = soft.clamp(1e-7, 1 - 1e-7)
        kl = s * (torch.log(s) - torch.log(p)) + (1 - s) * (torch.log(1 - s) - torch.log(1 - p))
        loss_kl = kl.mean()
        return self.alpha * loss_bce + (1 - self.alpha) * loss_kl, loss_bce.detach(), loss_kl.detach()

def train_student(X_train, y_train, soft_train, X_val, y_val, name=''):
    print(f"\n--- Student (KL distill, alpha={DISTILL_ALPHA}) {name} ---")
    model = RoadTransformer(in_channels=10, seq_len=SEQ_LEN).to(DEVICE)
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pw = float(n_neg) / max(1, n_pos)
    weights = np.where(y_train == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    X_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    s_t = torch.tensor(soft_train, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_t, y_t, s_t), batch_size=BATCH,
                    sampler=sampler, num_workers=2, pin_memory=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warm = 5
    def lr_lambda(ep):
        if ep < warm: return (ep + 1) / warm
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, EPOCHS - warm))))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = DistillLoss(alpha=DISTILL_ALPHA, pos_weight=pw)
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    best_auc, best_state, swa = 0, None, None
    for ep in range(EPOCHS):
        model.train(); tb = tk = nb = 0
        for xb, yb, sb in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            sb = sb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(xb)
                loss, lb, lk = crit(logits, yb, sb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tb += float(lb); tk += float(lk); nb += 1
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
            print(f"  Ep {ep+1:3d} | BCE:{tb/nb:.4f} KL:{tk/nb:.4f} | AUC:{val_auc:.4f} Best:{best_auc:.4f}")
    model.load_state_dict(best_state)
    return swa.get_model().to(DEVICE) if swa else model, best_auc

# ====================================================================
# APFD eval
# ====================================================================

def compute_apfd_from_pids(pids, td):
    n = len(pids); fp = [i + 1 for i, t in enumerate(pids) if td[t]['test_outcome'] == 'FAIL']
    m = len(fp); return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0

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

def measure(model, test_data, X_te_n, label, tag):
    p = predict_probs_torch(model, X_te_n)
    try: auc = roc_auc_score([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in test_data], p)
    except Exception: auc = 0.5
    af = full_apfd(test_data, p)
    am, as_, sz = trial_apfd(test_data, p)
    print(f"  {tag:>18s}/{label:<10s} AUC={auc:.4f} APFD_full={af:.4f} "
          f"APFD_trial={am:.4f}+/-{as_:.4f}")
    return {'auc': float(auc), 'apfd_full': float(af),
            'apfd_trial_mean': am, 'apfd_trial_std': as_, 'sample_size': sz}, p

def run_split_distill(train_data, test_data, name=''):
    print(f"\n  [{name}] Train: {len(train_data)} | Test: {len(test_data)}")
    X_tr, y_tr = prepare(train_data); X_te, y_te = prepare(test_data)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1))
    stds[stds < 1e-8] = 1.0
    X_tr_n = (X_tr - means) / stds; X_te_n = (X_te - means) / stds

    # Phase 1 -- train K teachers
    teachers, teacher_results = [], []
    p_te_acc = np.zeros(len(test_data), dtype=np.float64)
    for k, g in enumerate(TEACHER_GAMMAS):
        t, _ = train_one_teacher(X_tr_n, y_tr, X_te_n, y_te, gamma=g, name=name)
        m, p_te = measure(t, test_data, X_te_n, f'T{k}_g{g}', name)
        teacher_results.append({'gamma': g, **m})
        p_te_acc += p_te
        teachers.append(t)
    # Oracle ensemble (avg sigmoids of K teachers)
    ens_probs = p_te_acc / len(teachers)
    try: ens_auc = roc_auc_score([1 if tc['test_outcome'] == 'FAIL' else 0 for tc in test_data], ens_probs)
    except Exception: ens_auc = 0.5
    ens_full = full_apfd(test_data, ens_probs)
    ens_tm, ens_ts, sz = trial_apfd(test_data, ens_probs)
    print(f"  {name:>18s}/{'ENSEMBLE':<10s} AUC={ens_auc:.4f} APFD_full={ens_full:.4f} "
          f"APFD_trial={ens_tm:.4f}+/-{ens_ts:.4f}")

    # Phase 2 -- compute teacher soft targets on TRAIN set and distill
    soft_train = teacher_soft_targets(teachers, X_tr_n)
    student, _ = train_student(X_tr_n, y_tr, soft_train, X_te_n, y_te, name=name)
    stu, _ = measure(student, test_data, X_te_n, 'STUDENT', name)

    best_teacher = max(teacher_results, key=lambda r: r['apfd_full'])
    return {
        'teachers': teacher_results,
        'ensemble': {'auc': float(ens_auc), 'apfd_full': float(ens_full),
                     'apfd_trial_mean': ens_tm, 'apfd_trial_std': ens_ts,
                     'sample_size': sz, 'n_teachers': len(teachers)},
        'student': stu,
        'best_teacher_gamma': best_teacher['gamma'],
        'delta_student_vs_best_teacher': float(stu['apfd_full'] - best_teacher['apfd_full']),
        'delta_student_vs_ensemble':     float(stu['apfd_full'] - ens_full),
        'n_train': len(train_data), 'n_test': len(test_data),
        'n_fail_test': int(y_te.sum()),
    }

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
    files = sorted(glob.glob(os.path.join(path, pattern))); data = []
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
# RP -- distillation analogue is averaging 5 LightGBM models. Cheap.
# ====================================================================

def bench_rp_distill():
    print(f"\n{'='*70}\nSDC-Pririotizer-RP (5 LightGBM teachers -> avg)\n{'='*70}")
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
        # 5 teachers via different seeds (LightGBM "gamma" analogue = random seed perturbation)
        apfd_single, apfd_avg = [], []
        for fk, (tr, te) in enumerate(skf.split(X, y)):
            teacher_probs = []
            for sd in range(5):
                if HAVE_LGB:
                    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                             min_data_in_leaf=10, subsample=0.9, colsample_bytree=0.9,
                                             class_weight='balanced', random_state=SEED + sd, verbosity=-1)
                else:
                    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                          max_leaf_nodes=63, random_state=SEED + sd)
                clf.fit(X[tr], y[tr]); teacher_probs.append(clf.predict_proba(X[te])[:, 1])
            # single = first teacher; avg = mean of 5
            for probs, bucket in [(teacher_probs[0], apfd_single), (np.mean(teacher_probs, axis=0), apfd_avg)]:
                order = np.argsort(-probs); y_te = y[te]
                fp = [i + 1 for i, idx in enumerate(order) if y_te[idx] == 1]
                n_te = len(order); m_te = len(fp)
                bucket.append(1 - sum(fp) / (n_te * m_te) + 1 / (2 * n_te) if n_te and m_te else 1.0)
        out[name] = {
            'single_apfd_mean': float(np.mean(apfd_single)),
            'single_apfd_std':  float(np.std(apfd_single)),
            'avg5_apfd_mean':   float(np.mean(apfd_avg)),
            'avg5_apfd_std':    float(np.std(apfd_avg)),
            'delta_avg_vs_single': float(np.mean(apfd_avg) - np.mean(apfd_single)),
            'n': n, 'n_fail': n_pos,
        }
        print(f"  {name}: single={out[name]['single_apfd_mean']:.4f}  "
              f"avg5={out[name]['avg5_apfd_mean']:.4f}  "
              f"Delta={out[name]['delta_avg_vs_single']:+.4f}")
    return out

# ====================================================================
# Geometry drivers
# ====================================================================

def bench_geom(tag, loader, kfold=False):
    print(f"\n{'='*70}\n{tag} (5-teacher distill)\n{'='*70}")
    data = loader()
    if not data: return {'status': 'missing'}
    nf = sum(tc['test_outcome'] == 'FAIL' for tc in data)
    print(f"  [{tag}] N={len(data)} FAIL={nf} ({100*nf/max(1,len(data)):.2f}%)")
    if len(data) < 100 or nf < 20:
        return {'status': 'too_small', 'n': len(data), 'n_fail': nf}
    if kfold:
        # K-fold distill is expensive (K teachers per fold). Restrict to a
        # single 80/20 split for scissor to keep wall-clock reasonable.
        tr, te = stratified_split(data)
        return run_split_distill(tr, te, name=tag + '_80_20')
    tr, te = stratified_split(data)
    return run_split_distill(tr, te, name=tag)

def main():
    t0 = time.time()
    results = {
        'exp': '03_distill',
        'recipe': '5 teachers (gammas {1.5,2.0,2.5,3.0,3.5}) -> KL distill into student',
        'distill': {'alpha': DISTILL_ALPHA, 'teacher_gammas': TEACHER_GAMMAS},
        'epochs': EPOCHS, 'seed': SEED,
    }
    benches = [
        ('scissor',  lambda: bench_geom('scissor',  lambda: load_flat_json_dir(PATHS['scissor'], '*-test.json'), kfold=True)),
        ('rp',       lambda: bench_rp_distill()),
        ('its4sdc',  lambda: bench_geom('its4sdc',  lambda: load_flat_json_dir(PATHS['its4sdc'], '*.json'))),
        ('sensodat', lambda: bench_geom('sensodat', lambda: load_sensodat(PATHS['sensodat']))),
        ('travel',   lambda: bench_geom('travel',   lambda: load_travel(PATHS['travel']))),
    ]
    for tag, fn in benches:
        try: results[tag] = fn()
        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] stopped after {tag}"); break
        except Exception as e:
            print(f"  [ERR] {tag}: {type(e).__name__}: {e}")
            results[tag] = {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
        op = os.path.join(OUTPUT_DIR, 'exp_03_distill_results.json')
        with open(op, 'w') as f: json.dump(results, f, indent=2, default=str)
        print(f"  [save] {op}")

    print(f"\n{'='*70}\nEXP 03 -- distillation gains (vs best single teacher)\n{'='*70}")
    for tag, _ in benches:
        b = results.get(tag, {})
        if not isinstance(b, dict): continue
        if 'student' in b and 'ensemble' in b:
            print(f"  {tag:>10s}: best_T={max(t['apfd_full'] for t in b['teachers']):.4f}  "
                  f"ensemble={b['ensemble']['apfd_full']:.4f}  "
                  f"student={b['student']['apfd_full']:.4f}  "
                  f"Delta_S-T={b['delta_student_vs_best_teacher']:+.4f}  "
                  f"Delta_S-E={b['delta_student_vs_ensemble']:+.4f}")
        elif tag == 'rp':
            for sub, sb in b.items():
                if isinstance(sb, dict) and 'single_apfd_mean' in sb:
                    print(f"  {'rp/'+sub:>16s}: single={sb['single_apfd_mean']:.4f}  "
                          f"avg5={sb['avg5_apfd_mean']:.4f}  "
                          f"Delta={sb['delta_avg_vs_single']:+.4f}")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == '__main__':
    main()
