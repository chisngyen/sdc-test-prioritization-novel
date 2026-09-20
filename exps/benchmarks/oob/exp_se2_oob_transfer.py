"""
SE(2)-Invariant Cross-threshold Transfer on OOB
=========================================================================
Story: distribution-shift robustness UNDER an exact-invariance constraint.
We train one SE(2)-invariant Transformer per OOB threshold (0-1, 0-3, 0-5)
and evaluate zero-shot on every other threshold.

Architecture (from exps/exp02_SE2Equivariant.py):
  * 7-channel intrinsic features (seg len, |dheading|, signed curvature,
    dk/ds, d^2k/ds^2, normalized arclength, local-std of curvature).
    No absolute heading sin/cos, no absolute position -> by construction
    f(R r + t) = f(r), exactly.
  * Equivariant attention: relative-arclength bias only.

Outputs:
  * 3x3 APFD transfer matrix (rows = train src, cols = eval tgt)
  * Per-source SE(2) rotation probe (|Delta APFD| should be ~0 in float)
  * Saves: oob_se2_transfer_matrix.json, roadse2_oob_transfer_models.pt

Self-contained: paste this file as one Kaggle cell.
"""
import os, sys, json, time, math, copy, glob, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------- Path resolution (Kaggle + local) ----------
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

SEARCH_ROOTS = [
    '/kaggle/input',
    os.path.normpath(os.path.join(HERE, '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', 'data')),
    os.path.normpath(os.path.join(HERE, 'data')),
    os.getcwd(),
]

def _find_oob_folder(tag):
    target = f'Dataset-OOB-{tag}'
    seen = set()
    for root in SEARCH_ROOTS:
        if not root or not os.path.isdir(root) or root in seen:
            continue
        seen.add(root)
        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath)
            if base == target and any(fn.endswith('.json') for fn in filenames):
                return dirpath
            if base == target:
                for d in dirnames:
                    inner = os.path.join(dirpath, d)
                    try:
                        if any(fn.endswith('.json') for fn in os.listdir(inner)):
                            return inner
                    except OSError:
                        continue
    return None

def resolve_dir(tag):
    p = _find_oob_folder(tag)
    if p: return p
    raise FileNotFoundError(
        f"OOB-{tag} not found. Roots tried: {SEARCH_ROOTS}. "
        f"Need a 'Dataset-OOB-{tag}' folder containing *_test.json.")

OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(HERE, '..', '..', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
torch.set_float32_matmul_precision('high')
print(f"Device: {DEVICE} | bf16: {USE_BF16}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")

SEQ_LEN = 197

# ---------- Frame-INVARIANT 7-ch features ----------
# Drops absolute heading (sin/cos) and absolute position.
#   1. segment length              (intrinsic)
#   2. |delta heading|             (intrinsic, magnitude only)
#   3. signed curvature            (intrinsic)
#   4. dk/ds                       (intrinsic)
#   5. d^2k/ds^2                   (intrinsic)
#   6. cumulative arclength s/L    (parameterization invariant)
#   7. local std of curvature      (intrinsic)
def signed_curvature(pts):
    d = np.diff(pts, axis=0); ang = np.arctan2(d[:,1], d[:,0])
    dang = (np.diff(ang) + np.pi) % (2*np.pi) - np.pi
    seg = np.linalg.norm(d, axis=1)
    denom = 0.5*(seg[:-1] + seg[1:]) + 1e-8
    k = dang / denom
    return np.pad(k, (1,1), mode='constant')

def extract_invariant_7ch(pts_raw):
    pts = np.array(pts_raw, dtype=np.float64).reshape(-1,2); n = len(pts)
    d = np.diff(pts, axis=0); seg = np.linalg.norm(d, axis=1)
    seg_full = np.pad(seg, (0,1), mode='edge')
    ang = np.arctan2(d[:,1], d[:,0])
    dang = (np.diff(ang) + np.pi) % (2*np.pi) - np.pi
    abs_dang_full = np.pad(np.abs(dang), (1,1), mode='constant')
    k = signed_curvature(pts)
    dk = np.pad(np.diff(k), (0,1), mode='constant')
    ddk = np.pad(np.diff(dk), (0,1), mode='constant')
    s_cum = np.cumsum(seg_full); s_norm = s_cum / (s_cum[-1] + 1e-8)
    w = 11; lstd = np.zeros(n); hw = w//2
    for i in range(n):
        a, b = max(0, i-hw), min(n, i+hw+1); lstd[i] = np.std(k[a:b])
    return np.column_stack([seg_full, abs_dang_full, k, dk, ddk, s_norm, lstd]).astype(np.float32)

def resample_to_len(seq, target_len=SEQ_LEN):
    n, c = seq.shape
    if n == target_len: return seq
    x_old = np.linspace(0, 1, n); x_new = np.linspace(0, 1, target_len)
    out = np.empty((target_len, c), dtype=np.float32)
    for ch in range(c):
        out[:, ch] = np.interp(x_new, x_old, seq[:, ch])
    return out

# ---------- OOB loader ----------
def load_oob_dir(path, log_every=500):
    files = sorted(glob.glob(os.path.join(path, '*.json')))
    print(f"  {path}: {len(files)} files")
    data = []
    for i, fp in enumerate(files):
        try:
            with open(fp) as f: tc = json.load(f)
        except Exception:
            continue
        if not tc.get('is_valid', True): continue
        rp = tc.get('road_points'); out = tc.get('test_outcome')
        if not rp or out not in ('FAIL', 'PASS'): continue
        data.append({'_id': os.path.basename(fp), 'road_points': rp, 'test_outcome': out})
        if (i + 1) % log_every == 0:
            print(f"    parsed {i+1}/{len(files)}")
    return data

def get_pts(tc): return tc['road_points']
def is_fail(tc): return tc['test_outcome'] == 'FAIL'
def get_id(tc): return tc['_id']

def prepare_data(data, batch_print=2000):
    X, y = [], []
    for i, tc in enumerate(data):
        seq = extract_invariant_7ch(get_pts(tc))
        X.append(resample_to_len(seq, SEQ_LEN))
        y.append(1 if is_fail(tc) else 0)
        if (i + 1) % batch_print == 0:
            print(f"    feat {i+1}/{len(data)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# ---------- Equivariant attention block ----------
# Bias depends ONLY on relative arclength s_i - s_j (translation-invariant
# in s, and s itself is rotation-invariant) -> the whole tower is exactly
# SE(2)-invariant.
class InvariantBlock(nn.Module):
    def __init__(self, d_model=192, nhead=8, ff=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.rff = nn.Parameter(torch.randn(1, 32) * 2.0, requires_grad=False)
        self.rel_bias = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, nhead))

    def _rel_bias(self, s_norm):
        B, L = s_norm.shape
        ds = (s_norm.unsqueeze(2) - s_norm.unsqueeze(1)).unsqueeze(-1)
        feat = torch.sin(ds * self.rff)
        bias = self.rel_bias(feat)
        return bias.permute(0, 3, 1, 2)

    def forward(self, x, s_norm):
        B, Lp1, D = x.shape
        s_full = torch.cat([torch.zeros(B, 1, device=x.device), s_norm], dim=1)
        bias = self._rel_bias(s_full)
        nhead = bias.size(1); h = x.size(1)
        attn_mask = bias.reshape(B * nhead, h, h)
        z = self.n1(x)
        a, _ = self.attn(z, z, z, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x

class SE2RoadNet(nn.Module):
    def __init__(self, in_ch=7, d_model=192, depth=6, nhead=8, ff=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_ch, d_model),
                                  nn.LayerNorm(d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([InvariantBlock(d_model, nhead, ff, dropout)
                                     for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, 64), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):                                # x: (B, C, L)
        x = x.permute(0, 2, 1)                           # (B, L, C)
        s_norm = x[..., 5]                               # 6th ch = s/L
        h = self.proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        for b in self.blocks: h = b(h, s_norm)
        return self.head(h[:, 0]).squeeze(-1)

# ---------- Loss / SWA ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=1.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weight = torch.where(targets == 1, self.pos_weight, 1.0)
        bce = bce * weight
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return (self.alpha * (1 - pt).pow(self.gamma) * bce).mean()

class SWAModel:
    def __init__(self, model): self.model = copy.deepcopy(model); self.n = 0
    def update(self, new_model):
        self.n += 1; alpha = 1.0 / self.n
        for p_swa, p_new in zip(self.model.parameters(), new_model.parameters()):
            p_swa.data.mul_(1 - alpha).add_(p_new.data, alpha=alpha)
    def get_model(self): return self.model

# ---------- Chunked inference (rel-bias is O(B*L*L*32) per layer) ----------
@torch.no_grad()
def predict_chunked(model, X, chunk=64):
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    model.eval(); out = []
    for i in range(0, X.size(0), chunk):
        xb = X[i:i+chunk].to(DEVICE, non_blocking=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            logit = model(xb).float()
        out.append(logit.cpu())
    return torch.cat(out, dim=0).numpy()

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=80, batch_size=192, lr=5e-4,
                focal_gamma=2.0, swa_start=55, name=''):
    print(f"\n--- Train {name} | params={sum(p.numel() for p in model.parameters()):,} | "
          f"gamma={focal_gamma} | SWA@{swa_start} ---")
    model = model.to(DEVICE)
    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    pw = float(n_neg) / max(1, n_pos)
    weights = np.where(y_train == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    X_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                          sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)   # keep on CPU

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = FocalLoss(alpha=1.0, gamma=focal_gamma, pos_weight=pw)
    scaler = GradScaler(enabled=(torch.cuda.is_available() and not USE_BF16))
    best_auc = 0.0; best_state = None; swa_model = None

    for epoch in range(epochs):
        model.train(); total_loss = 0.0; nb = 0
        for xb, yb in train_dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                loss = criterion(model(xb), yb)
            if USE_BF16:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); nb += 1
        scheduler.step()
        if epoch >= swa_start:
            if swa_model is None:
                swa_model = SWAModel(model); print(f"  [SWA] start @ epoch {epoch+1}")
            else:
                swa_model.update(model)
        v_logit = predict_chunked(model, X_v, chunk=128)
        v_prob = 1.0 / (1.0 + np.exp(-v_logit))
        val_auc = roc_auc_score(y_val, v_prob)
        improved = val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or improved:
            tag = ' *' if improved else ''
            swa_tag = ' [SWA]' if epoch >= swa_start else ''
            print(f"  Ep {epoch+1:3d} | Loss:{total_loss/nb:.4f} | AUC:{val_auc:.4f} | Best:{best_auc:.4f}{tag}{swa_tag}")

    model.load_state_dict(best_state)
    swa_auc = 0.0
    if swa_model:
        sm = swa_model.get_model().to(DEVICE)
        sl = predict_chunked(sm, X_v, chunk=128)
        sp = 1.0 / (1.0 + np.exp(-sl))
        swa_auc = roc_auc_score(y_val, sp)
        print(f"  Best-ckpt AUC: {best_auc:.4f} | SWA AUC: {swa_auc:.4f} ({swa_model.n} snaps)")
    return model, best_auc, swa_model, swa_auc

# ---------- APFD eval (with rotation probe) ----------
def compute_apfd(pids, td):
    n = len(pids)
    fp = [i + 1 for i, t in enumerate(pids) if td[t]['test_outcome'] == 'FAIL']
    m = len(fp); return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0

def _feats(data, means, stds, rot_deg=0.0):
    if rot_deg == 0.0:
        seqs = [resample_to_len(extract_invariant_7ch(get_pts(tc)), SEQ_LEN) for tc in data]
    else:
        c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        seqs = []
        for tc in data:
            pts = np.array(get_pts(tc), dtype=np.float64) @ R.T
            seqs.append(resample_to_len(extract_invariant_7ch(pts.tolist()), SEQ_LEN))
    arr = np.array(seqs, dtype=np.float32)
    return (arr - means) / stds

def predict_probs(model, feats):
    Xt = torch.tensor(feats, dtype=torch.float32).permute(0, 2, 1)
    logit = predict_chunked(model, Xt, chunk=128)
    return 1.0 / (1.0 + np.exp(-logit))

def multi_trial_apfd(eval_data, model, means, stds, sample_size=None,
                     n_trials=30, name='', rot_deg=0.0):
    if sample_size is None:
        sample_size = max(50, int(0.3 * len(eval_data)))
    sample_size = min(sample_size, len(eval_data))
    feats_full = _feats(eval_data, means, stds, rot_deg=rot_deg)
    probs_full = predict_probs(model, feats_full)
    apfds = []
    for t in range(n_trials):
        rng = np.random.RandomState(42 + t)
        idx = rng.permutation(len(eval_data))[:sample_size]
        ed = [eval_data[i] for i in idx]
        td = {get_id(tc): tc for tc in ed}; ids = [get_id(tc) for tc in ed]
        probs = probs_full[idx]
        pids = [t for _, t in sorted(zip(probs, ids), key=lambda x: -x[0])]
        apfds.append(compute_apfd(pids, td))
    rotag = '' if rot_deg == 0.0 else f' [rot={rot_deg:+.0f}deg]'
    print(f"  {name:50s} APFD={np.mean(apfds):.4f}+/-{np.std(apfds):.4f} "
          f"({n_trials} trials, |S|={sample_size}){rotag}")
    return float(np.mean(apfds)), float(np.std(apfds))

# ---------- Protocol ----------
GAMMA = 2.5
EPOCHS = 80
N_TRIALS = 30
BATCH = 192
SWA_START = int(EPOCHS * 2 / 3)

def normalize(X, means, stds): return (X - means) / stds

def main():
    t0 = time.time()
    tags = ('0-1', '0-3', '0-5')

    print("\n" + "=" * 72)
    print("SE(2)-INVARIANT cross-threshold transfer on OOB")
    print("Theory: f(R r + t) = f(r). Verified by rotation probe (Delta ~ 0).")
    print("=" * 72)

    bundle = {}
    for tag in tags:
        path = resolve_dir(tag)
        data = load_oob_dir(path)
        y_all = np.array([1 if is_fail(tc) else 0 for tc in data])
        idx_tr, idx_te = train_test_split(np.arange(len(data)), test_size=0.2,
                                          stratify=y_all, random_state=42)
        train_data = [data[i] for i in idx_tr]
        test_data = [data[i] for i in idx_te]
        print(f"\n[Featurize OOB-{tag}] {len(train_data)} train / {len(test_data)} test")
        X_tr, y_tr = prepare_data(train_data)
        X_te, y_te = prepare_data(test_data)
        bundle[tag] = {'train_data': train_data, 'test_data': test_data,
                       'X_tr': X_tr, 'y_tr': y_tr, 'X_te': X_te, 'y_te': y_te}

    trained = {}
    for src in tags:
        b = bundle[src]
        means = b['X_tr'].mean(axis=(0, 1)); stds = b['X_tr'].std(axis=(0, 1))
        stds[stds < 1e-8] = 1.0
        X_tr_n = normalize(b['X_tr'], means, stds)
        X_te_n = normalize(b['X_te'], means, stds)
        model = SE2RoadNet(in_ch=7, d_model=192, depth=6, nhead=8, ff=512, dropout=0.1)
        model, auc, swa_m, swa_auc = train_model(
            model, X_tr_n, b['y_tr'], X_te_n, b['y_te'],
            epochs=EPOCHS, batch_size=BATCH, lr=5e-4,
            focal_gamma=GAMMA, swa_start=SWA_START,
            name=f'SE2 src=OOB-{src}')
        eval_model = swa_m.get_model() if swa_m else model
        trained[src] = {'model': eval_model, 'means': means, 'stds': stds,
                        'auc': auc, 'swa_auc': swa_auc}

    matrix = {src: {} for src in tags}
    print(f"\n{'='*70}\nSE(2) TRANSFER MATRIX (rows=train src, cols=eval tgt)\n{'='*70}")
    for src in tags:
        for tgt in tags:
            tgt_test = bundle[tgt]['test_data']
            apfd, std = multi_trial_apfd(tgt_test, trained[src]['model'],
                                         trained[src]['means'], trained[src]['stds'],
                                         n_trials=N_TRIALS,
                                         name=f'src={src} -> tgt={tgt}')
            matrix[src][tgt] = {'apfd': apfd, 'apfd_std': std}

    print(f"\n{'src \\ tgt':>10s}" + ''.join(f"{'OOB-'+t:>15s}" for t in tags))
    for src in tags:
        row = f"{'OOB-'+src:>10s}"
        for tgt in tags:
            v = matrix[src][tgt]['apfd']
            mark = '*' if src == tgt else ' '
            row += f"  {v:.4f}{mark}      "
        print(row)

    # ----- SE(2) rotation-invariance probe (within-threshold diagonal) -----
    print(f"\n{'='*70}\nROTATION-INVARIANCE PROBE (Delta APFD vs rot=0)\n{'='*70}")
    probe = {}
    for src in tags:
        tgt_test = bundle[src]['test_data']
        m, mu, sd = trained[src]['model'], trained[src]['means'], trained[src]['stds']
        base, _ = multi_trial_apfd(tgt_test, m, mu, sd, n_trials=N_TRIALS,
                                   name=f'probe src={src} rot=0', rot_deg=0.0)
        rots = {}
        for rot in [30.0, 60.0, 90.0, 180.0, -45.0]:
            a, _ = multi_trial_apfd(tgt_test, m, mu, sd, n_trials=N_TRIALS,
                                    name=f'probe src={src} rot={rot:+.0f}',
                                    rot_deg=rot)
            rots[str(rot)] = {'apfd': a, 'delta': a - base}
        max_abs = max(abs(v['delta']) for v in rots.values())
        print(f"  src={src}: max |Delta APFD| over 5 rotations = {max_abs:.6f}")
        probe[src] = {'base': base, 'rots': rots, 'max_abs_delta': max_abs}

    out = os.path.join(OUTPUT_DIR, 'oob_se2_transfer_matrix.json')
    payload = {
        'arch': 'SE2RoadNet (7ch invariant + rel-arclength attn)',
        'gamma': GAMMA, 'epochs': EPOCHS, 'n_trials': N_TRIALS,
        'matrix': matrix,
        'aucs': {s: {'auc': trained[s]['auc'], 'swa_auc': trained[s]['swa_auc']}
                 for s in tags},
        'rotation_probe': probe,
    }
    with open(out, 'w') as f: json.dump(payload, f, indent=2)
    print(f"\nSaved {out}")

    sd_path = os.path.join(OUTPUT_DIR, 'roadse2_oob_transfer_models.pt')
    torch.save({s: {'state': trained[s]['model'].state_dict(),
                    'means': trained[s]['means'].tolist(),
                    'stds': trained[s]['stds'].tolist(),
                    'arch': dict(in_ch=7, d_model=192, depth=6, nhead=8, ff=512)}
                for s in tags}, sd_path)
    print(f"Saved {sd_path}")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == '__main__':
    main()
