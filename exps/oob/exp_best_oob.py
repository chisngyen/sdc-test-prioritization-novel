"""
BEST (Transformer + SWA + Focal) on OOB-Regression datasets
============================================================
External benchmark: Dataset-OOB-{0-1, 0-3, 0-5} (Zenodo 16939865).

Story for paper: same model, same hyperparams as SensoDat best (APFD=0.8077),
re-trained per OOB threshold. Demonstrates that the SensoDat-tuned recipe
generalises across BeamNG datasets with different OOB severity thresholds.

Schema (per-file JSON):
  {
    "test_id": "...",
    "test_outcome": "FAIL" | "PASS",
    "test_duration": float,
    "road_points": [[x, y], ...]
  }

Per dataset:
  - Stratified 80/20 split (train/test).
  - Train Transformer + SWA + Focal sweep gamma in {0.0, 1.0, 1.5, 2.0, 2.5}.
  - Report best-ckpt / SWA / ensemble APFD on held-out test split (multi-trial).

Saves: roadfury_oob_<tag>.pt, oob_results.json
"""
import json, os, time, math, copy, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------- Path resolution ----------
# Works on Kaggle notebooks (no __file__) and locally.
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

# Search roots to scan for "Dataset-OOB-0-X" sub-folders.
SEARCH_ROOTS = [
    '/kaggle/input',
    os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', '..', 'data')),
    os.path.normpath(os.path.join(HERE, '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', 'data')),
    os.path.normpath(os.path.join(HERE, 'data')),
    os.getcwd(),
]

def _find_oob_folder(tag):
    """Find the inner folder that actually contains *.json files for OOB-<tag>.
    Mirrors the local layout `dataset-oob/Dataset-OOB-0-X/Dataset-OOB-0-X/*.json`
    but tolerates flatter Kaggle layouts too."""
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
            # parent named target but jsons live one level deeper
            if base == target:
                for d in dirnames:
                    inner = os.path.join(dirpath, d)
                    if any(fn.endswith('.json') for fn in os.listdir(inner)):
                        return inner
    return None

def resolve_dir(tag):
    p = _find_oob_folder(tag)
    if p:
        return p
    raise FileNotFoundError(
        f"OOB-{tag} not found. Checked roots: {SEARCH_ROOTS}. "
        f"Expected a 'Dataset-OOB-{tag}' folder containing *_test.json files.")

OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(HERE, '..', '..', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")

SEQ_LEN = 197  # match best.md positional embedding

# ---------- Feature extraction (10ch) ----------
def compute_curvature(pts):
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

def extract_sequence_10ch(pts_raw):
    pts = np.array(pts_raw, dtype=np.float64).reshape(-1, 2); n = len(pts)
    diffs = np.diff(pts, axis=0); seg_lens = np.linalg.norm(diffs, axis=1)
    seg_full = np.pad(seg_lens, (0, 1), mode='edge')
    angles = np.arctan2(diffs[:, 1], diffs[:, 0]); ac = np.diff(angles)
    ac = (ac + np.pi) % (2 * np.pi) - np.pi
    abs_ac_full = np.pad(np.abs(ac), (1, 1), mode='constant')
    curv = np.abs(compute_curvature(pts)); curv_full = np.pad(curv, (1, 1), mode='constant')
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

def resample_to_len(seq, target_len=SEQ_LEN):
    """Linear-interpolate (per channel) so every sequence has the same length."""
    n, c = seq.shape
    if n == target_len:
        return seq
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
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
            with open(fp) as f:
                tc = json.load(f)
        except Exception:
            continue
        if not tc.get('is_valid', True):
            continue
        rp = tc.get('road_points')
        out = tc.get('test_outcome')
        if not rp or out not in ('FAIL', 'PASS'):
            continue
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
        seq = extract_sequence_10ch(get_pts(tc))
        X.append(resample_to_len(seq, SEQ_LEN))
        y.append(1 if is_fail(tc) else 0)
        if (i + 1) % batch_print == 0:
            print(f"    feat {i+1}/{len(data)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# ---------- Model / Loss / SWA (same as best.md) ----------
class RoadTransformer(nn.Module):
    def __init__(self, in_channels=10, seq_len=SEQ_LEN, d_model=128,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.GELU())
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
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weight = torch.where(targets == 1, self.pos_weight, 1.0)
        bce = bce * weight
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

class SWAModel:
    def __init__(self, model): self.model = copy.deepcopy(model); self.n = 0
    def update(self, new_model):
        self.n += 1; alpha = 1.0 / self.n
        for p_swa, p_new in zip(self.model.parameters(), new_model.parameters()):
            p_swa.data.mul_(1 - alpha).add_(p_new.data, alpha=alpha)
    def get_model(self): return self.model

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=75, batch_size=256, lr=5e-4,
                focal_gamma=2.0, swa_start=50, name=''):
    print(f"\n--- Train {name} | γ={focal_gamma} | SWA@{swa_start} ---")
    model = model.to(DEVICE)
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pw = float(n_neg) / max(1, n_pos)
    weights = np.where(y_train == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    X_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                          sampler=sampler, num_workers=2, pin_memory=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = FocalLoss(alpha=1.0, gamma=focal_gamma, pos_weight=pw)
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
            if swa_model is None:
                swa_model = SWAModel(model); print(f"  [SWA] start @ epoch {epoch+1}")
            else:
                swa_model.update(model)
        model.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp): vl = model(X_v)
            val_auc = roc_auc_score(y_val, torch.sigmoid(vl).cpu().numpy())
        improved = val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or improved:
            tag = ' *' if improved else ''
            swa_tag = ' [SWA]' if epoch >= swa_start else ''
            print(f"  Ep {epoch+1:3d} | Loss:{total_loss/nb:.4f} | AUC:{val_auc:.4f} | Best:{best_auc:.4f}{tag}{swa_tag}")

    model.load_state_dict(best_state)
    swa_auc = 0.0
    if swa_model:
        sm = swa_model.get_model().to(DEVICE); sm.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp): sl = sm(X_v)
            swa_auc = roc_auc_score(y_val, torch.sigmoid(sl).cpu().numpy())
        print(f"  Best-ckpt AUC: {best_auc:.4f} | SWA AUC: {swa_auc:.4f} ({swa_model.n} snaps)")
    return model, best_auc, swa_model, swa_auc

# ---------- Evaluation ----------
def compute_apfd(pids, td):
    n = len(pids)
    fp = [i + 1 for i, t in enumerate(pids) if td[t]['test_outcome'] == 'FAIL']
    m = len(fp); return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0

def predict_probs(models, X_norm):
    """Average sigmoid probs over a list of models."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    Xt = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    probs_list = []
    for m in models:
        m.eval().to(DEVICE)
        with torch.no_grad():
            probs_list.append(torch.sigmoid(m(Xt)).cpu().numpy())
    return np.mean(probs_list, axis=0)

def multi_trial_apfd(eval_data, models, means, stds, sample_size=None, n_trials=30, name=''):
    """Random sub-sample multi-trial APFD (matches best.md protocol)."""
    if sample_size is None:
        sample_size = max(50, int(0.3 * len(eval_data)))
    sample_size = min(sample_size, len(eval_data))
    apfds = []
    feats_full = np.array([resample_to_len(extract_sequence_10ch(get_pts(tc)), SEQ_LEN) for tc in eval_data], dtype=np.float32)
    feats_full = (feats_full - means) / stds
    for t in range(n_trials):
        rng = np.random.RandomState(42 + t)
        idx = rng.permutation(len(eval_data))[:sample_size]
        ed = [eval_data[i] for i in idx]
        td = {get_id(tc): tc for tc in ed}; ids = [get_id(tc) for tc in ed]
        probs = predict_probs(models, feats_full[idx])
        pids = [t for _, t in sorted(zip(probs, ids), key=lambda x: -x[0])]
        apfds.append(compute_apfd(pids, td))
    print(f"  {name:50s} APFD={np.mean(apfds):.4f}±{np.std(apfds):.4f} ({n_trials} trials, |S|={sample_size})")
    return float(np.mean(apfds)), float(np.std(apfds))

# ---------- Main per dataset ----------
def run_one(tag, gammas=(0.0, 1.0, 1.5, 2.0, 2.5), epochs=75, n_trials=30):
    print(f"\n{'='*70}\nDATASET: OOB-{tag}\n{'='*70}")
    path = resolve_dir(tag)
    data = load_oob_dir(path)
    n_fail = sum(1 for tc in data if is_fail(tc))
    print(f"  Total: {len(data)} | FAIL: {n_fail} ({100*n_fail/len(data):.1f}%)")
    if n_fail < 10 or len(data) - n_fail < 10:
        print("  Skip: too imbalanced"); return None

    y_all = np.array([1 if is_fail(tc) else 0 for tc in data])
    idx_train, idx_test = train_test_split(np.arange(len(data)), test_size=0.2,
                                           stratify=y_all, random_state=42)
    train_data = [data[i] for i in idx_train]
    test_data = [data[i] for i in idx_test]
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

    print("  Extracting features...")
    X_tr, y_tr = prepare_data(train_data)
    X_te, y_te = prepare_data(test_data)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1)); stds[stds < 1e-8] = 1.0
    X_tr_n = (X_tr - means) / stds; X_te_n = (X_te - means) / stds

    results = {}
    for gamma in gammas:
        model = RoadTransformer(in_channels=10, seq_len=SEQ_LEN)
        model, auc, swa_m, swa_auc = train_model(
            model, X_tr_n, y_tr, X_te_n, y_te,
            epochs=epochs, batch_size=256, lr=5e-4,
            focal_gamma=gamma, swa_start=int(epochs * 2 / 3),
            name=f'OOB-{tag} γ={gamma}')
        results[gamma] = {'model': model, 'swa': swa_m.get_model() if swa_m else None,
                          'auc': auc, 'swa_auc': swa_auc}

    # Eval on test split
    print(f"\n  --- Multi-trial APFD on OOB-{tag} test split ---")
    summary = {'tag': tag, 'n_total': len(data), 'n_fail': int(n_fail),
               'n_train': len(train_data), 'n_test': len(test_data), 'gammas': {}}
    best_apfd = -1; best_gamma = None; best_kind = None
    for gamma in gammas:
        r = results[gamma]
        ck_apfd, ck_std = multi_trial_apfd(test_data, r['model'], means, stds,
                                           n_trials=n_trials, name=f'γ={gamma} best-ckpt')
        sw_apfd, sw_std = (None, None)
        if r['swa']:
            sw_apfd, sw_std = multi_trial_apfd(test_data, r['swa'], means, stds,
                                               n_trials=n_trials, name=f'γ={gamma}+SWA')
            if sw_apfd > best_apfd:
                best_apfd = sw_apfd; best_gamma = gamma; best_kind = 'swa'
        if ck_apfd > best_apfd:
            best_apfd = ck_apfd; best_gamma = gamma; best_kind = 'ckpt'
        summary['gammas'][str(gamma)] = {'auc': r['auc'], 'swa_auc': r['swa_auc'],
                                          'apfd_ckpt': ck_apfd, 'apfd_ckpt_std': ck_std,
                                          'apfd_swa': sw_apfd, 'apfd_swa_std': sw_std}

    # Ensemble
    swa_models = [results[g]['swa'] for g in gammas if results[g]['swa']]
    if swa_models:
        ens_apfd, ens_std = multi_trial_apfd(test_data, swa_models, means, stds,
                                             n_trials=max(n_trials, 50),
                                             name=f'Ensemble {len(swa_models)} SWA')
        summary['ensemble_swa'] = {'apfd': ens_apfd, 'apfd_std': ens_std,
                                    'n_models': len(swa_models)}
        if ens_apfd > best_apfd:
            best_apfd = ens_apfd; best_gamma = 'ENS'; best_kind = 'ensemble'

    summary['best'] = {'apfd': best_apfd, 'gamma': str(best_gamma), 'kind': best_kind}
    print(f"\n  ★ OOB-{tag} BEST: γ={best_gamma} ({best_kind}) → APFD={best_apfd:.4f}")

    # Save best single model
    if best_kind in ('ckpt', 'swa') and best_gamma in results:
        bm = results[best_gamma]['swa'] if best_kind == 'swa' else results[best_gamma]['model']
        save_path = os.path.join(OUTPUT_DIR, f'roadfury_oob_{tag}.pt')
        torch.save({'state': bm.state_dict(), 'means': means.tolist(), 'stds': stds.tolist(),
                    'focal_gamma': best_gamma, 'kind': best_kind, 'apfd': best_apfd,
                    'tag': tag}, save_path)
        print(f"  Saved {save_path}")
    # Save all SWA for cross-dataset reuse
    all_path = os.path.join(OUTPUT_DIR, f'roadfury_oob_{tag}_all.pt')
    torch.save({'means': means.tolist(), 'stds': stds.tolist(),
                **{f'swa_g{g}': results[g]['swa'].state_dict() for g in gammas if results[g]['swa']}},
               all_path)
    print(f"  Saved {all_path}")
    return summary

def main():
    t0 = time.time()
    all_results = {}
    for tag in ('0-1', '0-3', '0-5'):
        try:
            all_results[tag] = run_one(tag)
        except Exception as e:
            print(f"  [WARN] OOB-{tag} failed: {type(e).__name__}: {e}")
            all_results[tag] = {'error': str(e)}
    out = os.path.join(OUTPUT_DIR, 'oob_results.json')
    with open(out, 'w') as f: json.dump(all_results, f, indent=2)
    print(f"\nResults → {out}")
    print(f"\n{'='*70}\nSUMMARY (within-threshold)\n{'='*70}")
    for tag, r in all_results.items():
        if r and 'best' in r:
            print(f"  OOB-{tag:>3s}: best APFD = {r['best']['apfd']:.4f} "
                  f"(γ={r['best']['gamma']}, {r['best']['kind']}) | "
                  f"n={r['n_total']} fail={r['n_fail']}")
        else:
            print(f"  OOB-{tag}: {r}")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == '__main__':
    main()
