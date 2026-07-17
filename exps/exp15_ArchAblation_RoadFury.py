"""
Exp 15 -- RoadFury ARCHITECTURE ablation (why 4 blocks? why d_model=128?)
=========================================================================
Every prior exp fixes the RoadFury backbone at (d_model=128, nhead=8,
num_layers=4, dim_feedforward=512, dropout=0.1) and asks a reviewer to
take that on faith. This file *earns* those numbers: it sweeps each
architectural axis one-factor-at-a-time (OFAT) around the shipped
configuration, holding the winning training recipe (Focal gamma=2.5 +
SWA, 75 ep, batch=256, lr=5e-4) byte-for-byte identical, and reports
APFD +/- sigma, val-AUC, param count, and wall-clock per setting.

Reviewer questions this answers directly
----------------------------------------
  * "Why 4 Transformer blocks and not 3 / 5 / 6?"   -> DEPTH axis
  * "Why project 10 -> 128 and not 64 / 96 / 192 / 256?" -> WIDTH axis
  * "Why 8 heads? why FFN=512 (4x)? why dropout 0.1?"  -> NHEAD / FFN /
    DROPOUT axes (these jointly justify the d_model=128 choice, since
    128 = 8 heads x 16 head-dim and 512 = 4 x 128)
  * "Does the [CLS]-token readout matter vs mean-pool?"  -> POOL axis
  * "Does the 64-wide classifier head matter?"           -> HEAD_HID axis

Protocol (identical to exp00_Basline.py so numbers are comparable)
------------------------------------------------------------------
  * Train on SensoDat train split, early-select on SensoDat test AUC.
  * Report multi-trial APFD on the Competition split (956 tests, 30
    sub-trials of 287, seed 42+t) -- this is the exact protocol behind
    the headline best-single 0.8066 +/- 0.0124.
  * N_SEEDS models per config (default 3): the publication number is the
    APFD mean +/- sigma ACROSS SEEDS, because architecture variance
    across re-inits is what a reviewer should trust, not a single lucky
    run. Per-seed multi-trial sigma is also logged.

Baseline (the shipped RoadFury; the pivot every axis rotates around)
--------------------------------------------------------------------
  d_model=128  nhead=8  num_layers=4  dim_feedforward=512  dropout=0.1
  cls_hidden=64  pool='cls'   -> ~0.83 M params, ~3 min/train on H100

Outputs
-------
  * arch_ablation_results.json  -- full grid, machine-readable
  * ASCII tables per axis printed to stdout, ready to paste into
    exps/tracker.md or the paper (pure ASCII, cp1252-safe).

Runnable as a single pasted file on Kaggle (SEARCH_ROOTS auto-discovery)
or locally (../../data). Set FAST=True for a 1-seed/short-epoch smoke run.
"""
import os, sys, json, time, math, copy, glob, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------- config
FAST = False              # True -> 1 seed, 20 epochs, fewer values (smoke test)
N_SEEDS      = 3          # models trained per config (across-seed sigma)
EPOCHS       = 75
BATCH        = 256
LR           = 5e-4
GAMMA        = 2.5        # SensoDat winning focal gamma (frozen)
SWA_START    = 50
SEQ_LEN      = 197
N_TRIALS     = 30         # competition multi-trial count
COMP_LO, COMP_N = 334, 287   # exp00's fixed competition sub-trial window

# The shipped RoadFury configuration; every OFAT axis holds these fixed
# except the one being swept.
BASELINE_CFG = dict(d_model=128, nhead=8, num_layers=4,
                    dim_feedforward=512, dropout=0.1,
                    cls_hidden=64, pool='cls')

# Axes to sweep. Each entry: axis_key -> (param_name, [values]).
# ffn_mult is expressed as a multiplier of d_model (baseline 512 = 4x128).
# NOTE on orthogonality: the WIDTH axis scales dim_feedforward with d_model
# (holding the 4x ratio) so "width" is a single clean knob; the FFN axis
# then varies the ratio alone at d_model=128. Otherwise a fixed FFN=512
# would give d_model=64 an 8x FFN and confound the two axes.
AXES = {
    'DEPTH  (num_layers)':      ('num_layers',   [2, 3, 4, 5, 6]),
    'WIDTH  (d_model)':         ('d_model',      [64, 96, 128, 160, 192, 256]),
    'NHEAD  (attn heads)':      ('nhead',        [4, 8, 16]),
    'FFN    (ffn multiplier)':  ('ffn_mult',     [2, 4, 8]),
    'DROPOUT':                  ('dropout',      [0.0, 0.1, 0.2]),
    'HEAD_HID (classifier)':    ('cls_hidden',   [32, 64, 128]),
    'POOL   (readout)':         ('pool',         ['cls', 'mean']),
}
if FAST:
    N_SEEDS, EPOCHS = 1, 20
    AXES = {
        'DEPTH  (num_layers)': ('num_layers', [3, 4, 5]),
        'WIDTH  (d_model)':    ('d_model',    [96, 128, 192]),
    }

# ------------------------------------------------------------- discovery
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

SEARCH_ROOTS = [
    '/kaggle/input',
    os.path.normpath(os.path.join(HERE, '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', 'data')),
    os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', '..', 'data')),
    os.getcwd(),
]
OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') \
    else os.path.normpath(os.path.join(HERE, '..', 'models'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")


def find_file(basename):
    """First path named `basename` under any SEARCH_ROOT (nested mounts ok)."""
    seen = set()
    for root in SEARCH_ROOTS:
        if not root or not os.path.isdir(root) or root in seen:
            continue
        seen.add(root)
        direct = os.path.join(root, basename)
        if os.path.isfile(direct):
            return direct
        for dirpath, _, filenames in os.walk(root):
            if basename in filenames:
                return os.path.join(dirpath, basename)
    return None


# ------------------------------------------------------- feature pipeline
def compute_curvature(pts):
    n = len(pts); curv = np.zeros(max(0, n - 2))
    for i in range(n - 2):
        x1, y1 = pts[i]; x2, y2 = pts[i + 1]; x3, y3 = pts[i + 2]
        a = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        s = 0.5 * (a + b + c); at = s * (s - a) * (s - b) * (s - c)
        if at <= 1e-10:
            curv[i] = 0.0
        else:
            R = a * b * c / (4 * math.sqrt(at)); curv[i] = 1.0 / R if R > 0 else 0.0
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
        s, e = max(0, i - hw), min(n, i + hw + 1); local_std[i] = np.std(curv_full[s:e])
    curv_accel_full = np.pad(np.diff(curv_deriv_full), (0, 1), mode='constant')
    return np.column_stack([seg_full, abs_ac_full, curv_full, curv_deriv_full, cum_dist_norm,
                            heading_sin, heading_cos, rel_pos, local_std, curv_accel_full]).astype(np.float32)


def resample(seq, target_len=SEQ_LEN):
    n, c = seq.shape
    if n == target_len:
        return seq
    x_old = np.linspace(0, 1, n); x_new = np.linspace(0, 1, target_len)
    out = np.empty((target_len, c), dtype=np.float32)
    for ch in range(c):
        out[:, ch] = np.interp(x_new, x_old, seq[:, ch])
    return out


def get_pts(tc):
    pts = tc['road_points']
    if pts and isinstance(pts[0], dict):
        return [[p['x'], p['y']] for p in pts]
    return [[p[0], p[1]] for p in pts]


def get_outcome(tc):
    md = tc.get('meta_data')
    if isinstance(md, dict):
        ti = md.get('test_info') or {}
        return ti.get('test_outcome')
    return tc.get('test_outcome')


def is_fail(tc):
    return get_outcome(tc) == 'FAIL'


def get_id(tc):
    _id = tc.get('_id')
    if isinstance(_id, dict):
        return _id.get('$oid') or str(_id)
    return _id


def load_json(path):
    print(f"Loading {os.path.basename(path)} ..."); t0 = time.time()
    with open(path) as f:
        data = json.load(f)
    print(f"  {len(data)} tests in {time.time() - t0:.1f}s")
    return data


def prepare(data):
    X = np.array([resample(extract_sequence_10ch(get_pts(tc))) for tc in data], dtype=np.float32)
    y = np.array([1 if is_fail(tc) else 0 for tc in data], dtype=np.int64)
    return X, y


# --------------------------------------------------------------- model
class RoadTransformer(nn.Module):
    """Shipped RoadFury backbone, fully parameterised for ablation.

    `pool='cls'` reads the prepended [CLS] token (the shipped behaviour);
    `pool='mean'` averages the L road tokens instead, so the readout choice
    itself becomes an ablatable axis.
    """
    def __init__(self, in_channels=10, seq_len=SEQ_LEN, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.1,
                 cls_hidden=64, pool='cls'):
        super().__init__()
        self.pool = pool
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model),
                                        nn.LayerNorm(d_model), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, cls_hidden), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(cls_hidden, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1); B, L, C = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :L + 1, :]
        x = self.transformer(x)
        feat = x[:, 0, :] if self.pool == 'cls' else x[:, 1:, :].mean(dim=1)
        return self.classifier(feat).squeeze(-1)


def build_model(cfg):
    """cfg carries either dim_feedforward directly or ffn_mult (x d_model)."""
    c = dict(cfg)
    if 'ffn_mult' in c:
        c['dim_feedforward'] = int(c.pop('ffn_mult') * c['d_model'])
    return RoadTransformer(in_channels=10, seq_len=SEQ_LEN, **c)


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
    def get_model(self):
        return self.model


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------- train
def train_one(cfg, X_tr, y_tr, X_val, y_val, seed, epochs=EPOCHS):
    set_seed(seed)
    model = build_model(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
    pw = float(n_neg) / max(1, n_pos)
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    X_t = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(y_tr, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH,
                          sampler=sampler, num_workers=2, pin_memory=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = FocalLoss(alpha=1.0, gamma=GAMMA, pos_weight=pw)
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    swa_at = min(SWA_START, epochs - 5)
    best_auc = 0.0; best_state = None; swa_model = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        scheduler.step()
        if epoch >= swa_at:
            if swa_model is None:
                swa_model = SWAModel(model)
            else:
                swa_model.update(model)
        model.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp):
                vl = model(X_v)
            try:
                val_auc = roc_auc_score(y_val, torch.sigmoid(vl).cpu().numpy())
            except Exception:
                val_auc = 0.5
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    swa = swa_model.get_model().to(DEVICE) if swa_model else None
    swa_auc = 0.0
    if swa is not None:
        swa.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp):
                sl = swa(X_v)
            try:
                swa_auc = roc_auc_score(y_val, torch.sigmoid(sl).cpu().numpy())
            except Exception:
                swa_auc = 0.5
    return model, swa, float(best_auc), float(swa_auc), int(n_params)


# ------------------------------------------------------------ evaluate
def compute_apfd(pids, td):
    n = len(pids)
    fp = [i + 1 for i, t in enumerate(pids) if is_fail(td[t])]
    m = len(fp)
    return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0


def multi_trial(eval_data, model, means, stds, n_trials=N_TRIALS):
    """Competition multi-trial APFD, identical window to exp00_Basline.py.

    Returns (mean, sigma) of APFD over `n_trials` fixed sub-trials.
    """
    model.eval().to(DEVICE)
    feats = np.array([resample(extract_sequence_10ch(get_pts(tc))) for tc in eval_data],
                     dtype=np.float32)
    feats = (feats - means) / stds
    use_window = len(eval_data) >= COMP_LO + COMP_N
    sample_n = COMP_N if use_window else min(max(50, int(0.3 * len(eval_data))), len(eval_data))
    apfds = []
    for t in range(n_trials):
        rng = np.random.RandomState(42 + t)
        idx = rng.permutation(len(eval_data))
        idx = idx[COMP_LO:COMP_LO + COMP_N] if use_window else idx[:sample_n]
        ed = [eval_data[i] for i in idx]
        td = {get_id(tc): tc for tc in ed}; ids = [get_id(tc) for tc in ed]
        Xt = torch.tensor(feats[idx], dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
        with torch.no_grad():
            probs = torch.sigmoid(model(Xt)).cpu().numpy()
        pids = [t2 for _, t2 in sorted(zip(probs, ids), key=lambda x: -x[0])]
        apfds.append(compute_apfd(pids, td))
    return float(np.mean(apfds)), float(np.std(apfds))


# --------------------------------------------------- one config, N seeds
def eval_config(cfg, data, label):
    """Train N_SEEDS models for `cfg`, return aggregated metrics."""
    X_tr, y_tr = data['X_tr'], data['y_tr']
    X_te, y_te = data['X_te'], data['y_te']
    means, stds = data['means'], data['stds']
    eval_data = data['eval_data']

    per_seed = []
    t0 = time.time()
    for s in range(N_SEEDS):
        model, swa, auc, swa_auc, n_params = train_one(cfg, X_tr, y_tr, X_te, y_te, seed=42 + s)
        eval_model = swa if swa is not None else model
        apfd_m, apfd_s = multi_trial(eval_data, eval_model, means, stds)
        per_seed.append(dict(seed=42 + s, params=n_params, auc=auc, swa_auc=swa_auc,
                             apfd=apfd_m, apfd_sigma=apfd_s))
    wall = time.time() - t0

    apfds = np.array([r['apfd'] for r in per_seed])
    aucs = np.array([max(r['auc'], r['swa_auc']) for r in per_seed])
    out = dict(
        label=label, cfg={k: v for k, v in cfg.items()},
        params=per_seed[0]['params'],
        apfd_mean=float(apfds.mean()),
        apfd_sigma_across_seeds=float(apfds.std()),
        apfd_sigma_within_trial=float(np.mean([r['apfd_sigma'] for r in per_seed])),
        auc_mean=float(aucs.mean()),
        wall_sec=float(wall), n_seeds=N_SEEDS,
        per_seed=per_seed,
    )
    print(f"    {label:16s} | params={out['params']:>8,} | AUC={out['auc_mean']:.4f} "
          f"| APFD={out['apfd_mean']:.4f}+/-{out['apfd_sigma_across_seeds']:.4f} "
          f"(within {out['apfd_sigma_within_trial']:.4f}) | {wall/60:.1f}m")
    return out


# ------------------------------------------------------------ tables
def fmt_axis_table(axis_name, param, results, baseline_apfd):
    """ASCII table for one axis; marks winner and Delta vs shipped baseline."""
    lines = []
    lines.append(f"### {axis_name}")
    lines.append("")
    lines.append("| value | params   | AUC    | APFD-comp (30-trial)   | Delta vs base | wall |")
    lines.append("|-------|----------|--------|------------------------|---------------|------|")
    best = max(results, key=lambda r: r['apfd_mean'])
    for r in results:
        val = r['cfg'][param]
        star = ' *' if r is best else '  '
        base_tag = ' (base)' if _is_baseline_value(param, val) else ''
        delta = r['apfd_mean'] - baseline_apfd
        lines.append(
            f"| {str(val):>3}{star}| {r['params']:>8,} | {r['auc_mean']:.4f} "
            f"| {r['apfd_mean']:.4f} +/- {r['apfd_sigma_across_seeds']:.4f} "
            f"| {delta:+.4f}{base_tag:>7} | {r['wall_sec']/60:.1f}m |")
    lines.append("")
    lines.append(f"-> best on this axis: {param}={best['cfg'][param]} "
                 f"(APFD={best['apfd_mean']:.4f} +/- {best['apfd_sigma_across_seeds']:.4f})")
    lines.append("")
    return "\n".join(lines)


def _is_baseline_value(param, val):
    if param == 'ffn_mult':
        return int(val * BASELINE_CFG['d_model']) == BASELINE_CFG['dim_feedforward']
    return BASELINE_CFG.get(param) == val


# --------------------------------------------------------------- main
def main():
    t_start = time.time()

    train_p = find_file('sensodat_train.json')
    test_p = find_file('sensodat_test.json')
    comp_p = find_file('sdc-test-data.json')
    if not train_p or not test_p:
        print("[FATAL] sensodat_train.json / sensodat_test.json not found under:")
        for r in SEARCH_ROOTS:
            print(f"   {r}")
        sys.exit(1)

    train_data = load_json(train_p)
    test_data = load_json(test_p)
    comp_data = load_json(comp_p) if comp_p else test_data
    if not comp_p:
        print("[WARN] competition split not found -> multi-trial APFD falls back to SensoDat test.")

    n_fail = sum(1 for tc in train_data if is_fail(tc))
    print(f"Train: {len(train_data)} ({n_fail} FAIL = {100 * n_fail / len(train_data):.1f}%)")

    print("Extracting features ...")
    X_tr, y_tr = prepare(train_data)
    X_te, y_te = prepare(test_data)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1)); stds[stds < 1e-8] = 1.0
    X_tr = (X_tr - means) / stds; X_te = (X_te - means) / stds

    data = dict(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                means=means, stds=stds, eval_data=comp_data)

    print(f"\n{'=' * 70}")
    print("BASELINE (shipped RoadFury) -- the pivot for every axis")
    print(f"{'=' * 70}")
    print(f"  {BASELINE_CFG}")
    base = eval_config(BASELINE_CFG, data, 'baseline')
    baseline_apfd = base['apfd_mean']

    results = {
        'protocol': ('Focal(gamma=2.5)+SWA, 75ep, batch=256, lr=5e-4; '
                     'multi-trial APFD on Competition (287-of-956, 30 trials); '
                     f'{N_SEEDS} seeds/config, sigma across seeds'),
        'baseline_cfg': BASELINE_CFG,
        'baseline': base,
        'axes': {},
    }

    for axis_name, (param, values) in AXES.items():
        print(f"\n{'=' * 70}\nAXIS: {axis_name}\n{'=' * 70}")
        axis_res = []
        for val in values:
            cfg = dict(BASELINE_CFG)
            if param == 'ffn_mult':
                cfg.pop('dim_feedforward'); cfg['ffn_mult'] = val
            elif param == 'd_model':
                cfg['d_model'] = val
                cfg['dim_feedforward'] = 4 * val   # keep FFN ratio fixed at 4x
            else:
                cfg[param] = val
            # skip architecturally invalid combos (d_model must divide by nhead)
            dm = cfg.get('d_model', BASELINE_CFG['d_model'])
            nh = cfg.get('nhead', BASELINE_CFG['nhead'])
            if dm % nh != 0:
                print(f"    [skip] {param}={val}: d_model {dm} not divisible by nhead {nh}")
                continue
            # reuse the already-trained baseline row when the swept value IS the baseline
            if _is_baseline_value(param, val) and param != 'ffn_mult':
                row = dict(base); row['label'] = f'{param}={val}'
                row['cfg'] = {**{k: v for k, v in BASELINE_CFG.items()}}
                axis_res.append(row)
                print(f"    {param}={val:<6} (baseline, reusing) APFD={row['apfd_mean']:.4f}")
                continue
            row = eval_config(cfg, data, f'{param}={val}')
            axis_res.append(row)
        results['axes'][axis_name] = dict(param=param, rows=axis_res)

        # incremental save so a long run is never lost
        out_path = os.path.join(OUTPUT_DIR, 'arch_ablation_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [save] {out_path}")

    # ------------------------------------------------- ASCII report
    print(f"\n\n{'#' * 70}")
    print("# ARCHITECTURE ABLATION -- paste-ready tables (pure ASCII)")
    print(f"{'#' * 70}\n")
    print(f"Baseline (shipped): {BASELINE_CFG}")
    print(f"Baseline APFD-comp = {baseline_apfd:.4f} +/- "
          f"{base['apfd_sigma_across_seeds']:.4f}  ({base['params']:,} params)\n")
    report = [f"Baseline: {BASELINE_CFG}",
              f"Baseline APFD = {baseline_apfd:.4f} +/- {base['apfd_sigma_across_seeds']:.4f}", ""]
    for axis_name, blob in results['axes'].items():
        tbl = fmt_axis_table(axis_name, blob['param'], blob['rows'], baseline_apfd)
        print(tbl)
        report.append(tbl)

    report_path = os.path.join(OUTPUT_DIR, 'arch_ablation_report.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    print(f"[save] {report_path}")
    print(f"\nTOTAL TIME: {(time.time() - t_start) / 60:.1f} min "
          f"({N_SEEDS} seeds x {sum(len(v[1]) for v in AXES.values())} configs)")


if __name__ == '__main__':
    main()
