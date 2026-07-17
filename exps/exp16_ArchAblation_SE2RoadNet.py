"""
Exp 16 -- SE2RoadNet ARCHITECTURE ablation (why depth=6? why d_model=192?
why 32 RFF? does the equivariant rel-bias even help?)
=========================================================================
Sibling of exp15 (RoadFury ablation), but for the SE(2)-equivariant tower
of Exp 02. SE2RoadNet is a *different* animal: 7-channel frame-invariant
features (no absolute heading/position), a custom InvariantBlock whose
attention carries a RELATIVE-arclength bias from `n_rff` random Fourier
features, and a wider/deeper backbone (d_model=192, depth=6) than RoadFury.
This file earns each of those choices one-factor-at-a-time (OFAT), holding
the Exp-02 training recipe (Focal gamma=1.5 + SWA, 80 ep, batch=384,
lr=5e-4, bf16) byte-for-byte identical.

Reviewer questions this answers directly
----------------------------------------
  * "Why 6 blocks and not 3/4/5/8?"                 -> DEPTH axis
  * "Why d_model=192 and not 96/128/160/256?"       -> WIDTH axis
  * "Why 8 heads (192 = 8 x 24)?"                    -> NHEAD axis
  * "Why FFN=512?"                                   -> FFN axis
  * "Why 32 random Fourier features for the rel-bias? why scale 2.0?"
                                                     -> RFF / RFF_SCALE axes
  * "Does the equivariant relative-arclength bias actually contribute,
    or is invariance carried entirely by the 7-ch features?"  -> REL_BIAS
    axis (True vs False). This is the SIGNATURE SE2 ablation: turning the
    bias off keeps rotation invariance (features are already invariant)
    but isolates the bias's APFD contribution.
  * dropout / classifier-head width -> DROPOUT / HEAD_HID axes

Protocol (identical to exp02_SE2Equivariant.py so numbers are comparable)
-------------------------------------------------------------------------
  * 7-ch invariant features; train on SensoDat train, select on test AUC.
  * Multi-trial APFD on Competition (287-of-956, 30 trials, seed 42+t) --
    the exact protocol behind Exp 02's 0.8048 +/- 0.0118.
  * ROTATION-DELTA column: |APFD(rot=90) - APFD(rot=0)| single-pass on the
    full Competition split. Should stay ~0.0000 for EVERY config, which
    doubly serves the paper: it verifies no architectural knob accidentally
    breaks the invariance theorem.
  * N_SEEDS models/config: publication number is APFD mean +/- sigma across
    seeds; per-seed 30-trial sigma also logged.

Baseline (shipped SE2RoadNet, the pivot every axis rotates around)
------------------------------------------------------------------
  d_model=192  depth=6  nhead=8  ff=512  dropout=0.1
  n_rff=32  rff_scale=2.0  rel_hid=64  rel_bias=True  cls_hidden=64
  -> ~2.11 M params

COST WARNING
------------
SE2's rel-bias is O(B * L^2 * n_rff) PER LAYER -- roughly 8x slower than
RoadFury (~24 min/train on the Exp-02 hardware). The full grid at N_SEEDS=3
is ~a day. Defaults here use N_SEEDS=2 and you can trim `RUN_AXES` to run
one axis at a time. Set FAST=True for a smoke run first. A cost estimate is
printed before training starts.

Outputs: se2_arch_ablation_results.json + se2_arch_ablation_report.md
(paste-ready ASCII tables). Runnable as one pasted file on Kaggle.
"""
import os, sys, json, time, math, copy, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------- config
FAST = False              # True -> 1 seed, 20 epochs, DEPTH+WIDTH only (smoke)
N_SEEDS      = 2          # SE2 is ~8x slower than RoadFury; 2 keeps grid ~<day
EPOCHS       = 80
BATCH        = 384
LR           = 5e-4
GAMMA        = 1.5        # SE2 winning focal gamma (frozen)
SWA_START    = 55
SEQ_LEN      = 197
N_TRIALS     = 30
COMP_LO, COMP_N = 334, 287   # exp02's fixed competition sub-trial window
ROT_PROBE_DEG = 90.0         # single rotation used for the invariance-check column

# Shipped SE2RoadNet configuration; every OFAT axis holds these fixed
# except the one being swept. NOTE: unlike RoadFury, SE2's FFN is an
# ABSOLUTE 512 (not a multiple of d_model), so the WIDTH axis holds ff=512
# fixed and a dedicated FFN axis varies it -- documented below.
BASELINE_CFG = dict(d_model=192, depth=6, nhead=8, ff=512, dropout=0.1,
                    n_rff=32, rff_scale=2.0, rel_hid=64, rel_bias=True,
                    cls_hidden=64)

# axis_key -> (param_name, [values]).
AXES = {
    'DEPTH   (num blocks)':     ('depth',      [3, 4, 5, 6, 8]),
    'WIDTH   (d_model)':        ('d_model',    [96, 128, 160, 192, 256]),
    'NHEAD   (attn heads)':     ('nhead',      [4, 8, 16]),
    'FFN     (feedforward)':    ('ff',         [256, 512, 768, 1024]),
    'RFF     (rel-bias feats)': ('n_rff',      [16, 32, 64]),
    'RFF_SCALE (freq bw)':      ('rff_scale',  [1.0, 2.0, 4.0]),
    'REL_BIAS (equiv attn)':    ('rel_bias',   [True, False]),
    'DROPOUT':                  ('dropout',    [0.0, 0.1, 0.2]),
    'HEAD_HID (classifier)':    ('cls_hidden', [32, 64, 128]),
}
# Run only these axes (edit to run one at a time given SE2 cost).
RUN_AXES = list(AXES.keys())
if FAST:
    N_SEEDS, EPOCHS = 1, 20
    AXES = {
        'DEPTH   (num blocks)': ('depth',   [4, 6, 8]),
        'WIDTH   (d_model)':    ('d_model', [128, 192, 256]),
    }
    RUN_AXES = list(AXES.keys())

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
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
torch.set_float32_matmul_precision('high')
print(f"Device: {DEVICE} | bf16: {USE_BF16}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")


def find_file(basename):
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


# ----------------------------------------- frame-invariant 7-ch features
def signed_curvature(pts):
    d = np.diff(pts, axis=0); ang = np.arctan2(d[:, 1], d[:, 0])
    dang = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    seg = np.linalg.norm(d, axis=1)
    denom = 0.5 * (seg[:-1] + seg[1:]) + 1e-8
    k = dang / denom
    return np.pad(k, (1, 1), mode='constant')


def extract_invariant_7ch(pts_raw):
    pts = np.array(pts_raw, dtype=np.float64).reshape(-1, 2); n = len(pts)
    d = np.diff(pts, axis=0); seg = np.linalg.norm(d, axis=1)
    seg_full = np.pad(seg, (0, 1), mode='edge')
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    abs_dang_full = np.pad(np.abs(dang), (1, 1), mode='constant')
    k = signed_curvature(pts)
    dk = np.pad(np.diff(k), (0, 1), mode='constant')
    ddk = np.pad(np.diff(dk), (0, 1), mode='constant')
    s_cum = np.cumsum(seg_full); s_norm = s_cum / (s_cum[-1] + 1e-8)
    w = 11; lstd = np.zeros(n); hw = w // 2
    for i in range(n):
        a, b = max(0, i - hw), min(n, i + hw + 1); lstd[i] = np.std(k[a:b])
    return np.column_stack([seg_full, abs_dang_full, k, dk, ddk, s_norm, lstd]).astype(np.float32)


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
        return (md.get('test_info') or {}).get('test_outcome')
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


def rotate_pts(pts, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (np.array(pts, dtype=np.float64) @ R.T).tolist()


def prepare(data, rot_deg=0.0):
    feats = []
    for tc in data:
        pts = get_pts(tc)
        if rot_deg != 0.0:
            pts = rotate_pts(pts, rot_deg)
        feats.append(resample(extract_invariant_7ch(pts)))
    X = np.array(feats, dtype=np.float32)
    y = np.array([1 if is_fail(tc) else 0 for tc in data], dtype=np.int64)
    return X, y


# ------------------------------------------------------- SE2 model (param)
class InvariantBlock(nn.Module):
    """Equivariant attention block, fully parameterised for ablation.

    When `rel_bias=False` the relative-arclength attention bias is removed
    entirely (vanilla MHA). Rotation invariance is unaffected -- it comes
    from the 7-ch invariant features -- so this axis isolates *only* the
    bias's contribution to ranking quality.
    """
    def __init__(self, d_model=192, nhead=8, ff=512, dropout=0.1,
                 n_rff=32, rff_scale=2.0, rel_hid=64, rel_bias=True):
        super().__init__()
        self.use_bias = rel_bias
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        if self.use_bias:
            self.rff = nn.Parameter(torch.randn(1, n_rff) * rff_scale, requires_grad=False)
            self.rel_bias = nn.Sequential(nn.Linear(n_rff, rel_hid), nn.GELU(),
                                          nn.Linear(rel_hid, nhead))

    def _rel_bias(self, s_norm):
        B, L = s_norm.shape
        ds = (s_norm.unsqueeze(2) - s_norm.unsqueeze(1)).unsqueeze(-1)
        feat = torch.sin(ds * self.rff)
        bias = self.rel_bias(feat)
        return bias.permute(0, 3, 1, 2)

    def forward(self, x, s_norm):
        B, Lp1, D = x.shape
        z = self.n1(x)
        if self.use_bias:
            s_full = torch.cat([torch.zeros(B, 1, device=x.device), s_norm], dim=1)
            bias = self._rel_bias(s_full)
            nhead = bias.size(1)
            attn_mask = bias.reshape(B * nhead, Lp1, Lp1)
        else:
            attn_mask = None
        a, _ = self.attn(z, z, z, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class SE2RoadNet(nn.Module):
    def __init__(self, in_ch=7, d_model=192, depth=6, nhead=8, ff=512, dropout=0.1,
                 n_rff=32, rff_scale=2.0, rel_hid=64, rel_bias=True, cls_hidden=64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_ch, d_model),
                                  nn.LayerNorm(d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            InvariantBlock(d_model, nhead, ff, dropout, n_rff, rff_scale, rel_hid, rel_bias)
            for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, cls_hidden), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(cls_hidden, 1))

    def forward(self, x):                    # x: (B, C, L)
        x = x.permute(0, 2, 1)               # (B, L, C)
        s_norm = x[..., 5]                   # 6th channel = s/L (invariant param)
        h = self.proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        for b in self.blocks:
            h = b(h, s_norm)
        return self.head(h[:, 0]).squeeze(-1)


def build_model(cfg):
    return SE2RoadNet(in_ch=7, **cfg)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, pos_weight=1.0):
        super().__init__(); self.g = gamma; self.pw = pos_weight
    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        w = torch.where(y == 1, self.pw, 1.0); bce = bce * w
        pt = torch.where(y == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return ((1 - pt).pow(self.g) * bce).mean()


class SWAModel:
    def __init__(self, m): self.model = copy.deepcopy(m); self.n = 0
    def update(self, m):
        self.n += 1; a = 1.0 / self.n
        for p, q in zip(self.model.parameters(), m.parameters()):
            p.data.mul_(1 - a).add_(q.data, alpha=a)
    def get_model(self): return self.model


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_chunked(model, X, chunk=128):
    """X: (B, C, L) tensor (CPU or DEVICE). Returns float numpy logits."""
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    out = []; model.eval()
    for i in range(0, X.size(0), chunk):
        xb = X[i:i + chunk].to(DEVICE, non_blocking=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            logit = model(xb).float()
        out.append(logit.cpu())
    return torch.cat(out, dim=0).numpy()


# --------------------------------------------------------------- train
def train_one(cfg, X_tr, y_tr, X_va, y_va, seed, epochs=EPOCHS):
    set_seed(seed)
    model = build_model(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_pos = y_tr.sum(); pw = float(len(y_tr) - n_pos) / max(1, n_pos)
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)
    Xv = torch.tensor(X_va, dtype=torch.float32).permute(0, 2, 1)   # eval chunked

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warm = 5
    sched = optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm
        else max(0.01, 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm)))))
    crit = FocalLoss(gamma=GAMMA, pos_weight=pw)
    scaler = GradScaler(enabled=(not USE_BF16))
    swa_at = min(SWA_START, epochs - 5)
    best_auc, best_state, swa = 0.0, None, None

    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                loss = crit(model(xb), yb)
            if USE_BF16:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
        sched.step()
        if ep >= swa_at:
            if swa is None: swa = SWAModel(model)
            else: swa.update(model)
        model.eval()
        v = 1.0 / (1.0 + np.exp(-predict_chunked(model, Xv)))
        try:
            auc = roc_auc_score(y_va, v)
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}

    model.load_state_dict(best_state)
    swa_m = swa.get_model().to(DEVICE) if swa else None
    swa_auc = 0.0
    if swa_m is not None:
        sv = 1.0 / (1.0 + np.exp(-predict_chunked(swa_m, Xv)))
        try:
            swa_auc = roc_auc_score(y_va, sv)
        except Exception:
            swa_auc = 0.5
    return model, swa_m, float(best_auc), float(swa_auc), int(n_params)


# ------------------------------------------------------------ evaluate
def compute_apfd(pids, td):
    n = len(pids)
    fp = [i + 1 for i, t in enumerate(pids) if is_fail(td[t])]
    m = len(fp)
    return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0


def apfd_from_feats(model, feats, ids, td):
    """Single-pass APFD over a fixed feature matrix (feats: (N,L,C))."""
    Xt = torch.tensor(feats, dtype=torch.float32).permute(0, 2, 1)
    p = 1.0 / (1.0 + np.exp(-predict_chunked(model, Xt)))
    pids = [u for _, u in sorted(zip(p, ids), key=lambda z: -z[0])]
    return compute_apfd(pids, td)


def multi_trial(model, comp, feats0):
    """30-trial APFD on precomputed rot=0 features (exp02 window)."""
    ids_all = [get_id(tc) for tc in comp]
    use_window = len(comp) >= COMP_LO + COMP_N
    apfds = []
    for t in range(N_TRIALS):
        rng = np.random.RandomState(42 + t)
        idx = rng.permutation(len(comp))
        idx = idx[COMP_LO:COMP_LO + COMP_N] if use_window \
            else idx[:min(max(50, int(0.3 * len(comp))), len(comp))]
        ed = [comp[i] for i in idx]
        td = {get_id(tc): tc for tc in ed}
        ids = [ids_all[i] for i in idx]
        apfds.append(apfd_from_feats(model, feats0[idx], ids, td))
    return float(np.mean(apfds)), float(np.std(apfds))


# --------------------------------------------------- one config, N seeds
def eval_config(cfg, data, label):
    X_tr, y_tr = data['X_tr'], data['y_tr']
    X_te, y_te = data['X_te'], data['y_te']
    comp, feats0, feats_rot = data['comp'], data['comp_feats0'], data['comp_feats_rot']
    ids_all = [get_id(tc) for tc in comp]
    td_all = {get_id(tc): tc for tc in comp}

    per_seed = []
    t0 = time.time()
    for s in range(N_SEEDS):
        model, swa, auc, swa_auc, n_params = train_one(cfg, X_tr, y_tr, X_te, y_te, seed=42 + s)
        eval_model = swa if swa is not None else model
        apfd_m, apfd_s = multi_trial(eval_model, comp, feats0)
        rec = dict(seed=42 + s, params=n_params, auc=auc, swa_auc=swa_auc,
                   apfd=apfd_m, apfd_sigma=apfd_s)
        if s == 0:  # rotation-invariance check once per config (property is seed-agnostic)
            a0 = apfd_from_feats(eval_model, feats0, ids_all, td_all)
            a90 = apfd_from_feats(eval_model, feats_rot, ids_all, td_all)
            rec['rot_delta'] = abs(a90 - a0)
        per_seed.append(rec)
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
        rot_delta=float(per_seed[0].get('rot_delta', float('nan'))),
        wall_sec=float(wall), n_seeds=N_SEEDS, per_seed=per_seed,
    )
    print(f"    {label:18s} | params={out['params']:>9,} | AUC={out['auc_mean']:.4f} "
          f"| APFD={out['apfd_mean']:.4f}+/-{out['apfd_sigma_across_seeds']:.4f} "
          f"| rotD={out['rot_delta']:.4f} | {wall/60:.1f}m")
    return out


# ------------------------------------------------------------ tables
def _is_baseline_value(param, val):
    return BASELINE_CFG.get(param) == val


def fmt_axis_table(axis_name, param, results, baseline_apfd):
    lines = [f"### {axis_name}", "",
             "| value | params    | AUC    | APFD-comp (30-trial)   | rot-Delta | Delta vs base | wall |",
             "|-------|-----------|--------|------------------------|-----------|---------------|------|"]
    best = max(results, key=lambda r: r['apfd_mean'])
    for r in results:
        val = r['cfg'][param]
        star = ' *' if r is best else '  '
        base_tag = ' (base)' if _is_baseline_value(param, val) else ''
        delta = r['apfd_mean'] - baseline_apfd
        lines.append(
            f"| {str(val):>4}{star}| {r['params']:>9,} | {r['auc_mean']:.4f} "
            f"| {r['apfd_mean']:.4f} +/- {r['apfd_sigma_across_seeds']:.4f} "
            f"| {r['rot_delta']:.4f}   | {delta:+.4f}{base_tag:>7} | {r['wall_sec']/60:.1f}m |")
    lines += ["",
              f"-> best on this axis: {param}={best['cfg'][param]} "
              f"(APFD={best['apfd_mean']:.4f} +/- {best['apfd_sigma_across_seeds']:.4f})", ""]
    return "\n".join(lines)


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

    print("Extracting INVARIANT features (7-ch) ...")
    X_tr, y_tr = prepare(train_data)
    X_te, y_te = prepare(test_data)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1)); stds[stds < 1e-8] = 1.0
    X_tr = (X_tr - means) / stds; X_te = (X_te - means) / stds

    # Precompute Competition features ONCE (rot=0 and rot=ROT_PROBE_DEG) so
    # every config/seed/trial just does forward passes -- the expensive part
    # of SE2 is the model, not feature extraction.
    print(f"Precomputing Competition features (rot=0 and rot={ROT_PROBE_DEG:.0f}) ...")
    Xc0, _ = prepare(comp_data, rot_deg=0.0)
    Xcr, _ = prepare(comp_data, rot_deg=ROT_PROBE_DEG)
    comp_feats0 = (Xc0 - means) / stds
    comp_feats_rot = (Xcr - means) / stds

    data = dict(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                comp=comp_data, comp_feats0=comp_feats0, comp_feats_rot=comp_feats_rot)

    # ---- cost estimate ----
    n_cfg = 1 + sum(
        sum(1 for v in AXES[a][1] if not _is_baseline_value(AXES[a][0], v))
        for a in RUN_AXES)
    print(f"\n[plan] {n_cfg} unique configs x {N_SEEDS} seeds = {n_cfg * N_SEEDS} trainings.")
    print(f"[plan] SE2 ~ 24 min/train on Exp-02 hardware -> ~{n_cfg * N_SEEDS * 24 / 60:.1f} h "
          f"(edit RUN_AXES / N_SEEDS / FAST to shrink).")

    print(f"\n{'=' * 70}\nBASELINE (shipped SE2RoadNet) -- pivot for every axis\n{'=' * 70}")
    print(f"  {BASELINE_CFG}")
    base = eval_config(BASELINE_CFG, data, 'baseline')
    baseline_apfd = base['apfd_mean']

    results = {
        'protocol': ('Focal(gamma=1.5)+SWA, 80ep, batch=384, lr=5e-4, bf16; '
                     'multi-trial APFD on Competition (287-of-956, 30 trials); '
                     f'{N_SEEDS} seeds/config; rot-Delta = |APFD(90deg)-APFD(0)|'),
        'baseline_cfg': BASELINE_CFG, 'baseline': base, 'axes': {},
    }

    for axis_name in RUN_AXES:
        param, values = AXES[axis_name]
        print(f"\n{'=' * 70}\nAXIS: {axis_name}\n{'=' * 70}")
        axis_res = []
        for val in values:
            cfg = dict(BASELINE_CFG); cfg[param] = val
            dm = cfg['d_model']; nh = cfg['nhead']
            if dm % nh != 0:
                print(f"    [skip] {param}={val}: d_model {dm} not divisible by nhead {nh}")
                continue
            if _is_baseline_value(param, val):
                row = dict(base); row['label'] = f'{param}={val}'
                axis_res.append(row)
                print(f"    {param}={val} (baseline, reusing) APFD={row['apfd_mean']:.4f}")
                continue
            axis_res.append(eval_config(cfg, data, f'{param}={val}'))
        results['axes'][axis_name] = dict(param=param, rows=axis_res)

        out_path = os.path.join(OUTPUT_DIR, 'se2_arch_ablation_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [save] {out_path}")

    # -------------------- ASCII report --------------------
    print(f"\n\n{'#' * 70}")
    print("# SE2RoadNet ARCHITECTURE ABLATION -- paste-ready tables (pure ASCII)")
    print(f"{'#' * 70}\n")
    print(f"Baseline (shipped): {BASELINE_CFG}")
    print(f"Baseline APFD-comp = {baseline_apfd:.4f} +/- "
          f"{base['apfd_sigma_across_seeds']:.4f}  ({base['params']:,} params, "
          f"rot-Delta={base['rot_delta']:.4f})\n")
    report = [f"Baseline: {BASELINE_CFG}",
              f"Baseline APFD = {baseline_apfd:.4f} +/- {base['apfd_sigma_across_seeds']:.4f} "
              f"(rot-Delta={base['rot_delta']:.4f})", ""]
    for axis_name, blob in results['axes'].items():
        tbl = fmt_axis_table(axis_name, blob['param'], blob['rows'], baseline_apfd)
        print(tbl); report.append(tbl)

    report_path = os.path.join(OUTPUT_DIR, 'se2_arch_ablation_report.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    print(f"[save] {report_path}")
    print(f"\nTOTAL TIME: {(time.time() - t_start) / 60:.1f} min")


if __name__ == '__main__':
    main()
