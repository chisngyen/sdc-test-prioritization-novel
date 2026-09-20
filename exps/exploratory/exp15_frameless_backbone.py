"""
EXP 15 -- Frameless-input backbone ablation (RoadFury vs SE2RoadNet)
===================================================================
Controlled A/B on ONE shared input. We strip the three frame-dependent
channels from the RoadFury 10-ch pipeline and feed the resulting 7-ch
frame-invariant sequence to TWO backbones, changing nothing else:

  arm A (RoadFury) : RoadTransformer   -- absolute learnable positional
                     embedding (the ICST-2026 RoadFury backbone), in_ch=7
  arm B (SE2RoadNet): SE2RoadNet       -- relative-arclength attention bias
                     (the SE(2)-equivariant backbone from Exp 02), in_ch=7

Both arms share: the same 7-ch features, the same train/test/competition
split, the same seed, and the same recipe (Focal gamma=2.5 + SWA, 75 ep,
batch=256). The ONLY difference is the backbone. This isolates the
architecture's contribution once the input is already frame-invariant.

The three dropped channels (RoadFury 10-ch indices):
    5  heading_sin   -- sin(theta), leaks absolute orientation (frame)
    6  heading_cos   -- cos(theta), leaks absolute orientation (frame)
    7  rel_position  -- index-linspace position, redundant with cum_dist_norm

Kept 7-ch layout (drop {5,6,7} from the 10-ch column stack):
    0 seg_length | 1 abs_angle_change | 2 curvature(Menger) |
    3 curv_jerk  | 4 cum_dist_norm(=s/L) | 5 local_curv_std | 6 curv_accel
=> s/L now sits at index 4, so SE2RoadNet reads its arclength channel
   from S_INDEX=4 (not 5 as in the exp02 layout).

What the numbers mean
---------------------
- APFD-comp (multi 30) is the leaderboard metric in ../tracker.md. The
  10-ch RoadFury baseline is 0.8066 +/- 0.0124; the exp02 SE2RoadNet on
  its own 7-ch invariant pipeline is 0.8048 +/- 0.0118 (note: exp02 used
  gamma=1.5/batch=384/80ep, so its absolute number is not expected to match
  arm B here bit-for-bit -- what matters is A vs B under identical config).
- rot-Delta: both arms should be ~0 because the shared 7-ch input is
  already SO(2)-invariant. That is the point: dropping heading buys the
  rotation invariance at the INPUT; the backbone choice does not.

Self-contained: paste on Kaggle. Auto-discovers sensodat_train.json,
sensodat_test.json and sdc-test-data.json across common mount layouts.
"""

import json, os, time, math, copy, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

# ---------------- Paths (robust discovery) ----------------
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

SEARCH_ROOTS = [
    '/kaggle/input',
    os.path.normpath(os.path.join(HERE, '..', 'data')),
    os.path.normpath(os.path.join(HERE, '..', '..', 'data')),
    os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'data')),
    os.getcwd(),
]

def find_data_file(name):
    """Return the first path whose basename == name under any search root."""
    seen = set()
    for root in SEARCH_ROOTS:
        if not root or not os.path.isdir(root) or root in seen:
            continue
        seen.add(root)
        for dirpath, _, filenames in os.walk(root):
            if name in filenames:
                return os.path.join(dirpath, name)
    return None

TRAIN_PATH = find_data_file('sensodat_train.json')
TEST_PATH  = find_data_file('sensodat_test.json')
COMP_PATH  = find_data_file('sdc-test-data.json')
OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else \
    os.path.normpath(os.path.join(HERE, '..', 'models'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
torch.set_float32_matmul_precision('high')
print(f"Device: {DEVICE} | bf16: {USE_BF16}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# ---------------- Config (identical for both arms) ----------------
SEQ_LEN   = 197
GAMMA     = 2.5     # SensoDat winning focal gamma
EPOCHS    = 75
BATCH     = 256     # reduce to 128 if SE2RoadNet OOMs (rel-bias is O(B*L*L*32))
LR        = 5e-4
SWA_START = 50
N_TRIALS  = 30
SEED      = 42
ROTATIONS = [0.0, 30.0, 60.0, 90.0, 180.0, -45.0]

def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ==================================================================
# Feature extraction: RoadFury 10-ch (Menger) -> drop {5,6,7} -> 7-ch
# ==================================================================
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
            R = a * b * c / (4 * math.sqrt(at)); curv[i] = 1.0 / R if R > 0 else 0.0
    return curv

DROP_IDX = (5, 6, 7)                 # heading_sin, heading_cos, rel_position
KEEP_IDX = [0, 1, 2, 3, 4, 8, 9]     # 7 frame-invariant channels
S_INDEX  = KEEP_IDX.index(4)         # cum_dist_norm (s/L) lands at index 4

def extract_seq_10ch(pts_raw):
    pts = np.array(pts_raw, dtype=np.float64).reshape(-1, 2); n = len(pts)
    diffs = np.diff(pts, axis=0); seg_lens = np.linalg.norm(diffs, axis=1)
    seg_full = np.pad(seg_lens, (0, 1), mode='edge')
    angles = np.arctan2(diffs[:, 1], diffs[:, 0]); ac = np.diff(angles)
    ac = (ac + np.pi) % (2 * np.pi) - np.pi
    abs_ac_full = np.pad(np.abs(ac), (1, 1), mode='constant')
    curv = np.abs(_compute_curvature(pts)); curv_full = np.pad(curv, (1, 1), mode='constant')
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

def extract_frameless_7ch(pts_raw):
    """RoadFury 10-ch with the 3 frame-dependent columns removed."""
    return extract_seq_10ch(pts_raw)[:, KEEP_IDX]

def resample(seq, target_len=SEQ_LEN):
    n, c = seq.shape
    if n == target_len:
        return seq
    x_old = np.linspace(0, 1, n); x_new = np.linspace(0, 1, target_len)
    out = np.empty((target_len, c), dtype=np.float32)
    for ch in range(c):
        out[:, ch] = np.interp(x_new, x_old, seq[:, ch])
    return out

# ==================================================================
# Backbone A -- RoadFury (absolute positional embedding), in_ch=7
# ==================================================================
class RoadTransformer(nn.Module):
    def __init__(self, in_channels=7, seq_len=SEQ_LEN, d_model=128,
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
    def forward(self, x):                       # x: (B, C, L)
        x = x.permute(0, 2, 1); B, L, C = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :L + 1, :]
        x = self.transformer(x)
        return self.classifier(x[:, 0, :]).squeeze(-1)

# ==================================================================
# Backbone B -- SE2RoadNet (relative-arclength bias), in_ch=7
# ==================================================================
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

    def _rel_bias(self, s_full):                # s_full (B, L+1)
        ds = (s_full.unsqueeze(2) - s_full.unsqueeze(1)).unsqueeze(-1)   # (B,L+1,L+1,1)
        feat = torch.sin(ds * self.rff)                                  # (B,L+1,L+1,32)
        bias = self.rel_bias(feat)                                       # (B,L+1,L+1,nhead)
        return bias.permute(0, 3, 1, 2)                                  # (B,nhead,L+1,L+1)

    def forward(self, x, s_norm):
        B, Lp1, D = x.shape
        s_full = torch.cat([torch.zeros(B, 1, device=x.device), s_norm], dim=1)
        bias = self._rel_bias(s_full)
        nhead = bias.size(1)
        attn_mask = bias.reshape(B * nhead, Lp1, Lp1)
        z = self.n1(x)
        a, _ = self.attn(z, z, z, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x

class SE2RoadNet(nn.Module):
    def __init__(self, in_ch=7, d_model=192, depth=6, nhead=8, ff=512,
                 dropout=0.1, s_index=S_INDEX):
        super().__init__()
        self.s_index = s_index
        self.proj = nn.Sequential(nn.Linear(in_ch, d_model),
                                  nn.LayerNorm(d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([InvariantBlock(d_model, nhead, ff, dropout)
                                     for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, 64), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):                       # x: (B, C, L)
        x = x.permute(0, 2, 1)                  # (B, L, C)
        s_norm = x[..., self.s_index]           # arclength channel (s/L)
        h = self.proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        for b in self.blocks:
            h = b(h, s_norm)
        return self.head(h[:, 0]).squeeze(-1)

# ---------------- Loss / SWA ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=1.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weight = torch.where(targets == 1, self.pos_weight, 1.0); bce = bce * weight
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class SWAModel:
    def __init__(self, model):
        self.model = copy.deepcopy(model); self.n = 0
    def update(self, new_model):
        self.n += 1; a = 1.0 / self.n
        for p, q in zip(self.model.parameters(), new_model.parameters()):
            p.data.mul_(1 - a).add_(q.data, alpha=a)
    def get_model(self):
        return self.model

# ---------------- Chunked inference (SE2 rel-bias is memory-heavy) ----------------
@torch.no_grad()
def predict_chunked(model, X, chunk=128):
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    model.eval(); out = []
    for i in range(0, X.size(0), chunk):
        xb = X[i:i + chunk].to(DEVICE, non_blocking=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            out.append(model(xb).float().cpu())
    return torch.cat(out, dim=0).numpy()

# ---------------- Shared training loop ----------------
def train_backbone(model, X_tr, y_tr, X_va, y_va, name=''):
    set_seed(SEED)
    model = model.to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*64}\nTrain {name} | params={nparams:,} | gamma={GAMMA} | SWA@{SWA_START}\n{'='*64}")
    n_pos = y_tr.sum(); pw = float(len(y_tr) - n_pos) / max(1, n_pos)
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)
    Xv = torch.tensor(X_va, dtype=torch.float32).permute(0, 2, 1)   # kept on CPU; chunked to GPU

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warm = 5
    sched = optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm
        else max(0.01, 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, EPOCHS - warm)))))
    crit = FocalLoss(alpha=1.0, gamma=GAMMA, pos_weight=pw)
    scaler = GradScaler(enabled=(not USE_BF16) and torch.cuda.is_available())
    best_auc, best_state, swa = 0.0, None, None

    for ep in range(EPOCHS):
        model.train(); tot = 0.0; nb = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                loss = crit(model(xb), yb)
            if USE_BF16 or not torch.cuda.is_available():
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        if ep >= SWA_START:
            if swa is None:
                swa = SWAModel(model); print(f"  [SWA] start @ epoch {ep+1}")
            else:
                swa.update(model)
        v = 1.0 / (1.0 + np.exp(-predict_chunked(model, Xv, chunk=256)))
        try:
            auc = roc_auc_score(y_va, v)
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
            flag = ' *'
        else:
            flag = ''
        if (ep + 1) % 10 == 0 or flag:
            print(f"  Ep {ep+1:3d} | loss={tot/nb:.4f} | AUC={auc:.4f} | best={best_auc:.4f}{flag}")

    model.load_state_dict(best_state)
    eval_model = swa.get_model().to(DEVICE) if swa else model
    return eval_model, best_auc, nparams

# ---------------- APFD eval ----------------
def compute_apfd(pids, td):
    n = len(pids)
    fp = [i + 1 for i, t in enumerate(pids)
          if td[t]['meta_data']['test_info']['test_outcome'] == 'FAIL']
    m = len(fp)
    return 1 - sum(fp) / (n * m) + 1 / (2 * n) if n and m else 1.0

def _rot_matrix(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def _feats(data, means, stds, rot_deg=0.0):
    out = []
    R = _rot_matrix(rot_deg) if rot_deg != 0.0 else None
    for tc in data:
        pts = np.array(get_pts(tc), dtype=np.float64)
        if R is not None:
            pts = pts @ R.T
        f = resample(extract_frameless_7ch(pts.tolist()))
        out.append((f - means) / stds)
    return np.array(out, dtype=np.float32)

def eval_apfd(data, model, means, stds, name='', rot_deg=0.0):
    td = {get_id(tc): tc for tc in data}; ids = [get_id(tc) for tc in data]
    X = torch.tensor(_feats(data, means, stds, rot_deg), dtype=torch.float32).permute(0, 2, 1)
    p = 1.0 / (1.0 + np.exp(-predict_chunked(model, X, chunk=256)))
    pids = [t for _, t in sorted(zip(p, ids), key=lambda z: -z[0])]
    a = compute_apfd(pids, td)
    tag = '' if rot_deg == 0.0 else f' [rot={rot_deg:+.0f}]'
    print(f"  {name:44s} APFD={a:.4f}{tag}")
    return a

def multi_trial(data, model, means, stds, name='', n_trials=N_TRIALS):
    # Precompute features once; sub-sample 287 tests from a permuted 334 offset
    # (identical protocol to exp00/exp02 so numbers land in ../tracker.md).
    feats = _feats(data, means, stds, 0.0)
    ids_all = [get_id(tc) for tc in data]
    td_all = {get_id(tc): tc for tc in data}
    apfds = []
    for t in range(n_trials):
        rng = np.random.RandomState(42 + t); idx = rng.permutation(len(data))
        sub = idx[334:334 + 287]
        X = torch.tensor(feats[sub], dtype=torch.float32).permute(0, 2, 1)
        p = 1.0 / (1.0 + np.exp(-predict_chunked(model, X, chunk=256)))
        ids = [ids_all[i] for i in sub]
        pids = [u for _, u in sorted(zip(p, ids), key=lambda z: -z[0])]
        apfds.append(compute_apfd(pids, td_all))
    m_, s_ = float(np.mean(apfds)), float(np.std(apfds))
    print(f"  {name:44s} APFD={m_:.4f}+/-{s_:.4f} ({n_trials} trials)")
    return m_, s_

def rotation_probe(data, model, means, stds, name=''):
    print(f"  --- rotation probe: {name} ---")
    vals = [eval_apfd(data, model, means, stds, f'{name} rot', rot_deg=r) for r in ROTATIONS]
    delta = float(max(vals) - min(vals))
    print(f"  {name:44s} rot-Delta={delta:.4f}")
    return {'rotations': ROTATIONS, 'apfd': vals, 'delta': delta}

# ---------------- Data ----------------
def load_json(path):
    print(f"Loading {path}..."); t0 = time.time()
    with open(path) as f:
        data = json.load(f)
    print(f"  {len(data)} tests in {time.time()-t0:.1f}s")
    return data

def get_pts(tc):
    return [[p['x'], p['y']] for p in tc['road_points']]
def is_fail(tc):
    return tc['meta_data']['test_info']['test_outcome'] == 'FAIL'
def get_id(tc):
    return tc['_id']['$oid']

def prepare(data):
    X, y = [], []
    for i, tc in enumerate(data):
        X.append(resample(extract_frameless_7ch(get_pts(tc))))
        y.append(1 if is_fail(tc) else 0)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(data)}...")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# ---------------- Main ----------------
def main():
    t0 = time.time()
    print("\n" + "=" * 72)
    print("EXP 15 -- Frameless-input backbone ablation (RoadFury vs SE2RoadNet)")
    print("Shared 7-ch frame-invariant input; only the backbone changes.")
    print("=" * 72)
    for tag, p in [('train', TRAIN_PATH), ('test', TEST_PATH), ('comp', COMP_PATH)]:
        print(f"  {tag:5s}: {p}")
    if not (TRAIN_PATH and TEST_PATH and COMP_PATH):
        raise SystemExit("[FATAL] could not locate sensodat_train/test + sdc-test-data json.")

    train_data = load_json(TRAIN_PATH)
    test_data = load_json(TEST_PATH)
    comp_data = load_json(COMP_PATH)
    nft = sum(is_fail(tc) for tc in train_data)
    print(f"Train {len(train_data)} ({nft} FAIL={100*nft/len(train_data):.1f}%) | "
          f"Test {len(test_data)} | Comp {len(comp_data)}")

    print("\nExtracting 7-ch frameless features (drop {heading_sin, heading_cos, rel_position})...")
    X_tr, y_tr = prepare(train_data)
    X_te, y_te = prepare(test_data)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1)); stds[stds < 1e-8] = 1.0
    X_tr = (X_tr - means) / stds; X_te = (X_te - means) / stds
    print(f"  X_tr {X_tr.shape} | X_te {X_te.shape} | s/L channel index = {S_INDEX}")

    arms = {
        'RoadFury (7ch)':   lambda: RoadTransformer(in_channels=7, d_model=128, num_layers=4),
        'SE2RoadNet (7ch)': lambda: SE2RoadNet(in_ch=7, d_model=192, depth=6, s_index=S_INDEX),
    }

    results = {
        'setup': {'input': '7ch frameless (RoadFury 10ch minus {heading_sin,heading_cos,rel_position})',
                  'dropped_idx': list(DROP_IDX), 's_index': S_INDEX,
                  'recipe': f'Focal gamma={GAMMA} + SWA, {EPOCHS}ep, batch={BATCH}, seed={SEED}'},
        'arms': {}}

    for name, ctor in arms.items():
        ta = time.time()
        model, auc, nparams = train_backbone(ctor(), X_tr, y_tr, X_te, y_te, name=name)
        print(f"\n--- Eval {name} ---")
        apfd_sens = eval_apfd(test_data, model, means, stds, f'{name} SensoDat-test')
        apfd_m, apfd_s = multi_trial(comp_data, model, means, stds, f'{name} comp multi')
        probe = rotation_probe(comp_data, model, means, stds, name=name)
        results['arms'][name] = {
            'params': int(nparams), 'auc': float(auc),
            'apfd_sensodat': float(apfd_sens),
            'apfd_comp_mean': apfd_m, 'apfd_comp_std': apfd_s,
            'rot_delta': probe['delta'], 'rotation_probe': probe,
            'train_min': (time.time() - ta) / 60.0,
        }
        out = os.path.join(OUTPUT_DIR, 'exp15_frameless_results.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  [save] {out}")

    # ---- Comparison table ----
    print(f"\n{'='*84}\nFRAMELESS-INPUT BACKBONE ABLATION -- comparison\n{'='*84}")
    print(f"{'Backbone':<20} {'Params':>10} {'AUC':>8} {'APFD-Sens':>10} "
          f"{'APFD-comp(30)':>18} {'rotD':>7} {'min':>6}")
    for name in arms:
        r = results['arms'][name]
        print(f"{name:<20} {r['params']:>10,} {r['auc']:>8.4f} {r['apfd_sensodat']:>10.4f} "
              f"{r['apfd_comp_mean']:>10.4f}+/-{r['apfd_comp_std']:<5.4f} "
              f"{r['rot_delta']:>7.4f} {r['train_min']:>6.1f}")
    print("\nReference (../tracker.md): 10ch RoadFury 0.8066+/-0.0124 | "
          "exp02 SE2 (own 7ch pipeline, gamma=1.5) 0.8048+/-0.0118")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
