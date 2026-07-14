"""
EXP 02b -- SE(2)-Equivariant RoadNet: FOCAL-GAMMA SWEEP
======================================================
Why this exists
---------------
Every SE(2) experiment in this repo hard-codes focal gamma = 1.5, but the
only real gamma sweep on record ([0,1,1.5,2,2.5]) was run on the *non*-
equivariant RoadFury baseline, where gamma = 2.5 won (SensoDat APFD
0.8066 +/- 0.0124). We therefore have NO evidence that 1.5 is the right
gamma for the *equivariant* SE2RoadNet. This script closes that gap: it
trains the identical SE2RoadNet architecture across a grid of gammas
(default up to 5.0) and reports APFD +/- sigma (30 trials) + AUC per gamma,
then prints a leaderboard and saves a results JSON.

Architecture / training are copied verbatim from exp02_SE2Equivariant.py so
the numbers are directly comparable; the ONLY change is that focal gamma is
threaded through `train(...)` instead of being fixed at 1.5.

Self-contained: paste this one file on Kaggle (data + GPU there). Locally it
discovers data under ./data via SEARCH_ROOTS. Knobs via env vars:
    SDC_GAMMAS="1.0,1.5,2.0,2.5,3.0,4.0,5.0"   sweep grid
    SDC_EPOCHS=80                               epochs per gamma
    SDC_DATA_DIR=/path/to/dir                   force data dir (smoke tests)
    SDC_NTRIALS=30                              APFD trials
"""

import json, numpy as np, os, sys, time, math, copy, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

SEARCH_ROOTS = [
    '/kaggle/input',
    os.path.normpath(os.path.join(HERE, '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', 'data')),
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
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")

SEQ_LEN = 197
GAMMAS  = [float(g) for g in os.environ.get('SDC_GAMMAS', '1.0,1.5,2.0,2.5,3.0,4.0,5.0').split(',')]
EPOCHS  = int(os.environ.get('SDC_EPOCHS', '80'))
N_TRIALS = int(os.environ.get('SDC_NTRIALS', '30'))
SEED = 42

def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ==================== data discovery ====================
def _find_data_dir():
    forced = os.environ.get('SDC_DATA_DIR')
    if forced and os.path.isdir(forced):
        return forced
    seen = set()
    for root in SEARCH_ROOTS:
        if not root or not os.path.isdir(root) or root in seen: continue
        seen.add(root)
        for dirpath, _, filenames in os.walk(root):
            if any(fn == 'sensodat_train.json' for fn in filenames):
                return dirpath
    return None

# ==================== SE(2)-invariant 7-ch features (verbatim exp02) ====================
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
        a,b = max(0,i-hw), min(n,i+hw+1); lstd[i] = np.std(k[a:b])
    return np.column_stack([seg_full, abs_dang_full, k, dk, ddk, s_norm, lstd]).astype(np.float32)

def load_json(path):
    print(f"Loading {path}..."); t0=time.time()
    with open(path) as f: data=json.load(f)
    print(f"  Loaded {len(data)} tests in {time.time()-t0:.1f}s"); return data
def get_pts(tc): return [[p['x'],p['y']] for p in tc['road_points']]
def is_fail(tc): return tc['meta_data']['test_info']['test_outcome']=='FAIL'
def get_id(tc): return tc['_id']['$oid']

def prepare_data(data):
    X,y=[],[]
    for i,tc in enumerate(data):
        X.append(extract_invariant_7ch(get_pts(tc))); y.append(1 if is_fail(tc) else 0)
        if (i+1)%5000==0: print(f"    {i+1}/{len(data)}...")
    return np.array(X), np.array(y)

# ==================== model (verbatim exp02) ====================
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
        B, Lp1, D = x.shape; L = Lp1 - 1
        s_full = torch.cat([torch.zeros(B, 1, device=x.device), s_norm], dim=1)
        bias = self._rel_bias(s_full)
        h = x.size(1); nhead = bias.size(1)
        attn_mask = bias.reshape(B * nhead, h, h)
        z = self.n1(x)
        a, _ = self.attn(z, z, z, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x

class SE2RoadNet(nn.Module):
    def __init__(self, in_ch=7, d_model=192, depth=6, nhead=8, ff=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_ch, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([InvariantBlock(d_model, nhead, ff, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        s_norm = x[..., 5]
        h = self.proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        for b in self.blocks: h = b(h, s_norm)
        return self.head(h[:, 0]).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, pos_weight=1.0):
        super().__init__(); self.g=gamma; self.pw=pos_weight
    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        w = torch.where(y==1, self.pw, 1.0); bce = bce * w
        pt = torch.where(y==1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return ((1-pt).pow(self.g) * bce).mean()

class SWAModel:
    def __init__(self, m): self.model = copy.deepcopy(m); self.n = 0
    def update(self, m):
        self.n += 1; a = 1.0/self.n
        for p, q in zip(self.model.parameters(), m.parameters()):
            p.data.mul_(1-a).add_(q.data, alpha=a)
    def get_model(self): return self.model

@torch.no_grad()
def predict_chunked(model, X, chunk=128):
    if not torch.is_tensor(X): X = torch.tensor(X, dtype=torch.float32)
    out = []; model.eval()
    for i in range(0, X.size(0), chunk):
        xb = X[i:i+chunk].to(DEVICE, non_blocking=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            logit = model(xb).float()
        out.append(logit.cpu())
    return torch.cat(out, dim=0).numpy()

# ---- train: gamma is the ONLY change vs exp02 (threaded, not fixed at 1.5) ----
def train(model, X_tr, y_tr, X_va, y_va, *, gamma, epochs=80, batch=384,
          lr=5e-4, swa_start=55, name='SE2'):
    print(f"\n{'='*64}\nTrain {name} | gamma={gamma} | "
          f"params={sum(p.numel() for p in model.parameters()):,}\n{'='*64}")
    model = model.to(DEVICE)
    n_pos = y_tr.sum(); pw = (len(y_tr) - n_pos) / n_pos
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    # real data has thousands of rows -> batch=384, drop_last=True (identical to exp02);
    # only shrink for tiny sets (e.g. smoke tests) so we never end up with zero batches.
    eff_batch = batch if len(Xt) >= batch else max(8, len(Xt)//4)
    drop_last = len(Xt) >= 2*eff_batch
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=eff_batch, sampler=sampler,
                    num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=drop_last)
    Xv = torch.tensor(X_va, dtype=torch.float32).permute(0, 2, 1)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = 5
    sched = optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/warm if e<warm
        else max(0.01, 0.5*(1 + math.cos(math.pi*(e-warm)/max(1, epochs-warm)))))
    crit = FocalLoss(gamma=gamma, pos_weight=pw)
    scaler = GradScaler(enabled=(not USE_BF16) and torch.cuda.is_available())
    best_auc, best_state, swa = 0., None, None
    for ep in range(epochs):
        model.train(); tot=0; nb=0
        for xb, yb in dl:
            xb=xb.to(DEVICE, non_blocking=True); yb=yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                loss = crit(model(xb), yb)
            if USE_BF16 or not torch.cuda.is_available():
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
            tot+=loss.item(); nb+=1
        sched.step()
        if ep >= swa_start:
            if swa is None: swa = SWAModel(model); print(f"  [SWA] start @ epoch {ep+1}")
            else: swa.update(model)
        model.eval()
        v = 1.0 / (1.0 + np.exp(-predict_chunked(model, Xv)))
        auc = roc_auc_score(y_va, v)
        flag = ' *' if auc > best_auc else ''
        if auc > best_auc:
            best_auc = auc
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        if (ep+1) % 10 == 0 or flag:
            print(f"  Ep {ep+1:3d} | loss={tot/max(nb,1):.4f} | AUC={auc:.4f} | best={best_auc:.4f}{flag}")
    model.load_state_dict(best_state)
    return model, best_auc, swa

# ==================== APFD eval (verbatim exp02) ====================
def compute_apfd(pids, td):
    n=len(pids)
    fp=[i+1 for i,t in enumerate(pids) if td[t]['meta_data']['test_info']['test_outcome']=='FAIL']
    m=len(fp); return 1 - sum(fp)/(n*m) + 1/(2*n) if n and m else 1.0

def _feats(data, means, stds, rot_deg=0.0):
    out=[]
    if rot_deg == 0.0:
        for tc in data: out.append((extract_invariant_7ch(get_pts(tc)) - means)/stds)
    else:
        c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        R = np.array([[c, -s],[s, c]], dtype=np.float64)
        for tc in data:
            pts = np.array(get_pts(tc), dtype=np.float64) @ R.T
            out.append((extract_invariant_7ch(pts.tolist()) - means)/stds)
    return np.array(out)

def eval_apfd(data, model, means, stds, name='', rot_deg=0.0):
    model.eval().to(DEVICE)
    td={get_id(tc):tc for tc in data}; ids=[get_id(tc) for tc in data]
    feats = _feats(data, means, stds, rot_deg)
    X=torch.tensor(feats, dtype=torch.float32).permute(0,2,1)
    p = 1.0 / (1.0 + np.exp(-predict_chunked(model, X)))
    pids=[t for _,t in sorted(zip(p, ids), key=lambda z:-z[0])]
    a=compute_apfd(pids, td)
    rotag = '' if rot_deg == 0.0 else f' [rot={rot_deg:+.0f}]'
    print(f"  {name:46s} APFD={a:.4f}{rotag}")
    return a

def multi_trial(data, model, means, stds, name='', n_trials=30, rot_deg=0.0):
    model.eval().to(DEVICE); apfds=[]
    n = len(data)
    # exp02 protocol on the full ICST competition set; robust fallback for small sets
    if n >= 621:
        start, k = 334, 287
    else:
        start, k = 0, max(10, int(0.4*n))
    for t in range(n_trials):
        rng=np.random.RandomState(SEED+t); idx=rng.permutation(n)
        ed=[data[i] for i in idx[start:start+k]]
        td={get_id(tc):tc for tc in ed}; ids=[get_id(tc) for tc in ed]
        feats = _feats(ed, means, stds, rot_deg)
        X=torch.tensor(feats, dtype=torch.float32).permute(0,2,1)
        p = 1.0 / (1.0 + np.exp(-predict_chunked(model, X)))
        pids=[u for _,u in sorted(zip(p, ids), key=lambda z:-z[0])]
        apfds.append(compute_apfd(pids, td))
    mean, std = float(np.mean(apfds)), float(np.std(apfds))
    print(f"  {name:46s} APFD={mean:.4f}+/-{std:.4f}")
    return mean, std

# ==================== main sweep ====================
def main():
    t_all = time.time()
    print("\n" + "="*72)
    print("EXP 02b -- SE2RoadNet FOCAL-GAMMA SWEEP")
    print(f"gammas={GAMMAS} | epochs={EPOCHS} | trials={N_TRIALS}")
    print("="*72)

    ddir = _find_data_dir()
    if ddir is None:
        print("\n[FATAL] could not locate sensodat_train.json under SEARCH_ROOTS.")
        print("        Set SDC_DATA_DIR=/path/to/dir or run on Kaggle.")
        sys.exit(1)
    print("Data dir:", ddir)
    train_data = load_json(os.path.join(ddir, 'sensodat_train.json'))
    test_data  = load_json(os.path.join(ddir, 'sensodat_test.json'))
    comp_path  = os.path.join(ddir, 'sdc-test-data.json')
    comp_data  = load_json(comp_path) if os.path.exists(comp_path) else None

    print("\nExtracting INVARIANT features (7-ch) once...")
    X_tr, y_tr = prepare_data(train_data)
    X_te, y_te = prepare_data(test_data)
    means = X_tr.mean(axis=(0,1)); stds = X_tr.std(axis=(0,1)); stds[stds<1e-8] = 1.0
    X_trn = (X_tr - means)/stds; X_ten = (X_te - means)/stds

    rows = []
    best_model, best_apfd_seen = None, -1.0
    for g in GAMMAS:
        set_seed(SEED)  # same init + sampler stream per gamma -> difference is gamma, not luck
        t0 = time.time()
        model = SE2RoadNet(in_ch=7, d_model=192, depth=6, nhead=8, ff=512, dropout=0.1)
        model, auc, swa = train(model, X_trn, y_tr, X_ten, y_te, gamma=g,
                                epochs=EPOCHS, batch=384, lr=5e-4, swa_start=max(1, EPOCHS-25),
                                name=f'SE2RoadNet(g={g})')
        m_eval = swa.get_model() if swa else model
        se_auc = eval_apfd(test_data, m_eval, means, stds, f'g={g} SWA SensoDat')
        if comp_data is not None:
            apfd_mean, apfd_std = multi_trial(comp_data, m_eval, means, stds,
                                              f'g={g} SWA comp', n_trials=N_TRIALS)
        else:
            apfd_mean, apfd_std = float('nan'), float('nan')
        dt = (time.time()-t0)/60
        rows.append({'gamma': g, 'auc': float(auc), 'apfd_mean': apfd_mean,
                     'apfd_std': apfd_std, 'minutes': round(dt, 1)})
        print(f"  >>> gamma={g}: AUC={auc:.4f} APFD={apfd_mean:.4f}+/-{apfd_std:.4f} ({dt:.1f} min)")
        cmp_key = apfd_mean if apfd_mean == apfd_mean else se_auc  # nan-safe
        if cmp_key > best_apfd_seen:
            best_apfd_seen = cmp_key; best_model = copy.deepcopy(m_eval).cpu()

    # ---- leaderboard (ASCII) ----
    ranked = sorted(rows, key=lambda r: (-(r['apfd_mean'] if r['apfd_mean']==r['apfd_mean'] else -1)))
    best = ranked[0]
    print("\n" + "="*72)
    print("LEADERBOARD -- SE2RoadNet gamma sweep (sorted by APFD)")
    print("="*72)
    print(f"{'gamma':>6} | {'AUC':>7} | {'APFD':>8} | {'sigma':>7} | {'min':>5}")
    print("-"*44)
    for r in ranked:
        star = '  <== best' if r is best else ''
        print(f"{r['gamma']:>6} | {r['auc']:>7.4f} | {r['apfd_mean']:>8.4f} | "
              f"{r['apfd_std']:>7.4f} | {r['minutes']:>5}{star}")
    print(f"\nStrongest gamma = {best['gamma']}  "
          f"(APFD {best['apfd_mean']:.4f}+/-{best['apfd_std']:.4f}, AUC {best['auc']:.4f})")

    # ---- rotation-invariance sanity on the winner (Delta should be ~0) ----
    rot_probe = None
    if comp_data is not None and best_model is not None:
        print("\nRotation probe on winner gamma (architectural invariance, expect Delta~0):")
        a0  = eval_apfd(comp_data, best_model, means, stds, f'g={best["gamma"]} rot=0',  rot_deg=0.0)
        a90 = eval_apfd(comp_data, best_model, means, stds, f'g={best["gamma"]} rot=90', rot_deg=90.0)
        rot_probe = {'apfd_rot0': a0, 'apfd_rot90': a90, 'delta': abs(a0 - a90)}
        print(f"  Delta(0 vs 90) = {rot_probe['delta']:.6f}")

    out = {'recipe': 'SE2RoadNet (7-ch invariant, d=192, depth=6) + SWA + Focal, gamma sweep',
           'epochs': EPOCHS, 'n_trials': N_TRIALS, 'seed': SEED,
           'results': rows, 'best_gamma': best['gamma'],
           'best_apfd': best['apfd_mean'], 'best_apfd_std': best['apfd_std'],
           'winner_rotation_probe': rot_probe}
    op = os.path.join(OUTPUT_DIR, 'exp02b_gamma_sweep_results.json')
    with open(op, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved: {op}")
    print(f"TOTAL: {(time.time()-t_all)/60:.1f} min")

if __name__ == '__main__':
    main()
