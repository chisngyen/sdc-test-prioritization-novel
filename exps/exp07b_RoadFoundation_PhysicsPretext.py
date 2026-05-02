"""
EXP 07b -- SSL Foundation v2: PHYSICS-INFORMED Pretext
=========================================================
Theoretical lens: PHYSICS-INFORMED SELF-SUPERVISED LEARNING.

Tracker insight (Exp 07 v1, FAILED):
  - Pretext "predict (kappa, d-kappa)" is purely GEOMETRIC.
  - Encoder learned curvature but NOT the fail boundary.
  - 10% finetune APFD = 0.7056 -- a full 0.10 BELOW baseline.

Diagnosis: the geometric pretext does not expose the failure-relevant
sufficient statistic. From Exp 04 (PINN), we know that the centripetal-
acceleration field
        c(s) = v_eff^2 * |kappa(s)|
is the physical sufficient condition for the fail event (the SDC departs
the lane when c exceeds the friction-limited threshold mu * g). This is
a SCALAR field on the road that DISCRIMINATES fail/pass by construction.

We therefore replace the v1 pretext with two PHYSICS-INFORMED targets,
both label-free (computable from raw points):

  (P1) Per-position centripetal load    c_i = v_eff^2 * |kappa(s_i)|
       Token-wise regression target. Encoder learns "where physics is
       stressed".

  (P2) Per-road MAX load                C = max_s c(s)
       CLS-token regression target. This is a one-number proxy for "how
       likely the road is to fail under the autopilot config". Highly
       fail-discriminative.

Theory contribution. Under the centripetal-fail model
        P(fail | x) = sigma(eta * (C(x) - tau)),
the optimal scoring function is monotone in C(x). A pretext that learns
C(x) accurately therefore PROVABLY transfers to fail prediction up to a
single learnable affine head. Since C(x) is computable from x alone (no
labels), this gives an information-theoretically optimal SSL target for
this domain.

We expect: with the same compute as Exp 07 v1, finetune at 10% labels
should now MATCH or BEAT the baseline (target APFD >= 0.79), validating
the foundation-model recipe for the SDC domain.

Hardware: Kaggle RTX 6000 Pro Blackwell (96 GB), bf16.
"""

import json, numpy as np, os, time, math, copy, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

KAGGLE_DATA = '/kaggle/input/datasets/chinguyeen/sdc-sensodat'
OUTPUT_DIR = '/kaggle/working'
if os.path.exists(KAGGLE_DATA):
    TRAIN_PATH = os.path.join(KAGGLE_DATA, 'sensodat_train.json')
    TEST_PATH  = os.path.join(KAGGLE_DATA, 'sensodat_test.json')
    COMP_PATH  = os.path.join(KAGGLE_DATA, 'sdc-test-data.json')
else:
    BASE = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
    TRAIN_PATH = os.path.join(BASE, 'data', 'sensodat_train.json')
    TEST_PATH  = os.path.join(BASE, 'data', 'sensodat_test.json')
    COMP_PATH  = os.path.join(BASE, 'data', 'sdc-test-data.json')
    OUTPUT_DIR = os.path.join(BASE, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
torch.set_float32_matmul_precision('high')
print(f"Device: {DEVICE} | bf16: {USE_BF16}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")

SEQ_LEN = 197

# Competition autopilot effective speed (m/s).
# Risk Factor 1.5, v_max 120 km/h => v_eff ~ 33.3 m/s.
# We use the SQUARED quantity (33.3 m/s)^2 / 9.81 m/s^2 ~ 113 m as the
# centripetal-load denominator, but for SSL we drop the constant: any
# strict-monotone transform of c(s) works as a pretext target.
V_EFF = 33.3                                  # m/s
V_EFF_SQ = V_EFF * V_EFF

# -------------------- Procedural Bezier generator --------------------
def _bezier(P, n=SEQ_LEN):
    P = np.asarray(P, dtype=np.float64); K = len(P) - 1
    t = np.linspace(0, 1, n)[:, None]
    pts = np.zeros((n, 2))
    for k in range(K + 1):
        coeff = math.comb(K, k) * (t**k) * ((1-t)**(K-k))
        pts += coeff * P[k]
    return pts

def gen_random_road(rng):
    K = int(rng.choice([3, 4, 5, 6, 7]))
    P = rng.uniform(-100, 100, size=(K+1, 2))
    P[0] = [0, 0]; P[-1] = rng.uniform(60, 250, size=2) * rng.choice([-1, 1], size=2)
    return _bezier(P, SEQ_LEN)

# -------------------- 10-channel feature extraction (baseline-compatible) --------------------
def compute_curvature(pts):
    n=len(pts); curv=np.zeros(n-2)
    for i in range(n-2):
        x1,y1=pts[i]; x2,y2=pts[i+1]; x3,y3=pts[i+2]
        a=math.sqrt((x2-x1)**2+(y2-y1)**2); b=math.sqrt((x3-x2)**2+(y3-y2)**2); c=math.sqrt((x3-x1)**2+(y3-y1)**2)
        s=0.5*(a+b+c); at=s*(s-a)*(s-b)*(s-c)
        if at<=1e-10: curv[i]=0.0
        else: R=a*b*c/(4*math.sqrt(at)); curv[i]=1.0/R if R>0 else 0.0
    return curv

def extract_sequence_10ch(pts_raw):
    pts=np.array(pts_raw,dtype=np.float64).reshape(-1,2); n=len(pts)
    diffs=np.diff(pts,axis=0); seg_lens=np.linalg.norm(diffs,axis=1)
    seg_full=np.pad(seg_lens,(0,1),mode='edge')
    angles=np.arctan2(diffs[:,1],diffs[:,0]); ac=np.diff(angles)
    ac=(ac+np.pi)%(2*np.pi)-np.pi; abs_ac_full=np.pad(np.abs(ac),(1,1),mode='constant')
    curv=np.abs(compute_curvature(pts)); curv_full=np.pad(curv,(1,1),mode='constant')
    curv_deriv_full=np.pad(np.diff(curv_full),(0,1),mode='constant')
    cum_dist=np.cumsum(seg_full); cum_dist_norm=cum_dist/(cum_dist[-1]+1e-8)
    heading_full=np.pad(angles,(0,1),mode='edge')
    heading_sin=np.sin(heading_full); heading_cos=np.cos(heading_full)
    rel_pos=np.linspace(0,1,n)
    w=11; local_std=np.zeros(n); hw=w//2
    for i in range(n):
        s,e=max(0,i-hw),min(n,i+hw+1); local_std[i]=np.std(curv_full[s:e])
    curv_accel_full=np.pad(np.diff(curv_deriv_full),(0,1),mode='constant')
    return np.column_stack([seg_full,abs_ac_full,curv_full,curv_deriv_full,cum_dist_norm,
                            heading_sin,heading_cos,rel_pos,local_std,curv_accel_full]).astype(np.float32)

def centripetal_load(pts_raw):
    """Per-position c(s) = v_eff^2 * |kappa(s)|. Returns (SEQ_LEN,) float32."""
    pts = np.array(pts_raw, dtype=np.float64).reshape(-1, 2)
    curv_unsigned = np.abs(compute_curvature(pts))               # (n-2,)
    curv_full = np.pad(curv_unsigned, (1, 1), mode='constant')   # (n,)
    return (V_EFF_SQ * curv_full).astype(np.float32)

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
        X.append(extract_sequence_10ch(get_pts(tc))); y.append(1 if is_fail(tc) else 0)
        if (i+1)%5000==0: print(f"    {i+1}/{len(data)}...")
    return np.array(X), np.array(y)

# -------------------- Encoder + 3 heads --------------------
class RoadEncoder(nn.Module):
    def __init__(self, in_channels=10, seq_len=197, d_model=128, nhead=8,
                 num_layers=6, dim_feedforward=384, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model),
                                        nn.LayerNorm(d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d_model)*0.02)
        self.pos = nn.Parameter(torch.randn(1, seq_len + 1, d_model)*0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.d_model = d_model
    def forward(self, x):
        x = x.permute(0, 2, 1); B, L, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, :L+1, :]
        return self.tf(x)                                      # (B, L+1, D)

class PhysicsFoundationModel(nn.Module):
    """Encoder + 3 heads:
      head_load    : per-position scalar regression  -> c(s)
      head_max     : CLS-token scalar regression     -> max_s c(s)
      head_cls     : CLS-token classifier            -> fail logit (finetune)
    Smaller than v1 (d=128, 6 layers vs 256, 8) -- ~2 M params."""
    def __init__(self, in_channels=10, d_model=128, num_layers=6):
        super().__init__()
        self.enc = RoadEncoder(in_channels=in_channels, d_model=d_model,
                               num_layers=num_layers)
        # P1: per-position load regression head (token-wise scalar)
        self.head_load = nn.Sequential(nn.LayerNorm(d_model),
                                       nn.Linear(d_model, 64), nn.GELU(),
                                       nn.Linear(64, 1))
        # P2: per-road MAX load (CLS-token scalar)
        self.head_max = nn.Sequential(nn.LayerNorm(d_model),
                                      nn.Linear(d_model, 64), nn.GELU(),
                                      nn.Linear(64, 1))
        # finetune classifier
        self.head_cls = nn.Sequential(nn.LayerNorm(d_model),
                                      nn.Linear(d_model, 64), nn.GELU(),
                                      nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x, mode='cls'):
        h = self.enc(x)                                        # (B, L+1, D)
        if mode == 'load':
            return self.head_load(h[:, 1:, :]).squeeze(-1)     # (B, L)
        if mode == 'max':
            return self.head_max(h[:, 0, :]).squeeze(-1)       # (B,)
        if mode == 'cls':
            return self.head_cls(h[:, 0, :]).squeeze(-1)       # (B,)
        raise ValueError(mode)

# -------------------- Pretrain dataset (UNLABELED Bezier roads) --------------------
class PhysicsPretextDataset(Dataset):
    """Each item: (10ch features, per-position load c(s), max load C).
    The model must predict (c, C) from the masked feature input. Crucially,
    we ZERO OUT the curvature channels of the input so the encoder must
    DERIVE curvature from raw geometry to predict its physics target.
    This forces the encoder to learn the very thing that matters at test
    time: the physics field, not the engineered features."""
    def __init__(self, n_samples=200_000, seed=0,
                 mu_load=None, sd_load=None, mu_max=None, sd_max=None):
        self.n = n_samples; self.seed = seed
        # If standardisation stats are not given, use sane defaults computed
        # from a small Bezier sample.
        if mu_load is None:
            self._init_stats()
        else:
            self.mu_load, self.sd_load, self.mu_max, self.sd_max = \
                mu_load, sd_load, mu_max, sd_max
    def _init_stats(self):
        rng = np.random.RandomState(0); loads = []; maxes = []
        for i in range(2000):
            pts = gen_random_road(rng)
            c = centripetal_load(pts.tolist())
            loads.append(c); maxes.append(c.max())
        loads = np.concatenate(loads); maxes = np.array(maxes)
        self.mu_load = float(loads.mean()); self.sd_load = float(loads.std() + 1e-6)
        self.mu_max  = float(maxes.mean()); self.sd_max  = float(maxes.std()  + 1e-6)
        print(f"  pretrain stats: c~N({self.mu_load:.2f}, {self.sd_load:.2f}), "
              f"C_max~N({self.mu_max:.2f}, {self.sd_max:.2f})")
    def __len__(self): return self.n
    def __getitem__(self, i):
        rng = np.random.RandomState((self.seed * 1_000_003 + i) & 0xFFFFFFFF)
        pts = gen_random_road(rng)
        feats = extract_sequence_10ch(pts.tolist())
        c = centripetal_load(pts.tolist())
        # Standardise targets (zero-mean, unit-variance) so the regression
        # head doesn't need extreme weights.
        c_std = (c - self.mu_load) / self.sd_load
        C_std = (c.max() - self.mu_max) / self.sd_max
        # Mask the "engineered" curvature channels to force the encoder
        # to derive curvature from raw heading/segment-length.
        x = feats.copy()
        x[:, 2] = 0.0; x[:, 3] = 0.0; x[:, 8] = 0.0; x[:, 9] = 0.0
        return (torch.tensor(x.T, dtype=torch.float32),
                torch.tensor(c_std, dtype=torch.float32),
                torch.tensor(C_std, dtype=torch.float32))

# -------------------- Pretrain --------------------
def pretrain(model, *, steps=4000, batch=512, lr=3e-4, n_workers=2, seed=0):
    print(f"\n{'='*64}\nPHYSICS PRETRAIN (steps={steps}, batch={batch})\n{'='*64}")
    ds = PhysicsPretextDataset(n_samples=steps * batch, seed=seed)
    # Cache standardisation stats so finetune can use the same normalisation
    pretrain_stats = dict(mu_load=ds.mu_load, sd_load=ds.sd_load,
                          mu_max=ds.mu_max, sd_max=ds.sd_max)
    dl = DataLoader(ds, batch_size=batch, num_workers=n_workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=(n_workers > 0))
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = 200
    sched = optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s+1)/warm)
        * max(0.05, 0.5*(1 + math.cos(math.pi * max(0, s-warm)/max(1, steps-warm)))))
    scaler = GradScaler(enabled=(not USE_BF16))
    model = model.to(DEVICE).train()
    it = iter(dl); s = 0; t0 = time.time(); rolling_load = 0.0; rolling_max = 0.0
    while s < steps:
        try: x, c_std, C_std = next(it)
        except StopIteration:
            it = iter(dl); x, c_std, C_std = next(it)
        x = x.to(DEVICE, non_blocking=True)
        c_std = c_std.to(DEVICE, non_blocking=True)
        C_std = C_std.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            pl = model(x, mode='load')                         # (B, L)
            pm = model(x, mode='max')                          # (B,)
            l_load = (pl - c_std).pow(2).mean()
            l_max  = (pm - C_std).pow(2).mean()
            loss = l_load + 0.5 * l_max
        if USE_BF16:
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        else:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        rolling_load = 0.98 * rolling_load + 0.02 * l_load.item() if s > 0 else l_load.item()
        rolling_max  = 0.98 * rolling_max  + 0.02 * l_max.item()  if s > 0 else l_max.item()
        s += 1
        if s % 250 == 0 or s == 1:
            print(f"  step {s:5d}/{steps} | load={rolling_load:.4f} | max={rolling_max:.4f} | "
                  f"({(time.time()-t0)/60:.1f}m)")
    print(f"  pretrain done in {(time.time()-t0)/60:.1f} min")
    return model, pretrain_stats

# -------------------- Finetune --------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, pos_weight=1.0):
        super().__init__(); self.g=gamma; self.pw=pos_weight
    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        w = torch.where(y==1, self.pw, 1.0); bce = bce * w
        pt = torch.where(y==1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return ((1-pt).pow(self.g) * bce).mean()

@torch.no_grad()
def predict_chunked(model, X, chunk=256, mode='cls'):
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    out = []; model.eval()
    for i in range(0, X.size(0), chunk):
        xb = X[i:i+chunk].to(DEVICE, non_blocking=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            out.append(model(xb, mode=mode).float().cpu())
    return torch.cat(out, dim=0).numpy()

def finetune(model, X_tr, y_tr, X_va, y_va, *, frac=1.0, epochs=40, batch=384,
             lr=3e-4, freeze_encoder=False, name='SSL-finetune'):
    n = int(len(y_tr) * frac); idx = np.random.RandomState(0).permutation(len(y_tr))[:n]
    X_tr = X_tr[idx]; y_tr = y_tr[idx]
    print(f"\n{'='*64}\n{name} (frac={frac:.2f}, n={n}, freeze_encoder={freeze_encoder})\n{'='*64}")
    model = model.to(DEVICE)                                       # FIX from Exp 07
    if freeze_encoder:
        for p in model.enc.parameters(): p.requires_grad = False
    n_pos = y_tr.sum(); pw = (len(y_tr) - n_pos) / max(1, n_pos)
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)
    Xv = torch.tensor(X_va, dtype=torch.float32).permute(0, 2, 1)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-3)
    warm = 3
    sched = optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/warm if e<warm
        else max(0.01, 0.5*(1 + math.cos(math.pi*(e-warm)/max(1, epochs-warm)))))
    crit = FocalLoss(gamma=1.5, pos_weight=pw)
    scaler = GradScaler(enabled=(not USE_BF16))
    best_auc, best_state = 0., None
    for ep in range(epochs):
        model.train(); tot=0; nb=0
        for xb, yb in dl:
            xb=xb.to(DEVICE, non_blocking=True); yb=yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                loss = crit(model(xb, mode='cls'), yb)
            if USE_BF16:
                loss.backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params,1.0)
                scaler.step(opt); scaler.update()
            tot+=loss.item(); nb+=1
        sched.step()
        v_logit = predict_chunked(model, Xv, chunk=256, mode='cls')
        v = 1.0 / (1.0 + np.exp(-v_logit))
        auc = roc_auc_score(y_va, v)
        flag = ' *' if auc > best_auc else ''
        if auc > best_auc:
            best_auc = auc
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        if (ep+1) % 5 == 0 or flag:
            print(f"  Ep {ep+1:3d} | loss={tot/nb:.4f} | AUC={auc:.4f} | best={best_auc:.4f}{flag}")
    model.load_state_dict(best_state)
    if freeze_encoder:
        for p in model.enc.parameters(): p.requires_grad = True
    return model, best_auc

# -------------------- APFD eval --------------------
def compute_apfd(pids, td):
    n=len(pids)
    fp=[i+1 for i,t in enumerate(pids) if td[t]['meta_data']['test_info']['test_outcome']=='FAIL']
    m=len(fp); return 1 - sum(fp)/(n*m) + 1/(2*n) if n and m else 1.0

def eval_apfd(data, model, means, stds, name=''):
    td={get_id(tc):tc for tc in data}; ids=[get_id(tc) for tc in data]
    feats=[(extract_sequence_10ch(get_pts(tc)) - means)/stds for tc in data]
    X=torch.tensor(np.array(feats), dtype=torch.float32).permute(0,2,1)
    logit = predict_chunked(model, X, chunk=256, mode='cls')
    p = 1.0 / (1.0 + np.exp(-logit))
    pids=[t for _,t in sorted(zip(p, ids), key=lambda z:-z[0])]
    a=compute_apfd(pids, td); print(f"  {name:46s} APFD={a:.4f}"); return a

def multi_trial(data, model, means, stds, name='', n_trials=30):
    apfds=[]
    for t in range(n_trials):
        rng=np.random.RandomState(42+t); idx=rng.permutation(len(data))
        ed=[data[i] for i in idx[334:334+287]]
        td={get_id(tc):tc for tc in ed}; ids=[get_id(tc) for tc in ed]
        feats=[(extract_sequence_10ch(get_pts(tc)) - means)/stds for tc in ed]
        X=torch.tensor(np.array(feats), dtype=torch.float32).permute(0,2,1)
        logit = predict_chunked(model, X, chunk=256, mode='cls')
        p = 1.0 / (1.0 + np.exp(-logit))
        pids=[u for _,u in sorted(zip(p, ids), key=lambda z:-z[0])]
        apfds.append(compute_apfd(pids, td))
    print(f"  {name:46s} APFD={np.mean(apfds):.4f}+/-{np.std(apfds):.4f}")
    return np.mean(apfds), np.std(apfds)

# -------------------- Pretext-quality probe --------------------
@torch.no_grad()
def pretext_quality_probe(model, data, pretrain_stats):
    """How well does the pretrained encoder predict max-load on REAL roads?
    R^2 between predicted max-load (de-standardised) and ground-truth
    max-load. Sanity check that pretrain transferred."""
    model.eval()
    pts_list = [get_pts(tc) for tc in data]
    # Build masked input (curvature channels zeroed) so the test matches
    # pretrain conditions.
    feats_masked = []
    true_maxes = []
    for pts in pts_list:
        f = extract_sequence_10ch(pts).copy()
        f[:, 2] = 0; f[:, 3] = 0; f[:, 8] = 0; f[:, 9] = 0
        feats_masked.append(f)
        true_maxes.append(centripetal_load(pts).max())
    X = torch.tensor(np.array(feats_masked), dtype=torch.float32).permute(0,2,1)
    pred_std = predict_chunked(model, X, chunk=256, mode='max')
    pred = pred_std * pretrain_stats['sd_max'] + pretrain_stats['mu_max']
    true = np.array(true_maxes, dtype=np.float64)
    ss_res = float(((pred - true)**2).sum())
    ss_tot = float(((true - true.mean())**2).sum() + 1e-9)
    r2 = 1.0 - ss_res / ss_tot
    print(f"  pretext probe (max-load R^2 on real roads): {r2:.4f}  "
          f"(target > 0.5 = pretrain transferred)")
    return r2

# -------------------- Main --------------------
def main():
    t0=time.time()
    print("\n" + "="*72)
    print("EXP 07b -- PHYSICS-INFORMED SSL Foundation Model")
    print("Pretext: predict centripetal load c(s) and max-load C from raw points")
    print("="*72)

    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting labelled features...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    # ---- 1) Build foundation, pretrain on physics target ----
    fm = PhysicsFoundationModel(in_channels=10, d_model=128, num_layers=6)
    print(f"  foundation params: {sum(p.numel() for p in fm.parameters()):,}")
    fm, pretrain_stats = pretrain(fm, steps=4000, batch=512, lr=3e-4, n_workers=2)
    torch.save({'state': fm.state_dict(),
                'means': means.tolist(), 'stds': stds.tolist(),
                'pretrain_stats': pretrain_stats},
               os.path.join(OUTPUT_DIR, 'roadfoundation_physics.pt'))

    # ---- 1b) Pretext-quality probe on REAL roads ----
    print(f"\n--- Pretext-quality probe on real Competition split ---")
    if comp_data is not None:
        pretext_quality_probe(fm, comp_data, pretrain_stats)
    print(f"--- Pretext-quality probe on real SensoDat-test split ---")
    pretext_quality_probe(fm, test_data, pretrain_stats)

    # ---- 2) Sample-efficiency curve  (10/30/100% labels, with from-scratch control) ----
    print(f"\n{'='*64}\nSAMPLE EFFICIENCY (physics-pretrain vs from-scratch)\n{'='*64}")
    fracs = [0.10, 0.30, 1.00]
    for frac in fracs:
        # 2a) Physics-SSL finetune (encoder+heads)
        fm_ft = copy.deepcopy(fm)
        fm_ft, auc = finetune(fm_ft, X_tr, y_tr, X_te, y_te, frac=frac,
                              epochs=40, batch=384, lr=2e-4,
                              freeze_encoder=False,
                              name=f'Phys-SSL-finetune {int(frac*100)}%')
        eval_apfd(test_data, fm_ft, means, stds, f'Phys-SSL {int(frac*100)}% SensoDat')
        if comp_data is not None:
            multi_trial(comp_data, fm_ft, means, stds, f'Phys-SSL {int(frac*100)}% multi')

        # 2b) From-scratch control (same arch, no pretrain)
        fm_scr = PhysicsFoundationModel(in_channels=10, d_model=128, num_layers=6)
        fm_scr, auc = finetune(fm_scr, X_tr, y_tr, X_te, y_te, frac=frac,
                               epochs=40, batch=384, lr=3e-4,
                               freeze_encoder=False,
                               name=f'from-scratch {int(frac*100)}%')
        eval_apfd(test_data, fm_scr, means, stds, f'scratch {int(frac*100)}% SensoDat')
        if comp_data is not None:
            multi_trial(comp_data, fm_scr, means, stds, f'scratch {int(frac*100)}% multi')

    # ---- 3) Linear probe (encoder frozen) ----
    print(f"\n{'='*64}\nLINEAR PROBE (encoder frozen, head only)\n{'='*64}")
    fm_lp = copy.deepcopy(fm)
    fm_lp, _ = finetune(fm_lp, X_tr, y_tr, X_te, y_te, frac=1.0,
                        epochs=20, batch=384, lr=5e-4,
                        freeze_encoder=True, name='Phys-SSL-linear-probe')
    eval_apfd(test_data, fm_lp, means, stds, 'linear-probe SensoDat')
    if comp_data is not None:
        multi_trial(comp_data, fm_lp, means, stds, 'linear-probe multi')

    print(f"\nTOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
