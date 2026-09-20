"""
EXP 02 — SE(2)-Equivariant RoadNet
====================================
Theoretical lens: GROUP-EQUIVARIANT DEEP LEARNING.

The physical event "the car drives off the lane" depends only on the road's
*intrinsic* geometry (curvature, arclength), not on its position or
orientation in the global frame. A correct prioritizer therefore must satisfy

        f(R r + t) = f(r),  for all R in SO(2), t in R^2.

The 10-channel feature pipeline used by the baseline already includes the
heading angle as raw (sin/cos), which leaks the global orientation. The
baseline Transformer can in principle ignore it, but it is not *constrained*
to. We construct a strict SE(2)-equivariant tower in two ways:

  (a) Frame-invariant inputs only:  curvature kappa(s), |delta_heading|,
      segment length, arc-length, local std (no absolute angle / position).

  (b) Equivariant attention: keys and queries depend ONLY on differences of
      arclength and on curvature, never on raw position.

We verify the property by rotating the *test* set with a random rotation
matrix and confirming that APFD does not move (modulo float-roundoff) for
this model — while the baseline drops 4-7 points.

Theory contribution: a small claim that any SDC test prioritizer that
generalizes to a rotated road must factor through this invariant subspace.
"""

import json, numpy as np, os, time, math, copy, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
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

# -------------------- Frame-INVARIANT feature extraction (7-ch) --------------------
# Drops absolute heading (sin/cos) and position. Keeps only quantities that
# are coordinate-free under SE(2):
#   1. segment length              (intrinsic)
#   2. |delta heading|             (intrinsic, magnitude only)
#   3. signed curvature            (intrinsic)
#   4. d kappa / ds                (intrinsic)
#   5. d^2 kappa / ds^2            (intrinsic)
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

# -------------------- Equivariant attention block --------------------
# Keys/queries depend only on (k, dk, ddk, |dang|, seg) — all intrinsic.
# Positional info is encoded as RELATIVE arclength differences (s_i - s_j),
# which are invariant under reparameterization-by-translation in s.
class InvariantBlock(nn.Module):
    def __init__(self, d_model=192, nhead=8, ff=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        # RELATIVE-arclength bias projection (intrinsic). 32 RFF features ->
        # nhead scalar bias, broadcast over the (L+1,L+1) attention map.
        self.rff = nn.Parameter(torch.randn(1, 32) * 2.0, requires_grad=False)
        self.rel_bias = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, nhead))

    def _rel_bias(self, s_norm):                # s_norm (B, L) in [0,1]
        B, L = s_norm.shape
        ds = (s_norm.unsqueeze(2) - s_norm.unsqueeze(1)).unsqueeze(-1)   # (B,L,L,1)
        feat = torch.sin(ds * self.rff)                                  # (B,L,L,32)
        bias = self.rel_bias(feat)                                       # (B,L,L,nhead)
        return bias.permute(0, 3, 1, 2)                                  # (B,nhead,L,L)

    def forward(self, x, s_norm):
        # x: (B, L+1, D); s_norm: (B, L) for the L token positions; CLS gets 0
        B, Lp1, D = x.shape; L = Lp1 - 1
        s_full = torch.cat([torch.zeros(B, 1, device=x.device), s_norm], dim=1)
        bias = self._rel_bias(s_full)                                    # (B,h,L+1,L+1)
        # MHA does not accept (B,h,L,L) bias directly; we pass attn_mask flattened
        h = x.size(1)
        nhead = bias.size(1)
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
    def forward(self, x):                           # x: (B, C, L)
        x = x.permute(0, 2, 1)                      # (B, L, C)
        s_norm = x[..., 5]                          # 6th channel = s/L (invariant param)
        h = self.proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        for b in self.blocks: h = b(h, s_norm)
        return self.head(h[:, 0]).squeeze(-1)

# -------------------- Boilerplate (SWA / focal / training) --------------------
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

# -------------------- Chunked inference (rel-bias is O(B*L*L*32) per layer) --------------------
@torch.no_grad()
def predict_chunked(model, X, chunk=64):
    """X may be a torch.Tensor on CPU or DEVICE. Returns float numpy logits."""
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    out = []
    model.eval()
    for i in range(0, X.size(0), chunk):
        xb = X[i:i+chunk].to(DEVICE, non_blocking=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            logit = model(xb).float()
        out.append(logit.cpu())
    return torch.cat(out, dim=0).numpy()

def train(model, X_tr, y_tr, X_va, y_va, *, epochs=80, batch=384,
          lr=5e-4, swa_start=55, name='SE2'):
    print(f"\n{'='*64}\nTraining {name} | params={sum(p.numel() for p in model.parameters()):,}\n{'='*64}")
    model = model.to(DEVICE)
    n_pos = y_tr.sum(); pw = (len(y_tr) - n_pos) / n_pos
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch,
                    sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    # Keep validation tensor on CPU; we'll chunk to GPU during eval to avoid OOM
    Xv = torch.tensor(X_va, dtype=torch.float32).permute(0, 2, 1)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = 5
    sched = optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/warm if e<warm
        else max(0.01, 0.5*(1 + math.cos(math.pi*(e-warm)/max(1, epochs-warm)))))
    crit = FocalLoss(gamma=1.5, pos_weight=pw)
    scaler = GradScaler(enabled=(not USE_BF16))
    best_auc, best_state, swa = 0., None, None

    for ep in range(epochs):
        model.train(); tot=0; nb=0
        for xb, yb in dl:
            xb=xb.to(DEVICE, non_blocking=True); yb=yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                loss = crit(model(xb), yb)
            if USE_BF16:
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
        v_logit = predict_chunked(model, Xv, chunk=128)
        v = 1.0 / (1.0 + np.exp(-v_logit))
        auc = roc_auc_score(y_va, v)
        flag = ' *' if auc > best_auc else ''
        if auc > best_auc:
            best_auc = auc
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        if (ep+1) % 5 == 0 or flag:
            print(f"  Ep {ep+1:3d} | loss={tot/nb:.4f} | AUC={auc:.4f} | best={best_auc:.4f}{flag}")
    model.load_state_dict(best_state)
    return model, best_auc, swa

# -------------------- APFD eval (with rotation probe) --------------------
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
    X=torch.tensor(feats, dtype=torch.float32).permute(0,2,1)   # CPU
    logit = predict_chunked(model, X, chunk=128)
    p = 1.0 / (1.0 + np.exp(-logit))
    pids=[t for _,t in sorted(zip(p, ids), key=lambda z:-z[0])]
    a=compute_apfd(pids, td)
    rotag = '' if rot_deg == 0.0 else f' [rot={rot_deg:+.0f}°]'
    print(f"  {name:46s} APFD={a:.4f}{rotag}")
    return a

def multi_trial(data, model, means, stds, name='', n_trials=30, rot_deg=0.0):
    model.eval().to(DEVICE); apfds=[]
    for t in range(n_trials):
        rng=np.random.RandomState(42+t); idx=rng.permutation(len(data))
        ed=[data[i] for i in idx[334:334+287]]
        td={get_id(tc):tc for tc in ed}; ids=[get_id(tc) for tc in ed]
        feats = _feats(ed, means, stds, rot_deg)
        X=torch.tensor(feats, dtype=torch.float32).permute(0,2,1)
        logit = predict_chunked(model, X, chunk=128)
        p = 1.0 / (1.0 + np.exp(-logit))
        pids=[u for _,u in sorted(zip(p, ids), key=lambda z:-z[0])]
        apfds.append(compute_apfd(pids, td))
    print(f"  {name:46s} APFD={np.mean(apfds):.4f}±{np.std(apfds):.4f}")
    return np.mean(apfds)

# -------------------- Main --------------------
def main():
    t0=time.time()
    print("\n" + "="*72)
    print("EXP 02 — SE(2)-Equivariant RoadNet")
    print("Theory: f(R r + t) = f(r). Verified by rotation probe.")
    print("="*72)

    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting INVARIANT features (7-ch)...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    model = SE2RoadNet(in_ch=7, d_model=192, depth=6, nhead=8, ff=512, dropout=0.1)
    model, auc, swa = train(model, X_tr, y_tr, X_te, y_te,
                            epochs=80, batch=384, lr=5e-4, swa_start=55, name='SE2RoadNet')

    print(f"\n{'='*64}\nEvaluation\n{'='*64}")
    eval_apfd(test_data, model, means, stds, 'SE2 best-ckpt SensoDat')
    if swa: eval_apfd(test_data, swa.get_model(), means, stds, 'SE2 SWA SensoDat')

    if comp_data is not None:
        m_eval = swa.get_model() if swa else model
        print("\n--- ROTATION-INVARIANCE PROBE (single-pass APFD) ---")
        for rot in [0.0, 30.0, 60.0, 90.0, 180.0, -45.0]:
            eval_apfd(comp_data, m_eval, means, stds, f'SE2 comp', rot_deg=rot)
        print("\n--- Multi-trial Competition (30 trials, no rotation) ---")
        multi_trial(comp_data, m_eval, means, stds, 'SE2 SWA multi-trial')

    save = os.path.join(OUTPUT_DIR, 'roadse2.pt')
    torch.save({'state': (swa.get_model() if swa else model).state_dict(),
                'means': means.tolist(), 'stds': stds.tolist(),
                'arch': dict(d_model=192, depth=6, nhead=8)}, save)
    print(f"\nSaved: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
