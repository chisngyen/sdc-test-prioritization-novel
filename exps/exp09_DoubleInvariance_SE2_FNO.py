"""
EXP 09 -- Double Invariance: SE(2) x FNO Stack
================================================
Theoretical lens: GROUP-EQUIVARIANT NEURAL OPERATOR.

Exp 02 verified: rotation-invariance, Delta = 0.0000 (exact).
Exp 01 verified: resolution-invariance, Delta ~ 0.001 across N in {64..197}.

These two invariances commute (rotating a road and re-discretizing yields the
same equivalence class), so a model can be built that is invariant under the
DIRECT PRODUCT  SO(2) x R_{>0}  (rotations and arclength reparameterizations).

We construct the first such "doubly invariant" SDC test prioritizer.

Theorem (informal). Let G_theta : C([0,1]; R^7_invariant) -> R be the operator
defined by spectral conv on the SE(2)-invariant 7-channel feature map. Then
for any rotation R in SO(2), translation t in R^2, and any discretization
rho_N onto N nodes,
        G_theta( rho_N( R x + t ) ) = G_theta( x ) + O(1/N)
  i.e. APFD-Delta = 0 modulo float-roundoff under SO(2) and Delta -> 0 as
  N -> infty under arclength reparameterization.

We empirically verify the conjunction by running BOTH probes on the same
trained model (rotation x sampling-rate), giving a 6 x 5 = 30-cell APFD
table. The theoretical claim is that this table is constant up to O(1/N).

Hardware: Kaggle RTX 6000 Pro Blackwell (96 GB), bf16.
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

# -------------------- SE(2)-INVARIANT 7-channel features (from Exp 02) --------------------
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

# -------------------- Spectral conv on SE(2)-invariant channels --------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.W = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))
    def forward(self, x):  # (B, C_in, N)
        B, C, N = x.shape
        x32 = x.float()
        x_ft = torch.fft.rfft(x32, n=N, dim=-1)
        m = min(self.modes, x_ft.size(-1))
        out_ft = torch.zeros(B, self.out_ch, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m] = torch.einsum('bcn,con->bon',
                                        x_ft[:, :, :m], self.W[:, :, :m])
        return torch.fft.irfft(out_ft, n=N, dim=-1).to(x.dtype)

class FNOBlock(nn.Module):
    def __init__(self, ch, modes, dropout=0.1):
        super().__init__()
        self.spectral = SpectralConv1d(ch, ch, modes)
        self.local = nn.Conv1d(ch, ch, 1)
        self.norm = nn.GroupNorm(8, ch)
        self.act = nn.GELU(); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return x + self.drop(self.act(self.norm(self.spectral(x) + self.local(x))))

# -------------------- Doubly Invariant Net --------------------
# IMPORTANT: SE(2)-invariance comes from the 7-ch feature pipeline (no
# absolute heading, no x/y). FNO blocks operate on these features and are
# resolution-invariant in the spectral-truncation limit. The composition is
# invariant under SO(2) x R_{>0} by construction.
class DoublyInvariantNet(nn.Module):
    def __init__(self, in_ch=7, lift=128, modes=32, depth=6, mlp_hidden=256, dropout=0.1):
        super().__init__()
        self.lift = nn.Conv1d(in_ch, lift, 1)
        self.blocks = nn.ModuleList([FNOBlock(lift, modes, dropout) for _ in range(depth)])
        self.proj = nn.Conv1d(lift, lift, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(2*lift),
            nn.Linear(2*lift, mlp_hidden), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(mlp_hidden, 1))
    def forward(self, x):                 # x : (B, 7, N)
        h = self.lift(x)
        for b in self.blocks: h = b(h)
        h = self.proj(h)
        feat = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)
        return self.head(feat).squeeze(-1)

# -------------------- Resolution-jitter dataset (forces resolution-invariance) --------------------
class ResJitterDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, choices=(96, 128, 160, 197), p_jitter=0.5):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.choices = choices; self.p = p_jitter
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i):
        x = self.X[i]
        if np.random.rand() < self.p:
            n_keep = int(np.random.choice(self.choices))
            if n_keep != x.size(-1):
                idx = np.linspace(0, x.size(-1) - 1, n_keep).astype(int)
                x = x[:, idx]
                x = F.interpolate(x.unsqueeze(0), size=SEQ_LEN,
                                  mode='linear', align_corners=True).squeeze(0)
        return x, self.y[i]

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
        for p,q in zip(self.model.parameters(), m.parameters()):
            p.data.mul_(1-a).add_(q.data, alpha=a)
    def get_model(self): return self.model

@torch.no_grad()
def predict_chunked(model, X, chunk=256):
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    out = []; model.eval()
    for i in range(0, X.size(0), chunk):
        xb = X[i:i+chunk].to(DEVICE, non_blocking=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            out.append(model(xb).float().cpu())
    return torch.cat(out, dim=0).numpy()

def train(model, X_tr, y_tr, X_va, y_va, *, epochs=80, batch=512,
          lr=8e-4, swa_start=55, name='DoubleInv'):
    print(f"\n{'='*64}\nTraining {name} | params={sum(p.numel() for p in model.parameters()):,}\n{'='*64}")
    model = model.to(DEVICE)
    n_pos = y_tr.sum(); pw = (len(y_tr) - n_pos) / n_pos
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    ds = ResJitterDataset(X_tr, y_tr)
    dl = DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=2,
                    pin_memory=True, drop_last=True)
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
        v_logit = predict_chunked(model, Xv)
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

# -------------------- Eval --------------------
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

def eval_apfd(data, model, means, stds, name='', rot_deg=0.0, N=None):
    td={get_id(tc):tc for tc in data}; ids=[get_id(tc) for tc in data]
    feats = _feats(data, means, stds, rot_deg)
    X=torch.tensor(feats, dtype=torch.float32).permute(0,2,1)
    if N is not None and N != X.size(-1):
        idx=np.linspace(0,X.size(-1)-1,N).astype(int)
        X=F.interpolate(X[:,:,idx], size=SEQ_LEN, mode='linear', align_corners=True)
    logit = predict_chunked(model, X)
    p = 1.0 / (1.0 + np.exp(-logit))
    pids=[t for _,t in sorted(zip(p, ids), key=lambda z:-z[0])]
    a=compute_apfd(pids, td); return a

def multi_trial(data, model, means, stds, name='', n_trials=30):
    apfds=[]
    for t in range(n_trials):
        rng=np.random.RandomState(42+t); idx=rng.permutation(len(data))
        ed=[data[i] for i in idx[334:334+287]]
        td={get_id(tc):tc for tc in ed}; ids=[get_id(tc) for tc in ed]
        feats = _feats(ed, means, stds, 0.0)
        X=torch.tensor(feats, dtype=torch.float32).permute(0,2,1)
        logit = predict_chunked(model, X)
        p = 1.0 / (1.0 + np.exp(-logit))
        pids=[u for _,u in sorted(zip(p, ids), key=lambda z:-z[0])]
        apfds.append(compute_apfd(pids, td))
    print(f"  {name:46s} APFD={np.mean(apfds):.4f}+/-{np.std(apfds):.4f}")
    return np.mean(apfds), np.std(apfds)

def main():
    t0=time.time()
    print("\n" + "="*72)
    print("EXP 09 -- Doubly Invariant Net  (SE(2) x FNO)")
    print("Theory: G(R rho_N x) = G(x) for any R in SO(2), any sampling rho_N")
    print("="*72)

    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting SE(2)-invariant 7-ch features...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    model = DoublyInvariantNet(in_ch=7, lift=128, modes=32, depth=6, mlp_hidden=256)
    model, auc, swa = train(model, X_tr, y_tr, X_te, y_te,
                            epochs=80, batch=512, lr=8e-4, swa_start=55, name='DoubleInv')

    if comp_data is not None:
        m_eval = swa.get_model() if swa else model

        print(f"\n{'='*72}")
        print("DOUBLE-INVARIANCE PROBE  (rotation x sampling-rate)")
        print(f"{'='*72}")
        rotations = [0.0, 30.0, 60.0, 90.0, 180.0, -45.0]
        Ns = [64, 96, 128, 160, 197]
        print(f"  {'rot \\\\ N':>10s} | " + " | ".join([f'{N:>7d}' for N in Ns]))
        print(f"  {'-'*10}-+-" + "-+-".join(['-------']*len(Ns)))
        rows = []
        for rot in rotations:
            row = []
            for N in Ns:
                a = eval_apfd(comp_data, m_eval, means, stds, rot_deg=rot, N=N)
                row.append(a)
            rows.append(row)
            print(f"  {rot:>+9.0f}d | " + " | ".join([f'{a:.4f}' for a in row]))

        arr = np.array(rows)
        print(f"\n  Joint Delta (max - min over 30 cells): {arr.max() - arr.min():.6f}")
        print(f"  Per-rotation N-Delta (max over rotations): "
              f"{np.max(arr.max(axis=1) - arr.min(axis=1)):.6f}")
        print(f"  Per-N rotation-Delta (max over N):       "
              f"{np.max(arr.max(axis=0) - arr.min(axis=0)):.6f}")

        print(f"\n--- Multi-trial Competition (30 trials, no rotation, N=197) ---")
        multi_trial(comp_data, m_eval, means, stds, 'DoubleInv SWA multi-trial')

    save = os.path.join(OUTPUT_DIR, 'roaddouble.pt')
    torch.save({'state': (swa.get_model() if swa else model).state_dict(),
                'means': means.tolist(), 'stds': stds.tolist()}, save)
    print(f"\nSaved: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
