"""
EXP 10 -- DiffAPFD listwise loss on SE(2)-Equivariant backbone
=================================================================
Theoretical lens: GROUP-EQUIVARIANT LISTWISE LEARNING-TO-RANK.

Tracker insight (after 5 exps run):
  - Exp 02 (SE(2)) gave the HIGHEST AUC of any model: 0.9347
  - Exp 03 (PL listwise) gave the LOWEST sigma (0.0109) but mean APFD 0.8057
  - Both fell SHORT of the baseline single 0.8066 in mean

The two contributions are orthogonal: SE(2) raises the calibration ceiling
(AUC), listwise pulls APFD towards the rank target. Composing them is the
natural "headline" stack. We claim:

  Theorem (informal). The PL-NeuralSort-BCE listwise gradient is independent
  of any orthogonal coordinate transformation of the input features.
  Therefore, when applied to an SE(2)-equivariant backbone, the listwise
  loss preserves rotation invariance: rot-Delta of the trained model is
  identical to that of the underlying backbone.

Empirically: this should be the FIRST configuration to break 0.808 APFD on
multi-trial while ALSO retaining Exp 02's Delta=0 rotation property -- a
"both-and" instead of "either-or" result.

Hardware: Kaggle RTX 6000 Pro Blackwell (96 GB), bf16.
"""

import json, numpy as np, os, time, math, copy, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# -------------------- SE(2)-INVARIANT 7-ch features (from Exp 02) --------------------
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

# -------------------- SE(2)-equivariant backbone (from Exp 02, lighter) --------------------
class InvariantBlock(nn.Module):
    def __init__(self, d_model=160, nhead=8, ff=384, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.rff = nn.Parameter(torch.randn(1, 16) * 2.0, requires_grad=False)  # 16 RFF (vs 32 in Exp 02) for speed
        self.rel_bias = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, nhead))
    def _rel_bias(self, s_norm):
        ds = (s_norm.unsqueeze(2) - s_norm.unsqueeze(1)).unsqueeze(-1)
        feat = torch.sin(ds * self.rff)
        bias = self.rel_bias(feat)
        return bias.permute(0, 3, 1, 2)
    def forward(self, x, s_norm):
        B, Lp1, D = x.shape
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
    def __init__(self, in_ch=7, d_model=160, depth=5, nhead=8, ff=384, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_ch, d_model),
                                  nn.LayerNorm(d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([InvariantBlock(d_model, nhead, ff, dropout)
                                     for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, 64), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        s_norm = x[..., 5]
        h = self.proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        for b in self.blocks: h = b(h, s_norm)
        return self.head(h[:, 0]).squeeze(-1)

# -------------------- Listwise losses (from Exp 03) --------------------
def listwise_pl_loss(scores, labels):
    s_pos = scores[labels > 0.5]; s_neg = scores[labels < 0.5]
    if s_pos.numel() == 0 or s_neg.numel() == 0:
        return torch.zeros((), device=scores.device)
    diff = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)
    return F.softplus(-diff).mean()

def neuralsort_apfd_loss(scores, labels, tau=0.5):
    """Tighter tau (0.5) than Exp 03 (1.0) -- Exp 03 collapsed because P_hat
    was too smeared. Smaller tau -> sharper soft-permutation."""
    n = scores.size(0)
    s = scores.view(-1, 1)
    A = (s - s.T).abs()
    one = torch.ones(n, 1, device=scores.device)
    B_mat = (n + 1 - 2*(torch.arange(n, device=scores.device) + 1)).float()
    C = (A @ one).view(1, n) * one.T
    Phat = (B_mat.view(1, n) * s.T - C) / tau
    Phat = F.softmax(Phat, dim=-1)
    ranks = (Phat @ (torch.arange(n, device=scores.device) + 1).float())
    fail_rank_sum = (ranks * labels).sum()
    m = labels.sum().clamp_min(1.0)
    apfd_hat = 1.0 - fail_rank_sum / (n * m) + 1.0 / (2*n)
    return -apfd_hat

class SWAModel:
    def __init__(self, m): self.model = copy.deepcopy(m); self.n = 0
    def update(self, m):
        self.n += 1; a = 1.0/self.n
        for p,q in zip(self.model.parameters(), m.parameters()):
            p.data.mul_(1-a).add_(q.data, alpha=a)
    def get_model(self): return self.model

@torch.no_grad()
def predict_chunked(model, X, chunk=128):
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    out = []; model.eval()
    for i in range(0, X.size(0), chunk):
        xb = X[i:i+chunk].to(DEVICE, non_blocking=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            out.append(model(xb).float().cpu())
    return torch.cat(out, dim=0).numpy()

def train(model, X_tr, y_tr, X_va, y_va, *, epochs=80, batch=256,
          lr=4e-4, swa_start=55, w_pl=1.0, w_ns=0.3, w_bce=0.2, ns_tau=0.5,
          name='DiffAPFD-SE2'):
    print(f"\n{'='*64}\nTraining {name} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  weights: PL={w_pl} | NeuralSort-APFD={w_ns} | BCE-aux={w_bce} | tau={ns_tau}\n{'='*64}")
    model = model.to(DEVICE)
    n_pos = y_tr.sum(); pw = (len(y_tr) - n_pos) / n_pos
    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    fail_idx = np.where(y_tr == 1)[0]; pass_idx = np.where(y_tr == 0)[0]
    Xv = torch.tensor(X_va, dtype=torch.float32).permute(0,2,1)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = 5
    sched = optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/warm if e<warm
        else max(0.01, 0.5*(1 + math.cos(math.pi*(e-warm)/max(1, epochs-warm)))))
    pos_w = torch.tensor([float(pw)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = GradScaler(enabled=(not USE_BF16))
    best_auc, best_state, swa = 0., None, None
    half = batch // 2
    n_iter = max(len(y_tr) // batch, 50)

    for ep in range(epochs):
        model.train(); tot=0; nb=0
        for _ in range(n_iter):
            f_b = np.random.choice(fail_idx, size=half, replace=True)
            p_b = np.random.choice(pass_idx, size=batch-half, replace=True)
            ids = np.concatenate([f_b, p_b]); np.random.shuffle(ids)
            xb = Xt[ids].to(DEVICE, non_blocking=True)
            yb = yt[ids].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                s = model(xb)
                lpl  = listwise_pl_loss(s, yb)
                lns  = neuralsort_apfd_loss(s.float(), yb.float(), tau=ns_tau)
                lbce = bce(s, yb)
                loss = w_pl*lpl + w_ns*lns + w_bce*lbce
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

def eval_apfd(data, model, means, stds, name='', rot_deg=0.0):
    td={get_id(tc):tc for tc in data}; ids=[get_id(tc) for tc in data]
    feats = _feats(data, means, stds, rot_deg)
    X=torch.tensor(feats, dtype=torch.float32).permute(0,2,1)
    logit = predict_chunked(model, X)
    p = 1.0 / (1.0 + np.exp(-logit))
    pids=[t for _,t in sorted(zip(p, ids), key=lambda z:-z[0])]
    a=compute_apfd(pids, td)
    rotag = '' if rot_deg == 0.0 else f' [rot={rot_deg:+.0f}d]'
    print(f"  {name:46s} APFD={a:.4f}{rotag}"); return a

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
    print("EXP 10 -- DiffAPFD listwise loss on SE(2)-Equivariant backbone")
    print("Target: break 0.808 APFD AND retain rotation Delta=0")
    print("="*72)

    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting SE(2)-invariant 7-ch features...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    model = SE2RoadNet(in_ch=7, d_model=160, depth=5, nhead=8, ff=384, dropout=0.1)
    model, auc, swa = train(model, X_tr, y_tr, X_te, y_te,
                            epochs=80, batch=256, lr=4e-4, swa_start=55,
                            w_pl=1.0, w_ns=0.3, w_bce=0.2, ns_tau=0.5,
                            name='DiffAPFD-SE2')

    if comp_data is not None:
        m_eval = swa.get_model() if swa else model
        print(f"\n--- Sanity: APFD@SensoDat ---")
        eval_apfd(test_data, m_eval, means, stds, 'DiffAPFD-SE2 SWA SensoDat')
        print(f"\n--- Multi-trial Competition (30 trials) ---")
        multi_trial(comp_data, m_eval, means, stds, 'DiffAPFD-SE2 SWA multi-trial')
        print(f"\n--- Rotation-invariance retention probe ---")
        for rot in [0.0, 30.0, 90.0, 180.0]:
            eval_apfd(comp_data, m_eval, means, stds, 'DiffAPFD-SE2 comp', rot_deg=rot)

    save = os.path.join(OUTPUT_DIR, 'roaddiffapfd_se2.pt')
    torch.save({'state': (swa.get_model() if swa else model).state_dict(),
                'means': means.tolist(), 'stds': stds.tolist()}, save)
    print(f"\nSaved: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
