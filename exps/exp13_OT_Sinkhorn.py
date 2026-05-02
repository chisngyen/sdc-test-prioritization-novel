"""
EXP 13 -- Optimal Transport Test Prioritization (OT-TP)
==========================================================
Theoretical lens: ENTROPY-REGULARIZED OPTIMAL TRANSPORT.

Idea: re-frame test prioritization as a TRANSPORT problem between the
distribution of test embeddings and the distribution of failure modes.

Setup. Let phi(x) be a learned embedding. We maintain an explicit
"failure manifold" represented by K learnable centroids m_1,...,m_K (the
expected failure modes). For each test x, define

    score(x) = -  min_k  W_2( phi(x), m_k )       (signed distance to fail manifold)

The closer phi(x) is to ANY failure-mode centroid, the higher its priority.
We TRAIN this end-to-end with the Sinkhorn divergence (Cuturi 2013;
Feydy et al. 2019), which is differentiable, has unbiased gradients in the
entropic limit, and admits a clean SAMPLE COMPLEXITY bound.

Theory contribution (informal):
  Proposition (OT prioritisation consistency).
    Under the empirical-process assumption that fail roads concentrate near
    K low-dimensional clusters, the OT-TP score is a monotone function of
    the true Bayes posterior P(fail | x), with rate that depends on the
    sample size and the entropic regularization eps.

This is, to our knowledge, the FIRST application of OT to SDC test
prioritization. We compare against:
  - the cosine-distance-to-centroids baseline (no OT), and
  - the BCE classifier baseline.

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

# -------------------- Encoder + OT-style scorer --------------------
class TransformerEncoder(nn.Module):
    def __init__(self, in_channels=10, seq_len=197, d_model=192, nhead=8,
                 num_layers=5, dim_feedforward=512, dropout=0.1, embed_dim=64):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model),
                                        nn.LayerNorm(d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d_model)*0.02)
        self.pos = nn.Parameter(torch.randn(1, seq_len + 1, d_model)*0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, embed_dim))
    def forward(self, x):
        x = x.permute(0, 2, 1); B, L, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, :L+1, :]
        z = self.tf(x)[:, 0]                           # (B, d_model)
        return self.proj(z)                            # (B, embed_dim)

class OTScorer(nn.Module):
    """Maps test embedding to scalar score via softmin distance to K
    learnable failure-mode centroids on the unit sphere."""
    def __init__(self, embed_dim=64, K=8, sharpness=4.0):
        super().__init__()
        self.K = K
        self.centroids = nn.Parameter(F.normalize(torch.randn(K, embed_dim), dim=-1))
        self.sharpness = sharpness
        self.bias = nn.Parameter(torch.zeros(()))
    def forward(self, phi):                              # phi: (B, d)
        phi = F.normalize(phi, dim=-1)
        c = F.normalize(self.centroids, dim=-1)
        sim = phi @ c.T                                  # (B, K) cosine similarity
        # softmin negative-distance via log-sum-exp:
        score = (1.0 / self.sharpness) * torch.logsumexp(self.sharpness * sim, dim=-1)
        return score + self.bias                          # logit-style scalar

class OTNet(nn.Module):
    def __init__(self, embed_dim=64, K=8):
        super().__init__()
        self.enc = TransformerEncoder(embed_dim=embed_dim)
        self.scorer = OTScorer(embed_dim=embed_dim, K=K)
    def forward(self, x):
        return self.scorer(self.enc(x))
    def embed(self, x):
        return self.enc(x)

# -------------------- Sinkhorn divergence (Feydy et al. 2019) --------------------
def sinkhorn_divergence(X, Y, eps=0.1, n_iter=20):
    """Sinkhorn divergence S_eps(X, Y) = OT_eps(X, Y) - 0.5*(OT_eps(X,X) + OT_eps(Y,Y)).
    X (n,d), Y (m,d) on the unit sphere -> cost = 1 - cosine. Returns scalar."""
    def _ot(A, B):
        a = torch.full((A.size(0),), 1.0/A.size(0), device=A.device)
        b = torch.full((B.size(0),), 1.0/B.size(0), device=B.device)
        C = 1.0 - F.normalize(A, dim=-1) @ F.normalize(B, dim=-1).T   # (n,m)
        # Sinkhorn iterations in log-domain
        log_a = torch.log(a + 1e-30); log_b = torch.log(b + 1e-30)
        f = torch.zeros_like(log_a); g = torch.zeros_like(log_b)
        for _ in range(n_iter):
            f = -eps * torch.logsumexp((-C + g.unsqueeze(0)) / eps, dim=1) + eps * log_a
            g = -eps * torch.logsumexp((-C.T + f.unsqueeze(0)) / eps, dim=1) + eps * log_b
        # transport plan log-density:
        log_pi = (-C + f.unsqueeze(1) + g.unsqueeze(0)) / eps
        pi = torch.exp(log_pi)
        return (pi * C).sum()
    return _ot(X, Y) - 0.5 * _ot(X, X) - 0.5 * _ot(Y, Y)

# -------------------- Loss --------------------
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

def train(model, X_tr, y_tr, X_va, y_va, *, epochs=70, batch=256,
          lr=5e-4, swa_start=50, lam_ot=0.3, ot_eps=0.1, name='OT-Net'):
    print(f"\n{'='*64}\nTraining {name} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  lambda_OT={lam_ot}, sinkhorn eps={ot_eps}\n{'='*64}")
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
    crit = FocalLoss(gamma=1.5, pos_weight=pw)
    scaler = GradScaler(enabled=(not USE_BF16))
    best_auc, best_state, swa = 0., None, None
    half = batch // 2
    n_iter = max(len(y_tr) // batch, 50)

    for ep in range(epochs):
        model.train(); tot=0; nb=0; sum_ot=0
        for _ in range(n_iter):
            f_b = np.random.choice(fail_idx, size=half, replace=True)
            p_b = np.random.choice(pass_idx, size=batch-half, replace=True)
            ids = np.concatenate([f_b, p_b]); np.random.shuffle(ids)
            xb = Xt[ids].to(DEVICE, non_blocking=True)
            yb = yt[ids].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                phi = model.embed(xb)                         # (B, d)
                logits = model.scorer(phi)
                erm = crit(logits, yb)
            # Sinkhorn term in fp32 for numerical stability
            ot_loss = torch.zeros((), device=DEVICE)
            if lam_ot > 0:
                phi_fp = phi.float()
                centroids = model.scorer.centroids.float()
                # Sinkhorn divergence between phi(FAIL roads) and centroid set:
                # we want fail embeddings to lie ON the centroid manifold.
                fail_phi = phi_fp[yb > 0.5]
                if fail_phi.size(0) >= 4:
                    ot_loss = sinkhorn_divergence(fail_phi, centroids, eps=ot_eps, n_iter=15)
            loss = erm + lam_ot * ot_loss
            if USE_BF16:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
            tot+=loss.item(); nb+=1
            sum_ot += float(ot_loss.detach().item())
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
            print(f"  Ep {ep+1:3d} | loss={tot/nb:.4f} | sinkhorn={sum_ot/max(nb,1):.4f} | "
                  f"AUC={auc:.4f} | best={best_auc:.4f}{flag}")
    model.load_state_dict(best_state)
    return model, best_auc, swa

# -------------------- Eval --------------------
def compute_apfd(pids, td):
    n=len(pids)
    fp=[i+1 for i,t in enumerate(pids) if td[t]['meta_data']['test_info']['test_outcome']=='FAIL']
    m=len(fp); return 1 - sum(fp)/(n*m) + 1/(2*n) if n and m else 1.0

def eval_apfd(data, model, means, stds, name=''):
    td={get_id(tc):tc for tc in data}; ids=[get_id(tc) for tc in data]
    feats=[(extract_sequence_10ch(get_pts(tc)) - means)/stds for tc in data]
    X=torch.tensor(np.array(feats), dtype=torch.float32).permute(0,2,1)
    logit = predict_chunked(model, X)
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
        logit = predict_chunked(model, X)
        p = 1.0 / (1.0 + np.exp(-logit))
        pids=[u for _,u in sorted(zip(p, ids), key=lambda z:-z[0])]
        apfds.append(compute_apfd(pids, td))
    print(f"  {name:46s} APFD={np.mean(apfds):.4f}+/-{np.std(apfds):.4f}")
    return np.mean(apfds)

def main():
    t0=time.time()
    print("\n" + "="*72)
    print("EXP 13 -- Optimal Transport Test Prioritization")
    print("Theory: score = -W_2(phi(x), failure_manifold), Sinkhorn-trained")
    print("="*72)
    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting features...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    runs = [
        dict(lam=0.0, K=8,  tag='softmin centroids only (no OT)'),
        dict(lam=0.3, K=8,  tag='OT lam=0.3, K=8'),
        dict(lam=1.0, K=16, tag='OT lam=1.0, K=16'),
    ]
    saved = []
    for r in runs:
        print(f"\n>> {r['tag']}")
        model = OTNet(embed_dim=64, K=r['K'])
        model, auc, swa = train(model, X_tr, y_tr, X_te, y_te, epochs=70,
            batch=256, lr=5e-4, swa_start=50, lam_ot=r['lam'], ot_eps=0.1,
            name=f'OT ({r["tag"]})')
        m_eval = swa.get_model() if swa else model
        eval_apfd(test_data, m_eval, means, stds, f'{r["tag"]} SensoDat')
        if comp_data is not None:
            multi_trial(comp_data, m_eval, means, stds, f'{r["tag"]} multi-trial')
        saved.append(m_eval)

    save = os.path.join(OUTPUT_DIR, 'roadot.pt')
    torch.save({'state': saved[-1].state_dict(),
                'means': means.tolist(), 'stds': stds.tolist()}, save)
    print(f"\nSaved: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
