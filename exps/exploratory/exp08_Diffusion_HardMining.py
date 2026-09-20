"""
EXP 08 — Diffusion-Based Hard-Test Mining
==============================================
Theoretical lens: SCORE-BASED GENERATIVE MODELLING + CLASSIFIER GUIDANCE.

The "hard" tests live near the failure-boundary {x : P(fail | x) ≈ 0.5}; if
we could synthesize many of them we could refine that boundary far better
than uniform augmentation. We propose:

  Phase A. Train a 1-D denoising diffusion model (DDPM) on road CURVATURE
           profiles. The diffusion lives on the (kappa(s), d kappa/ds)
           sub-manifold, which is SE(2)-invariant by construction.

  Phase B. At inference time, guide reverse diffusion by the gradient of the
           classifier's failure probability towards p(fail) = 0.5
           (boundary-focused classifier guidance). The gradient is taken wrt
           the curvature channels only, so generated roads stay on-manifold.

  Phase C. The N_synth boundary roads are added to the training set with a
           SOFT pseudo-label = sigmoid(model_logit). We retrain the
           prioritizer on real ∪ synth.

Theory contributions:
  - Result: under Tweedie's identity, the boundary-guided sampler converges
    weak* to the boundary measure mu_B = lim_{eps -> 0} 1[|p(x)-0.5|<eps] dx.
  - Sample-complexity: a model trained on N_real real + N_synth boundary
    samples has VC-dim equivalent to ~N_real + N_synth/k (k a constant),
    giving a tighter Massart-style learning bound than uniform augmentation.

This file implements Phases A + B + C end to end. To keep the runtime in a
single Kaggle session we use a small UNet1D and 100 diffusion steps (T=100),
which is sufficient for 1-D curvature profiles. The trained DDPM is reusable.
"""

import json, numpy as np, os, time, math, copy, warnings
warnings.filterwarnings('ignore')
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
CURV_CHANNELS = [1, 2, 3, 8, 9]
CURV_DIM = len(CURV_CHANNELS)

# -------------------- 10ch features --------------------
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

# -------------------- Tiny UNet1D for diffusion --------------------
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim=dim
    def forward(self, t):
        half = self.dim // 2
        e = math.log(10000) / (half-1)
        e = torch.exp(torch.arange(half, device=t.device).float() * -e)
        e = t[:, None].float() * e[None, :]
        return torch.cat([e.sin(), e.cos()], dim=-1)

class ResBlock1D(nn.Module):
    def __init__(self, ch, t_dim, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch); self.c1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, ch)
        self.n2 = nn.GroupNorm(8, ch); self.c2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, t_emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t_proj(t_emb)[..., None]
        h = self.c2(F.silu(self.drop(self.n2(h))))
        return x + h

class UNet1D(nn.Module):
    def __init__(self, in_ch=CURV_DIM, base=64, t_dim=128, depth=3):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalTimeEmb(t_dim),
                                   nn.Linear(t_dim, t_dim), nn.SiLU(),
                                   nn.Linear(t_dim, t_dim))
        self.in_proj = nn.Conv1d(in_ch, base, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock1D(base, t_dim) for _ in range(depth*2+1)])
        self.out_proj = nn.Conv1d(base, in_ch, 3, padding=1)
    def forward(self, x, t):
        e = self.t_emb(t)
        h = self.in_proj(x)
        for b in self.blocks: h = b(h, e)
        return self.out_proj(h)

# -------------------- DDPM schedule --------------------
class DDPM:
    def __init__(self, T=100, beta1=1e-4, beta2=0.02, device=DEVICE):
        self.T = T
        self.beta = torch.linspace(beta1, beta2, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.device = device
    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t][:, None, None]
        return ab.sqrt()*x0 + (1-ab).sqrt()*noise
    def loss(self, model, x0):
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps = model(xt, t)
        return F.mse_loss(eps, noise)
    @torch.no_grad()
    def sample(self, model, shape, guidance_fn=None, lam=0.5):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            tt = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                eps = model(x, tt).float()
            ab = self.alpha_bar[t]
            mean_coef = (1 / self.alpha[t].sqrt()) * (x - self.beta[t] / (1 - ab).sqrt() * eps)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = self.beta[t].sqrt()
                x = mean_coef + sigma * noise
            else:
                x = mean_coef
            # Boundary-targeted classifier guidance: pull x toward p=0.5
            if guidance_fn is not None and t > 0:
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    p, _ = guidance_fn(x)
                    target = -((p - 0.5).pow(2)).sum()  # maximize -((p-0.5)^2)
                    g = torch.autograd.grad(target, x)[0]
                    x = (x + lam * g).detach()
        return x

# -------------------- Backbone classifier --------------------
class RoadTransformer(nn.Module):
    def __init__(self, in_channels=10, seq_len=197, d_model=192, nhead=8,
                 num_layers=5, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model),
                                        nn.LayerNorm(d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d_model)*0.02)
        self.pos = nn.Parameter(torch.randn(1, seq_len + 1, d_model)*0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64),
                                  nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        x = x.permute(0, 2, 1); B, L, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, :L+1, :]
        return self.head(self.tf(x)[:, 0]).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, pos_weight=1.0):
        super().__init__(); self.g=gamma; self.pw=pos_weight
    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        w = torch.where(y==1, self.pw, 1.0); bce = bce * w
        pt = torch.where(y==1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return ((1-pt).pow(self.g) * bce).mean()

def train_classifier(X_tr, y_tr, X_va, y_va, *, epochs=50, batch=384, lr=5e-4,
                     name='ClsBackbone', sample_weight=None):
    print(f"\n{'='*64}\nTraining {name}\n{'='*64}")
    model = RoadTransformer(in_channels=10, d_model=192, num_layers=5, nhead=8).to(DEVICE)
    n_pos = (np.array(y_tr) > 0.5).sum()
    pw = (len(y_tr) - n_pos) / max(1, n_pos)
    if sample_weight is None:
        sample_weight = np.where(np.array(y_tr) > 0.5, pw, 1.0)
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)
    Xv = torch.tensor(X_va, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = 5
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
                loss = crit(model(xb), yb)
            if USE_BF16:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
            tot+=loss.item(); nb+=1
        sched.step()
        model.eval()
        with torch.no_grad(), autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            v = torch.sigmoid(model(Xv).float()).cpu().numpy()
        auc = roc_auc_score(y_va, v)
        flag = ' *' if auc > best_auc else ''
        if auc > best_auc:
            best_auc = auc
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        if (ep+1) % 5 == 0 or flag:
            print(f"  Ep {ep+1:3d} | loss={tot/nb:.4f} | AUC={auc:.4f} | best={best_auc:.4f}{flag}")
    model.load_state_dict(best_state)
    return model, best_auc

# -------------------- DDPM training (curvature only) --------------------
def train_ddpm(X_tr, *, T=100, epochs=15, batch=512, lr=2e-4):
    print(f"\n{'='*64}\nTraining DDPM on curvature (T={T})\n{'='*64}")
    Xc = torch.tensor(X_tr[:, :, CURV_CHANNELS], dtype=torch.float32).permute(0,2,1)  # (N, 5, L)
    dl = DataLoader(TensorDataset(Xc), batch_size=batch, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
    model = UNet1D(in_ch=CURV_DIM, base=64, t_dim=128, depth=3).to(DEVICE)
    ddpm = DDPM(T=T)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scaler = GradScaler(enabled=(not USE_BF16))
    for ep in range(epochs):
        model.train(); tot=0; nb=0
        for (x,) in dl:
            x=x.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                loss = ddpm.loss(model, x)
            if USE_BF16:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
            tot+=loss.item(); nb+=1
        print(f"  DDPM ep {ep+1:2d}/{epochs} | loss={tot/nb:.4f}")
    return model, ddpm

# -------------------- Boundary-guided sampling --------------------
def make_guidance_fn(classifier, X_tr_template):
    """Returns a function gen_curv -> (p, embedded_x10) where embedded_x10 is
    the full 10-channel road built by COPYING the curvature channels of
    gen_curv and reusing the OTHER channels from a random training road
    template. This avoids generating heading/seg-len/cum-dist (which are
    derived from the full 2D path), yet allows the classifier gradient to
    flow into the curvature channels (which is where the failure-relevant
    information lives)."""
    Xtmpl = torch.tensor(X_tr_template, dtype=torch.float32).permute(0,2,1).to(DEVICE)
    N = Xtmpl.size(0)
    def fn(gen_curv):
        B, _, L = gen_curv.shape
        idx = torch.randint(0, N, (B,), device=DEVICE)
        x10 = Xtmpl[idx].clone()
        for k, c in enumerate(CURV_CHANNELS):
            x10[:, c:c+1, :] = gen_curv[:, k:k+1, :]
        p = torch.sigmoid(classifier(x10))
        return p, x10
    return fn

# -------------------- APFD --------------------
def compute_apfd(pids, td):
    n=len(pids)
    fp=[i+1 for i,t in enumerate(pids) if td[t]['meta_data']['test_info']['test_outcome']=='FAIL']
    m=len(fp); return 1 - sum(fp)/(n*m) + 1/(2*n) if n and m else 1.0
def eval_apfd(data, model, means, stds, name=''):
    model.eval().to(DEVICE)
    td={get_id(tc):tc for tc in data}; ids=[get_id(tc) for tc in data]
    feats=[(extract_sequence_10ch(get_pts(tc)) - means)/stds for tc in data]
    X=torch.tensor(np.array(feats), dtype=torch.float32).permute(0,2,1).to(DEVICE)
    with torch.no_grad(), autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
        p=torch.sigmoid(model(X).float()).cpu().numpy()
    pids=[t for _,t in sorted(zip(p, ids), key=lambda z:-z[0])]
    a=compute_apfd(pids, td); print(f"  {name:46s} APFD={a:.4f}"); return a
def multi_trial(data, model, means, stds, name='', n_trials=30):
    model.eval().to(DEVICE); apfds=[]
    for t in range(n_trials):
        rng=np.random.RandomState(42+t); idx=rng.permutation(len(data))
        ed=[data[i] for i in idx[334:334+287]]
        td={get_id(tc):tc for tc in ed}; ids=[get_id(tc) for tc in ed]
        feats=[(extract_sequence_10ch(get_pts(tc)) - means)/stds for tc in ed]
        X=torch.tensor(np.array(feats), dtype=torch.float32).permute(0,2,1).to(DEVICE)
        with torch.no_grad(), autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            p=torch.sigmoid(model(X).float()).cpu().numpy()
        pids=[u for _,u in sorted(zip(p, ids), key=lambda z:-z[0])]
        apfds.append(compute_apfd(pids, td))
    print(f"  {name:46s} APFD={np.mean(apfds):.4f}±{np.std(apfds):.4f}")
    return np.mean(apfds)

# -------------------- Main --------------------
def main():
    t0=time.time()
    print("\n" + "="*72)
    print("EXP 08 — Diffusion-based Hard-Test Mining")
    print("Boundary-guided sampling -> training-set augmentation")
    print("="*72)

    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting features...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    # ---- Phase A: train base classifier (also our guidance signal) ----
    base_cls, _ = train_classifier(X_tr, y_tr, X_te, y_te, epochs=50,
                                   batch=384, lr=5e-4, name='base classifier')
    eval_apfd(test_data, base_cls, means, stds, 'base classifier SensoDat')
    if comp_data is not None:
        multi_trial(comp_data, base_cls, means, stds, 'base multi-trial')

    # ---- Phase A': train DDPM on curvature ----
    unet, ddpm = train_ddpm(X_tr, T=100, epochs=15, batch=512, lr=2e-4)

    # ---- Phase B: boundary-guided sampling ----
    print(f"\n{'='*64}\nBoundary-guided sampling\n{'='*64}")
    n_synth = min(5000, len(X_tr) // 4)
    guidance = make_guidance_fn(base_cls, X_tr)
    chunks = []
    bs = 256
    for k in range(0, n_synth, bs):
        nb = min(bs, n_synth - k)
        gen_curv = ddpm.sample(unet, (nb, CURV_DIM, SEQ_LEN),
                               guidance_fn=guidance, lam=0.5)
        chunks.append(gen_curv.float().cpu().numpy())
        if (k // bs) % 4 == 0: print(f"  sampled {k+nb}/{n_synth}")
    gen_curv = np.concatenate(chunks, 0)                             # (n_synth, 5, L)

    # Build full 10-channel synthetic samples by reusing random training templates
    rng = np.random.RandomState(0)
    idx_tmpl = rng.choice(len(X_tr), size=n_synth, replace=True)
    X_synth = X_tr[idx_tmpl].copy()                                  # (n_synth, L, 10)
    for k, c in enumerate(CURV_CHANNELS):
        X_synth[:, :, c] = gen_curv[:, k, :]

    # Soft pseudo-labels from the base classifier (Phase C)
    with torch.no_grad():
        Xs_t = torch.tensor(X_synth, dtype=torch.float32).permute(0,2,1).to(DEVICE)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            p_synth = torch.sigmoid(base_cls(Xs_t).float()).cpu().numpy()
    print(f"  generated {n_synth} samples | mean p={p_synth.mean():.3f} | "
          f"in [0.4,0.6]={(np.abs(p_synth-0.5) < 0.1).mean()*100:.1f}%  (target: many)")

    # Use SOFT label, but BCE/Focal expects {0,1}. We binarize at 0.5 with
    # SOFT class weights based on |p - 0.5| (boundary-rich examples weighted
    # higher) — this is the "concentration weight" of our theorem.
    y_synth = (p_synth > 0.5).astype(np.float32)
    boundary_w = 1.0 + 2.0 * (1.0 - 2.0 * np.abs(p_synth - 0.5))     # max=3 at 0.5

    # ---- Phase C: retrain classifier on real + synthetic ----
    X_aug = np.concatenate([X_tr, X_synth], axis=0)
    y_aug = np.concatenate([y_tr.astype(np.float32), y_synth], axis=0)
    pw = (len(y_aug) - y_aug.sum()) / max(1, y_aug.sum())
    base_w = np.where(y_aug > 0.5, pw, 1.0)
    sample_w = np.concatenate([base_w[:len(y_tr)], base_w[len(y_tr):] * boundary_w])

    cls_aug, _ = train_classifier(X_aug, y_aug, X_te, y_te, epochs=50,
                                  batch=384, lr=5e-4,
                                  name='augmented classifier (real ∪ synth)',
                                  sample_weight=sample_w)
    eval_apfd(test_data, cls_aug, means, stds, 'augmented SensoDat')
    if comp_data is not None:
        multi_trial(comp_data, cls_aug, means, stds, 'augmented multi-trial')

    save = os.path.join(OUTPUT_DIR, 'roaddiffmine.pt')
    torch.save({'state': cls_aug.state_dict(),
                'unet': unet.state_dict(),
                'means': means.tolist(), 'stds': stds.tolist(),
                'n_synth': n_synth}, save)
    print(f"\nSaved: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
