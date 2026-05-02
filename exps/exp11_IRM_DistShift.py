"""
EXP 11 -- Invariant Risk Minimization for SensoDat -> Competition Gap
=======================================================================
Theoretical lens: INVARIANT RISK MINIMIZATION (Arjovsky, Bottou,
Gulrajani, Lopez-Paz, 2019).

Tracker insight: a 5-point APFD gap between SensoDat-test (0.756) and
Competition (0.807) appears in EVERY exp we have run -- even the strongest
SE(2)-equivariant model. Rotation/resolution invariance does NOT close this
gap, because the gap is about *failure-rate distribution shift*, not
coordinate-frame shift.

This is the FIRST distribution-shift attack on SDC test prioritization.

Idea: treat each LATENT failure mode (sharp turn, long straight, S-bend, ...)
as a separate ENVIRONMENT, then use IRM to find a representation phi such
that the OPTIMAL classifier on top of phi is the same across environments.

Concretely:
  L_IRM(phi, w) = sum_e R^e(w * phi)  +  lambda * sum_e || grad_w R^e(w * phi) ||^2  at w=1

  where R^e is the focal/BCE loss on environment e, and the gradient
  penalty enforces that w=1 is optimal in EVERY environment simultaneously.
  Theorem (Arjovsky 2019): under the bias-shift model, the IRM solution
  generalises to unseen environments at rate independent of environment
  count.

Environment construction here: cluster training roads into K=4 groups by
the (mean curvature, max curvature, total length) statistics via k-means.
This gives 4 latent "failure modes" without using any test-time information.

Implementation note: we use IRMv1 (linear-classifier-on-features form).
For the reported gap, we eval on the held-out Competition split (which is
a 5th, *unseen* environment) -- which is the OOD generalisation test.

Hardware: Kaggle RTX 6000 Pro Blackwell (96 GB), bf16.
"""

import json, numpy as np, os, time, math, copy, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.cluster import KMeans
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
    X,y,stats=[],[],[]
    for i,tc in enumerate(data):
        feat = extract_sequence_10ch(get_pts(tc))
        X.append(feat); y.append(1 if is_fail(tc) else 0)
        # 3-D summary statistic for environment clustering: (mean |curv|, max |curv|, total length)
        stats.append([feat[:,2].mean(), feat[:,2].max(), feat[:,0].sum()])
        if (i+1)%5000==0: print(f"    {i+1}/{len(data)}...")
    return np.array(X), np.array(y), np.array(stats, dtype=np.float32)

# -------------------- Backbone with explicit feature/classifier split --------------------
class IRMTransformer(nn.Module):
    """phi : input -> feat (d_model). w : feat -> logit (single linear).
    IRM penalizes the gradient of R^e at w=1 in the OUTPUT scale (a scalar
    multiplier on the linear classifier)."""
    def __init__(self, in_channels=10, seq_len=197, d_model=160, nhead=8,
                 num_layers=5, dim_feedforward=384, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model),
                                        nn.LayerNorm(d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(1, 1, d_model)*0.02)
        self.pos = nn.Parameter(torch.randn(1, seq_len + 1, d_model)*0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.feat_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)        # the "w" of IRMv1
    def features(self, x):
        x = x.permute(0, 2, 1); B, L, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, :L+1, :]
        return self.feat_norm(self.tf(x)[:, 0])
    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)

# -------------------- IRM penalty (IRMv1, Arjovsky 2019, eq. 5) --------------------
def irm_penalty(logits, y, pos_weight):
    """Gradient of weighted BCE at scale w=1, squared. Cheapest IRMv1 form."""
    w = torch.tensor(1.0, device=logits.device, requires_grad=True)
    weight = torch.where(y == 1, pos_weight, 1.0)
    bce = (F.binary_cross_entropy_with_logits(logits * w, y, reduction='none') * weight).mean()
    g = torch.autograd.grad(bce, w, create_graph=True)[0]
    return g.pow(2)

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

def train(model, X_tr, y_tr, env_tr, X_va, y_va, *, epochs=80, batch=256,
          lr=5e-4, swa_start=55, lam_irm=1.0, irm_warmup=20,
          name='IRM-Transformer'):
    print(f"\n{'='*64}\nTraining {name} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  IRM lambda={lam_irm}  warmup={irm_warmup} epochs (ERM only before)\n{'='*64}")
    model = model.to(DEVICE)
    n_pos = y_tr.sum(); pw_val = (len(y_tr) - n_pos) / n_pos
    pw = torch.tensor(float(pw_val), device=DEVICE)

    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    et = torch.tensor(env_tr, dtype=torch.long)
    Xv = torch.tensor(X_va, dtype=torch.float32).permute(0,2,1)

    n_envs = int(env_tr.max()) + 1
    env_idx = [np.where(env_tr == e)[0] for e in range(n_envs)]
    print(f"  environments: {n_envs}, sizes = {[len(s) for s in env_idx]}")

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = 5
    sched = optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/warm if e<warm
        else max(0.01, 0.5*(1 + math.cos(math.pi*(e-warm)/max(1, epochs-warm)))))
    crit = FocalLoss(gamma=1.5, pos_weight=pw_val)
    scaler = GradScaler(enabled=(not USE_BF16))
    best_auc, best_state, swa = 0., None, None

    per_env_batch = max(batch // n_envs, 32)
    n_iter = len(y_tr) // batch

    for ep in range(epochs):
        model.train(); tot=0; nb=0; sum_irm=0
        # IRM penalty ramp from 0 to lam_irm over irm_warmup epochs
        eff_lam = lam_irm * min(1.0, max(0.0, (ep - irm_warmup/2) / max(1, irm_warmup/2)))
        for _ in range(n_iter):
            # Build a batch with equal slots per env (stratified by env)
            xs, ys, es = [], [], []
            for e in range(n_envs):
                k = min(per_env_batch, len(env_idx[e]))
                idx = np.random.choice(env_idx[e], size=k, replace=(len(env_idx[e]) < k))
                xs.append(Xt[idx]); ys.append(yt[idx]); es.append(np.full(k, e))
            xb = torch.cat(xs, dim=0).to(DEVICE, non_blocking=True)
            yb = torch.cat(ys, dim=0).to(DEVICE, non_blocking=True)
            eb = torch.tensor(np.concatenate(es), dtype=torch.long, device=DEVICE)

            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                logits = model(xb)
                # ERM: focal loss over the batch
                erm = crit(logits, yb)
                # IRM v1 penalty (one per environment) requires create_graph=True
                # which means we go OUTSIDE bf16 autocast for the penalty.
            # IRM penalty in fp32:
            penalty = torch.zeros((), device=DEVICE)
            if eff_lam > 0:
                logits_fp = logits.float()
                for e in range(n_envs):
                    msk = (eb == e)
                    if msk.sum() < 4: continue
                    penalty = penalty + irm_penalty(logits_fp[msk], yb[msk].float(), pw)
                penalty = penalty / n_envs
            loss = erm + eff_lam * penalty
            if USE_BF16:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
            tot+=loss.item(); nb+=1
            sum_irm += float(penalty.detach().item())
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
            print(f"  Ep {ep+1:3d} | loss={tot/nb:.4f} | irm_pen={sum_irm/max(nb,1):.4e} | "
                  f"lam_eff={eff_lam:.3f} | AUC={auc:.4f} | best={best_auc:.4f}{flag}")
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
    print("EXP 11 -- Invariant Risk Minimization for SensoDat -> Competition gap")
    print("Theory: train on K=4 latent failure-mode environments via IRMv1")
    print("="*72)

    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting features and per-road statistics...")
    X_tr,y_tr,stats_tr=prepare_data(train_data)
    X_te,y_te,_=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    # Build environments via k-means on the 3-D summary statistics.
    # Standardize stats for clustering.
    K = 4
    s_mu = stats_tr.mean(axis=0); s_sd = stats_tr.std(axis=0) + 1e-8
    stats_norm = (stats_tr - s_mu) / s_sd
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(stats_norm)
    env_tr = km.labels_.astype(np.int64)
    print(f"  k-means environments (K={K}):")
    for e in range(K):
        msk = env_tr == e
        nf = y_tr[msk].sum()
        print(f"    env {e}: n={msk.sum()}, fail-rate={nf/msk.sum():.3f}")

    # Run two configs: ERM-only (lam=0) and IRM (lam=1.0)
    runs = [
        dict(lam=0.0,  tag='ERM-only (control)'),
        dict(lam=1.0,  tag='IRMv1 (lam=1.0)'),
        dict(lam=10.0, tag='IRMv1 (lam=10.0)'),
    ]
    saved=[]
    for r in runs:
        print(f"\n>> {r['tag']}")
        model = IRMTransformer(in_channels=10, d_model=160, num_layers=5,
                               nhead=8, dim_feedforward=384)
        model, auc, swa = train(model, X_tr, y_tr, env_tr, X_te, y_te,
            epochs=80, batch=256, lr=5e-4, swa_start=55,
            lam_irm=r['lam'], irm_warmup=20, name=f"{r['tag']}")
        m_eval = swa.get_model() if swa else model
        eval_apfd(test_data, m_eval, means, stds, f'{r["tag"]} SensoDat')
        if comp_data is not None:
            multi_trial(comp_data, m_eval, means, stds, f'{r["tag"]} multi-trial')
        saved.append(m_eval)

    save = os.path.join(OUTPUT_DIR, 'roadirm.pt')
    torch.save({'state': saved[-1].state_dict(),
                'means': means.tolist(), 'stds': stds.tolist(),
                'kmeans_centers': km.cluster_centers_.tolist(),
                'stats_mu': s_mu.tolist(), 'stats_sd': s_sd.tolist()}, save)
    print(f"\nSaved: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
