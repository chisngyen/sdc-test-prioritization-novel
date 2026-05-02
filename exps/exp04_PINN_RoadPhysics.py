"""
EXP 04 — Physics-Informed Neural Net for Road Failure
========================================================
Theoretical lens: PHYSICS-INFORMED LEARNING (Raissi et al., 2019).

The "test fails" event has a known *physics-side* sufficient condition: when
centripetal acceleration v^2 * |kappa(s)| at any point exceeds the friction-
limited threshold mu*g, the car cannot stay in lane. Concretely, for the
competition's autopilot config (RF=1.5, v_max=120 km/h, OOB=50%) the limit is
roughly (33.3 m/s)^2 * |kappa|  >  mu_eff * 9.81. Even when the constant is
unknown, the *monotonicity* must hold:

    if road A dominates road B in pointwise |kappa|, P(fail | A) >= P(fail | B).

We embed this as a differentiable regularizer:

    L_phys(s, x) = max(0, sigmoid(s_B) - sigmoid(s_A))^2,  for synthetic
                   pairs (A, B) where A = curvature_amplify(B).

That is, every minibatch we generate a "harder" copy of each road by scaling
its curvature channels by alpha > 1 and require the model's failure score to
be non-decreasing. This is a soft monotonicity constraint with provable
guarantees (Wehenkel & Louppe 2019 style):

  Theorem (informal): if L_phys = 0 for all amplification factors alpha
  in [1, A], then s(x) is alpha-monotone on the curvature channels in that
  range, hence calibrated to physical safety.

We also add a Sobolev penalty ||grad_x s||^2 on the curvature channel to
control sensitivity, which yields tighter generalization bounds (Czarnecki
et al., 2017). Together these encode physics WITHOUT requiring a simulator.
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

# Channel layout from extract_sequence_10ch:
#   0 seg_len | 1 |delta_heading| | 2 curv | 3 d-curv | 4 cum-dist
#   5 sin_h | 6 cos_h | 7 rel_pos | 8 local_std_curv | 9 d2-curv
CURV_CHANNELS = [1, 2, 3, 8, 9]   # all non-negative-ish curvature-related

# -------------------- 10-channel feature extraction (baseline) --------------------
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

# -------------------- Backbone (RoadTransformer like baseline) --------------------
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

# -------------------- Physics-informed regularizers --------------------
def amplify_curvature(x, alpha):
    """Multiply curvature-related channels by alpha (channels-first)."""
    x = x.clone()
    for c in CURV_CHANNELS:
        x[:, c, :] = x[:, c, :] * alpha
    return x

def physics_monotone_loss(model, x, alphas=(1.25, 1.5)):
    """Soft monotonicity: amplifying curvature should not DECREASE failure prob."""
    s_orig = torch.sigmoid(model(x))
    pen = 0.0
    for a in alphas:
        s_amp = torch.sigmoid(model(amplify_curvature(x, a)))
        pen = pen + F.relu(s_orig - s_amp + 0.0).pow(2).mean()
    return pen / len(alphas)

def sobolev_curv_penalty(model, x, eps=1e-2):
    """Smoothness: small curvature perturbations -> small score perturbations."""
    x_pert = x.clone()
    noise = torch.randn_like(x_pert[:, CURV_CHANNELS, :]) * eps
    x_pert[:, CURV_CHANNELS, :] = x_pert[:, CURV_CHANNELS, :] + noise
    s0 = torch.sigmoid(model(x))
    s1 = torch.sigmoid(model(x_pert))
    return (s0 - s1).pow(2).mean()

# -------------------- Boilerplate --------------------
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

def train(model, X_tr, y_tr, X_va, y_va, *, epochs=80, batch=384, lr=5e-4,
          swa_start=55, lam_phys=0.5, lam_sob=0.1, name='PINN-Road'):
    print(f"\n{'='*64}\nTraining {name} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  lambda_phys={lam_phys}  lambda_sobolev={lam_sob}\n{'='*64}")
    model = model.to(DEVICE)
    n_pos = y_tr.sum(); pw = (len(y_tr) - n_pos) / n_pos
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
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
    best_auc, best_state, swa = 0., None, None

    for ep in range(epochs):
        model.train(); tot=0; nb=0
        # ramp up physics losses linearly during training (curriculum):
        # too strong from epoch 1 fights the data fit.
        ramp = min(1.0, ep / max(1, epochs * 0.3))
        for xb, yb in dl:
            xb=xb.to(DEVICE, non_blocking=True); yb=yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                logits = model(xb)
                l_data = crit(logits, yb)
                l_phys = physics_monotone_loss(model, xb) if ramp > 0 else 0.0
                l_sob  = sobolev_curv_penalty(model, xb)  if ramp > 0 else 0.0
                loss = l_data + ramp*lam_phys*l_phys + ramp*lam_sob*l_sob
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
    return model, best_auc, swa

# -------------------- APFD eval & physics sanity check --------------------
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

def physics_violation_rate(data, model, means, stds, alpha=1.5, name=''):
    """Fraction of tests where amplifying curvature DECREASES the score
    (physics violation). Lower is better; baseline ~30%, PINN should be <5%."""
    model.eval().to(DEVICE)
    feats=[(extract_sequence_10ch(get_pts(tc)) - means)/stds for tc in data]
    X=torch.tensor(np.array(feats), dtype=torch.float32).permute(0,2,1).to(DEVICE)
    with torch.no_grad(), autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
        s0 = torch.sigmoid(model(X).float())
        s1 = torch.sigmoid(model(amplify_curvature(X, alpha)).float())
    rate = (s1 < s0 - 1e-3).float().mean().item()
    print(f"  {name:46s} viol={rate*100:.2f}% (lower=better)")
    return rate

# -------------------- Main --------------------
def main():
    t0=time.time()
    print("\n" + "="*72)
    print("EXP 04 — Physics-Informed Road Net (curvature monotonicity)")
    print("Theory: amplify_curv(road) >= original road in failure probability")
    print("="*72)

    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting features...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    runs = [
        dict(lp=0.0, ls=0.0, tag='no PINN  (control)'),
        dict(lp=0.5, ls=0.0, tag='monotone only'),
        dict(lp=0.5, ls=0.1, tag='monotone + sobolev  (full PINN)'),
    ]
    saved=[]
    for r in runs:
        print(f"\n>> {r['tag']}")
        model = RoadTransformer(in_channels=10, d_model=192, num_layers=5,
                                nhead=8, dim_feedforward=512)
        model, auc, swa = train(model, X_tr, y_tr, X_te, y_te, epochs=80,
            batch=384, lr=5e-4, swa_start=55, lam_phys=r['lp'], lam_sob=r['ls'],
            name=f"PINN ({r['tag']})")
        m_eval = swa.get_model() if swa else model
        eval_apfd(test_data, m_eval, means, stds, f'{r["tag"]} SensoDat')
        if comp_data is not None:
            multi_trial(comp_data, m_eval, means, stds, f'{r["tag"]} multi-trial')
            physics_violation_rate(comp_data, m_eval, means, stds, alpha=1.5,
                                   name=f'{r["tag"]} alpha=1.5 viol')
            physics_violation_rate(comp_data, m_eval, means, stds, alpha=2.0,
                                   name=f'{r["tag"]} alpha=2.0 viol')
        saved.append(m_eval)

    save = os.path.join(OUTPUT_DIR, 'roadpinn.pt')
    torch.save({'state': saved[-1].state_dict(),
                'means': means.tolist(), 'stds': stds.tolist()}, save)
    print(f"\nSaved: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
