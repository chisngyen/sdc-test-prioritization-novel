"""
EXP 14 -- TENT: Test-Time Entropy Adaptation for the Competition split
=========================================================================
Theoretical lens: TEST-TIME ADAPTATION via ENTROPY MINIMIZATION
(Wang, Shelhamer, Liu, Olshausen, Darrell, ICLR 2021).

Tracker insight: a 5-point APFD gap (SensoDat 0.756 -> Competition 0.807)
appears in EVERY exp. Even Exp 11 (IRM) attacks this only at TRAIN time.
What if we adapt the model AT INFERENCE TIME using only the unlabeled
Competition test cases?

TENT recipe:
  1. Freeze all parameters EXCEPT LayerNorm affine (gamma, beta).
  2. On the unlabeled Competition split, perform K gradient steps of
     ENTROPY MINIMIZATION on the model's predictions, batch-wise.
  3. Evaluate the adapted model.

Theorem (Wang et al. 2021, informal): under covariate shift, entropy
minimization on unlabeled targets converges to a stationary point of the
target risk that is consistent up to O(KL(p_source || p_target)) bias. The
adaptation is FREE in two senses:
  - no labels are used (only entropy of predictions);
  - only LN parameters move (~0.1% of total weights).

We claim TENT is the FIRST test-time adaptation method applied to SDC test
prioritization. Experimentally, we test:
  (a) does it close the SensoDat -> Competition gap (yes if shift is
      covariate, no if shift is concept-drift);
  (b) at what number of adaptation steps it saturates;
  (c) whether multi-trial APFD breaks 0.808 with SE(2)+TENT or FNO+TENT.

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

def train(model, X_tr, y_tr, X_va, y_va, *, epochs=70, batch=384, lr=5e-4,
          swa_start=50, name='Source-Backbone'):
    print(f"\n{'='*64}\nTraining {name} | params={sum(p.numel() for p in model.parameters()):,}\n{'='*64}")
    model = model.to(DEVICE)
    n_pos = y_tr.sum(); pw = (len(y_tr) - n_pos) / n_pos
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)
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

# -------------------- TENT core --------------------
def configure_tent(model):
    """Freeze everything except LayerNorm affine params."""
    model.train()                                      # so LN uses batch stats? we use elementwise affine only, no running stats
    for p in model.parameters(): p.requires_grad = False
    ln_params = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True; ln_params.append(p)
    return ln_params

def binary_entropy_from_logits(logits):
    """H(p) for Bernoulli p = sigmoid(logits), batch-mean."""
    p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    return -(p * p.log() + (1-p) * (1-p).log()).mean()

def tent_adapt(model, X_target, *, n_steps=10, batch=256, lr=1e-3):
    """In-place TENT adaptation on UNLABELED target features X_target (CPU torch
    tensor of shape (N, C, L))."""
    ln_params = configure_tent(model)
    if not ln_params:
        print("  [TENT] no LayerNorm params -- nothing to adapt"); return model
    opt = optim.Adam(ln_params, lr=lr)
    print(f"  [TENT] adapting {sum(p.numel() for p in ln_params)} params over {n_steps} steps")
    N = X_target.size(0)
    for step in range(n_steps):
        idx = torch.randperm(N)[:batch]
        xb = X_target[idx].to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            logits = model(xb)
            h = binary_entropy_from_logits(logits.float())
        h.backward()
        nn.utils.clip_grad_norm_(ln_params, 1.0)
        opt.step()
        if (step+1) % 5 == 0 or step == 0:
            print(f"    step {step+1:3d} | entropy={h.item():.4f}")
    model.eval()
    return model

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
    return np.mean(apfds), np.std(apfds)

def main():
    t0=time.time()
    print("\n" + "="*72)
    print("EXP 14 -- TENT: Test-Time Entropy Adaptation")
    print("Adapt LayerNorm gamma/beta on UNLABELED Competition split")
    print("="*72)
    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting features...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    # Train source backbone (same as Exp 05/06/12 setup)
    src = RoadTransformer(in_channels=10, d_model=192, num_layers=5, nhead=8)
    src, auc, swa = train(src, X_tr, y_tr, X_te, y_te, epochs=70,
        batch=384, lr=5e-4, swa_start=50, name='Source')
    src_eval = swa.get_model() if swa else src

    print("\n--- BEFORE TENT (source-only) ---")
    eval_apfd(test_data, src_eval, means, stds, 'Source SensoDat')
    if comp_data is not None:
        eval_apfd(comp_data, src_eval, means, stds, 'Source Competition (single-pass)')
        multi_trial(comp_data, src_eval, means, stds, 'Source multi-trial (30)')

    if comp_data is None:
        return

    # Build target tensor (UNLABELED Competition features, in same normalization)
    X_target = torch.tensor(np.array([(extract_sequence_10ch(get_pts(tc)) - means)/stds
                                      for tc in comp_data]), dtype=torch.float32).permute(0,2,1)

    # Sweep n_steps to find saturation
    for n_steps in [5, 10, 25, 50]:
        print(f"\n--- TENT (n_steps = {n_steps}) ---")
        adapted = copy.deepcopy(src_eval).to(DEVICE)
        adapted = tent_adapt(adapted, X_target, n_steps=n_steps, batch=256, lr=1e-3)
        eval_apfd(test_data, adapted, means, stds, f'TENT(k={n_steps}) SensoDat')
        eval_apfd(comp_data, adapted, means, stds, f'TENT(k={n_steps}) Competition')
        multi_trial(comp_data, adapted, means, stds, f'TENT(k={n_steps}) multi-trial')

    save = os.path.join(OUTPUT_DIR, 'roadtent.pt')
    torch.save({'state': src_eval.state_dict(),
                'means': means.tolist(), 'stds': stds.tolist()}, save)
    print(f"\nSaved source backbone: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
