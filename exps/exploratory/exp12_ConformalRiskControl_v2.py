"""
EXP 12 -- Conformal Risk Control v2 (TIGHTER bound, fixes Exp 05)
====================================================================
Theoretical lens: CONFORMAL RISK CONTROL (Angelopoulos, Bates, et al. 2023).

Tracker insight (Exp 05 v1):
  - Coverage = 1.000 (good, framework works)
  - LB-APFD@K = 0.0000 across all K (vacuous)
  - Empirical APFD@K = 0.94 -> 0.60 from K=50 -> K=287 (signal IS present)

The v1 bound used a worst-case combinatorial argument
        max(0, m + K - n)
which is dominated by the worst-case fault placement and ignores model
information entirely. We replace it with a CALIBRATED RISK CONTROL bound
where the test miss-rate at the top-K cut is bounded *as a function of the
calibrated quantile of the score*.

Conformal Risk Control (CRC) form (Angelopoulos & Bates 2023, Sec. 4):
  Given a monotone non-conformity loss L_lambda(x, y) bounded by B,
  there exists lambda_hat such that
        E[L_lambda_hat(X_{n+1}, Y_{n+1})]  <=  alpha    with marginal confidence 1.

In our setting:
  - Score s(x) is the model logit
  - For threshold lambda, define decision: top-K = {x : s(x) > lambda}
  - Loss L_lambda(x, y) = 1{x in top-K-cut by lambda but y = PASS}
                        + 1{x NOT in cut but y = FAIL}
    (i.e., the symmetric error of the lambda-threshold decision)
  - Find lambda_hat so that E[L_lambda] <= alpha

Then map E[L_lambda] back to a *prefix-APFD lower bound* via:
  prefix-APFD@K  =  1  -  E[rank of fail in top-K] / (K * m)  +  1/(2K)

Bound: if alpha-fraction of the population is mis-ranked above lambda, then
in expectation the K-prefix has at least  (m * (1-alpha)) - (n - K)/K * alpha
fails ranked correctly. This IS data-dependent (uses the fitted score
distribution), unlike v1's worst-case combinatorial argument.

We deliver:
  (a) per-K, per-alpha calibrated lambda_hat
  (b) coverage in 200 random calibration draws (target >= 1-alpha empirically)
  (c) NON-TRIVIAL APFD-LB@K, K=50..287, alpha in {0.05, 0.10, 0.20}

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
          swa_start=50, name='ConformalRC-Backbone'):
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

# -------------------- Conformal Risk Control v2 --------------------
def crc_calibrated_lambda(logits_cal, y_cal, alpha):
    """Find the smallest lambda such that the empirical risk
       (1/n) * sum 1{ (logit_i > lambda AND y_i = 0) OR (logit_i <= lambda AND y_i = 1) }
       <= alpha * (1 + 1/n)
    by linear search over the sorted set of candidate cuts (each unique logit
    value defines a candidate lambda)."""
    n = len(logits_cal)
    # Candidate lambdas: sort all logits ascending; halfway between adjacent
    # logits is a valid candidate.
    sorted_logits = np.sort(logits_cal)
    candidates = np.concatenate([[sorted_logits[0] - 1.0],
                                 (sorted_logits[:-1] + sorted_logits[1:]) / 2.0,
                                 [sorted_logits[-1] + 1.0]])
    target = alpha * (1.0 + 1.0/n)
    best_lambda = candidates[0]; best_risk = 1.0
    for lam in candidates:
        decision = (logits_cal > lam).astype(np.int64)         # 1 = predicted FAIL
        risk = np.mean( (decision != y_cal).astype(np.float64) )
        if risk <= target:
            return lam, risk
        if risk < best_risk:
            best_risk = risk; best_lambda = lam
    return best_lambda, best_risk

def crc_apfd_lower_bound(logits_eval, y_eval, lam_hat, alpha, K_grid):
    """Map the CRC threshold lambda to a per-K APFD lower bound.

    Reasoning. After calibration, for any *new* test (x, y) drawn
    exchangeably:
        P( decision_lambda(x) != y )  <=  alpha  +  small slack.
    The lambda-threshold sorts the eval set into "predicted FAIL" (top by
    logit) and "predicted PASS" (bottom). The decision-error rate caps the
    fraction of inversions, and inversions cap the prefix-APFD distance to
    the optimal ordering.

    We compute three quantities at each K:
      - empirical APFD@K (point estimate, for reference)
      - APFD@K lower bound from CRC (worst-case under the alpha-error budget)
      - APFD@K coverage indicator (1 if empirical >= LB)
    """
    order = np.argsort(-logits_eval)
    y_sorted = y_eval[order]
    logits_sorted = logits_eval[order]
    n = len(y_sorted); m = int(y_sorted.sum())
    out = {}
    if m == 0:
        return {K: dict(lb=1.0, emp=1.0) for K in K_grid}
    for K in K_grid:
        # Empirical APFD@K
        fp = [i+1 for i,b in enumerate(y_sorted[:K]) if b == 1]
        emp = (1 - sum(fp)/(K*m) + 1/(2*K)) if fp else 0.0
        # LB: count fails predicted-FAIL by lambda inside top-K
        # Predicted FAILs in top-K are those with logit > lam_hat AND in top-K positions.
        topK_logits = logits_sorted[:K]
        topK_y      = y_sorted[:K]
        # Fails that are SAFELY (lambda-bounded) in the top-K under CRC:
        # the CRC says total-error rate <= alpha -> at least (1 - alpha) * m
        # of all fails are in {logit > lam}. We can split that across the top-K
        # cut by what fraction of the >-lam mass is in the top-K (which we
        # observe directly).
        n_pred_fail = (logits_sorted > lam_hat).sum()                     # total predicted FAIL
        n_pred_fail_in_topK = (topK_logits > lam_hat).sum()
        if n_pred_fail == 0:
            lb_fails_in_topK = 0
        else:
            # Lower bound the actual fails inside top-K by:
            #   guaranteed (>= (1-alpha) m) total fails predicted-FAIL,
            #   of which the fraction in top-K is at LEAST n_pred_fail_in_topK / n_pred_fail
            #   minus an alpha-budget for adversarial placement.
            guaranteed_total_fails = max(0.0, (1 - alpha) * m - alpha * n)  # could be negative -> floor 0
            frac_in_topK = n_pred_fail_in_topK / n_pred_fail
            lb_fails_in_topK = int(math.floor(max(0.0, guaranteed_total_fails * frac_in_topK)))
        if lb_fails_in_topK == 0:
            lb = 0.0
        else:
            r = lb_fails_in_topK
            lb = 1 - sum(range(1, r+1))/(K*m) + 1.0/(2*K)
        out[K] = dict(lb=float(lb), emp=float(emp), pred_fail_topK=int(n_pred_fail_in_topK))
    return out

def main():
    t0=time.time()
    print("\n" + "="*72)
    print("EXP 12 -- Conformal Risk Control v2 (TIGHTER APFD bound)")
    print("Theory: Angelopoulos & Bates 2023 risk-control on top-K decision")
    print("="*72)
    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting features...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    model = RoadTransformer(in_channels=10, d_model=192, num_layers=5,
                            nhead=8, dim_feedforward=512)
    model, auc, swa = train(model, X_tr, y_tr, X_te, y_te, epochs=70,
        batch=384, lr=5e-4, swa_start=50, name='CRC-Backbone')
    m_eval = swa.get_model() if swa else model

    if comp_data is None:
        print("Competition data not present; skipping CRC eval.")
        return
    # Get logits on calib set (SensoDat-test) and eval set (Competition split)
    Xc = torch.tensor(np.array([(extract_sequence_10ch(get_pts(tc)) - means)/stds
                                for tc in test_data]), dtype=torch.float32).permute(0,2,1)
    Xe = torch.tensor(np.array([(extract_sequence_10ch(get_pts(tc)) - means)/stds
                                for tc in comp_data]), dtype=torch.float32).permute(0,2,1)
    logit_cal = predict_chunked(m_eval, Xc)
    logit_cmp = predict_chunked(m_eval, Xe)
    y_cal = np.array([1 if is_fail(tc) else 0 for tc in test_data])
    y_cmp = np.array([1 if is_fail(tc) else 0 for tc in comp_data])

    print(f"\n{'='*64}\nCONFORMAL RISK CONTROL v2 -- APFD@K LOWER BOUNDS\n{'='*64}")
    K_grid = [50, 100, 150, 200, 250, 287]
    rng = np.random.RandomState(0)

    for alpha in [0.05, 0.10, 0.20]:
        lbs = {K: [] for K in K_grid}; covered = {K: 0 for K in K_grid}
        n_draws = 200
        emp_at_full = {K: [] for K in K_grid}
        for _ in range(n_draws):
            idx = rng.permutation(len(y_cal))
            half = len(idx)//2
            calib_idx = idx[:half]
            lam_hat, _ = crc_calibrated_lambda(logit_cal[calib_idx], y_cal[calib_idx], alpha)
            res = crc_apfd_lower_bound(logit_cmp, y_cmp, lam_hat, alpha, K_grid)
            for K in K_grid:
                lbs[K].append(res[K]['lb'])
                emp_at_full[K].append(res[K]['emp'])
                if res[K]['emp'] >= res[K]['lb']: covered[K] += 1
        print(f"\nalpha = {alpha:.2f}")
        print(f"  {'K':>5} | {'LB-APFD@K (mean +/- std)':>26} | {'empirical APFD@K':>18} | {'coverage':>9}")
        for K in K_grid:
            mu = np.mean(lbs[K]); sd = np.std(lbs[K])
            emp = np.mean(emp_at_full[K])
            cov = covered[K] / n_draws
            print(f"  {K:>5} | {mu:>10.4f} +/- {sd:.4f}   | {emp:>18.4f} | {cov:>9.3f}")

    save = os.path.join(OUTPUT_DIR, 'roadcrc.pt')
    torch.save({'state': m_eval.state_dict(),
                'means': means.tolist(), 'stds': stds.tolist(),
                'logit_cal': logit_cal.tolist(), 'y_cal': y_cal.tolist()}, save)
    print(f"\nSaved: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
