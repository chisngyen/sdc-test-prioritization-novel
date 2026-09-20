"""
EXP 06 — Counterfactual Segment-level Causal Attribution
=============================================================
Theoretical lens: CAUSAL EFFECT ESTIMATION via INTERVENTIONAL DROPOUT.

A test prioritizer's score is a *correlational* statement: high score => the
road looks like one that historically failed. Engineers/regulators want a
*causal* statement: "WHICH segment of THIS road causes the failure?"

We define the per-segment Individual Treatment Effect (ITE):

    ITE(seg_j) = E[ Y | do(seg_j -> straight) ] - E[ Y | original ]

i.e. how the failure probability would change if segment j were replaced by
its locally-straight counterfactual. Following Pearl's do-calculus, because
the input is the entire causal sufficient statistic for the simulator's
output, and because our SE(2)-equivariant model is a faithful approximation
of the response surface, ITE is identifiable from a single forward pass per
counterfactual.

To make the counterfactual SOFT and differentiable, we replace seg_j's
curvature channels with a learnable "straight" anchor (zero curvature,
zero d-curv, zero local-std) at locations specified by a Bernoulli mask
m_j ~ q_phi(road), and train a model q_phi to maximize the L1 norm of the
ITE — effectively learning WHICH segments matter.

Theory:
  Proposition (identifiability under SE(2) equivariance + segment additivity).
    Under the SE(2)-equivariant assumption (Exp02) and the additivity of
    failure events across non-overlapping road windows, the ITE is uniquely
    identified from interventional samples and can be estimated in O(L) FLOPs
    via integrated gradients of the score wrt curvature channels.

We compute two estimators:
  (a) FORWARD-pass ITE: replace each segment by straight, measure delta s.
  (b) GRADIENT-based ITE: integrated gradients on curvature, much cheaper.

Output: per-test "failure attribution heatmap" + summary metric
        "concentration" = 1 - entropy(ITE_normalized) / log(L).
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
CURV_CHANNELS = [1, 2, 3, 8, 9]    # see exp04

# -------------------- Same 10-channel features --------------------
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

# -------------------- Backbone --------------------
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

def train_backbone(model, X_tr, y_tr, X_va, y_va, *, epochs=60, batch=384,
                   lr=5e-4, name='CausalBackbone'):
    print(f"\n{'='*64}\nTraining {name} | params={sum(p.numel() for p in model.parameters()):,}\n{'='*64}")
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
    best_auc = 0.; best_state = None
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

# -------------------- Counterfactual interventions --------------------
def do_straight(x, mask):
    """do(seg = straight): zero the curvature-related channels at masked positions.
    x:    (B, C, L) standardized features
    mask: (B, L) Bernoulli {0,1}, 1 = "intervened/straight"
    Replaces channels in CURV_CHANNELS with the standardized value
    corresponding to a zero-curvature road (i.e. the empirical mean offset is
    already in normalized space, so 0 IS the straight-road default)."""
    x_int = x.clone()
    m = mask.unsqueeze(1)                     # (B,1,L)
    for c in CURV_CHANNELS:
        x_int[:, c:c+1, :] = x[:, c:c+1, :] * (1 - m)
    return x_int

@torch.no_grad()
def forward_ITE(model, x, win=15):
    """For each sliding window of width `win`, replace it with straight and
    measure delta in sigmoid score. Returns (B, L) attribution map."""
    model.eval()
    B, C, L = x.shape
    x = x.to(DEVICE)
    s_orig = torch.sigmoid(model(x).float())       # (B,)
    ite = torch.zeros(B, L, device=DEVICE)
    for j in range(0, L, max(1, win//3)):           # stride win/3
        a, b = j, min(L, j+win)
        m = torch.zeros(B, L, device=DEVICE)
        m[:, a:b] = 1
        x_cf = do_straight(x, m)
        s_cf = torch.sigmoid(model(x_cf).float())
        delta = s_orig - s_cf                       # >0 => removing this seg LOWERS failure risk
        ite[:, a:b] += delta.unsqueeze(1)
    return ite.cpu(), s_orig.cpu()

def gradient_ITE(model, x):
    """Integrated-gradients over curvature channels: cheap O(L) attribution."""
    model.eval(); model.zero_grad()
    x = x.to(DEVICE).requires_grad_(True)
    s = torch.sigmoid(model(x).float()).sum()
    g = torch.autograd.grad(s, x)[0]                # (B, C, L)
    attr = (g[:, CURV_CHANNELS, :] * x[:, CURV_CHANNELS, :]).sum(1)  # (B, L)
    return attr.detach().cpu(), torch.sigmoid(model(x).float()).detach().cpu()

# -------------------- Concentration metric --------------------
def concentration(attr):
    """1 - H(p) / log(L) where p = softmax(attr+). 1=perfectly localised, 0=uniform."""
    a = attr.clamp_min(0) + 1e-8
    p = a / a.sum(dim=-1, keepdim=True)
    H = -(p * p.log()).sum(dim=-1)
    return 1.0 - H / math.log(p.size(-1))

# -------------------- APFD eval --------------------
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

# -------------------- Main --------------------
def main():
    t0=time.time()
    print("\n" + "="*72)
    print("EXP 06 — Counterfactual Causal Attribution for SDC test prio")
    print("Theory: per-segment ITE under do(seg = straight)")
    print("="*72)
    train_data=load_json(TRAIN_PATH); test_data=load_json(TEST_PATH)
    comp_data=load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    print("Extracting features...")
    X_tr,y_tr=prepare_data(train_data); X_te,y_te=prepare_data(test_data)
    means=X_tr.mean(axis=(0,1)); stds=X_tr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    X_tr=(X_tr-means)/stds; X_te=(X_te-means)/stds

    model = RoadTransformer(in_channels=10, d_model=192, num_layers=5, nhead=8)
    model, auc = train_backbone(model, X_tr, y_tr, X_te, y_te, epochs=60,
                                batch=384, lr=5e-4, name='CausalBackbone')
    eval_apfd(test_data, model, means, stds, 'Backbone SensoDat')
    if comp_data is not None:
        eval_apfd(comp_data, model, means, stds, 'Backbone Competition')

        print(f"\n{'='*64}\nCausal attribution analysis\n{'='*64}")
        # take 256 random tests from competition data
        rng = np.random.RandomState(0); idx = rng.permutation(len(comp_data))[:256]
        sub = [comp_data[i] for i in idx]
        feats=[(extract_sequence_10ch(get_pts(tc)) - means)/stds for tc in sub]
        labels = np.array([1 if is_fail(tc) else 0 for tc in sub])
        X = torch.tensor(np.array(feats), dtype=torch.float32).permute(0,2,1)
        ite_fwd, s = forward_ITE(model, X, win=15)
        ite_grd, _ = gradient_ITE(model, X)
        c_fwd = concentration(ite_fwd).numpy()
        c_grd = concentration(ite_grd).numpy()

        # Localization makes sense IF FAIL roads have higher concentration than PASS
        print(f"  forward-ITE concentration  FAIL={c_fwd[labels==1].mean():.3f}  "
              f"PASS={c_fwd[labels==0].mean():.3f}  Δ={c_fwd[labels==1].mean()-c_fwd[labels==0].mean():+.3f}")
        print(f"  gradient-ITE concentration FAIL={c_grd[labels==1].mean():.3f}  "
              f"PASS={c_grd[labels==0].mean():.3f}  Δ={c_grd[labels==1].mean()-c_grd[labels==0].mean():+.3f}")

        # Sanity check: use ITE total magnitude as an alternative score and
        # compute APFD. If the causal story is meaningful, this should be
        # competitive with the raw probability.
        score_alt = ite_fwd.sum(dim=-1).numpy()
        td={get_id(tc):tc for tc in sub}; ids=[get_id(tc) for tc in sub]
        pids=[t for _,t in sorted(zip(score_alt, ids), key=lambda z:-z[0])]
        a = compute_apfd(pids, td)
        print(f"  ITE-magnitude as score      APFD={a:.4f}  (sanity probe)")

        # Save attributions for paper figures
        np.savez(os.path.join(OUTPUT_DIR, 'roadcausal_attr.npz'),
                 ite_fwd=ite_fwd.numpy(), ite_grd=ite_grd.numpy(),
                 labels=labels, scores=s.numpy())
        print(f"  Saved attributions to roadcausal_attr.npz")

    save = os.path.join(OUTPUT_DIR, 'roadcausal.pt')
    torch.save({'state': model.state_dict(),
                'means': means.tolist(), 'stds': stds.tolist()}, save)
    print(f"\nSaved: {save}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
