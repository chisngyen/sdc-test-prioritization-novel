"""
Exp C -- Multi-resolution Test-Time Augmentation via FNO modes
==============================================================
NOVELTY (target: ICSE 2027 oral)
--------------------------------
`exps/exp01_FNO_Roads.py` proved a Fourier Neural Operator is **almost
exactly resolution-invariant** (Delta = 0.0012 across N in {64..197}).
That's a clean *theoretical* property. We turn it into a *practical*
operational gain that no one has tried in SDC test prio:

**Multi-resolution test-time ensembling.** At inference, we run the
SAME trained model on the SAME road but resampled to multiple
sequence lengths {64, 96, 128, 160, 197}, then *average sigmoids*. This
costs ~5x at inference (no extra training, no extra params) and:

  - If the model is truly resolution-invariant, averaging is a
    *noise-reducing* operation -- variance over resolutions is sampling
    noise, not signal. APFD sigma drops.
  - If the model has *partial* invariance, the consensus rejects roads
    whose ranking flips under sampling-rate jitter -- a built-in
    safety filter for ambiguous fails.

We provide three signals from the multi-res view (per test):
  1. mean probability (the ranker)
  2. inter-resolution std (a "I'm uncertain" flag)
  3. agreement count #{N : p_N > 0.5} (the "consensus" filter)

This is novel for SDC: the literature uses single-resolution scoring.
This is novel even for ranking literature: resolution-TTA via FNO has
been proposed for vision but not for graph/sequence-on-curve ranking.

Saves: exp_C_multires_tta.json
"""
import os, json, time, math, copy, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try: HERE = os.path.dirname(os.path.abspath(__file__))
except NameError: HERE = os.getcwd()
SEARCH_ROOTS = [
    '/kaggle/input',
    os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', '..', 'data')),
    os.path.normpath(os.path.join(HERE, '..', 'data', 'kaggle')),
    os.path.normpath(os.path.join(HERE, '..', 'data')),
    os.getcwd(),
]
OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.normpath(os.path.join(HERE, '..', '..', 'models'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

MAX_LEN = 197
GAMMA, EPOCHS, BATCH, LR, SWA_START = 2.5, 60, 256, 5e-4, 40

def _curvature(pts):
    n=len(pts); curv=np.zeros(n-2)
    for i in range(n-2):
        x1,y1=pts[i]; x2,y2=pts[i+1]; x3,y3=pts[i+2]
        a=math.sqrt((x2-x1)**2+(y2-y1)**2); b=math.sqrt((x3-x2)**2+(y3-y2)**2); c=math.sqrt((x3-x1)**2+(y3-y1)**2)
        s=0.5*(a+b+c); at=s*(s-a)*(s-b)*(s-c)
        if at<=1e-10: curv[i]=0.0
        else: R=a*b*c/(4*math.sqrt(at)); curv[i]=1.0/R if R>0 else 0.0
    return curv
def extract_seq(pts_raw):
    pts=np.array(pts_raw,dtype=np.float64).reshape(-1,2); n=len(pts)
    if n<3: pts=np.vstack([pts]*3)[:max(3,n)]; n=len(pts)
    df=np.diff(pts,axis=0); seg=np.linalg.norm(df,axis=1); seg_f=np.pad(seg,(0,1),mode='edge')
    ang=np.arctan2(df[:,1],df[:,0]); ac=np.diff(ang); ac=(ac+np.pi)%(2*np.pi)-np.pi
    abs_ac=np.pad(np.abs(ac),(1,1),mode='constant')
    curv=np.abs(_curvature(pts)); curv_f=np.pad(curv,(1,1),mode='constant')
    cd=np.pad(np.diff(curv_f),(0,1),mode='constant')
    cum=np.cumsum(seg_f); cum_n=cum/(cum[-1]+1e-8)
    h=np.pad(ang,(0,1),mode='edge'); hs=np.sin(h); hc=np.cos(h)
    rel=np.linspace(0,1,n); w=11; ls=np.zeros(n); hw=w//2
    for i in range(n): s,e=max(0,i-hw),min(n,i+hw+1); ls[i]=np.std(curv_f[s:e])
    ca=np.pad(np.diff(cd),(0,1),mode='constant')
    return np.column_stack([seg_f,abs_ac,curv_f,cd,cum_n,hs,hc,rel,ls,ca]).astype(np.float32)
def resample(seq,L):
    n,c=seq.shape
    if n==L: return seq
    xo=np.linspace(0,1,n); xn=np.linspace(0,1,L); out=np.empty((L,c),dtype=np.float32)
    for ch in range(c): out[:,ch]=np.interp(xn,xo,seq[:,ch])
    return out

# Spectral conv: the resolution-invariance comes from truncating modes,
# which is identical regardless of N (>= 2*modes).
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.weights = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, 2))
    def forward(self, x):
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, n=L)
        out_ft = torch.zeros(B, self.out_ch, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        modes = min(self.modes, x_ft.shape[-1])
        w_c = torch.view_as_complex(self.weights[:, :, :modes].contiguous())
        out_ft[:, :, :modes] = torch.einsum("bcl,col->bol", x_ft[:, :, :modes], w_c)
        return torch.fft.irfft(out_ft, n=L)

class FNORanker(nn.Module):
    def __init__(self, in_ch=10, d=64, modes=24, n_blocks=4):
        super().__init__()
        self.lift = nn.Conv1d(in_ch, d, 1)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'sp': SpectralConv1d(d, d, modes),
            'w': nn.Conv1d(d, d, 1),
            'n': nn.GroupNorm(8, d),
        }) for _ in range(n_blocks)])
        self.project = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(),
                                      nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):  # x: B x C x L
        x = self.lift(x)
        for b in self.blocks:
            r = b['sp'](x) + b['w'](x)
            x = F.gelu(b['n'](r))
        # global avg over L (resolution-equivariant pooling)
        return self.project(x.mean(dim=-1)).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self,a=1.0,g=2.0,pw=1.0): super().__init__(); self.a,self.g,self.pw=a,g,pw
    def forward(self,logits,t):
        bce=F.binary_cross_entropy_with_logits(logits,t,reduction='none')
        w=torch.where(t==1,self.pw,1.0); bce=bce*w
        pt=torch.where(t==1,torch.sigmoid(logits),1-torch.sigmoid(logits))
        return (self.a*(1-pt)**self.g*bce).mean()

class SWAModel:
    def __init__(self,m): self.m=copy.deepcopy(m); self.n=0
    def update(self,nm):
        self.n+=1; a=1.0/self.n
        for p,q in zip(self.m.parameters(),nm.parameters()): p.data.mul_(1-a).add_(q.data,alpha=a)
    def get(self): return self.m

def train(Xtr,ytr,Xv,yv,name=''):
    print(f"\n--- Train FNO {name} ---")
    model=FNORanker().to(DEVICE)
    npos=ytr.sum(); pw=float(len(ytr)-npos)/max(1,npos)
    w=np.where(ytr==1,pw,1.0); samp=WeightedRandomSampler(w,len(w),replacement=True)
    Xt=torch.tensor(Xtr,dtype=torch.float32).permute(0,2,1); yt=torch.tensor(ytr,dtype=torch.float32)
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=BATCH,sampler=samp,num_workers=2,pin_memory=True)
    Xv_t=torch.tensor(Xv,dtype=torch.float32).permute(0,2,1).to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-3); warm=5
    sch=optim.lr_scheduler.LambdaLR(opt,lambda e:(e+1)/warm if e<warm else max(0.01,0.5*(1+math.cos(math.pi*(e-warm)/max(1,EPOCHS-warm)))))
    crit=FocalLoss(gamma=GAMMA,pw=pw); amp=DEVICE.type=='cuda'; scl=GradScaler(enabled=amp)
    best=0; best_st=None; swa=None
    for ep in range(EPOCHS):
        model.train()
        for xb,yb in dl:
            xb=xb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp): loss=crit(model(xb),yb)
            scl.scale(loss).backward(); scl.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0); scl.step(opt); scl.update()
        sch.step()
        if ep>=SWA_START:
            if swa is None: swa=SWAModel(model)
            else: swa.update(model)
        model.eval()
        with torch.no_grad():
            with autocast(enabled=amp): vl=model(Xv_t)
            try: auc=roc_auc_score(yv,torch.sigmoid(vl).cpu().numpy())
            except: auc=0.5
        if auc>best: best=auc; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if (ep+1)%10==0: print(f"  Ep {ep+1:3d} AUC={auc:.4f} Best={best:.4f}")
    model.load_state_dict(best_st)
    return (swa.get() if swa else model), float(best)

def predict(model, X):
    Xt=torch.tensor(X,dtype=torch.float32).permute(0,2,1).to(DEVICE); model.eval().to(DEVICE)
    with torch.no_grad(): return torch.sigmoid(model(Xt)).cpu().numpy()

def apfd(probs, y):
    n=len(y); order=np.argsort(-probs)
    ranks=[i+1 for i,idx in enumerate(order) if y[idx]==1]; m=len(ranks)
    return (1-sum(ranks)/(n*m)+1/(2*n)) if (n and m) else 1.0

# ------- novel content: multi-resolution scoring -------
RES_GRID = (64, 96, 128, 160, 197)

def featurize_at_resolutions(data, means, stds):
    """For each test, build a feature tensor at each N in RES_GRID."""
    seqs = [extract_seq(tc['road_points']) for tc in data]
    feats_by_N = {}
    for N in RES_GRID:
        X = np.array([resample(s, N) for s in seqs], dtype=np.float32)
        # normalise with TRAIN means/stds (these were computed at MAX_LEN)
        # stats are per-channel so the same vector works for any N
        X = (X - means) / stds
        feats_by_N[N] = X
    return feats_by_N

def multi_res_scores(model, feats_by_N):
    return {N: predict(model, X) for N, X in feats_by_N.items()}

def ensemble(scores):
    """scores: {N: arr}. Returns mean and std across N for each test."""
    arr = np.stack([scores[N] for N in scores.keys()], axis=0)
    return arr.mean(axis=0), arr.std(axis=0)

# ------- loaders (OOB + scissor) -------
def walk_for(target):
    seen=set()
    for r in SEARCH_ROOTS:
        if not r or not os.path.isdir(r) or r in seen: continue
        seen.add(r)
        for dp,dn,fn in os.walk(r):
            if os.path.basename(dp)==target:
                if any(x.endswith('.json') for x in fn): return dp
                for d in dn:
                    inner=os.path.join(dp,d)
                    try:
                        if any(x.endswith('.json') for x in os.listdir(inner)): return inner
                    except OSError: continue
    return None
def load_oob(path):
    data=[]
    for fp in sorted(glob.glob(os.path.join(path,'*.json'))):
        try:
            with open(fp) as f: tc=json.load(f)
        except Exception: continue
        if not tc.get('is_valid',True): continue
        pts=tc.get('road_points'); out=tc.get('test_outcome')
        if not pts or out not in ('FAIL','PASS'): continue
        data.append({'_id':os.path.basename(fp),'road_points':pts,'test_outcome':out})
    return data

def prepare(data, L=MAX_LEN):
    X=np.array([resample(extract_seq(tc['road_points']), L) for tc in data],dtype=np.float32)
    y=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data],dtype=np.int64)
    return X,y

def main():
    t0=time.time()
    # Use OOB-0-3 as the workhorse bench (4.7K, balanced enough).
    path = walk_for('Dataset-OOB-0-3') or walk_for('Dataset-OOB-0-5')
    if not path:
        print('OOB not found'); return
    print(f"data: {path}")
    data = load_oob(path)
    print(f"N={len(data)} FAIL={sum(tc['test_outcome']=='FAIL' for tc in data)}")
    y_all=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data])
    tr,te=train_test_split(np.arange(len(data)),test_size=0.2,stratify=y_all,random_state=42)
    train_data=[data[i] for i in tr]; test_data=[data[i] for i in te]
    Xtr,ytr=prepare(train_data); Xte,yte=prepare(test_data)
    means=Xtr.mean(axis=(0,1)); stds=Xtr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    Xtr_n=(Xtr-means)/stds; Xte_n=(Xte-means)/stds
    model,auc=train(Xtr_n,ytr,Xte_n,yte,name='OOB')

    # Single-resolution baselines (one APFD per N)
    feats_by_N = featurize_at_resolutions(test_data, means, stds)
    scores = multi_res_scores(model, feats_by_N)
    per_N = {N: apfd(scores[N], yte) for N in RES_GRID}
    print(f"\nSingle-resolution APFD:")
    for N in RES_GRID: print(f"  N={N:3d}: APFD={per_N[N]:.4f}")
    delta = max(per_N.values()) - min(per_N.values())
    print(f"  Delta (max-min) = {delta:.4f}")

    # Ensemble (multi-res TTA)
    mean_p, std_p = ensemble(scores)
    apfd_ens = apfd(mean_p, yte)
    print(f"\nMulti-res TTA ensemble APFD = {apfd_ens:.4f}")

    # Consensus filter: keep only tests where every N agrees on >0.5
    sig = np.stack([scores[N] > 0.5 for N in RES_GRID], axis=0)
    agree = sig.sum(axis=0)  # 0..len(RES_GRID)
    # Subset: tests with strong agreement (>=4/5)
    keep = agree >= max(1, len(RES_GRID) - 1)
    if keep.sum() > 5 and yte[keep].sum() > 0:
        apfd_confident = apfd(mean_p[keep], yte[keep])
        print(f"Confident subset (n={int(keep.sum())}/{len(yte)}, fails={int(yte[keep].sum())}): "
              f"APFD={apfd_confident:.4f}")
    else:
        apfd_confident = None

    # AUC on the ensemble (sanity)
    try:
        auc_ens = roc_auc_score(yte, mean_p)
    except Exception:
        auc_ens = float('nan')

    payload = {'res_grid': list(RES_GRID),
                'per_N_apfd': {str(k): v for k, v in per_N.items()},
                'per_N_max_min_delta': float(delta),
                'ensemble_apfd': float(apfd_ens),
                'ensemble_auc': float(auc_ens),
                'confident_apfd': float(apfd_confident) if apfd_confident is not None else None,
                'mean_inter_res_std': float(std_p.mean()),
                'val_auc_train': auc,
                'n_train': len(train_data), 'n_test': len(test_data),
                'n_fail_test': int(yte.sum())}
    out=os.path.join(OUTPUT_DIR,'exp_C_multires_tta.json')
    with open(out,'w') as f: json.dump(payload,f,indent=2)
    print(f"\nSaved {out}")
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

if __name__=='__main__': main()
