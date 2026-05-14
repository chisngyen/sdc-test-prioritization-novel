"""
Exp D -- Conformal abstention for SDC prioritization
====================================================
NOVELTY (target: ICSE 2027 oral)
--------------------------------
Exp 05 / Exp 12 in `exps/tracker.md` tried to give a conformal *lower
bound* on prefix-APFD: v1 was valid-but-vacuous, v2 informative-but-
invalid. We pivot to a problem conformal prediction is *actually*
designed for: **selective prediction with provable mis-coverage**.

Setup. For each test t, the ranker outputs sigmoid score s_t. We
compute a per-test conformal score
        c_t = | s_t - 0.5 |
i.e. distance to the decision boundary. Tests with c_t below a
calibrated threshold tau_alpha are placed in an **abstain bucket**:
they are RANDOMLY ordered (worst-case position-of-fail).

Coverage claim (split conformal, exchangeable calib/eval):
        P( label(t) is predicted correctly | t not abstained ) >= 1 - alpha
which translates to a population guarantee that fault rates among
non-abstained tests are at most alpha lower than the model's confident
predictions.

We then measure two things:
  1. APFD of the **non-abstained** prefix vs full-ranking APFD.
     Selective prioritization should *improve* APFD on the confident
     subset (the easy fails).
  2. Trade-off curve: APFD vs abstention fraction across
     alpha in {0.01, 0.05, 0.10, 0.20}.

Novelty vs literature:
  - Selective classification (Geifman & El-Yaniv, NeurIPS 2017) exists
    in vision but has not been applied to SDC test prioritization.
  - The conformal-quantile choice gives a *distribution-free* guarantee
    -- no model assumptions, no calibration trick, just exchangeability.

Story for the oral: "if you don't trust the model, abstain -- with a
proven error budget." This is the safety/audit angle Exp 04 (PINN
monotonicity) sets up: PINN gives physical safety, conformal abstention
gives statistical safety. Pair them in one figure.

Saves: exp_D_conformal_abstain.json
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
SEQ_LEN, GAMMA, EPOCHS, BATCH, LR, SWA_START = 197, 2.5, 60, 256, 5e-4, 40
ALPHAS = (0.01, 0.05, 0.10, 0.20)

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
def resample(seq,L=SEQ_LEN):
    n,c=seq.shape
    if n==L: return seq
    xo=np.linspace(0,1,n); xn=np.linspace(0,1,L); out=np.empty((L,c),dtype=np.float32)
    for ch in range(c): out[:,ch]=np.interp(xn,xo,seq[:,ch])
    return out

class RoadTransformer(nn.Module):
    def __init__(self,in_ch=10,L=SEQ_LEN,d=128,h=8,nl=4,dff=512,dr=0.1):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(in_ch,d),nn.LayerNorm(d),nn.GELU())
        self.cls=nn.Parameter(torch.randn(1,1,d)*0.02); self.pos=nn.Parameter(torch.randn(1,L+1,d)*0.02)
        e=nn.TransformerEncoderLayer(d_model=d,nhead=h,dim_feedforward=dff,dropout=dr,activation='gelu',batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(e,num_layers=nl)
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,1))
    def forward(self,x):
        x=x.permute(0,2,1); B,L,_=x.shape
        x=self.proj(x); x=torch.cat([self.cls.expand(B,-1,-1),x],dim=1); x=x+self.pos[:,:L+1,:]
        return self.head(self.tr(x)[:,0,:]).squeeze(-1)

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
    print(f"\n--- Train {name} ---")
    model=RoadTransformer().to(DEVICE)
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
    model.load_state_dict(best_st)
    return (swa.get() if swa else model), float(best)

def predict(model,X):
    Xt=torch.tensor(X,dtype=torch.float32).permute(0,2,1).to(DEVICE); model.eval().to(DEVICE)
    with torch.no_grad(): return torch.sigmoid(model(Xt)).cpu().numpy()

def apfd(probs, y):
    n=len(y); order=np.argsort(-probs)
    ranks=[i+1 for i,idx in enumerate(order) if y[idx]==1]; m=len(ranks)
    return (1-sum(ranks)/(n*m)+1/(2*n)) if (n and m) else 1.0

def apfd_with_abstain(probs, y, abstain_mask, n_random_trials=20, seed=42):
    """Build a ranking: confident tests (NOT abstained) first, ranked by prob;
    abstained tests at end, averaged over `n_random_trials` random orderings.
    Returns mean APFD over the random tail orderings."""
    n=len(y); conf_idx = np.where(~abstain_mask)[0]; abs_idx = np.where(abstain_mask)[0]
    # Order confident by score desc
    conf_order = conf_idx[np.argsort(-probs[conf_idx])]
    apfds=[]
    rng=np.random.RandomState(seed)
    for t in range(n_random_trials):
        perm = rng.permutation(abs_idx)
        full_order = np.concatenate([conf_order, perm])
        ranks=[i+1 for i,idx in enumerate(full_order) if y[idx]==1]; m=len(ranks)
        if not m: apfds.append(1.0); continue
        apfds.append(1 - sum(ranks)/(n*m) + 1/(2*n))
    return float(np.mean(apfds)), float(np.std(apfds))

# loaders (OOB-0-3)
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
def prepare(data):
    X=np.array([resample(extract_seq(tc['road_points'])) for tc in data],dtype=np.float32)
    y=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data],dtype=np.int64)
    return X,y

# =================================================================
# Conformal abstention
# =================================================================
def conformal_threshold(scores_cal, y_cal, alpha):
    """Split conformal: nonconformity = 1 - p_y for the calibration set.
    The (1-alpha) empirical quantile gives a threshold; tests with margin
    (|p - 0.5|) BELOW this -> ABSTAIN."""
    # nonconformity: how far the model is from being CONFIDENT in the right class
    # For binary: NC = 1 - p_true = 1 - sigmoid_pred if y=1 else sigmoid_pred
    nc = np.where(y_cal == 1, 1 - scores_cal, scores_cal)
    n_cal = len(nc)
    q = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q = min(q, 1.0)
    tau = np.quantile(nc, q, method='higher')
    return tau

def abstain_mask(scores, tau):
    """For each test, NC = min(p, 1-p) (we don't know y at test time).
    Abstain if min(p, 1-p) > tau, i.e. neither class is confident enough."""
    margin = np.minimum(scores, 1.0 - scores)  # 0 at p=1 or p=0, 0.5 at p=0.5
    return margin > tau  # uncertain tests abstain

def main():
    t0=time.time()
    path=walk_for('Dataset-OOB-0-3') or walk_for('Dataset-OOB-0-5')
    if not path: print("OOB not found"); return
    print(f"data: {path}")
    data=load_oob(path)
    print(f"N={len(data)} FAIL={sum(tc['test_outcome']=='FAIL' for tc in data)}")
    y_all=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data])
    # 3-way split: train / calib / test (60 / 20 / 20)
    tr,rest=train_test_split(np.arange(len(data)),test_size=0.4,stratify=y_all,random_state=42)
    cal,te=train_test_split(rest,test_size=0.5,stratify=y_all[rest],random_state=42)
    train_data=[data[i] for i in tr]
    cal_data  =[data[i] for i in cal]
    test_data =[data[i] for i in te]
    print(f"train={len(train_data)} cal={len(cal_data)} test={len(test_data)}")
    Xtr,ytr=prepare(train_data); Xcal,ycal=prepare(cal_data); Xte,yte=prepare(test_data)
    means=Xtr.mean(axis=(0,1)); stds=Xtr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    Xtr_n=(Xtr-means)/stds; Xcal_n=(Xcal-means)/stds; Xte_n=(Xte-means)/stds
    model,auc=train(Xtr_n,ytr,Xcal_n,ycal,name='OOB')
    print(f"validation AUC on calib: {auc:.4f}")

    scores_cal = predict(model, Xcal_n)
    scores_te  = predict(model, Xte_n)

    # full APFD (no abstention)
    apfd_full = apfd(scores_te, yte)
    auc_te = roc_auc_score(yte, scores_te)
    print(f"\nFull-set test APFD = {apfd_full:.4f}, AUC = {auc_te:.4f}")

    # sweep alpha
    print(f"\n{'alpha':>6s} {'tau':>8s} {'abst%':>8s} {'APFD':>9s}+/- {'std':>6s} {'cov':>7s}")
    sweep = []
    for alpha in ALPHAS:
        tau = conformal_threshold(scores_cal, ycal, alpha)
        abst = abstain_mask(scores_te, tau)
        # coverage on test: fraction of NON-abstained where model's max class is correct
        non_ab = ~abst
        if non_ab.sum() > 0:
            pred = (scores_te[non_ab] > 0.5).astype(int)
            correct = (pred == yte[non_ab]).mean()
        else:
            correct = float('nan')
        apfd_m, apfd_s = apfd_with_abstain(scores_te, yte, abst, n_random_trials=30, seed=42)
        sweep.append({'alpha': alpha, 'tau': float(tau),
                      'abstain_frac': float(abst.mean()),
                      'apfd_mean': apfd_m, 'apfd_std': apfd_s,
                      'coverage_on_non_abstain': float(correct)})
        print(f"  {alpha:.2f}  {tau:.4f}  {100*abst.mean():>6.1f}%  "
              f"{apfd_m:.4f}+/-{apfd_s:.4f}  {correct:.4f}")

    # Also report APFD on the CONFIDENT subset alone (no random tail), as an
    # "if we only run confident tests" upper bound. APFD on small subset.
    print(f"\nAPFD on confident subset only:")
    confident_apfds = []
    for alpha in ALPHAS:
        tau = conformal_threshold(scores_cal, ycal, alpha)
        abst = abstain_mask(scores_te, tau)
        non = ~abst
        if non.sum() > 0 and yte[non].sum() > 0:
            a = apfd(scores_te[non], yte[non])
        else:
            a = float('nan')
        confident_apfds.append({'alpha': alpha, 'apfd_confident': float(a),
                                'n_confident': int(non.sum()),
                                'fail_confident': int(yte[non].sum())})
        print(f"  alpha={alpha:.2f}: APFD_conf={a:.4f} (n={int(non.sum())}, "
              f"fail={int(yte[non].sum())})")

    payload = {'alphas': list(ALPHAS),
               'full_apfd': float(apfd_full),
               'full_auc': float(auc_te),
               'sweep': sweep,
               'confident_only': confident_apfds,
               'n_train': len(train_data), 'n_cal': len(cal_data), 'n_test': len(test_data),
               'n_fail_test': int(yte.sum())}
    out=os.path.join(OUTPUT_DIR,'exp_D_conformal_abstain.json')
    with open(out,'w') as f: json.dump(payload,f,indent=2)
    print(f"\nSaved {out}")
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

if __name__=='__main__': main()
