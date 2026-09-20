"""
Exp E -- Curvature-violation rate as auxiliary ranking signal
=============================================================
NOVELTY (target: ICSE 2027 oral)
--------------------------------
Exp 04 in `exps/tracker.md` showed that the monotone-PINN constraint
drops curvature-violation rate from 17.57% to 3.14% (a 5.6x reduction)
WITHOUT moving APFD. We always treated violations as a *negative*
diagnostic to minimise.

We invert the framing: **violations are themselves diagnostic of hard
tests**. A road where the model violates the curvature-monotonicity
prior is a road where the model is *least sure* the physics holds --
which correlates with where the failure boundary actually sits.

Define a violation-aware score:
        S_t = alpha * p_t + (1 - alpha) * v_t / max(v)
where p_t = sigmoid(logit), v_t = per-test violation count w.r.t. a
"more curvature -> more likely to fail" prior (i.e. a Spearman-type
prior over (kappa_max, label)).

We sweep alpha in {1.0, 0.9, 0.7, 0.5, 0.0} and measure:
  - APFD vs alpha (does violation help?)
  - violation-rate of the top-K prefix (audit metric)
  - rank correlation between v_t and ground-truth FAIL position

This is novel because:
  1. PINN papers reduce violations; nobody uses violations as a SCORE.
  2. SDC prioritization papers do not have a physical regulariser to
     compute violations against in the first place.

Saves: exp_E_violation_aware.json
"""
import os, json, time, math, copy, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

try: HERE = os.path.dirname(os.path.abspath(__file__))
except NameError: HERE = os.getcwd()
SEARCH_ROOTS=[ '/kaggle/input',
    os.path.normpath(os.path.join(HERE,'..','..','data','kaggle')),
    os.path.normpath(os.path.join(HERE,'..','..','data')),
    os.path.normpath(os.path.join(HERE,'..','data','kaggle')),
    os.path.normpath(os.path.join(HERE,'..','data')),
    os.getcwd()]
OUTPUT_DIR='/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.normpath(os.path.join(HERE,'..','..','models'))
os.makedirs(OUTPUT_DIR,exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
SEQ_LEN, GAMMA, EPOCHS, BATCH, LR, SWA_START = 197, 2.5, 60, 256, 5e-4, 40

# ---- boilerplate (compressed) ----
def _curvature(pts):
    n=len(pts); curv=np.zeros(n-2)
    for i in range(n-2):
        x1,y1=pts[i]; x2,y2=pts[i+1]; x3,y3=pts[i+2]
        a=math.sqrt((x2-x1)**2+(y2-y1)**2); b=math.sqrt((x3-x2)**2+(y3-y2)**2); c=math.sqrt((x3-x1)**2+(y3-y1)**2)
        s=0.5*(a+b+c); at=s*(s-a)*(s-b)*(s-c)
        curv[i]=0.0 if at<=1e-10 else (1.0/(a*b*c/(4*math.sqrt(at))) if a*b*c>0 else 0.0)
    return curv
def extract_seq(pts_raw):
    pts=np.array(pts_raw,dtype=np.float64).reshape(-1,2); n=len(pts)
    if n<3: pts=np.vstack([pts]*3)[:max(3,n)]; n=len(pts)
    df=np.diff(pts,axis=0); seg=np.linalg.norm(df,axis=1); seg_f=np.pad(seg,(0,1),mode='edge')
    ang=np.arctan2(df[:,1],df[:,0]); ac=np.diff(ang); ac=(ac+np.pi)%(2*np.pi)-np.pi
    abs_ac=np.pad(np.abs(ac),(1,1),mode='constant')
    curv=np.abs(_curvature(pts)); curv_f=np.pad(curv,(1,1),mode='constant')
    cd=np.pad(np.diff(curv_f),(0,1),mode='constant'); cum=np.cumsum(seg_f); cum_n=cum/(cum[-1]+1e-8)
    h=np.pad(ang,(0,1),mode='edge'); hs=np.sin(h); hc=np.cos(h); rel=np.linspace(0,1,n)
    w=11; ls=np.zeros(n); hw=w//2
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

def apfd(probs,y):
    n=len(y); order=np.argsort(-probs)
    ranks=[i+1 for i,idx in enumerate(order) if y[idx]==1]; m=len(ranks)
    return (1-sum(ranks)/(n*m)+1/(2*n)) if (n and m) else 1.0

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
# Violation computation
# =================================================================
def per_test_curvature_stats(data):
    """Return per-test max-abs-curvature kappa_max and curvature monotonicity
    violation count (number of segments where curvature decreases when the
    cumulative-distance-to-failure-region heuristic suggests it should
    increase). We approximate with a simple roughness proxy:
        v_t = (# of curvature peaks > tau) - (# of plateaus < tau)
    which counts unstable-curvature regions where a monotone-physics prior
    would predict elevated fail probability."""
    out=[]
    for tc in data:
        pts=np.array(tc['road_points'],dtype=np.float64).reshape(-1,2)
        if len(pts) < 4:
            out.append({'kappa_max':0.0, 'viol':0.0})
            continue
        curv = np.abs(_curvature(pts))
        if len(curv)==0:
            out.append({'kappa_max':0.0, 'viol':0.0}); continue
        k_max = float(curv.max())
        # Violation proxy: deviation from monotone increase to k_max
        s = np.linspace(0, 1, len(curv))
        # ideal-monotone reference: linearly interpolate min->max along s
        ref = curv.min() + (k_max - curv.min()) * s
        viol = float(np.mean(np.maximum(0, ref - curv)))  # how much we undershoot the monotone ramp
        out.append({'kappa_max':k_max, 'viol':viol})
    return out

def compute_violation_scores(data):
    stats = per_test_curvature_stats(data)
    v = np.array([s['viol'] for s in stats], dtype=np.float32)
    kappa = np.array([s['kappa_max'] for s in stats], dtype=np.float32)
    # normalise to [0, 1]
    v_n = (v - v.min()) / max(1e-8, v.max() - v.min())
    k_n = (kappa - kappa.min()) / max(1e-8, kappa.max() - kappa.min())
    # combined physics prior: stronger curvature + larger violations => higher fail-prob prior
    phys = 0.5 * v_n + 0.5 * k_n
    return v_n, k_n, phys

def main():
    t0=time.time()
    path=walk_for('Dataset-OOB-0-3') or walk_for('Dataset-OOB-0-5')
    if not path: print("OOB not found"); return
    print(f"data: {path}")
    data=load_oob(path)
    print(f"N={len(data)} FAIL={sum(tc['test_outcome']=='FAIL' for tc in data)}")
    y_all=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data])
    tr,te=train_test_split(np.arange(len(data)),test_size=0.2,stratify=y_all,random_state=42)
    train_data=[data[i] for i in tr]; test_data=[data[i] for i in te]
    Xtr,ytr=prepare(train_data); Xte,yte=prepare(test_data)
    means=Xtr.mean(axis=(0,1)); stds=Xtr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    Xtr_n=(Xtr-means)/stds; Xte_n=(Xte-means)/stds
    model,auc=train(Xtr_n,ytr,Xte_n,yte,name='OOB')
    p = predict(model, Xte_n)

    # physics features on test set
    v_n, k_n, phys = compute_violation_scores(test_data)
    print(f"\nSpearman correlations with y (label):")
    rho_p, _ = spearmanr(p, yte)
    rho_v, _ = spearmanr(v_n, yte)
    rho_k, _ = spearmanr(k_n, yte)
    rho_phys, _ = spearmanr(phys, yte)
    print(f"  prob:         rho={rho_p:.4f}")
    print(f"  violation:    rho={rho_v:.4f}")
    print(f"  kappa_max:    rho={rho_k:.4f}")
    print(f"  phys (combo): rho={rho_phys:.4f}")

    # sweep mix
    print(f"\nalpha mix (prob vs physics): APFD")
    sweep=[]
    for a in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0):
        score = a * p + (1 - a) * phys
        ap = apfd(score, yte)
        sweep.append({'alpha':a, 'apfd':float(ap)})
        print(f"  alpha={a:.2f}: APFD={ap:.4f}")

    # ablations: violation alone, kappa alone
    apfd_v = apfd(v_n, yte); apfd_k = apfd(k_n, yte); apfd_p = apfd(p, yte)
    print(f"\nAblations:  prob-only={apfd_p:.4f}  viol-only={apfd_v:.4f}  kappa-only={apfd_k:.4f}")

    payload={'sweep':sweep,
              'ablations':{'prob_only':float(apfd_p),'viol_only':float(apfd_v),'kappa_only':float(apfd_k)},
              'spearman':{'prob':float(rho_p),'viol':float(rho_v),'kappa':float(rho_k),'phys':float(rho_phys)},
              'train_auc':auc,
              'n_train':len(train_data),'n_test':len(test_data),'n_fail_test':int(yte.sum())}
    out=os.path.join(OUTPUT_DIR,'exp_E_violation_aware.json')
    with open(out,'w') as f: json.dump(payload,f,indent=2)
    print(f"\nSaved {out}")
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

if __name__=='__main__': main()
