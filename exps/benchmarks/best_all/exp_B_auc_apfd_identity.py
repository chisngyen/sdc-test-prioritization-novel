"""
Exp B -- The exact AUC <-> APFD identity (no-ties, single split)
================================================================
NOVELTY (target: ICSE 2027 oral)
--------------------------------
`exps/tracker.md` repeats the pattern "AUC up, APFD down" across at
least five experiments (Exp 01, 02, 03, 04, 10) and calls it
"AUC/APFD divergence". We prove this is not a metric pathology but an
**evaluation-split artifact**, by deriving the EXACT identity:

    APFD  =  (1 - p) * AUC  +  p / 2                                (*)

where p = m / n is the FAIL prior of the EVALUATION split (m = #fails,
n = #tests, no ties).

Derivation (Mann-Whitney U for AUC, position sum for APFD):
  Let prio_t in {1..n} be rank of test t (1 = served first).
  APFD = 1 - (sum_{t in FAIL} prio_t) / (n*m) + 1/(2n).
  Count negatives-served-after-each-fail t:
      sum_t [(n - prio_t) - (# fails after t)]  =  m*(n-m)*AUC.
  Sum of (# fails after t) over t in FAIL is m(m-1)/2.
  Substituting:    sum_t prio_t = m*n - m(m-1)/2 - m*(n-m)*AUC
  Plug in APFD:    APFD = (1 - m/n)*AUC + m/(2n)  =  (1-p)*AUC + p/2.  QED.

Consequences (each is a paper bullet):
  1. On the same split with no ties, AUC and APFD carry *identical*
     ranker information up to the constants (1-p, p/2).
  2. The "AUC up, APFD down" pattern in `tracker.md` is impossible on
     the SAME split. The tracker measures AUC on SensoDat-test but
     APFD on Competition multi-trial; those have DIFFERENT priors p,
     and that's where the divergence enters.
  3. Cross-bench comparison of rankers should use **prior-adjusted
     AUC**, defined as AUC* = (APFD - p/2) / (1 - p). Any cross-bench
     APFD table without prior-adjustment is comparing distributions,
     not rankers.

Protocol
--------
We verify (*) empirically on every bench we have data for:
  - Predict held-out scores once.
  - Compute AUC and APFD on the SAME held-out test split.
  - Plot residual abs(APFD - [(1-p)AUC + p/2]) vs n. Expect ~ 0 at the
    machine-precision level when there are no ties; ~ 0.5/n with ties.
  - Repeat across 5 seeds for robustness.

Saves: exp_B_identity.json
"""
import os, json, time, math, copy, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---- boilerplate (paths / feature extraction / model / training) ----
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

def train_geom(Xtr,ytr,Xv,yv,name='',epochs=EPOCHS,bs=BATCH,lr=LR,gamma=GAMMA,swa_st=SWA_START):
    print(f"\n--- Train {name} ---")
    model=RoadTransformer().to(DEVICE)
    npos=ytr.sum(); pw=float(len(ytr)-npos)/max(1,npos)
    w=np.where(ytr==1,pw,1.0); samp=WeightedRandomSampler(w,len(w),replacement=True)
    Xt=torch.tensor(Xtr,dtype=torch.float32).permute(0,2,1); yt=torch.tensor(ytr,dtype=torch.float32)
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=bs,sampler=samp,num_workers=2,pin_memory=True)
    Xv_t=torch.tensor(Xv,dtype=torch.float32).permute(0,2,1).to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3); warm=5
    sch=optim.lr_scheduler.LambdaLR(opt,lambda e:(e+1)/warm if e<warm else max(0.01,0.5*(1+math.cos(math.pi*(e-warm)/max(1,epochs-warm)))))
    crit=FocalLoss(gamma=gamma,pw=pw); amp=DEVICE.type=='cuda'; scl=GradScaler(enabled=amp)
    best=0; best_st=None; swa=None
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            xb=xb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp): loss=crit(model(xb),yb)
            scl.scale(loss).backward(); scl.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0); scl.step(opt); scl.update()
        sch.step()
        if ep>=swa_st:
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

def predict(model,Xn):
    Xt=torch.tensor(Xn,dtype=torch.float32).permute(0,2,1).to(DEVICE); model.eval().to(DEVICE)
    with torch.no_grad(): return torch.sigmoid(model(Xt)).cpu().numpy()

def apfd_from_ranking(probs, y):
    """APFD on a single split given per-test probs and binary labels y."""
    n=len(y); order=np.argsort(-probs)
    ranks_of_pos=[i+1 for i,idx in enumerate(order) if y[idx]==1]
    m=len(ranks_of_pos)
    return (1 - sum(ranks_of_pos)/(n*m) + 1/(2*n)) if (n and m) else 1.0

def prepare(data):
    X=np.array([resample(extract_seq(tc['road_points'])) for tc in data],dtype=np.float32)
    y=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data],dtype=np.int64)
    return X,y

# ---- loaders ----
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

def find_scissor():
    seen=set()
    for r in SEARCH_ROOTS:
        if not r or not os.path.isdir(r) or r in seen: continue
        seen.add(r)
        for dp,_,fn in os.walk(r):
            if sum(1 for x in fn if x.endswith('-test.json'))>=50: return dp
    return None
def load_scissor(root):
    if not root: return []
    data=[]
    for fp in sorted(glob.glob(os.path.join(root,'*-test.json'))):
        try:
            with open(fp) as f: tc=json.load(f)
        except Exception: continue
        if tc.get('is_valid',True) is False: continue
        pts=tc.get('interpolated_road_points') or tc.get('road_points')
        out=tc.get('test_outcome')
        if not pts or out not in ('FAIL','PASS'): continue
        data.append({'_id':os.path.basename(fp),'road_points':pts,'test_outcome':out})
    return data

# =================================================================
# Experiment B: verify identity APFD = (1-p) AUC + p/2
# =================================================================
def predicted_apfd(auc, p):
    return (1.0 - p) * auc + 0.5 * p

def evaluate_identity(eval_data, model, means, stds, name='', n_seeds=5):
    """For each seed, take a fixed bootstrap of eval_data, compute AUC and APFD,
    and report residual r = APFD - [(1-p)AUC + p/2]."""
    feats=np.array([resample(extract_seq(tc['road_points'])) for tc in eval_data],dtype=np.float32)
    feats=(feats-means)/stds
    y=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in eval_data])
    out=[]
    for s in range(n_seeds):
        rng=np.random.RandomState(42+s)
        # bootstrap to introduce variation in p (otherwise identity is trivial)
        idx=rng.choice(len(eval_data), size=len(eval_data), replace=True)
        probs=predict(model, feats[idx]); y_s=y[idx]
        if y_s.sum()==0 or y_s.sum()==len(y_s):
            continue
        auc=roc_auc_score(y_s, probs)
        apfd_=apfd_from_ranking(probs, y_s)
        p=float(y_s.sum()/len(y_s))
        pred=predicted_apfd(auc, p)
        resid=apfd_-pred
        # ties detector
        n_ties = len(probs) - len(np.unique(probs))
        out.append({'seed':s,'auc':auc,'apfd':apfd_,'p':p,'predicted_apfd':pred,
                    'residual':resid,'n_unique_scores':int(len(np.unique(probs))),
                    'n_ties':int(n_ties)})
        print(f"  [{name} s={s}] AUC={auc:.4f} p={p:.4f} APFD={apfd_:.4f} "
              f"pred={pred:.4f} resid={resid:+.5f} ties={n_ties}/{len(probs)}")
    return out

def main():
    t0=time.time()
    # Use 3 benches that span fail-prior range: OOB-0-1 (low FAIL), OOB-0-5 (high), Scissor (~0.58)
    targets={}
    print("[load] OOB-0-1, OOB-0-5, Scissor")
    for tag in ('0-1','0-5'):
        p=walk_for(f'Dataset-OOB-{tag}')
        if p:
            d=load_oob(p)
            if len(d)>100 and 20<=sum(tc['test_outcome']=='FAIL' for tc in d):
                targets[f'oob_{tag}']=d
                print(f"  oob_{tag}: N={len(d)} FAIL={sum(tc['test_outcome']=='FAIL' for tc in d)}")
    rs=find_scissor()
    if rs:
        d=load_scissor(rs)
        if len(d)>=80: targets['scissor']=d; print(f"  scissor: N={len(d)}")

    if not targets:
        print("No benches found."); return

    payload={'identity':'APFD = (1-p)*AUC + p/2 (no ties)','benches':{}}

    for name,data in targets.items():
        print(f"\n{'='*70}\nBench: {name}\n{'='*70}")
        y=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data])
        tr,te=train_test_split(np.arange(len(data)),test_size=0.25,stratify=y,random_state=42)
        train_data=[data[i] for i in tr]; test_data=[data[i] for i in te]
        Xtr,ytr=prepare(train_data); Xte,yte=prepare(test_data)
        means=Xtr.mean(axis=(0,1)); stds=Xtr.std(axis=(0,1)); stds[stds<1e-8]=1.0
        Xtr_n=(Xtr-means)/stds; Xte_n=(Xte-means)/stds
        model,auc=train_geom(Xtr_n,ytr,Xte_n,yte,name=name)
        # core test: residuals over bootstrap seeds
        runs=evaluate_identity(test_data, model, means, stds, name=name, n_seeds=10)
        resids=[r['residual'] for r in runs]
        ties=[r['n_ties'] for r in runs]
        payload['benches'][name]={'train_auc':auc,
                                   'runs':runs,
                                   'residual_mean':float(np.mean(resids)) if resids else None,
                                   'residual_std':float(np.std(resids)) if resids else None,
                                   'max_abs_residual':float(np.max(np.abs(resids))) if resids else None,
                                   'mean_ties':float(np.mean(ties)) if ties else None}
        if resids:
            print(f"  ★ residual: mean={np.mean(resids):+.5f} max|.|={np.max(np.abs(resids)):.5f}"
                  f"  ties mean={np.mean(ties):.1f}")
            # bound on residual due to ties is ~ T/(2n*m); confirm visually
    out=os.path.join(OUTPUT_DIR,'exp_B_identity.json')
    with open(out,'w') as f: json.dump(payload,f,indent=2)
    print(f"\nSaved {out}")
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

if __name__=='__main__': main()
