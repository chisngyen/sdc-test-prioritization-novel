"""
Exp G -- Prefix-weighted Plackett-Luce listwise loss
====================================================
NOVELTY (target: ICSE 2027 oral)
--------------------------------
Exp 03 in `exps/tracker.md` showed a curious failure mode:
  - Plackett-Luce only:  AUC 0.9214 (HIGHER than baseline) but APFD
    0.8012 (LOWER than baseline).
The listwise loss pulled AUC up but DID NOT translate to top-K
prioritization. We now know why (Exp B identity): on a fixed split,
AUC determines APFD up to (1-p), p/2. PL loss treated all pairs
equally; APFD weighs *prefix* pairs more.

Fix: weight each contrastive comparison by where it lives in the
priority order. The novel loss:

    L_prefix = - sum_t  log( exp(s_t) / sum_{u in S_t} exp(s_u) )  *  w_t

where S_t is the suffix of positives starting at rank t, and the
weight w_t is the APFD-prefix-bias function:
    w_t = (n - rank_t) / n             (linear prefix weighting)
or  w_t = 1 / log(1 + rank_t)          (NDCG-style)

This is a *direct* APFD-aware listwise loss. To my knowledge:
  - ListNET, ApproxNDCG, NeuralSort all exist but none weight by
    APFD's specific 1 - r/n prefix kernel.
  - SDC test prio papers don't use listwise at all (mostly BCE + class
    imbalance).

We compare three losses head-to-head: BCE-focal baseline, vanilla PL,
prefix-weighted PL. Predict: prefix-weighted PL boosts APFD (top-K
matters more) while AUC may DROP slightly -- the trade Exp 03 missed.

Saves: exp_G_prefix_pl.json
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
SEARCH_ROOTS=['/kaggle/input',
    os.path.normpath(os.path.join(HERE,'..','..','data','kaggle')),
    os.path.normpath(os.path.join(HERE,'..','..','data')),
    os.path.normpath(os.path.join(HERE,'..','data','kaggle')),
    os.path.normpath(os.path.join(HERE,'..','data')),
    os.getcwd()]
OUTPUT_DIR='/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.normpath(os.path.join(HERE,'..','..','models'))
os.makedirs(OUTPUT_DIR,exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
SEQ_LEN, EPOCHS, BATCH, LR, SWA_START = 197, 60, 256, 5e-4, 40

# ---- feature extraction + model (compressed) ----
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

# ----- the novel losses -----
def plackett_luce_loss(logits, y):
    """Vanilla PL on a batch with explicit positives: sort positives by predicted
    score desc, then -sum_i log sigma_i where sigma_i = exp(s_i) / sum_{j: pos
    AND rank>=i} exp(s_j). y in {0,1}."""
    s = logits  # (B,)
    pos_idx = (y == 1).nonzero(as_tuple=True)[0]
    if pos_idx.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    # sort positives by descending score (the model's predicted order)
    pos_scores = s[pos_idx]
    order = torch.argsort(pos_scores, descending=True)
    pos_idx = pos_idx[order]
    # for each i, denominator = sum exp(s_j) for j in pos_idx[i:]
    # use logsumexp from the tail
    pos_s = s[pos_idx]
    # reverse cumulative logsumexp
    flipped = torch.flip(pos_s, dims=[0])
    cumlse = torch.logcumsumexp(flipped, dim=0)
    cumlse = torch.flip(cumlse, dims=[0])
    nll = -(pos_s - cumlse)
    return nll.mean()

def prefix_weighted_pl_loss(logits, y, weight_kind='linear'):
    """Prefix-weighted Plackett-Luce. Higher weight on top-rank terms,
    where APFD's prefix-bias kernel concentrates."""
    s = logits
    pos_idx = (y == 1).nonzero(as_tuple=True)[0]
    if pos_idx.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    order = torch.argsort(s[pos_idx], descending=True)
    pos_idx = pos_idx[order]
    pos_s = s[pos_idx]
    flipped = torch.flip(pos_s, dims=[0])
    cumlse = torch.flip(torch.logcumsumexp(flipped, dim=0), dims=[0])
    nll = -(pos_s - cumlse)  # per-positive NLL in order
    m = pos_s.numel()
    if weight_kind == 'linear':
        # w_i = (m - i) / m  : highest at i=0 (top), 0 at last
        w = torch.arange(m, device=s.device, dtype=s.dtype)
        w = (m - w) / m
    elif weight_kind == 'ndcg':
        w = 1.0 / torch.log2(torch.arange(m, device=s.device, dtype=s.dtype) + 2.0)
    else:
        w = torch.ones(m, device=s.device, dtype=s.dtype)
    w = w / w.sum() * m  # normalise so mean weight is 1
    return (nll * w).mean()

# ----- training drivers -----
class SWAModel:
    def __init__(self,m): self.m=copy.deepcopy(m); self.n=0
    def update(self,nm):
        self.n+=1; a=1.0/self.n
        for p,q in zip(self.m.parameters(),nm.parameters()): p.data.mul_(1-a).add_(q.data,alpha=a)
    def get(self): return self.m

def train_with_loss(loss_kind, Xtr, ytr, Xv, yv, name='', focal_gamma=2.5):
    print(f"\n--- Train [{loss_kind}] {name} ---")
    model=RoadTransformer().to(DEVICE)
    npos=ytr.sum(); pw=float(len(ytr)-npos)/max(1,npos)
    w=np.where(ytr==1,pw,1.0); samp=WeightedRandomSampler(w,len(w),replacement=True)
    Xt=torch.tensor(Xtr,dtype=torch.float32).permute(0,2,1); yt=torch.tensor(ytr,dtype=torch.float32)
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=BATCH,sampler=samp,num_workers=2,pin_memory=True)
    Xv_t=torch.tensor(Xv,dtype=torch.float32).permute(0,2,1).to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-3); warm=5
    sch=optim.lr_scheduler.LambdaLR(opt,lambda e:(e+1)/warm if e<warm else max(0.01,0.5*(1+math.cos(math.pi*(e-warm)/max(1,EPOCHS-warm)))))
    focal=FocalLoss(gamma=focal_gamma,pw=pw)
    amp=DEVICE.type=='cuda'; scl=GradScaler(enabled=amp)
    best=0; best_st=None; swa=None
    for ep in range(EPOCHS):
        model.train()
        for xb,yb in dl:
            xb=xb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logits=model(xb)
                if loss_kind=='bce_focal':
                    loss=focal(logits,yb)
                elif loss_kind=='pl':
                    loss=plackett_luce_loss(logits.float(),yb)+0.1*focal(logits,yb)
                elif loss_kind=='prefix_pl_linear':
                    loss=prefix_weighted_pl_loss(logits.float(),yb,'linear')+0.1*focal(logits,yb)
                elif loss_kind=='prefix_pl_ndcg':
                    loss=prefix_weighted_pl_loss(logits.float(),yb,'ndcg')+0.1*focal(logits,yb)
                else:
                    raise ValueError(loss_kind)
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
def apfd_at_k(probs, y, k):
    n=len(y); order=np.argsort(-probs)[:k]
    ranks=[i+1 for i,idx in enumerate(order) if y[idx]==1]; m=len(ranks)
    if not m: return 0.0
    return 1 - sum(ranks)/(k*m) + 1/(2*k)

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

def main():
    t0=time.time()
    path=walk_for('Dataset-OOB-0-3') or walk_for('Dataset-OOB-0-5')
    if not path: print("OOB not found"); return
    data=load_oob(path)
    print(f"N={len(data)} FAIL={sum(tc['test_outcome']=='FAIL' for tc in data)}")
    y_all=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data])
    tr,te=train_test_split(np.arange(len(data)),test_size=0.2,stratify=y_all,random_state=42)
    train_data=[data[i] for i in tr]; test_data=[data[i] for i in te]
    Xtr,ytr=prepare(train_data); Xte,yte=prepare(test_data)
    means=Xtr.mean(axis=(0,1)); stds=Xtr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    Xtr_n=(Xtr-means)/stds; Xte_n=(Xte-means)/stds

    results={}
    for loss_kind in ('bce_focal','pl','prefix_pl_linear','prefix_pl_ndcg'):
        model,auc=train_with_loss(loss_kind, Xtr_n, ytr, Xte_n, yte, name=path)
        p=predict(model,Xte_n)
        ap=apfd(p,yte)
        ap_50=apfd_at_k(p,yte,min(50,len(yte)))
        ap_100=apfd_at_k(p,yte,min(100,len(yte)))
        ap_200=apfd_at_k(p,yte,min(200,len(yte)))
        auc_te=roc_auc_score(yte,p)
        results[loss_kind]={'val_auc':auc,'test_auc':float(auc_te),
                             'apfd':float(ap),'apfd@50':float(ap_50),
                             'apfd@100':float(ap_100),'apfd@200':float(ap_200)}
        print(f"  ★ [{loss_kind}] AUC_te={auc_te:.4f} APFD={ap:.4f} @50={ap_50:.4f}"
              f" @100={ap_100:.4f} @200={ap_200:.4f}")

    out=os.path.join(OUTPUT_DIR,'exp_G_prefix_pl.json')
    with open(out,'w') as f: json.dump({'results':results,
                                        'n_train':len(train_data),'n_test':len(test_data)},f,indent=2)
    print(f"\nSaved {out}")
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

if __name__=='__main__': main()
