"""
Exp I -- Universal model + per-bench temperature scaling
========================================================
NOVELTY (target: ICSE 2027 oral)
--------------------------------
The standard SDC test-prio workflow is one-model-per-bench. Exp A
(this folder) measures the cost of NOT retraining via a transfer matrix.
But what if we could deploy ONE model and adapt to each bench with a
*single scalar*? That's what temperature scaling (Guo et al., 2017) was
designed for.

Setup:
  1. Train a "universal" Transformer on the UNION of bench train splits
     (SensoDat + OOB-{0-1,0-3,0-5} + Scissor + Travel).
  2. For each bench separately, fit a temperature T_b on a held-out
     calibration subset by minimising NLL of sigmoid(logit / T_b).
  3. At test time on bench b, divide logits by T_b. Compare APFD to:
     (a) the universal model with T = 1 (no calibration),
     (b) per-bench retraining (oracle upper bound),
     (c) Exp A's transfer-matrix off-diagonal numbers.

The novelty is positioning temperature scaling -- a calibration trick --
as a per-deployment **adaptation** layer over a frozen ranker. APFD is
*rank invariant under monotone score transforms*, so single-bench T_b
should NOT change within-bench APFD ... unless we additionally fuse
multiple per-bench score offsets (which we explore as a small head
"per-bench bias" b_b).

Why this matters for the oral:
  - It's the cheapest possible adaptation layer.
  - If it works, it kills the per-bench-retraining cost story in the
    SDC literature.
  - If it doesn't work, the result is itself interesting: SDC
    deployment cannot be reduced to calibration -- requires re-tuning.

Saves: exp_I_universal_T.json
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
SEQ_LEN, GAMMA, EPOCHS, BATCH, LR, SWA_START = 197, 2.5, 60, 256, 5e-4, 40

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

def train_geom(Xtr,ytr,Xv,yv,name=''):
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

def predict_logits(model,X):
    Xt=torch.tensor(X,dtype=torch.float32).permute(0,2,1).to(DEVICE); model.eval().to(DEVICE)
    with torch.no_grad(): return model(Xt).cpu().numpy()
def apfd(probs,y):
    n=len(y); order=np.argsort(-probs)
    ranks=[i+1 for i,idx in enumerate(order) if y[idx]==1]; m=len(ranks)
    return (1-sum(ranks)/(n*m)+1/(2*n)) if (n and m) else 1.0

# ----- loaders -----
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
def find_scissor():
    seen=set()
    for r in SEARCH_ROOTS:
        if not r or not os.path.isdir(r) or r in seen: continue
        seen.add(r)
        for dp,_,fn in os.walk(r):
            if sum(1 for x in fn if x.endswith('-test.json'))>=50: return dp
    return None
def find_travel():
    seen=set()
    for r in SEARCH_ROOTS:
        if not r or not os.path.isdir(r) or r in seen: continue
        seen.add(r)
        for dp,dn,_ in os.walk(r):
            if os.path.basename(dp)=='competition':
                for d in dn[:3]:
                    if 'generator' in d.lower(): return dp
    return None
def find_sensodat():
    seen=set()
    for r in SEARCH_ROOTS:
        if not r or not os.path.isdir(r) or r in seen: continue
        seen.add(r)
        for dp,dn,_ in os.walk(r):
            if 'sensodat' in os.path.basename(dp).lower():
                if any(d.endswith('.json') for d in os.listdir(dp)) or len(dn)>=3: return dp
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
def load_scissor(root):
    if not root: return []
    data=[]
    for fp in sorted(glob.glob(os.path.join(root,'*-test.json'))):
        try:
            with open(fp) as f: tc=json.load(f)
        except Exception: continue
        if tc.get('is_valid',True) is False: continue
        pts=tc.get('interpolated_road_points') or tc.get('road_points'); out=tc.get('test_outcome')
        if not pts or out not in ('FAIL','PASS'): continue
        data.append({'_id':os.path.basename(fp),'road_points':pts,'test_outcome':out})
    return data
def load_travel(root,max_files=None):
    if not root: return []
    data=[]
    for camp in sorted(os.listdir(root)):
        cp=os.path.join(root,camp)
        if not os.path.isdir(cp): continue
        for fp in glob.glob(os.path.join(cp,'test.*.json')):
            try:
                with open(fp) as f: tc=json.load(f)
            except Exception: continue
            if not tc.get('is_valid',True): continue
            pts=tc.get('interpolated_points') or tc.get('road_points'); out=tc.get('test_outcome')
            if not pts or out not in ('FAIL','PASS'): continue
            data.append({'_id':f'{camp}/{os.path.basename(fp)}','road_points':pts,'test_outcome':out})
            if max_files and len(data)>=max_files: break
        if max_files and len(data)>=max_files: break
    return data
def load_sensodat(root,max_files=None):
    if not root: return []
    data=[]
    for fp in glob.iglob(os.path.join(root,'**','*.json'),recursive=True):
        try:
            with open(fp) as f: tc=json.load(f)
        except Exception: continue
        if not tc.get('is_valid',True): continue
        pts=tc.get('road_points') or tc.get('interpolated_road_points') or tc.get('interpolated_points')
        out=tc.get('test_outcome')
        if not pts or out not in ('FAIL','PASS'): continue
        data.append({'_id':fp[len(root):].lstrip(os.sep),'road_points':pts,'test_outcome':out})
        if max_files and len(data)>=max_files: break
    return data
def prepare(data):
    X=np.array([resample(extract_seq(tc['road_points'])) for tc in data],dtype=np.float32)
    y=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data],dtype=np.int64)
    return X,y

def split_3way(data, cal_frac=0.15, test_frac=0.2, seed=42):
    y=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data])
    idx=np.arange(len(data))
    rest, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    yrest=y[rest]
    tr, cal = train_test_split(rest, test_size=cal_frac/(1-test_frac), stratify=yrest, random_state=seed)
    return [data[i] for i in tr], [data[i] for i in cal], [data[i] for i in te]

# ---- temperature fit ----
def fit_temperature(logits, y, lr=0.01, n_iter=200):
    logits_t = torch.tensor(logits, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    T = torch.nn.Parameter(torch.ones(1, device=DEVICE))
    bias = torch.nn.Parameter(torch.zeros(1, device=DEVICE))
    opt = torch.optim.LBFGS([T, bias], lr=lr, max_iter=n_iter)
    def closure():
        opt.zero_grad()
        scaled = (logits_t + bias) / T.clamp(min=0.05)
        loss = F.binary_cross_entropy_with_logits(scaled, y_t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=0.05)), float(bias.detach())

def main():
    t0=time.time()
    bench={}
    # load (with size caps to fit one training pass)
    print('[load] SensoDat'); r=find_sensodat(); bench['sensodat']=load_sensodat(r,max_files=6000) if r else []
    print(f'  N={len(bench["sensodat"])}')
    for tag in ('0-1','0-3','0-5'):
        p=walk_for(f'Dataset-OOB-{tag}'); bench[f'oob_{tag}']=load_oob(p) if p else []
        print(f'  oob_{tag}: N={len(bench[f"oob_{tag}"])}')
    rs=find_scissor(); bench['scissor']=load_scissor(rs) if rs else []
    print(f'  scissor: N={len(bench["scissor"])}')
    rt=find_travel(); bench['travel']=load_travel(rt,max_files=8000) if rt else []
    print(f'  travel: N={len(bench["travel"])}')

    keep=[k for k,v in bench.items() if len(v)>=200 and sum(tc['test_outcome']=='FAIL' for tc in v)>=20]
    print(f"using: {keep}")

    # 3-way split per bench
    splits={k: dict(zip(['train','cal','test'], split_3way(bench[k]))) for k in keep}

    # Universal training set = union of all train splits
    universal_train = []
    for k in keep: universal_train.extend(splits[k]['train'])
    print(f"\nUniversal training pool: {len(universal_train)}")

    # Use a sample of all-bench cals together as val for training
    all_cal=[]
    for k in keep: all_cal.extend(splits[k]['cal'])
    print(f"Universal val pool: {len(all_cal)}")

    Xtr_u, ytr_u = prepare(universal_train)
    Xval_u, yval_u = prepare(all_cal)
    mu_u=Xtr_u.mean(axis=(0,1)); sd_u=Xtr_u.std(axis=(0,1)); sd_u[sd_u<1e-8]=1.0
    Xtr_u_n=(Xtr_u-mu_u)/sd_u; Xval_u_n=(Xval_u-mu_u)/sd_u
    print(f"Pool fail rate: {ytr_u.mean():.3f}")
    universal, _ = train_geom(Xtr_u_n, ytr_u, Xval_u_n, yval_u, name='UNIVERSAL')

    # Evaluate per bench: (a) raw, (b) with per-bench T+bias, (c) per-bench retraining
    results={}
    for k in keep:
        cal_data = splits[k]['cal']; test_data = splits[k]['test']
        # universal logits
        Xcal,ycal = prepare(cal_data); Xte,yte = prepare(test_data)
        Xcal_u = (Xcal - mu_u)/sd_u; Xte_u = (Xte - mu_u)/sd_u
        lo_cal = predict_logits(universal, Xcal_u)
        lo_te  = predict_logits(universal, Xte_u)
        # raw
        p_raw = 1.0/(1.0+np.exp(-lo_te))
        apfd_raw = apfd(p_raw, yte); auc_raw = roc_auc_score(yte, p_raw)
        # temperature + bias
        T_b, bias_b = fit_temperature(lo_cal, ycal)
        lo_te_T = (lo_te + bias_b) / max(T_b, 0.05)
        p_T = 1.0/(1.0+np.exp(-lo_te_T))
        apfd_T = apfd(p_T, yte); auc_T = roc_auc_score(yte, p_T)
        # per-bench retraining (oracle)
        Xtr_b, ytr_b = prepare(splits[k]['train'])
        mu_b=Xtr_b.mean(axis=(0,1)); sd_b=Xtr_b.std(axis=(0,1)); sd_b[sd_b<1e-8]=1.0
        Xtr_b_n=(Xtr_b-mu_b)/sd_b; Xte_b_n=(Xte-mu_b)/sd_b
        Xcal_b_n=(Xcal-mu_b)/sd_b
        m_b, _ = train_geom(Xtr_b_n, ytr_b, Xcal_b_n, ycal, name=f'PER-BENCH {k}')
        p_b = 1.0/(1.0+np.exp(-predict_logits(m_b, Xte_b_n)))
        apfd_b = apfd(p_b, yte); auc_b = roc_auc_score(yte, p_b)
        results[k]={'apfd_raw':float(apfd_raw),'auc_raw':float(auc_raw),
                     'apfd_temp_bias':float(apfd_T),'auc_temp_bias':float(auc_T),
                     'apfd_per_bench':float(apfd_b),'auc_per_bench':float(auc_b),
                     'T':float(T_b),'bias':float(bias_b),
                     'n_train':len(splits[k]['train']),'n_test':len(test_data)}
        print(f"  {k:>12s}: raw={apfd_raw:.4f}  +T={apfd_T:.4f}  per-bench={apfd_b:.4f}"
              f"  (T={T_b:.3f}, bias={bias_b:.3f})")

    payload={'config':{'epochs':EPOCHS,'gamma':GAMMA},
              'benches':results,
              'universal_pool_n':len(universal_train),
              'universal_fail_rate':float(ytr_u.mean())}
    out=os.path.join(OUTPUT_DIR,'exp_I_universal_T.json')
    with open(out,'w') as f: json.dump(payload,f,indent=2)
    print(f"\nSaved {out}")
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

if __name__=='__main__': main()
