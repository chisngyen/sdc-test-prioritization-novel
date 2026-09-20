"""
Exp A -- Cross-bench k x k transfer matrix
==========================================
NOVELTY (target: ICSE 2027 oral)
--------------------------------
The SDC test-prioritization community evaluates within-bench (Birchler,
Panichella, Riccio). Nobody publishes a cross-bench transfer matrix
because nobody trains the same recipe across all public benches. Yet
the headline question of any "general SDC ranker" claim is: does a model
trained on bench i generalise to bench j?

We build a k x k matrix on k = 4 geometry benches (SensoDat, OOB-0-3,
Scissor pool, Travel pool), one cell per (train_src, eval_tgt) pair.
The diagonal is within-bench (sanity), the off-diagonal is the
empirical cost of distribution shift on APFD.

Hypothesis from `exps/tracker.md` (Section "OOB-Regression transfer"):
OOB-0-3 was the universal source there. We test whether that pattern
extends to non-OOB benches. If yes, OOB-0-3 becomes "the bench to
train on" for any deployable recipe.

Saves: exp_A_transfer_matrix.json
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
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")

SEQ_LEN, GAMMA, EPOCHS, BATCH, LR, SWA_START, N_TRIALS = 197, 2.5, 75, 256, 5e-4, 50, 30

# =================================================================
# GEOMETRY pipeline (feature extraction + model + train + APFD)
# =================================================================
def _curvature(pts):
    n = len(pts); curv = np.zeros(n - 2)
    for i in range(n - 2):
        x1,y1=pts[i]; x2,y2=pts[i+1]; x3,y3=pts[i+2]
        a=math.sqrt((x2-x1)**2+(y2-y1)**2); b=math.sqrt((x3-x2)**2+(y3-y2)**2); c=math.sqrt((x3-x1)**2+(y3-y1)**2)
        s=0.5*(a+b+c); at=s*(s-a)*(s-b)*(s-c)
        if at<=1e-10: curv[i]=0.0
        else: R=a*b*c/(4*math.sqrt(at)); curv[i]=1.0/R if R>0 else 0.0
    return curv

def extract_seq(pts_raw):
    pts=np.array(pts_raw,dtype=np.float64).reshape(-1,2); n=len(pts)
    if n<3: pts=np.vstack([pts]*3)[:max(3,n)]; n=len(pts)
    diffs=np.diff(pts,axis=0); seg=np.linalg.norm(diffs,axis=1)
    seg_full=np.pad(seg,(0,1),mode='edge')
    ang=np.arctan2(diffs[:,1],diffs[:,0]); ac=np.diff(ang); ac=(ac+np.pi)%(2*np.pi)-np.pi
    abs_ac_full=np.pad(np.abs(ac),(1,1),mode='constant')
    curv=np.abs(_curvature(pts)); curv_full=np.pad(curv,(1,1),mode='constant')
    curv_d=np.pad(np.diff(curv_full),(0,1),mode='constant')
    cumd=np.cumsum(seg_full); cumd_n=cumd/(cumd[-1]+1e-8)
    h=np.pad(ang,(0,1),mode='edge'); hs=np.sin(h); hc=np.cos(h)
    rel=np.linspace(0,1,n)
    w=11; ls=np.zeros(n); hw=w//2
    for i in range(n):
        s,e=max(0,i-hw),min(n,i+hw+1); ls[i]=np.std(curv_full[s:e])
    curv_a=np.pad(np.diff(curv_d),(0,1),mode='constant')
    return np.column_stack([seg_full,abs_ac_full,curv_full,curv_d,cumd_n,hs,hc,rel,ls,curv_a]).astype(np.float32)

def resample(seq, L=SEQ_LEN):
    n,c=seq.shape
    if n==L: return seq
    xo=np.linspace(0,1,n); xn=np.linspace(0,1,L)
    out=np.empty((L,c),dtype=np.float32)
    for ch in range(c): out[:,ch]=np.interp(xn,xo,seq[:,ch])
    return out

class RoadTransformer(nn.Module):
    def __init__(self, in_ch=10, L=SEQ_LEN, d=128, h=8, n_layers=4, dff=512, dr=0.1):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(in_ch,d),nn.LayerNorm(d),nn.GELU())
        self.cls=nn.Parameter(torch.randn(1,1,d)*0.02)
        self.pos=nn.Parameter(torch.randn(1,L+1,d)*0.02)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=h,dim_feedforward=dff,dropout=dr,
                                        activation='gelu',batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(enc,num_layers=n_layers)
        self.cls_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,1))
    def forward(self,x):
        x=x.permute(0,2,1); B,L,_=x.shape
        x=self.proj(x); x=torch.cat([self.cls.expand(B,-1,-1),x],dim=1)
        x=x+self.pos[:,:L+1,:]; x=self.tr(x)
        return self.cls_head(x[:,0,:]).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self,a=1.0,g=2.0,pw=1.0): super().__init__(); self.a,self.g,self.pw=a,g,pw
    def forward(self,logits,targets):
        bce=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
        w=torch.where(targets==1,self.pw,1.0); bce=bce*w
        pt=torch.where(targets==1,torch.sigmoid(logits),1-torch.sigmoid(logits))
        return (self.a*(1-pt)**self.g*bce).mean()

class SWAModel:
    def __init__(self,m): self.m=copy.deepcopy(m); self.n=0
    def update(self,nm):
        self.n+=1; a=1.0/self.n
        for p,q in zip(self.m.parameters(),nm.parameters()): p.data.mul_(1-a).add_(q.data,alpha=a)
    def get(self): return self.m

def train_geom(Xtr,ytr,Xv,yv,name='',epochs=EPOCHS,bs=BATCH,lr=LR,gamma=GAMMA,swa_st=SWA_START):
    print(f"\n--- Train {name} | g={gamma} | SWA@{swa_st} ---")
    model=RoadTransformer().to(DEVICE)
    npos=ytr.sum(); pw=float(len(ytr)-npos)/max(1,npos)
    w=np.where(ytr==1,pw,1.0); samp=WeightedRandomSampler(w,len(w),replacement=True)
    Xt=torch.tensor(Xtr,dtype=torch.float32).permute(0,2,1); yt=torch.tensor(ytr,dtype=torch.float32)
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=bs,sampler=samp,num_workers=2,pin_memory=True)
    Xv_t=torch.tensor(Xv,dtype=torch.float32).permute(0,2,1).to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3); warm=5
    sch=optim.lr_scheduler.LambdaLR(opt,lambda e:(e+1)/warm if e<warm
                                     else max(0.01,0.5*(1+math.cos(math.pi*(e-warm)/max(1,epochs-warm)))))
    crit=FocalLoss(gamma=gamma,pw=pw); amp=DEVICE.type=='cuda'; scl=GradScaler(enabled=amp)
    best_auc=0; best_state=None; swa=None
    for ep in range(epochs):
        model.train(); tl=0; nb=0
        for xb,yb in dl:
            xb=xb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp): loss=crit(model(xb),yb)
            scl.scale(loss).backward(); scl.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0); scl.step(opt); scl.update()
            tl+=loss.item(); nb+=1
        sch.step()
        if ep>=swa_st:
            if swa is None: swa=SWAModel(model)
            else: swa.update(model)
        model.eval()
        with torch.no_grad():
            with autocast(enabled=amp): vl=model(Xv_t)
            try: auc=roc_auc_score(yv,torch.sigmoid(vl).cpu().numpy())
            except: auc=0.5
        if auc>best_auc: best_auc=auc; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if (ep+1)%15==0: print(f"  Ep {ep+1:3d} L={tl/nb:.4f} AUC={auc:.4f} Best={best_auc:.4f}")
    model.load_state_dict(best_state)
    return (swa.get() if swa else model), float(best_auc)

def compute_apfd(pids, td):
    n=len(pids); fp=[i+1 for i,t in enumerate(pids) if td[t]['test_outcome']=='FAIL']
    return 1-sum(fp)/(n*len(fp))+1/(2*n) if n and fp else 1.0

def predict(model,Xn):
    Xt=torch.tensor(Xn,dtype=torch.float32).permute(0,2,1).to(DEVICE); model.eval().to(DEVICE)
    with torch.no_grad(): return torch.sigmoid(model(Xt)).cpu().numpy()

def multi_trial_apfd(eval_data,model,means,stds,n_trials=N_TRIALS,name=''):
    ss=min(max(50,int(0.3*len(eval_data))),len(eval_data))
    feats=np.array([resample(extract_seq(tc['road_points'])) for tc in eval_data],dtype=np.float32)
    feats=(feats-means)/stds
    apfds=[]
    for t in range(n_trials):
        rng=np.random.RandomState(42+t); idx=rng.permutation(len(eval_data))[:ss]
        ed=[eval_data[i] for i in idx]; td={tc['_id']:tc for tc in ed}; ids=[tc['_id'] for tc in ed]
        probs=predict(model,feats[idx])
        pids=[t for _,t in sorted(zip(probs,ids),key=lambda x:-x[0])]
        apfds.append(compute_apfd(pids,td))
    print(f"  {name:50s} APFD={np.mean(apfds):.4f}+/-{np.std(apfds):.4f}")
    return float(np.mean(apfds)),float(np.std(apfds))

def prepare(data):
    X=np.array([resample(extract_seq(tc['road_points'])) for tc in data],dtype=np.float32)
    y=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data],dtype=np.int64)
    return X,y

# =================================================================
# LOADERS (one per bench)
# =================================================================
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

def find_sensodat():
    seen=set()
    for r in SEARCH_ROOTS:
        if not r or not os.path.isdir(r) or r in seen: continue
        seen.add(r)
        for dp,dn,_ in os.walk(r):
            if 'sensodat' in os.path.basename(dp).lower():
                if any(d.endswith('.json') for d in os.listdir(dp)) or len(dn)>=3: return dp
    return None

def load_sensodat(root,max_files=None):
    if not root: return []
    data=[]; seen=0
    for fp in glob.iglob(os.path.join(root,'**','*.json'),recursive=True):
        seen+=1
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
            pts=tc.get('interpolated_points') or tc.get('road_points')
            out=tc.get('test_outcome')
            if not pts or out not in ('FAIL','PASS'): continue
            data.append({'_id':f'{camp}/{os.path.basename(fp)}','road_points':pts,'test_outcome':out,'campaign':camp})
            if max_files and len(data)>=max_files: break
        if max_files and len(data)>=max_files: break
    return data

# =================================================================
# Experiment A
# =================================================================
def split_train_test(data,frac=0.2,seed=42):
    y=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data])
    tr,te=train_test_split(np.arange(len(data)),test_size=frac,stratify=y,random_state=seed)
    return [data[i] for i in tr],[data[i] for i in te]

def main():
    t0=time.time()
    # Cap each bench so total wall-clock stays under ~90 min on T4.
    benches={}
    print("[load] SensoDat")
    r=find_sensodat(); benches['sensodat']=load_sensodat(r,max_files=8000) if r else []
    print(f"  N={len(benches['sensodat'])}")
    print("[load] OOB-0-3")
    r=walk_for('Dataset-OOB-0-3'); benches['oob_0_3']=load_oob(r) if r else []
    print(f"  N={len(benches['oob_0_3'])}")
    print("[load] Scissor")
    r=find_scissor(); benches['scissor']=load_scissor(r) if r else []
    print(f"  N={len(benches['scissor'])}")
    print("[load] Travel")
    r=find_travel(); benches['travel']=load_travel(r,max_files=10000) if r else []
    print(f"  N={len(benches['travel'])}")

    keep=[k for k,v in benches.items() if len(v)>=100 and sum(tc['test_outcome']=='FAIL' for tc in v)>=20]
    print(f"\nUsing benches: {keep}")

    # split each into 80/20 once
    splits={}
    for k in keep:
        tr,te=split_train_test(benches[k]); splits[k]={'train':tr,'test':te}

    # train one model per src
    trained={}
    for src in keep:
        Xtr,ytr=prepare(splits[src]['train']); Xte,yte=prepare(splits[src]['test'])
        means=Xtr.mean(axis=(0,1)); stds=Xtr.std(axis=(0,1)); stds[stds<1e-8]=1.0
        Xtr_n=(Xtr-means)/stds; Xte_n=(Xte-means)/stds
        model,auc=train_geom(Xtr_n,ytr,Xte_n,yte,name=f'src={src}')
        trained[src]={'model':model,'means':means,'stds':stds,'auc':auc}

    # k x k transfer matrix
    matrix={s:{} for s in keep}
    print(f"\n{'='*70}\nTRANSFER MATRIX (rows=train_src, cols=eval_tgt)\n{'='*70}")
    for src in keep:
        for tgt in keep:
            tgt_test=splits[tgt]['test']
            apfd,std=multi_trial_apfd(tgt_test,trained[src]['model'],
                                       trained[src]['means'],trained[src]['stds'],
                                       n_trials=N_TRIALS,name=f'{src} -> {tgt}')
            matrix[src][tgt]={'apfd':apfd,'apfd_std':std,'n_test':len(tgt_test)}

    # Pretty matrix
    print(f"\n{'src \\\\ tgt':>15s}"+''.join(f"{t:>16s}" for t in keep))
    for s in keep:
        row=f"{s:>15s}"
        for t in keep:
            v=matrix[s][t]['apfd']; mark='*' if s==t else ' '
            row+=f"   {v:.4f}{mark}      "
        print(row)

    payload={'recipe':f'Transformer+SWA+Focal(g={GAMMA}), ep={EPOCHS}',
             'benches':keep,
             'sizes':{k:{'train':len(splits[k]['train']),'test':len(splits[k]['test']),
                          'fail':sum(tc['test_outcome']=='FAIL' for tc in benches[k])}
                       for k in keep},
             'aucs':{s:trained[s]['auc'] for s in keep},
             'matrix':matrix}
    out=os.path.join(OUTPUT_DIR,'exp_A_transfer_matrix.json')
    with open(out,'w') as f: json.dump(payload,f,indent=2)
    print(f"\nSaved {out}")
    print(f"TOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

if __name__=='__main__': main()
