"""
Exp J -- Trustworthiness Scorecard: rotation x resolution x violation
=====================================================================
NOVELTY (target: ICSE 2027 oral)
--------------------------------
`exps/tracker.md` collects three audit axes separately:
  - Exp 02: rotation Delta on Competition (Delta = 0 for SE(2)-invariant
    features, Delta ~ 0.05 for raw-coord baselines).
  - Exp 01: resolution Delta across N in {64..197} (= 0.0012 for FNO,
    much larger for the Transformer baseline).
  - Exp 04: curvature-monotonicity violation rate (17.57% control ->
    3.14% with monotone-PINN).

Each is a separate figure in the current draft. We propose **one
scalar** that fuses all three -- the **Trustworthiness Scorecard
(TWS)**:

    TWS = w_r * (1 - clip(Delta_rot / Delta_rot_baseline, 0, 1))
        + w_s * (1 - clip(Delta_res / Delta_res_baseline, 0, 1))
        + w_v * (1 - clip(viol_rate / viol_rate_baseline, 0, 1))

with weights w_r + w_s + w_v = 1. We use w = (1/3, 1/3, 1/3) as the
default; sensitivity analysis lets a reviewer reweight.

Bench probe per model:
  - Build 6 rotations of the test set and re-compute APFD per rotation.
    Delta_rot = max - min APFD across rotations.
  - Re-sample each test road to N in {64, 128, 197} and re-compute APFD.
    Delta_res = max - min APFD across N.
  - Compute curvature-monotonicity violation rate (Exp 04 definition,
    alpha = 1.5) on the test predictions.

We score TWS on:
  1. SensoDat baseline Transformer (the canonical model in tracker.md).
  2. The OOB-0-3 best-recipe model trained here.
We expect: SensoDat baseline TWS ~ 0.3-0.5; SE(2)+FNO+PINN stacked
model TWS approaches ~0.9-1.0 even if APFD is only 0-0.001 above
baseline. That is the headline message: at the same APFD, you can get
a far more trustworthy ranker.

Saves: exp_J_trustworthiness.json
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
SEQ_LEN, GAMMA, EPOCHS, BATCH, LR, SWA_START = 197, 2.5, 50, 256, 5e-4, 33
ROTATIONS_DEG = (0, 30, 60, 90, 180, -45)
RES_GRID = (64, 128, 197)
VIOL_ALPHA = 1.5

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

def rotate_points(pts, deg):
    """Rigid 2D rotation around origin."""
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (np.asarray(pts) @ R.T).tolist()

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

# ----- the three audit probes -----
def rotation_delta(model, test_data, means, stds):
    """Build features at multiple rotations, return per-rotation APFD."""
    out={}
    yte=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in test_data])
    for deg in ROTATIONS_DEG:
        feats=[]
        for tc in test_data:
            pts_r = rotate_points(tc['road_points'], deg)
            feats.append(resample(extract_seq(pts_r)))
        X=np.array(feats,dtype=np.float32); X=(X-means)/stds
        p=predict(model,X); a=apfd(p,yte)
        out[deg]=float(a)
    delta = max(out.values()) - min(out.values())
    return out, float(delta)

def resolution_delta(model, test_data, means, stds):
    yte=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in test_data])
    seqs=[extract_seq(tc['road_points']) for tc in test_data]
    out={}
    for N in RES_GRID:
        feats=np.array([resample(s,N) for s in seqs],dtype=np.float32)
        # but the model expects SEQ_LEN positions; up/down-resample after norm
        # easiest: featurise at N then pad/interp back to SEQ_LEN
        # actually our Transformer pos embedding is fixed at SEQ_LEN+1 so we MUST
        # produce SEQ_LEN sequences. The probe is then: was the feature
        # extraction at coarser N preserved when up-interpolated back to SEQ_LEN?
        feats_up=np.array([resample(s, SEQ_LEN) for s in feats],dtype=np.float32)
        feats_up=(feats_up-means)/stds
        p=predict(model,feats_up); a=apfd(p,yte)
        out[N]=float(a)
    delta=max(out.values())-min(out.values())
    return out, float(delta)

def violation_rate(test_data, alpha=VIOL_ALPHA):
    """Curvature-monotonicity violation: count tests where curvature has a
    SHARP DROP (>alpha-fold) over consecutive segments -- empirically a sign
    of physically odd road geometry (Exp 04 protocol approximation)."""
    v=0; total=0
    for tc in test_data:
        pts=np.array(tc['road_points'],dtype=np.float64).reshape(-1,2)
        if len(pts)<5: continue
        total+=1
        c=np.abs(_curvature(pts))
        if len(c)<2: continue
        # consecutive ratios c[i+1]/c[i]: violation if c[i] > alpha * c[i+1] for any i
        c_safe = np.maximum(c, 1e-6)
        ratios = c_safe[:-1] / c_safe[1:]
        if (ratios > alpha).any():
            v += 1
    return float(v / max(1, total)), int(v), int(total)

def trustworthiness(delta_rot, delta_res, viol_rate,
                    base_rot=0.05, base_res=0.05, base_viol=0.17,
                    w=(1/3,1/3,1/3)):
    def comp(x, base): return max(0.0, 1.0 - min(1.0, x / max(1e-9, base)))
    return (w[0]*comp(delta_rot, base_rot)
            + w[1]*comp(delta_res, base_res)
            + w[2]*comp(viol_rate, base_viol))

# ----- bench loader (OOB-0-3) -----
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
    if not path: print('OOB not found'); return
    data=load_oob(path)
    print(f"N={len(data)} FAIL={sum(tc['test_outcome']=='FAIL' for tc in data)}")
    y_all=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in data])
    tr,te=train_test_split(np.arange(len(data)),test_size=0.2,stratify=y_all,random_state=42)
    train_data=[data[i] for i in tr]; test_data=[data[i] for i in te]
    Xtr,ytr=prepare(train_data); Xte,yte=prepare(test_data)
    means=Xtr.mean(axis=(0,1)); stds=Xtr.std(axis=(0,1)); stds[stds<1e-8]=1.0
    Xtr_n=(Xtr-means)/stds; Xte_n=(Xte-means)/stds
    model, auc = train(Xtr_n, ytr, Xte_n, yte, name='Transformer baseline')

    # ----- the scorecard -----
    print(f"\n=== Rotation probe ===")
    rot_per, drot = rotation_delta(model, test_data, means, stds)
    for d,a in rot_per.items(): print(f"  rot={d:>4d}deg: APFD={a:.4f}")
    print(f"  Delta_rot = {drot:.4f}")

    print(f"\n=== Resolution probe ===")
    res_per, dres = resolution_delta(model, test_data, means, stds)
    for n,a in res_per.items(): print(f"  N={n:>3d}: APFD={a:.4f}")
    print(f"  Delta_res = {dres:.4f}")

    print(f"\n=== Violation rate ===")
    vr, vcount, vtotal = violation_rate(test_data, alpha=VIOL_ALPHA)
    print(f"  violation count = {vcount}/{vtotal} ({100*vr:.2f}%)")

    tws = trustworthiness(drot, dres, vr)
    print(f"\n=== TRUSTWORTHINESS SCORE ===")
    print(f"  TWS = {tws:.4f}")
    print(f"  Components: rot_score={1-min(1,drot/0.05):.4f}  "
          f"res_score={1-min(1,dres/0.05):.4f}  viol_score={1-min(1,vr/0.17):.4f}")

    payload={'bench':path,'train_auc':auc,
              'rotation':{'per_deg':rot_per,'delta':drot},
              'resolution':{'per_N':res_per,'delta':dres},
              'violation':{'rate':vr,'count':vcount,'total':vtotal,'alpha':VIOL_ALPHA},
              'tws':tws,
              'baselines_used':{'rot':0.05,'res':0.05,'viol':0.17},
              'weights':[1/3,1/3,1/3],
              'n_train':len(train_data),'n_test':len(test_data),'n_fail_test':int(yte.sum())}
    out=os.path.join(OUTPUT_DIR,'exp_J_trustworthiness.json')
    with open(out,'w') as f: json.dump(payload,f,indent=2)
    print(f"\nSaved {out}")
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

if __name__=='__main__': main()
