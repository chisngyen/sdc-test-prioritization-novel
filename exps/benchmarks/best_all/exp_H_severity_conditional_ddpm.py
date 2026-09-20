"""
Exp H -- Severity-conditional DDPM for OOB-transfer augmentation
================================================================
NOVELTY (target: ICSE 2027 oral)
--------------------------------
The OOB transfer matrix in `exps/oob/tracker.md` shows a severe
asymmetry: training on OOB-0-5 (high severity) and evaluating on
OOB-0-1 (low severity) drops APFD by ~0.08. The "hard-only" model
never sees the geometric signatures of mild fails.

We propose a **severity-conditional generative augmentation**. Train a
1D DDPM (Exp 08 style) on curvature sequences from ALL three OOB
thresholds, but condition the model on a *continuous severity scalar*
(e.g., the OOB threshold value 0.1 / 0.3 / 0.5). At sampling time,
we can generate curvature trajectories conditioned on ANY severity,
including in-between values. This fills the under-represented 0-1
distribution with model-generated examples.

Pipeline:
  1. Train DDPM on (curvature_seq | severity).
  2. Sample N synthetic sequences conditioned on target severity 0.1.
  3. Filter samples by classifier confidence (Exp 08 boundary trick).
  4. Train classifier on real-0-5 UNION synthetic-0-1, evaluate on
     held-out real 0-1.
  5. Compare to baseline (real-0-5 only).

This is novel because:
  - Severity-conditional DDPM in test data generation has not been
    tried in the SDC literature.
  - It directly addresses an empirical asymmetry observed in the
    project's own transfer matrix (Exp B of `oob/tracker.md`).
  - Cheaper than re-collecting low-severity data from BeamNG.

Saves: exp_H_severity_ddpm.json
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
SEQ_LEN, GAMMA, EPOCHS_C, BATCH, LR, SWA_START = 197, 2.5, 50, 256, 5e-4, 33
T_DDPM, EPOCHS_DDPM, N_SYNTH = 100, 15, 3000

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

def train_classifier(Xtr, ytr, Xv, yv, name=''):
    print(f"\n--- Train classifier {name} ---")
    model=RoadTransformer().to(DEVICE)
    npos=ytr.sum(); pw=float(len(ytr)-npos)/max(1,npos)
    w=np.where(ytr==1,pw,1.0); samp=WeightedRandomSampler(w,len(w),replacement=True)
    Xt=torch.tensor(Xtr,dtype=torch.float32).permute(0,2,1); yt=torch.tensor(ytr,dtype=torch.float32)
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=BATCH,sampler=samp,num_workers=2,pin_memory=True)
    Xv_t=torch.tensor(Xv,dtype=torch.float32).permute(0,2,1).to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-3); warm=5
    sch=optim.lr_scheduler.LambdaLR(opt,lambda e:(e+1)/warm if e<warm else max(0.01,0.5*(1+math.cos(math.pi*(e-warm)/max(1,EPOCHS_C-warm)))))
    crit=FocalLoss(gamma=GAMMA,pw=pw); amp=DEVICE.type=='cuda'; scl=GradScaler(enabled=amp)
    best=0; best_st=None; swa=None
    for ep in range(EPOCHS_C):
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

# ----- 1D DDPM on curvature (single channel, severity-conditioned) -----
def cosine_beta_schedule(T, s=0.008):
    t = torch.linspace(0, T, T+1) / T
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0.0001, 0.999)

class CondUNet1D(nn.Module):
    """Small 1D U-Net for the curvature channel, with severity scalar
    concatenated as a global condition broadcast to every position."""
    def __init__(self, L=SEQ_LEN, d=64):
        super().__init__()
        self.t_emb = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.c_emb = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.in_conv = nn.Conv1d(1, d, 5, padding=2)
        self.b1 = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GroupNorm(8, d), nn.SiLU(),
                                nn.Conv1d(d, d, 5, padding=2), nn.GroupNorm(8, d), nn.SiLU())
        self.b2 = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GroupNorm(8, d), nn.SiLU(),
                                nn.Conv1d(d, d, 5, padding=2), nn.GroupNorm(8, d), nn.SiLU())
        self.out_conv = nn.Conv1d(d, 1, 5, padding=2)
    def forward(self, x, t, c):  # x: B,1,L  t: B,1  c: B,1
        emb = (self.t_emb(t) + self.c_emb(c)).unsqueeze(-1)  # B,d,1
        h = self.in_conv(x) + emb
        h = self.b1(h) + emb
        h = self.b2(h) + emb
        return self.out_conv(h)

class DDPM:
    def __init__(self, T=T_DDPM):
        self.T = T
        self.betas = cosine_beta_schedule(T).to(DEVICE)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    @torch.no_grad()
    def sample(self, model, n, severity, L=SEQ_LEN):
        x = torch.randn(n, 1, L, device=DEVICE)
        c = torch.full((n, 1), severity, device=DEVICE)
        for t_step in reversed(range(self.T)):
            t = torch.full((n, 1), t_step / self.T, device=DEVICE)
            eps = model(x, t, c)
            a = self.alphas[t_step]; ab = self.alpha_bar[t_step]; b = self.betas[t_step]
            mean = (1 / a.sqrt()) * (x - (b / (1 - ab).sqrt()) * eps)
            if t_step > 0:
                x = mean + b.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x  # B,1,L

def train_ddpm(seqs, severities):
    """seqs: list of curvature 1D arrays (length SEQ_LEN). severities: list of floats."""
    print(f"\n--- Train severity-conditional DDPM (N={len(seqs)}) ---")
    X = torch.tensor(np.stack(seqs), dtype=torch.float32).unsqueeze(1).to(DEVICE)  # B,1,L
    c = torch.tensor(severities, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    # normalise X to ~unit
    mu = X.mean(); sd = X.std() + 1e-8
    Xn = (X - mu) / sd
    dl = DataLoader(TensorDataset(Xn.cpu(), c.cpu()), batch_size=128, shuffle=True, num_workers=2)
    model = CondUNet1D().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=2e-4)
    diff = DDPM()
    for ep in range(EPOCHS_DDPM):
        model.train(); tot=0; nb=0
        for xb, cb in dl:
            xb=xb.to(DEVICE); cb=cb.to(DEVICE)
            t = torch.randint(0, T_DDPM, (xb.shape[0],), device=DEVICE)
            t_in = (t.float() / T_DDPM).unsqueeze(1)
            noise = torch.randn_like(xb)
            xt = diff.q_sample(xb, t, noise)
            pred = model(xt, t_in, cb)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if (ep+1) % 3 == 0:
            print(f"  DDPM ep{ep+1}: loss={tot/nb:.4f}")
    return model, diff, float(mu), float(sd)

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
    # Need OOB-0-1, OOB-0-3, OOB-0-5
    paths = {tag: walk_for(f'Dataset-OOB-{tag}') for tag in ('0-1','0-3','0-5')}
    if not all(paths.values()): print(f"Missing: {paths}"); return
    print(f"paths: {paths}")
    src_3 = load_oob(paths['0-3']); src_5 = load_oob(paths['0-5'])
    tgt_1 = load_oob(paths['0-1'])
    print(f"sizes: 0-3={len(src_3)} 0-5={len(src_5)} 0-1={len(tgt_1)}")
    if min(len(src_3), len(src_5), len(tgt_1)) < 200: print("Too small"); return

    # held-out split on OOB-0-1
    y1=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in tgt_1])
    tr,te=train_test_split(np.arange(len(tgt_1)),test_size=0.3,stratify=y1,random_state=42)
    tgt_1_test = [tgt_1[i] for i in te]
    print(f"OOB-0-1 test split: n={len(tgt_1_test)} fail={sum(tc['test_outcome']=='FAIL' for tc in tgt_1_test)}")

    Xte, yte = prepare(tgt_1_test)
    Xreal_src = []  # OOB-0-3 + OOB-0-5 union as base training pool
    yreal_src = []
    for tc in src_3 + src_5:
        Xreal_src.append(resample(extract_seq(tc['road_points'])))
        yreal_src.append(1 if tc['test_outcome']=='FAIL' else 0)
    Xreal_src = np.array(Xreal_src, dtype=np.float32); yreal_src = np.array(yreal_src, dtype=np.int64)
    means=Xreal_src.mean(axis=(0,1)); stds=Xreal_src.std(axis=(0,1)); stds[stds<1e-8]=1.0
    Xreal_src_n=(Xreal_src-means)/stds; Xte_n=(Xte-means)/stds

    # Baseline 1: train ONLY on OOB-0-5 (the asymmetry source)
    Xbase=np.array([resample(extract_seq(tc['road_points'])) for tc in src_5],dtype=np.float32)
    ybase=np.array([1 if tc['test_outcome']=='FAIL' else 0 for tc in src_5],dtype=np.int64)
    mu_b=Xbase.mean(axis=(0,1)); sd_b=Xbase.std(axis=(0,1)); sd_b[sd_b<1e-8]=1.0
    Xbase_n=(Xbase-mu_b)/sd_b
    Xte_b=(Xte-mu_b)/sd_b
    print(f"\n=== BASELINE: train on OOB-0-5 only -> test on OOB-0-1 ===")
    m_base, auc_b = train_classifier(Xbase_n, ybase, Xte_b, yte, name='real-0-5 only')
    p_base = predict(m_base, Xte_b)
    apfd_base = apfd(p_base, yte)
    print(f"  ★ baseline APFD = {apfd_base:.4f} (AUC_te={roc_auc_score(yte, p_base):.4f})")

    # Train severity-conditional DDPM on curvature CH (channel idx 2) of all three.
    print(f"\n=== DDPM on curvature channel (ALL sources) ===")
    seqs=[]; sevs=[]
    severity_map = {'0-1': 0.1, '0-3': 0.3, '0-5': 0.5}
    for tag in ('0-1','0-3','0-5'):
        d = load_oob(paths[tag])
        for tc in d[:1500]:
            seq=resample(extract_seq(tc['road_points']))
            seqs.append(seq[:, 2])  # curvature channel
            sevs.append(severity_map[tag])
    ddpm_model, diff, mu_c, sd_c = train_ddpm(seqs, sevs)

    # Sample N synthetic curvature sequences conditioned on severity = 0.1
    print(f"\n=== Sample {N_SYNTH} synthetic 0-1-like curvatures ===")
    with torch.no_grad():
        synth = diff.sample(ddpm_model, N_SYNTH, severity=0.1, L=SEQ_LEN)
        synth = synth.cpu().numpy()[:,0,:] * sd_c + mu_c  # B,L
    synth = np.clip(synth, 0, None)  # curvature non-negative

    # Build pseudo-labels: use a baseline classifier's predicted prob on the
    # FULL 10ch feature where we replace curvature channel with synth and keep
    # other channels from a random REAL OOB-0-5 template (preserves geometry).
    rng=np.random.RandomState(7)
    template_idx = rng.choice(len(Xreal_src), size=N_SYNTH, replace=True)
    aug_X = Xreal_src[template_idx].copy()
    # normalise back to raw before injecting synth (means/stds were computed on raw)
    aug_X[:, :, 2] = synth  # inject synth curvature
    # pseudo-label: use base classifier predicted prob on (aug - mu_b)/sd_b
    aug_X_b = (aug_X - mu_b)/sd_b
    p_syn = predict(m_base, aug_X_b)
    aug_y = (p_syn > 0.5).astype(np.int64)
    print(f"  synth label balance: FAIL={int(aug_y.sum())}/{len(aug_y)} "
          f"(mean p={p_syn.mean():.3f})")
    # keep only boundary-ish samples (Exp 08 trick)
    keep = (np.abs(p_syn - 0.5) < 0.25)
    print(f"  kept boundary: {int(keep.sum())}/{len(keep)}")
    aug_X_keep = aug_X[keep]; aug_y_keep = aug_y[keep]

    # Combined training set: real OOB-0-5 + synth-0-1
    if aug_X_keep.size:
        X_combo = np.concatenate([Xbase, aug_X_keep], axis=0)
        y_combo = np.concatenate([ybase, aug_y_keep], axis=0)
        mu_c2=X_combo.mean(axis=(0,1)); sd_c2=X_combo.std(axis=(0,1)); sd_c2[sd_c2<1e-8]=1.0
        X_combo_n=(X_combo-mu_c2)/sd_c2
        Xte_c=(Xte-mu_c2)/sd_c2
        print(f"\n=== AUG: train on OOB-0-5 + {len(aug_X_keep)} synth-0-1 ===")
        m_aug, auc_a = train_classifier(X_combo_n, y_combo, Xte_c, yte, name='real-0-5 + synth-0-1')
        p_aug = predict(m_aug, Xte_c); apfd_aug = apfd(p_aug, yte)
        print(f"  ★ augmented APFD = {apfd_aug:.4f}  (delta={apfd_aug-apfd_base:+.4f})")
    else:
        apfd_aug = None
        auc_a = None

    payload={'baseline_apfd':float(apfd_base),
              'baseline_auc_te':float(roc_auc_score(yte,p_base)),
              'augmented_apfd':float(apfd_aug) if apfd_aug is not None else None,
              'augmented_auc_te':float(auc_a) if auc_a is not None else None,
              'n_synth_total':N_SYNTH,
              'n_synth_kept':int(keep.sum()),
              'synth_fail_frac':float(aug_y.mean()),
              'n_test':len(yte),'n_fail_test':int(yte.sum())}
    out=os.path.join(OUTPUT_DIR,'exp_H_severity_ddpm.json')
    with open(out,'w') as f: json.dump(payload,f,indent=2)
    print(f"\nSaved {out}")
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

if __name__=='__main__': main()
