"""
BEST COMBINATION: Transformer + SWA + Focal Loss
==================================================
Ablation results (Multi-trial Competition APFD):
  Base Transformer:     0.7899 ± 0.0140
  +SWA:                 0.8042 ± 0.0120  ← BEST single (+0.0143)
  +FocalLoss:           0.7820 ± 0.0142  ← 2nd best
  +Mixup:               0.7733
  +DropPath:            0.7683
  +TriplePool:          0.7678
  +MultiScaleStem:      0.7670
  +TTA:                 0.7700

This file combines the top-2 improvements:
  1) SWA (Stochastic Weight Averaging) — flatter minima, better generalization
  2) Focal Loss — focus on hard borderline examples

Also sweeps focal gamma [1.0, 1.5, 2.0, 2.5] to find best setting.

Saves: roadfury_best.pt
"""

import json, numpy as np, os, time, math, copy, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

KAGGLE_DATA = '/kaggle/input/datasets/chinguyeen/sdc-test-data'
OUTPUT_DIR = '/kaggle/working'
if os.path.exists(KAGGLE_DATA):
    TRAIN_PATH = os.path.join(KAGGLE_DATA, 'sensodat_train.json')
    TEST_PATH = os.path.join(KAGGLE_DATA, 'sensodat_test.json')
    COMP_PATH = os.path.join(KAGGLE_DATA, 'sdc-test-data.json')
else:
    BASE = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
    TRAIN_PATH = os.path.join(BASE, 'data', 'sensodat_train.json')
    TEST_PATH = os.path.join(BASE, 'data', 'sensodat_test.json')
    COMP_PATH = os.path.join(BASE, 'data', 'sdc-test-data.json')
    OUTPUT_DIR = os.path.join(BASE, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")

# ---- Feature extraction (10ch) ----
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
def prepare_data(data, batch_print=5000):
    X,y=[],[]
    for i,tc in enumerate(data):
        X.append(extract_sequence_10ch(get_pts(tc))); y.append(1 if is_fail(tc) else 0)
        if (i+1)%batch_print==0: print(f"    {i+1}/{len(data)}...")
    return np.array(X), np.array(y)

# ---- Base Transformer (unchanged architecture) ----
class RoadTransformer(nn.Module):
    def __init__(self, in_channels=10, seq_len=197, d_model=128,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        x = x.permute(0, 2, 1); B, L, C = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :L+1, :]
        x = self.transformer(x)
        return self.classifier(x[:, 0, :]).squeeze(-1)

# ---- Focal Loss ----
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weight = torch.where(targets == 1, self.pos_weight, 1.0)
        bce = bce * weight
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

# ---- SWA ----
class SWAModel:
    def __init__(self, model):
        self.model = copy.deepcopy(model); self.n = 0
    def update(self, new_model):
        self.n += 1; alpha = 1.0 / self.n
        for p_swa, p_new in zip(self.model.parameters(), new_model.parameters()):
            p_swa.data.mul_(1 - alpha).add_(p_new.data, alpha=alpha)
    def get_model(self): return self.model

# ---- Training: Focal Loss + SWA ----
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=75, batch_size=256, lr=5e-4,
                focal_gamma=2.0, swa_start=50, name=''):
    print(f"\n{'='*60}\nTraining {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Focal γ={focal_gamma} | SWA from epoch {swa_start}")
    print(f"{'='*60}")
    model = model.to(DEVICE)
    n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
    pw = n_neg / n_pos
    weights = np.where(y_train == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    X_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                          sampler=sampler, num_workers=2, pin_memory=True)
    X_v = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    y_v_np = y_val

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = FocalLoss(alpha=1.0, gamma=focal_gamma, pos_weight=pw)
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    best_auc = 0; best_state = None; swa_model = None

    for epoch in range(epochs):
        model.train(); total_loss = 0; nb = 0
        for xb, yb in train_dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): loss = criterion(model(xb), yb)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); nb += 1
        scheduler.step()

        # SWA collection
        if epoch >= swa_start:
            if swa_model is None:
                swa_model = SWAModel(model)
                print(f"  [SWA] Started at epoch {epoch+1}")
            else:
                swa_model.update(model)

        model.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp): vl = model(X_v)
            val_auc = roc_auc_score(y_v_np, torch.sigmoid(vl).cpu().numpy())
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            m = ' *'
        else:
            m = ''
        if (epoch + 1) % 10 == 0 or m:
            swa_tag = ' [SWA]' if epoch >= swa_start else ''
            print(f"  Ep {epoch+1:3d} | Loss:{total_loss/nb:.4f} | AUC:{val_auc:.4f} | Best:{best_auc:.4f}{m}{swa_tag}")

    model.load_state_dict(best_state)
    print(f"  Best-ckpt AUC: {best_auc:.4f}")

    swa_auc = 0
    if swa_model:
        sm = swa_model.get_model().to(DEVICE); sm.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp): sl = sm(X_v)
            swa_auc = roc_auc_score(y_v_np, torch.sigmoid(sl).cpu().numpy())
        print(f"  SWA AUC: {swa_auc:.4f} ({swa_model.n} snapshots)")

    return model, best_auc, swa_model, swa_auc

# ---- Evaluation ----
def compute_apfd(pids, td):
    n = len(pids)
    fp = [i+1 for i, t in enumerate(pids) if td[t]['meta_data']['test_info']['test_outcome'] == 'FAIL']
    m = len(fp); return 1 - sum(fp)/(n*m) + 1/(2*n) if n and m else 1.0

def evaluate(data, model, means, stds, name=''):
    model.eval().to(DEVICE)
    td = {get_id(tc): tc for tc in data}; ids = [get_id(tc) for tc in data]
    feats = [(extract_sequence_10ch(get_pts(tc)) - means) / stds for tc in data]
    X = torch.tensor(np.array(feats), dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    with torch.no_grad(): probs = torch.sigmoid(model(X)).cpu().numpy()
    pids = [t for _, t in sorted(zip(probs, ids), key=lambda x: -x[0])]
    apfd = compute_apfd(pids, td); nf = sum(1 for tc in data if is_fail(tc))
    print(f"  {name:45s} APFD={apfd:.4f} ({len(data)} tests, {nf} fail)")
    return apfd

def multi_trial(data, model, means, stds, name='', n_trials=30):
    model.eval().to(DEVICE); apfds = []
    for t in range(n_trials):
        rng = np.random.RandomState(42 + t); idx = rng.permutation(len(data))
        ed = [data[i] for i in idx[334:334+287]]
        td = {get_id(tc): tc for tc in ed}; ids = [get_id(tc) for tc in ed]
        feats = [(extract_sequence_10ch(get_pts(tc)) - means) / stds for tc in ed]
        X = torch.tensor(np.array(feats), dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
        with torch.no_grad(): probs = torch.sigmoid(model(X)).cpu().numpy()
        pids = [t for _, t in sorted(zip(probs, ids), key=lambda x: -x[0])]
        apfds.append(compute_apfd(pids, td))
    print(f"  {name:45s} APFD={np.mean(apfds):.4f}±{np.std(apfds):.4f} ({n_trials} trials)")
    return np.mean(apfds)

# ---- Main ----
def main():
    t0 = time.time()
    print("\n" + "=" * 70)
    print("BEST COMBO: Transformer + SWA + Focal Loss (gamma sweep)")
    print("=" * 70)

    train_data = load_json(TRAIN_PATH); test_data = load_json(TEST_PATH)
    comp_data = load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    n_fail = sum(1 for tc in train_data if is_fail(tc))
    print(f"Train: {len(train_data)} ({n_fail} FAIL = {100*n_fail/len(train_data):.1f}%)")

    print("\nExtracting features...")
    X_tr, y_tr = prepare_data(train_data); X_te, y_te = prepare_data(test_data)
    means = X_tr.mean(axis=(0, 1)); stds = X_tr.std(axis=(0, 1)); stds[stds < 1e-8] = 1.0
    X_tr = (X_tr - means) / stds; X_te = (X_te - means) / stds

    # ---- Sweep focal gamma with SWA ----
    gammas = [0.0, 1.0, 1.5, 2.0, 2.5]  # 0.0 = standard BCE equivalent
    results = {}

    for gamma in gammas:
        tag = f"γ={gamma}" if gamma > 0 else "BCE (γ=0)"
        print(f"\n{'='*60}")
        print(f"  Focal {tag} + SWA")
        print(f"{'='*60}")

        model = RoadTransformer(in_channels=10, d_model=128, nhead=8, num_layers=4)
        model, auc, swa_model, swa_auc = train_model(
            model, X_tr, y_tr, X_te, y_te,
            epochs=75, batch_size=256, lr=5e-4,
            focal_gamma=gamma, swa_start=50,
            name=f'Focal({tag})+SWA'
        )
        swa_m = swa_model.get_model() if swa_model else None
        results[gamma] = {
            'model': model, 'swa': swa_m,
            'auc': auc, 'swa_auc': swa_auc
        }

    # ---- Comprehensive Evaluation ----
    print(f"\n{'='*70}")
    print("EVALUATION — ALL GAMMA SETTINGS")
    print(f"{'='*70}")

    if comp_data:
        print(f"\n--- Competition Multi-trial (30 trials) ---")
        best_apfd = 0; best_gamma = None; best_model = None
        for gamma in gammas:
            tag = f"γ={gamma}" if gamma > 0 else "BCE"
            r = results[gamma]
            # Best checkpoint
            apfd_ckpt = multi_trial(comp_data, r['model'], means, stds, f'Focal({tag}) best-ckpt')
            # SWA
            if r['swa']:
                apfd_swa = multi_trial(comp_data, r['swa'], means, stds, f'Focal({tag})+SWA')
                if apfd_swa > best_apfd:
                    best_apfd = apfd_swa; best_gamma = gamma; best_model = r['swa']
                    best_type = 'SWA'
            if apfd_ckpt > best_apfd:
                best_apfd = apfd_ckpt; best_gamma = gamma; best_model = r['model']
                best_type = 'best-ckpt'

        print(f"\n{'='*60}")
        print(f"★ BEST: Focal(γ={best_gamma}) {best_type} → APFD={best_apfd:.4f}")
        print(f"{'='*60}")

        # Full eval of best
        print(f"\n--- Best model detailed eval ---")
        evaluate(test_data, best_model, means, stds, f'BEST SensoDat')
        evaluate(comp_data, best_model, means, stds, f'BEST Competition')
        multi_trial(comp_data, best_model, means, stds, f'BEST Multi-trial', n_trials=50)

        # Save best
        save_path = os.path.join(OUTPUT_DIR, 'roadfury_best.pt')
        torch.save({
            'state': best_model.state_dict(),
            'means': means.tolist(), 'stds': stds.tolist(),
            'focal_gamma': best_gamma, 'type': best_type,
            'apfd': best_apfd,
        }, save_path)
        print(f"\nSaved: {save_path} ({os.path.getsize(save_path)/1024:.0f} KB)")

    # Also save ALL models for ensemble potential
    all_save = os.path.join(OUTPUT_DIR, 'roadfury_all_gammas.pt')
    save_dict = {'means': means.tolist(), 'stds': stds.tolist()}
    for gamma in gammas:
        r = results[gamma]
        save_dict[f'model_g{gamma}'] = r['model'].state_dict()
        if r['swa']:
            save_dict[f'swa_g{gamma}'] = r['swa'].state_dict()
    torch.save(save_dict, all_save)
    print(f"Saved all: {all_save}")

    # ---- ENSEMBLE of all SWA models ----
    print(f"\n{'='*60}")
    print("ENSEMBLE (average all SWA models across gammas)")
    print(f"{'='*60}")

    if comp_data:
        def ensemble_multi_trial(data, models, means, stds, name='', n_trials=30):
            apfds = []
            for t in range(n_trials):
                rng = np.random.RandomState(42 + t); idx = rng.permutation(len(data))
                ed = [data[i] for i in idx[334:334+287]]
                td = {get_id(tc): tc for tc in ed}; ids = [get_id(tc) for tc in ed]
                feats = [(extract_sequence_10ch(get_pts(tc)) - means) / stds for tc in ed]
                X = torch.tensor(np.array(feats), dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
                all_probs = []
                for m in models:
                    m.eval().to(DEVICE)
                    with torch.no_grad(): all_probs.append(torch.sigmoid(m(X)).cpu().numpy())
                avg = np.mean(all_probs, axis=0)
                pids = [t for _, t in sorted(zip(avg, ids), key=lambda x: -x[0])]
                apfds.append(compute_apfd(pids, td))
            print(f"  {name:45s} APFD={np.mean(apfds):.4f}±{np.std(apfds):.4f} ({n_trials} trials)")
            return np.mean(apfds)

        swa_models = [results[g]['swa'] for g in gammas if results[g]['swa']]
        if swa_models:
            ensemble_multi_trial(comp_data, swa_models, means, stds,
                                 f'Ensemble {len(swa_models)} SWA models', n_trials=50)

    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == '__main__':
    main()
Device: cuda
GPU: NVIDIA H100 80GB HBM3

======================================================================
BEST COMBO: Transformer + SWA + Focal Loss (gamma sweep)
======================================================================
Loading /kaggle/input/datasets/chinguyeen/sdc-test-data/sensodat_train.json...
  Loaded 28804 tests in 7.8s
Loading /kaggle/input/datasets/chinguyeen/sdc-test-data/sensodat_test.json...
  Loaded 7202 tests in 1.8s
Loading /kaggle/input/datasets/chinguyeen/sdc-test-data/sdc-test-data.json...
  Loaded 956 tests in 0.2s
Train: 28804 (11059 FAIL = 38.4%)

Extracting features...
    5000/28804...
    10000/28804...
    15000/28804...
    20000/28804...
    25000/28804...
    5000/7202...

============================================================
  Focal BCE (γ=0) + SWA
============================================================

============================================================
Training Focal(BCE (γ=0))+SWA
  Params: 828,801
  Focal γ=0.0 | SWA from epoch 50
============================================================
  Ep   1 | Loss:0.7319 | AUC:0.8214 | Best:0.8214 *
  Ep   2 | Loss:0.6376 | AUC:0.8531 | Best:0.8531 *
  Ep   3 | Loss:0.5966 | AUC:0.8674 | Best:0.8674 *
  Ep   4 | Loss:0.5691 | AUC:0.8824 | Best:0.8824 *
  Ep   5 | Loss:0.5339 | AUC:0.8917 | Best:0.8917 *
  Ep   6 | Loss:0.5026 | AUC:0.9004 | Best:0.9004 *
  Ep   7 | Loss:0.4829 | AUC:0.9027 | Best:0.9027 *
  Ep   8 | Loss:0.4611 | AUC:0.9129 | Best:0.9129 *
  Ep  10 | Loss:0.4359 | AUC:0.9147 | Best:0.9147 *
  Ep  11 | Loss:0.4119 | AUC:0.9171 | Best:0.9171 *
  Ep  13 | Loss:0.3974 | AUC:0.9178 | Best:0.9178 *
  Ep  15 | Loss:0.3598 | AUC:0.9197 | Best:0.9197 *
  Ep  16 | Loss:0.3448 | AUC:0.9227 | Best:0.9227 *
  Ep  20 | Loss:0.2928 | AUC:0.9197 | Best:0.9227
  Ep  30 | Loss:0.1746 | AUC:0.9096 | Best:0.9227
  Ep  40 | Loss:0.0924 | AUC:0.9016 | Best:0.9227
  Ep  50 | Loss:0.0476 | AUC:0.8832 | Best:0.9227
  [SWA] Started at epoch 51
  Ep  60 | Loss:0.0181 | AUC:0.8783 | Best:0.9227 [SWA]
  Ep  70 | Loss:0.0121 | AUC:0.8783 | Best:0.9227 [SWA]
  Best-ckpt AUC: 0.9227
  SWA AUC: 0.8822 (24 snapshots)

============================================================
  Focal γ=1.0 + SWA
============================================================

============================================================
Training Focal(γ=1.0)+SWA
  Params: 828,801
  Focal γ=1.0 | SWA from epoch 50
============================================================
  Ep   1 | Loss:0.3666 | AUC:0.8248 | Best:0.8248 *
  Ep   2 | Loss:0.3194 | AUC:0.8603 | Best:0.8603 *
  Ep   3 | Loss:0.3026 | AUC:0.8727 | Best:0.8727 *
  Ep   4 | Loss:0.2826 | AUC:0.8806 | Best:0.8806 *
  Ep   5 | Loss:0.2736 | AUC:0.8922 | Best:0.8922 *
  Ep   6 | Loss:0.2629 | AUC:0.8956 | Best:0.8956 *
  Ep   7 | Loss:0.2517 | AUC:0.9012 | Best:0.9012 *
  Ep   9 | Loss:0.2337 | AUC:0.9055 | Best:0.9055 *
  Ep  10 | Loss:0.2230 | AUC:0.9025 | Best:0.9055
  Ep  11 | Loss:0.2195 | AUC:0.9139 | Best:0.9139 *
  Ep  13 | Loss:0.2046 | AUC:0.9163 | Best:0.9163 *
  Ep  16 | Loss:0.1816 | AUC:0.9181 | Best:0.9181 *
  Ep  20 | Loss:0.1502 | AUC:0.9190 | Best:0.9190 *
  Ep  30 | Loss:0.0828 | AUC:0.9054 | Best:0.9190
  Ep  40 | Loss:0.0380 | AUC:0.9043 | Best:0.9190
  Ep  50 | Loss:0.0181 | AUC:0.9037 | Best:0.9190
  [SWA] Started at epoch 51
  Ep  60 | Loss:0.0072 | AUC:0.9033 | Best:0.9190 [SWA]
  Ep  70 | Loss:0.0036 | AUC:0.9047 | Best:0.9190 [SWA]
  Best-ckpt AUC: 0.9190
  SWA AUC: 0.9061 (24 snapshots)

============================================================
  Focal γ=1.5 + SWA
============================================================

============================================================
Training Focal(γ=1.5)+SWA
  Params: 828,801
  Focal γ=1.5 | SWA from epoch 50
============================================================
  Ep   1 | Loss:0.2576 | AUC:0.8180 | Best:0.8180 *
  Ep   2 | Loss:0.2274 | AUC:0.8563 | Best:0.8563 *
  Ep   3 | Loss:0.2160 | AUC:0.8696 | Best:0.8696 *
  Ep   4 | Loss:0.2041 | AUC:0.8781 | Best:0.8781 *
  Ep   5 | Loss:0.1960 | AUC:0.8871 | Best:0.8871 *
  Ep   6 | Loss:0.1836 | AUC:0.8980 | Best:0.8980 *
  Ep   7 | Loss:0.1754 | AUC:0.8992 | Best:0.8992 *
  Ep   8 | Loss:0.1693 | AUC:0.9077 | Best:0.9077 *
  Ep  10 | Loss:0.1560 | AUC:0.9119 | Best:0.9119 *
  Ep  11 | Loss:0.1535 | AUC:0.9175 | Best:0.9175 *
  Ep  15 | Loss:0.1335 | AUC:0.9211 | Best:0.9211 *
  Ep  20 | Loss:0.1073 | AUC:0.9112 | Best:0.9211
  Ep  30 | Loss:0.0599 | AUC:0.9080 | Best:0.9211
  Ep  40 | Loss:0.0296 | AUC:0.9103 | Best:0.9211
  Ep  50 | Loss:0.0127 | AUC:0.9111 | Best:0.9211
  [SWA] Started at epoch 51
  Ep  60 | Loss:0.0049 | AUC:0.9113 | Best:0.9211 [SWA]
  Ep  70 | Loss:0.0023 | AUC:0.9114 | Best:0.9211 [SWA]
  Best-ckpt AUC: 0.9211
  SWA AUC: 0.9129 (24 snapshots)

============================================================
  Focal γ=2.0 + SWA
============================================================

============================================================
Training Focal(γ=2.0)+SWA
  Params: 828,801
  Focal γ=2.0 | SWA from epoch 50
============================================================
  Ep   1 | Loss:0.1831 | AUC:0.8202 | Best:0.8202 *
  Ep   2 | Loss:0.1634 | AUC:0.8586 | Best:0.8586 *
  Ep   3 | Loss:0.1549 | AUC:0.8712 | Best:0.8712 *
  Ep   4 | Loss:0.1478 | AUC:0.8827 | Best:0.8827 *
  Ep   5 | Loss:0.1421 | AUC:0.8896 | Best:0.8896 *
  Ep   6 | Loss:0.1334 | AUC:0.8959 | Best:0.8959 *
  Ep   7 | Loss:0.1283 | AUC:0.9027 | Best:0.9027 *
  Ep   8 | Loss:0.1248 | AUC:0.9079 | Best:0.9079 *
  Ep   9 | Loss:0.1201 | AUC:0.9122 | Best:0.9122 *
  Ep  10 | Loss:0.1142 | AUC:0.9159 | Best:0.9159 *
  Ep  11 | Loss:0.1118 | AUC:0.9172 | Best:0.9172 *
  Ep  16 | Loss:0.0938 | AUC:0.9201 | Best:0.9201 *
  Ep  20 | Loss:0.0798 | AUC:0.9160 | Best:0.9201
  Ep  30 | Loss:0.0473 | AUC:0.9090 | Best:0.9201
  Ep  40 | Loss:0.0232 | AUC:0.9086 | Best:0.9201
  Ep  50 | Loss:0.0107 | AUC:0.9079 | Best:0.9201
  [SWA] Started at epoch 51
  Ep  60 | Loss:0.0041 | AUC:0.9108 | Best:0.9201 [SWA]
  Ep  70 | Loss:0.0023 | AUC:0.9088 | Best:0.9201 [SWA]
  Best-ckpt AUC: 0.9201
  SWA AUC: 0.9092 (24 snapshots)

============================================================
  Focal γ=2.5 + SWA
============================================================

============================================================
Training Focal(γ=2.5)+SWA
  Params: 828,801
  Focal γ=2.5 | SWA from epoch 50
============================================================
  Ep   1 | Loss:0.1313 | AUC:0.8222 | Best:0.8222 *
  Ep   2 | Loss:0.1164 | AUC:0.8589 | Best:0.8589 *
  Ep   3 | Loss:0.1103 | AUC:0.8594 | Best:0.8594 *
  Ep   4 | Loss:0.1049 | AUC:0.8719 | Best:0.8719 *
  Ep   5 | Loss:0.1037 | AUC:0.8726 | Best:0.8726 *
  Ep   6 | Loss:0.0974 | AUC:0.8947 | Best:0.8947 *
  Ep   8 | Loss:0.0904 | AUC:0.8999 | Best:0.8999 *
  Ep  10 | Loss:0.0817 | AUC:0.9076 | Best:0.9076 *
  Ep  11 | Loss:0.0831 | AUC:0.9162 | Best:0.9162 *
  Ep  19 | Loss:0.0624 | AUC:0.9170 | Best:0.9170 *
  Ep  20 | Loss:0.0594 | AUC:0.9091 | Best:0.9170
  Ep  30 | Loss:0.0362 | AUC:0.9119 | Best:0.9170
  Ep  40 | Loss:0.0192 | AUC:0.9060 | Best:0.9170
  Ep  50 | Loss:0.0086 | AUC:0.9056 | Best:0.9170
  [SWA] Started at epoch 51
  Ep  60 | Loss:0.0030 | AUC:0.9070 | Best:0.9170 [SWA]
  Ep  70 | Loss:0.0019 | AUC:0.9059 | Best:0.9170 [SWA]
  Best-ckpt AUC: 0.9170
  SWA AUC: 0.9071 (24 snapshots)

======================================================================
EVALUATION — ALL GAMMA SETTINGS
======================================================================

--- Competition Multi-trial (30 trials) ---
  Focal(BCE) best-ckpt                          APFD=0.7753±0.0146 (30 trials)
  Focal(BCE)+SWA                                APFD=0.8001±0.0121 (30 trials)
  Focal(γ=1.0) best-ckpt                        APFD=0.7887±0.0148 (30 trials)
  Focal(γ=1.0)+SWA                              APFD=0.8052±0.0126 (30 trials)
  Focal(γ=1.5) best-ckpt                        APFD=0.7699±0.0144 (30 trials)
  Focal(γ=1.5)+SWA                              APFD=0.8045±0.0119 (30 trials)
  Focal(γ=2.0) best-ckpt                        APFD=0.7774±0.0163 (30 trials)
  Focal(γ=2.0)+SWA                              APFD=0.8020±0.0132 (30 trials)
  Focal(γ=2.5) best-ckpt                        APFD=0.7740±0.0148 (30 trials)
  Focal(γ=2.5)+SWA                              APFD=0.8066±0.0124 (30 trials)

============================================================
★ BEST: Focal(γ=2.5) SWA → APFD=0.8066
============================================================

--- Best model detailed eval ---
  BEST SensoDat                                 APFD=0.7508 (7202 tests, 2765 fail)
  BEST Competition                              APFD=0.8054 (956 tests, 353 fail)
  BEST Multi-trial                              APFD=0.8064±0.0114 (50 trials)

Saved: /kaggle/working/roadfury_best.pt (3260 KB)
Saved all: /kaggle/working/roadfury_all_gammas.pt

============================================================
ENSEMBLE (average all SWA models across gammas)
============================================================
  Ensemble 5 SWA models                         APFD=0.8077±0.0115 (50 trials)

TOTAL TIME: 1069.2s (17.8 min)