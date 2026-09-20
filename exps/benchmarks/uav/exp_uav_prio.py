"""
UAV-Testing-Competition: prioritization with the SensoDat recipe
================================================================
Loads the JSON produced by ``gen_uav_dataset.py`` and runs the project's
canonical recipe -- Transformer + SWA + Focal (gamma=2.5) -- with the
same multi-trial APFD protocol used across all benchmarks (30 trials,
|S| = max(50, 0.3 * |test|), seeds 42..71).

Featurization
-------------
Each test = planned trajectory (path) + a list of obstacles. We resample
the path to ``SEQ_LEN`` waypoints and emit 10 obstacle-context channels
per waypoint:

    [ d, log1p(d), dx, dy, sin(yaw), cos(yaw),
      size_l, size_w, size_h, n_within_10m ]

where ``d`` is the 3D distance from the waypoint to the *nearest*
obstacle bounding box, ``dx/dy`` point to that obstacle's centre, and
``n_within_10m`` is the count of obstacles whose centre lies within 10m
of the waypoint. This keeps the (batch, 10, 197) shape so the existing
``RoadTransformer`` checkpoint architecture is reused as-is.

Protocol
--------
- 80/20 stratified split by ``test_outcome``.
- Train 75 epochs, batch 256, lr 5e-4, SWA from epoch 50.
- Two model snapshots: best-val-AUC (``best``) and SWA average (``swa``).
- Report APFD mean +/- std across 30 trials for both; also the all-data
  single-shot APFD.

Output
------
``models/uav_prio_<mode>_results.json`` + console scoreboard.
"""
import os, sys, json, math, copy, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT  = os.path.join(ROOT, 'models')
os.makedirs(OUT, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    # PyTorch 2.9 + Python 3.14 CPU build segfaults inside MultiHeadAttention
    # under threaded MKL. Pinning to 1 thread is the cheapest workaround.
    torch.set_num_threads(1)

SEQ_LEN   = 197
GAMMA     = 2.5
EPOCHS    = 75
BATCH     = 256
LR        = 5e-4
SWA_START = 50
N_TRIALS  = 30


# ---- Featurization --------------------------------------------------
def resample_path(path, target=SEQ_LEN):
    p = np.asarray(path, dtype=np.float64)
    if len(p) < 2:
        p = np.vstack([p, p]) if len(p) else np.zeros((2, 3))
    # densify before resampling so equal-arc spacing makes sense
    segs = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(segs)])
    if s[-1] < 1e-6:
        return np.tile(p[0], (target, 1)).astype(np.float64)
    snew = np.linspace(0, s[-1], target)
    out = np.empty((target, 3))
    for k in range(3):
        out[:, k] = np.interp(snew, s, p[:, k])
    return out


def _box_dist3d(p, obs):
    cx, cy, cz = obs['x'], obs['y'], obs['z'] + obs['h'] / 2.0
    hl, hw, hh = obs['l'] / 2.0, obs['w'] / 2.0, obs['h'] / 2.0
    th = math.radians(obs['r'])
    c, s = math.cos(-th), math.sin(-th)
    dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
    rx = dx * c - dy * s
    ry = dx * s + dy * c
    qx = max(0.0, abs(rx) - hl)
    qy = max(0.0, abs(ry) - hw)
    qz = max(0.0, abs(dz) - hh)
    return math.sqrt(qx * qx + qy * qy + qz * qz)


def featurize(rec):
    path = resample_path(rec['path'])
    obs_list = rec['obstacles']
    feats = np.zeros((SEQ_LEN, 10), dtype=np.float32)
    if not obs_list:
        return feats
    for i, p in enumerate(path):
        # find nearest obstacle
        dists = [_box_dist3d(p, o) for o in obs_list]
        j = int(np.argmin(dists))
        d = dists[j]; o = obs_list[j]
        feats[i, 0] = d
        feats[i, 1] = math.log1p(d)
        feats[i, 2] = p[0] - o['x']
        feats[i, 3] = p[1] - o['y']
        th = math.radians(o['r'])
        feats[i, 4] = math.sin(th)
        feats[i, 5] = math.cos(th)
        feats[i, 6] = o['l']
        feats[i, 7] = o['w']
        feats[i, 8] = o['h']
        # neighbour count within 10m of waypoint centre
        cnt = 0
        for o2 in obs_list:
            if math.hypot(p[0] - o2['x'], p[1] - o2['y']) <= 10.0:
                cnt += 1
        feats[i, 9] = cnt
    return feats


# ---- Model (lifted from exp_best_all.py) ----------------------------
class RoadTransformer(nn.Module):
    def __init__(self, in_channels=10, seq_len=SEQ_LEN, d_model=128,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model),
                                         nn.LayerNorm(d_model), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, L, C = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :L + 1, :]
        x = self.transformer(x)
        return self.classifier(x[:, 0, :]).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.alpha, self.gamma, self.pos_weight = alpha, gamma, pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        w = torch.where(targets == 1, self.pos_weight, 1.0)
        bce = bce * w
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class SWA:
    def __init__(self, model):
        self.model = copy.deepcopy(model); self.n = 0
    def update(self, new):
        self.n += 1; a = 1.0 / self.n
        for p_s, p_n in zip(self.model.parameters(), new.parameters()):
            p_s.data.mul_(1 - a).add_(p_n.data, alpha=a)


def train(X_tr, y_tr, X_va, y_va):
    model = RoadTransformer().to(DEVICE)
    n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
    pw = float(n_neg) / max(1, n_pos)
    weights = np.where(y_tr == 1, pw, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    Xt = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, sampler=sampler,
                    num_workers=0, pin_memory=True)
    Xv_arr = torch.tensor(X_va, dtype=torch.float32).permute(0, 2, 1)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    warmup = 5
    def lrf(ep):
        if ep < warmup: return (ep + 1) / warmup
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, EPOCHS - warmup))))
    sch = optim.lr_scheduler.LambdaLR(opt, lrf)
    crit = FocalLoss(alpha=1.0, gamma=GAMMA, pos_weight=pw)
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    best_auc = 0.0; best_state = None; swa = None
    for ep in range(EPOCHS):
        model.train(); tot = 0.0; nb = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sch.step()
        if ep >= SWA_START:
            swa = SWA(model) if swa is None else (swa.update(model) or swa)
        model.eval()
        with torch.no_grad():
            probs = []
            for s in range(0, len(Xv_arr), 64):
                xv = Xv_arr[s:s+64].to(DEVICE)
                with autocast(enabled=use_amp):
                    probs.append(torch.sigmoid(model(xv)).cpu().numpy())
            vp = np.concatenate(probs)
            try: vauc = roc_auc_score(y_va, vp)
            except Exception: vauc = 0.5
        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 15 == 0:
            print(f"  ep {ep+1:3d} | loss {tot/nb:.4f} | val AUC {vauc:.4f} | best {best_auc:.4f}")
    model.load_state_dict(best_state)
    return model, (swa.model.to(DEVICE) if swa else None), best_auc


def predict(model, X, bs=64):
    Xt = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xt), bs):
            out.append(torch.sigmoid(model(Xt[s:s+bs].to(DEVICE))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def apfd(order, td):
    n = len(order)
    fp = [i + 1 for i, t in enumerate(order) if td[t]['test_outcome'] == 'FAIL']
    m = len(fp)
    return 1 - sum(fp) / (n * m) + 1 / (2 * n) if (n and m) else 1.0


def multi_trial_apfd(eval_data, X_eval, model, means, stds, name=''):
    sample = max(50, int(0.3 * len(eval_data)))
    sample = min(sample, len(eval_data))
    Xn = (X_eval - means) / stds
    apfds = []
    for t in range(N_TRIALS):
        rng = np.random.RandomState(42 + t)
        idx = rng.permutation(len(eval_data))[:sample]
        ed = [eval_data[i] for i in idx]
        td = {tc['_id']: tc for tc in ed}
        ids = [tc['_id'] for tc in ed]
        probs = predict(model, Xn[idx])
        order = [ids[k] for k in np.argsort(-probs)]
        apfds.append(apfd(order, td))
    apfds = np.array(apfds)
    print(f"  [{name}] APFD = {apfds.mean():.4f} +/- {apfds.std():.4f} "
          f"(min {apfds.min():.4f}, max {apfds.max():.4f})")
    return apfds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['surrogate', 'sim'], default='surrogate')
    ap.add_argument('--dataset', type=str, default=None,
                    help='Override path to uav_dataset_<mode>.json')
    ap.add_argument('--batch', type=int, default=None,
                    help='Override batch size (default: 256 on GPU, 32 on CPU; '
                         'torch 2.9 + py3.14 CPU segfaults at large batches).')
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()
    global BATCH, EPOCHS, SWA_START
    if args.batch is not None:
        BATCH = args.batch
    elif DEVICE.type == 'cpu':
        BATCH = 32
    if args.epochs is not None:
        EPOCHS = args.epochs
        SWA_START = max(1, int(EPOCHS * 2 / 3))

    ds_path = args.dataset or os.path.join(ROOT, 'data', 'uav',
                                            f'uav_dataset_{args.mode}.json')
    if not os.path.exists(ds_path):
        sys.exit(f"Dataset not found: {ds_path}. Run gen_uav_dataset.py first.")
    print(f"Device: {DEVICE}")
    with open(ds_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    n_fail = sum(1 for r in data if r['test_outcome'] == 'FAIL')
    print(f"[data] {len(data)} tests | FAIL {n_fail} ({100*n_fail/len(data):.1f}%)")

    t0 = time.time()
    X = np.stack([featurize(r) for r in data]).astype(np.float32)
    y = np.array([1 if r['test_outcome'] == 'FAIL' else 0 for r in data],
                 dtype=np.float32)
    print(f"[feat] X shape {X.shape}  | featurize {time.time()-t0:.1f}s")

    idx = np.arange(len(data))
    tr, va = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]

    means = X_tr.reshape(-1, X.shape[-1]).mean(axis=0)
    stds  = X_tr.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    X_trn = (X_tr - means) / stds
    X_van = (X_va - means) / stds

    best_model, swa_model, val_auc = train(X_trn, y_tr, X_van, y_va)
    print(f"[train] best val AUC = {val_auc:.4f}  ({time.time()-t0:.1f}s)")

    eval_data = [data[i] for i in va]
    X_eval = X[va]
    apfd_best = multi_trial_apfd(eval_data, X_eval, best_model, means, stds, 'best')
    apfd_swa  = None
    if swa_model is not None:
        apfd_swa = multi_trial_apfd(eval_data, X_eval, swa_model, means, stds, 'swa')

    # All-data single-shot APFD (no subsampling)
    Xn = (X_eval - means) / stds
    td_all = {r['_id']: r for r in eval_data}
    probs_all = predict(best_model, Xn)
    order_all = [eval_data[k]['_id'] for k in np.argsort(-probs_all)]
    apfd_all_best = apfd(order_all, td_all)
    print(f"[all-data] APFD (best) = {apfd_all_best:.4f}")

    res = dict(
        mode=args.mode, n_tests=len(data), n_fail=int(n_fail),
        val_auc=float(val_auc),
        apfd_best_mean=float(apfd_best.mean()),
        apfd_best_std=float(apfd_best.std()),
        apfd_swa_mean=float(apfd_swa.mean()) if apfd_swa is not None else None,
        apfd_swa_std=float(apfd_swa.std()) if apfd_swa is not None else None,
        apfd_all_data=float(apfd_all_best),
        wall_sec=time.time() - t0,
    )
    out_path = os.path.join(OUT, f"uav_prio_{args.mode}_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"\n=== UAV-{args.mode} ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"[done] wrote {out_path}")


if __name__ == '__main__':
    main()
