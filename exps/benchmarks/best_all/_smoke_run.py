"""Smoke run for exp_best_all.py loaders + pipeline.
Reduced settings (3 epochs, 1500 cap/bench, 5 trials) — verify nothing crashes
and APFD comes out as a plausible number, not a publication metric.
"""
import sys, os, time, json, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_best_all as M

# Shrink expensive knobs.
M.EPOCHS = 3
M.SWA_START = 2
M.N_TRIALS = 5
M.BATCH = 128

CAP = 1500  # per-bench sample cap to keep training quick on CPU

def cap_data(data, n=CAP, seed=42):
    if len(data) <= n: return data
    rng = random.Random(seed)
    return rng.sample(data, n)

results = {}

def run(tag, root_fn, load_fn, args=()):
    print(f"\n{'#'*70}\n# {tag}\n{'#'*70}")
    t0 = time.time()
    root = root_fn(*args) if args else root_fn()
    print(f"  root: {root}")
    if not root:
        results[tag] = {'status': 'missing'}; return
    data = load_fn(root)
    nf = sum(tc['test_outcome'] == 'FAIL' for tc in data)
    print(f"  N={len(data)} | FAIL={nf} | load_time={time.time()-t0:.1f}s")
    if len(data) < 100 or nf < 20:
        results[tag] = {'status': 'too_small', 'n': len(data), 'n_fail': nf}; return
    data = cap_data(data)
    nf2 = sum(tc['test_outcome'] == 'FAIL' for tc in data)
    print(f"  capped: N={len(data)} | FAIL={nf2}")
    tr, te = M.stratified_split(data, test_size=0.2)
    r = M.run_geom_split(tr, te, name=tag, n_trials=M.N_TRIALS)
    print(f"  --> APFD={r['apfd_mean']:.4f}+/-{r['apfd_std']:.4f} AUC={r['auc']:.4f}")
    results[tag] = r

run('sensodat',     M.find_sensodat_root, M.load_sensodat)
run('oob_its4sdc',  M.find_oob_single_root, M.load_oob_dir)
run('scissor',      M.find_scissor_root, M.load_scissor)
run('travel',       M.find_travel_root, M.load_travel)

# RP: quick LightGBM check on one CSV only
print(f"\n{'#'*70}\n# rp_quick (BeamNG_RF_1 only)\n{'#'*70}")
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
base = M.find_rp_base()
csv = M.find_rp_csv(base, 'datasets/fullroad/BeamNG_AI/BeamNG_RF_1/BeamNG_RF_1_Complete.csv')
df = pd.read_csv(csv)
y = (df['safety'].astype(str).str.lower() == 'unsafe').astype(int).values
X = df.drop(columns=[c for c in df.columns if c in {'start_time','end_time','safety'}]).apply(pd.to_numeric, errors='coerce').fillna(0.0).values
print(f"  N={len(y)} FAIL={int(y.sum())} feats={X.shape[1]}")
try: import lightgbm as lgb; HAVE_LGB=True
except Exception: HAVE_LGB=False
apfds=[]
for fk,(tr,te) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X,y)):
    if HAVE_LGB:
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                                 min_data_in_leaf=10, class_weight='balanced',
                                 random_state=42, verbosity=-1)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)
    clf.fit(X[tr], y[tr])
    probs = clf.predict_proba(X[te])[:,1]; y_te = y[te]
    order = np.argsort(-probs); fp = [pos+1 for pos,idx in enumerate(order) if y_te[idx]==1]
    apfds.append(1 - sum(fp)/(len(order)*len(fp)) + 1/(2*len(order)) if fp else 1.0)
print(f"  --> APFD (3-fold)= {np.mean(apfds):.4f}+/-{np.std(apfds):.4f}")
results['rp_BeamNG_RF_1'] = {'apfd_mean': float(np.mean(apfds)),
                              'apfd_std': float(np.std(apfds)),
                              'classifier': 'LightGBM' if HAVE_LGB else 'HistGradientBoosting'}

out = os.path.join(M.OUTPUT_DIR, 'smoke_results.json')
with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
print(f"\nSaved {out}")
print("\n=== SMOKE SUMMARY ===")
for k,v in results.items():
    if isinstance(v, dict) and 'apfd_mean' in v:
        print(f"  {k:>20s}: APFD={v['apfd_mean']:.4f}+/-{v['apfd_std']:.4f}")
    else:
        print(f"  {k:>20s}: {v}")
