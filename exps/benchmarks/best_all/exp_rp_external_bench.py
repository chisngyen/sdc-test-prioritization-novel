"""
External benchmark: SDC-Pririotizer-RP datasets
================================================
Birchler et al. "Automated Test Cases Prioritization for SDCs in Virtual
Environments" (Zenodo / SDC-Pririotizer-RP) provides PRE-EXTRACTED tabular
features (no road_points). Our Transformer is geometry-based, so we use a
gradient-boosted classifier (LightGBM, fallback HistGradientBoosting) to
rank tests by predicted-unsafe probability and report APFD.

This is a REPRODUCTION / external bench — paper subsection should phrase it
as "comparison on Birchler et al.'s feature set" rather than as our model
generalising. The Transformer scores live in exp_best_oob*.py.

Datasets (full-road feature CSVs):
  - BeamNG_AI/BeamNG_RF_1
  - BeamNG_AI/BeamNG_RF_1_5
  - BeamNG_AI/BeamNG_RF_2
  - Driver_AI/DriverAI

Protocol per dataset:
  - 5-fold stratified CV on `safety` ∈ {safe, unsafe}; unsafe → fail (1).
  - Train LightGBM (or HGB), predict prob(unsafe) on held-out fold.
  - Rank fold tests by predicted prob desc, compute APFD.
  - Report mean ± std APFD over folds.

Saves: rp_external_bench.json
"""
import os, json, glob, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

# Auto-discover SDC-Pririotizer-RP base under Kaggle / local roots.
def _find_rp_base():
    roots = [
        '/kaggle/input',
        os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'kaggle')),
        os.path.normpath(os.path.join(HERE, '..', '..', 'data')),
        os.path.normpath(os.path.join(HERE, '..', 'data', 'kaggle')),
        os.path.normpath(os.path.join(HERE, '..', 'data')),
        os.getcwd(),
    ]
    seen = set()
    for root in roots:
        if not root or not os.path.isdir(root) or root in seen: continue
        seen.add(root)
        for dirpath, dirnames, _ in os.walk(root):
            if 'datasets' in dirnames and os.path.basename(dirpath) == 'SDC-Pririotizer-RP':
                return dirpath
    return None

BASE = _find_rp_base() or '/kaggle/input/datasets/chiboiz/sdc-pririotizer-rp/SDC-Pririotizer-RP'
OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(HERE, '..', '..', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _find_csv(rel_under_base):
    """Try BASE/rel first; fall back to walking BASE for the filename."""
    direct = os.path.join(BASE, rel_under_base)
    if os.path.isfile(direct): return direct
    target_name = os.path.basename(rel_under_base)
    for dirpath, _, filenames in os.walk(BASE):
        if target_name in filenames:
            return os.path.join(dirpath, target_name)
    return direct  # return expected path so error is informative

DATASETS = {
    'BeamNG_RF_1':   _find_csv('datasets/fullroad/BeamNG_AI/BeamNG_RF_1/BeamNG_RF_1_Complete.csv'),
    'BeamNG_RF_1_5': _find_csv('datasets/fullroad/BeamNG_AI/BeamNG_RF_1_5/BeamNG_RF_1_5_Complete.csv'),
    'BeamNG_RF_2':   _find_csv('datasets/fullroad/BeamNG_AI/BeamNG_RF_2/BeamNG_RF_2_Complete.csv'),
    'DriverAI':      _find_csv('datasets/fullroad/Driver_AI/DriverAI_Complete.csv'),
}

LABEL_COL = 'safety'
DROP_COLS = {'start_time', 'end_time', LABEL_COL}


def compute_apfd(rank_indices, y_true):
    """rank_indices: positions in priority order; y_true: 1=fail."""
    n = len(rank_indices)
    fp = [pos + 1 for pos, idx in enumerate(rank_indices) if y_true[idx] == 1]
    m = len(fp)
    if not n or not m: return 1.0
    return 1 - sum(fp) / (n * m) + 1 / (2 * n)


def load_dataset(path):
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"No '{LABEL_COL}' column in {path}: {df.columns.tolist()}")
    y = (df[LABEL_COL].astype(str).str.lower() == 'unsafe').astype(int).values
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()
    # Drop non-numeric; coerce remaining
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return X.values.astype(np.float32), y, feat_cols


def fit_predict(X_tr, y_tr, X_te):
    if HAVE_LGB:
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_data_in_leaf=10, subsample=0.9, colsample_bytree=0.9,
            class_weight='balanced', random_state=42, verbosity=-1)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=63, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def run_dataset(name, path, n_splits=5):
    print(f"\n{'='*70}\nRP External: {name}\n  {path}\n{'='*70}")
    if not os.path.isfile(path):
        print(f"  [SKIP] file missing"); return {'error': 'missing'}
    X, y, feat_cols = load_dataset(path)
    n_pos = int(y.sum()); n = len(y)
    print(f"  N={n} | unsafe(fail)={n_pos} ({100*n_pos/n:.1f}%) | features={len(feat_cols)}")
    if n_pos < n_splits or (n - n_pos) < n_splits:
        print("  [SKIP] not enough class samples for CV")
        return {'n': n, 'n_fail': n_pos, 'error': 'too_imbalanced'}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    apfds, aucs = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        probs = fit_predict(X[tr_idx], y[tr_idx], X[te_idx])
        y_te = y[te_idx]
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = float('nan')
        # Priority order: highest predicted unsafe prob first
        order = np.argsort(-probs)
        apfd = compute_apfd(order, y_te)
        apfds.append(apfd); aucs.append(auc)
        print(f"  fold {fold+1}/{n_splits}: AUC={auc:.4f} | APFD={apfd:.4f} "
              f"(|S|={len(te_idx)}, fail={int(y_te.sum())})")

    res = {'n': n, 'n_fail': n_pos, 'n_features': len(feat_cols),
           'apfd_mean': float(np.mean(apfds)), 'apfd_std': float(np.std(apfds)),
           'auc_mean': float(np.nanmean(aucs)), 'auc_std': float(np.nanstd(aucs)),
           'fold_apfds': [float(a) for a in apfds], 'fold_aucs': [float(a) for a in aucs],
           'classifier': 'LightGBM' if HAVE_LGB else 'HistGradientBoosting'}
    print(f"  ★ {name}: APFD = {res['apfd_mean']:.4f} ± {res['apfd_std']:.4f} | "
          f"AUC = {res['auc_mean']:.4f}")
    return res


def main():
    t0 = time.time()
    print(f"BASE = {BASE}")
    print(f"Classifier = {'LightGBM' if HAVE_LGB else 'HistGradientBoosting'}")
    out = {}
    for name, path in DATASETS.items():
        try:
            out[name] = run_dataset(name, path)
        except Exception as e:
            print(f"  [ERR] {name}: {type(e).__name__}: {e}")
            out[name] = {'error': str(e)}

    print(f"\n{'='*70}\nRP EXTERNAL SUMMARY\n{'='*70}")
    for name, r in out.items():
        if 'apfd_mean' in r:
            print(f"  {name:>15s}: APFD={r['apfd_mean']:.4f}±{r['apfd_std']:.4f} | "
                  f"AUC={r['auc_mean']:.4f} | n={r['n']} fail={r['n_fail']}")
        else:
            print(f"  {name:>15s}: {r}")

    out_path = os.path.join(OUTPUT_DIR, 'rp_external_bench.json')
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
