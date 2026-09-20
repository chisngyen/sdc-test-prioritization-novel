"""
SE(2)-aware reproduction of SDC-Pririotizer-RP bench
=====================================================
Birchler et al. (Zenodo / SDC-Pririotizer-RP) ship PRE-AGGREGATED tabular
features (no road_points). Honest framing of the SE(2) story here:

  * RP features are scalar AGGREGATES of geometry (angle statistics, turn
    counts, distances, durations). By construction they are intrinsic to
    the road shape and do not depend on the global frame.
  * Consequently, ANY ranker trained on those features is automatically
    SE(2)-invariant -- the invariance is at the feature-engineering layer,
    not the architecture layer. The corresponding OOB result uses
    SE2RoadNet on raw road_points (architectural invariance).

This script:
  (1) Audits each column and tags it `intrinsic` / `extrinsic` / `meta`
      from its name (frontier: anything containing 'x', 'y', 'pos',
      'heading_abs' would be extrinsic; none should exist in RP fullroad).
  (2) Trains LightGBM (fallback HGB) on the INTRINSIC subset only.
  (3) 5-fold stratified CV, APFD + AUC mean/std.
  (4) Saves audit + numbers to rp_se2_external_bench.json so the paper
      can cite "no extrinsic features dropped".

Datasets (full-road feature CSVs):
  - BeamNG_AI/BeamNG_RF_1
  - BeamNG_AI/BeamNG_RF_1_5
  - BeamNG_AI/BeamNG_RF_2
  - Driver_AI/DriverAI
"""
import os, json, time, warnings, re
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

def _find_rp_base():
    roots = [
        '/kaggle/input',
        os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'kaggle')),
        os.path.normpath(os.path.join(HERE, '..', '..', 'data')),
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
    direct = os.path.join(BASE, rel_under_base)
    if os.path.isfile(direct): return direct
    target_name = os.path.basename(rel_under_base)
    for dirpath, _, filenames in os.walk(BASE):
        if target_name in filenames:
            return os.path.join(dirpath, target_name)
    return direct

DATASETS = {
    'BeamNG_RF_1':   _find_csv('datasets/fullroad/BeamNG_AI/BeamNG_RF_1/BeamNG_RF_1_Complete.csv'),
    'BeamNG_RF_1_5': _find_csv('datasets/fullroad/BeamNG_AI/BeamNG_RF_1_5/BeamNG_RF_1_5_Complete.csv'),
    'BeamNG_RF_2':   _find_csv('datasets/fullroad/BeamNG_AI/BeamNG_RF_2/BeamNG_RF_2_Complete.csv'),
    'DriverAI':      _find_csv('datasets/fullroad/Driver_AI/DriverAI_Complete.csv'),
}

LABEL_COL = 'safety'
META_COLS = {'start_time', 'end_time', LABEL_COL, 'test_id', 'id', 'name'}

# ---------- SE(2) feature audit ----------
# Heuristic: an aggregate scalar feature is "extrinsic" iff its name implies
# the global frame. Intrinsic = depends only on the road's intrinsic shape.
EXTRINSIC_PATTERNS = [
    r'^x(_|$)', r'^y(_|$)',                # raw coords
    r'_x(_|$)', r'_y(_|$)',
    r'pos(_|$)', r'position',              # absolute position
    r'heading_(abs|mean|min|max|median)',  # absolute heading; |dheading| is fine
    r'orientation',
]
INTRINSIC_PATTERNS = [
    r'angle', r'curv', r'turn', r'distance', r'length', r'duration',
    r'count', r'num_', r'speed', r'accel', r'jerk', r'std', r'mean',
    r'median', r'min', r'max', r'safety', r'segment',
]

def classify_feature(name):
    n = name.lower()
    for pat in EXTRINSIC_PATTERNS:
        if re.search(pat, n):
            for ipat in [r'angle', r'curv', r'turn', r'distance', r'length', r'duration', r'count']:
                if re.search(ipat, n):
                    return 'intrinsic'
            return 'extrinsic'
    for pat in INTRINSIC_PATTERNS:
        if re.search(pat, n): return 'intrinsic'
    return 'unknown'


def compute_apfd(rank_indices, y_true):
    n = len(rank_indices)
    fp = [pos + 1 for pos, idx in enumerate(rank_indices) if y_true[idx] == 1]
    m = len(fp)
    if not n or not m: return 1.0
    return 1 - sum(fp) / (n * m) + 1 / (2 * n)


def load_and_audit(path):
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"No '{LABEL_COL}' column in {path}")
    y = (df[LABEL_COL].astype(str).str.lower() == 'unsafe').astype(int).values
    feat_cols_all = [c for c in df.columns if c not in META_COLS]
    audit = {c: classify_feature(c) for c in feat_cols_all}
    intrinsic = [c for c, t in audit.items() if t == 'intrinsic']
    extrinsic = [c for c, t in audit.items() if t == 'extrinsic']
    unknown   = [c for c, t in audit.items() if t == 'unknown']
    # SE(2)-invariant subset: intrinsic + unknown (defensively kept since
    # they don't match extrinsic patterns either). Extrinsic are dropped.
    use_cols = intrinsic + unknown
    X = df[use_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return X.values.astype(np.float32), y, audit, use_cols, extrinsic


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
    print(f"\n{'='*70}\nSE(2)-aware RP: {name}\n  {path}\n{'='*70}")
    if not os.path.isfile(path):
        print("  [SKIP] file missing"); return {'error': 'missing'}
    X, y, audit, use_cols, extrinsic = load_and_audit(path)
    n_pos = int(y.sum()); n = len(y)
    print(f"  N={n} | unsafe(fail)={n_pos} ({100*n_pos/n:.1f}%)")
    print(f"  features: {len(audit)} total -> {len(use_cols)} kept (intrinsic+unknown), "
          f"{len(extrinsic)} dropped as extrinsic")
    if extrinsic:
        print(f"    DROPPED extrinsic: {extrinsic}")
    if n_pos < n_splits or (n - n_pos) < n_splits:
        print("  [SKIP] not enough class samples for CV")
        return {'n': n, 'n_fail': n_pos, 'error': 'too_imbalanced',
                'feature_audit': audit, 'use_cols': use_cols, 'extrinsic_dropped': extrinsic}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    apfds, aucs = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        probs = fit_predict(X[tr_idx], y[tr_idx], X[te_idx])
        y_te = y[te_idx]
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = float('nan')
        order = np.argsort(-probs)
        apfd = compute_apfd(order, y_te)
        apfds.append(apfd); aucs.append(auc)
        print(f"  fold {fold+1}/{n_splits}: AUC={auc:.4f} | APFD={apfd:.4f} "
              f"(|S|={len(te_idx)}, fail={int(y_te.sum())})")

    res = {
        'n': n, 'n_fail': n_pos,
        'n_features_total': len(audit),
        'n_features_kept': len(use_cols),
        'n_features_dropped_extrinsic': len(extrinsic),
        'extrinsic_dropped': extrinsic,
        'feature_audit': audit,
        'apfd_mean': float(np.mean(apfds)), 'apfd_std': float(np.std(apfds)),
        'auc_mean': float(np.nanmean(aucs)), 'auc_std': float(np.nanstd(aucs)),
        'fold_apfds': [float(a) for a in apfds],
        'fold_aucs': [float(a) for a in aucs],
        'classifier': 'LightGBM' if HAVE_LGB else 'HistGradientBoosting',
        'se2_invariance_basis': 'feature_engineering',
    }
    print(f"  ★ {name}: APFD = {res['apfd_mean']:.4f} ± {res['apfd_std']:.4f} | "
          f"AUC = {res['auc_mean']:.4f}")
    return res


def main():
    t0 = time.time()
    print(f"BASE = {BASE}")
    print(f"Classifier = {'LightGBM' if HAVE_LGB else 'HistGradientBoosting'}")
    print("\nNOTE: RP features are pre-aggregated scalars. SE(2) invariance")
    print("      is enforced at FEATURE-ENGINEERING (by Birchler et al.),")
    print("      not by model architecture. We audit feature names to verify")
    print("      no extrinsic (frame-dependent) features leak in.\n")

    out = {}
    for name, path in DATASETS.items():
        try:
            out[name] = run_dataset(name, path)
        except Exception as e:
            print(f"  [ERR] {name}: {type(e).__name__}: {e}")
            out[name] = {'error': str(e)}

    print(f"\n{'='*70}\nSE(2)-AWARE RP SUMMARY\n{'='*70}")
    for name, r in out.items():
        if 'apfd_mean' in r:
            print(f"  {name:>15s}: APFD={r['apfd_mean']:.4f}±{r['apfd_std']:.4f} | "
                  f"AUC={r['auc_mean']:.4f} | n={r['n']} fail={r['n_fail']} | "
                  f"feats kept={r['n_features_kept']}/{r['n_features_total']} "
                  f"(extrinsic dropped={r['n_features_dropped_extrinsic']})")
        else:
            print(f"  {name:>15s}: {r}")

    out_path = os.path.join(OUTPUT_DIR, 'rp_se2_external_bench.json')
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"\nTOTAL TIME: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
