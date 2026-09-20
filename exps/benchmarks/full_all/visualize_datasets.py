"""
Visualize the 5 benchmarks used by exp_full_all.py
==================================================
Standalone EDA script. For each dataset:
  * print N, FAIL%, road-length distribution
  * plot 4 sample roads (2 PASS, 2 FAIL) as x-y trajectory
  * plot the curvature time-series of one PASS + one FAIL sample
For RP (tabular):
  * label balance per CSV + histogram of 2 numeric features
A final summary figure compares N / FAIL% / road-length across benchmarks.

Run:
  python visualize_datasets.py                     # auto-discover paths
  python visualize_datasets.py --max 60            # cap roads per dataset

Outputs (PNG):
  ./dataset_viz/<bench>_samples.png
  ./dataset_viz/<bench>_lengths.png
  ./dataset_viz/_summary.png
"""
import os, sys, json, glob, math, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAVE_PD = True
except Exception:
    HAVE_PD = False

# ---------- Path discovery (Kaggle first, then local data/) ----------
# Notebook cells have no __file__, so fall back to cwd.
_HERE = os.path.dirname(os.path.abspath(globals().get('__file__', os.getcwd())))
SEARCH_ROOTS = [
    '/kaggle/input/datasets',
    os.path.join(_HERE, '..', '..', 'data'),
    os.path.join(_HERE, '..', '..'),
    os.getcwd(),
]

def find_first(rel_parts):
    """Try each search root + rel path, return first existing."""
    for root in SEARCH_ROOTS:
        cand = os.path.join(root, *rel_parts)
        if os.path.isdir(cand) or os.path.isfile(cand):
            return cand
    return None

PATHS = {
    'sensodat':  find_first(['chinguyeen', 'sdc-sensodat']),
    'scissor':   find_first(['chinguyeen', 'sdc-scissor',
                              'christianbirchler-org-sdc-scissor-faf11b2',
                              'sample_tests']),
    'its4sdc':   find_first(['chiboiz', 'its4sdc', 'executed-10000']),
    'travel':    find_first(['chiboiz', 'sdc-travel', 'competition']),
    'rp_base':   find_first(['chiboiz', 'sdc-pririotizer-rp',
                              'SDC-Pririotizer-RP']),
}

OUT = '/kaggle/working/dataset_viz' if os.path.isdir('/kaggle/working') \
      else os.path.join(_HERE, 'dataset_viz')
os.makedirs(OUT, exist_ok=True)

# ====================================================================
# Loaders (copied verbatim from exp_full_all.py so this file is standalone)
# ====================================================================

def _normalize_points(pts_raw):
    if not pts_raw: return np.zeros((0, 2), dtype=np.float64)
    first = pts_raw[0]
    if isinstance(first, dict):
        return np.array([[p['x'], p['y']] for p in pts_raw], dtype=np.float64)
    arr = np.asarray(pts_raw, dtype=np.float64)
    if arr.ndim == 1: arr = arr.reshape(-1, 2)
    elif arr.ndim == 2 and arr.shape[1] >= 2: arr = arr[:, :2]
    return arr

def _compute_curvature(pts):
    n = len(pts); curv = np.zeros(max(0, n - 2))
    for i in range(n - 2):
        x1, y1 = pts[i]; x2, y2 = pts[i + 1]; x3, y3 = pts[i + 2]
        a = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        s = 0.5 * (a + b + c); at = s * (s - a) * (s - b) * (s - c)
        if at <= 1e-10:
            curv[i] = 0.0
        else:
            R = a * b * c / (4 * math.sqrt(at))
            curv[i] = 1.0 / R if R > 0 else 0.0
    return curv

def load_sensodat(root, cap=None):
    if not root or not os.path.isdir(root): return []
    full = os.path.join(root, 'sensodat_full.json')
    candidates = [full] if os.path.isfile(full) else [
        p for p in (os.path.join(root, n) for n in ('sensodat_train.json', 'sensodat_test.json'))
        if os.path.isfile(p)
    ]
    data = []
    for fp in candidates:
        try:
            with open(fp) as f: items = json.load(f)
        except Exception as e:
            print(f"  [WARN] {fp}: {e}"); continue
        if not isinstance(items, list): continue
        for tc in items:
            md = tc.get('meta_data') or {}
            ti = md.get('test_info') or {}
            if isinstance(ti, str):
                try: import ast; ti = ast.literal_eval(ti)
                except Exception: continue
            if ti.get('is_valid') is False: continue
            out = ti.get('test_outcome')
            if out not in ('FAIL', 'PASS'): continue
            pts = tc.get('road_points')
            if not pts: continue
            data.append({'_id': str(tc.get('_id')), 'road_points': pts,
                         'test_outcome': out})
            if cap and len(data) >= cap: return data
    return data

def load_flat_json_dir(path, pattern='*.json', cap=None):
    if not path or not os.path.isdir(path): return []
    files = sorted(glob.glob(os.path.join(path, pattern)))
    data = []
    for fp in files:
        try:
            with open(fp) as f: tc = json.load(f)
        except Exception:
            continue
        if tc.get('is_valid', True) is False: continue
        pts = tc.get('road_points') or tc.get('interpolated_road_points')
        out = tc.get('test_outcome')
        if not pts or out not in ('FAIL', 'PASS'): continue
        data.append({'_id': os.path.basename(fp), 'road_points': pts,
                     'test_outcome': out})
        if cap and len(data) >= cap: return data
    return data

def load_travel(root, cap=None):
    if not root or not os.path.isdir(root): return []
    data = []
    for camp in sorted(os.listdir(root)):
        cp = os.path.join(root, camp)
        if not os.path.isdir(cp): continue
        for fp in glob.glob(os.path.join(cp, 'test.*.json')):
            try:
                with open(fp) as f: tc = json.load(f)
            except Exception:
                continue
            if not tc.get('is_valid', True): continue
            pts = tc.get('interpolated_points') or tc.get('road_points')
            out = tc.get('test_outcome')
            if not pts or out not in ('FAIL', 'PASS'): continue
            data.append({'_id': f'{camp}/{os.path.basename(fp)}',
                         'campaign': camp, 'road_points': pts,
                         'test_outcome': out})
            if cap and len(data) >= cap: return data
    return data

# ====================================================================
# Pipeline trace (raw JSON -> tensor that the model actually sees)
# ====================================================================

SEQ_LEN = 197   # same as exp_full_all.py
CH_NAMES = ['seg_len', 'abs_dheading', 'curv', 'curv_deriv',
            'cum_dist_norm', 'head_sin', 'head_cos',
            'rel_pos', 'local_std', 'curv_accel']

def _extract_seq_10ch(pts_raw):
    """Verbatim from exp_full_all.py — returns (N, 10) float32 array."""
    pts = _normalize_points(pts_raw); n = len(pts)
    if n < 3:
        pts = np.vstack([pts] * 3)[:max(3, n)] if n else np.zeros((3, 2)); n = len(pts)
    diffs = np.diff(pts, axis=0); seg_lens = np.linalg.norm(diffs, axis=1)
    seg_full = np.pad(seg_lens, (0, 1), mode='edge')
    angles = np.arctan2(diffs[:, 1], diffs[:, 0]); ac = np.diff(angles)
    ac = (ac + np.pi) % (2 * np.pi) - np.pi
    abs_ac_full = np.pad(np.abs(ac), (1, 1), mode='constant')
    curv = np.abs(_compute_curvature(pts))
    curv_full = np.pad(curv, (1, 1), mode='constant')
    curv_deriv_full = np.pad(np.diff(curv_full), (0, 1), mode='constant')
    cum_dist = np.cumsum(seg_full)
    cum_dist_norm = cum_dist / (cum_dist[-1] + 1e-8)
    heading_full = np.pad(angles, (0, 1), mode='edge')
    heading_sin = np.sin(heading_full); heading_cos = np.cos(heading_full)
    rel_pos = np.linspace(0, 1, n)
    w = 11; local_std = np.zeros(n); hw = w // 2
    for i in range(n):
        s, e = max(0, i - hw), min(n, i + hw + 1)
        local_std[i] = np.std(curv_full[s:e])
    curv_accel_full = np.pad(np.diff(curv_deriv_full), (0, 1), mode='constant')
    return np.column_stack([seg_full, abs_ac_full, curv_full, curv_deriv_full,
                            cum_dist_norm, heading_sin, heading_cos,
                            rel_pos, local_std, curv_accel_full]).astype(np.float32)

def _resample(seq, target_len=SEQ_LEN):
    n, c = seq.shape
    if n == target_len: return seq
    x_old = np.linspace(0, 1, n); x_new = np.linspace(0, 1, target_len)
    out = np.empty((target_len, c), dtype=np.float32)
    for ch in range(c):
        out[:, ch] = np.interp(x_new, x_old, seq[:, ch])
    return out

def inspect_pipeline(data, label):
    """Trace ONE PASS + ONE FAIL sample through the geometry pipeline,
    printing the array shape and a few values at every step."""
    if not data:
        print(f"  [{label}] no data to inspect"); return
    print(f"\n  --- pipeline trace: {label} ---")
    samples = []
    for outcome in ('PASS', 'FAIL'):
        hit = next((t for t in data if t['test_outcome'] == outcome), None)
        if hit: samples.append((outcome, hit))
    for outcome, tc in samples:
        pts_raw = tc['road_points']
        first = pts_raw[0]
        raw_type = type(first).__name__
        if isinstance(first, dict): raw_repr = f"dict keys={list(first.keys())}"
        else:                       raw_repr = f"{raw_type} value={first}"

        pts = _normalize_points(pts_raw)
        seq10 = _extract_seq_10ch(pts_raw)
        resampled = _resample(seq10)
        # batch of 1, then permute to model layout
        batched   = resampled[None, ...]                    # (1, 197, 10)
        permuted  = np.transpose(batched, (0, 2, 1))        # (1, 10, 197)

        print(f"    [{outcome}] _id={str(tc.get('_id'))[:40]}")
        print(f"      raw road_points  : list len={len(pts_raw):4d}  "
              f"first[0]={raw_repr}")
        print(f"      _normalize_points: shape={pts.shape}  dtype={pts.dtype}  "
              f"x_range=[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
              f"y_range=[{pts[:,1].min():.1f},{pts[:,1].max():.1f}]")
        print(f"      extract_seq_10ch : shape={seq10.shape}  dtype={seq10.dtype}")
        print(f"      resample(197)    : shape={resampled.shape}  dtype={resampled.dtype}")
        print(f"      batched          : shape={batched.shape}  (B, L, C)")
        print(f"      permuted -> model: shape={permuted.shape}  (B, C, L) "
              f"<-- this enters RoadTransformer.forward(x)")
        # per-channel stats of the resampled (pre-normalize) sequence
        print(f"      channel ranges (pre z-score):")
        for i, name in enumerate(CH_NAMES):
            col = resampled[:, i]
            print(f"        ch{i} {name:14s}: "
                  f"min={col.min():+.3e}  max={col.max():+.3e}  "
                  f"mean={col.mean():+.3e}")

def inspect_rp(rp_results):
    if not rp_results: return
    print("\n  --- pipeline trace: RP (tabular) ---")
    name, blob = next(iter(rp_results.items()))
    X = blob['sample_features']
    print(f"    [{name}] X.shape={X.shape}  dtype={X.dtype}  "
          f"(rows=tests, cols=pre-extracted features)")
    print(f"      y derived from `safety == 'unsafe'` -> (N,) int64 0/1")
    print(f"      goes straight into LightGBM.fit(X, y) -- NO geometry pipeline")

# ====================================================================
# Visualization helpers
# ====================================================================

def summarize(data, label):
    n = len(data)
    if n == 0:
        print(f"  [{label}] EMPTY"); return None
    nf = sum(1 for t in data if t['test_outcome'] == 'FAIL')
    lengths = np.array([len(_normalize_points(t['road_points'])) for t in data])
    print(f"  [{label}] N={n}  FAIL={nf} ({100*nf/n:.1f}%)  "
          f"road_len: min={lengths.min()} med={int(np.median(lengths))} "
          f"max={lengths.max()} mean={lengths.mean():.1f}")
    return {'n': n, 'n_fail': nf, 'lengths': lengths, 'label': label}

def plot_samples(data, label, out_path, seed=42):
    """4 sample roads (2 PASS, 2 FAIL) + curvature of 1 PASS + 1 FAIL."""
    rng = np.random.RandomState(seed)
    pass_idx = [i for i, t in enumerate(data) if t['test_outcome'] == 'PASS']
    fail_idx = [i for i, t in enumerate(data) if t['test_outcome'] == 'FAIL']
    if len(pass_idx) < 2 or len(fail_idx) < 2:
        print(f"  [{label}] not enough samples to plot"); return
    pick_pass = rng.choice(pass_idx, 2, replace=False)
    pick_fail = rng.choice(fail_idx, 2, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(f'{label}  -  road samples (top: trajectory, bottom: curvature)',
                 fontsize=13, fontweight='bold')

    samples = [('PASS', pick_pass[0], 'tab:green'),
               ('PASS', pick_pass[1], 'tab:green'),
               ('FAIL', pick_fail[0], 'tab:red'),
               ('FAIL', pick_fail[1], 'tab:red')]

    for col, (tag, idx, color) in enumerate(samples):
        pts = _normalize_points(data[idx]['road_points'])
        ax = axes[0, col]
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5)
        ax.scatter(pts[0, 0], pts[0, 1], color='black', s=40, zorder=5,
                   marker='o', label='start')
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=60, zorder=5,
                   marker='X', label='end')
        ax.set_title(f'{tag}  |  N={len(pts)}', fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='best')

        curv = _compute_curvature(pts)
        ax2 = axes[1, col]
        ax2.plot(np.abs(curv), color=color, lw=1.0)
        ax2.set_title(f'|curvature| (peak={np.abs(curv).max():.4f})', fontsize=9)
        ax2.set_xlabel('waypoint index'); ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{label}] -> {out_path}")

def plot_length_hist(stats_list, out_path):
    """Histogram of road-points length per dataset, one subplot each."""
    valid = [s for s in stats_list if s is not None]
    if not valid:
        print("  no length stats to plot"); return
    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
    for ax, s in zip(axes[0], valid):
        ax.hist(s['lengths'], bins=40, color='steelblue', alpha=0.85)
        ax.set_title(f"{s['label']}\nN={s['n']}  FAIL={100*s['n_fail']/s['n']:.1f}%",
                     fontsize=10)
        ax.set_xlabel('road_points length'); ax.set_ylabel('count')
        ax.grid(alpha=0.3)
    plt.suptitle('Road-points length distribution per benchmark',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")

def plot_rp_overview(rp_results, out_path):
    """Bar + 2 feature histograms for the tabular RP datasets."""
    if not rp_results:
        print("  no RP results"); return
    names = list(rp_results.keys())
    n_total = [rp_results[k]['n'] for k in names]
    n_fail  = [rp_results[k]['n_fail'] for k in names]
    pct_fail = [100 * f / max(1, n) for f, n in zip(n_fail, n_total)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, n_total, 0.4, label='total', color='steelblue')
    ax.bar(x + 0.2, n_fail, 0.4, label='unsafe', color='tab:red')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=9)
    ax.set_title('RP — samples / unsafe count'); ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x, pct_fail, color=['tab:gray']*len(names))
    for i, v in enumerate(pct_fail): ax.text(i, v + 1, f'{v:.0f}%', ha='center')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel('unsafe %')
    ax.set_title('RP — class balance'); ax.grid(alpha=0.3, axis='y')

    # Histogram of 2 numeric features pooled across CSVs
    ax = axes[2]
    pooled = rp_results[names[0]].get('sample_features')
    if pooled is not None and pooled.shape[1] >= 2:
        ax.hist(pooled[:, 0], bins=30, alpha=0.6, label=f'feat[0]')
        ax.hist(pooled[:, 1], bins=30, alpha=0.6, label=f'feat[1]')
        ax.set_title(f'RP {names[0]} — first 2 numeric features')
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")

def plot_summary(stats_list, out_path):
    valid = [s for s in stats_list if s is not None]
    if not valid: return
    labels = [s['label'] for s in valid]
    ns     = [s['n']     for s in valid]
    pcts   = [100 * s['n_fail'] / s['n'] for s in valid]
    meds   = [int(np.median(s['lengths'])) for s in valid]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(valid))
    for ax, vals, title, color in [
        (axes[0], ns,   'N (total tests)',          'steelblue'),
        (axes[1], pcts, 'FAIL %',                    'tab:red'),
        (axes[2], meds, 'median road_points length', 'tab:purple'),
    ]:
        ax.bar(x, vals, color=color)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.1f}' if isinstance(v, float) else str(v),
                    ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_title(title); ax.grid(alpha=0.3, axis='y')
    plt.suptitle('Cross-benchmark summary', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")

# ====================================================================
# Main
# ====================================================================

def main(max_per_ds=None):
    # In a notebook, sys.argv is the kernel's argv; parse only when run as script.
    if __name__ == '__main__' and 'ipykernel' not in sys.modules:
        ap = argparse.ArgumentParser()
        ap.add_argument('--max', type=int, default=None,
                        help='Cap roads loaded per dataset (speed-up for EDA).')
        args = ap.parse_args()
        max_per_ds = args.max
    class _A: pass
    args = _A(); args.max = max_per_ds

    print(f"OUT dir: {OUT}")
    for k, v in PATHS.items():
        print(f"  PATH[{k:9s}] = {v}")

    stats = []

    # 1. SensoDat
    print("\n--- SensoDat ---")
    d = load_sensodat(PATHS['sensodat'], cap=args.max)
    s = summarize(d, 'SensoDat'); stats.append(s)
    if d:
        plot_samples(d, 'SensoDat', os.path.join(OUT, 'sensodat_samples.png'))
        inspect_pipeline(d, 'SensoDat')

    # 2. Scissor
    print("\n--- SDC-Scissor ---")
    d = load_flat_json_dir(PATHS['scissor'], pattern='*-test.json', cap=args.max)
    s = summarize(d, 'Scissor'); stats.append(s)
    if d:
        plot_samples(d, 'Scissor', os.path.join(OUT, 'scissor_samples.png'))
        inspect_pipeline(d, 'Scissor')

    # 3. its4sdc
    print("\n--- its4sdc ---")
    d = load_flat_json_dir(PATHS['its4sdc'], pattern='*.json', cap=args.max)
    s = summarize(d, 'its4sdc'); stats.append(s)
    if d:
        plot_samples(d, 'its4sdc', os.path.join(OUT, 'its4sdc_samples.png'))
        inspect_pipeline(d, 'its4sdc')

    # 4. Travel
    print("\n--- sdc-travel ---")
    d = load_travel(PATHS['travel'], cap=args.max)
    s = summarize(d, 'Travel'); stats.append(s)
    if d:
        plot_samples(d, 'Travel', os.path.join(OUT, 'travel_samples.png'))
        # campaign breakdown
        from collections import Counter
        camp_counter = Counter(t.get('campaign', '?') for t in d)
        print(f"  campaigns seen: {len(camp_counter)}  "
              f"(top 5: {camp_counter.most_common(5)})")
        inspect_pipeline(d, 'Travel')

    # 5. RP (tabular)
    print("\n--- SDC-Prioritizer-RP (tabular) ---")
    rp_results = {}
    if HAVE_PD and PATHS['rp_base']:
        sets = {
            'BeamNG_RF_1':   'datasets/fullroad/BeamNG_AI/BeamNG_RF_1/BeamNG_RF_1_Complete.csv',
            'BeamNG_RF_1_5': 'datasets/fullroad/BeamNG_AI/BeamNG_RF_1_5/BeamNG_RF_1_5_Complete.csv',
            'BeamNG_RF_2':   'datasets/fullroad/BeamNG_AI/BeamNG_RF_2/BeamNG_RF_2_Complete.csv',
            'DriverAI':      'datasets/fullroad/Driver_AI/DriverAI_Complete.csv',
        }
        for name, rel in sets.items():
            path = os.path.join(PATHS['rp_base'], rel)
            if not os.path.isfile(path):
                print(f"  [SKIP] {name}: {rel}"); continue
            df = pd.read_csv(path)
            y = (df['safety'].astype(str).str.lower() == 'unsafe').astype(int).values
            feat_cols = [c for c in df.columns if c not in {'start_time', 'end_time', 'safety'}]
            X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
            print(f"  {name}: N={len(df)}  unsafe={y.sum()} ({100*y.sum()/len(df):.1f}%)  "
                  f"feats={len(feat_cols)} cols={feat_cols[:5]}...")
            rp_results[name] = {'n': len(df), 'n_fail': int(y.sum()),
                                'sample_features': X[:2000]}
        if rp_results:
            plot_rp_overview(rp_results, os.path.join(OUT, 'rp_overview.png'))
            inspect_rp(rp_results)
    else:
        print("  [SKIP] pandas missing or RP path not found")

    # Cross-benchmark summary
    print("\n--- summary figures ---")
    plot_length_hist(stats, os.path.join(OUT, '_lengths.png'))
    plot_summary(stats,    os.path.join(OUT, '_summary.png'))

    print(f"\nDONE. Open: {OUT}")

if __name__ == '__main__':
    main()
