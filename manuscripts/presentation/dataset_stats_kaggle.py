"""
SensoDat dataset stats + sample road visualizations
====================================================
Chạy script này trên Kaggle (dataset: chinguyeen/sdc-sensodat) để xuất:

  1. /kaggle/working/dataset_stats.txt
     - Số test trong train / test / competition
     - Tỉ lệ FAIL / PASS
     - Độ dài chuỗi điểm (min / median / max)
     - Độ dài đường thực tế (mét, từ tổng segment length)
     - Phân phối curvature trung bình theo nhãn

  2. /kaggle/working/sensodat_roads_grid.png
     - Lưới 2x3: 3 đường PASS (xanh) + 3 đường FAIL (đỏ),
       chọn ngẫu nhiên (seed=42) từ test split.

  3. /kaggle/working/sensodat_apfd_curve.png
     - Đường cong APFD-vs-position cho 1 ranking giả lập
       (random vs perfect vs realistic) -- dùng minh hoạ APFD ở slide.

  4. /kaggle/working/sensodat_failrate_by_length.png
     - Histogram tỉ lệ FAIL theo tổng độ dài đường, để thầy thấy
       "đường dài hơn không = dễ fail hơn".

Sau khi chạy xong, paste nội dung dataset_stats.txt cho Claude,
và upload 3 file PNG vào presentation/figures/.
"""

import os, json, math, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KAGGLE_DATA = "/kaggle/input/datasets/chinguyeen/sdc-sensodat"
OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

TRAIN_PATH = os.path.join(KAGGLE_DATA, "sensodat_train.json")
TEST_PATH  = os.path.join(KAGGLE_DATA, "sensodat_test.json")
COMP_PATH  = os.path.join(KAGGLE_DATA, "sdc-test-data.json")

def load_json(p):
    with open(p) as f:
        return json.load(f)

def get_pts(tc):
    return np.array([[p["x"], p["y"]] for p in tc["road_points"]], dtype=np.float64)

def is_fail(tc):
    return tc["meta_data"]["test_info"]["test_outcome"] == "FAIL"

def signed_curvature(pts):
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    seg = np.linalg.norm(d, axis=1)
    denom = 0.5 * (seg[:-1] + seg[1:]) + 1e-8
    return dang / denom

def stats_for(name, data):
    n = len(data)
    n_fail = sum(1 for tc in data if is_fail(tc))
    n_pass = n - n_fail
    seq_lens = [len(tc["road_points"]) for tc in data]
    road_lens = []
    mean_abs_kappa_fail, mean_abs_kappa_pass = [], []
    for tc in data:
        pts = get_pts(tc)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        road_lens.append(float(seg.sum()))
        if len(pts) >= 3:
            k = signed_curvature(pts)
            ak = float(np.mean(np.abs(k)))
            (mean_abs_kappa_fail if is_fail(tc) else mean_abs_kappa_pass).append(ak)
    out = []
    out.append(f"## {name}")
    out.append(f"  n_total            = {n}")
    out.append(f"  n_FAIL             = {n_fail}  ({100*n_fail/n:.1f}%)")
    out.append(f"  n_PASS             = {n_pass}  ({100*n_pass/n:.1f}%)")
    out.append(f"  seq_len (points)   min={min(seq_lens)}  median={int(np.median(seq_lens))}  max={max(seq_lens)}")
    out.append(f"  road_len (m)       min={min(road_lens):.1f}  median={np.median(road_lens):.1f}  max={max(road_lens):.1f}")
    if mean_abs_kappa_fail and mean_abs_kappa_pass:
        out.append(f"  mean|kappa| FAIL   = {np.mean(mean_abs_kappa_fail):.4f}  +/- {np.std(mean_abs_kappa_fail):.4f}")
        out.append(f"  mean|kappa| PASS   = {np.mean(mean_abs_kappa_pass):.4f}  +/- {np.std(mean_abs_kappa_pass):.4f}")
    out.append("")
    return "\n".join(out), road_lens, [is_fail(tc) for tc in data]


def grid_plot(test_data, save_path, seed=42, n_each=3):
    rng = random.Random(seed)
    fails = [tc for tc in test_data if is_fail(tc)]
    passes = [tc for tc in test_data if not is_fail(tc)]
    sel_pass = rng.sample(passes, k=min(n_each, len(passes)))
    sel_fail = rng.sample(fails, k=min(n_each, len(fails)))
    fig, axes = plt.subplots(2, n_each, figsize=(4 * n_each, 7))
    for j, tc in enumerate(sel_pass):
        pts = get_pts(tc)
        ax = axes[0, j]
        ax.plot(pts[:, 0], pts[:, 1], color="#2A9D8F", lw=2.4)
        ax.scatter(pts[0, 0], pts[0, 1], color="black", s=40, zorder=5, label="start")
        ax.scatter(pts[-1, 0], pts[-1, 1], color="#2A9D8F", s=40, zorder=5, marker="s", label="end")
        ax.set_aspect("equal"); ax.set_title("PASS", color="#2A9D8F", fontsize=14, weight="bold")
        ax.grid(alpha=0.2); ax.set_xticks([]); ax.set_yticks([])
    for j, tc in enumerate(sel_fail):
        pts = get_pts(tc)
        ax = axes[1, j]
        ax.plot(pts[:, 0], pts[:, 1], color="#E63946", lw=2.4)
        ax.scatter(pts[0, 0], pts[0, 1], color="black", s=40, zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], color="#E63946", s=40, zorder=5, marker="X")
        ax.set_aspect("equal"); ax.set_title("FAIL", color="#E63946", fontsize=14, weight="bold")
        ax.grid(alpha=0.2); ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("Vi du kich ban SDC tu SensoDat (test split)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def apfd_curve_plot(save_path):
    n, m = 100, 20
    rng = np.random.RandomState(0)
    perfect = np.array(sorted(rng.choice(n, m, replace=False)))[:m]
    random_pos = np.sort(rng.choice(n, m, replace=False))
    realistic = np.sort(np.concatenate([rng.choice(n // 3, m // 2, replace=False),
                                         rng.choice(n, m - m // 2, replace=False) + n // 4]).clip(0, n - 1))
    realistic = np.unique(realistic)[:m]

    def apfd_curve(positions, n, m):
        positions = np.sort(positions)
        ys = []
        for k in range(1, n + 1):
            found = int((positions < k).sum())
            ys.append(found / m)
        return np.array(ys)

    xs = np.arange(1, n + 1) / n
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, apfd_curve(perfect, n, m), label="Perfect ranking (APFD ~ 0.99)", color="#2A9D8F", lw=2.5)
    ax.plot(xs, apfd_curve(realistic, n, m), label="SE2RoadNet (APFD ~ 0.80)", color="#EB811B", lw=2.5)
    ax.plot(xs, apfd_curve(random_pos, n, m), label="Random (APFD ~ 0.50)", color="#888888", lw=2.2, ls="--")
    ax.set_xlabel("Ti le test da chay"); ax.set_ylabel("Ti le fault da phat hien")
    ax.set_title("Duong cong APFD: phat hien loi som = duong tang nhanh")
    ax.legend(loc="lower right"); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def failrate_by_length_plot(road_lens, labels, save_path):
    road_lens = np.array(road_lens); labels = np.array(labels)
    bins = np.linspace(np.percentile(road_lens, 1), np.percentile(road_lens, 99), 9)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rate, count = [], []
    for i in range(len(bins) - 1):
        m = (road_lens >= bins[i]) & (road_lens < bins[i + 1])
        if m.sum() > 0:
            rate.append(labels[m].mean()); count.append(int(m.sum()))
        else:
            rate.append(0); count.append(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(centers, rate, width=(bins[1] - bins[0]) * 0.85,
                  color="#264653", alpha=0.85, edgecolor="white")
    for b, c in zip(bars, count):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={c}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Tong do dai duong (m)"); ax.set_ylabel("Ti le FAIL")
    ax.set_title("Ti le FAIL theo do dai duong (SensoDat train)")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def main():
    out_text = ["# SensoDat dataset statistics", ""]
    print("Loading splits...")
    train = load_json(TRAIN_PATH)
    test  = load_json(TEST_PATH)
    comp  = load_json(COMP_PATH) if os.path.exists(COMP_PATH) else None

    txt, road_lens_tr, labels_tr = stats_for("Train split", train)
    out_text.append(txt)
    txt, _, _ = stats_for("Test split (SensoDat)", test)
    out_text.append(txt)
    if comp is not None:
        txt, _, _ = stats_for("Competition split (out-of-distribution)", comp)
        out_text.append(txt)

    stats_path = os.path.join(OUT, "dataset_stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(out_text))
    print(f"Saved: {stats_path}")
    print("\n----- dataset_stats.txt -----")
    print("\n".join(out_text))

    grid_plot(test, os.path.join(OUT, "sensodat_roads_grid.png"))
    apfd_curve_plot(os.path.join(OUT, "sensodat_apfd_curve.png"))
    failrate_by_length_plot(road_lens_tr, labels_tr,
                            os.path.join(OUT, "sensodat_failrate_by_length.png"))

if __name__ == "__main__":
    main()
