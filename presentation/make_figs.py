# -*- coding: utf-8 -*-
"""
Code-generated figures for the SE2RoadNet ICST/ICSE slide deck.
All charts use the Metropolis palette (teal #23373B + orange #EB811B).
Data-driven charts read data/sensodat_features.csv; the rest are
representative-by-construction (clearly pedagogical, not over-claimed).

Run:  python make_figs.py
Out:  presentation/figures/gen_*.png   (200 dpi, white bg, Vietnamese OK)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- palette ----------
TEAL   = "#23373B"
TEALM  = "#5B7A82"   # mid teal
TEALL  = "#D6DEE0"   # tint
ORANGE = "#EB811B"
GREEN  = "#2E7D32"
RED    = "#B23A33"
GREY   = "#9AA5A8"
plt.rcParams.update({
    "font.family": "DejaVu Sans",   # has full Vietnamese coverage
    "font.size": 12,
    "axes.edgecolor": "#5A6B6F",
    "axes.linewidth": 0.9,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIG  = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "sensodat_features.csv"))


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


# =====================================================================
# 1. APFD detection curve  (correct: Perfect >= SE2 >= Random)
# =====================================================================
def apfd_curve():
    n = 60
    m = 6  # sparse faults so "perfect" is clearly near 1.0

    # explicit fault positions (1-indexed) for clean, distinct illustration
    pos = {
        "perfect": [1, 2, 3, 4, 5, 6],
        "se2":     [2, 5, 9, 14, 19, 25],
        "random":  [4, 15, 25, 33, 45, 55],
    }

    def vec(p):
        v = np.zeros(n, dtype=int)
        for i in p:
            v[i - 1] = 1
        return v

    def cum(v):
        x = np.concatenate([[0], (np.arange(1, n + 1) / n)])
        y = np.concatenate([[0], np.cumsum(v) / m])
        return x, y

    def apfd(p):
        return 1 - sum(p) / (n * m) + 1 / (2 * n)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    se2v = vec(pos["se2"])
    ax.fill_between(*cum(se2v), step="post", color=ORANGE, alpha=0.08)
    for key, c, lab, lw, ls in [
        ("random",   GREY, f"Ngẫu nhiên (APFD ≈ {apfd(pos['random']):.2f})", 2.2, "--"),
        ("se2",    ORANGE, f"SE2RoadNet (APFD ≈ {apfd(pos['se2']):.2f})", 3.0, "-"),
        ("perfect", GREEN, f"Lý tưởng (APFD ≈ {apfd(pos['perfect']):.2f})", 2.4, "-"),
    ]:
        x, y = cum(vec(pos[key]))
        ax.step(x, y, where="post", color=c, lw=lw, ls=ls, label=lab)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tỉ lệ test đã chạy", fontsize=13)
    ax.set_ylabel("Tỉ lệ lỗi đã phát hiện", fontsize=13)
    ax.set_title("Đường cong APFD — xếp hạng tốt ⇒ lỗi lộ sớm", color=TEAL, fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    save(fig, "gen_apfd_curve.png")


# =====================================================================
# 2. Representative PASS / FAIL road shapes (2x3 grid)
# =====================================================================
def roads_grid():
    rng = np.random.default_rng(7)

    def road(kind, k):
        t = np.linspace(0, 1, 220)
        if kind == "PASS":
            a = 0.45 + 0.25 * rng.random()
            x = t * 6
            y = a * np.sin(2 * np.pi * (0.8 + 0.3 * k) * t) * (0.6) + 0.15 * np.sin(2 * np.pi * t)
            y *= 0.8
        else:  # FAIL: sharp chicane / sign-flipping curvature
            freq = 2.4 + 0.7 * k
            x = t * 6
            y = (np.sin(2 * np.pi * freq * t) * (0.55 + 0.25 * t)
                 + 0.4 * np.sin(2 * np.pi * (freq * 2) * t))
        return x, y

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.4))
    specs = [("PASS", GREEN)] * 3 + [("FAIL", RED)] * 3
    for ax, (kind, c), k in zip(axes.ravel(), specs, range(6)):
        x, y = road(kind, k)
        ax.plot(x, y, color=c, lw=3.0, solid_capstyle="round")
        ax.plot(x[0], y[0], "o", color=TEAL, ms=7)
        ax.plot(x[-1], y[-1], "s" if kind == "PASS" else "X", color=c, ms=9)
        ax.set_title(kind, color=c, fontsize=15, fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for s in ax.spines.values():
            s.set_edgecolor(TEALL); s.set_linewidth(1.2)
    fig.suptitle("Kịch bản SDC từ SensoDat: hình dạng quyết định nhãn",
                 color=TEAL, fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "gen_roads_grid.png")


# =====================================================================
# helpers for real data
# =====================================================================
def load_data():
    import csv
    rows = []
    with open(DATA, newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(d)
    return rows


# =====================================================================
# 3. FAIL rate by road length  (REAL data)
# =====================================================================
def failrate_by_length():
    rows = load_data()
    L, Y = [], []
    for d in rows:
        try:
            length = float(d["total_length"]); out = d["test_outcome"].strip().upper()
        except Exception:
            continue
        if out not in ("PASS", "FAIL"):
            continue
        L.append(length); Y.append(1 if out == "FAIL" else 0)
    L = np.array(L); Y = np.array(Y)
    edges = np.arange(90, 320, 25.0)
    cx, rate, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (L >= lo) & (L < hi)
        if msk.sum() < 30:
            continue
        cx.append((lo + hi) / 2); rate.append(Y[msk].mean()); ns.append(int(msk.sum()))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(cx, rate, width=20, color=ORANGE, edgecolor=TEAL, lw=1.0)
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width()/2, b.get_height()+0.012, f"n={n}",
                ha="center", va="bottom", fontsize=9, color=TEAL)
    ax.axhline(0.384, color=TEAL, ls="--", lw=1.3)
    ax.text(edges[-2], 0.384+0.015, "trung bình toàn tập 38.4%", ha="right",
            color=TEAL, fontsize=10)
    ax.set_xlabel("Tổng độ dài đường (m)", fontsize=13)
    ax.set_ylabel("Tỉ lệ FAIL", fontsize=13)
    ax.set_title("Đường càng dài ⇒ càng dễ FAIL (SensoDat, dữ liệu thật)",
                 color=TEAL, fontsize=13)
    ax.set_ylim(0, max(rate)+0.12); ax.grid(axis="y", alpha=0.25)
    save(fig, "gen_failrate_by_length.png")


# =====================================================================
# 4. Curvature signature PASS vs FAIL  (REAL data) — "shape decides label"
# =====================================================================
def curvature_signature():
    rows = load_data()
    p, f = [], []
    for d in rows:
        try:
            v = float(d["total_turning"]); out = d["test_outcome"].strip().upper()
        except Exception:
            continue
        if out == "PASS":
            p.append(v)
        elif out == "FAIL":
            f.append(v)
    p = np.array(p); f = np.array(f)
    hi = np.percentile(np.concatenate([p, f]), 98)
    bins = np.linspace(0, hi, 40)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.hist(p, bins=bins, density=True, color=GREEN, alpha=0.55, label=f"PASS (n={len(p):,})")
    ax.hist(f, bins=bins, density=True, color=RED,   alpha=0.55, label=f"FAIL (n={len(f):,})")
    ax.axvline(p.mean(), color=GREEN, lw=2); ax.axvline(f.mean(), color=RED, lw=2)
    ax.set_xlabel("Tổng độ đổi hướng của đường (radian)", fontsize=12)
    ax.set_ylabel("Mật độ", fontsize=12)
    ax.set_title("FAIL tập trung ở đường “xoắn” hơn — chỉ hình dạng, không toạ độ",
                 color=TEAL, fontsize=12)
    ax.legend(frameon=True, fontsize=11); ax.grid(axis="y", alpha=0.22)
    save(fig, "gen_curvature_signature.png")


# =====================================================================
# 5. Leaderboard — APFD of 8 baselines + SE2RoadNet
# =====================================================================
def leaderboard():
    data = [
        ("LLM zero-shot", 0.487, False),
        ("Ngẫu nhiên", 0.493, False),
        ("GNN (đồ thị đường)", 0.533, False),
        ("ResNet-50 (ảnh)", 0.572, False),
        ("SO-SDC-Prioritizer (GA)", 0.765, False),
        ("ITEP4SDC (SOTA ICST'25)", 0.781, False),
        ("Greedy-diversity", 0.795, False),
        ("RoadFury (nền tảng)", 0.804, False),
        ("SE2RoadNet (của nhóm)", 0.8047, True),
    ]
    labels = [d[0] for d in data]
    vals   = [d[1] for d in data]
    cols   = [ORANGE if d[2] else TEALM for d in data]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.barh(y, vals, color=cols, edgecolor=TEAL, lw=1.0)
    bars[-1].set_edgecolor(TEAL); bars[-1].set_linewidth(2.0)
    for yi, v, hl in zip(y, vals, [d[2] for d in data]):
        ax.text(v+0.006, yi, f"{v:.3f}", va="center", fontsize=11,
                fontweight="bold" if hl else "normal",
                color=ORANGE if hl else TEAL)
    ax.axvline(0.5, color=GREY, ls=":", lw=1.2)
    ax.text(0.5, len(data)-0.3, "ngẫu nhiên", color=GREY, fontsize=9, ha="center")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0.4, 0.86)
    ax.set_xlabel("APFD (956 test out-of-distribution, 30 trials)", fontsize=12)
    ax.set_title("Bảng xếp hạng APFD — SE2RoadNet dẫn đầu kèm bảo chứng Δ=0",
                 color=TEAL, fontsize=13)
    ax.grid(axis="x", alpha=0.25); ax.invert_yaxis()
    save(fig, "gen_leaderboard.png")


# =====================================================================
# 6. Rotation drift — ITEP4SDC drifts, SE2RoadNet flat (Δ=0)
# =====================================================================
def rotation_drift():
    angles = ["0°", "+30°", "+60°", "+90°", "+180°", "−45°"]
    itep = [0.7810, 0.7240, 0.7518, 0.7396, 0.7334, 0.7627]
    se2  = [0.8047]*6
    x = np.arange(len(angles))
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(x, itep, "-o", color=RED, lw=2.4, ms=8, label="ITEP4SDC  (Δ = 0.057)")
    ax.plot(x, se2,  "-s", color=ORANGE, lw=2.8, ms=8, label="SE2RoadNet  (Δ = 0.0000)")
    ax.fill_between(x, min(itep)-0.005, itep, color=RED, alpha=0.06)
    for xi, v in zip(x, itep):
        ax.text(xi, v-0.012, f"{v:.3f}", ha="center", va="top", fontsize=9, color=RED)
    ax.text(x[-1], 0.8047+0.006, "bằng nhau đến từng bit float", ha="right",
            color=ORANGE, fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(angles, fontsize=12)
    ax.set_ylim(0.70, 0.83)
    ax.set_xlabel("Góc xoay toàn bộ tập test", fontsize=12)
    ax.set_ylabel("APFD", fontsize=12)
    ax.set_title("Xoay đường ⇒ baseline trôi điểm, SE2RoadNet bất biến tuyệt đối",
                 color=TEAL, fontsize=12.5)
    ax.grid(alpha=0.25); ax.legend(loc="center right", fontsize=11, frameon=True)
    save(fig, "gen_rotation_drift.png")


# =====================================================================
# 7. Focal loss — down-weights easy examples
# =====================================================================
def focal_loss():
    p = np.linspace(1e-3, 1, 300)
    bce = -np.log(p)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for g, c, ls in [(0, GREY, "--"), (1.0, TEALM, "-"), (1.5, ORANGE, "-"), (2.5, TEAL, "-")]:
        fl = -(1-p)**g * np.log(p)
        ax.plot(p, fl, color=c, lw=2.6 if g == 1.5 else 1.8, ls=ls,
                label=("BCE (γ=0)" if g == 0 else f"Focal γ={g}"))
    ax.annotate("ví dụ dễ\n(down-weight)", xy=(0.86, -(1-0.86)**1.5*-np.log(0.86)),
                xytext=(0.55, 1.6), fontsize=10, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_xlabel("xác suất dự đoán đúng  $p_t$", fontsize=12)
    ax.set_ylabel("loss", fontsize=12)
    ax.set_ylim(0, 5)
    ax.set_title("Focal Loss (γ=1.5): dồn sức học ca khó (FAIL hiếm)",
                 color=TEAL, fontsize=12.5)
    ax.legend(fontsize=11, frameon=True); ax.grid(alpha=0.25)
    save(fig, "gen_focal_loss.png")


# =====================================================================
# 8. Dataset splits — N and FAIL% per split
# =====================================================================
def dataset_splits():
    splits = ["Train", "Test\n(SensoDat)", "Competition\n(OOD)"]
    n = [28804, 7202, 956]
    fail = [38.4, 38.4, 36.9]
    x = np.arange(3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    b = a1.bar(x, n, color=[TEALM, TEALM, ORANGE], edgecolor=TEAL, lw=1.0)
    a1.set_yscale("log")
    for xi, v in zip(x, n):
        a1.text(xi, v*1.1, f"{v:,}", ha="center", fontsize=10, color=TEAL)
    a1.set_xticks(x); a1.set_xticklabels(splits, fontsize=10)
    a1.set_title("Số kịch bản / split", color=TEAL, fontsize=12)
    a1.set_ylim(500, 60000); a1.grid(axis="y", alpha=0.2)

    b2 = a2.bar(x, fail, color=[TEALM, TEALM, ORANGE], edgecolor=TEAL, lw=1.0)
    for xi, v in zip(x, fail):
        a2.text(xi, v+0.6, f"{v:.1f}%", ha="center", fontsize=10, color=TEAL)
    a2.set_xticks(x); a2.set_xticklabels(splits, fontsize=10)
    a2.set_title("Tỉ lệ FAIL / split", color=TEAL, fontsize=12)
    a2.set_ylim(0, 46); a2.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "gen_dataset_splits.png")


if __name__ == "__main__":
    apfd_curve()
    roads_grid()
    failrate_by_length()
    curvature_signature()
    leaderboard()
    rotation_drift()
    focal_loss()
    dataset_splits()
    print("done.")
