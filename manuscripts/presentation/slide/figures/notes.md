# Slide Asset & Reference Notes — RoadFury → SE2RoadNet

Nguồn của tài nguyên ngoài + số liệu trích dẫn trong deck, để mở ra khi thầy hỏi kiểm chứng.
Deck này (giống template CGAR) **tự vẽ toàn bộ sơ đồ/biểu đồ bằng SVG/CSS** → chỉ còn logo + video là ảnh ngoài.

## 1. Ảnh raster ngoài (cần dẫn nguồn — đã liệt kê ở slide "Sources" cuối + _pptx_build/build_pptx.py)

| Ảnh | File | Nguồn |
|-----|------|-------|
| Logo HCMUS | `google-slides/uploads/hcmus.png` | https://www.hcmus.edu.vn |
| Logo ICST 2026 | `google-slides/uploads/icst_logo.jpg` | https://conf.researchr.org/home/icst-2026 |
| Đường thật + histogram độ cong (s07) | `uploads/roads_grid.png`, `uploads/curvature_hist.png` | Vẽ từ bộ dữ liệu SensoDat — https://github.com/christianbirchler-org/sensodat |
| Ảnh chụp mô phỏng (s07) | `uploads/beamng_lane.png` | BeamNG.tech — https://www.beamng.tech/ |
| Sơ đồ kiến trúc TikZ (s11, s15) | `uploads/fig_arch.png`, `uploads/fig_arch_se2.png` | Tự vẽ bằng TikZ (`figures/fig_arch*.tex`) — KHÔNG cần nguồn ngoài |
| Thumbnail video | (chưa có — placeholder vẽ bằng CSS) | Khung hình từ video của nhóm; thay trước khi nộp |

## 2. Link video (track Method 60%)
- Google Drive: https://drive.google.com/file/d/1JC0NY3qfW-if9cM74Zi3d0VaTcA-1el_/view?usp=sharing
- Facebook reel: https://www.facebook.com/reel/1528755028625867

## 3. Số liệu trích dẫn / dữ liệu
- **SensoDat** (Birchler et al., MSR 2024) — bộ dữ liệu gốc; mọi biểu đồ fail-rate/độ dài vẽ lại từ split này.
  https://github.com/christianbirchler-org/sensodat
- APFD baselines (Random 0.493, GNN 0.533, ResNet-50 0.572, SO-SDC 0.765, ITS4SDC 0.781,
  Greedy 0.795, RoadFury 0.804, SE2RoadNet 0.805): **nhóm tái lập trong cùng harness** (956 test OOD, 30 trials).
  RoadFury & SE2RoadNet là phương pháp của nhóm → không cần cite ngoài.
  - Verified vs literature (Birchler et al., TOSEM 2023, arXiv:2107.09614): Greedy 79.5% ✓, SO-SDC 76.5% ✓,
    Random 49.9% (~0.493). SO-SDC = ref [2].
  - **ITS4SDC** = Güllü, Shah, Pfahl, "An LSTM-based Test Selection Method for SDC", ICST 2025 SDC Tool
    Competition (arXiv:2501.03881). Là **bi-LSTM** trên 2 đặc trưng chuỗi (góc đoạn + chiều dài), test
    SELECTION. Paper chỉ công bố F1=0.89, KHÔNG có APFD → 0.781 + bảng xoay là **nhóm tái lập**. LƯU Ý:
    paper định nghĩa góc là "góc giữa 2 đoạn kề" (turning angle, vốn bất biến xoay) — drift Δ=0.057 chỉ
    đúng nếu bản tái lập của nhóm dùng góc/heading tuyệt đối (phụ thuộc khung). Cần sẵn log cho Q&A.
- Rotation probe (ITS4SDC Δ=0.057 vs SE2RoadNet Δ=0.0000), multi-trial 0.8048±0.0118,
  resolution Δ≈0.0012: thí nghiệm của nhóm.

## 4. Hình tự vẽ bằng SVG/CSS (KHÔNG cần dẫn nguồn ngoài)
Tất cả: road infographics, INPUT→SCORER→OUTPUT, APFD curve, fail-rate bars, 3 hướng related work,
scatter 2 trục, RoadFury & SE2RoadNet pipeline, 4-panel SE(2), Δs attention heatmap, focal-loss curve,
rotation-drift line, leaderboard bars, 2×2 future, sơ đồ chứng minh bất biến → tự vẽ trong deck.
