# Presentation: SE(2)-Equivariant RoadNet

Slide tiếng Việt cho buổi báo cáo phương pháp **Exp 02** với thầy.

## Files

- `se2_slides.tex` -- slide chính (LaTeX, Beamer metropolis theme).
- `se2_speaker_script.md` -- script thuyết trình ~18-22 phút.
- `dataset_stats_kaggle.py` -- chạy trên Kaggle để xuất số liệu SensoDat
  + 3 ảnh minh hoạ.
- `gemini_prompts/README.md` -- 8 prompt Gemini cho ảnh hero.
- `figures/` -- nơi đặt **toàn bộ** ảnh slide (Gemini outputs + Kaggle
  outputs).

## Quy trình tạo slide hoàn chỉnh (3 bước)

### Bước 1. Chạy Kaggle script lấy số liệu thật

1. Mở notebook Kaggle mới, attach dataset `chinguyeen/sdc-sensodat`.
2. Paste toàn bộ nội dung [`dataset_stats_kaggle.py`](dataset_stats_kaggle.py)
   vào cell và run.
3. Sau khi chạy xong (~2-5 phút), download 4 file output từ
   `/kaggle/working/`:
   - `dataset_stats.txt`
   - `sensodat_roads_grid.png`
   - `sensodat_apfd_curve.png`
   - `sensodat_failrate_by_length.png`
4. Copy 3 file PNG vào [`figures/`](figures/).
5. Paste nội dung `dataset_stats.txt` vào chat cho Claude -- tui sẽ
   điền số thật vào bảng Dataset trên slide 7.

### Bước 2. Tạo ảnh hero bằng Gemini

1. Mở [`gemini_prompts/README.md`](gemini_prompts/README.md).
2. Với mỗi prompt, paste vào Gemini 2.5 Pro (hoặc Imagen 4), tạo ảnh,
   save vào `figures/` với **đúng tên file** đã ghi trong prompt:
   - `fig_hero_sdc.png`
   - `fig_problem_motivation.png`
   - `fig_rotation_problem.png`
   - `fig_se2_intuition.png` *(slide 10 đang dùng tikz, ảnh này optional)*
   - `fig_pipeline_blocks.png` *(slide 11 dùng tikz, optional)*
   - `fig_attention_relative.png`
   - `fig_eval_protocol.png`
   - `fig_takeaways.png`

Slide có thể compile **trước khi có ảnh** -- nó sẽ chỉ trống chỗ ảnh
chứ không lỗi. Nhưng để demo cho thầy thì nên có hết.

### Bước 3. Compile

```powershell
cd presentation
pdflatex se2_slides.tex
pdflatex se2_slides.tex   # chạy 2 lần cho TOC/refs
```

Output: `se2_slides.pdf` -- 24 slide aspect ratio 16:9.

## Cấu trúc slide (24 slides)

| # | Phần | Nội dung |
|---|------|----------|
| 1 | Title | Tên đề tài + tác giả |
| 2 | Agenda | 6 phần |
| 3-4 | I. Tổng quan | Bài toán SDC test prio, motivation |
| 5-8 | II. Input/Output + Dataset | Formal IO, APFD, SensoDat, ví dụ |
| 9-17 | **III. Phương pháp (top-down)** | 2 điểm mù -> định lý -> pipeline -> 4 bước drill-down |
| 18-20 | IV. Đánh giá | Protocol, $\Delta=0$, AUC cao nhất |
| 21 | V. Leaderboard | So sánh 8 baselines |
| 22-24 | VI. Kết luận | Take-aways, future work, Q&A |

## Phong cách

- **Theme**: metropolis (giống `summary_proposal.tex`).
- **Palette**: dark teal `#23373B` + cam `#EB811B` + success green.
- **Tiếng Việt**: `vietnam.sty` + `[utf8]inputenc`.
- **Toán**: chỉ dùng khi cần làm rõ (định lý SE(2), công thức APFD).
- **Math symbol**: ASCII trong code snippet, LaTeX inline với `$...$`.
- **Top-down method**: slide 10 (định lý) -> 11 (pipeline 4 box) -> 12-16
  (drill-down từng bước).

## Khi nào cần báo cho Claude

- [ ] Sau khi chạy Kaggle script: paste `dataset_stats.txt` để điền
      bảng slide 7.
- [ ] Nếu Gemini không ra ảnh đẹp: báo prompt nào fail, Claude refine.
- [ ] Nếu thầy có góp ý nội dung: gửi Claude để revise slide tương ứng.
- [ ] Nếu pdflatex báo lỗi unicode: paste error log để fix.
