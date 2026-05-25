# Exp 02 — SE(2)-Equivariant RoadNet — Video Pipeline (ManimCE)

Bộ animation 3b1b-style giải thích pipeline của `exps/exp02_SE2Equivariant.py`:
từ road points raw đến predicted FAIL probability, đi qua từng feature, từng
attention block, từng head layer.

## Storyboard (9 scenes ~ 7 phút)

| # | File | Scene | Dài | Nội dung |
|---|------|-------|-----|----------|
| 0 | `scene_00_context.py`      | `Context`           | 47s | Project context: SDC competition, metric APFD, project map, Exp 02 trong bức tranh tổng thể. |
| 1 | `scene_01_intro.py`        | `Intro`             | 47s | Bài toán: vì sao baseline bị "vỡ" khi xoay. Đặt phương trình `f(Rr + t) = f(r)`. |
| 2 | `scene_02_input.py`        | `InputPoints`       | 28s | Input = chuỗi điểm `(x_i, y_i)`. Tensor shape `(L, 2)`. |
| 3 | `scene_03_features.py`     | `FeatureExtract`    | 81s | 7-channel intrinsic features. Mỗi feature có công thức + viz trên road thật. |
| 4 | `scene_04_invariance.py`   | `RotationProof`     | 30s | Hai road song song (gốc + xoay 60°) → 7 features **bằng nhau từng con số**. |
| 5 | `scene_05_architecture.py` | `Architecture`      | 27s | Tổng quan SE2RoadNet: Linear → CLS → 6 InvariantBlocks → head. |
| 6 | `scene_06_attention.py`    | `AttentionBlock`    | 32s | Tại sao **relative-arclength bias** làm attention bất biến với phép xoay. |
| 7 | `scene_06b_compute.py`     | `ComputeWalkthrough`| 93s | **Walkthrough numeric chi tiết**: 1 tensor đi xuyên model với số thật ở mọi bước. |
| 8 | `scene_07_results.py`      | `Results`           | 37s | APFD bar chart 6 góc xoay, Δ = 0.0000, scoreboard. |

Tổng: **~7 phút** ở 480p15. File ghép sẵn: `full_video_480p15.mp4`.

## Render

```powershell
# Preview low quality (nhanh, 480p)
manim -pql scene_01_intro.py Intro

# Production quality (1080p60)
manim -qh scene_01_intro.py Intro

# 4K
manim -qk scene_01_intro.py Intro

# GIF
manim --format gif -qm scene_03_features.py FeatureExtract
```

Render tất cả scenes:

```powershell
python render_all.py        # -ql preview
python render_all.py --hq   # -qh final
```

Output mặc định ManimCE: `media/videos/<scene_file>/<quality>/<SceneName>.mp4`.

## Ghép thành 1 video tổng hợp

Sau khi đã render xong các scenes, ghép lại bằng ffmpeg (đã sẵn ở
`C:\Program Files\ffmpeg-*\bin\ffmpeg.exe` hoặc trên PATH):

```powershell
python concat_all.py          # ghép 480p15 (preview)  -> full_video_480p15.mp4
python concat_all.py --hq     # ghép 1080p60 (final)   -> full_video_1080p60.mp4
python concat_all.py --4k     # ghép 2160p60           -> full_video_2160p60.mp4
```

Script dùng ffmpeg concat demuxer với `-c copy` -- không re-encode nên chỉ
mất vài giây. Kết quả là một file `full_video_<quality>.mp4` đặt ngay trong
folder này.

Tham khảo: bản preview 480p15 đầy đủ dài **5:15**, kích thước **~4.5 MB**.

## Convention

- Tất cả scene tự đứng độc lập, chỉ import từ `common.py`.
- Road mẫu được sinh trong `common.py` để mọi scene cùng nhìn một con đường.
- Màu sắc bám theo feature (xem `common.FEATURE_COLORS`) — giữ nhất quán xuyên suốt video.
- Code Manim, comment, LaTeX viết bằng English (giảm khả năng lỗi font ở Windows).

## Narration (Vietnamese)

Mỗi scene có một file MP3 narration tiếng Việt trong `narration/`, sinh
bằng Edge-TTS với giọng `vi-VN-HoaiMyNeural`.  Audio được attach vào
`construct()` qua `attach_narration(self, "scene_NN")` và scene tự
extend wait ở cuối qua `seal_narration(self, "scene_NN")` -- đảm bảo
video không cắt giữa câu nói.

```powershell
pip install edge-tts
python narration/generate.py              # all 9 scenes
python narration/generate.py scene_03     # chỉ scene_03
```

Edit `narration/scripts.py` (NARRATION dict) để chỉnh lời, rồi chạy lại
`generate.py`.  Mỗi lần chạy cập nhật `narration/audio_durations.json`
mà các scene đọc để pad wait cho khớp.

## Phụ thuộc

```powershell
pip install manim==0.20.1 edge-tts
manim checkhealth
```

Trên Windows cần thêm: MiKTeX (cho LaTeX), FFmpeg (đi kèm Manim), và Pango
(đi kèm `manimpango`).
