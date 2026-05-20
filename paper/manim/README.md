# RoadFury — 3blue1brown-style explainer video

Single file: [`road_fury_video.py`](road_fury_video.py). Nine scenes that
tell the whole paper story:

| # | Scene class         | What it shows                                                 |
|---|---------------------|---------------------------------------------------------------|
| 1 | `Title`             | RoadFury title card with a road that draws itself             |
| 2 | `Problem`           | 1,000 tests, only a few fail — which to run first?            |
| 3 | `APFDExplained`     | Colour-coded APFD formula + intuition                         |
| 4 | `AggregationLoss`   | Two roads with identical 3-number summaries fail differently  |
| 5 | `FeatureExtraction` | Resample to L=197 and extract 10 geometry channels            |
| 6 | `TransformerView`   | [CLS] token, self-attention, 4 Pre-LN encoder layers          |
| 7 | `SWAVisualization`  | SWA averages SGD snapshots into a flatter minimum (0.788→0.804) |
| 8 | `Results`           | Bar chart: Random / LLM / GNN / CNN / ITEP4SDC / RoadFury     |
| 9 | `Conclusion`        | "Sequence > average." + APFD 0.804 + repo URL                 |

## Install Manim Community once

```powershell
pip install manim
manim checkhealth
```

On Windows you also need MiKTeX (LaTeX) and `ffmpeg` on PATH for the
math scenes — `manim checkhealth` will flag anything missing.

## Render a single scene

```powershell
cd paper/manim
manim -pql  road_fury_video.py Title            # quick 480p draft
manim -pqh  road_fury_video.py Results          # 1080p final
manim -pqh --format gif road_fury_video.py Problem
```

`-p` previews, `-q{l,m,h,k}` picks quality (low / medium / high / 4K).

## Render the whole story

Render each scene to its own `.mp4`, then concat with ffmpeg:

```powershell
$scenes = @("Title","Problem","APFDExplained","AggregationLoss",
            "FeatureExtraction","TransformerView","SWAVisualization",
            "Results","Conclusion")
foreach ($s in $scenes) { manim -qh road_fury_video.py $s }

# Concatenate (paths come from Manim's media/videos/road_fury_video/1080p60/)
$list = $scenes | ForEach-Object { "file '$pwd\media\videos\road_fury_video\1080p60\$_.mp4'" }
$list | Set-Content -Encoding ascii concat.txt
ffmpeg -f concat -safe 0 -i concat.txt -c copy roadfury_full.mp4
```

(The `FullStory` class in the script is a convenience that chains
everything in one shot, but the per-scene + ffmpeg concat workflow is
more reliable and lets you re-render single scenes.)

## Editing notes

- Palette constants at the top of the file (`C_HI`, `C_BLUE`, …) match
  the 3b1b warm-on-dark look.
- All numbers cited in scenes (`0.788 → 0.804`, `APFD 0.493 / 0.487 /
  0.533 / 0.572 / 0.781 / 0.804`) come directly from
  [`../paper.tex`](../paper.tex) Tables I–II.
- To tweak pacing, change the `run_time=` arguments on each `self.play`
  call — most are between 0.8 s and 2.4 s.
