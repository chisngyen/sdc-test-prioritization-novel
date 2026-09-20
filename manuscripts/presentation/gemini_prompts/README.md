# Gemini image prompts cho slide SE(2)-Equivariant

Mỗi prompt thiết kế cho **Gemini 2.5 Pro / Imagen 4** (hoặc bất kỳ
text-to-image model nào). Output mong muốn: PNG 16:9 hoặc square,
phong cách **academic illustration**, **flat design**, **palette teal +
cam** (matches metropolis theme: `#23373B` teal + `#EB811B` cam).

Sau khi tạo, save vào `presentation/figures/` với tên đúng để file `.tex`
biên dịch không lỗi.

---

## fig_hero_sdc.png (slide tổng quan)

> A clean academic-style illustration in flat vector design.
> Subject: a blue autonomous self-driving car (top-down 3/4 view) driving
> along a smoothly curving asphalt road that winds through a green countryside.
> The road has visible lane markings. A subtle digital overlay (thin orange
> dashed lines and small geometric markers) traces the road's curvature and
> heading, suggesting that an AI is analyzing the road's shape, not just
> the picture. The overall tone is technical but inviting. White background,
> minimal shadow. Aspect ratio 16:9. No text. Style: clean infographic /
> editorial illustration. Color palette: dark teal #23373B, accent orange
> #EB811B, soft white.

---

## fig_problem_motivation.png (slide "vì sao bài toán quan trọng")

> An academic infographic illustrating the cost of regression testing for
> self-driving car simulators. Left half: a stack of dozens of small road
> tiles (each showing a different curved road) labeled "thousands of test
> scenarios". Right half: a large stopwatch icon with the label "many CPU
> hours". A central red arrow points from "all tests" to "prioritized few"
> with the label "we must rank them". Flat editorial illustration style,
> palette: deep teal #23373B, warm orange #EB811B, neutral grey. White
> background. Aspect ratio 16:9. No technical jargon -- this is a motivation
> slide for an academic talk.

---

## fig_rotation_problem.png (slide "hai điểm yếu của baseline")

> An academic diagram in flat vector style. Show the same curved road
> drawn three times, side-by-side: (1) original orientation, (2) rotated
> 30 degrees, (3) rotated 90 degrees. The same blue car is placed at the
> start of each road. Above each road, a "score" gauge shows different
> numbers (e.g. 0.81, 0.74, 0.71). Below the diagram, a red sad-face emoji
> with the caption "the model gives different scores to the same road --
> wrong!". Editorial illustration style, palette teal/orange, white
> background. Aspect ratio 16:9. Subtle dashed coordinate axes behind
> each road to suggest the world frame.

---

## fig_se2_intuition.png (slide "ý tưởng SE(2) equivariance")

> A conceptual academic illustration. A curved road is shown floating in
> 3D space, with three semi-transparent rotated copies of itself overlapping.
> Above the stack of roads, a glowing orange box labeled "f(road)" outputs
> a single fixed score "0.804" with a green check mark. Below the stack,
> a short caption reads "same shape -> same score, by construction".
> Flat editorial illustration with subtle glow effects, palette teal +
> orange + soft success-green for the check mark. White background.
> Aspect ratio 16:9. No text on the road itself; let the visual carry the
> "rotation equivariance" idea.

---

## fig_pipeline_blocks.png (slide "pipeline tổng thể")

> A clean horizontal 4-step pipeline diagram, academic flat style. From left
> to right, four rounded rectangle boxes connected by arrows:
> Box 1: "Raw road points (x, y)" with a small icon of scattered dots.
> Box 2: "7-channel SE(2)-invariant features" with a small icon of a road
> with curvature markings.
> Box 3: "Equivariant Transformer (6 layers)" with a small icon of
> attention heads connected in a graph.
> Box 4: "FAIL probability + ranking" with a small icon of an ordered list.
> Arrows are orange. Box outlines are dark teal. Background is soft grey.
> Caption below the pipeline: "SE2RoadNet". Aspect ratio 16:9, wide.

---

## fig_attention_relative.png (slide "attention với relative arclength")

> A schematic of self-attention with relative positional bias.
> Show 6 token boxes in a row labeled s_1 ... s_6, each representing a
> point on a road. Between two highlighted tokens (s_3 and s_5), draw a
> bold orange arrow with the label "delta s = s_5 - s_3" -- emphasizing
> the _relative_ distance, not absolute position. To the side, a small
> formula plate displays "bias = MLP(sin(delta_s \* w))". Below, a
> mini-attention heatmap (6x6) shows that attention strength decays with
> |delta_s|. Academic vector style, palette dark teal + orange, white
> background. Aspect ratio 16:9.

---

## fig_eval_protocol.png (slide "giao thức đánh giá")

> An academic infographic showing the evaluation protocol. Three vertical
> panels:
> Left panel: "Single-pass APFD" -- a road list with a single ranked
> ordering and one APFD number 0.8047.
> Middle panel: "Multi-trial APFD (30 trials)" -- 30 small subsampled
> test sets piled on top of each other, with a histogram of APFD values
> showing mean 0.8048 +/- 0.0118.
> Right panel: "Rotation probe" -- the same road list rotated by 6 angles
> (0, +30, +60, +90, +180, -45 deg) all producing the identical APFD =
> 0.8047 with a big green "Delta = 0" badge.
> Connect the three panels with subtle teal arrows. Flat editorial style,
> palette teal + orange + green for the headline result. Aspect ratio 16:9.

---

## fig_takeaways.png (slide kết luận)

> A simple, calm closing illustration: a self-driving car silhouette on
> a winding road, with three glowing badges floating above it labeled
> "Rotation-invariant", "Sampling-robust", "Audit-readable". The road
> stretches into a soft sunrise horizon. Editorial illustration style,
> warm-but-academic palette (teal road, orange sky, white space).
> Aspect ratio 16:9. No body text -- badges only.

---

## (Optional) fig_failure_examples.png

Nếu bạn KHÔNG muốn dùng grid PNG từ Kaggle, có thể yêu cầu Gemini render:

> Two side-by-side schematic road examples in flat editorial style.
> Left: a smooth, gently curving road with a green check icon labeled
> "PASS". Right: an aggressive S-curve / chicane with a red X icon labeled
> "FAIL". Below each road, three small bars compare metrics
> (curvature, length, std). Palette teal/orange/green/red. Aspect ratio 16:9.

(Nhưng tui khuyên dùng ảnh thật từ Kaggle qua `sensodat_roads_grid.png`
cho mục này -- thuyết phục hơn vì là đường thật trong dataset.)
