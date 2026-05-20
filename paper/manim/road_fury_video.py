"""
RoadFury: a 3blue1brown-style explainer video.

Render any single scene:
    manim -pql road_fury_video.py SceneName

Render the whole story (chain them in a playlist or use --save_sections):
    manim -pqh road_fury_video.py Title
    manim -pqh road_fury_video.py Problem
    manim -pqh road_fury_video.py APFDExplained
    manim -pqh road_fury_video.py AggregationLoss
    manim -pqh road_fury_video.py FeatureExtraction
    manim -pqh road_fury_video.py TransformerView
    manim -pqh road_fury_video.py SWAVisualization
    manim -pqh road_fury_video.py Results
    manim -pqh road_fury_video.py Conclusion

Quality flags: -ql (480p draft), -qm (720p), -qh (1080p), -qk (4K).
"""

from manim import *
import numpy as np


# ----------------------------------------------------------------------------
# 3b1b-ish palette (black background is Manim default)
# ----------------------------------------------------------------------------
C_BG = "#0c0c0c"
C_ROAD = "#cccccc"
C_PASS = "#3ec96d"   # 3b1b green
C_FAIL = "#ee5253"   # 3b1b red
C_HI = "#ffd166"     # warm highlight
C_BLUE = "#5fa8d3"   # 3b1b sky blue
C_PURPLE = "#b497d6"
C_TEAL = "#5fd3c2"


def road_curve(t, kind="curvy"):
    """Parametric road for visualisation. t in [0, 1]."""
    x = 6.0 * (t - 0.5)
    if kind == "curvy":
        y = 0.9 * np.sin(2.0 * PI * t) + 0.4 * np.sin(6.0 * PI * t)
    elif kind == "straight":
        y = 0.05 * np.sin(8.0 * PI * t)
    elif kind == "hairpin":
        y = 1.2 * np.sin(3.0 * PI * t) * np.exp(-((t - 0.5) ** 2) * 4)
    elif kind == "smooth":
        y = 0.8 * np.sin(PI * t)
    else:
        y = 0.0
    return np.array([x, y, 0.0])


# ============================================================================
# SCENE 1 — Title
# ============================================================================
class Title(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # Road that draws itself across the screen and curves into the title.
        road = ParametricFunction(
            lambda t: road_curve(t, "curvy"),
            t_range=[0, 1],
            color=C_ROAD,
            stroke_width=6,
        )
        road_glow = road.copy().set_stroke(color=C_BLUE, width=14, opacity=0.25)

        self.play(Create(road_glow, run_time=2.2), Create(road, run_time=2.2))
        self.wait(0.3)

        title = Text("RoadFury", font_size=96, weight=BOLD, color=C_HI)
        subtitle = Text(
            "Teaching a Transformer to read roads",
            font_size=34,
            color=C_ROAD,
        )
        venue = Text(
            "ICST 2026  ·  SDC Testing Competition",
            font_size=24,
            color=C_BLUE,
        )

        title.to_edge(UP, buff=1.2)
        subtitle.next_to(title, DOWN, buff=0.35)
        venue.next_to(subtitle, DOWN, buff=0.6)

        self.play(FadeIn(title, shift=DOWN * 0.4), run_time=0.9)
        self.play(Write(subtitle), run_time=1.0)
        self.play(FadeIn(venue, shift=UP * 0.2), run_time=0.8)
        self.wait(2.0)

        self.play(
            *[FadeOut(m) for m in [road, road_glow, title, subtitle, venue]],
            run_time=1.0,
        )


# ============================================================================
# SCENE 2 — The Problem
# ============================================================================
class Problem(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # Question framing.
        q = Text(
            "You have 1,000 simulator tests.",
            font_size=42,
            color=C_ROAD,
        )
        q2 = Text(
            "Only a few will actually crash the car.",
            font_size=42,
            color=C_ROAD,
        )
        q3 = Text(
            "Which ones do you run first?",
            font_size=46,
            color=C_HI,
            weight=BOLD,
        )

        VGroup(q, q2, q3).arrange(DOWN, buff=0.55).move_to(ORIGIN)
        self.play(Write(q), run_time=1.2)
        self.play(Write(q2), run_time=1.4)
        self.play(Write(q3), run_time=1.4)
        self.wait(1.4)
        self.play(FadeOut(VGroup(q, q2, q3)), run_time=0.7)

        # A queue of tests. Most pass (green), a few fail (red).
        N = 14
        n_fail = 3
        rng = np.random.default_rng(7)
        fail_idx = set(rng.choice(N, size=n_fail, replace=False))

        tests = VGroup()
        for i in range(N):
            box = Square(side_length=0.55, color=C_PASS, fill_opacity=0.85)
            tests.add(box)
        tests.arrange(RIGHT, buff=0.18).shift(UP * 0.4)
        for i in fail_idx:
            tests[i].set_color(C_FAIL).set_fill(C_FAIL, opacity=0.95)

        label_random = Text("Random order  (most early runs reveal nothing)",
                            font_size=26, color=C_ROAD).next_to(tests, UP, buff=0.5)

        self.play(LaggedStart(*[FadeIn(t, shift=DOWN * 0.2) for t in tests],
                              lag_ratio=0.05),
                  FadeIn(label_random),
                  run_time=1.6)
        self.wait(1.0)

        # Highlight fails: scattered across the queue.
        marks = VGroup(*[
            Cross(stroke_color=C_HI, stroke_width=5).scale(0.25).move_to(tests[i])
            for i in fail_idx
        ])
        self.play(FadeIn(marks), run_time=0.8)
        self.wait(1.0)

        # Now: a clever ordering puts fails first.
        target_positions = [t.get_center() for t in tests]
        order = list(fail_idx) + [i for i in range(N) if i not in fail_idx]

        anims = []
        for slot, src in enumerate(order):
            anims.append(tests[src].animate.move_to(target_positions[slot]))
        for slot, src in enumerate(order):
            if src in fail_idx:
                anims.append(
                    marks[list(fail_idx).index(src)]
                    .animate.move_to(target_positions[slot])
                )

        good = Text("RoadFury order  (fails come first)",
                    font_size=26, color=C_HI).next_to(tests, UP, buff=0.5)
        self.play(ReplacementTransform(label_random, good), *anims, run_time=2.0)
        self.wait(1.5)

        self.play(FadeOut(VGroup(tests, marks, good)), run_time=0.7)


# ============================================================================
# SCENE 3 — APFD explained
# ============================================================================
class APFDExplained(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text("How do we score an ordering?",
                     font_size=40, color=C_ROAD).to_edge(UP)
        self.play(Write(title))
        self.wait(0.4)

        # APFD formula, colour-coded.
        apfd = MathTex(
            r"\text{APFD}(\pi) \,=\, 1 \,-\,",
            r"\frac{\sum_{i=1}^{m} TF_i}{n \cdot m}",
            r"\,+\, \frac{1}{2n}",
            font_size=54,
        )
        apfd[1].set_color(C_HI)
        apfd.next_to(title, DOWN, buff=0.7)
        self.play(Write(apfd), run_time=2.2)

        legend = VGroup(
            Text("n  = number of tests",  font_size=26, color=C_ROAD),
            Text("m  = number of failing tests", font_size=26, color=C_ROAD),
            Text("TF = rank where the i-th fault is found",
                 font_size=26, color=C_HI),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        legend.next_to(apfd, DOWN, buff=0.6)
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.2) for t in legend],
                              lag_ratio=0.25), run_time=1.6)
        self.wait(1.2)

        intuition = Text(
            "Faults found earlier  ⇒  smaller TF sum  ⇒  higher APFD.",
            font_size=28, color=C_PASS,
        )
        intuition.to_edge(DOWN, buff=0.7)
        self.play(Write(intuition), run_time=1.6)
        self.wait(2.0)

        self.play(FadeOut(VGroup(title, apfd, legend, intuition)), run_time=0.8)


# ============================================================================
# SCENE 4 — Why aggregation throws away the road
# ============================================================================
class AggregationLoss(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        header = Text("Prior tools summarise each road as 3 numbers",
                      font_size=34, color=C_ROAD).to_edge(UP)
        self.play(Write(header))

        # Two roads, very different shape, same mean curvature.
        road_a = ParametricFunction(
            lambda t: np.array([4*(t-0.5),
                                0.6*np.sin(2*PI*t) - 0.6*np.sin(4*PI*t),
                                0]),
            t_range=[0, 1], color=C_BLUE, stroke_width=5,
        ).shift(LEFT * 3.4 + DOWN * 0.4)
        road_b = ParametricFunction(
            lambda t: np.array([4*(t-0.5),
                                1.1*np.sin(PI*t)*(2*t-1),
                                0]),
            t_range=[0, 1], color=C_TEAL, stroke_width=5,
        ).shift(RIGHT * 3.4 + DOWN * 0.4)

        lab_a = Text("Road A",  font_size=26, color=C_BLUE).next_to(road_a, UP, buff=0.4)
        lab_b = Text("Road B",  font_size=26, color=C_TEAL).next_to(road_b, UP, buff=0.4)

        self.play(Create(road_a), Create(road_b), FadeIn(lab_a), FadeIn(lab_b), run_time=1.8)
        self.wait(0.4)

        # Show identical 3-number summary.
        summary = VGroup(
            Text("mean κ   = 0.42", font_size=26, color=C_ROAD),
            Text("mean lat. = 0.07", font_size=26, color=C_ROAD),
            Text("std        = 0.31", font_size=26, color=C_ROAD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        summary.to_edge(DOWN, buff=0.6)
        equals = Text("=  identical fingerprint", font_size=28, color=C_HI)
        equals.next_to(summary, RIGHT, buff=0.5)

        self.play(FadeIn(summary, shift=UP * 0.2), run_time=1.0)
        self.play(Write(equals), run_time=1.2)
        self.wait(1.4)

        # Stamp the verdict.
        verdict = Text("…but they fail in totally different places.",
                       font_size=30, color=C_FAIL).to_edge(UP, buff=1.4)
        self.play(ReplacementTransform(header, verdict), run_time=1.0)

        # Mark hot spots on each road.
        hot_a = Dot(road_a.point_from_proportion(0.18), color=C_FAIL, radius=0.13)
        hot_b = Dot(road_b.point_from_proportion(0.72), color=C_FAIL, radius=0.13)
        self.play(GrowFromCenter(hot_a), GrowFromCenter(hot_b), run_time=0.9)
        self.play(Flash(hot_a, color=C_FAIL), Flash(hot_b, color=C_FAIL))
        self.wait(1.6)

        self.play(FadeOut(VGroup(verdict, road_a, road_b, lab_a, lab_b,
                                 summary, equals, hot_a, hot_b)),
                  run_time=0.8)


# ============================================================================
# SCENE 5 — Resample + 10-channel feature extraction
# ============================================================================
class FeatureExtraction(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text("RoadFury keeps the whole sequence",
                     font_size=38, color=C_HI).to_edge(UP)
        self.play(Write(title))
        self.wait(0.3)

        # Raw road -> 197 sampled points.
        road = ParametricFunction(
            lambda t: road_curve(t, "curvy"),
            t_range=[0, 1], color=C_BLUE, stroke_width=5,
        ).shift(UP * 0.6)
        self.play(Create(road), run_time=1.6)

        # Dots representing L=197 resampled points (we show 40 for clarity).
        n_show = 40
        dots = VGroup(*[
            Dot(road.point_from_proportion(i / (n_show - 1)),
                color=C_HI, radius=0.05)
            for i in range(n_show)
        ])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.03),
                  run_time=1.4)

        l_caption = MathTex(r"L = 197 \text{ uniformly resampled points}",
                            font_size=34, color=C_ROAD)
        l_caption.next_to(road, DOWN, buff=0.4)
        self.play(Write(l_caption), run_time=1.1)
        self.wait(0.6)

        # Pull out 10 channels next to the road.
        channels = [
            "f0  segment length",
            "f1  |Δθ|  angle change",
            "f2  Menger curvature  κ",
            "f3  curvature jerk  Δκ",
            "f4  cumulative distance",
            "f5  sin θ",
            "f6  cos θ",
            "f7  relative position",
            "f8  local curvature std",
            "f9  curvature acceleration",
        ]
        ch_text = VGroup(*[
            Text(c, font="Monospace", font_size=22, color=C_ROAD)
            for c in channels
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        ch_text.to_edge(RIGHT, buff=0.6).shift(DOWN * 0.4)

        box = SurroundingRectangle(ch_text, color=C_TEAL, buff=0.2,
                                   corner_radius=0.15)
        ch_label = Text("10 geometry channels  (197 × 10)",
                        font_size=24, color=C_TEAL,
                        weight=BOLD).next_to(box, UP, buff=0.18)

        self.play(
            FadeOut(l_caption),
            ReplacementTransform(road.copy().set_color(C_TEAL), box),
            run_time=0.8,
        )
        self.play(Write(ch_label),
                  LaggedStart(*[FadeIn(t, shift=LEFT * 0.2) for t in ch_text],
                              lag_ratio=0.08),
                  run_time=2.2)
        self.wait(2.0)

        self.play(FadeOut(VGroup(title, road, dots, box, ch_label, ch_text)),
                  run_time=0.8)


# ============================================================================
# SCENE 6 — Inside RoadFury: step-by-step architecture walkthrough
#
# Mirrors exps/exp00_Basline.py::RoadTransformer.forward exactly:
#   (B,10,197) → permute → (B,197,10)
#   → Linear(10→128) + LN + GELU              (B,197,128)
#   → prepend [CLS]                            (B,198,128)
#   → + sinusoidal pos embedding               (B,198,128)
#   → Pre-LN encoder × 4 :
#         LN → MHA(8 heads, d_k=16) + residual
#         LN → FFN(128→512→128, GELU) + residual
#   → pool [CLS]                               (B,128)
#   → LN → Linear(128→64) → GELU → Dropout(0.2)
#   → Linear(64→1) → sigmoid                   ŷ ∈ [0,1]
# ============================================================================


def tensor_block(rows, cols, h=2.6, w=1.4, color=C_BLUE, opacity=0.75,
                 depth=0.18, grid_rows=None, grid_cols=None,
                 max_grid_lines=24):
    """A 3D-ish tensor visualisation. `rows × cols` is the conceptual shape;
    `grid_rows / grid_cols` control how many cell lines actually get drawn."""
    front = Rectangle(height=h, width=w, color=color,
                      fill_opacity=opacity, stroke_width=2)
    grp = VGroup(front)
    if depth > 0:
        ur = front.get_corner(UR); dr = front.get_corner(DR); ul = front.get_corner(UL)
        shift = RIGHT * depth + UP * depth * 0.6
        side = Polygon(ur, ur + shift, dr + shift, dr,
                       color=color, fill_opacity=opacity * 0.55,
                       stroke_width=1.4)
        top = Polygon(ul, ur, ur + shift, ul + shift,
                      color=color, fill_opacity=opacity * 0.40,
                      stroke_width=1.4)
        grp.add(side, top)
    gr = grid_rows if grid_rows is not None else min(rows, max_grid_lines)
    gc = grid_cols if grid_cols is not None else min(cols, max_grid_lines)
    if gr > 1:
        for i in range(1, gr):
            y = front.get_top()[1] - i * (h / gr)
            grp.add(Line(np.array([front.get_left()[0], y, 0]),
                         np.array([front.get_right()[0], y, 0]),
                         stroke_width=0.6, stroke_color=WHITE,
                         stroke_opacity=0.30))
    if gc > 1:
        for j in range(1, gc):
            x = front.get_left()[0] + j * (w / gc)
            grp.add(Line(np.array([x, front.get_top()[1], 0]),
                         np.array([x, front.get_bottom()[1], 0]),
                         stroke_width=0.6, stroke_color=WHITE,
                         stroke_opacity=0.30))
    return grp


class TransformerView(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ───────────────────────── Header ─────────────────────────
        title = Text("Inside RoadFury", font_size=40, color=C_HI, weight=BOLD)
        sub = Text("road points  →  failure score, step by step",
                   font_size=24, color=C_ROAD)
        head = VGroup(title, sub).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.25)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2), run_time=1.2)

        # Shape tracker at the bottom; gets replaced as the tensor morphs.
        shape = MathTex(r"x\,:\;(B,\, 10,\, 197)", font_size=34, color=C_HI)
        shape.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(shape), run_time=0.5)

        # ───────────────── PHASE 1 — input tensor ─────────────────
        T_in = tensor_block(rows=10, cols=197, h=3.0, w=0.55, color=C_BLUE,
                            grid_rows=10, grid_cols=18)
        T_in.move_to(LEFT * 4.5 + DOWN * 0.1)

        feat_names = ["f0  length", "f1  |Δθ|", "f2  κ", "f3  Δκ", "f4  d/D",
                      "f5  sinθ", "f6  cosθ", "f7  pos", "f8  std(κ)", "f9  Δ²κ"]
        feat_lbls = VGroup()
        for i, name in enumerate(feat_names):
            y = T_in[0].get_top()[1] - (i + 0.5) * (3.0 / 10)
            t = Text(name, font_size=14, color=C_BLUE, font="Consolas")
            t.move_to([T_in[0].get_left()[0] - 1.05, y, 0])
            feat_lbls.add(t)

        cap_in = Text("10 channels × 197 points",
                      font_size=22, color=C_BLUE).next_to(T_in, DOWN, buff=0.3)

        self.play(FadeIn(T_in, scale=0.92), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.15)
                                for l in feat_lbls], lag_ratio=0.05),
                  run_time=1.4)
        self.play(FadeIn(cap_in, shift=UP * 0.15), run_time=0.5)
        self.wait(0.4)

        # Permute (10, 197) → (197, 10): rotate the tensor & shape label.
        shape2 = MathTex(r"x\,:\;(B,\, 197,\, 10)",
                         font_size=34, color=C_HI).to_edge(DOWN, buff=0.35)
        self.play(
            Rotate(VGroup(T_in, feat_lbls), angle=-PI / 2, about_point=T_in.get_center()),
            VGroup(T_in, feat_lbls).animate.move_to(LEFT * 4.5 + DOWN * 0.5),
            FadeOut(cap_in),
            ReplacementTransform(shape, shape2),
            run_time=1.4,
        )
        shape = shape2
        self.wait(0.5)

        # ───────────── PHASE 2 — input projection 10→128 ─────────────
        T_proj = tensor_block(rows=197, cols=128, h=0.55, w=3.0, color=C_TEAL,
                              grid_rows=2, grid_cols=24)
        T_proj.move_to(LEFT * 0.1 + DOWN * 0.5)

        proj_pill = self._pill("Linear  10 → 128",
                               "LayerNorm  +  GELU", C_TEAL)
        proj_pill.move_to(UP * 1.0)

        arrow1 = Arrow(T_in.get_right(), T_proj.get_left(),
                       color=C_TEAL, stroke_width=4, buff=0.15,
                       max_tip_length_to_length_ratio=0.12)
        self.play(FadeOut(feat_lbls), run_time=0.4)
        self.play(GrowArrow(arrow1), FadeIn(proj_pill, shift=DOWN * 0.2),
                  run_time=0.9)
        self.play(TransformFromCopy(T_in, T_proj), run_time=1.1)

        shape3 = MathTex(r"x\,:\;(B,\, 197,\, 128)",
                         font_size=34, color=C_HI).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(shape, shape3), run_time=0.5)
        shape = shape3
        self.wait(0.3)

        # ───────────── PHASE 3 — prepend [CLS] token ─────────────
        cls_cell = Square(side_length=0.42, color=C_HI, fill_opacity=0.95,
                          stroke_width=2.5)
        cls_lbl = Text("[CLS]", font_size=12, color=C_BG, weight=BOLD)
        cls_lbl.move_to(cls_cell)
        cls_token = VGroup(cls_cell, cls_lbl)
        cls_token.next_to(T_proj, LEFT, buff=0.12).shift(UP * 0.05)

        cls_caption = Text("Prepend a learnable [CLS] token",
                           font_size=22, color=C_HI).next_to(T_proj, UP, buff=0.55)
        self.play(FadeIn(cls_caption, shift=DOWN * 0.2), run_time=0.6)
        self.play(GrowFromCenter(cls_token), run_time=0.9)
        self.play(Flash(cls_token, color=C_HI, flash_radius=0.4))

        shape4 = MathTex(r"x\,:\;(B,\, 198,\, 128)",
                         font_size=34, color=C_HI).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(shape, shape4), run_time=0.5)
        shape = shape4
        self.wait(0.2)

        # ───────────── PHASE 4 — positional embedding ─────────────
        pos_caption = Text("Add positional embeddings",
                           font_size=22, color=C_PURPLE)\
            .next_to(T_proj, UP, buff=0.55)
        self.play(ReplacementTransform(cls_caption, pos_caption), run_time=0.5)

        # Sinusoidal "ribbon" hovering above the sequence.
        ax_pos = Axes(x_range=[0, 198, 50], y_range=[-1.2, 1.2, 1],
                      x_length=3.2, y_length=0.6, tips=False,
                      axis_config={"include_ticks": False, "stroke_width": 0.8,
                                   "color": C_PURPLE})
        sin_curve = ax_pos.plot(lambda t: 0.9 * np.sin(t * 0.16),
                                color=C_PURPLE, stroke_width=2.4)
        pos_ribbon = VGroup(ax_pos, sin_curve).move_to(T_proj.get_center() + UP * 1.0)
        plus_sign = MathTex("+", font_size=44, color=C_PURPLE)\
            .next_to(pos_ribbon, RIGHT, buff=0.15)

        self.play(Create(pos_ribbon), Write(plus_sign), run_time=1.0)
        # Animate the ribbon descending into the sequence (merge by addition).
        self.play(pos_ribbon.animate.move_to(T_proj.get_center())
                  .set_opacity(0.0),
                  T_proj[0].animate.set_color(C_PURPLE).set_fill(C_PURPLE, opacity=0.6),
                  FadeOut(plus_sign),
                  run_time=1.1)
        self.wait(0.3)

        # Bundle current sequence (cls + projected + pos-added) for later moves.
        seq_block = VGroup(T_proj, cls_token)
        self.play(FadeOut(VGroup(proj_pill, arrow1, T_in, pos_caption)),
                  seq_block.animate.move_to(LEFT * 4.8 + DOWN * 0.5),
                  run_time=0.8)

        # ───────────── PHASE 5 — one encoder layer in detail ─────────────
        layer_title = Text("× 4 Pre-LN Encoder Layers",
                           font_size=26, color=C_PURPLE, weight=BOLD)
        layer_title.next_to(head, DOWN, buff=0.15)
        self.play(FadeIn(layer_title, shift=DOWN * 0.2), run_time=0.5)

        # Sub-blocks across the middle/right of the screen.
        ln1 = self._pill("LayerNorm", "", C_PURPLE, w=2.0, h=0.7)
        mha = self._pill("Multi-Head Attention",
                         "8 heads · d_k=16", C_PURPLE, w=3.0, h=1.1)
        plus_a = MathTex(r"\oplus", font_size=42, color=C_HI)

        ln2 = self._pill("LayerNorm", "", C_PURPLE, w=2.0, h=0.7)
        ffn = self._pill("Feed-Forward",
                         "Linear 128→512  GELU  Linear 512→128",
                         C_PURPLE, w=4.2, h=1.1)
        plus_b = MathTex(r"\oplus", font_size=42, color=C_HI)

        row = VGroup(ln1, mha, plus_a, ln2, ffn, plus_b)\
            .arrange(RIGHT, buff=0.25).next_to(seq_block, RIGHT, buff=0.5)
        row.shift(DOWN * 0.15)

        self.play(LaggedStart(
            FadeIn(ln1, shift=RIGHT * 0.2),
            FadeIn(mha, shift=RIGHT * 0.2),
            FadeIn(plus_a),
            FadeIn(ln2, shift=RIGHT * 0.2),
            FadeIn(ffn, shift=RIGHT * 0.2),
            FadeIn(plus_b),
            lag_ratio=0.18,
        ), run_time=1.8)

        # Residual loops (curve above MHA and FFN).
        res_a = ArcBetweenPoints(
            ln1.get_top() + UP * 0.08, plus_a.get_top() + UP * 0.08,
            angle=-PI / 2.3, color=C_HI, stroke_width=2.5,
        ).add_tip(tip_length=0.15)
        res_b = ArcBetweenPoints(
            ln2.get_top() + UP * 0.08, plus_b.get_top() + UP * 0.08,
            angle=-PI / 2.3, color=C_HI, stroke_width=2.5,
        ).add_tip(tip_length=0.15)
        res_lbl_a = Text("residual", font_size=14, color=C_HI)\
            .next_to(res_a, UP, buff=0.05)
        res_lbl_b = Text("residual", font_size=14, color=C_HI)\
            .next_to(res_b, UP, buff=0.05)
        self.play(Create(res_a), Create(res_b),
                  FadeIn(res_lbl_a), FadeIn(res_lbl_b),
                  run_time=1.0)
        self.wait(0.3)

        # Pop-out: 8 attention heads inside MHA.
        heads = VGroup()
        for k in range(8):
            cell = Square(side_length=0.30, color=C_PURPLE, fill_opacity=0.7)
            heads.add(cell)
        heads.arrange_in_grid(rows=2, cols=4, buff=0.06).move_to(mha.get_center())
        head_lbl = Text("8 heads", font_size=14, color=WHITE).next_to(heads, DOWN, buff=0.04)

        self.play(FadeOut(mha[1]),  # hide the inner text of mha pill
                  FadeIn(heads, scale=0.7),
                  FadeIn(head_lbl), run_time=0.8)
        # Pulse heads in sequence.
        self.play(LaggedStart(*[
            Flash(h, color=C_HI, flash_radius=0.18, line_length=0.08)
            for h in heads
        ], lag_ratio=0.06), run_time=1.2)

        # Stacking ×4.
        stack_lbl = MathTex(r"\times 4", font_size=44, color=C_HI)\
            .next_to(row, DOWN, buff=0.2)
        self.play(Write(stack_lbl), run_time=0.6)

        # Echo plates behind the row to suggest 4 stacked layers.
        echos = VGroup()
        for i in range(1, 4):
            r = SurroundingRectangle(row, buff=0.18 + 0.07 * i,
                                     color=C_PURPLE, stroke_width=1.2,
                                     stroke_opacity=0.5 - 0.12 * i)
            r.shift(RIGHT * 0.12 * i + UP * 0.07 * i)
            echos.add(r)
        self.play(LaggedStart(*[Create(e) for e in echos], lag_ratio=0.2),
                  run_time=1.2)
        self.wait(0.6)

        # ───────────── PHASE 6 — pool [CLS] ─────────────
        # Collapse the row + echoes into a small icon, then pluck CLS.
        layer_icon = VGroup(row, res_a, res_b, res_lbl_a, res_lbl_b,
                            heads, head_lbl, stack_lbl, echos)
        self.play(layer_icon.animate.scale(0.35).move_to(ORIGIN + DOWN * 0.3)
                  .set_opacity(0.45),
                  run_time=0.8)

        pool_caption = Text("Pool the [CLS] token",
                            font_size=22, color=C_HI).move_to(UP * 0.9)
        self.play(FadeIn(pool_caption, shift=DOWN * 0.2), run_time=0.5)

        cls_vec = Rectangle(height=0.45, width=3.0, color=C_HI,
                            fill_opacity=0.85, stroke_width=2)
        cls_vec.move_to(RIGHT * 2.3 + DOWN * 0.3)
        cls_vec_lbl = MathTex(r"(B,\, 128)", font_size=26, color=C_HI)\
            .next_to(cls_vec, UP, buff=0.15)

        self.play(ReplacementTransform(cls_token.copy(), cls_vec),
                  FadeIn(cls_vec_lbl), run_time=1.1)
        self.wait(0.3)

        # ───────────── PHASE 7 — classifier head ─────────────
        head_caption = Text("Classifier head",
                            font_size=22, color=C_HI).move_to(pool_caption.get_center())
        self.play(ReplacementTransform(pool_caption, head_caption), run_time=0.4)

        # Funnel: 128 → 64 → 1
        funnel = VGroup(
            self._pill("LN  +  Linear  128→64",
                       "GELU  ·  Dropout 0.2", C_PURPLE, w=3.4, h=0.95),
            self._pill("Linear  64→1", "", C_PURPLE, w=2.4, h=0.7),
            self._pill("σ  (sigmoid)", "", C_HI, w=1.8, h=0.7),
        ).arrange(DOWN, buff=0.25).next_to(cls_vec, DOWN, buff=0.45)

        for pill in funnel:
            self.play(FadeIn(pill, shift=DOWN * 0.15), run_time=0.5)
        self.wait(0.3)

        # ───────────── PHASE 8 — output ŷ ─────────────
        yhat_dot = Dot(radius=0.18, color=C_HI)
        yhat_lbl = MathTex(r"\hat y \in [0,1]",
                           font_size=34, color=C_HI)
        out_group = VGroup(yhat_dot, yhat_lbl).arrange(RIGHT, buff=0.3)
        out_group.next_to(funnel, DOWN, buff=0.35)
        self.play(GrowFromCenter(yhat_dot), Write(yhat_lbl), run_time=0.8)
        self.play(Flash(yhat_dot, color=C_HI, flash_radius=0.5))

        shape5 = MathTex(r"\hat y\,:\;(B,\, 1)",
                         font_size=34, color=C_PASS).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(shape, shape5), run_time=0.6)
        shape = shape5

        # ───────────── PHASE 9 — param count + ranking outcome ─────────────
        params = Text("829 K parameters in total",
                      font_size=22, color=C_PASS).next_to(out_group, DOWN, buff=0.25)
        self.play(FadeIn(params, shift=UP * 0.2), run_time=0.7)
        self.wait(2.4)

        # Final wipe.
        self.play(
            *[FadeOut(m) for m in self.mobjects if m is not None],
            run_time=0.9,
        )

    # ─────────────────── helpers ───────────────────
    def _pill(self, top, bottom, color, w=2.6, h=0.85):
        """Rounded pill with optional two lines of text."""
        body = RoundedRectangle(corner_radius=0.12, height=h, width=w,
                                color=color, fill_opacity=0.18,
                                stroke_width=2)
        if bottom:
            t1 = Text(top, font_size=18, color=color, weight=BOLD)
            t2 = Text(bottom, font_size=14, color=C_ROAD)
            txt = VGroup(t1, t2).arrange(DOWN, buff=0.05)
        else:
            txt = Text(top, font_size=18, color=color, weight=BOLD)
        txt.move_to(body.get_center())
        return VGroup(body, txt)


# ============================================================================
# SCENE 7 — SWA on the loss landscape
# ============================================================================
class SWAVisualization(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text("Why Stochastic Weight Averaging (SWA) wins",
                     font_size=34, color=C_ROAD).to_edge(UP)
        self.play(Write(title))

        # 1-D loss landscape: wiggly valley.
        ax = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 2.2, 0.5],
            x_length=10,
            y_length=4.0,
            tips=False,
            axis_config={"include_numbers": False, "color": C_ROAD},
        ).shift(DOWN * 0.5)

        loss = ax.plot(
            lambda x: 0.4 + 0.6 * np.cos(2.2 * x) * np.exp(-0.18 * x * x)
                      + 0.05 * x * x,
            color=C_BLUE,
            stroke_width=4,
        )
        loss_label = Text("loss landscape", font_size=22, color=C_BLUE)\
            .next_to(ax, UP, buff=0.1)

        self.play(Create(ax), Create(loss), FadeIn(loss_label), run_time=1.6)
        self.wait(0.3)

        # Last-K SGD checkpoints bouncing around a flat region.
        rng = np.random.default_rng(11)
        # Pick points around the flat minimum near x=-1.7.
        xs = -1.7 + rng.uniform(-0.45, 0.45, size=8)
        ys = [0.4 + 0.6 * np.cos(2.2 * x) * np.exp(-0.18 * x * x)
              + 0.05 * x * x for x in xs]
        ckpts = VGroup(*[
            Dot(ax.c2p(x, y), color=C_FAIL, radius=0.07)
            for x, y in zip(xs, ys)
        ])

        ckpt_lab = Text("epoch 50–75 SGD snapshots",
                        font_size=22, color=C_FAIL).to_edge(LEFT, buff=0.7).shift(UP * 1.0)

        self.play(LaggedStart(*[GrowFromCenter(d) for d in ckpts],
                              lag_ratio=0.15),
                  FadeIn(ckpt_lab),
                  run_time=1.8)
        self.wait(0.4)

        # Average them -> SWA point at mean(x).
        swa_x = float(np.mean(xs))
        swa_y = 0.4 + 0.6 * np.cos(2.2 * swa_x) * np.exp(-0.18 * swa_x * swa_x) \
                + 0.05 * swa_x * swa_x
        swa_dot = Dot(ax.c2p(swa_x, swa_y), color=C_HI, radius=0.13)
        swa_label = Text("SWA average  (flatter, wider optimum)",
                         font_size=24, color=C_HI).next_to(swa_dot, UP, buff=0.35)

        self.play(
            *[ReplacementTransform(c.copy(), swa_dot) for c in ckpts],
            run_time=1.6,
        )
        self.play(Write(swa_label), run_time=1.0)
        self.play(Flash(swa_dot, color=C_HI, flash_radius=0.4))
        self.wait(0.4)

        # Headline numbers.
        gain = Text("APFD  0.788  →  0.804     (+0.016, only SWA improved)",
                    font_size=26, color=C_PASS).to_edge(DOWN, buff=0.5)
        self.play(Write(gain), run_time=1.6)
        self.wait(2.2)

        self.play(FadeOut(VGroup(title, ax, loss, loss_label, ckpts, ckpt_lab,
                                 swa_dot, swa_label, gain)), run_time=0.8)


# ============================================================================
# SCENE 8 — Results bar chart
# ============================================================================
class Results(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text("RoadFury vs. four other paradigms (956 tests, 30 trials)",
                     font_size=30, color=C_ROAD).to_edge(UP)
        self.play(Write(title))

        methods = [
            ("Random",            0.493, C_ROAD),
            ("LLM zero-shot",     0.487, C_PURPLE),
            ("GNN (road graph)",  0.533, C_TEAL),
            ("ResNet-50 visual",  0.572, C_BLUE),
            ("ITEP4SDC (prior)",  0.781, C_HI),
            ("RoadFury (ours)",   0.804, C_PASS),
        ]

        chart = BarChart(
            values=[m[1] for m in methods],
            bar_names=["Random", "LLM", "GNN", "CNN", "ITEP4SDC", "RoadFury"],
            y_range=[0, 1.0, 0.2],
            y_length=4.5,
            x_length=11.0,
            bar_colors=[m[2] for m in methods],
            bar_fill_opacity=0.9,
        ).shift(DOWN * 0.3)
        chart.y_axis.set_color(C_ROAD)
        chart.x_axis.set_color(C_ROAD)

        self.play(Create(chart.axes), run_time=0.9)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in chart.bars],
                              lag_ratio=0.18),
                  run_time=2.4)

        # Value labels above each bar.
        value_labels = VGroup()
        for bar, (_, v, _) in zip(chart.bars, methods):
            lbl = Text(f"{v:.3f}", font_size=22, color=C_ROAD)
            lbl.next_to(bar, UP, buff=0.08)
            value_labels.add(lbl)
        self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.15) for l in value_labels],
                              lag_ratio=0.12),
                  run_time=1.4)

        # Highlight RoadFury.
        rf_bar = chart.bars[-1]
        rf_box = SurroundingRectangle(rf_bar, color=C_HI, buff=0.05,
                                      corner_radius=0.05)
        self.play(Create(rf_box), run_time=0.7)
        self.play(Flash(value_labels[-1], color=C_HI, flash_radius=0.5))
        self.wait(2.0)

        self.play(FadeOut(VGroup(title, chart, value_labels, rf_box)),
                  run_time=0.8)


# ============================================================================
# SCENE 9 — Conclusion
# ============================================================================
class Conclusion(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        l1 = Text("Sequence  >  average.", font_size=72, color=C_HI, weight=BOLD)
        l2 = Text("Preserve the road; let attention find the danger.",
                  font_size=32, color=C_ROAD)
        l3 = Text("APFD  =  0.804  ±  0.012     (state of the art)",
                  font_size=32, color=C_PASS)
        l4 = Text("github.com/chisngyen/sdc-test-prioritization-novel",
                  font="Monospace", font_size=22, color=C_BLUE)

        group = VGroup(l1, l2, l3, l4).arrange(DOWN, buff=0.55).move_to(ORIGIN)

        self.play(FadeIn(l1, shift=DOWN * 0.3), run_time=1.0)
        self.play(Write(l2), run_time=1.4)
        self.play(Write(l3), run_time=1.4)
        self.play(FadeIn(l4, shift=UP * 0.2), run_time=0.9)
        self.wait(3.0)

        self.play(FadeOut(group), run_time=1.0)


# ============================================================================
# A single "playlist" scene that chains everything (optional).
# ============================================================================
class FullStory(Scene):
    """Concatenate all scenes in one render. Use -qh; expect ~4 min runtime."""

    def construct(self):
        for SceneCls in [
            Title, Problem, APFDExplained, AggregationLoss,
            FeatureExtraction, TransformerView, SWAVisualization,
            Results, Conclusion,
        ]:
            sub = SceneCls()
            sub.camera = self.camera
            sub.renderer = self.renderer
            sub.construct()
