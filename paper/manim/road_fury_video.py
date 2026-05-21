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

        # Road that draws itself across the screen, then slides to the lower
        # third so the title block sits in clear space above it.
        road = ParametricFunction(
            lambda t: road_curve(t, "curvy"),
            t_range=[0, 1],
            color=C_ROAD,
            stroke_width=6,
        )
        road_glow = road.copy().set_stroke(color=C_BLUE, width=14, opacity=0.25)

        self.play(Create(road_glow, run_time=2.2), Create(road, run_time=2.2))
        self.wait(0.3)
        self.play(
            road.animate.scale(0.85).shift(DOWN * 2.1),
            road_glow.animate.scale(0.85).shift(DOWN * 2.1),
            run_time=0.8,
        )

        title = Text("RoadFury", font_size=84, weight=BOLD, color=C_HI)
        subtitle = Text(
            "Teaching a Transformer to read roads",
            font_size=30,
            color=C_ROAD,
        )
        venue = Text(
            "ICST 2026  ·  SDC Testing Competition",
            font_size=22,
            color=C_BLUE,
        )

        VGroup(title, subtitle, venue).arrange(DOWN, buff=0.35).move_to(UP * 1.4)

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

        # Raw road -> 197 sampled points. Sized + shifted left so the 10-channel
        # box on the right has clear horizontal room.
        road = ParametricFunction(
            lambda t: road_curve(t, "curvy"),
            t_range=[0, 1], color=C_BLUE, stroke_width=5,
        ).scale(0.7).shift(UP * 0.6 + LEFT * 2.0)
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

        feat_names = ["f0 len", "f1 |Δθ|", "f2 κ", "f3 Δκ", "f4 d/D",
                      "f5 sinθ", "f6 cosθ", "f7 pos", "f8 σκ", "f9 Δ²κ"]
        feat_lbls = VGroup()
        for i, name in enumerate(feat_names):
            y = T_in[0].get_top()[1] - (i + 0.5) * (3.0 / 10)
            t = Text(name, font_size=12, color=C_BLUE, font="Consolas")
            t.move_to([T_in[0].get_left()[0] - 0.7, y, 0])
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

        # Sub-blocks compacted so the whole row fits to the right of seq_block.
        ln1 = self._pill("LN", "", C_PURPLE, w=0.9, h=0.55)
        mha = self._pill("MHA", "8 heads · d_k=16", C_PURPLE, w=2.0, h=0.95)
        plus_a = MathTex(r"\oplus", font_size=32, color=C_HI)

        ln2 = self._pill("LN", "", C_PURPLE, w=0.9, h=0.55)
        ffn = self._pill("FFN", "128→512→128", C_PURPLE, w=2.0, h=0.95)
        plus_b = MathTex(r"\oplus", font_size=32, color=C_HI)

        row = VGroup(ln1, mha, plus_a, ln2, ffn, plus_b)\
            .arrange(RIGHT, buff=0.15)
        row.next_to(seq_block, RIGHT, buff=0.35).shift(DOWN * 0.2)

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
        # Clear the encoder visualisation entirely so the head has room.
        layer_icon = VGroup(row, res_a, res_b, res_lbl_a, res_lbl_b,
                            heads, head_lbl, stack_lbl, echos)
        self.play(FadeOut(layer_icon), FadeOut(layer_title), run_time=0.6)

        pool_caption = Text("Pool the [CLS] token",
                            font_size=22, color=C_HI).move_to(UP * 2.0)
        self.play(FadeIn(pool_caption, shift=DOWN * 0.2), run_time=0.5)

        cls_vec = Rectangle(height=0.45, width=2.2, color=C_HI,
                            fill_opacity=0.85, stroke_width=2)
        cls_vec.move_to(LEFT * 4.6 + UP * 0.8)
        cls_vec_lbl = MathTex(r"(B,\, 128)", font_size=24, color=C_HI)\
            .next_to(cls_vec, UP, buff=0.12)

        self.play(ReplacementTransform(cls_token.copy(), cls_vec),
                  FadeOut(seq_block),
                  FadeIn(cls_vec_lbl), run_time=1.1)
        self.wait(0.3)

        # ───────────── PHASE 7 — classifier head (horizontal flow) ─────────────
        head_caption = Text("Classifier head",
                            font_size=22, color=C_HI).move_to(pool_caption.get_center())
        self.play(ReplacementTransform(pool_caption, head_caption), run_time=0.4)

        funnel = VGroup(
            self._pill("LN + Linear", "128→64 · GELU · Drop", C_PURPLE,
                       w=2.6, h=0.85),
            self._pill("Linear", "64→1", C_PURPLE, w=1.5, h=0.7),
            self._pill("σ", "sigmoid", C_HI, w=1.0, h=0.7),
        ).arrange(RIGHT, buff=0.3)
        funnel.next_to(cls_vec, RIGHT, buff=0.55).align_to(cls_vec, UP).shift(DOWN * 0.2)

        arrow_to_funnel = Arrow(cls_vec.get_right(), funnel.get_left(),
                                color=C_PURPLE, stroke_width=3, buff=0.08,
                                max_tip_length_to_length_ratio=0.20)
        self.play(GrowArrow(arrow_to_funnel), run_time=0.4)
        for pill in funnel:
            self.play(FadeIn(pill, shift=RIGHT * 0.15), run_time=0.4)
        self.wait(0.3)

        # ───────────── PHASE 8 — output ŷ ─────────────
        yhat_dot = Dot(radius=0.18, color=C_HI)
        yhat_lbl = MathTex(r"\hat y \in [0,1]",
                           font_size=30, color=C_HI)
        out_group = VGroup(yhat_dot, yhat_lbl).arrange(RIGHT, buff=0.25)
        out_group.next_to(funnel, RIGHT, buff=0.55)
        arrow_out = Arrow(funnel.get_right(), out_group.get_left(),
                          color=C_HI, stroke_width=3, buff=0.08,
                          max_tip_length_to_length_ratio=0.25)
        self.play(GrowArrow(arrow_out), run_time=0.3)
        self.play(GrowFromCenter(yhat_dot), Write(yhat_lbl), run_time=0.8)
        self.play(Flash(yhat_dot, color=C_HI, flash_radius=0.5))

        shape5 = MathTex(r"\hat y\,:\;(B,\, 1)",
                         font_size=30, color=C_PASS).to_edge(DOWN, buff=0.4)
        self.play(ReplacementTransform(shape, shape5), run_time=0.6)
        shape = shape5

        # ───────────── PHASE 9 — param count ─────────────
        params = Text("829 K parameters in total",
                      font_size=22, color=C_PASS).move_to(DOWN * 1.8)
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
# ARCHITECTURE DEEP DIVE — six scenes that walk through each layer pedagogically
#
# Mirrors exps/exp00_Basline.py::RoadTransformer step by step:
#   ArchInputProj   →  ArchCLSPos      →  ArchAttention
#   ArchFFNRes      →  ArchStack       →  ArchPoolHead
#
# Each scene begins with the question it answers, walks the math, shows the
# tensor shape change, and ends with a one-sentence intuition. Render them in
# order to teach the full forward pass.
# ============================================================================


def _pill(top, bottom, color, w=2.6, h=0.85, fs_top=18, fs_bot=14):
    """Module-level rounded pill (used by the deep-dive scenes)."""
    body = RoundedRectangle(corner_radius=0.12, height=h, width=w,
                            color=color, fill_opacity=0.18, stroke_width=2)
    if bottom:
        t1 = Text(top,    font_size=fs_top, color=color, weight=BOLD)
        t2 = Text(bottom, font_size=fs_bot, color=C_ROAD)
        txt = VGroup(t1, t2).arrange(DOWN, buff=0.05)
    else:
        txt = Text(top, font_size=fs_top, color=color, weight=BOLD)
    txt.move_to(body.get_center())
    return VGroup(body, txt)


def _teach(lines, color=C_ROAD, size=20, align=LEFT, buff=0.12):
    """Multi-line teaching caption. Empty strings act as paragraph breaks.

    Empty Text("") collapses to a zero-bbox mobject and breaks
    VGroup.arrange(DOWN, aligned_edge=LEFT) -- subsequent lines end up
    stacked on top of earlier ones. Use an invisible single-glyph spacer
    so every entry has a real line-height for arrange() to work with.
    """
    mobjs = []
    for l in lines:
        if l == "":
            mobjs.append(Text("M", font_size=size, color=color).set_opacity(0))
        else:
            mobjs.append(Text(l, font_size=size, color=color))
    return VGroup(*mobjs).arrange(DOWN, aligned_edge=align, buff=buff)


def _section_header(scene, n_of_total, title_text):
    """Common header for deep-dive scenes."""
    badge = Text(n_of_total, font_size=20, color=C_BG, weight=BOLD)
    badge_box = RoundedRectangle(corner_radius=0.08, width=1.6, height=0.45,
                                 color=C_HI, fill_opacity=1.0, stroke_width=0)
    badge.move_to(badge_box)
    title = Text(title_text, font_size=30, color=C_HI, weight=BOLD)
    head = VGroup(VGroup(badge_box, badge), title)\
        .arrange(RIGHT, buff=0.3).to_edge(UP, buff=0.3)
    scene.play(FadeIn(head[0], shift=DOWN * 0.1), Write(head[1]), run_time=0.9)
    return head


# ----------------------------------------------------------------------------
# DEEP DIVE 1 / 6  —  Input tensor + linear projection 10 → 128
# ----------------------------------------------------------------------------
class ArchInputProj(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        head = _section_header(self, "Step 1/6",
                               "Each road point becomes a 128-dim embedding")

        # ===== Phase A : one road point -> 10 numbers =====
        road = ParametricFunction(
            lambda t: road_curve(t, "curvy") * 0.45 + LEFT * 4.3 + UP * 1.6,
            t_range=[0, 1], color=C_BLUE, stroke_width=4,
        )
        self.play(Create(road), run_time=1.0)

        focus_t = 0.62
        focus_pt = road.point_from_proportion(focus_t)
        dot = Dot(focus_pt, color=C_HI, radius=0.10)
        self.play(GrowFromCenter(dot))
        self.play(Flash(dot, color=C_HI, flash_radius=0.3))

        # 10-feature column: compact cells + concise labels so it fits in frame.
        feats = [("f0", "segment length"),
                 ("f1", "|dtheta|  angle change"),
                 ("f2", "kappa  curvature"),
                 ("f3", "d-kappa  jerk"),
                 ("f4", "d/D  cum. distance"),
                 ("f5", "sin theta"),
                 ("f6", "cos theta"),
                 ("f7", "i/L  position"),
                 ("f8", "std kappa  local"),
                 ("f9", "d2-kappa  accel.")]
        col = VGroup()
        for fid, name in feats:
            cell = Rectangle(width=0.45, height=0.24, color=C_BLUE,
                             fill_opacity=0.7, stroke_width=1.2)
            num = Text(fid, font_size=11, color=WHITE, weight=BOLD).move_to(cell)
            lbl = Text(name, font_size=12, color=C_ROAD,
                       font="Consolas").next_to(cell, RIGHT, buff=0.10)
            col.add(VGroup(cell, num, lbl))
        col.arrange(DOWN, aligned_edge=LEFT, buff=0.03)
        # Place column to the right of the road; keep it vertically centered.
        col.move_to(RIGHT * 1.2 + DOWN * 0.4)

        arr0 = Arrow(dot.get_right() + RIGHT * 0.05, col.get_left() + LEFT * 0.05,
                     color=C_HI, stroke_width=3.0, buff=0.05,
                     max_tip_length_to_length_ratio=0.06)
        arr_lbl = Text("10 numbers per point",
                       font_size=18, color=C_HI).next_to(arr0, UP, buff=0.10)
        self.play(GrowArrow(arr0), FadeIn(arr_lbl), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in col],
                              lag_ratio=0.06), run_time=1.6)
        self.wait(0.8)

        intuit1 = _teach([
            "197 road points  x  10 features each",
            "= one (197, 10) matrix per test case.",
        ], color=C_ROAD, size=20).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(intuit1, shift=UP * 0.2), run_time=0.8)
        self.wait(0.6)

        # ===== Phase B : project 10 -> 128 =====
        # Clear phase-A artwork. Mat (197,10) re-appears top-left.
        mat = tensor_block(rows=197, cols=10, h=0.50, w=2.6, color=C_BLUE,
                           grid_rows=2, grid_cols=10)
        mat_lbl = MathTex(r"(197,\, 10)", font_size=24, color=C_BLUE)
        mat_group = VGroup(mat, mat_lbl)
        mat_lbl.next_to(mat, DOWN, buff=0.15)
        mat_group.move_to(LEFT * 4.2 + UP * 1.3)

        self.play(
            FadeOut(VGroup(road, dot, arr0, arr_lbl, intuit1)),
            TransformFromCopy(col, mat),
            FadeIn(mat_lbl, shift=UP * 0.1),
            FadeOut(col),
            run_time=1.0,
        )

        # Right side: question + answer (compact).
        q = Text("Why widen 10 -> 128?", font_size=22, color=C_HI, weight=BOLD)
        ans = _teach([
            "10 raw numbers are too narrow to mix",
            "curvature, heading, and jerk together.",
            "",
            "128 'opinion channels' give the model",
            "room to combine features into many",
            "richer signals per token.",
        ], size=16, color=C_ROAD, buff=0.08)
        right_panel = VGroup(q, ans).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        right_panel.move_to(RIGHT * 2.0 + UP * 1.0)
        self.play(Write(q), run_time=0.6)
        self.play(FadeIn(ans, shift=UP * 0.15), run_time=1.4)
        self.wait(0.5)

        # Centered formula block, well clear of mat and right_panel.
        eq = MathTex(
            r"x_{\text{emb}}",
            r"\,=\,",
            r"\text{GELU}\!\big(\text{LN}(",
            r"x",
            r"\,W",
            r"+ b",
            r")\big)",
            font_size=30,
        )
        eq[0].set_color(C_TEAL)
        eq[3].set_color(C_BLUE)
        eq[4].set_color(C_HI)

        shape_eq = MathTex(
            r"(197,\,10)\,\times\,(10,\,128)\,\to\,(197,\,128)",
            font_size=24, color=C_ROAD,
        )
        formula = VGroup(eq, shape_eq).arrange(DOWN, buff=0.18)
        formula.move_to(LEFT * 0.4 + DOWN * 1.1)
        self.play(Write(eq), run_time=1.4)
        self.play(Write(shape_eq), run_time=1.0)
        self.wait(0.4)

        # Output tensor mat2 (197, 128) at bottom-left, vertical arrow from mat.
        mat2 = tensor_block(rows=197, cols=128, h=0.50, w=3.6, color=C_TEAL,
                            grid_rows=2, grid_cols=22)
        mat2_lbl = MathTex(r"(197,\, 128)", font_size=24, color=C_TEAL)
        mat2_lbl.next_to(mat2, DOWN, buff=0.15)
        mat2.move_to(LEFT * 4.0 + DOWN * 1.2)
        mat2_lbl.next_to(mat2, DOWN, buff=0.15)
        arr1 = Arrow(mat.get_bottom() + DOWN * 0.05,
                     mat2.get_top() + UP * 0.05,
                     color=C_TEAL, stroke_width=3, buff=0.05,
                     max_tip_length_to_length_ratio=0.20)
        self.play(GrowArrow(arr1), run_time=0.4)
        self.play(TransformFromCopy(mat, mat2),
                  FadeIn(mat2_lbl, shift=UP * 0.15),
                  run_time=1.2)
        self.wait(0.4)

        intuit2 = Text("Each road point now has a 128-dim 'feature personality'.",
                       font_size=20, color=C_PASS, weight=BOLD).to_edge(DOWN, buff=0.25)
        self.play(Write(intuit2), run_time=1.4)
        self.wait(1.6)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ----------------------------------------------------------------------------
# DEEP DIVE 2 / 6  —  [CLS] token + positional embeddings
# ----------------------------------------------------------------------------
class ArchCLSPos(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        head = _section_header(self, "Step 2/6",
                               "Why we need [CLS] and positional embeddings")

        # ===== Problem: Transformer is permutation-invariant =====
        q = Text("A pure Transformer doesn't see order.",
                 font_size=26, color=C_HI, weight=BOLD).next_to(head, DOWN, buff=0.5)
        self.play(Write(q), run_time=0.9)

        n = 7
        toks_a = VGroup(*[Square(side_length=0.55, color=C_TEAL, fill_opacity=0.7)
                          for _ in range(n)]).arrange(RIGHT, buff=0.12)
        toks_a.shift(UP * 0.4)
        hues = [C_BLUE, C_TEAL, C_PURPLE, C_HI, C_PASS, C_FAIL, C_ROAD]
        for i, c in enumerate(hues):
            toks_a[i].set_color(c).set_fill(c, opacity=0.75)
            num = Text(str(i + 1), font_size=18, color=C_BG,
                       weight=BOLD).move_to(toks_a[i])
            toks_a[i] = VGroup(toks_a[i], num)
        toks_a = VGroup(*toks_a)

        lbl_a = Text("road points in order  ->  prediction p",
                     font_size=20, color=C_ROAD).next_to(toks_a, DOWN, buff=0.3)
        self.play(FadeIn(toks_a, shift=DOWN * 0.2), FadeIn(lbl_a), run_time=0.9)

        order = [3, 0, 6, 2, 4, 1, 5]
        positions = [t.get_center() for t in toks_a]
        shuf_anims = []
        for new_slot, src in enumerate(order):
            shuf_anims.append(toks_a[src].animate.move_to(positions[new_slot]))
        lbl_b = Text("shuffled  ->  same prediction p   (without positions!)",
                     font_size=20, color=C_FAIL).next_to(toks_a, DOWN, buff=0.3)
        self.play(*shuf_anims, ReplacementTransform(lbl_a, lbl_b), run_time=1.6)
        self.wait(0.8)
        self.play(FadeOut(VGroup(toks_a, lbl_b, q)), run_time=0.5)

        # ===== Solution 1: sinusoidal positional embeddings =====
        h1 = Text("Fix #1: inject position with sinusoids",
                  font_size=24, color=C_PURPLE, weight=BOLD)\
            .next_to(head, DOWN, buff=0.4)
        self.play(Write(h1), run_time=0.7)

        formula = MathTex(
            r"PE_{(\text{pos},\, 2i)} \,=\, \sin\!\big(\text{pos}/10000^{2i/d}\big)",
            r"\\",
            r"PE_{(\text{pos},\, 2i+1)} \,=\, \cos\!\big(\text{pos}/10000^{2i/d}\big)",
            font_size=26,
        ).next_to(h1, DOWN, buff=0.3)
        formula.set_color_by_gradient(C_PURPLE, C_HI)
        self.play(Write(formula), run_time=1.8)
        self.wait(0.3)

        # Sinusoid axes: keep total width (ax + labels) under ~10 so it stays
        # in-frame after centering.
        ax = Axes(x_range=[0, 12, 2], y_range=[-1.3, 1.3, 1],
                  x_length=5.6, y_length=1.5, tips=False,
                  axis_config={"stroke_width": 0.6,
                               "color": C_ROAD, "include_ticks": False})
        sines = VGroup(
            ax.plot(lambda t: np.sin(0.8 * t), color=C_PURPLE, stroke_width=2.4),
            ax.plot(lambda t: np.sin(0.35 * t), color=C_BLUE,   stroke_width=2.4),
            ax.plot(lambda t: np.sin(0.15 * t), color=C_TEAL,   stroke_width=2.4),
        )
        labels = VGroup(
            Text("dim 0  high freq", font_size=14, color=C_PURPLE),
            Text("dim 32 mid freq",  font_size=14, color=C_BLUE),
            Text("dim 96 low freq",  font_size=14, color=C_TEAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        sin_block = VGroup(VGroup(ax, sines), labels).arrange(RIGHT, buff=0.35)
        sin_block.next_to(formula, DOWN, buff=0.35)
        self.play(Create(ax), Create(sines), FadeIn(labels), run_time=2.0)

        intuit_pe = Text(
            "Every position gets a unique 128-d 'fingerprint' added to its embedding.",
            font_size=20, color=C_PASS).to_edge(DOWN, buff=0.4)
        self.play(Write(intuit_pe), run_time=1.4)
        self.wait(1.4)

        self.play(FadeOut(VGroup(h1, formula, sin_block, intuit_pe)),
                  run_time=0.6)

        # ===== Solution 2: [CLS] token =====
        h2 = Text("Fix #2: add a learnable [CLS] token",
                  font_size=24, color=C_HI, weight=BOLD)\
            .next_to(head, DOWN, buff=0.4)
        self.play(Write(h2), run_time=0.7)

        # Concise intuition text — kept narrow so the token-row sits below cleanly.
        intuit_cls = _teach([
            "We need ONE vector that summarises the whole road.",
            "",
            "Option A: average all 197 token vectors",
            "          (every token weighs the same).",
            "",
            "Option B: prepend a learnable token that ASKS",
            "          every other token what matters.",
            "          Attention turns it into the summariser.",
        ], size=16, color=C_ROAD, buff=0.10)
        intuit_cls.next_to(h2, DOWN, buff=0.25)
        intuit_cls.to_edge(LEFT, buff=0.50)
        if intuit_cls.get_right()[0] > -0.2:
            sf = (-0.2 - intuit_cls.get_left()[0]) / intuit_cls.get_width()
            intuit_cls.scale(sf).to_edge(LEFT, buff=0.50)
        self.play(FadeIn(intuit_cls, shift=UP * 0.15), run_time=1.6)

        # Token row: CLS + 7 content tokens, placed centred to the RIGHT of the
        # text panel so nothing overflows the frame.
        n = 7
        toks = VGroup(*[Square(side_length=0.46, color=C_TEAL, fill_opacity=0.55)
                        for _ in range(n)]).arrange(RIGHT, buff=0.10)
        cls = Square(side_length=0.46, color=C_HI, fill_opacity=0.95)
        cls_inner = Text("[CLS]", font_size=11, color=C_BG, weight=BOLD).move_to(cls)
        cls_grp = VGroup(cls, cls_inner)
        row = VGroup(cls_grp, toks).arrange(RIGHT, buff=0.20)
        row.move_to(RIGHT * 2.6 + DOWN * 0.5)

        self.play(FadeIn(row, shift=DOWN * 0.2), run_time=0.8)

        # Curved arrows from each content token UP-and-OVER into CLS so they
        # don't overlap into a single horizontal line.
        arrows_in = VGroup()
        for k, t in enumerate(toks):
            start = t.get_top() + UP * 0.05
            end = cls.get_top() + UP * 0.05
            # angle grows with distance so arcs don't all stack on the same line.
            angle = -PI / 6 - 0.10 * k
            a = CurvedArrow(start, end, angle=angle,
                            color=C_HI, stroke_width=1.6, tip_length=0.12)
            a.set_stroke(opacity=0.75)
            arrows_in.add(a)
        self.play(LaggedStart(*[Create(a) for a in arrows_in],
                              lag_ratio=0.08), run_time=1.4)
        self.play(Flash(cls, color=C_HI, flash_radius=0.4))
        self.wait(1.4)

        shape_final = MathTex(
            r"\text{sequence shape: }(197,\,128)\,\to\,(198,\,128)",
            font_size=26, color=C_PASS).to_edge(DOWN, buff=0.35)
        self.play(Write(shape_final), run_time=1.2)
        self.wait(1.6)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ----------------------------------------------------------------------------
# DEEP DIVE 3 / 6  —  Scaled dot-product attention, then multi-head
# ----------------------------------------------------------------------------
class ArchAttention(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        head = _section_header(self, "Step 3/6",
                               "Self-attention: each token decides what to listen to")

        # ===== Q, K, V =====
        q_txt = Text("Goal: weighted summary of all other tokens per token.",
                     font_size=21, color=C_ROAD).next_to(head, DOWN, buff=0.4)
        self.play(Write(q_txt), run_time=1.2)
        self.wait(0.3)

        legend = VGroup(
            VGroup(Square(0.28, color=C_FAIL, fill_opacity=0.9),
                   Text(" Q  query — what I look for",
                        font_size=19, color=C_FAIL)).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.28, color=C_BLUE, fill_opacity=0.9),
                   Text(" K  key   — what I offer",
                        font_size=19, color=C_BLUE)).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.28, color=C_PASS, fill_opacity=0.9),
                   Text(" V  value — my content",
                        font_size=19, color=C_PASS)).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(LEFT, buff=0.5).shift(DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in legend],
                              lag_ratio=0.2), run_time=1.4)

        # Compact formula block placed centrally to the right of legend.
        qkv_formula = MathTex(
            r"Q=xW_Q,\quad K=xW_K,\quad V=xW_V",
            font_size=28,
        ).set_color_by_gradient(C_FAIL, C_PASS)
        qkv_formula.move_to(RIGHT * 2.6 + UP * 0.3)
        intuit_qkv = _teach([
            "Three different linear projections of x.",
            "Q, K, V each have shape (198, 16)",
            "for one head with d_k = 16.",
        ], color=C_ROAD, size=17).next_to(qkv_formula, DOWN, buff=0.3)
        self.play(Write(qkv_formula), run_time=1.4)
        self.play(FadeIn(intuit_qkv, shift=UP * 0.1), run_time=1.0)
        self.wait(1.0)

        self.play(FadeOut(VGroup(q_txt, legend, qkv_formula, intuit_qkv)),
                  run_time=0.5)

        # ===== Scaled dot-product formula =====
        f1 = MathTex(
            r"A \;=\; \mathrm{softmax}\!\left(",
            r"\frac{Q\,K^{\top}}{\sqrt{d_k}}",
            r"\right)",
            font_size=44,
        )
        f1[0].set_color(C_ROAD); f1[1].set_color(C_HI); f1[2].set_color(C_ROAD)
        f1.next_to(head, DOWN, buff=0.5)
        self.play(Write(f1), run_time=1.6)

        why_scale = _teach([
            "Why sqrt(dk)?  Q·K dot products grow with dk.",
            "Without scaling, softmax saturates -> gradients die.",
            "Dividing by sqrt(dk) keeps variance near 1.",
        ], size=18, color=C_ROAD).next_to(f1, DOWN, buff=0.4)
        self.play(FadeIn(why_scale, shift=UP * 0.15), run_time=1.2)
        self.wait(1.0)
        self.play(FadeOut(VGroup(f1, why_scale)), run_time=0.5)

        # ===== Attention heat-map (8×8 toy) =====
        L = 8
        rng = np.random.default_rng(13)
        scores = np.abs(rng.standard_normal((L, L))) + 0.1
        scores = scores / scores.sum(axis=1, keepdims=True)

        cell_sz = 0.48
        heatmap = VGroup()
        for i in range(L):
            for j in range(L):
                v = float(scores[i, j])
                sq = Square(side_length=cell_sz,
                            color=C_HI, fill_opacity=min(0.95, 0.15 + 2.5 * v),
                            stroke_width=0.4, stroke_color=BLACK)
                sq.move_to(np.array([j * cell_sz, -i * cell_sz, 0]))
                heatmap.add(sq)
        # Shift heatmap to the LEFT third so caption fits on the right.
        heatmap.move_to(LEFT * 2.5 + DOWN * 0.2)

        x_lbl = Text("key (token j)", font_size=17, color=C_BLUE)\
            .next_to(heatmap, UP, buff=0.18)
        y_lbl = Text("query (token i)", font_size=17, color=C_FAIL)\
            .next_to(heatmap, LEFT, buff=0.18).rotate(PI / 2)
        self.play(FadeIn(heatmap), FadeIn(x_lbl), FadeIn(y_lbl), run_time=1.0)

        row_caption = _teach([
            "Row 3 = attention weights",
            "of query-token 3 over all keys.",
            "",
            "These 8 numbers sum to 1.",
            "Bigger square = louder voice.",
        ], size=17, color=C_HI)
        row_caption.next_to(heatmap, RIGHT, buff=0.6)
        self.play(FadeIn(row_caption, shift=RIGHT * 0.15), run_time=1.2)

        row3 = VGroup(*[heatmap[3 * L + j] for j in range(L)])
        rect3 = SurroundingRectangle(row3, color=C_HI, buff=0.04,
                                     stroke_width=3, corner_radius=0.04)
        self.play(Create(rect3), run_time=0.6)
        self.play(Flash(rect3, color=C_HI, flash_radius=0.4))
        self.wait(1.2)

        out_formula = MathTex(
            r"\text{output}_i \;=\; \sum_{j} A_{ij}\,V_{j}",
            font_size=32, color=C_PASS,
        ).to_edge(DOWN, buff=0.5)
        gather_lbl = Text("(token i pulls a weighted blend of every V_j)",
                          font_size=17, color=C_ROAD).next_to(out_formula, UP, buff=0.15)
        self.play(Write(out_formula), run_time=1.2)
        self.play(FadeIn(gather_lbl, shift=DOWN * 0.1), run_time=0.7)
        self.wait(1.4)

        self.play(FadeOut(VGroup(heatmap, x_lbl, y_lbl, row_caption,
                                 rect3, out_formula, gather_lbl)),
                  run_time=0.5)

        # ===== Multi-head =====
        h_multi = Text("8 heads in parallel — that's Multi-Head Attention",
                       font_size=25, color=C_PURPLE, weight=BOLD)\
            .next_to(head, DOWN, buff=0.4)
        self.play(Write(h_multi), run_time=0.7)

        intuit_multi = _teach([
            "d_model=128 split into 8 heads x d_k=16.",
            "",
            "Each head learns a different relation",
            "(curvature, long-range, jerk spikes, ...).",
            "",
            "Concat all heads -> 128.  Final linear mixes.",
        ], size=17, color=C_ROAD)
        intuit_multi.next_to(h_multi, DOWN, buff=0.25)
        intuit_multi.to_edge(LEFT, buff=0.55)
        if intuit_multi.get_right()[0] > -0.2:
            sf = (-0.2 - intuit_multi.get_left()[0]) / intuit_multi.get_width()
            intuit_multi.scale(sf).to_edge(LEFT, buff=0.55)
        self.play(FadeIn(intuit_multi, shift=UP * 0.15), run_time=1.2)

        # 8 small heat-maps. Each is L2×L2 cells of 0.15.
        # 2 rows × 4 cols with buff=0.20. Total width ≈ 4*(5*0.15+0.20)=3.8.
        # Placed to the RIGHT of intuit_multi.
        L2 = 5
        mini_maps = VGroup()
        for hi in range(8):
            grid = VGroup()
            rngh = np.random.default_rng(20 + hi)
            sc = np.abs(rngh.standard_normal((L2, L2))) + 0.1
            sc = sc / sc.sum(axis=1, keepdims=True)
            for i in range(L2):
                for j in range(L2):
                    s = Square(side_length=0.15,
                               color=C_PURPLE,
                               fill_opacity=min(0.95, 0.15 + 3 * float(sc[i, j])),
                               stroke_width=0.2, stroke_color=BLACK)
                    s.move_to(np.array([j * 0.15, -i * 0.15, 0]))
                    grid.add(s)
            mini_maps.add(grid)
        mini_maps.arrange_in_grid(rows=2, cols=4, buff=0.22)
        mini_maps.next_to(intuit_multi, RIGHT, buff=0.5)
        head_label = Text("8 attention heads", font_size=17, color=C_PURPLE)\
            .next_to(mini_maps, UP, buff=0.12)

        self.play(FadeIn(head_label),
                  LaggedStart(*[FadeIn(g, scale=0.85) for g in mini_maps],
                              lag_ratio=0.10),
                  run_time=1.8)
        self.wait(1.2)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ----------------------------------------------------------------------------
# DEEP DIVE 4 / 6  —  FFN + residual + LayerNorm (Pre-LN)
# ----------------------------------------------------------------------------
class ArchFFNRes(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        head = _section_header(self, "Step 4/6",
                               "After attention: per-token FFN, residuals and Pre-LN")

        # ===== FFN funnel diagram =====
        h1 = Text("Feed-Forward Network:  128 -> 512 -> 128",
                  font_size=23, color=C_PURPLE, weight=BOLD)\
            .next_to(head, DOWN, buff=0.4)
        self.play(Write(h1), run_time=0.7)

        # Scale mid_block width down so the whole trio fits comfortably.
        in_block  = tensor_block(rows=1, cols=128, h=0.38, w=1.1, color=C_TEAL,
                                 grid_rows=1, grid_cols=8)
        mid_block = tensor_block(rows=1, cols=512, h=0.38, w=3.6, color=C_HI,
                                 grid_rows=1, grid_cols=20)
        out_block = tensor_block(rows=1, cols=128, h=0.38, w=1.1, color=C_TEAL,
                                 grid_rows=1, grid_cols=8)

        in_lbl  = MathTex(r"128", font_size=22, color=C_TEAL).next_to(in_block,  DOWN, buff=0.12)
        mid_lbl = MathTex(r"512", font_size=22, color=C_HI  ).next_to(mid_block, DOWN, buff=0.12)
        out_lbl = MathTex(r"128", font_size=22, color=C_TEAL).next_to(out_block, DOWN, buff=0.12)

        trio = VGroup(VGroup(in_block, in_lbl),
                      VGroup(mid_block, mid_lbl),
                      VGroup(out_block, out_lbl))\
            .arrange(RIGHT, buff=0.35).next_to(h1, DOWN, buff=0.35)

        op1 = MathTex(r"\xrightarrow{\;W_1,\,\text{GELU}\;}",
                      font_size=24, color=C_PURPLE)\
            .move_to(midpoint(in_block.get_right(), mid_block.get_left()))
        op2 = MathTex(r"\xrightarrow{\;W_2\;}",
                      font_size=24, color=C_PURPLE)\
            .move_to(midpoint(mid_block.get_right(), out_block.get_left()))

        self.play(FadeIn(in_block), FadeIn(in_lbl), run_time=0.5)
        self.play(Write(op1), FadeIn(mid_block), FadeIn(mid_lbl), run_time=0.8)
        self.play(Write(op2), FadeIn(out_block), FadeIn(out_lbl), run_time=0.7)

        why_ffn = _teach([
            "Each token processed independently.",
            "4x expansion (128->512) gives room",
            "for richer nonlinear mixing,",
            "then squeezed back to 128.",
        ], size=17, color=C_ROAD).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(why_ffn, shift=UP * 0.15), run_time=1.1)
        self.wait(1.2)

        # Shrink the FFN block to the LEFT so the Pre-LN diagram has room.
        ffn_group = VGroup(h1, trio, op1, op2, why_ffn)
        self.play(ffn_group.animate.scale(0.52).to_edge(LEFT, buff=0.35).shift(UP * 0.2),
                  run_time=0.8)

        # ===== Pre-LN wiring diagram =====
        h2 = Text("Pre-LN: normalise BEFORE each sub-layer",
                  font_size=21, color=C_HI, weight=BOLD)
        h2.to_edge(RIGHT, buff=0.4).shift(UP * 2.5)
        self.play(Write(h2), run_time=0.6)

        # Pipeline pills — sized compactly and positioned in the right half.
        x_node   = _pill("x",   "",            C_TEAL,   w=0.50, h=0.50, fs_top=16)
        ln1_node = _pill("LN",  "",            C_PURPLE, w=0.50, h=0.50, fs_top=13)
        mha_node = _pill("MHA", "8 heads",     C_PURPLE, w=1.10, h=0.80, fs_top=14, fs_bot=10)
        add1     = MathTex(r"\oplus", font_size=28, color=C_HI)
        ln2_node = _pill("LN",  "",            C_PURPLE, w=0.50, h=0.50, fs_top=13)
        ffn_node = _pill("FFN", "128->512->128", C_PURPLE, w=1.40, h=0.80, fs_top=14, fs_bot=10)
        add2     = MathTex(r"\oplus", font_size=28, color=C_HI)
        y_node   = _pill("y",   "",            C_TEAL,   w=0.50, h=0.50, fs_top=16)

        pipeline = VGroup(x_node, ln1_node, mha_node, add1,
                          ln2_node, ffn_node, add2, y_node)\
            .arrange(RIGHT, buff=0.10)
        pipeline.next_to(h2, DOWN, buff=0.35)
        # If pipeline overflows right edge, scale it to fit.
        max_w = 13.0  # safe frame width
        if pipeline.get_width() > max_w * 0.55:
            pipeline.scale(max_w * 0.55 / pipeline.get_width())
        pipeline.next_to(h2, DOWN, buff=0.35)

        self.play(LaggedStart(*[FadeIn(p, shift=RIGHT * 0.12) for p in pipeline],
                              lag_ratio=0.10), run_time=1.6)

        # Residual arcs over the pipeline.
        res1 = ArcBetweenPoints(x_node.get_top() + UP * 0.04,
                                add1.get_top() + UP * 0.04,
                                angle=-PI / 2.4, color=C_HI, stroke_width=2.0)
        res1.add_tip(tip_length=0.11)
        res2 = ArcBetweenPoints(add1.get_top() + UP * 0.04,
                                add2.get_top() + UP * 0.04,
                                angle=-PI / 2.4, color=C_HI, stroke_width=2.0)
        res2.add_tip(tip_length=0.11)
        res_lbl = Text("residual", font_size=13, color=C_HI)\
            .move_to(midpoint(res1.get_top(), res2.get_top()) + UP * 0.18)
        self.play(Create(res1), Create(res2), FadeIn(res_lbl), run_time=1.0)
        self.wait(0.3)

        # Explanation below pipeline, right-aligned, not overlapping ffn_group.
        why_res = _teach([
            "Residual x->+x: gradients flow through identity,",
            "no signal vanishes at any depth.",
            "",
            "Pre-LN: normalise BEFORE each sub-layer.",
            "Result: no warm-up, stable training (Xiong 2020).",
        ], size=16, color=C_ROAD, buff=0.09)
        why_res.next_to(pipeline, DOWN, buff=0.40).to_edge(RIGHT, buff=0.4)
        self.play(FadeIn(why_res, shift=UP * 0.12), run_time=1.5)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ----------------------------------------------------------------------------
# DEEP DIVE 5 / 6  —  Stacking 4 encoder layers, what depth buys you
# ----------------------------------------------------------------------------
class ArchStack(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        head = _section_header(self, "Step 5/6",
                               "Stacking 4 encoder layers builds hierarchy")

        # ── Left panel: short text lines, strictly in the left half ─────────
        intuit = _teach([
            "Each pass through one block:",
            "  LN->MHA->+x  then  LN->FFN->+x",
            "adds one more context layer.",
            "",
            "L1: local  (adjacent curvatures)",
            "L2: medium (consecutive bends)",
            "L3: long   (whole road shape)",
            "L4: high   (failure-risk signal)",
        ], size=18, color=C_ROAD, buff=0.10)
        intuit.next_to(head, DOWN, buff=0.4)
        intuit.to_edge(LEFT, buff=0.40)
        # Safety clamp: scale down if right edge would enter right half.
        if intuit.get_right()[0] > -0.2:
            sf = (-0.2 - intuit.get_left()[0]) / intuit.get_width()
            intuit.scale(sf).to_edge(LEFT, buff=0.40)
        self.play(FadeIn(intuit, shift=UP * 0.15), run_time=1.4)

        # ── Right panel: 4 layer boxes centered at x = +2.5 ─────────────────
        cls_states = [C_ROAD, C_TEAL, C_BLUE, C_PURPLE, C_HI]
        layer_boxes = VGroup()
        for i in range(4):
            box = RoundedRectangle(corner_radius=0.10, height=0.70, width=2.9,
                                   color=C_PURPLE, fill_opacity=0.20, stroke_width=2)
            in_t  = _pill("in",  "", cls_states[i],     w=0.42, h=0.36, fs_top=11)
            arr   = MathTex(r"\to", font_size=19, color=C_PURPLE)
            op    = Text(f"L{i+1}: LN->MHA->LN->FFN",
                         font_size=11, color=C_PURPLE)
            arr2  = MathTex(r"\to", font_size=19, color=C_PURPLE)
            out_t = _pill("out", "", cls_states[i + 1], w=0.42, h=0.36, fs_top=11)
            inner = VGroup(in_t, arr, op, arr2, out_t).arrange(RIGHT, buff=0.08)
            inner.move_to(box.get_center())
            layer_boxes.add(VGroup(box, inner))
        layer_boxes.arrange(DOWN, buff=0.14)
        layer_boxes.move_to(RIGHT * 2.6 + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(l, shift=LEFT * 0.2) for l in layer_boxes],
                              lag_ratio=0.22), run_time=2.0)
        self.wait(0.8)

        bracket = Brace(layer_boxes, RIGHT, color=C_HI, buff=0.10)
        cls_tag = Text("[CLS] refines layer by layer",
                       font_size=16, color=C_HI)\
            .next_to(layer_boxes, DOWN, buff=0.28)
        self.play(GrowFromCenter(bracket), Write(cls_tag), run_time=1.0)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ----------------------------------------------------------------------------
# DEEP DIVE 6 / 6  —  Pool [CLS] + classifier head + sigmoid
# ----------------------------------------------------------------------------
class ArchPoolHead(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        head = _section_header(self, "Step 6/6",
                               "From token soup to one number: y_hat in [0,1]")

        # ===== Pool [CLS] — top row of tokens =====
        seq = VGroup(*[Square(side_length=0.46,
                              color=C_TEAL if i > 0 else C_HI,
                              fill_opacity=0.85 if i == 0 else 0.55)
                       for i in range(10)])
        seq.arrange(RIGHT, buff=0.10).next_to(head, DOWN, buff=0.45)
        cls_inner = Text("[CLS]", font_size=11, color=C_BG, weight=BOLD)\
            .move_to(seq[0])
        seq_lbl = MathTex(r"(B,\,198,\,128)\;\text{after 4 layers}",
                          font_size=22, color=C_ROAD).next_to(seq, UP, buff=0.20)
        self.play(FadeIn(seq, shift=DOWN * 0.2), FadeIn(cls_inner),
                  FadeIn(seq_lbl), run_time=0.9)
        self.wait(0.3)

        # CLS vector extracted below the token row.
        cls_vec = Rectangle(width=3.4, height=0.42, color=C_HI,
                            fill_opacity=0.9, stroke_width=2)
        cls_vec_lbl = MathTex(r"(B,\,128)", font_size=22, color=C_HI)
        cls_vec_lbl.next_to(cls_vec, RIGHT, buff=0.20)
        cls_vec.move_to(LEFT * 2.8 + UP * 0.2)
        cls_vec_lbl.next_to(cls_vec, RIGHT, buff=0.20)

        why_pool = _teach([
            "Keep only x[:,0,:]  — the CLS row.",
            "After 4 attention layers it has",
            "absorbed signals from all 197 tokens.",
        ], size=17, color=C_ROAD, buff=0.10)
        why_pool.next_to(cls_vec, DOWN, buff=0.30).align_to(cls_vec, LEFT)

        self.play(FadeIn(cls_vec, scale=0.9), FadeIn(cls_vec_lbl), run_time=1.0)
        self.play(FadeIn(why_pool, shift=UP * 0.12), run_time=1.0)
        self.wait(1.0)

        self.play(FadeOut(VGroup(seq, cls_inner, seq_lbl, why_pool)),
                  cls_vec.animate.to_edge(UP, buff=1.4).to_edge(LEFT, buff=0.8),
                  run_time=0.7)
        cls_vec_lbl.next_to(cls_vec, RIGHT, buff=0.20)

        # ===== Classifier head — vertical funnel on the LEFT =====
        h2 = Text("Classifier head:  LN->Linear->GELU->Dropout->Linear->sigmoid",
                  font_size=18, color=C_HI, weight=BOLD)
        h2.next_to(cls_vec, DOWN, buff=0.35).to_edge(LEFT, buff=0.6)
        self.play(Write(h2), run_time=0.8)

        # Funnel: 128 -> 64 -> 1 stacked vertically with tighter spacing.
        n128 = Rectangle(width=3.2, height=0.36, color=C_TEAL,
                         fill_opacity=0.75, stroke_width=1.5)
        n128_lbl = MathTex(r"128", font_size=20, color=C_TEAL)\
            .next_to(n128, LEFT, buff=0.12)
        n64 = Rectangle(width=1.6, height=0.36, color=C_BLUE,
                        fill_opacity=0.85, stroke_width=1.5)
        n64_lbl = MathTex(r"64", font_size=20, color=C_BLUE)\
            .next_to(n64, LEFT, buff=0.12)
        n1 = Circle(radius=0.16, color=C_HI, fill_opacity=0.95)
        n1_lbl = MathTex(r"1", font_size=20, color=C_HI)\
            .next_to(n1, LEFT, buff=0.12)

        funnel = VGroup(VGroup(n128_lbl, n128),
                        VGroup(n64_lbl, n64),
                        VGroup(n1_lbl, n1))\
            .arrange(DOWN, buff=0.60)
        funnel.next_to(h2, DOWN, buff=0.28).align_to(h2, LEFT).shift(RIGHT * 0.6)

        op_a = MathTex(r"\xrightarrow[\text{GELU, Dropout}]{\text{Linear }128\to64}",
                       font_size=16, color=C_BLUE)
        op_b = MathTex(r"\xrightarrow{\text{Linear }64\to1}",
                       font_size=16, color=C_HI)
        op_a.move_to(midpoint(n128.get_bottom(), n64.get_top()))
        op_b.move_to(midpoint(n64.get_bottom(), n1.get_top()))

        self.play(FadeIn(n128), FadeIn(n128_lbl), run_time=0.4)
        self.play(Write(op_a), FadeIn(n64), FadeIn(n64_lbl), run_time=0.9)
        self.play(Write(op_b), FadeIn(n1), FadeIn(n1_lbl), run_time=0.9)
        self.wait(0.3)

        # ===== Sigmoid — RIGHT half of screen =====
        ax = Axes(x_range=[-5, 5, 5], y_range=[0, 1.05, 0.5],
                  x_length=4.0, y_length=2.0, tips=False,
                  axis_config={"stroke_width": 0.8, "color": C_ROAD,
                               "include_ticks": True, "include_numbers": False})
        sig = ax.plot(lambda z: 1.0 / (1.0 + np.exp(-z)),
                      color=C_HI, stroke_width=3.0)
        sig_lbl = MathTex(r"\sigma(z)=\frac{1}{1+e^{-z}}",
                          font_size=22, color=C_HI).next_to(ax, UP, buff=0.10)
        sigma_group = VGroup(ax, sig, sig_lbl)
        sigma_group.to_edge(RIGHT, buff=0.55).move_to(
            np.array([sigma_group.get_center()[0], funnel.get_center()[1], 0]))

        self.play(Create(ax), Create(sig), Write(sig_lbl), run_time=1.4)

        z_val = 1.4
        y_val = 1.0 / (1.0 + np.exp(-z_val))
        dot = Dot(ax.c2p(z_val, y_val), color=C_PASS, radius=0.08)
        proj = DashedLine(dot.get_center(), ax.c2p(0, y_val),
                          color=C_PASS, stroke_width=1.2)
        yhat_lbl = MathTex(r"\hat y", font_size=26, color=C_PASS)\
            .next_to(proj.get_end(), LEFT, buff=0.12)
        self.play(GrowFromCenter(dot), Create(proj), Write(yhat_lbl), run_time=0.9)
        self.play(Flash(dot, color=C_PASS, flash_radius=0.3))
        self.wait(0.3)

        # ===== Final messages at bottom =====
        final = Text("y_hat  =  P(this road causes a simulation failure)",
                     font_size=22, color=C_PASS, weight=BOLD).to_edge(DOWN, buff=0.35)
        sort_step = Text("Tests sorted by y_hat descending -> fail-first order.",
                         font_size=19, color=C_ROAD).next_to(final, UP, buff=0.12)
        self.play(Write(sort_step), run_time=0.9)
        self.play(Write(final), run_time=1.0)
        self.wait(2.2)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.9)


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
