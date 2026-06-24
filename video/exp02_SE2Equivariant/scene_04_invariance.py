"""
Scene 04 -- Rotation invariance, proven by computation.

Same road, two orientations.  Extract the 7 intrinsic channels at the
same index from both copies; every number matches to machine precision.

Render:  manim -pql scene_04_invariance.py RotationProof
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, Circle, Dot, Line,
    Write, FadeIn, FadeOut, Create, LaggedStart,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, BLUE_A, YELLOW, GREY_A, RED, GREEN_A,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_road, rotate_points, to_scene_coords, extract_invariant_7ch,
    ROAD_COLOR, POINT_COLOR, FEATURE_COLORS, FEATURE_NAMES, FEATURE_KEYS,
    fmt,
)
from theme import (
    TEXT, MUTED, PRIMARY, ACCENT, GOOD, WARN, BAD, RULE,
    section_header, transition, hold,
    title, subtitle, footer, body_text, caption,
    body_formula, inline_math, accent_box,
    attach_narration, seal_narration,
    MATH_INLINE,
)


FOCUS_IDX = 12
ROT_DEG = 60.0


def _column_box(rows: list[VGroup], col_idx: int, *,
                color=ACCENT, padding: float = 0.12) -> Rectangle:
    cells = [r[col_idx] for r in rows]
    xs = [c.get_x() for c in cells]
    ys = [c.get_y() for c in cells]
    width  = max(c.width  for c in cells) + 2 * padding
    height = (max(ys) - min(ys)) + max(c.height for c in cells) + 2 * padding
    box = Rectangle(width=width, height=height,
                    stroke_color=color, stroke_width=3, fill_opacity=0.0)
    box.move_to([float(np.mean(xs)), float(np.mean(ys)), 0])
    return box


class RotationProof(Scene):
    def construct(self):
        attach_narration(self, "scene_04")
        head = title("Rotation invariance, proven by computation")
        ul = Line(
            head.get_corner(DOWN + LEFT) + DOWN * 0.10,
            head.get_corner(DOWN + RIGHT) + DOWN * 0.10,
            color=PRIMARY, stroke_width=2,
        )
        sub = MathTex(
            r"\phi(R\,\mathbf{r}) \;=\; \phi(\mathbf{r})"
            r"\qquad \text{for every } R \in SO(2)",
            font_size=26, color=PRIMARY,
        ).move_to([0, 2.55, 0])
        self.play(Write(head), Create(ul), run_time=0.8)
        self.play(FadeIn(sub, shift=DOWN * 0.1), run_time=0.5)
        self.header = VGroup(head, ul, sub)

        # ---------------- two roads side by side ----------------
        # Small scale + low centre so the *rotated* copy (which becomes tall)
        # stays under the subtitle AND well above the delta table below.
        base = sample_road(n=20) * 0.40
        rotated = rotate_points(base, ROT_DEG)

        L = np.array([-3.7, 1.10, 0.0])
        R = np.array([+3.7, 1.10, 0.0])

        Lc = to_scene_coords(base)    + L
        Rc = to_scene_coords(rotated) + R

        left_road  = VMobject(stroke_color=PRIMARY, stroke_width=6)
        left_road.set_points_smoothly(Lc)
        right_road = VMobject(stroke_color=PRIMARY, stroke_width=6)
        right_road.set_points_smoothly(Rc)

        left_dots  = VGroup(*[Dot(p, radius=0.055, color=ACCENT) for p in Lc])
        right_dots = VGroup(*[Dot(p, radius=0.055, color=ACCENT) for p in Rc])

        left_lbl  = Text("original", font_size=22, color=MUTED).next_to(left_road,  DOWN, buff=0.20)
        right_lbl = Tex(r"rotated by $60^{\circ}$", color=MUTED).scale_to_fit_height(0.28)
        right_lbl.next_to(right_road, DOWN, buff=0.20)

        self.play(Create(left_road), Create(right_road), run_time=1.4)
        self.play(
            LaggedStart(*[FadeIn(d, scale=1.4) for d in left_dots],  lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=1.4) for d in right_dots], lag_ratio=0.04),
            run_time=1.0,
        )
        self.play(FadeIn(left_lbl), FadeIn(right_lbl), run_time=0.5)

        focus_l = Dot(Lc[FOCUS_IDX], radius=0.11, color=ACCENT)
        focus_r = Dot(Rc[FOCUS_IDX], radius=0.11, color=ACCENT)
        ring_l = Circle(radius=0.22, color=ACCENT, stroke_width=2).move_to(focus_l)
        ring_r = Circle(radius=0.22, color=ACCENT, stroke_width=2).move_to(focus_r)
        idx_l = inline_math(f"i={FOCUS_IDX}", color=ACCENT).next_to(focus_l, UP, buff=0.18)
        idx_r = inline_math(f"i={FOCUS_IDX}", color=ACCENT).next_to(focus_r, UP, buff=0.18)
        self.play(
            FadeIn(focus_l), FadeIn(focus_r),
            Create(ring_l), Create(ring_r),
            Write(idx_l), Write(idx_r), run_time=0.7,
        )
        hold(self, 0.5)

        self.scene_top = VGroup(
            left_road, right_road, left_dots, right_dots,
            focus_l, focus_r, ring_l, ring_r, idx_l, idx_r,
            left_lbl, right_lbl,
        )

        # ------------- table of 7 features for both sides -------------
        feats_l = extract_invariant_7ch(base)
        feats_r = extract_invariant_7ch(rotated)

        # No further shift -- the roads are already placed high enough.
        # Build the table immediately below.
        col_xs = [-4.3, -1.3, 1.6, 4.4]
        header_row_y = -0.55
        col_titles = ["channel", "original", "rotated", r"|\Delta|"]
        col_colors = [MUTED, PRIMARY, PRIMARY, ACCENT]
        header_cells = []
        for x, ttl, col in zip(col_xs, col_titles, col_colors):
            if ttl.startswith("|"):
                m = inline_math(ttl, color=col)
            else:
                m = Text(ttl, font_size=22, color=col)
            m.move_to([x, header_row_y, 0])
            header_cells.append(m)
        header_grp = VGroup(*header_cells)

        rule = Line(
            [col_xs[0] - 0.5, header_row_y - 0.30, 0],
            [col_xs[-1] + 0.5, header_row_y - 0.30, 0],
            color=GREY_A, stroke_width=1,
        )

        rows = []
        for r_i, key in enumerate(FEATURE_KEYS):
            y = header_row_y - 0.35 - 0.32 * r_i
            vl = feats_l[FOCUS_IDX, FEATURE_KEYS.index(key)]
            vr = feats_r[FOCUS_IDX, FEATURE_KEYS.index(key)]
            delta = abs(vl - vr)

            cells = VGroup(
                MathTex(FEATURE_NAMES[key], font_size=22,
                        color=FEATURE_COLORS[key]),
                Text(fmt(vl, 4), font_size=18, color=TEXT),
                Text(fmt(vr, 4), font_size=18, color=TEXT),
                Text(f"{delta:.2e}" if delta > 0 else "0.0000",
                     font_size=18, color=GOOD if delta < 1e-6 else WARN),
            )
            for c, x in zip(cells, col_xs):
                c.move_to([x, y, 0])
            rows.append(cells)

        self.play(FadeIn(header_grp, shift=UP * 0.10),
                  Create(rule), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.08) for r in rows],
                              lag_ratio=0.14, run_time=2.0))
        hold(self, 0.5)

        delta_box = _column_box(rows, 3, color=GOOD)
        self.play(Create(delta_box), run_time=0.7)
        verdict = Text(
            "Every feature matches to machine precision.",
            font_size=24, color=GOOD,
        ).move_to([0, -3.30, 0])
        self.play(Write(verdict), run_time=0.9)
        hold(self, 1.6)

        self.table_block = VGroup(header_grp, rule, *rows, delta_box, verdict)

        # ----------- baseline contrast: sin/cos leak ----------
        # Clear BOTH the table and the two roads, so the contrast block owns a
        # clean canvas (the roads used to linger under the leak table).
        self.play(FadeOut(self.table_block), FadeOut(self.scene_top), run_time=0.5)

        contrast_head = body_text(
            "What about the baseline? It feeds raw heading sin and cos.",
            color=WARN,
        ).move_to([0, 1.20, 0])
        self.play(Write(contrast_head), run_time=1.0)

        v_l = base[FOCUS_IDX + 1] - base[FOCUS_IDX]
        v_r = rotated[FOCUS_IDX + 1] - rotated[FOCUS_IDX]
        ang_l = np.arctan2(v_l[1], v_l[0])
        ang_r = np.arctan2(v_r[1], v_r[0])
        sin_l, cos_l = np.sin(ang_l), np.cos(ang_l)
        sin_r, cos_r = np.sin(ang_r), np.cos(ang_r)

        leak_table = VGroup(
            VGroup(
                MathTex(r"\sin\theta_i", font_size=26, color=BAD),
                Text(fmt(sin_l, 3), font_size=22, color=TEXT),
                Text(fmt(sin_r, 3), font_size=22, color=TEXT),
                Text(f"{abs(sin_l - sin_r):.3f}", font_size=22, color=BAD),
            ).arrange(RIGHT, buff=1.4),
            VGroup(
                MathTex(r"\cos\theta_i", font_size=26, color=BAD),
                Text(fmt(cos_l, 3), font_size=22, color=TEXT),
                Text(fmt(cos_r, 3), font_size=22, color=TEXT),
                Text(f"{abs(cos_l - cos_r):.3f}", font_size=22, color=BAD),
            ).arrange(RIGHT, buff=1.4),
        ).arrange(DOWN, buff=0.30).next_to(contrast_head, DOWN, buff=0.40)
        self.play(FadeIn(leak_table, shift=UP * 0.12), run_time=0.8)

        leak_msg = body_text(
            "These rotate with the road.  The baseline must learn to ignore them.",
            color=BAD,
        ).scale(0.85).next_to(leak_table, DOWN, buff=0.30)
        self.play(Write(leak_msg), run_time=1.0)
        hold(self, 2.0)

        # ----------- closer -----------
        transition(self)
        closer = Text(
            "Invariance for free.  Now: how the 7 channels become a score.",
            font_size=28, color=TEXT,
        )
        self.play(FadeIn(closer, shift=UP * 0.15))
        hold(self, 1.6)
        # Hold the closer under the remaining narration instead of cutting to black.
        seal_narration(self, "scene_04")
        self.play(FadeOut(closer), run_time=0.5)
