"""
Scene 04 -- Rotation invariance, proven by computation.

Same road, two orientations.  Extract the 7 intrinsic channels at the
same index from both copies; every number matches to machine precision.

Render:
    manim -pql scene_04_invariance.py RotationProof
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Circle, Dot, Line, Arrow,
    DashedLine, Arc, Table,
    Write, FadeIn, FadeOut, Create, Uncreate, ReplacementTransform,
    Indicate, Flash, LaggedStart, Transform, Rotate, AnimationGroup,
    ValueTracker, always_redraw, DecimalNumber, MoveAlongPath,
    UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL, ORIGIN, PI, DEGREES,
    WHITE, BLUE, BLUE_A, BLUE_B, BLUE_D, BLUE_E, YELLOW, ORANGE, RED, RED_A,
    GREEN, GREEN_A, GREY, GREY_A, GREY_B, GREY_C, GOLD, PINK, MAROON, TEAL,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_road, rotate_points, to_scene_coords, extract_invariant_7ch,
    ROAD_COLOR, POINT_COLOR, FEATURE_COLORS, FEATURE_NAMES, FEATURE_KEYS,
    fmt,
)


FOCUS_IDX = 12
ROT_DEG = 60.0


class RotationProof(Scene):
    def construct(self):
        # ---------- title --------------------------------------------------- #
        title = Text("Rotation invariance, proven by computation",
                     font_size=34, color=WHITE).to_edge(UP, buff=0.4)
        sub = MathTex(
            r"\phi(R\,\mathbf{r}) \;=\; \phi(\mathbf{r})"
            r"\qquad \text{for every } R \in SO(2)",
            font_size=26, color=BLUE_A,
        ).next_to(title, DOWN, buff=0.2)
        self.play(Write(title), Write(sub))
        self.wait(0.6)

        # ---------- two roads side-by-side ---------------------------------- #
        base = sample_road(n=20) * 0.7
        rotated = rotate_points(base, ROT_DEG)

        left_center = np.array([-3.5, 0.4, 0])
        right_center = np.array([3.5, 0.4, 0])

        left_coords  = to_scene_coords(base) + left_center
        right_coords = to_scene_coords(rotated) + right_center

        left_road  = VMobject(stroke_color=ROAD_COLOR, stroke_width=6)
        left_road.set_points_smoothly(left_coords)
        right_road = VMobject(stroke_color=ROAD_COLOR, stroke_width=6)
        right_road.set_points_smoothly(right_coords)

        left_dots  = VGroup(*[Dot(p, radius=0.05, color=POINT_COLOR) for p in left_coords])
        right_dots = VGroup(*[Dot(p, radius=0.05, color=POINT_COLOR) for p in right_coords])

        left_lbl  = Text("original", font_size=22, color=GREY_A).next_to(left_road, DOWN, buff=0.3)
        right_lbl = Tex(r"rotated by $60^{\circ}$",
                        font_size=28, color=GREY_A).next_to(right_road, DOWN, buff=0.3)

        self.play(Create(left_road), Create(right_road), run_time=1.6)
        self.play(
            LaggedStart(*[FadeIn(d, scale=1.4) for d in left_dots],
                        lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=1.4) for d in right_dots],
                        lag_ratio=0.04),
            run_time=1.2,
        )
        self.play(FadeIn(left_lbl), FadeIn(right_lbl))
        self.wait(0.4)

        # Mark the focus point on both
        focus_l = Dot(left_coords[FOCUS_IDX], radius=0.11, color=YELLOW)
        focus_r = Dot(right_coords[FOCUS_IDX], radius=0.11, color=YELLOW)
        ring_l = Circle(radius=0.20, color=YELLOW, stroke_width=2).move_to(focus_l)
        ring_r = Circle(radius=0.20, color=YELLOW, stroke_width=2).move_to(focus_r)
        idx_lbl_l = MathTex(f"i={FOCUS_IDX}", font_size=22,
                            color=YELLOW).next_to(focus_l, UP, buff=0.18)
        idx_lbl_r = MathTex(f"i={FOCUS_IDX}", font_size=22,
                            color=YELLOW).next_to(focus_r, UP, buff=0.18)
        self.play(FadeIn(focus_l), FadeIn(focus_r),
                  Create(ring_l), Create(ring_r),
                  Write(idx_lbl_l), Write(idx_lbl_r))
        self.wait(0.6)

        # ---------- table of 7 features for both sides ---------------------- #
        feats_l = extract_invariant_7ch(base)
        feats_r = extract_invariant_7ch(rotated)

        rows = []
        for k, key in enumerate(FEATURE_KEYS):
            vl = feats_l[FOCUS_IDX, k]
            vr = feats_r[FOCUS_IDX, k]
            delta = abs(vl - vr)
            rows.append((key, vl, vr, delta))

        # Move everything up to make room for a bottom table
        self.play(
            VGroup(left_road, right_road, left_dots, right_dots,
                   focus_l, focus_r, ring_l, ring_r,
                   idx_lbl_l, idx_lbl_r, left_lbl, right_lbl,
                   title, sub).animate.shift(UP * 1.2),
            run_time=1.0,
        )

        # Header row
        header = VGroup(
            Text("channel", font_size=22, color=GREY_A),
            Text("original", font_size=22, color=BLUE_A),
            Text("rotated",  font_size=22, color=BLUE_A),
            Tex(r"$|\Delta|$", font_size=24, color=YELLOW),
        ).arrange(RIGHT, buff=1.5)
        header.to_edge(DOWN, buff=2.6).shift(DOWN * 0.0)

        # Build numerical rows
        col_x = [c.get_x() for c in header]
        row_mobs = []
        for r_i, (key, vl, vr, delta) in enumerate(rows):
            y = header.get_y() - 0.45 * (r_i + 1)
            cells = VGroup(
                MathTex(FEATURE_NAMES[key], font_size=22,
                        color=FEATURE_COLORS[key]),
                Text(fmt(vl, 4), font_size=20, color=WHITE),
                Text(fmt(vr, 4), font_size=20, color=WHITE),
                Text(f"{delta:.2e}" if delta > 0 else "0.0000",
                     font_size=20, color=GREEN_A if delta < 1e-6 else ORANGE),
            )
            for cell, x in zip(cells, col_x):
                cell.move_to([x, y, 0])
            row_mobs.append(cells)

        self.play(FadeIn(header, shift=UP * 0.05))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.08) for r in row_mobs],
                              lag_ratio=0.15, run_time=2.4))
        self.wait(0.5)

        # Highlight the delta column
        delta_box = SurroundingBoxLine(
            row_mobs, col_idx=3, color=GREEN_A, padding=0.12,
        )
        self.play(Create(delta_box), run_time=0.9)
        self.wait(0.4)

        verdict = Text(
            "Every feature matches to machine precision.",
            font_size=26, color=GREEN_A,
        ).next_to(delta_box, DOWN, buff=0.25)
        self.play(Write(verdict))
        self.wait(1.6)

        # ---------- contrast: baseline 10-ch leaks the angle ---------------- #
        self.play(
            FadeOut(VGroup(header, *row_mobs, delta_box, verdict)),
            run_time=0.8,
        )

        contrast_title = Text(
            "What about the baseline? It feeds raw heading sin θ, cos θ.",
            font_size=24, color=ORANGE,
        ).move_to([0, -1.2, 0])
        self.play(Write(contrast_title), run_time=1.4)

        # angle of segment i in both copies
        v_l = base[FOCUS_IDX + 1] - base[FOCUS_IDX]
        v_r = rotated[FOCUS_IDX + 1] - rotated[FOCUS_IDX]
        ang_l = np.arctan2(v_l[1], v_l[0])
        ang_r = np.arctan2(v_r[1], v_r[0])
        sin_l, cos_l = np.sin(ang_l), np.cos(ang_l)
        sin_r, cos_r = np.sin(ang_r), np.cos(ang_r)

        leak_table = VGroup(
            VGroup(MathTex(r"\sin\theta_i", font_size=24, color=RED_A),
                   Text(fmt(sin_l, 3), font_size=22, color=WHITE),
                   Text(fmt(sin_r, 3), font_size=22, color=WHITE),
                   Text(f"{abs(sin_l - sin_r):.3f}", font_size=22, color=RED)
                   ).arrange(RIGHT, buff=1.2),
            VGroup(MathTex(r"\cos\theta_i", font_size=24, color=RED_A),
                   Text(fmt(cos_l, 3), font_size=22, color=WHITE),
                   Text(fmt(cos_r, 3), font_size=22, color=WHITE),
                   Text(f"{abs(cos_l - cos_r):.3f}", font_size=22, color=RED)
                   ).arrange(RIGHT, buff=1.2),
        ).arrange(DOWN, buff=0.25).next_to(contrast_title, DOWN, buff=0.35)

        self.play(FadeIn(leak_table, shift=UP * 0.15))
        self.wait(0.4)

        leak_msg = Text(
            "These rotate with the road. The baseline must LEARN to ignore them — and rarely fully does.",
            font_size=20, color=RED,
        ).next_to(leak_table, DOWN, buff=0.3)
        self.play(Write(leak_msg))
        self.wait(2.0)

        # ---------- closer ---------------------------------------------------#
        self.play(FadeOut(VGroup(
            contrast_title, leak_table, leak_msg,
            left_road, right_road, left_dots, right_dots,
            focus_l, focus_r, ring_l, ring_r,
            idx_lbl_l, idx_lbl_r, left_lbl, right_lbl,
            title, sub,
        )))

        closer = Text(
            "Invariance for free.  Now: how the 7 channels become a score.",
            font_size=28, color=WHITE,
        )
        self.play(FadeIn(closer, shift=UP * 0.15))
        self.wait(1.6)
        self.play(FadeOut(closer))


# ----------------------------------------------------------------------------- #
# Tiny utility: a "surrounding box" that fits one column of a virtual table.
# We pass it the row VGroups and an integer column index.
# ----------------------------------------------------------------------------- #
def SurroundingBoxLine(rows, col_idx: int, *, color=YELLOW, padding=0.1):
    cells = [r[col_idx] for r in rows]
    xs = [c.get_x() for c in cells]
    ys = [c.get_y() for c in cells]
    width = max(c.width for c in cells) + 2 * padding
    height = (max(ys) - min(ys)) + max(c.height for c in cells) + 2 * padding
    box = Rectangle(width=width, height=height,
                    stroke_color=color, stroke_width=3, fill_opacity=0.0)
    box.move_to([np.mean(xs), np.mean(ys), 0])
    return box
