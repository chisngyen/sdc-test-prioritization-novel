"""
Scene 02 -- Input representation.

The raw test case is a sequence of (x, y) road points.  We show the
sequence as a tensor, then point out that this representation already
commits to a particular global frame -- which is exactly what we need
to factor out.

Render:  manim -pql scene_02_input.py InputPoints
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex,
    Rectangle, Dot, Line, Arrow, NumberPlane,
    Write, FadeIn, FadeOut, Create, LaggedStart,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    UR, DR,
    WHITE, BLUE_D, YELLOW, GREY_A,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_road, to_scene_coords,
    ROAD_COLOR, POINT_COLOR, fmt,
)
from theme import (
    TEXT, MUTED, PRIMARY, ACCENT, GOOD, WARN, BAD, RULE,
    section_header, transition, hold,
    title, subtitle, footer, body_text, caption,
    body_formula, inline_math,
    panel, accent_box,
    attach_narration, seal_narration,
    MATH_BODY, MATH_INLINE,
)


class InputPoints(Scene):
    def construct(self):
        attach_narration(self, "scene_02")
        header = section_header(
            self, "Step 1.  The input is a sequence of points",
            "Each test case = an ordered list of (x, y) coordinates.",
        )

        # ----------------------- coordinate frame + road -- #
        plane = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-1.6, 1.6, 1],
            x_length=9.8, y_length=4.0,
            background_line_style={
                "stroke_color": BLUE_D, "stroke_width": 1, "stroke_opacity": 0.22,
            },
            axis_config={"include_numbers": False, "stroke_opacity": 0.45,
                         "stroke_color": GREY_A},
        ).shift(DOWN * 0.55)

        pts = sample_road(n=20) * 0.85
        coords = to_scene_coords(pts) + np.array([0, -0.55, 0])

        road_line = VMobject(stroke_color=PRIMARY, stroke_width=7)
        road_line.set_points_smoothly(coords)

        self.play(FadeIn(plane, run_time=0.6))
        self.play(Create(road_line), run_time=1.6)

        dots = VGroup(*[Dot(p, radius=0.078, color=ACCENT) for p in coords])
        self.play(
            LaggedStart(*[FadeIn(d, scale=1.5) for d in dots],
                        lag_ratio=0.06, run_time=2.0),
        )
        hold(self, 0.3)

        # ----------------- label first three coords -- #
        labels = VGroup()
        for i, direction in enumerate([UR, DR, UR]):
            x, y, _ = coords[i]
            lab = inline_math(
                rf"(x_{{{i+1}}},\,y_{{{i+1}}}) = "
                rf"({fmt(x, 2)},\,{fmt(y, 2)})",
                color=ACCENT,
            )
            lab.next_to(dots[i], direction, buff=0.18)
            labels.add(lab)

        self.play(
            LaggedStart(*[Write(l) for l in labels],
                        lag_ratio=0.35, run_time=1.8),
        )

        # arrows showing order
        order_arrows = VGroup(*[
            Arrow(a, b, buff=0.10, stroke_width=3, color=WARN,
                  max_tip_length_to_length_ratio=0.4)
            for a, b in zip(coords[:3], coords[1:4])
        ])
        self.play(
            LaggedStart(*[Create(a) for a in order_arrows],
                        lag_ratio=0.30, run_time=1.0),
        )
        order_cap = caption("ordered along the road", color=WARN, italic=False)
        order_cap.next_to(plane, DOWN, buff=0.20)
        self.play(FadeIn(order_cap, shift=UP * 0.10), run_time=0.5)
        hold(self, 0.6)

        # ------------------ collapse to a tensor -- #
        self.play(
            FadeOut(labels),
            FadeOut(order_arrows),
            FadeOut(order_cap),
            run_time=0.5,
        )

        # Squeeze the plane to the left
        plane_grp = VGroup(plane, road_line, dots)
        self.play(plane_grp.animate.scale(0.62).to_edge(LEFT, buff=0.6).shift(DOWN * 0.1),
                  run_time=1.0)

        # Tensor on the right
        tensor_lbl = body_text("As a tensor:", color=PRIMARY)
        tensor_lbl.move_to([3.4, 1.0, 0])

        tensor = MathTex(
            r"\mathbf{r} \;=\; "
            r"\begin{bmatrix} x_1 & y_1 \\ x_2 & y_2 \\ \vdots & \vdots \\ x_L & y_L \end{bmatrix}"
            r" \in \mathbb{R}^{\,L \times 2}",
            font_size=38,
        ).next_to(tensor_lbl, DOWN, buff=0.35)
        tensor.set_color_by_tex(r"\mathbf{r}", ACCENT)

        shape = inline_math(r"L \in [64,\, 197]", color=GOOD)
        shape.next_to(tensor, DOWN, buff=0.45)

        self.play(Write(tensor_lbl), run_time=0.5)
        self.play(Write(tensor), run_time=1.8)
        self.play(FadeIn(shape, shift=UP * 0.12), run_time=0.5)
        hold(self, 1.0)

        # ----------------- the catch: frame-dependent -- #
        warn_box = Rectangle(
            width=12.0, height=0.75, stroke_color=BAD, stroke_width=2,
            fill_color="#2a0e10", fill_opacity=0.85,
        ).move_to([0, -3.40, 0])
        warn_text = Text(
            "BUT  (x, y) depends on where the origin is, and which way is north.",
            font_size=22, color=BAD,
        ).move_to(warn_box.get_center())

        self.play(FadeIn(warn_box), Write(warn_text), run_time=1.0)
        hold(self, 2.0)

        # ---------------- outro -- #
        transition(self)
        end = Text(
            "Next: 7 numbers per point that rotation cannot touch.",
            font_size=30, color=TEXT,
        )
        self.play(FadeIn(end, shift=UP * 0.15))
        hold(self, 1.6)
        self.play(FadeOut(end))
        seal_narration(self, "scene_02")
