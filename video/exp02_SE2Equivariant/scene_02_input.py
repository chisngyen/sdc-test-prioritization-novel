"""
Scene 02 -- Input representation.

The raw test case is a sequence of (x, y) road points.  We show the
sequence, then highlight that this representation already commits to a
particular global frame -- which is exactly what we need to factor out.

Render:
    manim -pql scene_02_input.py InputPoints
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, Square, Circle, Dot, Line, Arrow, DashedLine, NumberPlane,
    Write, FadeIn, FadeOut, Create, Uncreate, ReplacementTransform,
    Indicate, Flash, LaggedStart, Transform,
    UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL, ORIGIN, PI, DEGREES,
    WHITE, BLUE, BLUE_A, BLUE_B, BLUE_D, BLUE_E, YELLOW, ORANGE, RED,
    GREEN, GREEN_A, GREY, GREY_A, GREY_C, GOLD, PINK, MAROON,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_road, to_scene_coords, make_road,
    ROAD_COLOR, POINT_COLOR, fmt,
)


class InputPoints(Scene):
    def construct(self):
        # ---------- 1. Header ----------------------------------------------
        title = Text("Step 1 — The input is a sequence of points",
                     font_size=34, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1.2)

        # ---------- 2. Coordinate frame + road -----------------------------
        plane = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-2, 2, 1],
            x_length=10, y_length=5,
            background_line_style={
                "stroke_color": BLUE_D, "stroke_width": 1, "stroke_opacity": 0.25,
            },
            axis_config={"include_numbers": False, "stroke_opacity": 0.4},
        ).shift(DOWN * 0.4)

        pts = sample_road(n=20) * 0.9
        coords = to_scene_coords(pts + np.array([0, -0.4]))   # shift to match plane

        road_line = VMobject(stroke_color=ROAD_COLOR, stroke_width=8)
        road_line.set_points_smoothly(coords)

        self.play(FadeIn(plane, run_time=0.8))
        self.play(Create(road_line), run_time=1.6)

        # ---------- 3. Reveal the sample points one by one -----------------
        dots = VGroup(*[Dot(p, radius=0.075, color=POINT_COLOR) for p in coords])
        self.play(
            LaggedStart(*[FadeIn(d, scale=1.6) for d in dots],
                        lag_ratio=0.08, run_time=2.4)
        )
        self.wait(0.4)

        # ---------- 4. Annotate the first three coords ----------------------
        labels = VGroup()
        directions = [UR, DR, UR]
        for i, dir_ in enumerate(directions):
            x, y, _ = coords[i]
            lab = MathTex(
                rf"(x_{{{i+1}}},\,y_{{{i+1}}}) = "
                rf"({fmt(x, 2)},\,{fmt(y, 2)})",
                font_size=22, color=YELLOW,
            )
            lab.next_to(dots[i], dir_, buff=0.18)
            labels.add(lab)

        self.play(LaggedStart(*[Write(l) for l in labels],
                              lag_ratio=0.4, run_time=2.2))
        self.wait(0.6)

        # connect them with a chevron to remind: ORDERED
        order_arrows = VGroup()
        for a, b in zip(coords[:3], coords[1:4]):
            order_arrows.add(
                Arrow(a, b, buff=0.10, stroke_width=3, color=ORANGE,
                      max_tip_length_to_length_ratio=0.4)
            )
        self.play(LaggedStart(*[Create(a) for a in order_arrows],
                              lag_ratio=0.3, run_time=1.2))
        order_lbl = Text("ordered along the road",
                         font_size=20, color=ORANGE).next_to(plane, DOWN, buff=0.25)
        self.play(FadeIn(order_lbl, shift=UP * 0.1))
        self.wait(0.6)

        # ---------- 5. Compress into a tensor on the right ------------------
        self.play(
            FadeOut(labels), FadeOut(order_arrows), FadeOut(order_lbl),
            plane.animate.scale(0.55).to_edge(LEFT, buff=0.6).shift(DOWN * 0.3),
            road_line.animate.scale(0.55).move_to(
                plane.copy().scale(0.55).to_edge(LEFT, buff=0.6).shift(DOWN * 0.3).get_center()
            ),
            dots.animate.scale(0.55).move_to(
                plane.copy().scale(0.55).to_edge(LEFT, buff=0.6).shift(DOWN * 0.3).get_center()
            ),
            run_time=1.2,
        )
        # The animate-on-copy gymnastics above can drift; lock road + dots to the
        # plane centre to be safe.
        target_center = plane.get_center()
        road_line.move_to(target_center)
        dots.move_to(target_center)

        tensor_title = Text("As a tensor:", font_size=24, color=BLUE_A)
        tensor_title.next_to(plane, RIGHT, buff=1.0).shift(UP * 1.1)
        self.play(Write(tensor_title))

        tensor = MathTex(
            r"\mathbf{r} \;=\; "
            r"\begin{bmatrix} x_1 & y_1 \\ x_2 & y_2 \\ \vdots & \vdots \\ x_L & y_L \end{bmatrix}"
            r"\in \mathbb{R}^{\,L \times 2}",
            font_size=36,
        ).next_to(tensor_title, DOWN, buff=0.3)
        tensor.set_color_by_tex(r"\mathbf{r}", YELLOW)

        self.play(Write(tensor), run_time=2.0)

        shape = MathTex(r"L = 197", font_size=28, color=GREEN_A)
        shape.next_to(tensor, DOWN, buff=0.4)
        self.play(FadeIn(shape, shift=UP * 0.15))
        self.wait(0.8)

        # ---------- 6. The catch -- this representation leaks the frame -----
        warn_bar = Rectangle(width=7.4, height=0.9,
                             stroke_color=RED, stroke_width=2,
                             fill_color="#2a0e10", fill_opacity=0.85)
        warn_bar.to_edge(DOWN, buff=0.6)
        warn_text = Text(
            "But (x, y) depends on where the origin is, and which way is north.",
            font_size=22, color=RED,
        ).move_to(warn_bar.get_center())

        self.play(FadeIn(warn_bar), Write(warn_text))
        self.wait(1.4)
        self.play(Indicate(warn_text, scale_factor=1.05, color=YELLOW), run_time=1.2)
        self.wait(0.8)

        # ---------- 7. Outro -------------------------------------------------
        self.play(FadeOut(VGroup(
            title, plane, road_line, dots, tensor_title, tensor, shape,
            warn_bar, warn_text,
        )))

        end = Text(
            "Next: 7 numbers per point that the rotation cannot touch.",
            font_size=30, color=WHITE,
        )
        self.play(FadeIn(end, shift=UP * 0.2))
        self.wait(1.6)
        self.play(FadeOut(end))
