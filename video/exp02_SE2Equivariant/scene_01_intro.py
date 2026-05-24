"""
Scene 01 -- Intro.

Sets the stage: SDC test prioritization, why rotation breaks a vanilla
prioritizer, and the equivariance equation we want to satisfy.

Render:
    manim -pql scene_01_intro.py Intro
    manim -qh  scene_01_intro.py Intro     # final
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, Text, MathTex, Tex, Title,
    Rectangle, Square, Circle, Triangle, Dot, Line, Arrow, Cross,
    Write, FadeIn, FadeOut, Create, Uncreate, ReplacementTransform,
    Indicate, Flash, Rotate, MoveAlongPath, LaggedStart, AnimationGroup,
    ValueTracker, always_redraw, DecimalNumber,
    UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL, ORIGIN, PI, DEGREES,
    WHITE, BLUE, BLUE_A, BLUE_B, BLUE_D, BLUE_E, YELLOW, ORANGE, RED, RED_A,
    GREEN, GREEN_B, GREY, GREY_A, GREY_C, GOLD, PINK, MAROON, TEAL,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import sample_road, rotate_points, make_road, to_scene_coords


class Intro(Scene):
    def construct(self):
        # ---------- 1. Title card --------------------------------------------
        title = Text(
            "SE(2)-Equivariant RoadNet",
            font_size=56, color=WHITE, weight="BOLD",
        )
        subtitle = Text(
            "A prioritizer that is mathematically incapable of caring "
            "about rotation.",
            font_size=24, color=BLUE_A,
        ).next_to(title, DOWN, buff=0.4)
        tag = Text(
            "exps/exp02_SE2Equivariant.py",
            font_size=18, color=GREY_A, slant="ITALIC",
        ).next_to(subtitle, DOWN, buff=0.35)

        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(subtitle, shift=UP * 0.2), FadeIn(tag, shift=UP * 0.1))
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, subtitle, tag)))

        # ---------- 2. The setup: a road + a car ----------------------------
        pts = sample_road(n=36) * 0.85
        coords = to_scene_coords(pts)
        road_group = make_road(pts, show_dots=False, stroke_width=8)
        road_group.move_to(ORIGIN)
        coords = road_group[0].points  # rely on the smoothed VMobject path

        header = Text("A self-driving test case is a road shape.",
                      font_size=28, color=WHITE).to_edge(UP, buff=0.6)
        self.play(Write(header), run_time=1.2)
        self.play(Create(road_group), run_time=2.0)

        # The car -- triangular wedge
        car = Triangle(color=YELLOW, fill_color=YELLOW, fill_opacity=1.0)
        car.scale(0.18).rotate(-PI / 2)

        # Drive the car along the road
        path = road_group  # VMobject (its first sub-mobject is the smooth line)
        # MoveAlongPath needs the smooth path mobject directly
        smooth_path = path
        car.move_to(coords[0] if hasattr(coords, "__len__") else ORIGIN)
        self.play(FadeIn(car, scale=0.7))
        self.play(MoveAlongPath(car, smooth_path, rate_func=lambda t: t), run_time=3.0)
        self.wait(0.4)

        # ---------- 3. The model gives a score ------------------------------
        score_box = Rectangle(width=4.3, height=1.4, color=BLUE_E,
                              fill_color="#0e1a2b", fill_opacity=0.95).to_edge(DOWN, buff=0.6)
        score_label = Text("Baseline predicts:", font_size=22,
                           color=BLUE_A).next_to(score_box, UP, buff=0.2)
        score_text = MathTex(r"P(\text{FAIL}) = 0.85", font_size=42, color=RED)
        score_text.move_to(score_box.get_center())

        self.play(FadeOut(header))
        self.play(FadeIn(score_box), Write(score_label))
        self.play(Write(score_text))
        self.wait(0.8)

        # ---------- 4. Rotate the road -------------------------------------
        rot_label = Text("Now rotate the road by 60 degrees...",
                         font_size=28, color=WHITE).to_edge(UP, buff=0.6)
        self.play(Write(rot_label))
        rotation_grp = VGroup(road_group, car)
        self.play(Rotate(rotation_grp, angle=60 * DEGREES, about_point=ORIGIN),
                  run_time=2.0)
        self.wait(0.4)

        # Same physics -- new score, much worse
        new_score = MathTex(r"P(\text{FAIL}) = 0.32", font_size=42, color=RED_A)
        new_score.move_to(score_box.get_center())
        drop_arrow = MathTex(r"\Delta = -0.53", font_size=28, color=ORANGE)
        drop_arrow.next_to(score_box, RIGHT, buff=0.3)

        self.play(ReplacementTransform(score_text, new_score))
        self.play(FadeIn(drop_arrow, shift=LEFT * 0.3))
        self.play(Flash(new_score, color=RED, line_length=0.3, num_lines=14,
                        flash_radius=0.7), run_time=1.2)
        self.wait(0.8)

        # ---------- 5. The punchline ----------------------------------------
        punch = Text(
            "Same physics. Same crash. Only the camera angle changed.",
            font_size=26, color=YELLOW,
        ).move_to(rot_label)
        self.play(FadeOut(rot_label))
        self.play(Write(punch), run_time=1.6)
        self.wait(1.0)

        # Clear bottom panel, keep the road centred
        self.play(
            FadeOut(score_box), FadeOut(score_label),
            FadeOut(new_score), FadeOut(drop_arrow), FadeOut(car),
        )
        # Unrotate the road back to canonical position to host the equation
        self.play(
            Rotate(road_group, angle=-60 * DEGREES, about_point=ORIGIN),
            road_group.animate.scale(0.6).to_edge(UP, buff=1.2),
            FadeOut(punch),
            run_time=1.6,
        )

        # ---------- 6. The equivariance equation ----------------------------
        want = Text("What we want, by construction:",
                    font_size=26, color=BLUE_A).move_to([0, 0.7, 0])
        eq = MathTex(
            r"f(R\,\mathbf{r} + \mathbf{t}) \;=\; f(\mathbf{r})",
            font_size=64,
        ).move_to([0, -0.3, 0])
        eq.set_color_by_tex("R", YELLOW)
        eq.set_color_by_tex(r"\mathbf{t}", ORANGE)
        eq.set_color_by_tex("f", BLUE_B)

        cond = MathTex(
            r"\forall \, R \in SO(2),\ \mathbf{t} \in \mathbb{R}^2",
            font_size=32, color=GREY_A,
        ).next_to(eq, DOWN, buff=0.4)

        self.play(Write(want))
        self.play(Write(eq), run_time=1.6)
        self.play(FadeIn(cond, shift=UP * 0.2))
        self.wait(1.4)

        # Underline the key idea
        ul = Line(eq.get_corner(DL) + DOWN * 0.05,
                  eq.get_corner(DR) + DOWN * 0.05,
                  color=YELLOW, stroke_width=4)
        self.play(Create(ul), run_time=0.8)

        plan = Text(
            "Achieved by feeding the network only intrinsic geometry.",
            font_size=24, color=WHITE,
        ).next_to(cond, DOWN, buff=0.7)
        self.play(Write(plan))
        self.wait(2.0)

        # ---------- 7. Outro -------------------------------------------------
        self.play(FadeOut(VGroup(want, eq, cond, ul, plan, road_group)))
        end_card = Text(
            "Next: what the model sees.", font_size=34, color=WHITE,
        )
        self.play(FadeIn(end_card, shift=UP * 0.2))
        self.wait(1.4)
        self.play(FadeOut(end_card))
