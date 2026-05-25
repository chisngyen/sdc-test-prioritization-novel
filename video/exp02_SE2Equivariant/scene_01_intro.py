"""
Scene 01 -- The rotation problem.

Why rotation breaks a vanilla prioritizer, and the one-line equivariance
equation we want to satisfy.

Render:  manim -pql scene_01_intro.py Intro
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex,
    Rectangle, Triangle, Line,
    Write, FadeIn, FadeOut, Create, ReplacementTransform, Rotate,
    MoveAlongPath, LaggedStart,
    UP, DOWN, LEFT, RIGHT, ORIGIN, PI, DEGREES,
    WHITE, YELLOW, RED, RED_A, GREY_A, BLUE_A,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import sample_road, to_scene_coords, make_road
from theme import (
    TEXT, MUTED, PRIMARY, ACCENT, BAD, WARN, RULE,
    section_header, transition, hold,
    title, subtitle, footer, body_text,
    big_formula, body_formula, inline_math,
    panel, accent_box,
    attach_narration, seal_narration,
    MATH_BIG,
)


class Intro(Scene):
    def construct(self):
        attach_narration(self, "scene_01")
        self._title_card()
        self._the_test()
        self._rotate_and_break()
        self._equation()
        self._closer()
        seal_narration(self, "scene_01")

    # ----------------------------------------------------------- title -- #
    def _title_card(self):
        big = Text("SE(2)-Equivariant RoadNet",
                   font_size=58, color=TEXT, weight="BOLD")
        sub = Text("A prioritizer that is mathematically incapable",
                   font_size=24, color=PRIMARY).next_to(big, DOWN, buff=0.45)
        sub2 = Text("of caring about rotation.",
                    font_size=24, color=PRIMARY).next_to(sub, DOWN, buff=0.10)
        tag = Text("exps/exp02_SE2Equivariant.py",
                   font_size=18, color=MUTED, slant="ITALIC").next_to(sub2, DOWN, buff=0.55)
        block = VGroup(big, sub, sub2, tag).move_to(ORIGIN)

        self.play(Write(big), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.15),
                  FadeIn(sub2, shift=UP * 0.10), run_time=0.7)
        self.play(FadeIn(tag, shift=UP * 0.05), run_time=0.5)
        hold(self, 1.4)
        self.play(FadeOut(block), run_time=0.6)

    # --------------------------------------------------- a test = a road -- #
    def _the_test(self):
        header = section_header(
            self, "A test case is a road shape.",
            "We score it: higher number -> more likely to crash.",
        )

        # Build the road centered slightly low so the score panel fits below.
        # Use a small scale so the 60-degree rotation later does not push
        # the road into the title region.
        pts = sample_road(n=36) * 0.65
        road = make_road(pts, show_dots=False, stroke_width=8)
        road.move_to([0, 0.45, 0])
        self.road_group = road

        car = Triangle(color=ACCENT, fill_color=ACCENT, fill_opacity=1.0)
        car.scale(0.20).rotate(-PI / 2)
        car.move_to(road.points[0] if hasattr(road, "points") and len(road.points) > 0 else ORIGIN)

        self.play(Create(road), run_time=1.6)
        self.play(FadeIn(car, scale=0.7), run_time=0.5)
        # Drive the car along the road
        self.play(MoveAlongPath(car, road, rate_func=lambda t: t), run_time=2.8)
        self.car = car
        self.header = header

        # Show the score panel (predict FAIL probability)
        score_panel = panel(width=4.6, height=1.30, color=PRIMARY,
                            fill_opacity=0.18, stroke_width=2, rounded=True)
        score_panel.move_to([0, -2.65, 0])
        score_lbl = Text("baseline predicts:", font_size=22, color=PRIMARY)
        score_lbl.next_to(score_panel, UP, buff=0.18)
        score_val = MathTex(r"P(\text{FAIL}) \;=\; 0.85",
                            font_size=40, color=BAD)
        score_val.move_to(score_panel.get_center())

        self.play(FadeIn(score_panel), Write(score_lbl), run_time=0.7)
        self.play(Write(score_val), run_time=0.8)
        hold(self, 0.8)

        self.score_panel = score_panel
        self.score_lbl = score_lbl
        self.score_val = score_val

    # --------------------------------------------------- rotate the road -- #
    def _rotate_and_break(self):
        # Replace header
        new_header = section_header(self, "Same road. Rotated 60 degrees.",
                                    "Same physics. Same crash.")
        self.play(FadeOut(self.header), run_time=0.3)
        self.header = new_header

        rotation_grp = VGroup(self.road_group, self.car)
        self.play(Rotate(rotation_grp, angle=60 * DEGREES,
                         about_point=self.road_group.get_center()),
                  run_time=1.8)
        hold(self, 0.4)

        # New score, much worse
        new_val = MathTex(r"P(\text{FAIL}) \;=\; 0.32",
                          font_size=40, color=BAD).move_to(self.score_val.get_center())
        delta = MathTex(r"\Delta \;=\; -0.53",
                        font_size=26, color=WARN).next_to(self.score_panel, RIGHT, buff=0.35)

        self.play(ReplacementTransform(self.score_val, new_val), run_time=0.9)
        self.play(FadeIn(delta, shift=LEFT * 0.2), run_time=0.6)
        hold(self, 1.0)

        punch = body_text(
            "Only the camera angle changed.  Yet the score collapsed.",
            color=ACCENT,
        ).move_to([0, -1.30, 0])
        self.play(Write(punch), run_time=1.0)
        hold(self, 1.6)

        # tear down everything but the road
        self.play(
            FadeOut(self.score_panel),
            FadeOut(self.score_lbl),
            FadeOut(new_val),
            FadeOut(delta),
            FadeOut(self.car),
            FadeOut(punch),
            FadeOut(self.header),
            run_time=0.6,
        )
        # Reset road to canonical orientation, scale & move out of the way
        self.play(
            Rotate(self.road_group, angle=-60 * DEGREES,
                   about_point=self.road_group.get_center()),
            self.road_group.animate.scale(0.55).to_edge(UP, buff=1.10),
            run_time=1.2,
        )

    # ----------------------------------------------------- the equation -- #
    def _equation(self):
        prompt = body_text("What we want, by construction:", color=PRIMARY)
        prompt.move_to([0, 0.70, 0])

        eq = MathTex(
            r"f\!\bigl(R\,\mathbf{r} + \mathbf{t}\bigr) \;=\; f(\mathbf{r})",
            font_size=MATH_BIG + 12,
        ).move_to([0, -0.30, 0])
        eq.set_color_by_tex("R", ACCENT)
        eq.set_color_by_tex(r"\mathbf{t}", WARN)
        eq.set_color_by_tex("f", BLUE_A)

        cond = MathTex(
            r"\forall \, R \in SO(2),\ \mathbf{t} \in \mathbb{R}^2",
            font_size=28, color=MUTED,
        ).next_to(eq, DOWN, buff=0.45)

        self.play(Write(prompt), run_time=0.7)
        self.play(Write(eq), run_time=1.6)
        self.play(FadeIn(cond, shift=UP * 0.10), run_time=0.6)
        hold(self, 1.0)

        ul = Line(
            eq.get_corner(DOWN + LEFT) + DOWN * 0.12,
            eq.get_corner(DOWN + RIGHT) + DOWN * 0.12,
            color=ACCENT, stroke_width=4,
        )
        self.play(Create(ul), run_time=0.6)
        hold(self, 0.6)

        plan = body_text(
            "Achieved by feeding the network only intrinsic geometry.",
            color=TEXT,
        ).next_to(cond, DOWN, buff=0.75)
        self.play(Write(plan), run_time=1.0)
        hold(self, 2.0)

        self.outgoing = VGroup(prompt, eq, ul, cond, plan, self.road_group)

    # ---------------------------------------------------------- closer --- #
    def _closer(self):
        self.play(FadeOut(self.outgoing), run_time=0.6)
        end = Text("Next: what the model actually sees.",
                   font_size=32, color=TEXT)
        self.play(FadeIn(end, shift=UP * 0.15))
        hold(self, 1.5)
        self.play(FadeOut(end))
