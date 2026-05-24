"""
Scene 00 -- Project context.

Before we dive into Exp 02 specifically, frame the whole problem:
    - what is SDC test prioritization?
    - what is APFD?
    - where does Exp 02 sit in the project's experiment ladder?

Render:
    manim -pql scene_00_context.py Context
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Circle, Dot, Line, Arrow,
    DashedLine, Triangle,
    Write, FadeIn, FadeOut, Create, ReplacementTransform,
    Indicate, Flash, LaggedStart, Transform, GrowFromEdge,
    ValueTracker, always_redraw, DecimalNumber,
    UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL, ORIGIN, PI, DEGREES,
    WHITE, BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E,
    YELLOW, YELLOW_A, ORANGE, RED, RED_A,
    GREEN, GREEN_A, GREY, GREY_A, GREY_B, GREY_C, GOLD, PINK, MAROON, TEAL,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from layout import title, subtitle, footer, clear, replace_title
from common import sample_road, to_scene_coords, ROAD_COLOR


class Context(Scene):
    def construct(self):
        self._part_a_title()
        self._part_b_the_problem()
        self._part_c_apfd_metric()
        self._part_d_project_map()
        self._part_e_today()

    # ------------------------------------------------------------- a. title
    def _part_a_title(self):
        big = Text("Self-Driving Car", font_size=60, color=WHITE, weight="BOLD")
        big.move_to([0, 0.7, 0])
        sub = Text("Test Prioritization", font_size=60, color=BLUE_A, weight="BOLD")
        sub.next_to(big, DOWN, buff=0.2)
        tag = Text("Why rotation should not change your prediction",
                   font_size=22, color=GREY_A, slant="ITALIC").next_to(sub, DOWN, buff=0.6)
        self.play(Write(big), run_time=1.1)
        self.play(Write(sub), run_time=1.0)
        self.play(FadeIn(tag, shift=UP * 0.15))
        self.wait(1.2)
        self.play(FadeOut(big), FadeOut(sub), FadeOut(tag))

    # ----------------------------------------------------- b. the problem
    def _part_b_the_problem(self):
        t = title("The setup")
        s = subtitle("Self-driving simulators run tests; some crash, most don't.")
        self.play(Write(t), FadeIn(s, shift=UP * 0.15))

        # A grid of road thumbnails: green = pass, red = fail
        rng = np.random.default_rng(0)
        n_cols, n_rows = 8, 3
        tiles = VGroup()
        for r in range(n_rows):
            for c in range(n_cols):
                pts = sample_road(n=12) * 0.12
                pts[:, 1] += rng.normal(scale=0.04, size=pts.shape[0])
                road = VMobject(stroke_color=ROAD_COLOR, stroke_width=2)
                road.set_points_smoothly(to_scene_coords(pts))
                # Frame around the road
                fail = rng.random() < 0.30
                frame_col = RED if fail else GREEN_A
                frame = Rectangle(width=0.95, height=0.55,
                                  stroke_color=frame_col, stroke_width=2,
                                  fill_color=frame_col, fill_opacity=0.10)
                road.move_to(frame.get_center())
                tile = VGroup(frame, road)
                tiles.add(tile)
        tiles.arrange_in_grid(n_rows, n_cols, buff=0.18)
        tiles.move_to([0, 0.2, 0]).scale_to_fit_width(11.0)
        self.play(LaggedStart(*[FadeIn(t, scale=1.05) for t in tiles],
                              lag_ratio=0.02, run_time=1.6))
        self.wait(0.3)

        # Legend
        leg_fail = VGroup(
            Rectangle(width=0.4, height=0.25, stroke_color=RED, stroke_width=2,
                      fill_color=RED, fill_opacity=0.15),
            Text("crash (FAIL)", font_size=20, color=RED),
        ).arrange(RIGHT, buff=0.18)
        leg_pass = VGroup(
            Rectangle(width=0.4, height=0.25, stroke_color=GREEN_A, stroke_width=2,
                      fill_color=GREEN_A, fill_opacity=0.15),
            Text("safe (PASS)", font_size=20, color=GREEN_A),
        ).arrange(RIGHT, buff=0.18)
        legend = VGroup(leg_fail, leg_pass).arrange(RIGHT, buff=1.0)
        legend.move_to([0, -2.2, 0])
        self.play(FadeIn(legend, shift=UP * 0.2))

        msg = footer("Out of ~10K tests, only ~30% crash. We want the crashes to surface first.")
        self.play(Write(msg))
        self.wait(1.8)

        clear(self)

    # ----------------------------------------------------- c. APFD metric
    def _part_c_apfd_metric(self):
        t = title("The metric: APFD")
        self.play(Write(t))

        # Formula
        eq = MathTex(
            r"\mathrm{APFD} \;=\; 1 \;-\; "
            r"\frac{\sum_{i=1}^{m} \mathrm{pos}(f_i)}{n \cdot m} "
            r"\;+\; \frac{1}{2n}",
            font_size=40,
        ).move_to([0, 2.05, 0])
        eq.set_color_by_tex("APFD", YELLOW)
        eq.set_color_by_tex("pos", RED)
        self.play(Write(eq), run_time=2.0)

        # Two ranked queues: "good" and "bad"
        n_tests = 12
        rng = np.random.default_rng(3)
        fails_good = [1, 2, 4]                # FAILs near front
        fails_bad  = [8, 10, 11]              # FAILs near back

        def make_queue(fail_positions, y, label_text, label_color):
            row = VGroup()
            for i in range(n_tests):
                is_fail = i in fail_positions
                col = RED if is_fail else GREEN_A
                cell = Square(side_length=0.45, stroke_color=col, stroke_width=2,
                              fill_color=col, fill_opacity=0.45)
                num = Text(str(i + 1), font_size=14, color=WHITE)
                num.move_to(cell.get_center())
                row.add(VGroup(cell, num))
            row.arrange(RIGHT, buff=0.08)
            row.move_to([0, y, 0])
            lbl = Text(label_text, font_size=20, color=label_color).next_to(
                row, LEFT, buff=0.35
            )
            return VGroup(row, lbl)

        good = make_queue(fails_good, 0.30, "good ranking", YELLOW)
        bad  = make_queue(fails_bad, -1.10, "bad ranking",  GREY_A)

        # Compute APFD values for the two queues (m=3 FAILs, n=12)
        def apfd(positions, n=12):
            return 1 - sum(p + 1 for p in positions) / (n * len(positions)) + 1 / (2 * n)

        apfd_good = apfd(fails_good)
        apfd_bad  = apfd(fails_bad)

        good_val = MathTex(rf"\mathrm{{APFD}} = {apfd_good:.3f}",
                           font_size=28, color=YELLOW).next_to(good, RIGHT, buff=0.35)
        bad_val  = MathTex(rf"\mathrm{{APFD}} = {apfd_bad:.3f}",
                           font_size=28, color=RED_A).next_to(bad, RIGHT, buff=0.35)

        self.play(FadeIn(good, shift=RIGHT * 0.15), run_time=0.8)
        self.play(Write(good_val))
        self.play(FadeIn(bad, shift=RIGHT * 0.15), run_time=0.8)
        self.play(Write(bad_val))
        self.wait(0.5)

        msg = footer("FAILs at the front -> higher APFD. We are optimising this number.")
        self.play(Write(msg))
        self.wait(2.0)

        clear(self)

    # ----------------------------------------------------- d. project map
    def _part_d_project_map(self):
        t = title("The project: 1 baseline + 14 experiments")
        self.play(Write(t))

        # Build a vertical ladder of cards.  Each card: experiment label + APFD.
        items = [
            ("00", "Baseline (Transformer + SWA + Focal)", 0.8077, "ensemble", BLUE_A),
            ("01", "FNO Roads — resolution invariance",    0.8067, r"$\Delta_N \!=\! 0.001$", TEAL),
            ("02", "SE(2)-Equivariant RoadNet  (today!)",   0.8048, r"$\Delta_{\mathrm{rot}}\!=\!0.0000$, AUC 0.9347", YELLOW),
            ("03", "Differentiable APFD (listwise)",        0.8057, r"$\sigma$=0.0109 (lowest)", BLUE_A),
            ("04", "PINN -- curvature monotonicity",        0.8055, r"viol. 17.6\% $\to$ 3.1\% (5.6$\times$)", ORANGE),
            ("10", "DiffAPFD on SE(2) backbone",            0.8049, "AUC 0.9385 (highest)", PINK),
        ]
        cards = VGroup()
        for idx, name, apfd_val, note, col in items:
            box = RoundedRectangle(
                width=10.2, height=0.55, corner_radius=0.10,
                stroke_color=col, stroke_width=2.5,
                fill_color=col, fill_opacity=0.10,
            )
            tag = Text(f"#{idx}", font_size=18, color=col, weight="BOLD")
            tag.move_to(box.get_left() + RIGHT * 0.45)
            label = Text(name, font_size=18, color=WHITE)
            label.move_to(box.get_left() + RIGHT * 1.50, aligned_edge=LEFT)
            val = MathTex(rf"\mathrm{{APFD}}={apfd_val:.4f}",
                          font_size=18, color=col)
            val.move_to(box.get_right() + LEFT * 2.45, aligned_edge=RIGHT)
            n_obj = Tex(note, font_size=16, color=GREY_A)
            n_obj.move_to(box.get_right() + LEFT * 0.25, aligned_edge=RIGHT)
            cards.add(VGroup(box, tag, label, val, n_obj))
        cards.arrange(DOWN, buff=0.12).move_to([0, -0.2, 0])

        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.08) for c in cards],
                              lag_ratio=0.12, run_time=2.4))

        # Highlight Exp 02
        target = cards[2]
        ring = Rectangle(
            width=target[0].width + 0.15, height=target[0].height + 0.15,
            stroke_color=YELLOW, stroke_width=4, fill_opacity=0,
        ).move_to(target.get_center())
        self.play(Create(ring), run_time=0.8)
        self.wait(1.6)

        msg = footer("This video walks through #02 from input road to final score.")
        self.play(Write(msg))
        self.wait(2.0)

        clear(self)

    # ----------------------------------------------------- e. today
    def _part_e_today(self):
        t = title("Today: Exp 02 — SE(2)-Equivariant RoadNet")
        self.play(Write(t))

        bullets_text = [
            ("1. Input",     "raw road points (L, 2)",                   BLUE_A),
            ("2. Features",  "7 channels of intrinsic geometry",         ORANGE),
            ("3. Invariance","prove rotation cannot change the input",   GREEN_A),
            ("4. Model",     "SE2RoadNet: 6 InvariantBlocks + CLS head", BLUE_C),
            ("5. Compute",   "watch a tensor flow through the network",  YELLOW),
            ("6. Results",   r"$\Delta\,\mathrm{APFD}_{\mathrm{rot}}\!=\!0.0000$, AUC $0.9347$", PINK),
        ]
        rows = VGroup()
        for tag, txt, col in bullets_text:
            tag_t = Text(tag, font_size=24, color=col, weight="BOLD")
            if "Delta" in txt or "0.0000" in txt or "AUC" in txt:
                body = Tex(txt, font_size=24, color=WHITE)
            else:
                body = Text(txt, font_size=22, color=WHITE)
            row = VGroup(tag_t, body).arrange(RIGHT, buff=0.5, aligned_edge=DOWN)
            rows.add(row)
        rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to([0, -0.1, 0])

        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.08) for r in rows],
                              lag_ratio=0.18, run_time=2.6))
        self.wait(2.2)

        clear(self)
        end = Text("Let's begin.", font_size=40, color=YELLOW).move_to(ORIGIN)
        self.play(FadeIn(end, shift=UP * 0.2))
        self.wait(1.2)
        self.play(FadeOut(end))
