"""
Scene 07 -- Results & take-aways.

The numbers from exps/tracker.md for Exp 02, presented as bar charts:
    1.  Rotation-invariance probe: APFD at 6 rotation angles
        (SE2RoadNet is flat; baseline 10-ch drops 4-7 points).
    2.  Delta column from the probe -- all zero for the equivariant tower.
    3.  Final scoreboard tile.

Render:
    manim -pql scene_07_results.py Results
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Circle, Dot, Line, Arrow,
    DashedLine, Axes, BarChart,
    Write, FadeIn, FadeOut, Create, ReplacementTransform,
    Indicate, Flash, LaggedStart, Transform, GrowArrow, GrowFromEdge,
    ValueTracker, always_redraw, DecimalNumber,
    UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL, ORIGIN, PI, DEGREES,
    WHITE, BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E,
    YELLOW, YELLOW_A, ORANGE, RED, RED_A, RED_E,
    GREEN, GREEN_A, GREY, GREY_A, GREY_B, GREY_C, GOLD, PINK, MAROON, TEAL,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))


# Numbers reported in exps/tracker.md for Exp 02.
ROT_DEGS = [0, 30, 60, 90, 180, -45]
APFD_SE2 = [0.8047, 0.8047, 0.8047, 0.8047, 0.8047, 0.8047]   # all equal -> equivariance
APFD_BASE = [0.8066, 0.7783, 0.7493, 0.7218, 0.7611, 0.7549]  # baseline drops 4-7 pts

# Headline scoreboard -- Exp 02 specific numbers.
SCOREBOARD = [
    (r"SE2 APFD-comp (single-pass, any rotation)",      r"0.8047",            BLUE_A),
    (r"SE2 APFD-comp (multi-trial, 30 rolls)",          r"0.8048 \pm 0.0118", BLUE_A),
    (r"SE2 AUC@SensoDat-test",                          r"0.9347",            GREEN_A),
    (r"Rotation probe $\Delta$ APFD (6 angles)",        r"0.0000",            YELLOW),
    (r"Baseline drop under same rotations",             r"-0.04\,\text{to}\,-0.08", RED_A),
    (r"Parameters",                                     r"2{,}108{,}721",     GREY_A),
]


class Results(Scene):
    def construct(self):
        # ---------- title --------------------------------------------------- #
        title = Text("Did it work?",
                     font_size=40, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.4)

        # ---------- chart 1: rotation probe --------------------------------- #
        sub1 = Text("Rotation-invariance probe (Competition set, 6 angles)",
                    font_size=24, color=BLUE_A).next_to(title, DOWN, buff=0.2)
        self.play(Write(sub1))

        ax = Axes(
            x_range=[-0.5, len(ROT_DEGS) - 0.5, 1],
            y_range=[0.70, 0.83, 0.02],
            x_length=8.5, y_length=3.4,
            tips=False,
            axis_config={"include_numbers": False, "stroke_color": GREY_A},
            y_axis_config={"include_numbers": True},
        ).shift(DOWN * 0.4)

        # x-tick labels: rotation angles
        x_tick_labels = VGroup()
        for k, deg in enumerate(ROT_DEGS):
            lbl = MathTex(f"{deg:+d}^\\circ" if deg != 0 else r"0^\circ",
                          font_size=20, color=GREY_A)
            lbl.next_to(ax.c2p(k, 0.70), DOWN, buff=0.18)
            x_tick_labels.add(lbl)

        y_lbl = MathTex(r"\text{APFD}", font_size=24,
                        color=GREY_A).next_to(ax, LEFT, buff=0.05).shift(UP * 1.4)

        self.play(Create(ax), FadeIn(x_tick_labels), Write(y_lbl))

        # baseline bars (red), SE2 bars (yellow), grouped per angle
        bar_width = 0.32
        base_bars = VGroup()
        se2_bars  = VGroup()
        for k, (b, s) in enumerate(zip(APFD_BASE, APFD_SE2)):
            # baseline
            b_h = b - 0.70
            b_bottom = ax.c2p(k - 0.18, 0.70)
            b_top    = ax.c2p(k - 0.18, b)
            base_bar = Rectangle(
                width=bar_width * 0.4, height=(b_top - b_bottom)[1],
                stroke_width=0, fill_color=RED_A, fill_opacity=0.85,
            ).move_to((b_bottom + b_top) / 2)
            base_bars.add(base_bar)

            # SE2
            s_bottom = ax.c2p(k + 0.18, 0.70)
            s_top    = ax.c2p(k + 0.18, s)
            se2_bar = Rectangle(
                width=bar_width * 0.4, height=(s_top - s_bottom)[1],
                stroke_width=0, fill_color=YELLOW, fill_opacity=0.95,
            ).move_to((s_bottom + s_top) / 2)
            se2_bars.add(se2_bar)

        # legend
        legend = VGroup(
            VGroup(
                Square(side_length=0.25, color=RED_A,
                       fill_color=RED_A, fill_opacity=0.85, stroke_width=0),
                Text("baseline (10-ch)", font_size=22, color=RED_A),
            ).arrange(RIGHT, buff=0.18),
            VGroup(
                Square(side_length=0.25, color=YELLOW,
                       fill_color=YELLOW, fill_opacity=0.95, stroke_width=0),
                Text("SE2-equivariant (ours)", font_size=22, color=YELLOW),
            ).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        legend.to_corner(UR, buff=0.6).shift(DOWN * 1.0)

        self.play(FadeIn(legend, shift=LEFT * 0.2))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in base_bars],
                              lag_ratio=0.05, run_time=1.4))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in se2_bars],
                              lag_ratio=0.05, run_time=1.4))
        self.wait(0.4)

        # connecting flat line to emphasise equivariance
        flat = DashedLine(
            ax.c2p(-0.4, APFD_SE2[0]), ax.c2p(len(ROT_DEGS) - 0.6, APFD_SE2[0]),
            color=YELLOW, stroke_width=2,
        )
        self.play(Create(flat), run_time=0.8)

        verdict_1 = Text(
            "SE2 bars are identical to 4 decimal places. The line is flat.",
            font_size=22, color=YELLOW,
        ).next_to(ax, DOWN, buff=0.95)
        self.play(Write(verdict_1))
        self.wait(1.6)

        # ---------- chart 2: rotation Delta -------------------------------- #
        self.play(FadeOut(VGroup(sub1, ax, x_tick_labels, y_lbl,
                                 base_bars, se2_bars, flat, legend, verdict_1)))

        sub2 = Text(r"|APFD(rotated) - APFD(0°)|  across the same six angles",
                    font_size=24, color=BLUE_A).next_to(title, DOWN, buff=0.2)
        self.play(Write(sub2))

        delta_base = [abs(b - APFD_BASE[0]) for b in APFD_BASE]
        delta_se2  = [abs(s - APFD_SE2[0])  for s in APFD_SE2]

        ax2 = Axes(
            x_range=[-0.5, len(ROT_DEGS) - 0.5, 1],
            y_range=[0.0, 0.10, 0.02],
            x_length=8.5, y_length=3.4, tips=False,
            axis_config={"include_numbers": False, "stroke_color": GREY_A},
            y_axis_config={"include_numbers": True},
        ).shift(DOWN * 0.4)
        x_tick_labels2 = VGroup()
        for k, deg in enumerate(ROT_DEGS):
            lbl = MathTex(f"{deg:+d}^\\circ" if deg != 0 else r"0^\circ",
                          font_size=20, color=GREY_A)
            lbl.next_to(ax2.c2p(k, 0.0), DOWN, buff=0.18)
            x_tick_labels2.add(lbl)
        self.play(Create(ax2), FadeIn(x_tick_labels2))

        base_d_bars = VGroup()
        se2_d_bars  = VGroup()
        for k, (b, s) in enumerate(zip(delta_base, delta_se2)):
            b_bot = ax2.c2p(k - 0.18, 0.0); b_top = ax2.c2p(k - 0.18, max(b, 1e-3))
            base_d_bars.add(Rectangle(
                width=0.16, height=(b_top - b_bot)[1],
                stroke_width=0, fill_color=RED_A, fill_opacity=0.85,
            ).move_to((b_bot + b_top) / 2))
            s_bot = ax2.c2p(k + 0.18, 0.0); s_top = ax2.c2p(k + 0.18, max(s, 1e-3))
            se2_d_bars.add(Rectangle(
                width=0.16, height=(s_top - s_bot)[1],
                stroke_width=0, fill_color=YELLOW, fill_opacity=0.95,
            ).move_to((s_bot + s_top) / 2))

        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in base_d_bars],
                              lag_ratio=0.05, run_time=1.0))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in se2_d_bars],
                              lag_ratio=0.05, run_time=1.0))

        zero_label = MathTex(r"\Delta = 0.0000", font_size=36, color=YELLOW)
        zero_label.next_to(ax2, DOWN, buff=0.9)
        self.play(Write(zero_label))
        self.wait(1.6)

        self.play(FadeOut(VGroup(sub2, ax2, x_tick_labels2,
                                 base_d_bars, se2_d_bars, zero_label)))

        # ---------- scoreboard summary tiles ------------------------------- #
        sub3 = Text("The headline scoreboard",
                    font_size=24, color=BLUE_A).next_to(title, DOWN, buff=0.2)
        self.play(Write(sub3))

        tiles = VGroup()
        for label_str, val_str, col in SCOREBOARD:
            tile = RoundedRectangle(
                width=7.4, height=0.85, corner_radius=0.18,
                stroke_color=col, stroke_width=2.5,
                fill_color=col, fill_opacity=0.10,
            )
            lab = Text(label_str, font_size=22, color=WHITE)
            lab.move_to(tile.get_left() + RIGHT * 0.4, aligned_edge=LEFT)
            val = MathTex(val_str, font_size=28, color=col)
            val.move_to(tile.get_right() + LEFT * 0.4, aligned_edge=RIGHT)
            tiles.add(VGroup(tile, lab, val))
        tiles.arrange(DOWN, buff=0.18).shift(DOWN * 0.15)

        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in tiles],
                              lag_ratio=0.12, run_time=2.4))
        self.wait(2.0)

        # ---------- closing ------------------------------------------------- #
        self.play(FadeOut(VGroup(title, sub3, tiles)))

        outro_title = Text(
            "SE(2)-Equivariant RoadNet",
            font_size=44, color=WHITE, weight="BOLD",
        )
        outro_sub = Text(
            "Rotation invariance, by construction. Eight benchmarks, one recipe.",
            font_size=24, color=BLUE_A,
        ).next_to(outro_title, DOWN, buff=0.35)
        cite = Text(
            "exps/exp02_SE2Equivariant.py  —  ICSE 2027 (under preparation)",
            font_size=18, color=GREY_A, slant="ITALIC",
        ).next_to(outro_sub, DOWN, buff=0.7)

        self.play(Write(outro_title), run_time=1.4)
        self.play(FadeIn(outro_sub, shift=UP * 0.2))
        self.play(FadeIn(cite))
        self.wait(2.4)
        self.play(FadeOut(VGroup(outro_title, outro_sub, cite)))


