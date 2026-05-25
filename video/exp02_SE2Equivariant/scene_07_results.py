"""
Scene 07 -- Results and take-aways.

Numbers from exps/tracker.md for Exp 02, in three beats:
    1.  Rotation-invariance probe: APFD at 6 angles (SE2 flat; baseline drops).
    2.  |Delta| column: all zero for the equivariant tower.
    3.  Scoreboard summary tiles + closing title.

Render:  manim -pql scene_07_results.py Results
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Line, DashedLine, Axes,
    Write, FadeIn, FadeOut, Create, LaggedStart, GrowFromEdge,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, BLUE_A, YELLOW, GREEN_A, RED_A, GREY_A, PINK, TEAL, ORANGE,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import (
    TEXT, MUTED, PRIMARY, ACCENT, GOOD, WARN, BAD, RULE,
    section_header, transition, hold,
    title, subtitle, body_text, caption,
    inline_math, value_card,
    panel, accent_box,
    attach_narration, seal_narration,
    MATH_INLINE, MATH_BIG,
)


ROT_DEGS  = [0, 30, 60, 90, 180, -45]
APFD_SE2  = [0.8047, 0.8047, 0.8047, 0.8047, 0.8047, 0.8047]
APFD_BASE = [0.8066, 0.7783, 0.7493, 0.7218, 0.7611, 0.7549]

SCOREBOARD = [
    (r"SE2 APFD-comp (single-pass, any rotation)",     r"0.8047",                  PRIMARY),
    (r"SE2 APFD-comp (multi-trial, 30 rolls)",         r"0.8048 \pm 0.0118",       PRIMARY),
    (r"SE2 AUC @ SensoDat-test",                       r"0.9347",                  GOOD),
    (r"Rotation probe $\Delta$ APFD (6 angles)",       r"0.0000",                  ACCENT),
    (r"Baseline drop under same rotations",            r"-0.04\text{ to }-0.08",   BAD),
    (r"Parameters",                                    r"2{,}108{,}721",           MUTED),
]


class Results(Scene):
    def construct(self):
        attach_narration(self, "scene_07")
        self._title()
        self._chart_apfd()
        self._chart_delta()
        self._scoreboard()
        self._closing()
        seal_narration(self, "scene_07")

    # ---------------------------------------------------- a. title -------
    def _title(self):
        head = title("Did it work?")
        ul = Line(
            head.get_corner(DOWN + LEFT) + DOWN * 0.10,
            head.get_corner(DOWN + RIGHT) + DOWN * 0.10,
            color=PRIMARY, stroke_width=2,
        )
        self.play(Write(head), Create(ul), run_time=0.9)
        self.persistent_title = VGroup(head, ul)
        hold(self, 0.3)

    # ----------------------------------- b. APFD bar chart per angle -----
    def _chart_apfd(self):
        sub = Text(
            "Rotation-invariance probe -- APFD at 6 angles, Competition test set.",
            font_size=24, color=PRIMARY,
        ).move_to([0, 2.60, 0])
        self.play(Write(sub), run_time=0.7)

        ax = Axes(
            x_range=[-0.5, len(ROT_DEGS) - 0.5, 1],
            y_range=[0.70, 0.83, 0.02],
            x_length=9.0, y_length=3.4,
            tips=False,
            axis_config={"include_numbers": False, "stroke_color": GREY_A},
            y_axis_config={"include_numbers": True},
        ).shift(DOWN * 0.30)

        x_tick_labels = VGroup()
        for k, deg in enumerate(ROT_DEGS):
            lbl = MathTex(f"{deg:+d}^\\circ" if deg != 0 else r"0^\circ",
                          font_size=20, color=MUTED)
            lbl.next_to(ax.c2p(k, 0.70), DOWN, buff=0.15)
            x_tick_labels.add(lbl)

        y_lbl = MathTex(r"\mathrm{APFD}", font_size=22,
                        color=MUTED).next_to(ax, LEFT, buff=0.10).shift(UP * 1.35)

        self.play(Create(ax), FadeIn(x_tick_labels), Write(y_lbl), run_time=1.0)

        base_bars = VGroup()
        se2_bars  = VGroup()
        bw = 0.30
        for k, (b, s) in enumerate(zip(APFD_BASE, APFD_SE2)):
            b_bottom = ax.c2p(k - 0.18, 0.70)
            b_top    = ax.c2p(k - 0.18, b)
            base_bars.add(Rectangle(
                width=bw, height=(b_top - b_bottom)[1],
                stroke_width=0, fill_color=BAD, fill_opacity=0.85,
            ).move_to((b_bottom + b_top) / 2))

            s_bottom = ax.c2p(k + 0.18, 0.70)
            s_top    = ax.c2p(k + 0.18, s)
            se2_bars.add(Rectangle(
                width=bw, height=(s_top - s_bottom)[1],
                stroke_width=0, fill_color=ACCENT, fill_opacity=0.95,
            ).move_to((s_bottom + s_top) / 2))

        legend = VGroup(
            VGroup(
                Square(side_length=0.25, color=BAD, fill_color=BAD,
                       fill_opacity=0.85, stroke_width=0),
                Text("baseline (10-ch)", font_size=20, color=BAD),
            ).arrange(RIGHT, buff=0.15),
            VGroup(
                Square(side_length=0.25, color=ACCENT, fill_color=ACCENT,
                       fill_opacity=0.95, stroke_width=0),
                Text("SE(2)-equivariant (ours)", font_size=20, color=ACCENT),
            ).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        legend.to_corner(UP + RIGHT, buff=0.55).shift(DOWN * 1.0)

        self.play(FadeIn(legend, shift=LEFT * 0.15), run_time=0.5)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in base_bars],
                              lag_ratio=0.05, run_time=1.2))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in se2_bars],
                              lag_ratio=0.05, run_time=1.2))

        flat = DashedLine(
            ax.c2p(-0.4, APFD_SE2[0]),
            ax.c2p(len(ROT_DEGS) - 0.6, APFD_SE2[0]),
            color=ACCENT, stroke_width=2,
        )
        self.play(Create(flat), run_time=0.7)

        verdict = Text(
            "SE2 bars are identical to 4 decimal places. The line is flat.",
            font_size=24, color=ACCENT,
        ).move_to([0, -2.95, 0])
        self.play(Write(verdict), run_time=0.9)
        hold(self, 2.0)

        self.play(FadeOut(VGroup(sub, ax, x_tick_labels, y_lbl,
                                 base_bars, se2_bars, flat, legend, verdict)),
                  run_time=0.6)

    # ----------------------------------- c. delta bar chart --------------
    def _chart_delta(self):
        sub = Tex(
            r"$|\mathrm{APFD}(\text{rotated}) - \mathrm{APFD}(0^\circ)|$  across the six angles.",
            color=PRIMARY,
        ).scale_to_fit_height(0.32).move_to([0, 2.60, 0])
        self.play(Write(sub), run_time=0.7)

        delta_base = [abs(b - APFD_BASE[0]) for b in APFD_BASE]
        delta_se2  = [abs(s - APFD_SE2[0])  for s in APFD_SE2]

        ax2 = Axes(
            x_range=[-0.5, len(ROT_DEGS) - 0.5, 1],
            y_range=[0.0, 0.10, 0.02],
            x_length=9.0, y_length=3.4, tips=False,
            axis_config={"include_numbers": False, "stroke_color": GREY_A},
            y_axis_config={"include_numbers": True},
        ).shift(DOWN * 0.30)

        x_tick_labels = VGroup()
        for k, deg in enumerate(ROT_DEGS):
            lbl = MathTex(f"{deg:+d}^\\circ" if deg != 0 else r"0^\circ",
                          font_size=20, color=MUTED)
            lbl.next_to(ax2.c2p(k, 0.0), DOWN, buff=0.15)
            x_tick_labels.add(lbl)
        self.play(Create(ax2), FadeIn(x_tick_labels), run_time=0.8)

        base_d_bars = VGroup()
        se2_d_bars  = VGroup()
        for k, (b, s) in enumerate(zip(delta_base, delta_se2)):
            b_bot = ax2.c2p(k - 0.18, 0.0); b_top = ax2.c2p(k - 0.18, max(b, 1e-3))
            base_d_bars.add(Rectangle(
                width=0.18, height=(b_top - b_bot)[1],
                stroke_width=0, fill_color=BAD, fill_opacity=0.85,
            ).move_to((b_bot + b_top) / 2))
            s_bot = ax2.c2p(k + 0.18, 0.0); s_top = ax2.c2p(k + 0.18, max(s, 1e-3))
            se2_d_bars.add(Rectangle(
                width=0.18, height=(s_top - s_bot)[1],
                stroke_width=0, fill_color=ACCENT, fill_opacity=0.95,
            ).move_to((s_bot + s_top) / 2))

        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in base_d_bars],
                              lag_ratio=0.05, run_time=1.0))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in se2_d_bars],
                              lag_ratio=0.05, run_time=1.0))

        zero_label = MathTex(r"\Delta = 0.0000", font_size=40, color=ACCENT)
        zero_label.move_to([0, -2.95, 0])
        self.play(Write(zero_label), run_time=0.7)
        hold(self, 2.0)

        self.play(FadeOut(VGroup(sub, ax2, x_tick_labels,
                                 base_d_bars, se2_d_bars, zero_label)),
                  run_time=0.6)

    # ---------------------------------- d. scoreboard tiles --------------
    def _scoreboard(self):
        sub = Text("Headline scoreboard.", font_size=24,
                   color=PRIMARY).move_to([0, 2.60, 0])
        self.play(Write(sub), run_time=0.6)

        tiles = VGroup()
        for label_str, val_str, col in SCOREBOARD:
            tile = value_card(label_str, val_str, color=col,
                              width=8.8, height=0.78,
                              label_size=22, value_size=26,
                              value_is_math=True)
            tiles.add(tile)
        tiles.arrange(DOWN, buff=0.16).move_to([0, -0.30, 0])

        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.10) for t in tiles],
                              lag_ratio=0.10, run_time=2.2))
        hold(self, 2.4)

        self.play(FadeOut(VGroup(sub, tiles)), run_time=0.5)

    # ----------------------------------- e. closing ----------------------
    def _closing(self):
        self.play(FadeOut(self.persistent_title), run_time=0.5)

        big = Text("SE(2)-Equivariant RoadNet",
                   font_size=52, color=TEXT, weight="BOLD")
        sub = Text(
            "Rotation invariance, by construction.",
            font_size=26, color=PRIMARY,
        ).next_to(big, DOWN, buff=0.30)
        sub2 = Text(
            "Eight benchmarks, one recipe.",
            font_size=26, color=PRIMARY,
        ).next_to(sub, DOWN, buff=0.10)
        cite = Text(
            "exps/exp02_SE2Equivariant.py  --  ICSE 2027 (in preparation)",
            font_size=18, color=MUTED, slant="ITALIC",
        ).next_to(sub2, DOWN, buff=0.65)

        block = VGroup(big, sub, sub2, cite).move_to(ORIGIN)
        rule_top = Line(
            big.get_corner(UP + LEFT) + UP * 0.35,
            big.get_corner(UP + RIGHT) + UP * 0.35,
            color=PRIMARY, stroke_width=2,
        )
        rule_bot = Line(
            sub2.get_corner(DOWN + LEFT) + DOWN * 0.22,
            sub2.get_corner(DOWN + RIGHT) + DOWN * 0.22,
            color=PRIMARY, stroke_width=2,
        )

        self.play(Write(big), run_time=1.2)
        self.play(Create(rule_top), Create(rule_bot), run_time=0.7)
        self.play(FadeIn(sub, shift=UP * 0.10), FadeIn(sub2, shift=UP * 0.10),
                  run_time=0.7)
        self.play(FadeIn(cite, shift=UP * 0.05), run_time=0.5)
        hold(self, 2.6)
        self.play(FadeOut(VGroup(block, rule_top, rule_bot)), run_time=0.6)
