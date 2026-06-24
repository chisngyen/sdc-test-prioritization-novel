"""
Scene 00 -- Project context.

Frames the problem before we dive into Exp 02:
    a. title splash
    b. what SDC test prioritization is, why it's a sorting problem
    c. APFD -- the single metric we are optimising
    d. the project's experiment ladder, with Exp 02 highlighted
    e. what this video will walk through

Render:  manim -pql scene_00_context.py Context
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Dot, Line,
    Write, FadeIn, FadeOut, Create, LaggedStart,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, BLUE_A, BLUE_B, YELLOW, GREEN_A, RED, GREY_A, ORANGE,
    PINK, TEAL,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import sample_road, to_scene_coords
from theme import (
    TEXT, MUTED, PRIMARY, ACCENT, GOOD, WARN, BAD, RULE,
    section_header, swap_header, transition, hold,
    title, subtitle, footer, body_text, caption,
    big_formula, body_formula, inline_math, value_card,
    accent_box, divider,
    attach_narration, seal_narration,
    TITLE_FS, BODY_FS, MATH_BODY, MATH_INLINE,
)


class Context(Scene):
    def construct(self):
        attach_narration(self, "scene_00")
        self._splash()
        self._setup()
        self._metric()
        self._project_map()
        self._today()
        # Hold "Let's begin." through the narration tail instead of cutting to black.
        seal_narration(self, "scene_00")
        self.play(FadeOut(self.opener), run_time=0.5)

    # ----------------------------------------------------------- a. splash --
    def _splash(self):
        big = Text("Self-Driving Car", font_size=64, color=TEXT, weight="BOLD")
        sub = Text("Test Prioritization", font_size=64, color=ACCENT, weight="BOLD")
        sub.next_to(big, DOWN, buff=0.25)
        tag = Text(
            "When rotating the road must not change the prediction.",
            font_size=24, color=MUTED, slant="ITALIC",
        ).next_to(sub, DOWN, buff=0.55)

        block = VGroup(big, sub, tag).move_to(ORIGIN)
        rule_top = Line(
            big.get_corner(UP + LEFT) + UP * 0.35,
            big.get_corner(UP + RIGHT) + UP * 0.35,
            color=PRIMARY, stroke_width=2,
        )
        rule_bot = Line(
            sub.get_corner(DOWN + LEFT) + DOWN * 0.20,
            sub.get_corner(DOWN + RIGHT) + DOWN * 0.20,
            color=PRIMARY, stroke_width=2,
        )

        self.play(Write(big), run_time=1.0)
        self.play(Write(sub), run_time=0.9)
        self.play(Create(rule_top), Create(rule_bot), run_time=0.7)
        self.play(FadeIn(tag, shift=UP * 0.1), run_time=0.6)
        hold(self, 1.4)
        self.play(FadeOut(VGroup(big, sub, tag, rule_top, rule_bot)), run_time=0.6)

    # ------------------------------------------------------------ b. setup --
    def _setup(self):
        header = section_header(
            self, "The setup",
            "Simulators run ~10k road tests. ~30% crash. We want crashes first.",
        )

        rng = np.random.default_rng(0)
        n_cols, n_rows = 8, 3
        tiles = VGroup()
        for r in range(n_rows):
            for c in range(n_cols):
                pts = sample_road(n=12) * 0.12
                pts[:, 1] += rng.normal(scale=0.04, size=pts.shape[0])
                road = VMobject(stroke_color=PRIMARY, stroke_width=2)
                road.set_points_smoothly(to_scene_coords(pts))
                is_fail = rng.random() < 0.30
                col = BAD if is_fail else GOOD
                frame = Rectangle(
                    width=1.05, height=0.62, stroke_color=col, stroke_width=2,
                    fill_color=col, fill_opacity=0.10,
                )
                road.move_to(frame.get_center())
                tiles.add(VGroup(frame, road))
        tiles.arrange_in_grid(n_rows, n_cols, buff=0.16)
        tiles.scale_to_fit_width(11.2).move_to([0, 0.05, 0])

        self.play(
            LaggedStart(*[FadeIn(t, scale=1.03) for t in tiles],
                        lag_ratio=0.02, run_time=1.4),
        )

        leg = VGroup(
            VGroup(
                Square(side_length=0.32, color=BAD, fill_color=BAD,
                       fill_opacity=0.18, stroke_width=2),
                Text("crash  (FAIL)", font_size=20, color=BAD),
            ).arrange(RIGHT, buff=0.18),
            VGroup(
                Square(side_length=0.32, color=GOOD, fill_color=GOOD,
                       fill_opacity=0.18, stroke_width=2),
                Text("safe  (PASS)", font_size=20, color=GOOD),
            ).arrange(RIGHT, buff=0.18),
        ).arrange(RIGHT, buff=1.2).move_to([0, -2.30, 0])
        self.play(FadeIn(leg, shift=UP * 0.15), run_time=0.7)

        cap = footer("FAILs are rare. Surfacing them first is the entire game.")
        self.play(Write(cap), run_time=0.9)
        hold(self, 2.0)

        transition(self)

    # ----------------------------------------------------------- c. metric --
    def _metric(self):
        header = section_header(
            self, "The metric: APFD",
            "Average Position of Failure Detection (higher = better).",
        )

        eq = big_formula(
            r"\mathrm{APFD} \;=\; 1 \;-\; "
            r"\dfrac{\sum_{i=1}^{m}\mathrm{pos}(f_i)}{n\,m} "
            r"\;+\; \dfrac{1}{2n}"
        ).move_to([0, 1.50, 0])
        eq.set_color_by_tex("APFD", ACCENT)
        eq.set_color_by_tex("pos", BAD)
        self.play(Write(eq), run_time=1.8)
        hold(self, 0.4)

        # Two ranked queues
        n_tests = 12
        fails_good = [1, 2, 4]
        fails_bad  = [8, 10, 11]

        def make_queue(fail_positions, y, lbl_text, lbl_color):
            row = VGroup()
            for i in range(n_tests):
                is_fail = i in fail_positions
                col = BAD if is_fail else GOOD
                cell = Square(
                    side_length=0.48, stroke_color=col, stroke_width=2,
                    fill_color=col, fill_opacity=0.45,
                )
                num = Text(str(i + 1), font_size=14, color=WHITE)
                num.move_to(cell.get_center())
                row.add(VGroup(cell, num))
            row.arrange(RIGHT, buff=0.10)
            row.move_to([0.5, y, 0])
            lab = Text(lbl_text, font_size=22, color=lbl_color)
            lab.next_to(row, LEFT, buff=0.40)
            return VGroup(row, lab)

        def apfd(positions, n=12):
            return 1 - sum(p + 1 for p in positions) / (n * len(positions)) + 1 / (2 * n)

        good = make_queue(fails_good, -0.30, "good ranking", ACCENT)
        bad  = make_queue(fails_bad,  -1.55, "bad ranking",  MUTED)
        good_val = inline_math(rf"\mathrm{{APFD}} = {apfd(fails_good):.3f}",
                               color=ACCENT).next_to(good, RIGHT, buff=0.35)
        bad_val  = inline_math(rf"\mathrm{{APFD}} = {apfd(fails_bad):.3f}",
                               color=BAD).next_to(bad, RIGHT, buff=0.35)

        self.play(FadeIn(good, shift=RIGHT * 0.15), run_time=0.6)
        self.play(Write(good_val), run_time=0.6)
        self.play(FadeIn(bad, shift=RIGHT * 0.15), run_time=0.6)
        self.play(Write(bad_val), run_time=0.6)

        cap = footer("Each red square is a crash. The closer to the front, the higher APFD.")
        self.play(Write(cap), run_time=0.9)
        hold(self, 2.0)

        transition(self)

    # -------------------------------------------------------- d. ladder ----
    def _project_map(self):
        header = section_header(
            self, "The leaderboard",
            "8 methods on 956 OOD tests (30 trials); only ours is rotation-invariant.",
        )

        # Competitor leaderboard, mirrored 1-to-1 from presentation/se2_slides.tex:
        # (method, approach, APFD, row color, is_ours).
        items = [
            ("Random",             "--",              "0.493", MUTED,  False),
            ("GNN",                "graph",           "0.533", MUTED,  False),
            ("ResNet-50",          "image",           "0.572", MUTED,  False),
            ("SO-SDC-Prioritizer", "GA (TOSEM'23)",   "0.765", BLUE_A, False),
            ("ITEP4SDC",           "MLP (ICST'25)",   "0.781", BLUE_A, False),
            ("Greedy-diversity",   "heuristic",       "0.795", BLUE_A, False),
            ("RoadFury",           "Transformer+SWA", "0.804", GOOD,   False),
            ("SE2RoadNet",         "SE(2)-equiv.",    "0.805", ACCENT, True),
        ]
        rows = VGroup()
        for name, approach, apfd, col, is_ours in items:
            box = RoundedRectangle(
                width=11.0, height=0.46, corner_radius=0.09,
                stroke_color=col, stroke_width=2.5 if is_ours else 2.0,
                fill_color=col, fill_opacity=0.12 if is_ours else 0.05,
            )
            nm = Text(name, font_size=18, color=col,
                      weight="BOLD" if is_ours else "NORMAL")
            nm.move_to(box.get_left() + RIGHT * 0.40, aligned_edge=LEFT)
            ap = Text(approach, font_size=15, color=MUTED, slant="ITALIC")
            ap.move_to(box.get_left() + RIGHT * 3.55, aligned_edge=LEFT)
            val = MathTex(apfd, font_size=22, color=col)
            val.move_to(box.get_right() + LEFT * 1.75, aligned_edge=RIGHT)
            inv = (MathTex(r"\Delta=0", font_size=20, color=GOOD) if is_ours
                   else MathTex(r"\times", font_size=22, color=BAD))
            inv.move_to(box.get_right() + LEFT * 0.45, aligned_edge=RIGHT)
            rows.add(VGroup(box, nm, ap, val, inv))
        rows.arrange(DOWN, buff=0.08).move_to([0, -0.40, 0])

        apfd_hdr = Text("APFD", font_size=14, color=MUTED)
        apfd_hdr.next_to(rows[0], UP, buff=0.10).align_to(rows[0][3], RIGHT)
        inv_hdr = Text("rot-inv?", font_size=14, color=MUTED)
        inv_hdr.next_to(rows[0], UP, buff=0.10).align_to(rows[0][4], RIGHT)

        self.play(
            LaggedStart(*[FadeIn(r, shift=UP * 0.08) for r in rows],
                        lag_ratio=0.08, run_time=2.2),
        )
        self.play(FadeIn(apfd_hdr), FadeIn(inv_hdr), run_time=0.4)

        ring = accent_box(rows[-1], color=ACCENT, buff=0.04, stroke_width=3)
        self.play(Create(ring), run_time=0.7)
        hold(self, 1.6)

        cap = footer("Same APFD as the best baseline -- plus a rotation guarantee no one else has.")
        self.play(Write(cap), run_time=0.9)
        hold(self, 2.0)

        transition(self)

    # ------------------------------------------------------------ e. today --
    def _today(self):
        header = section_header(
            self, "Today: Exp 02",
            "SE(2)-Equivariant RoadNet.",
        )

        rows = [
            ("1.", "Input",      "raw road points  (L, 2)",                  BLUE_A),
            ("2.", "Features",   "7 intrinsic channels per point",           WARN),
            ("3.", "Invariance", "rotation cannot change the input",         GOOD),
            ("4.", "Model",      "Linear + CLS + 6 InvariantBlocks + head",  BLUE_B),
            ("5.", "Compute",    "one tensor flowing through the network",   ACCENT),
            ("6.", "Results",    r"$\Delta\,\mathrm{APFD}_{\mathrm{rot}} = 0.0000$,  AUC $= 0.934$", PINK),
        ]
        # Three left-aligned columns at fixed x (number / head / body) so the
        # body column starts at one constant x for every row -- no ragged edge.
        lines = VGroup()
        y0, ROW_DY = 1.25, 0.62
        for i, (num, head, body, col) in enumerate(rows):
            n = Text(num,  font_size=24, color=col, weight="BOLD")
            h = Text(head, font_size=24, color=col, weight="BOLD")
            if "Delta" in body or "AUC" in body:
                b = Tex(body, color=TEXT).scale_to_fit_height(0.30)
            else:
                b = Text(body, font_size=22, color=TEXT)
            y = y0 - i * ROW_DY
            n.move_to([-4.7, y, 0], aligned_edge=LEFT)
            h.move_to([-4.1, y, 0], aligned_edge=LEFT)
            b.move_to([-1.3, y, 0], aligned_edge=LEFT)
            lines.add(VGroup(n, h, b))

        self.play(
            LaggedStart(*[FadeIn(r, shift=UP * 0.08) for r in lines],
                        lag_ratio=0.15, run_time=2.4),
        )
        hold(self, 2.0)

        transition(self)
        opener = Text("Let's begin.", font_size=44, color=ACCENT)
        self.play(FadeIn(opener, shift=UP * 0.15))
        hold(self, 1.2)
        self.opener = opener
