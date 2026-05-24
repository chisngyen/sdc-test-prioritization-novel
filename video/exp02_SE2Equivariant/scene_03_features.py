"""
Scene 03 -- 7-channel invariant feature extraction.

The heart of the rotation-invariance proof.  For each of the 7 channels
we show:
    (a) its formula,
    (b) a geometric picture on the actual road,
    (c) the numerical value at a sample index.

Render:
    manim -pql scene_03_features.py FeatureExtract
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Circle, Dot, Line, Arrow,
    DashedLine, Arc, ArcBetweenPoints, Polygon,
    Axes, NumberPlane, BarChart,
    Write, FadeIn, FadeOut, Create, Uncreate, ReplacementTransform,
    Indicate, Flash, LaggedStart, Transform, AnimationGroup, Wait,
    ValueTracker, always_redraw, DecimalNumber,
    UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL, ORIGIN, PI, DEGREES,
    WHITE, BLUE, BLUE_A, BLUE_B, BLUE_D, BLUE_E, YELLOW, YELLOW_A, ORANGE, RED,
    GREEN, GREEN_A, GREEN_B, GREY, GREY_A, GREY_B, GREY_C, GREY_D,
    GOLD, PINK, MAROON, TEAL, PURPLE, PURPLE_A,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_road, to_scene_coords, make_road,
    extract_invariant_7ch, normalise_to_unit_band,
    ROAD_COLOR, POINT_COLOR, FEATURE_COLORS, FEATURE_NAMES,
    FEATURE_DESC, FEATURE_KEYS, fmt,
)


# Index of the sample point we focus on for value read-outs.
FOCUS_IDX = 12


# ----------------------------------------------------------------------------- #
# Helpers building the recurring sub-mobjects
# ----------------------------------------------------------------------------- #
def build_chip_row(active_idx: int) -> VGroup:
    """Seven coloured chips bottom-left, the active one inflated."""
    chips = VGroup()
    for i, key in enumerate(FEATURE_KEYS):
        c = FEATURE_COLORS[key]
        if i == active_idx:
            chip = RoundedRectangle(
                width=0.55, height=0.55, corner_radius=0.1,
                stroke_color=c, stroke_width=4,
                fill_color=c, fill_opacity=0.85,
            )
            chip.set_z_index(2)
        else:
            chip = RoundedRectangle(
                width=0.45, height=0.45, corner_radius=0.1,
                stroke_color=c, stroke_width=2,
                fill_color=c, fill_opacity=0.25,
            )
        chips.add(chip)
    chips.arrange(RIGHT, buff=0.12).to_corner(DL, buff=0.45)
    return chips


def heat_bar(values: np.ndarray, color, *, width=4.6, height=0.45):
    """Tiny coloured bar plot encoding the 1-D channel along arclength."""
    v = np.asarray(values, dtype=np.float64)
    v_norm = normalise_to_unit_band(np.abs(v - np.median(v)), 0.05, 1.0)
    bars = VGroup()
    n = len(v)
    bw = width / n
    for i, h in enumerate(v_norm):
        bar = Rectangle(
            width=bw * 0.85, height=height * float(h),
            stroke_width=0, fill_color=color, fill_opacity=0.9,
        )
        bar.move_to([-width / 2 + (i + 0.5) * bw, 0, 0], aligned_edge=DOWN)
        bars.add(bar)
    return bars


# ============================================================================ #
class FeatureExtract(Scene):
    def construct(self):
        # ---------------------------------------------------------- title -- #
        big_title = Text("7 numbers per point that rotation cannot move",
                         font_size=34, color=WHITE).to_edge(UP, buff=0.4)
        self.play(Write(big_title), run_time=1.4)

        # ---------------------------------------------------------- road -- #
        pts = sample_road(n=20) * 0.95
        feats = extract_invariant_7ch(pts)

        # Position the road in the upper half of the screen
        road_offset = np.array([0.0, 0.8, 0.0])
        coords = to_scene_coords(pts) + road_offset

        road_line = VMobject(stroke_color=ROAD_COLOR, stroke_width=7)
        road_line.set_points_smoothly(coords)
        dots = VGroup(*[Dot(p, radius=0.06, color=POINT_COLOR) for p in coords])

        self.play(Create(road_line), run_time=1.4)
        self.play(LaggedStart(*[FadeIn(d, scale=1.3) for d in dots],
                              lag_ratio=0.03, run_time=1.1))
        self.wait(0.3)
        self.play(FadeOut(big_title))

        # ------------------------------------------------------ side rail -- #
        rail_title = MathTex(
            r"\text{feature}(\mathbf{r}) \in \mathbb{R}^{L \times 7}",
            font_size=28, color=BLUE_A,
        ).to_edge(UP, buff=0.45)
        self.play(FadeIn(rail_title, shift=DOWN * 0.2))

        # ============== run each feature in turn ============================ #
        self._channel_segment_length(coords, feats, dots, road_line)
        self._channel_heading_change(coords, feats, dots, road_line)
        self._channel_curvature(coords, feats, dots, road_line)
        self._channel_curvature_rate(coords, feats, dots, road_line)
        self._channel_curvature_accel(coords, feats, dots, road_line)
        self._channel_arclength(coords, feats, dots, road_line)
        self._channel_local_std(coords, feats, dots, road_line)

        # ------------------------------------- final wrap-up: stacked panel - #
        self._final_stack(coords, feats, road_line, dots)

        self.wait(1.0)

    # -------------------------------------------------------- channel 1 ---- #
    def _channel_segment_length(self, coords, feats, dots, road_line):
        key = "seg"
        chips = build_chip_row(0)
        self.play(FadeIn(chips, shift=UP * 0.1))

        # Highlight ONE segment
        i = FOCUS_IDX
        a, b = coords[i], coords[i + 1]
        seg = Line(a, b, color=FEATURE_COLORS[key], stroke_width=10)
        bracket = DashedLine(a + UP * 0.25, b + UP * 0.25,
                             color=FEATURE_COLORS[key], stroke_width=2)
        dlabel = MathTex(r"\Delta s_i", font_size=28,
                         color=FEATURE_COLORS[key]).next_to(bracket, UP, buff=0.1)

        formula_grp, value_text, desc, bar = self._panel(
            key,
            formula=r"\Delta s_i \;=\; \|\mathbf{r}_i - \mathbf{r}_{i-1}\|",
            value=feats[i, 0],
            channel_idx=0, feats=feats,
        )

        self.play(Create(seg), Create(bracket), Write(dlabel))
        self.play(Write(formula_grp), run_time=1.4)
        self.play(FadeIn(desc, shift=UP * 0.05), FadeIn(value_text, shift=UP * 0.05))
        self.play(FadeIn(bar, shift=UP * 0.05))
        self.wait(1.3)

        self.play(FadeOut(VGroup(seg, bracket, dlabel, formula_grp,
                                 value_text, desc, bar, chips)))

    # -------------------------------------------------------- channel 2 ---- #
    def _channel_heading_change(self, coords, feats, dots, road_line):
        key = "dangle"
        chips = build_chip_row(1)
        self.play(FadeIn(chips, shift=UP * 0.1))

        i = FOCUS_IDX
        prev_arrow = Arrow(coords[i - 1], coords[i], buff=0,
                           color=BLUE_B, stroke_width=5,
                           max_tip_length_to_length_ratio=0.35)
        next_arrow = Arrow(coords[i], coords[i + 1], buff=0,
                           color=FEATURE_COLORS[key], stroke_width=5,
                           max_tip_length_to_length_ratio=0.35)
        # Angle wedge between the two arrow directions
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        wedge = Arc(radius=0.35, start_angle=a1, angle=(a2 - a1 + PI) % (2 * PI) - PI,
                    color=YELLOW, stroke_width=4, arc_center=coords[i])
        wedge_lbl = MathTex(r"|\Delta\theta_i|", font_size=24,
                            color=YELLOW).next_to(wedge, UR, buff=0.05)

        formula_grp, value_text, desc, bar = self._panel(
            key,
            formula=r"|\Delta\theta_i| \;=\; "
                    r"\big|\angle(\mathbf{r}_{i+1}{-}\mathbf{r}_i) "
                    r"\;-\; \angle(\mathbf{r}_i{-}\mathbf{r}_{i-1})\big|",
            value=feats[i, 1],
            channel_idx=1, feats=feats,
            sub=r"\text{absolute value $\Rightarrow$ flip-invariant.}",
        )

        self.play(Create(prev_arrow), Create(next_arrow))
        self.play(Create(wedge), Write(wedge_lbl))
        self.play(Write(formula_grp), run_time=1.6)
        self.play(FadeIn(desc, shift=UP * 0.05), FadeIn(value_text, shift=UP * 0.05))
        self.play(FadeIn(bar, shift=UP * 0.05))
        self.wait(1.6)
        self.play(FadeOut(VGroup(prev_arrow, next_arrow, wedge, wedge_lbl,
                                 formula_grp, value_text, desc, bar, chips)))

    # -------------------------------------------------------- channel 3 ---- #
    def _channel_curvature(self, coords, feats, dots, road_line):
        key = "kappa"
        chips = build_chip_row(2)
        self.play(FadeIn(chips, shift=UP * 0.1))

        i = FOCUS_IDX
        # Highlight an osculating arc as a visual mnemonic for kappa = 1/R.
        glow = Dot(coords[i], radius=0.16,
                   color=FEATURE_COLORS[key], fill_opacity=0.7)
        radius = 1.0 / max(abs(feats[i, 2]), 0.18)
        # Direction perpendicular to local tangent
        v = coords[i + 1] - coords[i - 1]
        tangent = v / (np.linalg.norm(v) + 1e-8)
        normal = np.array([-tangent[1], tangent[0], 0.0])
        sign = 1.0 if feats[i, 2] >= 0 else -1.0
        center = coords[i] + sign * normal * radius
        osculating = Circle(radius=radius, color=FEATURE_COLORS[key],
                            stroke_width=3).move_to(center)
        osc_lbl = MathTex(r"R = 1/\kappa_i", font_size=22,
                          color=FEATURE_COLORS[key]).next_to(osculating, UP, buff=0.1)

        formula_grp, value_text, desc, bar = self._panel(
            key,
            formula=r"\kappa_i \;=\; \dfrac{\Delta\theta_i}{\tfrac{1}{2}"
                    r"(\Delta s_{i-1} + \Delta s_i)}",
            value=feats[i, 2],
            channel_idx=2, feats=feats,
            sub=r"\text{signed: }+\text{ left bend},\,-\text{ right bend}.",
            value_prec=4,
        )

        self.play(FadeIn(glow))
        self.play(Create(osculating), Write(osc_lbl), run_time=1.4)
        self.play(Write(formula_grp), run_time=1.6)
        self.play(FadeIn(desc, shift=UP * 0.05), FadeIn(value_text, shift=UP * 0.05))
        self.play(FadeIn(bar, shift=UP * 0.05))
        self.wait(1.8)
        self.play(FadeOut(VGroup(glow, osculating, osc_lbl,
                                 formula_grp, value_text, desc, bar, chips)))

    # -------------------------------------------------------- channel 4 ---- #
    def _channel_curvature_rate(self, coords, feats, dots, road_line):
        key = "dkappa"
        chips = build_chip_row(3)
        self.play(FadeIn(chips, shift=UP * 0.1))

        # Draw kappa along the road as a thin coloured trace under the road
        kappa = feats[:, 2]
        n = len(coords)
        below = coords - np.array([0, 1.2, 0])
        scale = 0.7 / (np.max(np.abs(kappa)) + 1e-8)
        trace_pts = [
            np.array([below[i, 0], below[i, 1] + kappa[i] * scale, 0.0])
            for i in range(n)
        ]
        kappa_trace = VMobject(stroke_color=FEATURE_COLORS["kappa"], stroke_width=4)
        kappa_trace.set_points_smoothly(trace_pts)
        axis = Line(below[0] + LEFT * 0.0, below[-1] + RIGHT * 0.0,
                    color=GREY_C, stroke_width=2)
        axis_lbl = MathTex(r"\kappa(s)", font_size=22,
                           color=FEATURE_COLORS["kappa"]).next_to(kappa_trace, LEFT, buff=0.15)

        # Highlight the slope at FOCUS_IDX
        i = FOCUS_IDX
        p_now = trace_pts[i]
        p_next = trace_pts[i + 1]
        slope_line = Line(p_now, p_next, color=FEATURE_COLORS[key], stroke_width=8)

        formula_grp, value_text, desc, bar = self._panel(
            key,
            formula=r"\kappa'_i \;=\; \dfrac{\kappa_{i+1} - \kappa_i}{\Delta s_i}"
                    r" \;\approx\; \dfrac{d\kappa}{ds}\bigg|_i",
            value=feats[i, 3],
            channel_idx=3, feats=feats,
            sub=r"\text{a smooth bend vs.\ a sudden snap.}",
            value_prec=4,
        )

        self.play(Create(axis), Create(kappa_trace), Write(axis_lbl), run_time=1.6)
        self.play(Create(slope_line), run_time=0.5)
        self.play(Write(formula_grp), run_time=1.4)
        self.play(FadeIn(desc, shift=UP * 0.05), FadeIn(value_text, shift=UP * 0.05))
        self.play(FadeIn(bar, shift=UP * 0.05))
        self.wait(1.4)

        self.play(FadeOut(VGroup(axis, kappa_trace, axis_lbl, slope_line,
                                 formula_grp, value_text, desc, bar, chips)))

    # -------------------------------------------------------- channel 5 ---- #
    def _channel_curvature_accel(self, coords, feats, dots, road_line):
        key = "ddkappa"
        chips = build_chip_row(4)
        self.play(FadeIn(chips, shift=UP * 0.1))

        i = FOCUS_IDX
        formula_grp, value_text, desc, bar = self._panel(
            key,
            formula=r"\kappa''_i \;=\; \dfrac{\kappa'_{i+1} - \kappa'_i}{\Delta s_i}"
                    r"\;\approx\;\dfrac{d^2\kappa}{ds^2}\bigg|_i",
            value=feats[i, 4],
            channel_idx=4, feats=feats,
            sub=r"\text{measures \emph{jerk} in road shape.}",
            value_prec=4,
        )

        # Visual: emphasise points where ddk is large (likely failure-prone spots)
        ddk = np.abs(feats[:, 4])
        heat = normalise_to_unit_band(ddk, 0.0, 1.0)
        flash_dots = VGroup()
        for j in range(len(coords)):
            r = 0.045 + 0.10 * heat[j]
            d = Dot(coords[j], radius=r,
                    color=FEATURE_COLORS[key], fill_opacity=0.85)
            flash_dots.add(d)

        self.play(LaggedStart(*[FadeIn(d, scale=1.3) for d in flash_dots],
                              lag_ratio=0.03, run_time=1.2))
        self.play(Write(formula_grp), run_time=1.4)
        self.play(FadeIn(desc, shift=UP * 0.05), FadeIn(value_text, shift=UP * 0.05))
        self.play(FadeIn(bar, shift=UP * 0.05))
        self.wait(1.4)

        self.play(FadeOut(VGroup(flash_dots, formula_grp, value_text,
                                 desc, bar, chips)))

    # -------------------------------------------------------- channel 6 ---- #
    def _channel_arclength(self, coords, feats, dots, road_line):
        key = "s_norm"
        chips = build_chip_row(5)
        self.play(FadeIn(chips, shift=UP * 0.1))

        # Animate a "progress" dot sweeping along the road, with a 0..1 readout
        sweep = Dot(coords[0], radius=0.13, color=FEATURE_COLORS[key])
        progress_tracker = ValueTracker(0.0)
        # We will use the smooth path mobject for proportional travel
        path = road_line
        sweep.add_updater(
            lambda m: m.move_to(path.point_from_proportion(
                float(np.clip(progress_tracker.get_value(), 0.0, 1.0))
            ))
        )

        readout = always_redraw(
            lambda: MathTex(
                rf"s/L = {progress_tracker.get_value():.2f}",
                font_size=28, color=FEATURE_COLORS[key],
            ).to_corner(UR, buff=0.6)
        )

        formula_grp, value_text, desc, bar = self._panel(
            key,
            formula=r"\dfrac{s_i}{L} \;=\; "
                    r"\dfrac{\sum_{j\le i} \Delta s_j}{\sum_{j} \Delta s_j}",
            value=feats[FOCUS_IDX, 5],
            channel_idx=5, feats=feats,
            sub=r"\text{parameterization-invariant `where am I?'}",
            value_prec=3,
        )

        self.play(FadeIn(sweep), FadeIn(readout))
        self.play(progress_tracker.animate.set_value(1.0),
                  run_time=3.0, rate_func=lambda t: t)
        self.play(progress_tracker.animate.set_value(FOCUS_IDX / (len(coords) - 1)),
                  run_time=0.8)
        self.play(Write(formula_grp), run_time=1.4)
        self.play(FadeIn(desc, shift=UP * 0.05), FadeIn(value_text, shift=UP * 0.05))
        self.play(FadeIn(bar, shift=UP * 0.05))
        self.wait(1.3)

        sweep.clear_updaters()
        self.play(FadeOut(VGroup(sweep, readout, formula_grp, value_text,
                                 desc, bar, chips)))

    # -------------------------------------------------------- channel 7 ---- #
    def _channel_local_std(self, coords, feats, dots, road_line):
        key = "lstd"
        chips = build_chip_row(6)
        self.play(FadeIn(chips, shift=UP * 0.1))

        i = FOCUS_IDX
        # Window of 11 points centred on i
        hw = 5
        a, b = max(0, i - hw), min(len(coords), i + hw + 1)
        window_dots = VGroup(*[
            Dot(coords[j], radius=0.10,
                color=FEATURE_COLORS[key], fill_opacity=0.9)
            for j in range(a, b)
        ])
        window_box = Rectangle(
            width=abs(coords[b - 1, 0] - coords[a, 0]) + 0.4, height=0.7,
            stroke_color=FEATURE_COLORS[key], stroke_width=3,
            fill_opacity=0.0,
        ).move_to(np.mean(coords[a:b], axis=0))
        win_lbl = Text("window = 11", font_size=18,
                       color=FEATURE_COLORS[key]).next_to(window_box, DOWN, buff=0.1)

        formula_grp, value_text, desc, bar = self._panel(
            key,
            formula=r"\sigma_\kappa(i) \;=\; "
                    r"\mathrm{std}\bigl(\kappa_{i-5},\ldots,\kappa_{i+5}\bigr)",
            value=feats[i, 6],
            channel_idx=6, feats=feats,
            sub=r"\text{`roughness' near point $i$ -- still rotation-free.}",
            value_prec=4,
        )

        self.play(Create(window_box), FadeIn(win_lbl))
        self.play(LaggedStart(*[FadeIn(d, scale=1.4) for d in window_dots],
                              lag_ratio=0.05, run_time=1.0))
        self.play(Write(formula_grp), run_time=1.4)
        self.play(FadeIn(desc, shift=UP * 0.05), FadeIn(value_text, shift=UP * 0.05))
        self.play(FadeIn(bar, shift=UP * 0.05))
        self.wait(1.4)

        self.play(FadeOut(VGroup(window_dots, window_box, win_lbl,
                                 formula_grp, value_text, desc, bar, chips)))

    # ----------------------------------------------- shared panel builder -- #
    def _panel(self, key, *, formula, value, channel_idx, feats,
               sub=None, value_prec=3):
        color = FEATURE_COLORS[key]
        name_tex = FEATURE_NAMES[key]

        title = MathTex(
            rf"\text{{ch.\ }}{channel_idx + 1}:\quad {name_tex}",
            font_size=36,
        )
        title[0].set_color(color)

        formula_eq = MathTex(formula, font_size=30, color=WHITE)
        desc_lbl = Text(FEATURE_DESC[key], font_size=22, color=color, slant="ITALIC")

        value_eq = MathTex(
            rf"{name_tex} \;=\; {fmt(float(value), value_prec)}",
            font_size=32, color=color,
        )

        # The lower band stack (formula + description)
        top_stack = VGroup(title, formula_eq).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        if sub is not None:
            sub_eq = Tex(sub, font_size=22, color=GREY_A)
            top_stack.add(sub_eq)
            top_stack.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        top_stack.to_edge(DOWN, buff=0.45).shift(LEFT * 2.0)

        # The right-side value + heat bar
        value_eq.to_corner(DR, buff=0.6).shift(UP * 0.7)
        bar = heat_bar(feats[:, channel_idx], color)
        bar.next_to(value_eq, DOWN, buff=0.25, aligned_edge=RIGHT)

        return top_stack, value_eq, desc_lbl, bar

    # ----------------------------------------- final summary: 7-channel matrix
    def _final_stack(self, coords, feats, road_line, dots):
        title = Text("All 7 channels together",
                     font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Build a small heat bar per channel, stacked
        bars_group = VGroup()
        labels_group = VGroup()
        for k, key in enumerate(FEATURE_KEYS):
            bar = heat_bar(feats[:, k], FEATURE_COLORS[key], width=8.0, height=0.32)
            lab = MathTex(FEATURE_NAMES[key], font_size=22,
                          color=FEATURE_COLORS[key])
            row = VGroup(lab, bar).arrange(RIGHT, buff=0.25)
            bars_group.add(row)
        bars_group.arrange(DOWN, buff=0.18).to_edge(DOWN, buff=0.55)

        self.play(FadeOut(road_line), FadeOut(dots))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in bars_group],
                              lag_ratio=0.10, run_time=2.0))
        self.wait(0.8)

        # punchline
        tagline = Text(
            "Every one is a function of distances and angles only.",
            font_size=24, color=YELLOW,
        ).next_to(title, DOWN, buff=0.25)
        self.play(Write(tagline))
        self.wait(1.6)

        self.play(FadeOut(VGroup(title, tagline, bars_group)))
