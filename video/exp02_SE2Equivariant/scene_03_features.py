"""
Scene 03 -- 7-channel invariant feature extraction.

For each of the 7 channels we show:
    (a) a coloured chip in a chip row (which channel we are on),
    (b) a geometric picture on the actual road,
    (c) the formula,
    (d) the numerical value at a sample index,
    (e) a heat bar that encodes the channel along arclength.

Render:  manim -pql scene_03_features.py FeatureExtract
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Circle, Dot, Line, Arrow,
    DashedLine, Arc,
    Write, FadeIn, FadeOut, Create, LaggedStart,
    ValueTracker, always_redraw,
    UP, DOWN, LEFT, RIGHT, ORIGIN, PI, UR,
    WHITE, BLUE_A, YELLOW, GREY_A,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_road, to_scene_coords, extract_invariant_7ch, normalise_to_unit_band,
    ROAD_COLOR, POINT_COLOR, FEATURE_COLORS, FEATURE_NAMES,
    FEATURE_DESC, FEATURE_KEYS, fmt,
)
from theme import (
    TEXT, MUTED, PRIMARY, ACCENT, GOOD, WARN, BAD, RULE,
    section_header, transition, hold,
    title, subtitle, footer, body_text, caption,
    body_formula, inline_math, panel,
    attach_narration, seal_narration,
    MATH_BODY, MATH_INLINE, MATH_SMALL,
)


FOCUS_IDX = 12


# ----------------------------------------------------------------- helpers --
def build_chip_row(active_idx: int) -> VGroup:
    chips = VGroup()
    for i, key in enumerate(FEATURE_KEYS):
        col = FEATURE_COLORS[key]
        active = i == active_idx
        size = 0.48 if active else 0.38
        fill = 0.85 if active else 0.18
        stroke = 3.0 if active else 1.6
        chip = RoundedRectangle(
            width=size, height=size, corner_radius=0.09,
            stroke_color=col, stroke_width=stroke,
            fill_color=col, fill_opacity=fill,
        )
        idx_lab = Text(str(i + 1), font_size=14,
                       color="#1a1a1a" if active else col,
                       weight="BOLD")
        idx_lab.move_to(chip.get_center())
        chips.add(VGroup(chip, idx_lab))
    # Position centered just below the subtitle, above the road.
    chips.arrange(RIGHT, buff=0.16).move_to([0, 2.10, 0])
    return chips


def heat_bar(values: np.ndarray, color, *, width=4.5, height=0.42) -> VGroup:
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


# ===========================================================================
class FeatureExtract(Scene):
    def construct(self):
        attach_narration(self, "scene_03")
        # Title that stays the whole scene
        head = title("7 numbers per point that rotation cannot move")
        underline = Line(
            head.get_corner(DOWN + LEFT) + DOWN * 0.10,
            head.get_corner(DOWN + RIGHT) + DOWN * 0.10,
            color=PRIMARY, stroke_width=2,
        )
        rail = MathTex(
            r"\mathrm{features}(\mathbf{r}) \in \mathbb{R}^{L \times 7}",
            font_size=28, color=PRIMARY,
        ).move_to([0, 2.55, 0])
        self.play(Write(head), Create(underline), run_time=0.9)
        self.play(FadeIn(rail, shift=DOWN * 0.10), run_time=0.5)
        self.persistent_header = VGroup(head, underline, rail)

        # Road that stays the whole scene (top half)
        pts = sample_road(n=20) * 0.90
        feats = extract_invariant_7ch(pts)
        road_offset = np.array([0.0, 0.55, 0.0])
        coords = to_scene_coords(pts) + road_offset

        road_line = VMobject(stroke_color=PRIMARY, stroke_width=7)
        road_line.set_points_smoothly(coords)
        dots = VGroup(*[Dot(p, radius=0.06, color=ACCENT) for p in coords])

        self.play(Create(road_line), run_time=1.3)
        self.play(LaggedStart(*[FadeIn(d, scale=1.3) for d in dots],
                              lag_ratio=0.03, run_time=1.0))
        hold(self, 0.3)

        self.road_line = road_line
        self.dots = dots
        self.coords = coords
        self.feats = feats

        # Run each feature channel
        self._channel_segment_length()
        self._channel_heading_change()
        self._channel_curvature()
        self._channel_curvature_rate()
        self._channel_curvature_accel()
        self._channel_arclength()
        self._channel_local_std()

        # Final stack
        self._final_stack()

    # ----------------- per-channel scaffolding ----------------
    def _panel_bundle(self, key: str, *, formula: str, value: float,
                      channel_idx: int, sub: str | None = None,
                      value_prec: int = 3):
        """Build the four bottom mobjects: chips, formula+desc stack,
        numeric value, heat bar.  Positioned in stable slots."""
        color = FEATURE_COLORS[key]
        name_tex = FEATURE_NAMES[key]

        ch_title = MathTex(
            rf"\text{{ch.\ }}{channel_idx + 1}:\quad {name_tex}",
            font_size=34,
        )
        ch_title[0].set_color(color)

        formula_eq = MathTex(formula, font_size=30, color=TEXT)
        desc_lbl = Text(FEATURE_DESC[key], font_size=22, color=color,
                        slant="ITALIC")

        bottom_stack = VGroup(ch_title, formula_eq, desc_lbl).arrange(
            DOWN, buff=0.22, aligned_edge=LEFT,
        )
        if sub is not None:
            sub_eq = Tex(sub, font_size=22, color=MUTED)
            bottom_stack.add(sub_eq)
            bottom_stack.arrange(DOWN, buff=0.20, aligned_edge=LEFT)
        bottom_stack.to_corner(DOWN + LEFT, buff=0.55).shift(UP * 0.10)

        value_eq = MathTex(
            rf"{name_tex} \;=\; {fmt(float(value), value_prec)}",
            font_size=32, color=color,
        )
        value_eq.to_corner(DOWN + RIGHT, buff=0.55).shift(UP * 1.45)

        bar = heat_bar(self.feats[:, channel_idx], color)
        bar.next_to(value_eq, DOWN, buff=0.30, aligned_edge=RIGHT)

        return bottom_stack, value_eq, bar

    def _show_panel(self, bottom_stack, value_eq, bar):
        self.play(Write(bottom_stack), run_time=1.3)
        self.play(FadeIn(value_eq, shift=UP * 0.05), run_time=0.5)
        self.play(FadeIn(bar, shift=UP * 0.05), run_time=0.5)

    def _clear_panel(self, *artifacts):
        self.play(*[FadeOut(a) for a in artifacts], run_time=0.45)

    # ----------------------------------------------------- ch.1 segment ---
    def _channel_segment_length(self):
        key = "seg"
        chips = build_chip_row(0)
        self.play(FadeIn(chips, shift=DOWN * 0.08), run_time=0.5)
        self.chips = chips

        i = FOCUS_IDX
        a, b = self.coords[i], self.coords[i + 1]
        seg = Line(a, b, color=FEATURE_COLORS[key], stroke_width=11)
        bracket = DashedLine(a + UP * 0.25, b + UP * 0.25,
                             color=FEATURE_COLORS[key], stroke_width=2)
        dlabel = inline_math(r"\Delta s_i", color=FEATURE_COLORS[key])
        dlabel.next_to(bracket, UP, buff=0.10)

        bs, val, bar = self._panel_bundle(
            key,
            formula=r"\Delta s_i \;=\; \|\mathbf{r}_i - \mathbf{r}_{i-1}\|",
            value=self.feats[i, 0], channel_idx=0,
        )

        self.play(Create(seg), Create(bracket), Write(dlabel), run_time=0.9)
        self._show_panel(bs, val, bar)
        hold(self, 1.2)
        self._clear_panel(seg, bracket, dlabel, bs, val, bar, chips)

    # ---------------------------------------------------- ch.2 heading ----
    def _channel_heading_change(self):
        key = "dangle"
        chips = build_chip_row(1)
        self.play(FadeIn(chips, shift=DOWN * 0.08), run_time=0.5)

        i = FOCUS_IDX
        prev_arrow = Arrow(self.coords[i - 1], self.coords[i], buff=0,
                           color=BLUE_A, stroke_width=5,
                           max_tip_length_to_length_ratio=0.32)
        next_arrow = Arrow(self.coords[i], self.coords[i + 1], buff=0,
                           color=FEATURE_COLORS[key], stroke_width=5,
                           max_tip_length_to_length_ratio=0.32)
        v1 = self.coords[i] - self.coords[i - 1]
        v2 = self.coords[i + 1] - self.coords[i]
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        wedge = Arc(radius=0.35, start_angle=a1,
                    angle=(a2 - a1 + PI) % (2 * PI) - PI,
                    color=ACCENT, stroke_width=4,
                    arc_center=self.coords[i])
        wedge_lbl = inline_math(r"|\Delta\theta_i|", color=ACCENT)
        wedge_lbl.next_to(wedge, UR, buff=0.05)

        bs, val, bar = self._panel_bundle(
            key,
            formula=r"|\Delta\theta_i| \;=\; "
                    r"\big|\angle(\mathbf{r}_{i+1}{-}\mathbf{r}_i) - "
                    r"\angle(\mathbf{r}_i{-}\mathbf{r}_{i-1})\big|",
            value=self.feats[i, 1], channel_idx=1,
            sub=r"absolute value $\Rightarrow$ also flip-invariant.",
        )
        self.play(Create(prev_arrow), Create(next_arrow), run_time=0.6)
        self.play(Create(wedge), Write(wedge_lbl), run_time=0.6)
        self._show_panel(bs, val, bar)
        hold(self, 1.4)
        self._clear_panel(prev_arrow, next_arrow, wedge, wedge_lbl,
                          bs, val, bar, chips)

    # --------------------------------------------------- ch.3 curvature ---
    def _channel_curvature(self):
        key = "kappa"
        chips = build_chip_row(2)
        self.play(FadeIn(chips, shift=DOWN * 0.08), run_time=0.5)

        i = FOCUS_IDX
        glow = Dot(self.coords[i], radius=0.15,
                   color=FEATURE_COLORS[key], fill_opacity=0.7)
        radius = 1.0 / max(abs(self.feats[i, 2]), 0.18)
        v = self.coords[i + 1] - self.coords[i - 1]
        tangent = v / (np.linalg.norm(v) + 1e-8)
        normal = np.array([-tangent[1], tangent[0], 0.0])
        sign = 1.0 if self.feats[i, 2] >= 0 else -1.0
        center = self.coords[i] + sign * normal * radius
        osculating = Circle(radius=radius, color=FEATURE_COLORS[key],
                            stroke_width=3).move_to(center)
        osc_lbl = inline_math(r"R \;=\; 1/\kappa_i", color=FEATURE_COLORS[key])
        osc_lbl.next_to(osculating, UP, buff=0.10)

        bs, val, bar = self._panel_bundle(
            key,
            formula=r"\kappa_i \;=\; \dfrac{\Delta\theta_i}{\tfrac{1}{2}"
                    r"(\Delta s_{i-1} + \Delta s_i)}",
            value=self.feats[i, 2], channel_idx=2, value_prec=4,
            sub=r"signed: $+$ left bend, $-$ right bend.",
        )

        self.play(FadeIn(glow), run_time=0.4)
        self.play(Create(osculating), Write(osc_lbl), run_time=1.0)
        self._show_panel(bs, val, bar)
        hold(self, 1.4)
        self._clear_panel(glow, osculating, osc_lbl, bs, val, bar, chips)

    # ----------------------------------------------- ch.4 curvature rate -
    def _channel_curvature_rate(self):
        key = "dkappa"
        chips = build_chip_row(3)
        self.play(FadeIn(chips, shift=DOWN * 0.08), run_time=0.5)

        kappa = self.feats[:, 2]
        n = len(self.coords)
        below = self.coords - np.array([0, 1.40, 0])
        scale = 0.55 / (np.max(np.abs(kappa)) + 1e-8)
        trace_pts = [
            np.array([below[i, 0], below[i, 1] + kappa[i] * scale, 0.0])
            for i in range(n)
        ]
        kappa_trace = VMobject(stroke_color=FEATURE_COLORS["kappa"], stroke_width=4)
        kappa_trace.set_points_smoothly(trace_pts)
        axis = Line(below[0], below[-1], color=GREY_A, stroke_width=2)
        axis_lbl = inline_math(r"\kappa(s)", color=FEATURE_COLORS["kappa"])
        axis_lbl.next_to(kappa_trace, LEFT, buff=0.18)

        i = FOCUS_IDX
        slope_line = Line(trace_pts[i], trace_pts[i + 1],
                          color=FEATURE_COLORS[key], stroke_width=9)

        bs, val, bar = self._panel_bundle(
            key,
            formula=r"\kappa'_i \;=\; \dfrac{\kappa_{i+1} - \kappa_i}{\Delta s_i}"
                    r" \;\approx\; \dfrac{d\kappa}{ds}\Big|_i",
            value=self.feats[i, 3], channel_idx=3, value_prec=4,
            sub=r"smooth bend vs.\ sudden snap.",
        )

        self.play(Create(axis), Create(kappa_trace), Write(axis_lbl), run_time=1.2)
        self.play(Create(slope_line), run_time=0.4)
        self._show_panel(bs, val, bar)
        hold(self, 1.3)
        self._clear_panel(axis, kappa_trace, axis_lbl, slope_line,
                          bs, val, bar, chips)

    # ----------------------------------------- ch.5 curvature acceleration
    def _channel_curvature_accel(self):
        key = "ddkappa"
        chips = build_chip_row(4)
        self.play(FadeIn(chips, shift=DOWN * 0.08), run_time=0.5)

        ddk = np.abs(self.feats[:, 4])
        heat = normalise_to_unit_band(ddk, 0.0, 1.0)
        flash_dots = VGroup(*[
            Dot(self.coords[j],
                radius=0.045 + 0.10 * float(heat[j]),
                color=FEATURE_COLORS[key], fill_opacity=0.85)
            for j in range(len(self.coords))
        ])

        bs, val, bar = self._panel_bundle(
            key,
            formula=r"\kappa''_i \;=\; \dfrac{\kappa'_{i+1} - \kappa'_i}{\Delta s_i}"
                    r" \;\approx\; \dfrac{d^2\kappa}{ds^2}\Big|_i",
            value=self.feats[FOCUS_IDX, 4], channel_idx=4, value_prec=4,
            sub=r"detects \emph{jerks} in road shape.",
        )

        self.play(
            LaggedStart(*[FadeIn(d, scale=1.3) for d in flash_dots],
                        lag_ratio=0.03, run_time=1.0),
        )
        self._show_panel(bs, val, bar)
        hold(self, 1.3)
        self._clear_panel(flash_dots, bs, val, bar, chips)

    # --------------------------------------------------- ch.6 arclength --
    def _channel_arclength(self):
        key = "s_norm"
        chips = build_chip_row(5)
        self.play(FadeIn(chips, shift=DOWN * 0.08), run_time=0.5)

        sweep = Dot(self.coords[0], radius=0.13, color=FEATURE_COLORS[key])
        prog = ValueTracker(0.0)
        sweep.add_updater(
            lambda m: m.move_to(
                self.road_line.point_from_proportion(
                    float(np.clip(prog.get_value(), 0.0, 1.0))
                )
            )
        )
        readout = always_redraw(
            lambda: MathTex(
                rf"s/L \;=\; {prog.get_value():.2f}",
                font_size=28, color=FEATURE_COLORS[key],
            ).to_corner(UP + RIGHT, buff=0.55).shift(DOWN * 0.95)
        )

        bs, val, bar = self._panel_bundle(
            key,
            formula=r"\dfrac{s_i}{L} \;=\; \dfrac{\sum_{j \le i}\Delta s_j}{\sum_{j}\Delta s_j}",
            value=self.feats[FOCUS_IDX, 5], channel_idx=5, value_prec=3,
            sub=r"parameterization-invariant `where am I?'.",
        )

        self.play(FadeIn(sweep), FadeIn(readout), run_time=0.4)
        self.play(prog.animate.set_value(1.0),
                  run_time=2.6, rate_func=lambda t: t)
        self.play(prog.animate.set_value(FOCUS_IDX / (len(self.coords) - 1)),
                  run_time=0.7)
        self._show_panel(bs, val, bar)
        hold(self, 1.2)
        sweep.clear_updaters()
        self._clear_panel(sweep, readout, bs, val, bar, chips)

    # ---------------------------------------------------- ch.7 local std -
    def _channel_local_std(self):
        key = "lstd"
        chips = build_chip_row(6)
        self.play(FadeIn(chips, shift=DOWN * 0.08), run_time=0.5)

        i = FOCUS_IDX
        hw = 5
        a, b = max(0, i - hw), min(len(self.coords), i + hw + 1)
        window_dots = VGroup(*[
            Dot(self.coords[j], radius=0.10,
                color=FEATURE_COLORS[key], fill_opacity=0.9)
            for j in range(a, b)
        ])
        window_box = Rectangle(
            width=abs(self.coords[b - 1, 0] - self.coords[a, 0]) + 0.45,
            height=0.75, stroke_color=FEATURE_COLORS[key], stroke_width=3,
            fill_opacity=0.0,
        ).move_to(np.mean(self.coords[a:b], axis=0))
        win_lbl = Text("window = 11", font_size=18,
                       color=FEATURE_COLORS[key]).next_to(window_box, DOWN, buff=0.10)

        bs, val, bar = self._panel_bundle(
            key,
            formula=r"\sigma_\kappa(i) \;=\; "
                    r"\mathrm{std}\bigl(\kappa_{i-5},\,\ldots,\,\kappa_{i+5}\bigr)",
            value=self.feats[i, 6], channel_idx=6, value_prec=4,
            sub=r"`roughness' near point $i$ -- still rotation-free.",
        )

        self.play(Create(window_box), FadeIn(win_lbl), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(d, scale=1.3) for d in window_dots],
                              lag_ratio=0.04, run_time=0.9))
        self._show_panel(bs, val, bar)
        hold(self, 1.3)
        self._clear_panel(window_dots, window_box, win_lbl,
                          bs, val, bar, chips)

    # ------------------------------------------------- final 7-bar stack -
    def _final_stack(self):
        self.play(FadeOut(self.road_line), FadeOut(self.dots),
                  FadeOut(self.persistent_header), run_time=0.6)

        head = title("All 7 channels together")
        ul = Line(
            head.get_corner(DOWN + LEFT) + DOWN * 0.10,
            head.get_corner(DOWN + RIGHT) + DOWN * 0.10,
            color=PRIMARY, stroke_width=2,
        )
        self.play(Write(head), Create(ul), run_time=0.8)

        bars_group = VGroup()
        for k, key in enumerate(FEATURE_KEYS):
            bar = heat_bar(self.feats[:, k], FEATURE_COLORS[key],
                           width=8.5, height=0.32)
            lab = inline_math(FEATURE_NAMES[key], color=FEATURE_COLORS[key])
            lab.scale(1.05)
            row = VGroup(lab, bar).arrange(RIGHT, buff=0.25)
            bars_group.add(row)
        bars_group.arrange(DOWN, buff=0.20).move_to([0, -0.10, 0])

        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.10) for r in bars_group],
                              lag_ratio=0.10, run_time=2.0))
        hold(self, 0.8)

        tag = body_text(
            "Every one is a function of distances and angles only.",
            color=ACCENT,
        ).move_to([0, -3.0, 0])
        self.play(Write(tag), run_time=1.0)
        hold(self, 1.8)

        transition(self)
        seal_narration(self, "scene_03")
