"""
Scene 06 -- Inside one InvariantBlock (concept level).

Two beats:
    A.  the block is standard transformer machinery (LN -> MHA -> + -> LN -> FFN -> +),
    B.  the *one* twist that makes the attention rotation-invariant is the
        relative-arclength bias B_rel.

Numeric detail lives in scene_06b_compute.py.

Render:  manim -pql scene_06_attention.py AttentionBlock
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Line, DashedLine, Arrow,
    Write, FadeIn, FadeOut, Create, LaggedStart, GrowArrow,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, BLUE_A, BLUE_C, YELLOW, GREEN_A, GREY_A, ORANGE,
    PURPLE_A, TEAL,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import (
    TEXT, MUTED, PRIMARY, ACCENT, GOOD, WARN, BAD, RULE,
    section_header, transition, hold,
    title, subtitle, body_text, caption,
    big_formula, body_formula, inline_math,
    panel, accent_box, chip,
    attach_narration, seal_narration,
    MATH_BIG, MATH_BODY,
)


def lyr_box(name: str, *, color=BLUE_C, width: float = 2.4,
            height: float = 0.75, font_size: int = 20) -> VGroup:
    box = RoundedRectangle(
        width=width, height=height, corner_radius=0.10,
        stroke_color=color, stroke_width=2.5,
        fill_color=color, fill_opacity=0.14,
    )
    lab = Text(name, font_size=font_size, color=color, weight="BOLD")
    lab.move_to(box.get_center())
    return VGroup(box, lab)


class AttentionBlock(Scene):
    def construct(self):
        attach_narration(self, "scene_06")
        self._block_diagram()
        self._why_rel_bias_invariant()
        self._outro()
        # Hold the outro card under the narration tail, then wipe (no black tail).
        seal_narration(self, "scene_06")
        transition(self)

    # -------------------------------------- A. block diagram -----------
    def _block_diagram(self):
        head = section_header(
            self, "Inside one InvariantBlock",
            "Standard transformer machinery, with one twist in the attention.",
        )
        self.head = head

        x_in = MathTex(r"H", font_size=32, color=WARN)
        ln1  = lyr_box("LayerNorm", color=GREEN_A, width=1.95)
        mha  = lyr_box("Multi-Head Attention", color=ACCENT, width=2.85)
        add1 = lyr_box("+", color=GREY_A, width=0.55)
        ln2  = lyr_box("LayerNorm", color=GREEN_A, width=1.85)
        ffn  = lyr_box("FFN", color=BLUE_C, width=1.20)
        add2 = lyr_box("+", color=GREY_A, width=0.55)
        x_out= MathTex(r"H'", font_size=32, color=WARN)

        chain = VGroup(x_in, ln1, mha, add1, ln2, ffn, add2, x_out)
        chain.arrange(RIGHT, buff=0.22).move_to([0, 0.0, 0])

        self.play(FadeIn(x_in, shift=RIGHT * 0.15), run_time=0.4)
        prev = x_in
        for cur in chain[1:]:
            arr = Arrow(prev.get_right(), cur.get_left(), buff=0.04,
                        stroke_width=3, color=GREY_A,
                        max_tip_length_to_length_ratio=0.22)
            self.play(GrowArrow(arr), FadeIn(cur, shift=RIGHT * 0.10),
                      run_time=0.35)
            prev = cur

        skip1 = DashedLine(
            ln1.get_top() + UP * 0.05, add1.get_top() + UP * 0.05,
            color=ACCENT, stroke_width=2,
        )
        skip2 = DashedLine(
            ln2.get_top() + UP * 0.05, add2.get_top() + UP * 0.05,
            color=ACCENT, stroke_width=2,
        )
        skip_lbl = caption("residual skip connections",
                           color=ACCENT, italic=False).move_to([0, 1.50, 0])
        self.play(Create(skip1), Create(skip2), Write(skip_lbl), run_time=0.8)
        hold(self, 1.2)

        ring = accent_box(mha, color=ACCENT, buff=0.10, stroke_width=3)
        zoom_msg = caption(
            "Everything else is standard.  The interesting bit is in the yellow box.",
            color=MUTED,
        ).move_to([0, -2.20, 0])
        self.play(Create(ring), Write(zoom_msg), run_time=1.0)
        hold(self, 2.6)

        transition(self)

    # ------------------------- B. why B_rel makes attn invariant -----------
    def _why_rel_bias_invariant(self):
        head = section_header(
            self, "Why the attention is rotation-invariant",
            "The attention bias depends only on differences of arclength.",
        )

        eq = MathTex(
            r"\mathrm{Attn} \;=\; "
            r"\mathrm{softmax}\!\left("
            r"\underbrace{\frac{Q K^{\top}}{\sqrt{d}}}_{\text{standard}}"
            r"\;+\;"
            r"\underbrace{B^{\mathrm{rel}}}_{\text{ours}}"
            r"\right) V",
            font_size=38,
        ).move_to([0, 1.10, 0])
        self.play(Write(eq), run_time=1.8)
        hold(self, 1.2)

        stages = [
            (r"s_i",                                       PURPLE_A),
            (r"\Delta s_{ij} = s_i - s_j",                 PURPLE_A),
            (r"\sin\!\bigl(\Delta s_{ij}\,\omega\bigr)",   TEAL),
            (r"\mathrm{MLP}",                              BLUE_C),
            (r"B^{\mathrm{rel}}_{ij}",                     ACCENT),
        ]
        chips = VGroup()
        for txt, col in stages:
            chips.add(chip(txt, color=col, width=2.2, height=0.70,
                           font_size=20, math=True, fill_opacity=0.12))
        chips.arrange(RIGHT, buff=0.24).move_to([0, -0.60, 0])

        arrows = VGroup(*[
            Arrow(a.get_right(), b.get_left(), buff=0.06, stroke_width=3,
                  color=GREY_A, max_tip_length_to_length_ratio=0.22)
            for a, b in zip(chips[:-1], chips[1:])
        ])
        self.play(
            LaggedStart(*[FadeIn(c, scale=1.05) for c in chips],
                        *[GrowArrow(a) for a in arrows],
                        lag_ratio=0.09, run_time=2.2),
        )
        hold(self, 1.4)

        claim_box = RoundedRectangle(
            width=12.0, height=1.20, corner_radius=0.15,
            stroke_color=GOOD, stroke_width=3,
            fill_color=GOOD, fill_opacity=0.06,
        ).move_to([0, -2.20, 0])
        claim_text = Tex(
            r"Rotation moves each $\mathbf{r}_i$, but it never changes the "
            r"\emph{arclength} along the road.  "
            r"$\Delta s_{ij}$ invariant $\Rightarrow$ "
            r"$B^{\mathrm{rel}}_{ij}$ invariant $\Rightarrow$ "
            r"attention invariant.",
            font_size=24, color=TEXT,
        ).move_to(claim_box.get_center())
        self.play(FadeIn(claim_box), FadeIn(claim_text), run_time=1.4)
        hold(self, 4.6)

        transition(self)

    # ---------------------------------------------- C. outro -------------
    def _outro(self):
        t = title("Next: every multiplication, with real numbers.")
        ul = Line(
            t.get_corner(DOWN + LEFT) + DOWN * 0.10,
            t.get_corner(DOWN + RIGHT) + DOWN * 0.10,
            color=PRIMARY, stroke_width=2,
        )
        msg = Text(
            "One tensor, end-to-end through SE2RoadNet,\n"
            "every step shown with the actual matrix.",
            font_size=26, color=TEXT, line_spacing=1.2,
        ).move_to([0, 0.0, 0])
        self.play(Write(t), Create(ul), run_time=0.8)
        self.play(FadeIn(msg, shift=UP * 0.15), run_time=0.8)
        hold(self, 3.0)
