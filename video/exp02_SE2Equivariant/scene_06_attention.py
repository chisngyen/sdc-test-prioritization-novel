"""
Scene 06 -- Inside one InvariantBlock (concept-level).

Detailed numeric computation lives in scene_06b_compute.py.  This scene
stays at the conceptual level: why the relative-arclength bias is the
piece that makes the *attention* mechanism rotation-invariant, and how
the rest of the block (residual + FFN) is just standard transformer
machinery.

We follow a strict screen-clear discipline so nothing overlaps.

Render:
    manim -pql scene_06_attention.py AttentionBlock
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Circle, Dot, Line, Arrow,
    DashedLine,
    Write, FadeIn, FadeOut, Create, ReplacementTransform,
    Indicate, Flash, LaggedStart, Transform, GrowArrow,
    ValueTracker, always_redraw, DecimalNumber,
    UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL, ORIGIN, PI, DEGREES,
    WHITE, BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E,
    YELLOW, YELLOW_A, ORANGE, RED, RED_A,
    GREEN, GREEN_A, GREY, GREY_A, GREY_B, GREY_C, GOLD, PINK, MAROON, TEAL,
    PURPLE, PURPLE_A,
)
from manim.utils.color import interpolate_color

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from layout import title, subtitle, footer, clear


def lyr_box(name: str, *, color=BLUE_C, width=2.4, height=0.7,
            font_size: int = 20) -> VGroup:
    box = RoundedRectangle(
        width=width, height=height, corner_radius=0.12,
        stroke_color=color, stroke_width=3,
        fill_color=color, fill_opacity=0.18,
    )
    lbl = Text(name, font_size=font_size, color=color, weight="BOLD")
    lbl.move_to(box.get_center())
    return VGroup(box, lbl)


def heat_matrix(values: np.ndarray, *, cell=0.30,
                color_lo=BLUE_E, color_hi=YELLOW) -> VGroup:
    v = np.asarray(values, dtype=np.float64)
    vmin, vmax = float(v.min()), float(v.max())
    span = max(vmax - vmin, 1e-9)
    rows, cols = v.shape
    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            alpha = float(np.clip((v[i, j] - vmin) / span, 0, 1))
            c = interpolate_color(color_lo, color_hi, alpha)
            sq = Square(side_length=cell, stroke_width=0.4,
                        stroke_color=GREY_C, fill_color=c, fill_opacity=0.95)
            sq.move_to([(j - (cols - 1) / 2) * cell,
                        ((rows - 1) / 2 - i) * cell, 0.0])
            grid.add(sq)
    return grid


class AttentionBlock(Scene):
    def construct(self):
        self._part_a_block_diagram()
        self._part_b_why_rel_bias_is_invariant()
        self._part_c_outro()

    # ------------------------------------------------------- a. block diagram
    def _part_a_block_diagram(self):
        t = title("Inside one InvariantBlock")
        s = subtitle("standard transformer block, with one extra twist in the attention")
        self.play(Write(t), FadeIn(s, shift=UP * 0.15))

        # Lay out the chain
        x_in = MathTex(r"H", font_size=30, color=ORANGE)
        ln1 = lyr_box("LayerNorm", color=GREEN_A)
        mha = lyr_box("Multi-Head Attention", color=YELLOW, width=2.9)
        add1 = lyr_box("+", color=GREY_A, width=0.6)
        ln2 = lyr_box("LayerNorm", color=GREEN_A)
        ffn = lyr_box("FFN", color=BLUE_C, width=1.4)
        add2 = lyr_box("+", color=GREY_A, width=0.6)
        x_out = MathTex(r"H'", font_size=30, color=ORANGE)

        chain = VGroup(x_in, ln1, mha, add1, ln2, ffn, add2, x_out)
        chain.arrange(RIGHT, buff=0.24).move_to([0, 0.0, 0])

        # Step-by-step reveal
        self.play(FadeIn(x_in, shift=RIGHT * 0.15))
        prev = x_in
        for cur in chain[1:]:
            arr = Arrow(prev.get_right(), cur.get_left(), buff=0.04,
                        stroke_width=3, color=GREY_A,
                        max_tip_length_to_length_ratio=0.25)
            self.play(GrowArrow(arr), FadeIn(cur, shift=RIGHT * 0.1), run_time=0.40)
            prev = cur

        # Residual streams (curved/dashed up over the +)
        skip1 = DashedLine(
            ln1.get_top() + UP * 0.05, add1.get_top() + UP * 0.05,
            color=YELLOW, stroke_width=2,
        )
        skip2 = DashedLine(
            ln2.get_top() + UP * 0.05, add2.get_top() + UP * 0.05,
            color=YELLOW, stroke_width=2,
        )
        skip_lbl = Text("residual skip connections",
                        font_size=22, color=YELLOW).move_to([0, 1.4, 0])
        self.play(Create(skip1), Create(skip2), Write(skip_lbl))
        self.wait(0.5)

        # Pull out the MHA block as the focus of the rest of the scene
        focus = Rectangle(
            width=mha[0].width + 0.2, height=mha[0].height + 0.2,
            stroke_color=YELLOW, stroke_width=4, fill_opacity=0,
        ).move_to(mha.get_center())
        zoom_msg = footer("Everything else is standard. The interesting bit is in the yellow box.")
        self.play(Create(focus), Write(zoom_msg))
        self.wait(1.6)

        clear(self)

    # ---------------------------------------------- b. why B^rel is invariant
    def _part_b_why_rel_bias_is_invariant(self):
        t = title("Why the attention is rotation-invariant")
        s = subtitle("the attention bias depends only on differences of arclength")
        self.play(Write(t), FadeIn(s, shift=UP * 0.15))

        # Attention formula -- show both terms, box the right one
        eq = MathTex(
            r"\mathrm{Attn} = "
            r"\mathrm{softmax}\!\left("
            r"\underbrace{\frac{Q K^{\top}}{\sqrt{d}}}_{\text{standard}}"
            r"\;+\;"
            r"\underbrace{B^{\mathrm{rel}}}_{\text{ours}}"
            r"\right) V",
            font_size=36,
        ).move_to([0, 1.1, 0])
        self.play(Write(eq), run_time=2.0)
        self.wait(0.5)

        # The chain that builds B^rel: s -> ds -> sin(ds*omega) -> MLP -> B
        chain_stages = [
            (r"s_i", PURPLE_A),
            (r"\Delta s_{ij} = s_i - s_j", PURPLE_A),
            (r"\sin\!\bigl(\Delta s_{ij}\,\omega\bigr)", ORANGE),
            (r"\mathrm{MLP}", BLUE_C),
            (r"B^{\mathrm{rel}}_{ij}", YELLOW),
        ]
        chain_grp = VGroup()
        for txt, col in chain_stages:
            chip = RoundedRectangle(
                width=2.2, height=0.7, corner_radius=0.12,
                stroke_color=col, stroke_width=2.5,
                fill_color=col, fill_opacity=0.15,
            )
            tex = MathTex(txt, font_size=22, color=col).move_to(chip.get_center())
            chain_grp.add(VGroup(chip, tex))
        chain_grp.arrange(RIGHT, buff=0.30).move_to([0, -0.5, 0])

        arrows = VGroup()
        for prev, cur in zip(chain_grp[:-1], chain_grp[1:]):
            arrows.add(Arrow(prev.get_right(), cur.get_left(),
                             buff=0.06, stroke_width=3, color=GREY_A,
                             max_tip_length_to_length_ratio=0.22))
        self.play(LaggedStart(*[FadeIn(c, scale=1.05) for c in chain_grp],
                              *[GrowArrow(a) for a in arrows],
                              lag_ratio=0.10, run_time=2.4))
        self.wait(0.6)

        # The critical claim -- s_i and ds_ij are intrinsic
        claim_box = RoundedRectangle(
            width=11.5, height=1.05, corner_radius=0.15,
            stroke_color=GREEN_A, stroke_width=3,
            fill_color=GREEN_A, fill_opacity=0.08,
        ).move_to([0, -2.05, 0])
        claim_text = Tex(
            r"Rotation moves each $\mathbf{r}_i$, but it never changes "
            r"the \emph{arclength} along the road.  "
            r"$\Delta s_{ij}$ is invariant $\Rightarrow$ "
            r"$B^{\mathrm{rel}}_{ij}$ is invariant $\Rightarrow$ "
            r"attention is invariant.",
            font_size=22, color=WHITE,
        ).move_to(claim_box.get_center())
        self.play(FadeIn(claim_box), Write(claim_text), run_time=2.2)
        self.wait(2.4)

        clear(self)

    # ------------------------------------------------------------ c. outro
    def _part_c_outro(self):
        t = title("Detailed numerical walkthrough next")
        m = Text(
            "Watch one tensor flow end-to-end through the model,\n"
            "every multiplication shown with actual numbers.",
            font_size=26, color=WHITE,
        ).move_to([0, 0.0, 0])
        cite = footer("scene_06b_compute.py")
        self.play(Write(t))
        self.play(FadeIn(m, shift=UP * 0.15))
        self.play(Write(cite))
        self.wait(1.8)
        clear(self)
