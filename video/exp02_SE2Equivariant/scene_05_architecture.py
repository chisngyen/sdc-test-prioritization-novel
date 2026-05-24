"""
Scene 05 -- SE2RoadNet architecture (block diagram).

We zoom out and watch the whole network from input tensor to scalar
FAIL score, then zoom into one of the six identical InvariantBlocks so
the next scene can dissect it.

Render:
    manim -pql scene_05_architecture.py Architecture
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Circle, Dot, Line, Arrow,
    DashedLine, Triangle,
    Write, FadeIn, FadeOut, Create, Uncreate, ReplacementTransform,
    Indicate, Flash, LaggedStart, Transform, AnimationGroup, GrowArrow,
    ValueTracker, always_redraw, DecimalNumber,
    UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL, ORIGIN, PI, DEGREES,
    WHITE, BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E,
    YELLOW, YELLOW_A, ORANGE, RED, RED_A,
    GREEN, GREEN_A, GREY, GREY_A, GREY_B, GREY_C, GOLD, PINK, MAROON, TEAL,
    PURPLE, PURPLE_A,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))


# ----------------------------------------------------------------------------- #
# Helpers for layer blocks
# ----------------------------------------------------------------------------- #
def layer_block(label: str, shape: str, *, color=BLUE_B, width=2.2, height=0.9):
    """A rounded rectangle with a layer name on top and the tensor shape below."""
    box = RoundedRectangle(
        width=width, height=height, corner_radius=0.12,
        stroke_color=color, stroke_width=3,
        fill_color=color, fill_opacity=0.18,
    )
    name = Text(label, font_size=20, color=color, weight="BOLD")
    name.move_to(box.get_center() + UP * 0.18)
    sh = MathTex(shape, font_size=20, color=GREY_A)
    sh.move_to(box.get_center() + DOWN * 0.20)
    return VGroup(box, name, sh)


def flow_arrow(a, b, color=GREY_A):
    return Arrow(
        a.get_right(), b.get_left(), buff=0.08,
        stroke_width=4, color=color,
        max_tip_length_to_length_ratio=0.25,
    )


class Architecture(Scene):
    def construct(self):
        # ---------- title --------------------------------------------------- #
        title = Text("SE2RoadNet — bird's-eye view",
                     font_size=36, color=WHITE).to_edge(UP, buff=0.45)
        sub = MathTex(
            r"(L,7) \to (L+1,192) \to \cdots \to \mathbb{R}",
            font_size=26, color=BLUE_A,
        ).next_to(title, DOWN, buff=0.2)
        self.play(Write(title), Write(sub))
        self.wait(0.4)

        # ---------- input ---------------------------------------------------- #
        x_in = layer_block(
            "7-ch invariant input", r"(L,\,7)", color=ORANGE, width=2.7,
        )
        x_in.to_edge(LEFT, buff=0.6).shift(DOWN * 0.5)
        self.play(FadeIn(x_in, shift=RIGHT * 0.3))

        # ---------- projection ---------------------------------------------- #
        proj = layer_block(
            "Linear + LN + GELU", r"(L,\,192)", color=BLUE_C, width=2.5,
        ).next_to(x_in, RIGHT, buff=0.7)
        arrow1 = flow_arrow(x_in, proj)
        self.play(GrowArrow(arrow1))
        self.play(FadeIn(proj, shift=RIGHT * 0.3))

        # ---------- CLS prepend --------------------------------------------- #
        cls_box = layer_block(
            "prepend CLS", r"(L{+}1,\,192)", color=YELLOW_A, width=2.4,
        ).next_to(proj, RIGHT, buff=0.7)
        arrow2 = flow_arrow(proj, cls_box)
        self.play(GrowArrow(arrow2))
        self.play(FadeIn(cls_box, shift=RIGHT * 0.3))
        # small visual for CLS: a yellow square plus L grey squares
        cls_visual = self._cls_strip()
        cls_visual.next_to(cls_box, DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(s, scale=1.4) for s in cls_visual],
                              lag_ratio=0.05, run_time=0.9))
        self.wait(0.4)

        # ---------- six InvariantBlocks ------------------------------------ #
        blocks_title = MathTex(r"\times\,6 \;\text{InvariantBlocks}",
                               font_size=24, color=BLUE_B)
        blocks_title.to_edge(DOWN, buff=2.8).shift(LEFT * 1.6)
        self.play(Write(blocks_title))

        block_row = VGroup()
        for k in range(6):
            b = RoundedRectangle(
                width=1.0, height=1.4, corner_radius=0.12,
                stroke_color=BLUE_B, stroke_width=3,
                fill_color=BLUE_B, fill_opacity=0.22,
            )
            lab = Text(f"block {k+1}", font_size=14, color=BLUE_A)
            lab.move_to(b.get_center())
            block_row.add(VGroup(b, lab))
        block_row.arrange(RIGHT, buff=0.18)
        block_row.next_to(blocks_title, RIGHT, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in block_row],
                              lag_ratio=0.10, run_time=1.8))

        # connect CLS box to first block & last block to next stage
        connect_in = Arrow(
            cls_box.get_bottom() + DOWN * 0.1,
            block_row[0].get_top() + UP * 0.05,
            buff=0.08, stroke_width=4, color=GREY_A,
            max_tip_length_to_length_ratio=0.18,
        )
        self.play(GrowArrow(connect_in))
        self.wait(0.4)

        # equation under the block row -- emphasise weight-sharing per layer
        wshared = Tex(
            r"Same shape, same params count, different weights per block.",
            font_size=22, color=GREY_A,
        ).next_to(block_row, DOWN, buff=0.35)
        self.play(FadeIn(wshared, shift=UP * 0.1))
        self.wait(0.8)

        # ---------- head ----------------------------------------------------- #
        head_grp = VGroup(
            layer_block("take CLS", r"(1,\,192)", color=YELLOW, width=2.0),
            layer_block("LN", r"(1,\,192)", color=GREEN_A, width=1.4),
            layer_block(r"Linear 192 $\to$ 64", r"(1,\,64)", color=GREEN_A, width=2.1),
            layer_block("GELU + Dropout", r"(1,\,64)", color=GREEN_A, width=2.1),
            layer_block(r"Linear 64 $\to$ 1", r"(1,\,1)", color=RED, width=1.9),
        ).arrange(RIGHT, buff=0.25).to_edge(RIGHT, buff=0.5).shift(UP * 1.1)

        connect_out = Arrow(
            block_row[-1].get_right(),
            head_grp[0].get_left(),
            buff=0.1, stroke_width=4, color=GREY_A,
            max_tip_length_to_length_ratio=0.18,
        )
        # If head goes off the visible right edge, drop it to the same row
        head_grp.scale_to_fit_width(7.0)
        head_grp.next_to(block_row, RIGHT, buff=0.5).shift(UP * 0.0)
        connect_out = Arrow(
            block_row[-1].get_right(),
            head_grp[0].get_left(),
            buff=0.08, stroke_width=4, color=GREY_A,
            max_tip_length_to_length_ratio=0.22,
        )

        self.play(GrowArrow(connect_out))
        self.play(LaggedStart(*[FadeIn(h, shift=LEFT * 0.15) for h in head_grp],
                              lag_ratio=0.10, run_time=2.0))

        # ---------- output --------------------------------------------------- #
        out_box = RoundedRectangle(
            width=2.0, height=0.8, corner_radius=0.12,
            stroke_color=RED, stroke_width=3,
            fill_color=RED, fill_opacity=0.18,
        )
        out_lbl = MathTex(r"P(\text{FAIL})", font_size=26, color=RED)
        out_lbl.move_to(out_box.get_center())
        out_grp = VGroup(out_box, out_lbl)
        out_grp.next_to(head_grp, DOWN, buff=0.5)

        arrow_to_out = Arrow(
            head_grp[-1].get_bottom(), out_box.get_top(),
            buff=0.08, stroke_width=4, color=RED,
            max_tip_length_to_length_ratio=0.25,
        )
        self.play(GrowArrow(arrow_to_out))
        self.play(FadeIn(out_grp, scale=1.1))
        self.wait(0.8)

        # ---------- zoom into one block: tease scene 06 --------------------- #
        zoom_msg = Text(
            "Next: inside one InvariantBlock.",
            font_size=24, color=YELLOW,
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(zoom_msg))

        # highlight block 3 with a magnification ring
        target = block_row[2]
        ring = Circle(radius=0.95, color=YELLOW, stroke_width=4)
        ring.move_to(target.get_center())
        self.play(Create(ring), Flash(target, color=YELLOW,
                                      flash_radius=0.9, num_lines=14), run_time=1.2)
        self.wait(1.2)

        # clear
        self.play(FadeOut(VGroup(
            title, sub, x_in, proj, cls_box, cls_visual,
            arrow1, arrow2, connect_in, connect_out, arrow_to_out,
            block_row, blocks_title, wshared,
            head_grp, out_grp, ring, zoom_msg,
        )))

    # ---- the CLS / token strip used to visualise the prepend ---------------- #
    def _cls_strip(self) -> VGroup:
        sq_cls = Square(side_length=0.30, color=YELLOW, fill_color=YELLOW,
                        fill_opacity=0.9, stroke_width=2)
        cls_t = Text("CLS", font_size=12, color="#1a1a1a")
        cls_t.move_to(sq_cls.get_center())
        cls_grp = VGroup(sq_cls, cls_t)

        token_grp = VGroup(cls_grp)
        for _ in range(10):
            sq = Square(side_length=0.28, color=BLUE_C,
                        fill_color=BLUE_C, fill_opacity=0.35, stroke_width=1.5)
            token_grp.add(sq)
        dots = MathTex(r"\cdots", font_size=22, color=GREY_A)
        token_grp.add(dots)
        token_grp.arrange(RIGHT, buff=0.05)
        return token_grp
