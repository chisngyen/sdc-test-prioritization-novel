"""
Scene 05 -- SE2RoadNet architecture (block diagram).

A clean, two-row block diagram: input pipeline on the top row, head on
the bottom row, six InvariantBlocks in the middle.

Render:  manim -pql scene_05_architecture.py Architecture
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Line, Arrow,
    Write, FadeIn, FadeOut, Create, LaggedStart, GrowArrow,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, BLUE_A, BLUE_C, YELLOW, GREEN_A, GREY_A, RED, ORANGE,
    YELLOW_A,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import (
    TEXT, MUTED, PRIMARY, ACCENT, GOOD, WARN, BAD, RULE,
    section_header, transition, hold,
    title, subtitle, body_text, caption,
    inline_math,
    panel, accent_box, flow_arrow,
    attach_narration, seal_narration,
    MATH_BODY, MATH_INLINE,
)


def lyr_block(top: str, shape: str, *, color=BLUE_C,
              width: float = 2.3, height: float = 1.05) -> VGroup:
    box = RoundedRectangle(
        width=width, height=height, corner_radius=0.10,
        stroke_color=color, stroke_width=2.5,
        fill_color=color, fill_opacity=0.10,
    )
    name = Text(top, font_size=20, color=color, weight="BOLD")
    name.move_to(box.get_center() + UP * 0.22)
    sh = MathTex(shape, font_size=22, color=MUTED)
    sh.move_to(box.get_center() + DOWN * 0.22)
    return VGroup(box, name, sh)


def connector(a, b, *, color=MUTED) -> Arrow:
    return Arrow(
        a.get_right(), b.get_left(),
        buff=0.08, stroke_width=3.5, color=color,
        max_tip_length_to_length_ratio=0.22,
    )


class Architecture(Scene):
    def construct(self):
        attach_narration(self, "scene_05")
        head = title("SE2RoadNet -- bird's-eye view")
        ul = Line(
            head.get_corner(DOWN + LEFT) + DOWN * 0.10,
            head.get_corner(DOWN + RIGHT) + DOWN * 0.10,
            color=PRIMARY, stroke_width=2,
        )
        sub = MathTex(
            r"(L,\,7) \to (L,\,192) \to (L{+}1,\,192) \to \cdots \to \mathbb{R}",
            font_size=28, color=PRIMARY,
        ).move_to([0, 2.55, 0])
        self.play(Write(head), Create(ul), run_time=0.8)
        self.play(FadeIn(sub, shift=DOWN * 0.10), run_time=0.5)

        # ============================================== TOP ROW ============
        x_in = lyr_block("7-ch invariant input", r"(L,\,7)", color=WARN, width=2.7)
        proj = lyr_block("Linear + LN + GELU", r"(L,\,192)", color=BLUE_C, width=2.6)
        cls  = lyr_block("prepend CLS", r"(L{+}1,\,192)", color=YELLOW_A, width=2.5)

        top_row = VGroup(x_in, proj, cls).arrange(RIGHT, buff=0.55)
        top_row.move_to([0, 1.30, 0])
        a1 = connector(x_in, proj)
        a2 = connector(proj, cls)

        self.play(FadeIn(x_in, shift=RIGHT * 0.2), run_time=0.5)
        self.play(GrowArrow(a1), run_time=0.35)
        self.play(FadeIn(proj, shift=RIGHT * 0.2), run_time=0.5)
        self.play(GrowArrow(a2), run_time=0.35)
        self.play(FadeIn(cls, shift=RIGHT * 0.2), run_time=0.5)

        # ============================================ MIDDLE ROW ===========
        block_row = VGroup()
        for k in range(6):
            box = RoundedRectangle(
                width=1.10, height=1.20, corner_radius=0.10,
                stroke_color=BLUE_C, stroke_width=2.5,
                fill_color=BLUE_C, fill_opacity=0.16,
            )
            lab = Text(f"block {k+1}", font_size=14, color=BLUE_A)
            lab.move_to(box.get_center())
            block_row.add(VGroup(box, lab))
        block_row.arrange(RIGHT, buff=0.22).move_to([0, -0.35, 0])

        blocks_caption = MathTex(r"6 \times \text{InvariantBlock}",
                                 font_size=26, color=BLUE_C)
        blocks_caption.next_to(block_row, LEFT, buff=0.55)

        # Connect CLS row down into block 1
        down_arrow = Arrow(
            cls.get_bottom() + DOWN * 0.04,
            block_row[0].get_top() + UP * 0.04,
            buff=0.06, stroke_width=3.5, color=GREY_A,
            max_tip_length_to_length_ratio=0.20,
        )
        self.play(GrowArrow(down_arrow), run_time=0.4)
        self.play(Write(blocks_caption),
                  LaggedStart(*[FadeIn(b, shift=UP * 0.10) for b in block_row],
                              lag_ratio=0.08, run_time=1.8))
        hold(self, 0.3)

        block_shape = inline_math(r"\text{shape stays } (L{+}1,\,192)",
                                  color=MUTED).next_to(block_row, DOWN, buff=0.25)
        self.play(FadeIn(block_shape, shift=UP * 0.10), run_time=0.5)

        # ============================================ BOTTOM ROW ==========
        head_grp = VGroup(
            lyr_block("take CLS",       r"(1,\,192)", color=ACCENT,   width=1.85, height=0.90),
            lyr_block("LN",             r"(1,\,192)", color=GREEN_A, width=1.25, height=0.90),
            lyr_block(r"Linear 192 to 64", r"(1,\,64)",  color=GREEN_A, width=2.30, height=0.90),
            lyr_block("GELU + Dropout",   r"(1,\,64)",  color=GREEN_A, width=2.15, height=0.90),
            lyr_block(r"Linear 64 to 1",   r"(1,\,1)",   color=BAD,    width=2.0,  height=0.90),
        ).arrange(RIGHT, buff=0.22)
        head_grp.scale_to_fit_width(11.5)
        head_grp.move_to([0, -2.00, 0])

        block_to_head = Arrow(
            block_row[-1].get_bottom() + DOWN * 0.04,
            head_grp[0].get_top() + UP * 0.05,
            buff=0.06, stroke_width=3.5, color=GREY_A,
            max_tip_length_to_length_ratio=0.20,
        )
        self.play(FadeOut(block_shape), run_time=0.3)
        self.play(GrowArrow(block_to_head), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(h, shift=LEFT * 0.10) for h in head_grp],
                              lag_ratio=0.10, run_time=1.8))

        # Output sits to the right of the head, on the same row -- keeps the
        # bottom area free for the teaser caption.
        out_box = RoundedRectangle(
            width=2.0, height=0.75, corner_radius=0.12,
            stroke_color=BAD, stroke_width=3,
            fill_color=BAD, fill_opacity=0.15,
        )
        out_lbl = MathTex(r"P(\text{FAIL})", font_size=26, color=BAD)
        out_lbl.move_to(out_box.get_center())
        out_grp = VGroup(out_box, out_lbl)
        out_grp.next_to(head_grp, DOWN, buff=0.40)
        arrow_to_out = Arrow(
            head_grp[-1].get_bottom(), out_box.get_top(),
            buff=0.05, stroke_width=4, color=BAD,
            max_tip_length_to_length_ratio=0.25,
        )
        self.play(GrowArrow(arrow_to_out), FadeIn(out_grp, scale=1.05), run_time=0.7)
        hold(self, 1.0)

        # zoom marker on one block -> tease the next scene
        target = block_row[2]
        ring = accent_box(target, color=ACCENT, buff=0.10, stroke_width=3.5)
        self.play(Create(ring), run_time=0.7)
        hold(self, 1.0)

        transition(self)
        seal_narration(self, "scene_05")

        transition(self)
