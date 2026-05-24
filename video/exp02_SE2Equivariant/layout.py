"""
Screen-layout discipline for the Exp 02 video.

Every scene reserves these vertical zones, top to bottom:

    TITLE_Y       ~ +3.20   (one-line, font 32-40)
    SUBTITLE_Y    ~ +2.55   (one-line, font 22-26)
    CANVAS_TOP_Y  ~ +2.00   (top of the main drawing area)
    CANVAS_MID_Y  ~  0.00
    CANVAS_BOT_Y  ~ -2.00
    FOOTER_Y      ~ -3.20   (one-line caption)

The scene area is 14.22 wide x 8.00 tall in Manim default 16:9.

Use `clear(scene)` between conceptual phases.  Pile-on overlap is the
single biggest source of the video looking amateurish, so we are strict
about removing what we're done with.
"""
from __future__ import annotations

from manim import (
    VGroup, Mobject, FadeOut, FadeIn, Scene, UP, DOWN, LEFT, RIGHT,
    WHITE, GREY_A, GREY_B, BLUE_A, YELLOW,
    Text, MathTex, Tex,
)


TITLE_Y    =  3.25
SUBTITLE_Y =  2.55
CANVAS_TOP =  2.00
CANVAS_MID =  0.00
CANVAS_BOT = -2.00
FOOTER_Y   = -3.30

CANVAS_LEFT_X  = -6.5
CANVAS_RIGHT_X = +6.5


def title(text: str, *, font_size: int = 36, color=WHITE) -> Text:
    t = Text(text, font_size=font_size, color=color, weight="BOLD")
    t.move_to([0.0, TITLE_Y, 0.0])
    return t


def subtitle(text: str, *, font_size: int = 24, color=BLUE_A,
             tex: bool = False) -> Mobject:
    """`tex=True` if the text mixes prose and inline math (`$...$`).
    `tex=False` (default) for plain text -- avoids LaTeX entirely."""
    if tex:
        m = Tex(text, font_size=font_size + 4, color=color)
    else:
        m = Text(text, font_size=font_size, color=color)
    m.move_to([0.0, SUBTITLE_Y, 0.0])
    return m


def footer(text: str, *, font_size: int = 22, color=GREY_A, tex: bool = False) -> Mobject:
    if tex:
        m = Tex(text, font_size=font_size, color=color)
    else:
        m = Text(text, font_size=font_size, color=color, slant="ITALIC")
    m.move_to([0.0, FOOTER_Y, 0.0])
    return m


def at_canvas(mob: Mobject, *, x: float = 0.0, y: float = CANVAS_MID) -> Mobject:
    mob.move_to([x, y, 0.0])
    return mob


def clear(scene: Scene, *, keep=()):
    """Fade out every active mobject in the scene except those in `keep`."""
    keep_set = set(keep)
    survivors = [m for m in scene.mobjects if m not in keep_set]
    if survivors:
        scene.play(*[FadeOut(m) for m in survivors], run_time=0.5)


def replace_title(scene: Scene, old: Mobject, new_text: str, **kwargs) -> Text:
    """Swap the screen's current title for a new one in one play call."""
    new = title(new_text, **kwargs)
    scene.play(FadeOut(old, shift=DOWN * 0.15),
               FadeIn(new, shift=DOWN * 0.15), run_time=0.6)
    return new
