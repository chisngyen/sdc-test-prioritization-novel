"""
Visual theme for the Exp 02 video.  3b1b-flavoured: limited palette, sharp
rectangles, generous whitespace, single accent per frame.

Every scene goes through these helpers so colours, font sizes, and
positions stay consistent across the 9-scene pipeline.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Sequence

from manim import (
    VGroup, VMobject, Mobject, Scene,
    Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Line, DashedLine, SurroundingRectangle,
    FadeIn, FadeOut, Write, Create, AnimationGroup,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, BLACK, BLUE_A, BLUE_B, BLUE_D, BLUE_E,
    YELLOW, ORANGE, RED, RED_A, RED_E,
    GREEN_A, GREY_A, GREY_B, GREY_C, GREY_D, TEAL,
)


# ---------------------------------------------------------------- palette ---
# Stick to a small, semantic palette.  Any other colour is decoration.
TEXT          = WHITE              # primary body
MUTED         = GREY_A             # captions, footers
PRIMARY       = BLUE_A             # subtitles, neutral structure
ACCENT        = YELLOW             # the thing the viewer should look at
GOOD          = GREEN_A            # things that worked, proofs, deltas = 0
WARN          = ORANGE             # "watch out", contrast
BAD           = RED_A              # baseline failure
RULE          = GREY_C             # axis lines, dividers
PANEL_FILL    = "#0e1a2b"          # background of any solid card
SOFT_FILL     = "#101820"


# ---------------------------------------------------------------- geometry --
TITLE_Y       =  3.30
SUBTITLE_Y    =  2.65
CANVAS_TOP    =  2.10
CANVAS_MID    =  0.00
CANVAS_BOT    = -2.10
FOOTER_Y      = -3.35

CANVAS_LEFT_X  = -6.40
CANVAS_RIGHT_X =  6.40


# ----------------------------------------------------------------- sizes ----
TITLE_FS      = 40
SUBTITLE_FS   = 26
BODY_FS       = 24
CAPTION_FS    = 20
TINY_FS       = 16

MATH_BIG      = 44
MATH_BODY     = 32
MATH_INLINE   = 24
MATH_SMALL    = 20


# ============================================================================
# Title / subtitle / footer.  Always positioned the same way.
# ============================================================================
def title(text: str, *, color=TEXT) -> Text:
    t = Text(text, font_size=TITLE_FS, color=color, weight="BOLD")
    t.move_to([0.0, TITLE_Y, 0.0])
    return t


def subtitle(text: str, *, color=PRIMARY, tex: bool = False) -> Mobject:
    if tex:
        m = Tex(text, color=color)
        m.scale_to_fit_height(0.34)
    else:
        m = Text(text, font_size=SUBTITLE_FS, color=color)
    m.move_to([0.0, SUBTITLE_Y, 0.0])
    return m


def footer(text: str, *, color=MUTED, tex: bool = False) -> Mobject:
    if tex:
        m = Tex(text, color=color)
        m.scale_to_fit_height(0.28)
    else:
        m = Text(text, font_size=CAPTION_FS, color=color, slant="ITALIC")
    m.move_to([0.0, FOOTER_Y, 0.0])
    return m


def section_header(scene: Scene, title_text: str, subtitle_text: str | None = None,
                   *, tex_subtitle: bool = False) -> VGroup:
    """Animate in a title (+ optional subtitle) and return the VGroup so the
    caller can keep / fade it on its own schedule."""
    t = title(title_text)
    underline = Line(
        t.get_corner(DOWN + LEFT) + DOWN * 0.10,
        t.get_corner(DOWN + RIGHT) + DOWN * 0.10,
        color=PRIMARY, stroke_width=2,
    )
    header = VGroup(t, underline)
    if subtitle_text is not None:
        s = subtitle(subtitle_text, tex=tex_subtitle)
        header.add(s)
        scene.play(
            Write(t, run_time=0.9),
            Create(underline, run_time=0.6),
            FadeIn(s, shift=DOWN * 0.12, run_time=0.6),
        )
    else:
        scene.play(
            Write(t, run_time=0.9),
            Create(underline, run_time=0.6),
        )
    return header


def swap_header(scene: Scene, old: VGroup, new_title: str,
                new_subtitle: str | None = None, *, tex_subtitle: bool = False) -> VGroup:
    scene.play(FadeOut(old, shift=UP * 0.2, run_time=0.5))
    return section_header(scene, new_title, new_subtitle, tex_subtitle=tex_subtitle)


# ============================================================================
# Transitions
# ============================================================================
def transition(scene: Scene, *, keep: Sequence[Mobject] = (), run_time: float = 0.55):
    """Fade out everything currently on the scene except `keep`.  No drama,
    just a clean wipe."""
    keep_set = set(keep)
    survivors = [m for m in scene.mobjects if m not in keep_set]
    if survivors:
        scene.play(*[FadeOut(m) for m in survivors], run_time=run_time)


def hold(scene: Scene, seconds: float = 1.0):
    """Tiny alias so the cadence reads at a glance."""
    scene.wait(seconds)


# ============================================================================
# Panels / cards / chips -- sharp-edged, low-fill, single stroke.
# ============================================================================
def panel(*, width: float, height: float, color=PRIMARY,
          fill_opacity: float = 0.06, stroke_width: float = 2.0,
          rounded: bool = False, corner_radius: float = 0.12) -> Mobject:
    if rounded:
        return RoundedRectangle(
            width=width, height=height, corner_radius=corner_radius,
            stroke_color=color, stroke_width=stroke_width,
            fill_color=color, fill_opacity=fill_opacity,
        )
    return Rectangle(
        width=width, height=height,
        stroke_color=color, stroke_width=stroke_width,
        fill_color=color, fill_opacity=fill_opacity,
    )


def value_card(label: str, value: str, *, color=PRIMARY,
               width: float = 8.0, height: float = 0.78,
               label_size: int = BODY_FS, value_size: int = MATH_INLINE,
               value_is_math: bool = True) -> VGroup:
    """A wide row: text on the left, monospaced numeric on the right, thin
    stroke, almost transparent fill.  The workhorse of scoreboards."""
    box = panel(width=width, height=height, color=color,
                fill_opacity=0.05, stroke_width=1.8, rounded=True)
    lab = Text(label, font_size=label_size, color=TEXT)
    lab.move_to(box.get_left() + RIGHT * 0.35, aligned_edge=LEFT)
    if value_is_math:
        val = MathTex(value, font_size=value_size, color=color)
    else:
        val = Text(value, font_size=value_size, color=color)
    val.move_to(box.get_right() + LEFT * 0.35, aligned_edge=RIGHT)
    return VGroup(box, lab, val)


def chip(label: str, *, color=PRIMARY, width: float = 1.6, height: float = 0.55,
         font_size: int = TINY_FS, math: bool = False, fill_opacity: float = 0.18) -> VGroup:
    box = RoundedRectangle(
        width=width, height=height, corner_radius=0.12,
        stroke_color=color, stroke_width=2,
        fill_color=color, fill_opacity=fill_opacity,
    )
    if math:
        lab = MathTex(label, font_size=font_size + 2, color=color)
    else:
        lab = Text(label, font_size=font_size, color=color, weight="BOLD")
    lab.move_to(box.get_center())
    return VGroup(box, lab)


def accent_box(mob: Mobject, *, color=ACCENT, buff: float = 0.12,
               stroke_width: float = 2.5) -> Rectangle:
    return SurroundingRectangle(
        mob, color=color, buff=buff, stroke_width=stroke_width,
    ).set_fill(opacity=0.0)


def divider(width: float = 12.0, *, color=RULE, stroke_width: float = 1.2,
            y: float = 0.0) -> Line:
    return Line([-width / 2, y, 0], [width / 2, y, 0],
                color=color, stroke_width=stroke_width)


# ============================================================================
# Math / text builders with consistent defaults
# ============================================================================
def big_formula(tex_str: str, *, color=TEXT, scale: float = 1.0) -> MathTex:
    m = MathTex(tex_str, font_size=MATH_BIG, color=color)
    if scale != 1.0:
        m.scale(scale)
    return m


def body_formula(tex_str: str, *, color=TEXT) -> MathTex:
    return MathTex(tex_str, font_size=MATH_BODY, color=color)


def inline_math(tex_str: str, *, color=TEXT) -> MathTex:
    return MathTex(tex_str, font_size=MATH_INLINE, color=color)


def body_text(text: str, *, color=TEXT) -> Text:
    return Text(text, font_size=BODY_FS, color=color)


def caption(text: str, *, color=MUTED, italic: bool = True) -> Text:
    return Text(text, font_size=CAPTION_FS, color=color,
                slant="ITALIC" if italic else "NORMAL")


# ============================================================================
# Flow helpers (arrows, connectors)
# ============================================================================
def flow_arrow(a: Mobject, b: Mobject, *, color=MUTED,
               stroke_width: float = 3.5, buff: float = 0.10,
               tip_ratio: float = 0.22):
    from manim import Arrow
    return Arrow(
        a.get_right(), b.get_left(),
        buff=buff, stroke_width=stroke_width, color=color,
        max_tip_length_to_length_ratio=tip_ratio,
    )


def labeled_arrow(a, b, label_text: str, *, color=MUTED,
                  label_size: int = TINY_FS, math: bool = False):
    from manim import Arrow
    ar = Arrow(a, b, buff=0.10, stroke_width=3, color=color,
               max_tip_length_to_length_ratio=0.22)
    if math:
        lab = MathTex(label_text, font_size=label_size + 4, color=color)
    else:
        lab = Text(label_text, font_size=label_size, color=color)
    lab.next_to(ar.get_center(), UP, buff=0.10)
    return VGroup(ar, lab)


# ============================================================================
# Pulse / emphasise -- 3b1b style is gentle, not jarring.
# ============================================================================
def pulse(mob: Mobject, *, scale: float = 1.06, color=None):
    """Returns an Animation that scales the mob up briefly.  Use in place of
    Flash / Indicate for a calmer feel."""
    from manim import Succession
    target = mob.copy()
    if color is not None:
        target.set_color(color)
    target.scale(scale)
    return Succession(
        mob.animate.scale(scale),
        mob.animate.scale(1.0 / scale),
        lag_ratio=1.0,
    )


def color_tex(mob: MathTex, mapping: dict) -> MathTex:
    """Apply set_color_by_tex for every (substring -> colour) in `mapping`."""
    for sub, col in mapping.items():
        mob.set_color_by_tex(sub, col)
    return mob


# ============================================================================
# End-of-scene helpers
# ============================================================================
def end_card(scene: Scene, text: str, *, color=TEXT, run_time: float = 1.6):
    """A simple `Next: ...` card, then fade out."""
    card = Text(text, font_size=32, color=color)
    scene.play(FadeIn(card, shift=UP * 0.15, run_time=0.7))
    scene.wait(run_time)
    scene.play(FadeOut(card, run_time=0.5))


# ============================================================================
# Narration / audio sync
# ============================================================================
_NARRATION_DIR = os.path.join(os.path.dirname(__file__), "narration")
_DURATIONS_PATH = os.path.join(_NARRATION_DIR, "audio_durations.json")


def _load_durations() -> dict:
    if not os.path.isfile(_DURATIONS_PATH):
        return {}
    try:
        with open(_DURATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def attach_narration(scene: Scene, key: str) -> None:
    """Attach the scene's narration MP3 so it plays from t=0.

    Silently no-ops if the file is missing -- the visual scene still works
    when audio is not yet generated, which keeps rapid-iteration sane.
    """
    mp3_rel = os.path.join("narration", f"{key}.mp3")
    mp3_abs = os.path.join(os.path.dirname(__file__), mp3_rel)
    if not os.path.isfile(mp3_abs):
        return
    scene.add_sound(mp3_abs)


def seal_narration(scene: Scene, key: str, *, tail: float = 0.6) -> None:
    """Pad the end of the scene so that the video covers the full audio.

    Reads the duration from `narration/audio_durations.json` (produced by
    `narration/generate.py`).  Adds `wait(gap + tail)` if the scene is
    shorter than the audio.  `tail` gives a small breathing room after
    the voice stops.
    """
    durations = _load_durations()
    if key not in durations:
        return
    audio_dur = float(durations[key]) + tail
    elapsed = float(getattr(scene.renderer, "time", 0.0) or 0.0)
    gap = audio_dur - elapsed
    if gap > 0.05:
        scene.wait(gap)
