"""
Shared utilities for the SE(2)-Equivariant RoadNet video.

Mirrors the feature pipeline from exps/exp02_SE2Equivariant.py so the
on-screen numbers match what the model actually sees at training time.
"""

from __future__ import annotations
import numpy as np
from manim import (
    VGroup, VMobject, Line, Dot, Arrow, Arc,
    BLUE, BLUE_A, BLUE_B, BLUE_D, BLUE_E,
    TEAL, GREEN, GREEN_A, YELLOW, YELLOW_A, ORANGE, RED, RED_A,
    PURPLE, PURPLE_A, PINK, GREY, GREY_A, WHITE, GOLD, MAROON,
    LEFT, RIGHT, UP, DOWN, ORIGIN, PI,
)


# -----------------------------------------------------------------------------
# Colour palette.  Each of the 7 invariant channels gets a stable hue so the
# viewer can track "this colour = this quantity" across scenes.
# -----------------------------------------------------------------------------
ROAD_COLOR      = BLUE_A
ROAD_FILL       = "#101820"
POINT_COLOR     = YELLOW
ACCENT          = GOLD

FEATURE_COLORS = {
    "seg":      BLUE_B,      # segment length
    "dangle":   ORANGE,      # |delta heading|
    "kappa":    YELLOW,      # signed curvature
    "dkappa":   GREEN,       # d kappa / ds
    "ddkappa":  TEAL,        # d^2 kappa / ds^2
    "s_norm":   PURPLE_A,    # cumulative arclength / L
    "lstd":     PINK,        # local std of kappa
}
FEATURE_NAMES = {
    "seg":     r"\Delta s_i",
    "dangle":  r"|\Delta \theta_i|",
    "kappa":   r"\kappa_i",
    "dkappa":  r"\kappa'_i",
    "ddkappa": r"\kappa''_i",
    "s_norm":  r"s_i / L",
    "lstd":    r"\sigma_\kappa(i)",
}
FEATURE_DESC = {
    "seg":     "segment length",
    "dangle":  "|heading change|",
    "kappa":   "signed curvature",
    "dkappa":  "curvature rate",
    "ddkappa": "curvature acceleration",
    "s_norm":  "arclength fraction",
    "lstd":    "local curvature noise",
}
FEATURE_KEYS = ["seg", "dangle", "kappa", "dkappa", "ddkappa", "s_norm", "lstd"]


# -----------------------------------------------------------------------------
# Demo road.  Hand-crafted to show a straight start, a long left bend, a
# sharper right bend, and a wiggle at the end -- so every feature has
# non-trivial values to draw.
# -----------------------------------------------------------------------------
def sample_road(n: int = 32, scale: float = 1.0, seed: int = 7) -> np.ndarray:
    """Return road points shaped (n, 2) in scene units (not normalised)."""
    t = np.linspace(0, 1, n)
    x = 6.0 * (t - 0.5)
    # piecewise-ish y
    y = (
        0.20 * np.sin(2.0 * np.pi * t * 1.4)
        + 0.55 * np.sin(np.pi * t)
        - 0.35 * (t - 0.5) ** 2
    )
    pts = np.column_stack([x, y]) * scale
    return pts.astype(np.float64)


def rotate_points(pts: np.ndarray, deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T


# -----------------------------------------------------------------------------
# Feature pipeline -- bit-for-bit clone of extract_invariant_7ch().
# Returns (n, 7) feature matrix with channels:
#   0 seg, 1 |dang|, 2 kappa, 3 dkappa, 4 ddkappa, 5 s/L, 6 lstd
# -----------------------------------------------------------------------------
def _signed_curvature(pts: np.ndarray) -> np.ndarray:
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    seg = np.linalg.norm(d, axis=1)
    denom = 0.5 * (seg[:-1] + seg[1:]) + 1e-8
    k = dang / denom
    return np.pad(k, (1, 1), mode="constant")


def extract_invariant_7ch(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    d = np.diff(pts, axis=0)
    seg = np.linalg.norm(d, axis=1)
    seg_full = np.pad(seg, (0, 1), mode="edge")
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    abs_dang_full = np.pad(np.abs(dang), (1, 1), mode="constant")
    k = _signed_curvature(pts)
    dk = np.pad(np.diff(k), (0, 1), mode="constant")
    ddk = np.pad(np.diff(dk), (0, 1), mode="constant")
    s_cum = np.cumsum(seg_full)
    s_norm = s_cum / (s_cum[-1] + 1e-8)
    w = 11
    lstd = np.zeros(n)
    hw = w // 2
    for i in range(n):
        a, b = max(0, i - hw), min(n, i + hw + 1)
        lstd[i] = np.std(k[a:b])
    feats = np.column_stack([seg_full, abs_dang_full, k, dk, ddk, s_norm, lstd])
    return feats.astype(np.float64)


# -----------------------------------------------------------------------------
# Visual helpers -- turn an (n, 2) numpy array into a smooth Manim road.
# -----------------------------------------------------------------------------
def to_scene_coords(pts: np.ndarray) -> np.ndarray:
    """Embed (n, 2) into (n, 3) with z=0 so Manim can use them."""
    z = np.zeros((len(pts), 1))
    return np.concatenate([pts, z], axis=1)


def make_road(
    pts: np.ndarray,
    *,
    stroke_color=ROAD_COLOR,
    stroke_width: float = 6.0,
    show_dots: bool = True,
    dot_radius: float = 0.045,
    dot_color=POINT_COLOR,
):
    """A smooth road VMobject with optional dots at sample points."""
    coords = to_scene_coords(pts)
    road = VMobject(stroke_color=stroke_color, stroke_width=stroke_width)
    road.set_points_smoothly(coords)
    if not show_dots:
        return road
    dots = VGroup(*[Dot(p, radius=dot_radius, color=dot_color) for p in coords])
    return VGroup(road, dots)


def make_polyline(pts: np.ndarray, color=ROAD_COLOR, stroke_width: float = 4.0) -> VMobject:
    """A piecewise-linear polyline.  Useful when we want to highlight a
    specific segment (segment-length feature, heading vectors, etc.)."""
    coords = to_scene_coords(pts)
    line = VMobject(stroke_color=color, stroke_width=stroke_width)
    line.set_points_as_corners(coords)
    return line


def segment_arrows(pts: np.ndarray, color=BLUE_A, stroke_width: float = 3.0) -> VGroup:
    """One Arrow per consecutive pair (good for visualising heading)."""
    coords = to_scene_coords(pts)
    arrows = VGroup()
    for a, b in zip(coords[:-1], coords[1:]):
        arrows.add(
            Arrow(a, b, buff=0, stroke_width=stroke_width,
                  color=color, max_tip_length_to_length_ratio=0.25)
        )
    return arrows


# -----------------------------------------------------------------------------
# Small numeric helpers
# -----------------------------------------------------------------------------
def fmt(v: float, prec: int = 3) -> str:
    """Compact fixed-point format with trailing-zero trim."""
    s = f"{v:.{prec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def normalise_to_unit_band(values: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Linearly remap any 1-D array into [lo, hi].  Used purely for visual
    encoding (eg. heat bars under the road)."""
    v = np.asarray(values, dtype=np.float64)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-12:
        return np.full_like(v, 0.5 * (lo + hi))
    return lo + (v - vmin) * (hi - lo) / (vmax - vmin)
