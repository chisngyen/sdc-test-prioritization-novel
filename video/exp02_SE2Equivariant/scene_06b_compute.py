"""
Scene 06b -- End-to-end numerical compute walkthrough.

Replays the actual tensor flow through a toy-sized SE2RoadNet (L=4, d=4,
1 head) so the viewer can watch every multiplication: linear projection,
CLS prepend, LayerNorm, Q/K/V, scaled dot products, relative-arclength
bias, softmax, attention @ V, residual, FFN, x6 stack, CLS extract,
final head -> P(FAIL).

Same equations as the full-size model, just smaller numbers.

Render:  manim -pql scene_06b_compute.py ComputeWalkthrough
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Line, Arrow, Brace,
    MoveToTarget,
    Write, FadeIn, FadeOut, Create, ReplacementTransform,
    LaggedStart, GrowArrow,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, BLUE_A, BLUE_C, BLUE_E, YELLOW, GREEN_A, ORANGE, RED,
    GREY_A, GREY_C, PINK,
)
from manim.utils.color import interpolate_color

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import (
    TEXT, MUTED, PRIMARY, ACCENT, GOOD, WARN, BAD, RULE,
    section_header, transition, hold,
    title, subtitle, body_text, caption,
    body_formula, inline_math, chip,
    attach_narration, seal_narration,
    MATH_INLINE, MATH_BIG,
)


# ============================================================================
# Toy dimensions and stable random numbers
# ============================================================================
SEED = 7
L_TOY = 4
D_TOY = 4
F_TOY = 7


def _gen_input() -> np.ndarray:
    return np.array([
        [ 0.30, -0.10,  0.05,  0.02,  0.01,  0.10,  0.04],
        [ 0.45,  0.20,  0.30,  0.05, -0.04,  0.40,  0.10],
        [-0.20,  0.80, -0.50,  0.10,  0.15,  0.70,  0.30],
        [ 0.10,  0.30,  0.10, -0.02,  0.00,  1.00,  0.18],
    ], dtype=np.float64)


def _gen_weights(seed: int, shape: tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape) * 0.4


# ============================================================================
# Matrix-grid mobject
# ============================================================================
def matrix_grid(values: np.ndarray, *,
                cell: float = 0.45,
                show_numbers: bool = True,
                font_size: int = 16,
                color_lo=BLUE_E, color_hi=YELLOW,
                base_color=None,
                stroke=GREY_C,
                fmt: str = "{:+.2f}",
                vmin: float | None = None,
                vmax: float | None = None) -> VGroup:
    v = np.asarray(values, dtype=np.float64)
    vmin = float(v.min()) if vmin is None else vmin
    vmax = float(v.max()) if vmax is None else vmax
    span = max(vmax - vmin, 1e-9)
    rows, cols = v.shape

    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            if base_color is not None:
                cell_color = base_color
                opacity = 0.55
            else:
                alpha = float(np.clip((v[i, j] - vmin) / span, 0.0, 1.0))
                cell_color = interpolate_color(color_lo, color_hi, alpha)
                opacity = 0.92
            sq = Square(side_length=cell,
                        stroke_color=stroke, stroke_width=1.0,
                        fill_color=cell_color, fill_opacity=opacity)
            sq.move_to([(j - (cols - 1) / 2) * cell,
                        ((rows - 1) / 2 - i) * cell, 0.0])
            cell_grp = VGroup(sq)
            if show_numbers:
                txt = Text(fmt.format(v[i, j]), font_size=font_size, color=WHITE)
                txt.move_to(sq.get_center())
                cell_grp.add(txt)
            grid.add(cell_grp)
    return grid


def mark_cls_row(grid: VGroup, *, n_cols: int):
    """Paint the first row of a matrix_grid yellow (CLS slot)."""
    for cell_grp in grid[:n_cols]:
        cell_grp[0].set_fill(ACCENT, opacity=0.85)
        if len(cell_grp) > 1:
            cell_grp[1].set_color("#1a1a1a")


def labeled_grid(values, label_tex: str, shape_tex: str, *,
                 lbl_color, cell=0.45, font_size=14, **kwargs) -> VGroup:
    grid = matrix_grid(values, cell=cell, font_size=font_size, **kwargs)
    lbl = MathTex(label_tex, font_size=26, color=lbl_color).next_to(grid, UP, buff=0.18)
    shp = MathTex(shape_tex, font_size=22, color=MUTED).next_to(grid, DOWN, buff=0.18)
    return VGroup(grid, lbl, shp)


# ============================================================================
# Outline helpers
# ============================================================================
def row_outline(grid: VGroup, row_idx: int, n_rows: int, n_cols: int,
                *, color=ACCENT) -> Rectangle:
    cells = [grid[row_idx * n_cols + j][0] for j in range(n_cols)]
    x0 = cells[0].get_left()[0]
    x1 = cells[-1].get_right()[0]
    y  = cells[0].get_center()[1]
    h  = cells[0].height + 0.06
    return Rectangle(width=x1 - x0 + 0.06, height=h,
                     stroke_color=color, stroke_width=3,
                     fill_opacity=0).move_to([(x0 + x1) / 2, y, 0])


def col_outline(grid: VGroup, col_idx: int, n_rows: int, n_cols: int,
                *, color=ACCENT) -> Rectangle:
    cells = [grid[i * n_cols + col_idx][0] for i in range(n_rows)]
    y0 = cells[-1].get_bottom()[1]
    y1 = cells[0].get_top()[1]
    x  = cells[0].get_center()[0]
    w  = cells[0].width + 0.06
    return Rectangle(width=w, height=y1 - y0 + 0.06,
                     stroke_color=color, stroke_width=3,
                     fill_opacity=0).move_to([x, (y0 + y1) / 2, 0])


def cell_outline(grid: VGroup, i: int, j: int, n_rows: int, n_cols: int,
                 *, color=ACCENT) -> Rectangle:
    cell = grid[i * n_cols + j][0]
    return Rectangle(width=cell.width + 0.10, height=cell.height + 0.10,
                     stroke_color=color, stroke_width=3,
                     fill_opacity=0).move_to(cell.get_center())


# ============================================================================
# Scene
# ============================================================================
class ComputeWalkthrough(Scene):
    def construct(self):
        attach_narration(self, "scene_06b")
        self.x_in   = _gen_input()
        self.W_proj = _gen_weights(11, (F_TOY, D_TOY))
        self.W_Q    = _gen_weights(21, (D_TOY, D_TOY))
        self.W_K    = _gen_weights(22, (D_TOY, D_TOY))
        self.W_V    = _gen_weights(23, (D_TOY, D_TOY))
        self.W_h1   = _gen_weights(41, (D_TOY, 3))
        self.W_h2   = _gen_weights(42, (3, 1))

        self._overview()
        self._linear_projection()
        self._cls_prepend()
        self._one_attention_block()
        self._six_blocks()
        self._head_and_score()
        seal_narration(self, "scene_06b")

    # ------------------------------------------------------ a. overview ---
    def _overview(self):
        header = section_header(
            self, "Compute walkthrough -- one tensor, end-to-end",
            r"toy dimensions for clarity: $L=4$ instead of $197$, $d=4$ instead of $192$.",
            tex_subtitle=True,
        )

        steps = [
            ("step 1", r"input $(L,7) \to$ Linear $\to (L,d)$",            BLUE_A),
            ("step 2", r"prepend CLS $\to (L{+}1,d)$",                     ACCENT),
            ("step 3", r"$6 \times$ InvariantBlock  (LN, MHA, FFN)",        BLUE_C),
            ("step 4", r"take CLS row $\to$ head $\to P(\mathrm{FAIL})$",   BAD),
        ]
        rows = VGroup()
        for tg, txt, col in steps:
            tag = Text(tg, font_size=22, color=col, weight="BOLD")
            md  = Tex(txt, color=TEXT).scale_to_fit_height(0.32)
            row = VGroup(tag, md).arrange(RIGHT, buff=0.50, aligned_edge=DOWN)
            rows.add(row)
        rows.arrange(DOWN, buff=0.32, aligned_edge=LEFT).move_to([0, -0.40, 0])

        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.10) for r in rows],
                              lag_ratio=0.18, run_time=2.0))
        hold(self, 1.6)
        transition(self)

    # ----------------------------------------------- b. linear projection -
    def _linear_projection(self):
        header = section_header(
            self, "Step 1.  Linear projection",
            r"each row of $X$ is multiplied by $W_{\mathrm{proj}}\in\mathbb{R}^{7\times d}$.",
            tex_subtitle=True,
        )

        x_grp = labeled_grid(self.x_in, "X", r"(L,\,7)",
                             lbl_color=WARN, cell=0.55).move_to([-4.5, -0.35, 0])
        at = MathTex(r"\times", font_size=46, color=TEXT).move_to([-1.45, -0.35, 0])
        w_grp = labeled_grid(self.W_proj, r"W_{\mathrm{proj}}", r"(7,\,d)",
                             lbl_color=PRIMARY, cell=0.55).move_to([0.5, -0.35, 0])
        eq = MathTex(r"=", font_size=46, color=TEXT).move_to([2.60, -0.35, 0])

        H0 = self.x_in @ self.W_proj
        self.h0 = H0
        h_grp = labeled_grid(H0, r"H_0", r"(L,\,d)",
                             lbl_color=GOOD, cell=0.55,
                             color_lo=BLUE_E, color_hi=GOOD).move_to([4.5, -0.35, 0])

        self.play(FadeIn(x_grp, shift=UP * 0.10), run_time=0.8)
        self.play(Write(at), FadeIn(w_grp, shift=UP * 0.10), run_time=0.8)
        self.play(Write(eq), FadeIn(h_grp, shift=UP * 0.10), run_time=1.0)

        r1 = row_outline(x_grp[0], 1, L_TOY, F_TOY, color=WARN)
        c1 = col_outline(w_grp[0], 1, F_TOY, D_TOY, color=PRIMARY)
        o1 = cell_outline(h_grp[0], 1, 1, L_TOY, D_TOY, color=GOOD)
        formula = MathTex(
            r"H_0[i,j] = \sum_{k=1}^{7} X[i,k]\, W_{\mathrm{proj}}[k,j]",
            font_size=24, color=GOOD,
        ).move_to([0, -2.65, 0])
        self.play(Create(r1), Create(c1), Create(o1), run_time=0.7)
        self.play(Write(formula), run_time=0.9)
        hold(self, 2.0)
        transition(self)

    # ---------------------------------------------------- c. CLS prepend --
    def _cls_prepend(self):
        header = section_header(
            self, "Step 2.  Prepend a learnable CLS token",
            "The network writes its final answer into row 0.",
        )

        h_grp = labeled_grid(self.h0, r"H_0", r"(L,\,d)",
                             lbl_color=GOOD, cell=0.55,
                             color_lo=BLUE_E, color_hi=GOOD).move_to([-3.5, -0.30, 0])
        self.play(FadeIn(h_grp, shift=UP * 0.10), run_time=0.7)

        cls_vec = np.zeros((1, D_TOY))
        cls_data = np.concatenate([cls_vec, self.h0], axis=0)
        self.h1 = cls_data

        h1_grp = labeled_grid(cls_data, r"H \;=\; [\,\mathrm{CLS};\, H_0]",
                              r"(L{+}1,\,d)",
                              lbl_color=TEXT, cell=0.55,
                              color_lo=BLUE_E, color_hi=GOOD).move_to([3.3, -0.30, 0])
        mark_cls_row(h1_grp[0], n_cols=D_TOY)

        cls_brace = Brace(h1_grp[0][:D_TOY], LEFT, color=ACCENT)
        cls_lbl = Text("CLS = learnable",
                       font_size=18, color=ACCENT).next_to(cls_brace, LEFT, buff=0.12)

        arrow = Arrow(h_grp.get_right(), h1_grp.get_left(),
                      buff=0.18, stroke_width=4, color=ACCENT,
                      max_tip_length_to_length_ratio=0.18)
        self.play(GrowArrow(arrow), run_time=0.6)
        self.play(FadeIn(h1_grp, shift=RIGHT * 0.10), run_time=0.9)
        self.play(Create(cls_brace), Write(cls_lbl), run_time=0.7)
        hold(self, 2.0)

        transition(self)

    # ----------------------------------------- d. one attention block ----
    def _one_attention_block(self):
        header = section_header(
            self, "Step 3.  Inside one InvariantBlock",
            "LN  ->  Q/K/V  ->  QK + B_rel  ->  softmax  ->  AV  ->  +residual",
        )

        # --- show H pre-block on the left ---
        H_pre = self.h1
        H_grid_grp = labeled_grid(H_pre, r"H", r"(L{+}1,\,d)",
                                  lbl_color=TEXT, cell=0.40, font_size=12,
                                  color_lo=BLUE_E, color_hi=GOOD).move_to([-5.4, -0.30, 0])
        mark_cls_row(H_grid_grp[0], n_cols=D_TOY)
        self.play(FadeIn(H_grid_grp, shift=RIGHT * 0.10), run_time=0.7)

        # --- LayerNorm ---
        mu    = H_pre.mean(axis=1, keepdims=True)
        sigma = H_pre.std(axis=1, keepdims=True) + 1e-5
        H_ln  = (H_pre - mu) / sigma

        ln_eq = MathTex(
            r"\widehat{H}_{i,j} = \dfrac{H_{i,j} - \mu_i}{\sigma_i}",
            font_size=26, color=GOOD,
        ).move_to([-2.4, 1.05, 0])
        H_ln_grp = labeled_grid(H_ln, r"\widehat{H}", r"(L{+}1,\,d)",
                                lbl_color=GOOD, cell=0.40, font_size=12,
                                color_lo=BLUE_E, color_hi=GOOD).move_to([-2.4, -0.50, 0])
        mark_cls_row(H_ln_grp[0], n_cols=D_TOY)
        self.play(Write(ln_eq), run_time=0.7)
        self.play(FadeIn(H_ln_grp, shift=RIGHT * 0.10), run_time=0.7)
        hold(self, 0.5)

        # --- Q, K, V projections ---
        Q = H_ln @ self.W_Q
        K = H_ln @ self.W_K
        V = H_ln @ self.W_V

        qkv_eq = MathTex(
            r"Q = \widehat{H}W_Q,\; K = \widehat{H}W_K,\; V = \widehat{H}W_V",
            font_size=22, color=TEXT,
        ).move_to([3.4, 2.0, 0])
        self.play(Write(qkv_eq), run_time=1.0)

        qkv_data = [("Q", PRIMARY, Q), ("K", PINK, K), ("V", GOOD, V)]
        qkv_grids = []
        for (letter, col, M), ypos in zip(qkv_data, [1.55, 0.55, -0.45]):
            g = matrix_grid(M, cell=0.28, font_size=10, color_lo=BLUE_E,
                            color_hi=col, fmt="{:+.1f}")
            mark_cls_row(g, n_cols=D_TOY)
            lbl = MathTex(letter, font_size=22, color=col).next_to(g, UP, buff=0.10)
            grp = VGroup(g, lbl).move_to([3.6, ypos, 0])
            qkv_grids.append(grp)
        for g in qkv_grids:
            self.play(FadeIn(g, shift=LEFT * 0.10), run_time=0.4)
        hold(self, 0.3)

        # --- Compute QK^T / sqrt(d), then add B_rel ---
        self.play(FadeOut(VGroup(H_grid_grp, ln_eq, H_ln_grp, qkv_eq)),
                  run_time=0.5)
        self.play(
            qkv_grids[0].animate.move_to([-4.6, 1.45, 0]).scale(0.85),
            qkv_grids[1].animate.move_to([-1.7, 1.45, 0]).scale(0.85),
            qkv_grids[2].animate.move_to([+1.2, 1.45, 0]).scale(0.85),
            run_time=0.9,
        )

        QKt = (Q @ K.T) / np.sqrt(D_TOY)
        s_pos = np.array([0.0, 0.0, 0.33, 0.66, 1.0])
        ds = s_pos[:, None] - s_pos[None, :]
        B_rel = 0.5 * np.tanh(2.0 * ds) + 0.1 * np.cos(4 * ds)
        S_plus = QKt + B_rel

        S_grid = matrix_grid(QKt, cell=0.36, font_size=10,
                             color_lo=BLUE_E, color_hi=ACCENT)
        S_lbl  = MathTex(r"S = \tfrac{QK^{\!\top}}{\sqrt{d}}",
                         font_size=22, color=ACCENT).next_to(S_grid, UP, buff=0.12)
        S_grp = VGroup(S_grid, S_lbl).move_to([-4.5, -1.4, 0])

        plus = MathTex("+", font_size=44, color=TEXT).move_to([-2.0, -1.4, 0])
        B_grid = matrix_grid(B_rel, cell=0.36, font_size=10,
                             color_lo=BLUE_E, color_hi=ACCENT, fmt="{:+.2f}")
        B_lbl = MathTex(r"B^{\mathrm{rel}}", font_size=22,
                        color=ACCENT).next_to(B_grid, UP, buff=0.12)
        B_grp = VGroup(B_grid, B_lbl).move_to([-0.3, -1.4, 0])
        B_eq  = MathTex(r"B^{\mathrm{rel}}_{ij} = \mathrm{MLP}(\sin(\Delta s_{ij}\omega))",
                        font_size=22, color=ACCENT).move_to([-0.3, 0.40, 0])

        eq2 = MathTex("=", font_size=44, color=TEXT).move_to([+1.95, -1.4, 0])
        SP_grid = matrix_grid(S_plus, cell=0.36, font_size=10,
                              color_lo=BLUE_E, color_hi=ACCENT)
        SP_lbl  = MathTex(r"S + B^{\mathrm{rel}}", font_size=22,
                          color=ACCENT).next_to(SP_grid, UP, buff=0.12)
        SP_grp = VGroup(SP_grid, SP_lbl).move_to([+4.2, -1.4, 0])

        self.play(Write(B_eq), run_time=0.6)
        self.play(FadeIn(S_grp, shift=UP * 0.10), run_time=0.5)
        self.play(Write(plus), FadeIn(B_grp, shift=UP * 0.10), run_time=0.7)
        self.play(Write(eq2), FadeIn(SP_grp, shift=UP * 0.10), run_time=0.7)
        hold(self, 0.8)

        # --- Softmax row-wise ---
        ex = np.exp(S_plus - S_plus.max(axis=1, keepdims=True))
        A = ex / ex.sum(axis=1, keepdims=True)

        self.play(FadeOut(VGroup(qkv_grids[0], qkv_grids[1],
                                 S_grp, plus, B_grp, B_eq, eq2)),
                  run_time=0.5)

        sm_eq = MathTex(
            r"A_{ij} = \mathrm{softmax}_j\,(S+B^{\mathrm{rel}})_{ij}",
            font_size=26, color=WARN,
        ).move_to([-3.4, 0.80, 0])
        A_grid = matrix_grid(A, cell=0.38, font_size=10,
                             color_lo=BLUE_E, color_hi=WARN, fmt="{:.2f}")
        A_lbl  = MathTex(r"A", font_size=22, color=WARN).next_to(A_grid, UP, buff=0.12)
        A_grp = VGroup(A_grid, A_lbl).move_to([-3.4, -0.85, 0])

        self.play(ReplacementTransform(SP_grp, A_grp), Write(sm_eq), run_time=1.0)
        row_sum_note = caption("each row sums to 1", color=WARN, italic=False).move_to([-3.4, -2.20, 0])
        self.play(Write(row_sum_note), run_time=0.5)

        # --- O = A V ---
        out = A @ V
        self.out_block = out
        mat_eq = MathTex(r"O = A\,V", font_size=28, color=GOOD).move_to([1.3, 0.80, 0])
        qkv_grids[2].generate_target()
        qkv_grids[2].target.move_to([1.3, -0.85, 0])
        self.play(MoveToTarget(qkv_grids[2]), run_time=0.7)

        O_grid = matrix_grid(out, cell=0.38, font_size=10,
                             color_lo=BLUE_E, color_hi=GOOD, fmt="{:+.2f}")
        mark_cls_row(O_grid, n_cols=D_TOY)
        O_lbl = MathTex(r"O", font_size=22, color=GOOD).next_to(O_grid, UP, buff=0.12)
        O_grp = VGroup(O_grid, O_lbl).move_to([5.0, -0.85, 0])
        self.play(Write(mat_eq), FadeIn(O_grp, shift=LEFT * 0.10), run_time=0.8)
        hold(self, 0.8)

        # --- residual + FFN follow-up (collapsed) ---
        self.play(FadeOut(VGroup(A_grp, sm_eq, row_sum_note,
                                 qkv_grids[2], O_grp, mat_eq)),
                  run_time=0.5)

        H_pre_grp = labeled_grid(self.h1, r"H", r"",
                                 lbl_color=TEXT, cell=0.36, font_size=11,
                                 color_lo=BLUE_E, color_hi=GOOD).move_to([-4.4, -0.30, 0])
        mark_cls_row(H_pre_grp[0], n_cols=D_TOY)

        plus_res = MathTex("+", font_size=40, color=ACCENT).move_to([-2.4, -0.30, 0])
        O_again = matrix_grid(out, cell=0.36, font_size=11,
                              color_lo=BLUE_E, color_hi=GOOD, fmt="{:+.2f}")
        mark_cls_row(O_again, n_cols=D_TOY)
        O_again_lbl = MathTex(r"O", font_size=22, color=GOOD).next_to(O_again, UP, buff=0.12)
        O_again_grp = VGroup(O_again, O_again_lbl).move_to([-0.9, -0.30, 0])
        eq_sym = MathTex("=", font_size=40, color=ACCENT).move_to([0.8, -0.30, 0])

        H_post = self.h1 + out
        Hp_grid = matrix_grid(H_post, cell=0.36, font_size=11,
                              color_lo=BLUE_E, color_hi=GOOD, fmt="{:+.2f}")
        mark_cls_row(Hp_grid, n_cols=D_TOY)
        Hp_lbl = MathTex(r"H'", font_size=22, color=GOOD).next_to(Hp_grid, UP, buff=0.12)
        Hp_grp = VGroup(Hp_grid, Hp_lbl).move_to([2.7, -0.30, 0])

        res_eq = Tex(r"residual:\quad $H' = H + \mathrm{Attn}(\widehat{H})$",
                     color=ACCENT).scale_to_fit_height(0.30).move_to([0, 1.50, 0])
        self.play(Write(res_eq), run_time=0.7)
        self.play(FadeIn(H_pre_grp, shift=UP * 0.10), run_time=0.5)
        self.play(Write(plus_res), FadeIn(O_again_grp, shift=UP * 0.10), run_time=0.5)
        self.play(Write(eq_sym), FadeIn(Hp_grp, shift=UP * 0.10), run_time=0.5)
        hold(self, 0.7)

        ffn_note = Tex(
            r"then: $\mathrm{LN}(\cdot) \to \mathrm{FFN}_{d \to 4d \to d} \to +\text{residual}$",
            color=PRIMARY,
        ).scale_to_fit_height(0.30).move_to([0, -2.40, 0])
        self.play(Write(ffn_note), run_time=0.8)
        hold(self, 1.2)

        self.H_after_block = H_post
        transition(self)

    # ----------------------------------------------- e. x6 blocks ---------
    def _six_blocks(self):
        header = section_header(
            self, "Step 3 (continued).  Six identical blocks in sequence",
            "Shape stays  (L+1, d)  the whole way; CLS row accumulates.",
        )

        tensor_w = 1.10
        blocks = VGroup()
        for k in range(7):
            grid = Rectangle(
                width=tensor_w, height=2.2,
                stroke_color=PRIMARY, stroke_width=2,
                fill_color=PRIMARY, fill_opacity=0.10,
            )
            cls_band = Rectangle(
                width=tensor_w * 0.96, height=0.30,
                stroke_width=0, fill_color=ACCENT, fill_opacity=0.85,
            ).move_to(grid.get_top() + DOWN * 0.18)
            blocks.add(VGroup(grid, cls_band))
        blocks.arrange(RIGHT, buff=0.50).move_to([0, -0.10, 0])

        block_chips = VGroup()
        for k in range(6):
            mid = (blocks[k].get_right() + blocks[k + 1].get_left()) / 2
            box = RoundedRectangle(
                width=0.42, height=0.32, corner_radius=0.06,
                stroke_color=BLUE_C, stroke_width=1.5,
                fill_color=BLUE_C, fill_opacity=0.45,
            ).move_to(mid)
            block_chips.add(box)

        cap_top = caption("yellow band = CLS row -- the global summary",
                          color=ACCENT, italic=False).next_to(blocks, UP, buff=0.45)
        cap_bot = MathTex(r"6 \times \text{InvariantBlock}",
                          font_size=26, color=BLUE_C).next_to(blocks, DOWN, buff=0.45)

        self.play(FadeIn(blocks[0]), run_time=0.4)
        for k in range(6):
            self.play(
                FadeIn(block_chips[k], scale=1.2),
                FadeIn(blocks[k + 1], shift=RIGHT * 0.15),
                run_time=0.30,
            )
        self.play(Write(cap_top), Write(cap_bot), run_time=1.0)
        hold(self, 1.6)

        transition(self)

    # ------------------------------------------ f. head and score ---------
    def _head_and_score(self):
        header = section_header(
            self, "Step 4.  Take CLS, push through head, read the score",
            "One row becomes one number.  That number is the prediction.",
        )

        # Big tensor representation
        big_tensor = Rectangle(
            width=1.55, height=2.0,
            stroke_color=PRIMARY, stroke_width=2,
            fill_color=PRIMARY, fill_opacity=0.14,
        ).move_to([-5.1, -0.10, 0])
        cls_band = Rectangle(
            width=1.50, height=0.32, stroke_width=0,
            fill_color=ACCENT, fill_opacity=0.9,
        ).move_to(big_tensor.get_top() + DOWN * 0.18)
        cls_text = Text("CLS row", font_size=14, color="#1a1a1a").move_to(cls_band)
        tensor_lbl = MathTex(r"H^{(6)}", font_size=22,
                             color=PRIMARY).next_to(big_tensor, DOWN, buff=0.18)

        self.play(FadeIn(big_tensor), FadeIn(cls_band), FadeIn(cls_text),
                  Write(tensor_lbl), run_time=0.8)
        hold(self, 0.4)

        # Detach CLS row
        cls_band.generate_target()
        cls_band.target.move_to([-2.2, 1.10, 0]).scale(1.3)
        cls_text.generate_target()
        cls_text.target.move_to(cls_band.target.get_center())

        cls_vec_lbl = MathTex(r"\mathbf{z}_{\mathrm{CLS}} \in \mathbb{R}^{d}",
                              font_size=24, color=ACCENT).next_to(cls_band.target, UP, buff=0.18)
        self.play(MoveToTarget(cls_band), MoveToTarget(cls_text),
                  FadeIn(cls_vec_lbl, shift=DOWN * 0.10), run_time=0.9)

        # Toy numeric vector visual
        z = np.random.default_rng(99).standard_normal(D_TOY) * 0.6
        z_grid = matrix_grid(z.reshape(1, -1), cell=0.40, font_size=12,
                             color_lo=BLUE_E, color_hi=ACCENT,
                             fmt="{:+.2f}").move_to(cls_band.target.get_center())
        self.play(FadeOut(cls_band), FadeOut(cls_text), run_time=0.3)
        self.play(FadeIn(z_grid), run_time=0.4)

        # Head MLP boxes
        head1 = RoundedRectangle(width=1.7, height=0.70, corner_radius=0.10,
                                 stroke_color=GOOD, stroke_width=2.5,
                                 fill_color=GOOD, fill_opacity=0.15)
        head1_lbl = MathTex(r"\mathrm{LN}\ +\ d \!\to\! 64",
                            font_size=18, color=GOOD).move_to(head1.get_center())
        head1_grp = VGroup(head1, head1_lbl).move_to([0.20, 1.10, 0])

        head2 = RoundedRectangle(width=1.7, height=0.70, corner_radius=0.10,
                                 stroke_color=GOOD, stroke_width=2.5,
                                 fill_color=GOOD, fill_opacity=0.15)
        head2_lbl = MathTex(r"\mathrm{GELU}\ +\ 64 \!\to\! 1",
                            font_size=18, color=GOOD).move_to(head2.get_center())
        head2_grp = VGroup(head2, head2_lbl).move_to([2.80, 1.10, 0])

        a1 = Arrow(z_grid.get_right(), head1_grp.get_left(), buff=0.08,
                   stroke_width=3, color=ACCENT,
                   max_tip_length_to_length_ratio=0.20)
        a2 = Arrow(head1_grp.get_right(), head2_grp.get_left(), buff=0.08,
                   stroke_width=3, color=ACCENT,
                   max_tip_length_to_length_ratio=0.20)

        # Logit + sigmoid + final probability
        h1 = np.maximum(0, z @ self.W_h1)
        h2 = (h1 @ self.W_h2).squeeze()
        prob = 1.0 / (1.0 + np.exp(-h2))

        logit_tag = MathTex(rf"\mathrm{{logit}} = {h2:+.3f}",
                            font_size=24, color=BAD).move_to([5.0, 1.10, 0])
        a3 = Arrow(head2_grp.get_right(), logit_tag.get_left(),
                   buff=0.10, stroke_width=3, color=BAD,
                   max_tip_length_to_length_ratio=0.22)

        self.play(GrowArrow(a1), FadeIn(head1_grp, shift=RIGHT * 0.10), run_time=0.5)
        self.play(GrowArrow(a2), FadeIn(head2_grp, shift=RIGHT * 0.10), run_time=0.5)
        self.play(GrowArrow(a3), Write(logit_tag), run_time=0.6)
        hold(self, 0.4)

        sig_box = RoundedRectangle(width=1.6, height=0.70, corner_radius=0.10,
                                   stroke_color=BAD, stroke_width=2.5,
                                   fill_color=BAD, fill_opacity=0.15).move_to([0.8, -1.00, 0])
        sig_lbl = MathTex(r"\sigma(\cdot)", font_size=26, color=BAD).move_to(sig_box.get_center())
        sig_grp = VGroup(sig_box, sig_lbl)

        prob_tag = MathTex(rf"P(\text{{FAIL}}) = {prob:.3f}",
                           font_size=34, color=BAD).move_to([3.8, -1.00, 0])

        a4 = Arrow(logit_tag.get_bottom(), sig_grp.get_top(),
                   buff=0.10, stroke_width=3, color=BAD,
                   max_tip_length_to_length_ratio=0.22)
        a5 = Arrow(sig_grp.get_right(), prob_tag.get_left(),
                   buff=0.10, stroke_width=3, color=BAD,
                   max_tip_length_to_length_ratio=0.22)

        self.play(GrowArrow(a4), FadeIn(sig_grp, shift=DOWN * 0.10), run_time=0.6)
        self.play(GrowArrow(a5), Write(prob_tag), run_time=0.7)

        # Subtle scale-up emphasis (no Flash)
        self.play(prob_tag.animate.scale(1.10), run_time=0.4)
        self.play(prob_tag.animate.scale(1 / 1.10), run_time=0.3)
        hold(self, 1.6)

        cap = caption("That single scalar is what we sort by for the APFD ranking.",
                      color=MUTED).move_to([0, -3.30, 0])
        self.play(Write(cap), run_time=0.8)
        hold(self, 1.8)

        transition(self)
