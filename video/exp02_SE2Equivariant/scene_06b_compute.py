"""
Scene 06b -- End-to-end numerical compute walkthrough.

We replay the *actual* tensor flow through a toy-sized SE2RoadNet
(L=4 instead of 197, d_model=4 instead of 192, 1 head instead of 8) so
the viewer can watch every multiplication: linear projection, CLS
prepend, LayerNorm, Q/K/V, scaled dot products, relative-arclength bias,
softmax, attention @ V, residual, FFN, the x6 block stack, CLS extract,
and the final head producing P(FAIL).

Same equations as the full-size model -- just smaller numbers so they
fit on the screen.

Render:
    manim -pql scene_06b_compute.py ComputeWalkthrough
"""
from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Circle, Dot, Line, Arrow,
    DashedLine, Triangle, Brace,
    Write, FadeIn, FadeOut, Create, ReplacementTransform,
    Indicate, Flash, LaggedStart, Transform, GrowArrow,
    ValueTracker, always_redraw, DecimalNumber,
    UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL, ORIGIN, PI, DEGREES,
    WHITE, BLACK, BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E,
    YELLOW, YELLOW_A, ORANGE, RED, RED_A,
    GREEN, GREEN_A, GREEN_B, GREY, GREY_A, GREY_B, GREY_C, GREY_D,
    GOLD, PINK, MAROON, TEAL, PURPLE, PURPLE_A,
)
from manim.utils.color import interpolate_color

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from layout import title, subtitle, footer, clear


# ============================================================================
# Numerical helpers -- pin a random seed so the on-screen numbers are stable.
# ============================================================================
SEED = 7
rng = np.random.default_rng(SEED)

L_TOY = 4   # token count (in the real model: 197)
D_TOY = 4   # model dimension (real: 192)
F_TOY = 7   # number of input features (matches real model)

def _gen_input():
    """A plausible (L, 7) input tensor.  Numbers chosen to look like
    normalised features: roughly zero-mean, magnitudes <= 2."""
    return np.array([
        [ 0.30, -0.10,  0.05,  0.02,  0.01,  0.10,  0.04],
        [ 0.45,  0.20,  0.30,  0.05, -0.04,  0.40,  0.10],
        [-0.20,  0.80, -0.50,  0.10,  0.15,  0.70,  0.30],
        [ 0.10,  0.30,  0.10, -0.02,  0.00,  1.00,  0.18],
    ], dtype=np.float64)


def _gen_weights(seed: int, shape: tuple[int, int]) -> np.ndarray:
    r = np.random.default_rng(seed)
    W = r.standard_normal(shape) * 0.4
    return W


# ============================================================================
# Matrix-as-mobject helpers
# ============================================================================
def matrix_grid(values: np.ndarray, *,
                cell=0.45,
                show_numbers: bool = True,
                font_size: int = 16,
                color_lo=BLUE_E, color_hi=YELLOW,
                base_color=None,
                stroke=BLUE_D,
                fmt: str = "{:+.2f}",
                vmin: float | None = None,
                vmax: float | None = None) -> VGroup:
    """Render a 2-D ndarray as a coloured grid with numbers inside."""
    v = np.asarray(values, dtype=np.float64)
    if vmin is None: vmin = float(v.min())
    if vmax is None: vmax = float(v.max())
    span = max(vmax - vmin, 1e-9)
    rows, cols = v.shape

    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            if base_color is not None:
                cell_color = base_color
                opacity = 0.6
            else:
                alpha = float(np.clip((v[i, j] - vmin) / span, 0.0, 1.0))
                cell_color = interpolate_color(color_lo, color_hi, alpha)
                opacity = 0.95
            sq = Square(side_length=cell,
                        stroke_color=stroke, stroke_width=1.2,
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


def shape_tag(rows: int, cols: int, *, color=GREY_A, font_size=20) -> MathTex:
    return MathTex(rf"({rows},\,{cols})", font_size=font_size, color=color)


# ============================================================================
# The scene
# ============================================================================
class ComputeWalkthrough(Scene):
    def construct(self):
        # Cache numbers we'll need over the whole scene
        self.x_in   = _gen_input()                          # (L, 7)
        self.W_proj = _gen_weights(11, (F_TOY, D_TOY))      # (7, 4)
        self.W_Q    = _gen_weights(21, (D_TOY, D_TOY))      # (4, 4)
        self.W_K    = _gen_weights(22, (D_TOY, D_TOY))
        self.W_V    = _gen_weights(23, (D_TOY, D_TOY))
        self.W_ff1  = _gen_weights(31, (D_TOY, 8))          # FFN up
        self.W_ff2  = _gen_weights(32, (8, D_TOY))          # FFN down
        self.W_h1   = _gen_weights(41, (D_TOY, 3))          # head linear 1
        self.W_h2   = _gen_weights(42, (3, 1))              # head linear 2

        self._part_a_overview_banner()
        self._part_b_input_to_proj()
        self._part_c_cls_prepend()
        self._part_d_one_attention_block()
        self._part_e_x6_blocks()
        self._part_f_cls_extract_head()

    # ----------- helper outlines on a matrix_grid -------------------------
    def _row_outline(self, grid: VGroup, row_idx: int, n_rows: int, n_cols: int,
                     *, color=YELLOW) -> Rectangle:
        cells = [grid[row_idx * n_cols + j][0] for j in range(n_cols)]
        x0 = cells[0].get_left()[0]
        x1 = cells[-1].get_right()[0]
        y  = cells[0].get_center()[1]
        h  = cells[0].height + 0.06
        return Rectangle(width=x1 - x0 + 0.06, height=h,
                         stroke_color=color, stroke_width=3,
                         fill_opacity=0).move_to([(x0 + x1) / 2, y, 0])

    def _col_outline(self, grid: VGroup, col_idx: int, n_rows: int, n_cols: int,
                     *, color=YELLOW) -> Rectangle:
        cells = [grid[i * n_cols + col_idx][0] for i in range(n_rows)]
        y0 = cells[-1].get_bottom()[1]
        y1 = cells[0].get_top()[1]
        x  = cells[0].get_center()[0]
        w  = cells[0].width + 0.06
        return Rectangle(width=w, height=y1 - y0 + 0.06,
                         stroke_color=color, stroke_width=3,
                         fill_opacity=0).move_to([x, (y0 + y1) / 2, 0])

    def _cell_outline(self, grid: VGroup, i: int, j: int, n_rows: int, n_cols: int,
                      *, color=YELLOW) -> Rectangle:
        cell = grid[i * n_cols + j][0]
        return Rectangle(width=cell.width + 0.10, height=cell.height + 0.10,
                         stroke_color=color, stroke_width=3,
                         fill_opacity=0).move_to(cell.get_center())

    # ----------------------------------------------------------- a. banner
    def _part_a_overview_banner(self):
        t = title("Compute walkthrough: one tensor, end-to-end")
        s = subtitle(
            "toy dims for clarity: L=4 instead of 197, d=4 instead of 192, 1 head",
        )
        self.play(Write(t), FadeIn(s, shift=UP * 0.15))
        self.wait(0.8)

        steps = [
            ("step 1", r"input  $(L,7)$  $\to$  Linear  $(L,d)$",        BLUE_A),
            ("step 2", r"prepend CLS  $\to$  $(L{+}1,d)$",               YELLOW),
            ("step 3", r"6$\times$ InvariantBlock  (LN + MHA + FFN)",     BLUE_C),
            ("step 4", r"take CLS row $\to$ head $\to$ $P(\mathrm{FAIL})$", RED),
        ]
        rows = VGroup()
        for tag, txt, col in steps:
            tg = Text(tag, font_size=22, color=col, weight="BOLD")
            md = Tex(txt, font_size=24, color=WHITE)
            row = VGroup(tg, md).arrange(RIGHT, buff=0.5, aligned_edge=DOWN)
            rows.add(row)
        rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([0, -0.4, 0])

        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in rows],
                              lag_ratio=0.18, run_time=2.0))
        self.wait(1.6)
        clear(self)

    # ------------------------------------------------- b. input -> proj
    def _part_b_input_to_proj(self):
        t = title("Step 1.  Linear projection")
        s = subtitle(r"each row of $X$ is multiplied by $W_{\mathrm{proj}} \in \mathbb{R}^{7 \times d}$",
                     tex=True)
        self.play(Write(t), FadeIn(s, shift=UP * 0.15))

        # Input matrix on the left
        x_grid = matrix_grid(self.x_in, cell=0.55, font_size=14)
        x_lbl = MathTex(r"X", font_size=26, color=ORANGE).next_to(x_grid, UP, buff=0.2)
        x_shape = shape_tag(L_TOY, F_TOY, color=GREY_A).next_to(x_grid, DOWN, buff=0.2)
        x_grp = VGroup(x_grid, x_lbl, x_shape)
        x_grp.move_to([-4.5, -0.3, 0])

        # @W
        at_sym = MathTex(r"\times", font_size=44, color=WHITE)
        at_sym.move_to([-1.4, -0.3, 0])

        w_grid = matrix_grid(self.W_proj, cell=0.55, font_size=14)
        w_lbl = MathTex(r"W_{\mathrm{proj}}", font_size=26, color=BLUE_A).next_to(w_grid, UP, buff=0.2)
        w_shape = shape_tag(F_TOY, D_TOY).next_to(w_grid, DOWN, buff=0.2)
        w_grp = VGroup(w_grid, w_lbl, w_shape)
        w_grp.move_to([0.4, -0.3, 0])

        # =
        eq_sym = MathTex(r"=", font_size=44, color=WHITE).move_to([2.6, -0.3, 0])

        # Compute X @ W_proj
        h0 = self.x_in @ self.W_proj                       # (L, D)
        self.h0 = h0
        h_grid = matrix_grid(h0, cell=0.55, font_size=14, color_lo=BLUE_E, color_hi=GREEN_A)
        h_lbl = MathTex(r"H_0", font_size=26, color=GREEN_A).next_to(h_grid, UP, buff=0.2)
        h_shape = shape_tag(L_TOY, D_TOY, color=GREEN_A).next_to(h_grid, DOWN, buff=0.2)
        h_grp = VGroup(h_grid, h_lbl, h_shape)
        h_grp.move_to([4.4, -0.3, 0])

        self.play(FadeIn(x_grp, shift=UP * 0.1))
        self.play(Write(at_sym), FadeIn(w_grp, shift=UP * 0.1))
        self.play(Write(eq_sym), FadeIn(h_grp, shift=UP * 0.1), run_time=1.2)

        # Highlight: row 1 of X * column 1 of W -> entry (1,1) of H
        row1 = self._row_outline(x_grid, 1, L_TOY, F_TOY, color=ORANGE)
        col1 = self._col_outline(w_grid, 1, F_TOY, D_TOY, color=BLUE_A)
        out1 = self._cell_outline(h_grid, 1, 1, L_TOY, D_TOY, color=GREEN_A)
        formula = MathTex(
            r"H_0[i, j] = \sum_{k=1}^{7} X[i, k]\, W_{\mathrm{proj}}[k, j]",
            font_size=22, color=GREEN_A,
        ).move_to([0, -2.5, 0])
        self.play(Create(row1), Create(col1), Create(out1), run_time=0.8)
        self.play(Write(formula))
        self.wait(2.2)

        clear(self)

    # ----------------------------------------------- c. CLS prepend
    def _part_c_cls_prepend(self):
        t = title("Step 2.  Prepend a learnable CLS token")
        s = subtitle("the network will write its final answer into row 0")
        self.play(Write(t), FadeIn(s, shift=UP * 0.15))

        # Show H_0 on the left
        h_grid = matrix_grid(self.h0, cell=0.55, font_size=14,
                             color_lo=BLUE_E, color_hi=GREEN_A)
        h_lbl = MathTex(r"H_0", font_size=26, color=GREEN_A).next_to(h_grid, UP, buff=0.2)
        h_shape = shape_tag(L_TOY, D_TOY, color=GREEN_A).next_to(h_grid, DOWN, buff=0.2)
        h_grp = VGroup(h_grid, h_lbl, h_shape).move_to([-3.5, -0.2, 0])

        self.play(FadeIn(h_grp, shift=UP * 0.1))

        # CLS row (in yellow, all zeros initially)
        cls_vec = np.zeros((1, D_TOY))
        cls_grid = matrix_grid(cls_vec, cell=0.55, font_size=14,
                               base_color=YELLOW)
        cls_lbl = Text("CLS", font_size=18, color="#222")
        # overlay CLS text on the row
        cls_lbl.move_to(cls_grid.get_center())
        cls_brace = Brace(cls_grid, LEFT, color=YELLOW)
        cls_btxt = Text("learnable", font_size=18,
                        color=YELLOW).next_to(cls_brace, LEFT, buff=0.12)

        # Stacked into (5, 4)
        h1_data = np.concatenate([cls_vec, self.h0], axis=0)
        self.h1 = h1_data
        h1_grid = matrix_grid(h1_data, cell=0.55, font_size=14,
                              color_lo=BLUE_E, color_hi=GREEN_A,
                              base_color=None)
        # Recolor row 0 yellow:
        row0_cells = h1_grid[:D_TOY]                  # cells in row 0
        for cell_grp in row0_cells:
            cell_grp[0].set_fill(YELLOW, opacity=0.85)
            if len(cell_grp) > 1:
                cell_grp[1].set_color("#222")
        h1_lbl = MathTex(r"H \;=\; [\,\mathrm{CLS}; H_0]", font_size=24,
                         color=WHITE).next_to(h1_grid, UP, buff=0.2)
        h1_shape = shape_tag(L_TOY + 1, D_TOY, color=BLUE_A).next_to(h1_grid, DOWN, buff=0.2)
        h1_grp = VGroup(h1_grid, h1_lbl, h1_shape).move_to([3.2, -0.2, 0])

        arrow = Arrow(h_grp.get_right(), h1_grp.get_left(),
                      buff=0.15, stroke_width=4, color=YELLOW,
                      max_tip_length_to_length_ratio=0.18)

        self.play(Create(cls_grid.move_to(h1_grid.get_top() + DOWN * 0.275)),
                  Write(cls_lbl.move_to(cls_grid.get_center())),
                  Create(cls_brace.next_to(cls_grid, LEFT)),
                  Write(cls_btxt.next_to(cls_brace, LEFT, buff=0.12)),
                  run_time=1.0)
        self.play(GrowArrow(arrow))
        self.play(FadeIn(h1_grp, shift=RIGHT * 0.1))
        self.wait(1.5)

        clear(self)

    # ------------------------------------------- d. one attention block
    def _part_d_one_attention_block(self):
        t = title("Step 3.  Inside one InvariantBlock")
        s = subtitle("LayerNorm  to  Q/K/V  to  QK + B_rel  to  softmax  to  AV  to  +residual")
        self.play(Write(t), FadeIn(s, shift=UP * 0.15))

        # Show H pre-block on the left
        H_pre = self.h1
        H_grid = matrix_grid(H_pre, cell=0.42, font_size=12,
                             color_lo=BLUE_E, color_hi=GREEN_A)
        # mark CLS row yellow
        for c in H_grid[:D_TOY]:
            c[0].set_fill(YELLOW, opacity=0.85)
            if len(c) > 1: c[1].set_color("#222")
        H_lbl = MathTex(r"H", font_size=22, color=WHITE).next_to(H_grid, UP, buff=0.15)
        H_grp = VGroup(H_grid, H_lbl).move_to([-5.6, -0.2, 0])
        self.play(FadeIn(H_grp, shift=RIGHT * 0.1))

        # ---- (d1) LayerNorm row-wise --------------------------------------
        # LN(h)_ij = (h_ij - mu_i) / sigma_i
        mu = H_pre.mean(axis=1, keepdims=True)
        sigma = H_pre.std(axis=1, keepdims=True) + 1e-5
        H_ln = (H_pre - mu) / sigma

        ln_eq = MathTex(
            r"\widehat{H}_{i,j} = \frac{H_{i,j} - \mu_i}{\sigma_i}",
            font_size=24, color=GREEN_A,
        ).move_to([-2.6, 1.0, 0])
        H_ln_grid = matrix_grid(H_ln, cell=0.42, font_size=12,
                                color_lo=BLUE_E, color_hi=GREEN_A)
        # CLS row stays yellow
        for c in H_ln_grid[:D_TOY]:
            c[0].set_fill(YELLOW, opacity=0.85)
            if len(c) > 1: c[1].set_color("#222")
        H_ln_lbl = MathTex(r"\widehat{H}", font_size=22, color=GREEN_A).next_to(H_ln_grid, UP, buff=0.15)
        H_ln_grp = VGroup(H_ln_grid, H_ln_lbl).move_to([-2.5, -0.6, 0])

        self.play(Write(ln_eq))
        self.play(FadeIn(H_ln_grp, shift=RIGHT * 0.1))
        self.wait(0.7)

        # ---- (d2) Q, K, V projections -------------------------------------
        Q = H_ln @ self.W_Q
        K = H_ln @ self.W_K
        V = H_ln @ self.W_V

        qkv_box_specs = [
            ("Q", BLUE_A,  Q,   1.6),
            ("K", PINK,    K,   0.6),
            ("V", GREEN_A, V,  -0.4),
        ]
        qkv_grids = []
        for letter, col, M, ypos in qkv_box_specs:
            g = matrix_grid(M, cell=0.32, font_size=10, color_lo=BLUE_E,
                            color_hi=col, fmt="{:+.1f}")
            for c in g[:D_TOY]:
                c[0].set_fill(YELLOW, opacity=0.85)
                if len(c) > 1: c[1].set_color("#222")
            lbl = MathTex(letter, font_size=22, color=col).next_to(g, UP, buff=0.10)
            grp = VGroup(g, lbl).move_to([3.6, ypos, 0])
            qkv_grids.append(grp)

        qkv_eq = MathTex(
            r"Q = \widehat{H}W_Q,\quad K = \widehat{H}W_K,\quad V = \widehat{H}W_V",
            font_size=22, color=WHITE,
        ).move_to([3.0, 2.0, 0])
        self.play(Write(qkv_eq), run_time=1.4)
        for g in qkv_grids:
            self.play(FadeIn(g, shift=LEFT * 0.1), run_time=0.45)
        self.wait(0.4)

        # Clear LN intermediate, keep H on the left
        self.play(FadeOut(VGroup(ln_eq, H_ln_grp, qkv_eq)))
        # Re-arrange Q/K/V into a tight row across the top so we have room
        # below for the matrix-multiply visuals.
        self.play(
            qkv_grids[0].animate.move_to([-4.4, 1.4, 0]).scale(0.9),
            qkv_grids[1].animate.move_to([-1.4, 1.4, 0]).scale(0.9),
            qkv_grids[2].animate.move_to([+1.6, 1.4, 0]).scale(0.9),
            FadeOut(H_grp),
            run_time=1.0,
        )

        # ---- (d3) QK^T / sqrt(d) -----------------------------------------
        QKt = (Q @ K.T) / np.sqrt(D_TOY)
        self.QKt = QKt

        attn_eq = MathTex(
            r"S = \frac{Q K^{\!\top}}{\sqrt{d}}",
            font_size=28, color=YELLOW,
        ).move_to([4.5, 1.6, 0])
        S_grid = matrix_grid(QKt, cell=0.40, font_size=11,
                             color_lo=BLUE_E, color_hi=YELLOW)
        S_lbl = MathTex(r"S", font_size=22, color=YELLOW).next_to(S_grid, UP, buff=0.15)
        S_grp = VGroup(S_grid, S_lbl).move_to([4.5, 0.0, 0])

        self.play(Write(attn_eq))
        self.play(FadeIn(S_grp, shift=UP * 0.1))
        self.wait(0.6)

        # ---- (d4) Add B^rel ----------------------------------------------
        # We synthesise a small (5, 5) relative-arclength bias matrix.
        s_pos = np.array([0.0, 0.0, 0.33, 0.66, 1.0])   # CLS gets s=0
        ds = s_pos[:, None] - s_pos[None, :]
        B_rel = 0.5 * np.tanh(2.0 * ds) + 0.1 * np.cos(4 * ds)
        S_plus = QKt + B_rel
        self.S_plus = S_plus

        # Slide S left, then bring in B and =
        self.play(
            S_grp.animate.move_to([-4.5, -1.2, 0]),
            attn_eq.animate.move_to([-4.5, 0.4, 0]).scale(0.7),
        )

        plus = MathTex("+", font_size=44, color=WHITE).move_to([-2.0, -1.2, 0])
        B_grid = matrix_grid(B_rel, cell=0.40, font_size=11,
                             color_lo=BLUE_E, color_hi=YELLOW, fmt="{:+.2f}")
        B_lbl = MathTex(r"B^{\mathrm{rel}}", font_size=22, color=YELLOW).next_to(B_grid, UP, buff=0.15)
        B_eq = MathTex(r"B^{\mathrm{rel}}_{ij} = \mathrm{MLP}(\sin(\Delta s_{ij}\, \omega))",
                       font_size=20, color=YELLOW).move_to([-0.2, 0.4, 0])
        B_grp = VGroup(B_grid, B_lbl).move_to([-0.2, -1.2, 0])

        eq2 = MathTex("=", font_size=44, color=WHITE).move_to([+2.0, -1.2, 0])

        SP_grid = matrix_grid(S_plus, cell=0.40, font_size=11,
                              color_lo=BLUE_E, color_hi=YELLOW)
        SP_lbl = MathTex(r"S + B^{\mathrm{rel}}", font_size=22, color=YELLOW).next_to(SP_grid, UP, buff=0.15)
        SP_grp = VGroup(SP_grid, SP_lbl).move_to([+4.3, -1.2, 0])

        self.play(Write(B_eq))
        self.play(Write(plus), FadeIn(B_grp, shift=UP * 0.1))
        self.play(Write(eq2), FadeIn(SP_grp, shift=UP * 0.1), run_time=1.0)
        self.wait(0.6)

        # ---- (d5) Softmax row-wise ----------------------------------------
        # Softmax across the K-dimension (columns)
        ex = np.exp(S_plus - S_plus.max(axis=1, keepdims=True))
        A = ex / ex.sum(axis=1, keepdims=True)
        self.A = A

        self.play(FadeOut(VGroup(qkv_grids[0], qkv_grids[1],
                                 attn_eq, S_grp, plus, B_grp, eq2, B_eq)))

        sm_eq = MathTex(
            r"A_{ij} = \mathrm{softmax}_j(S+B^{\mathrm{rel}})_{ij}",
            font_size=26, color=ORANGE,
        ).move_to([-3.7, 0.8, 0])

        A_grid = matrix_grid(A, cell=0.42, font_size=11,
                             color_lo=BLUE_E, color_hi=ORANGE, fmt="{:.2f}")
        A_lbl = MathTex(r"A", font_size=22, color=ORANGE).next_to(A_grid, UP, buff=0.15)
        A_grp = VGroup(A_grid, A_lbl).move_to([-3.7, -1.0, 0])

        row_sum_note = Tex(r"each row sums to $1$", font_size=22,
                           color=ORANGE).move_to([-3.7, -2.3, 0])

        # Transform S+B grid to A grid
        self.play(
            ReplacementTransform(SP_grp, A_grp),
            Write(sm_eq),
        )
        self.play(Write(row_sum_note))
        self.wait(0.5)

        # ---- (d6) Output = A @ V ------------------------------------------
        out = A @ V
        self.out = out

        mat_eq = MathTex(r"O = A\,V",
                         font_size=28, color=GREEN_A).move_to([1.2, 0.8, 0])
        # Show V still
        qkv_grids[2].generate_target()
        qkv_grids[2].target.move_to([1.2, -1.0, 0])
        self.play(MoveToTarget(qkv_grids[2]))

        O_grid = matrix_grid(out, cell=0.42, font_size=11,
                             color_lo=BLUE_E, color_hi=GREEN_A, fmt="{:+.2f}")
        for c in O_grid[:D_TOY]:
            c[0].set_fill(YELLOW, opacity=0.85)
            if len(c) > 1: c[1].set_color("#222")
        O_lbl = MathTex(r"O", font_size=22, color=GREEN_A).next_to(O_grid, UP, buff=0.15)
        O_grp = VGroup(O_grid, O_lbl).move_to([5.0, -1.0, 0])

        self.play(Write(mat_eq))
        self.play(FadeIn(O_grp, shift=LEFT * 0.1))
        self.wait(0.8)

        # ---- (d7) Residual + LayerNorm + FFN (compressed) -----------------
        self.play(FadeOut(VGroup(A_grp, sm_eq, row_sum_note,
                                 qkv_grids[2], O_grp, mat_eq)))

        # Compressed visual: original H + O = H' (with residual),
        # then "FFN" block diagram
        H_pre_grid = matrix_grid(self.h1, cell=0.42, font_size=12,
                                 color_lo=BLUE_E, color_hi=GREEN_A)
        for c in H_pre_grid[:D_TOY]:
            c[0].set_fill(YELLOW, opacity=0.85)
            if len(c) > 1: c[1].set_color("#222")
        H_pre_lbl = MathTex(r"H", font_size=20, color=WHITE).next_to(H_pre_grid, UP, buff=0.12)
        H_pre_grp = VGroup(H_pre_grid, H_pre_lbl).move_to([-4.8, -0.3, 0])

        plus_res = MathTex("+", font_size=40, color=YELLOW).move_to([-3.0, -0.3, 0])

        O_again = matrix_grid(out, cell=0.42, font_size=12,
                              color_lo=BLUE_E, color_hi=GREEN_A, fmt="{:+.2f}")
        for c in O_again[:D_TOY]:
            c[0].set_fill(YELLOW, opacity=0.85)
            if len(c) > 1: c[1].set_color("#222")
        O_again_lbl = MathTex(r"O", font_size=20, color=GREEN_A).next_to(O_again, UP, buff=0.12)
        O_again_grp = VGroup(O_again, O_again_lbl).move_to([-1.4, -0.3, 0])

        eq_res = MathTex("=", font_size=40, color=YELLOW).move_to([0.4, -0.3, 0])

        H_post = self.h1 + out
        Hp_grid = matrix_grid(H_post, cell=0.42, font_size=12,
                              color_lo=BLUE_E, color_hi=GREEN_A, fmt="{:+.2f}")
        for c in Hp_grid[:D_TOY]:
            c[0].set_fill(YELLOW, opacity=0.85)
            if len(c) > 1: c[1].set_color("#222")
        Hp_lbl = MathTex(r"H'", font_size=22, color=GREEN_A).next_to(Hp_grid, UP, buff=0.12)
        Hp_grp = VGroup(Hp_grid, Hp_lbl).move_to([2.4, -0.3, 0])

        res_eq = Tex(r"residual:\ \ $H' = H + \mathrm{Attn}(\widehat{H})$",
                     font_size=22, color=YELLOW).move_to([0, 1.6, 0])

        self.play(Write(res_eq))
        self.play(FadeIn(H_pre_grp, shift=UP * 0.1))
        self.play(Write(plus_res), FadeIn(O_again_grp, shift=UP * 0.1))
        self.play(Write(eq_res), FadeIn(Hp_grp, shift=UP * 0.1))
        self.wait(0.8)

        # FFN follow-up (block-level, no numbers)
        ffn_note = Tex(
            r"Then: $\mathrm{LN}(\,\cdot\,) \to \mathrm{FFN}_{d\to 4d\to d}\to +\text{residual}$",
            font_size=22, color=BLUE_A,
        ).move_to([0, -2.3, 0])
        self.play(Write(ffn_note))
        self.wait(1.2)

        # store final H' for the next sub-scene
        self.H_after_block = H_post
        clear(self)

    # ------------------------------------------------- e. ×6 blocks
    def _part_e_x6_blocks(self):
        t = title("Step 3 (continued).  Six identical blocks, one after another")
        self.play(Write(t))

        # Six little stacked tensors connecting one to the next
        tensor_w = 1.1
        spacing = 1.6
        blocks = VGroup()
        for k in range(7):  # 7 tensors with 6 blocks between them
            grid = Rectangle(
                width=tensor_w, height=2.2,
                stroke_color=BLUE_A, stroke_width=2,
                fill_color=BLUE_A, fill_opacity=0.15,
            )
            # mark CLS row
            cls_band = Rectangle(
                width=tensor_w * 0.96, height=0.30,
                stroke_width=0, fill_color=YELLOW, fill_opacity=0.85,
            ).move_to(grid.get_top() + DOWN * 0.18)
            stack = VGroup(grid, cls_band)
            blocks.add(stack)
        blocks.arrange(RIGHT, buff=0.55).move_to([0, -0.2, 0])

        block_lbls = VGroup()
        for k in range(6):
            mid = (blocks[k].get_right() + blocks[k + 1].get_left()) / 2
            box = RoundedRectangle(
                width=0.46, height=0.30, corner_radius=0.05,
                stroke_color=BLUE_C, stroke_width=1.5,
                fill_color=BLUE_C, fill_opacity=0.4,
            ).move_to(mid)
            block_lbls.add(box)

        block_caption = Text(r"6× InvariantBlock", font_size=22,
                             color=BLUE_C).next_to(blocks, DOWN, buff=0.5)
        shape_caption = MathTex(r"\text{shape stays }(L{+}1,\,d)\text{ throughout}",
                                font_size=22, color=GREY_A).next_to(block_caption, DOWN, buff=0.15)
        cls_caption = Tex(r"yellow band = CLS row; it accumulates the global summary",
                          font_size=20, color=YELLOW).next_to(blocks, UP, buff=0.5)

        self.play(FadeIn(blocks[0]))
        for k in range(6):
            self.play(
                FadeIn(block_lbls[k], scale=1.3),
                FadeIn(blocks[k + 1], shift=RIGHT * 0.2),
                run_time=0.35,
            )
        self.play(Write(cls_caption), Write(block_caption), Write(shape_caption),
                  run_time=1.6)
        self.wait(1.6)

        clear(self)

    # ----------------------------------- f. CLS extract + head
    def _part_f_cls_extract_head(self):
        t = title("Step 4.  Take CLS, push through head, read score")
        self.play(Write(t))

        # Big stylised tensor on the left -- only its CLS row will travel
        big_tensor = Rectangle(
            width=1.6, height=2.0,
            stroke_color=BLUE_A, stroke_width=2,
            fill_color=BLUE_A, fill_opacity=0.18,
        ).move_to([-5.2, 0.0, 0])
        cls_band_big = Rectangle(
            width=1.55, height=0.32, stroke_width=0,
            fill_color=YELLOW, fill_opacity=0.9,
        ).move_to(big_tensor.get_top() + DOWN * 0.18)
        cls_text = Text("CLS row", font_size=14, color="#222").move_to(cls_band_big)
        tensor_lbl = MathTex(r"H^{(6)}", font_size=22, color=BLUE_A).next_to(big_tensor, DOWN, buff=0.18)
        self.play(FadeIn(big_tensor), FadeIn(cls_band_big), FadeIn(cls_text), Write(tensor_lbl))
        self.wait(0.3)

        # Detach CLS row, fly to the right as a 1-D vector
        cls_band_big.generate_target()
        cls_band_big.target.move_to([-2.2, 1.0, 0]).scale(1.3)
        cls_text.generate_target()
        cls_text.target.move_to(cls_band_big.target.get_center())

        cls_vec_lbl = MathTex(r"\mathbf{z}_{\mathrm{CLS}} \in \mathbb{R}^{d}",
                              font_size=24, color=YELLOW).next_to(cls_band_big.target, UP, buff=0.2)
        self.play(MoveToTarget(cls_band_big), MoveToTarget(cls_text),
                  FadeIn(cls_vec_lbl, shift=DOWN * 0.1))
        self.wait(0.3)

        # Head MLP: d -> 64 (toy 3) -> 1
        # Compute small numeric example
        z = np.random.default_rng(99).standard_normal(D_TOY) * 0.6
        h1 = np.maximum(0, z @ self.W_h1)           # GELU approx by ReLU for the demo
        h2 = (h1 @ self.W_h2).squeeze()             # scalar logit
        prob = 1.0 / (1.0 + np.exp(-h2))

        # Vector of d toy values
        z_grid = matrix_grid(z.reshape(1, -1), cell=0.42, font_size=12,
                             color_lo=BLUE_E, color_hi=YELLOW, fmt="{:+.2f}").move_to(
            cls_band_big.target.get_center()
        )
        # We overlay z_grid on top of the CLS band visual
        self.play(FadeOut(cls_band_big), FadeOut(cls_text))
        self.play(FadeIn(z_grid))

        # Arrow to MLP linear 1
        head_box1 = RoundedRectangle(width=1.6, height=0.7, corner_radius=0.1,
                                     stroke_color=GREEN_A, stroke_width=2.5,
                                     fill_color=GREEN_A, fill_opacity=0.18)
        head_box1_lbl = MathTex(r"\mathrm{LN}\ +\ d\!\to\!64", font_size=18,
                                color=GREEN_A).move_to(head_box1.get_center())
        head1_grp = VGroup(head_box1, head_box1_lbl).move_to([0.0, 0.6, 0])

        head_box2 = RoundedRectangle(width=1.6, height=0.7, corner_radius=0.1,
                                     stroke_color=GREEN_A, stroke_width=2.5,
                                     fill_color=GREEN_A, fill_opacity=0.18)
        head_box2_lbl = MathTex(r"\mathrm{GELU}\ +\ 64\!\to\!1", font_size=18,
                                color=GREEN_A).move_to(head_box2.get_center())
        head2_grp = VGroup(head_box2, head_box2_lbl).move_to([2.5, 0.6, 0])

        a1 = Arrow(z_grid.get_right(), head1_grp.get_left(), buff=0.08,
                   stroke_width=3, color=YELLOW,
                   max_tip_length_to_length_ratio=0.18)
        a2 = Arrow(head1_grp.get_right(), head2_grp.get_left(), buff=0.08,
                   stroke_width=3, color=YELLOW,
                   max_tip_length_to_length_ratio=0.18)

        # Final logit value
        logit_tag = MathTex(rf"\mathrm{{logit}} = {h2:+.3f}",
                            font_size=24, color=RED).move_to([4.7, 0.6, 0])
        a3 = Arrow(head2_grp.get_right(), logit_tag.get_left(), buff=0.1,
                   stroke_width=3, color=RED,
                   max_tip_length_to_length_ratio=0.20)

        self.play(GrowArrow(a1), FadeIn(head1_grp, shift=RIGHT * 0.1))
        self.play(GrowArrow(a2), FadeIn(head2_grp, shift=RIGHT * 0.1))
        self.play(GrowArrow(a3), Write(logit_tag))
        self.wait(0.5)

        # sigmoid → probability
        sig_box = RoundedRectangle(width=1.5, height=0.7, corner_radius=0.1,
                                   stroke_color=RED, stroke_width=2.5,
                                   fill_color=RED, fill_opacity=0.18).move_to([0.6, -1.0, 0])
        sig_lbl = MathTex(r"\sigma(\cdot)", font_size=24, color=RED).move_to(sig_box.get_center())
        sig_grp = VGroup(sig_box, sig_lbl)

        prob_tag = MathTex(rf"P(\text{{FAIL}}) = {prob:.3f}",
                           font_size=32, color=RED).move_to([3.6, -1.0, 0])

        a4 = Arrow(logit_tag.get_bottom(), sig_grp.get_top(),
                   buff=0.1, stroke_width=3, color=RED,
                   max_tip_length_to_length_ratio=0.22)
        a5 = Arrow(sig_grp.get_right(), prob_tag.get_left(), buff=0.1,
                   stroke_width=3, color=RED,
                   max_tip_length_to_length_ratio=0.22)

        self.play(GrowArrow(a4), FadeIn(sig_grp, shift=DOWN * 0.1))
        self.play(GrowArrow(a5), Write(prob_tag))
        self.play(Flash(prob_tag, color=RED, flash_radius=0.6, num_lines=14), run_time=1.2)
        self.wait(1.6)

        outro = footer("That single scalar is what we sort by for the APFD ranking.")
        self.play(Write(outro))
        self.wait(1.8)

        clear(self)


# Manim 0.20.x exports MoveToTarget at the package top-level; keep the import
# here just in case the user is on a slimmer build.
from manim import MoveToTarget   # noqa: E402
