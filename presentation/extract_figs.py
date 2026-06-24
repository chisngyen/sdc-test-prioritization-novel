# -*- coding: utf-8 -*-
"""
Extract every tikzpicture (and key display formulas) from se2_slides.tex and
emit a standalone `assets/figs.tex` (preview/tightpage) that reuses the deck's
own TikZ -> compiling gives PDF/PNG figures pixel-identical to the deck.
Records figure index -> nearest \begin{frame} title in assets/figmap.txt.
"""
import re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

src = io.open("se2_slides.tex", encoding="utf-8").read()

PREAMBLE = r"""\documentclass{article}
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{3pt}
\usepackage[utf8]{inputenc}
\usepackage{vietnam}
\usepackage{amsmath,amssymb}
\usepackage[table]{xcolor}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{pifont}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, fit, backgrounds, decorations.pathreplacing,
                shapes.geometric, calc, decorations.pathmorphing}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
\definecolor{mDarkTeal}{HTML}{23373B}
\definecolor{mLightBrown}{HTML}{EB811B}
\definecolor{softgray}{HTML}{F2F4F7}
\definecolor{tealtint}{HTML}{D6DEE0}
\colorlet{navydeep}{mDarkTeal}\colorlet{navymid}{mDarkTeal!75}
\colorlet{accent}{mLightBrown}\colorlet{success}{green!50!black}
\colorlet{alert}{red!70!black}\colorlet{lightnavy}{tealtint}
\definecolor{cRot}{HTML}{B85450}\definecolor{cRes}{HTML}{C8A951}
\definecolor{cInv}{HTML}{2C3E6B}\definecolor{cRotL}{HTML}{F2E6E5}
\definecolor{cResL}{HTML}{F5F0E0}\definecolor{cInvL}{HTML}{EEF0F5}
\definecolor{cHard}{HTML}{B85450}\definecolor{cSoft}{HTML}{C8A951}
\definecolor{cUpper}{HTML}{2E86AB}
\begin{document}
"""

# ---- collect (title, tikz-body) in document order ----
frame_re = re.compile(r"\\begin\{frame\}(?:\[[^\]]*\])?\{([^}]*)\}")
def title_before(pos):
    last = None
    for m in frame_re.finditer(src, 0, pos):
        last = m.group(1)
    return last or "?"

figs = []   # (label, latex)
# tikzpictures (handle optional [..] options already inside \begin{tikzpicture}...)
i = 0
idx = 0
while True:
    b = src.find(r"\begin{tikzpicture}", i)
    if b < 0: break
    e = src.find(r"\end{tikzpicture}", b)
    block = src[b:e+len(r"\end{tikzpicture}")]
    idx += 1
    figs.append((f"T{idx:02d} :: {title_before(b)}", block))
    i = e + 10

# ---- display formulas to render (label, latex) ----
FORMULAS = [
    ("F_io_R",   r"$\mathcal{R} = \{p_1, \dots, p_N\} \subset \mathbb{R}^2$"),
    ("F_io_f",   r"$f_\theta : \mathcal{R} \mapsto [0,1]$"),
    ("F_io_pi",  r"$\pi^{*} = \arg\max_{\pi}\; \mathrm{APFD}(\pi)$"),
    ("F_theorem",r"$\boxed{\; f_\theta(R\,\mathcal{R} + t) \;=\; f_\theta(\mathcal{R}) \;}$"),
    ("F_bias",   r"$\mathrm{bias}_{ij} = \mathrm{MLP}\bigl(\sin(\Delta s_{ij}\cdot\omega)\bigr) \in \mathbb{R}^{8}$"),
    ("F_focal",  r"$\mathcal{L} = -\,\alpha\,(1-\hat{p}_t)^{\gamma}\,\log \hat{p}_t, \qquad \gamma = 1.5$"),
    ("F_apfd",   r"$\displaystyle \mathrm{APFD}(\pi) = 1 - \frac{\sum_{i=1}^{m} TF_i}{n\cdot m} + \frac{1}{2n}$"),
    ("F_bk_good",r"$\displaystyle \mathrm{APFD} = 1 - \frac{1+2}{5\cdot 2} + \frac{1}{10} = \mathbf{0.80}$"),
    ("F_bk_bad", r"$\displaystyle \mathrm{APFD} = 1 - \frac{4+5}{5\cdot 2} + \frac{1}{10} = \mathbf{0.20}$"),
    ("F_kappa",  r"$\kappa_i = \Delta\theta_i / \Delta s_i$"),
    ("F_dk",     r"$d\kappa/ds$"),
    ("F_d2k",    r"$d^2\kappa/ds^2$"),
    ("F_snorm",  r"$s_{\text{norm}} = s/L \in [0,1]$"),
    ("F_sigma",  r"$\sigma_{\text{local}}(\kappa)$"),
]

out = [PREAMBLE]
mapping = []
n = 0
for label, block in figs:
    n += 1
    out.append(f"% ===== fig {n}: {label} =====\n\\begin{{preview}}\n{block}\n\\end{{preview}}\n")
    mapping.append(f"{n:02d}  TIKZ  {label}")
for label, tex in FORMULAS:
    n += 1
    out.append(f"% ===== fig {n}: {label} =====\n\\begin{{preview}}\n{tex}\n\\end{{preview}}\n")
    mapping.append(f"{n:02d}  FORM  {label}")
out.append(r"\end{document}")

io.open("assets/figs.tex","w",encoding="utf-8").write("\n".join(out))
io.open("assets/figmap.txt","w",encoding="utf-8").write("\n".join(mapping))
print(f"tikz figures: {len(figs)} | formulas: {len(FORMULAS)} | total pages: {n}")
print("wrote assets/figs.tex + assets/figmap.txt")
