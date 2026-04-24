# LaTeX Proposal Slides

This folder contains a professional Beamer deck for proposing an idea in the SDC test prioritization context.

## Files
- `main.tex`: main slide source
- `Pipeline.png`: copied from repo root
- `example.png`: copied from repo root

## Build
From repository root:

```bash
cd slides/latex_proposal
latexmk -pdf main.tex
```

If `latexmk` is not available:

```bash
pdflatex main.tex
pdflatex main.tex
```

## Quick Customization
- Update title/author/institute/date in `main.tex` preamble.
- Replace timeline and contribution bullets with your exact project scope.
- Keep image paths local in this folder for portability.
