# Project CLAUDE.md — SDC Test Prioritization (SOICT 2026: CliffordTrajNet)

## Status (updated 2026-09-21)

- **This repository is the SOICT 2026 record.** The submitted paper is
  **CliffordTrajNet** (Clifford algebra + continuous state-space, SDC and UAV),
  source in `manuscripts/paper/soict/` (`main.tex`, Springer `llncs`,
  single-blind, 12 pages excluding references, full paper due 2026-09-25).
  It builds cleanly with two `pdflatex` passes plus `bibtex`: 14 pages in
  total, references start on page 12, no errors, no undefined references.
- The **SE2RoadNet** story below is the earlier lineage of this work, and
  `manuscripts/paper/soict/outline.md` still describes it. It is not the
  submitted paper. The core numbers further down are SE2RoadNet numbers.
- The **FSE 2027 extension** continues in a separate private repository. The
  old "ICSE 2027" todo list moved there.
- Data is not stored in Git; see `docs/DATA.md`.

## Target venue & narrative (SE2RoadNet lineage)

- **Venue**: SOICT (Symposium on Information and Communication Technology).
  - Main track / AI & Software Engineering.
- **Paper pitch (one sentence)**: *"A theory-driven, geometry- and physics-grounded Transformer for SDC test prioritization that is provably rotation-invariant, resolution-invariant, and curvature-monotone -- achieving state-of-the-art APFD across public benchmarks without per-task tuning."*
- **Core pillars**:
  (a) **Geometric guarantees**: Exact $SE(2)$ rotation invariance ($\Delta = 0.0000$) via coordinate-free features and relative attention bias.
  (b) **Physical regularization**: Physics-informed loss (PINN) enforcing curvature monotonicity, cutting violation rates by 5.6x.
  (c) **Empirical generalization**: SOTA or tie across benchmarks (SensoDat, Scissor, OOB, Travel) with a unified recipe.

## Story arc (storytelling, not a feature list)

1. **Problem framing**: SDC test prioritization is brittle to road
   rotations, sampling-rate shifts, and unphysical predictions. Existing
   recipes are tuned per-benchmark or aggregate sequences into lossy scalars.
2. **Architecture**: SE2RoadNet with 7-ch coordinate-free features +
   relative-arclength attention bias + SWA + Focal loss.
3. **Theoretical contributions** (Exp 01, 02, 04):
   - **Resolution invariance** (FNO / continuous curve formulation).
   - **Exact rotation invariance** (SE(2) probe, $\Delta = 0.0000$).
   - **Curvature-monotonicity** (PINN probe, 5.6x violation rate drop).
4. **Empirical contributions**:
   - APFD across public benchmarks (SensoDat, Scissor, OOB, Travel).
   - Cross-threshold and cross-bench transfer evaluations.

## What lives where (don't move without updating tracker.md)

```
manuscripts/
  paper/                 Paper drafts (icst2026_roadfury.tex, related_work.tex, figures, soict draft).
  presentation/          Beamer slides (se2_slides.tex), PPTX decks, speaker scripts.
exps/
  core/                  Primary theory-driven models (exp00, exp01, exp02, exp03, exp04, exp10).
  probes/                Invariance and ablation probes (exp00a, exp02b, exp02c, exp04b).
  benchmarks/            Cross-benchmark evaluations (best_all, oob, scissor, travel, uav, full_all).
  exploratory/           Auxiliary and exploratory experiments (exp05..exp09, exp11..exp16).
  tracker.md             Headline scoreboard for SensoDat (the canonical leaderboard; keep it ASCII).
  best.md                Recipe specification of the winner.
  results/               JSON results from runs.
docs/                    Detailed Vietnamese math and architecture breakdown (hai_method_chi_tiet.tex).
sensodat/                SensoDat dataset module / loaders.
data/                    Local mirrors of public datasets.
```

Each folder under `exps/benchmarks/{oob,scissor,travel,best_all}/` has its own
`tracker.md`. **The SensoDat tracker at `exps/tracker.md` is the master
scoreboard** for theory exps; per-bench trackers are for cross-bench
generalisation.

## Working preferences (durable, project-level)

- **Vietnamese** for discussion / explanation; **English** for code,
  LaTeX, paper content, commit messages.
- **No trailing summaries** after edits -- I read the diff.
- **No unnecessary comments** in code; comments only when WHY is
  non-obvious (a hidden invariant, a workaround, a subtle constraint).
- **Storytelling first**: when drafting paper sections or slides, lead
  with the narrative arc and put numbers where they support the arc;
  don't dump tables.
- **Cite exact numbers**: APFD ± sigma, AUC, wall-clock, params.
- **Pure ASCII** in `tracker.md` files (cp1252 mojibake on Windows).
- **Figures path** in the paper: `../figures/rqX/filename.pdf`.
- **LaTeX**: compile twice for TOC/refs; report errors only (skip chktex).
- **Commit messages**: concise, imperative mood ("add", "fix", "refine").

## Conventions for new experiments

- **Self-contained scripts** runnable on Kaggle by pasting one file.
- Use `SEARCH_ROOTS` pattern (see `exps/best_all/exp_best_all.py`) so the
  script discovers data on both Kaggle and local layouts without args.
- Save artifacts to `OUTPUT_DIR = /kaggle/working or ../../models`.
- Multi-trial APFD: **30 trials**, sample size = `max(50, 0.3 * |test|)`.
- For seed control: `random_state=42` everywhere (split, sampler, etc.).
- Always emit a `*_results.json` next to the saved model.
- For ablations: report **mean and sigma** across trials; the sigma is
  often the more important publication number.

## Numbers to beat / cite (as of 2026-05-14)

- **SensoDat best-single**: APFD = **0.8066 ± 0.0124** (Transformer + SWA
  + Focal gamma=2.5, ~3 min).
- **SensoDat 5-config ensemble**: APFD = **0.8077 ± 0.0115**.
- **Highest project AUC**: **0.9385** (Exp 10 DiffAPFD on SE(2)).
- **Curvature violation rate**: control 17.57% -> monotone-PINN 3.14%
  (5.6x reduction) at the same APFD.
- **OOB transfer matrix**: best off-diagonal source = OOB-0-3 (single
  model lands within 0.05 APFD on all three thresholds).
- **RP LightGBM**: APFD 0.84 / 0.76 / 0.52 / 0.89 on
  RF_1 / RF_1_5 / RF_2 / DriverAI (RF_2 ceiling is ~0.52 because 95% FAIL).

## Open items carried to the FSE 2027 extension repository

These were the "ICSE 2027 angle" todos. They are no longer worked on here.

- [ ] Run `exps/best_all/exp_best_all.py` end-to-end and fill the empty
      tracker tables. **This is the headline figure**.
- [ ] Per-bench **rotation-Delta probe** (Exp 02 protocol, 6 rotations).
- [ ] Per-bench **resolution-Delta probe** (Exp 01 protocol, N in {64..197}).
- [ ] Per-bench **curvature-violation rate** (Exp 04 protocol, alpha=1.5).
- [ ] **Cross-bench transfer matrix** (5 benches x 5 benches; OOB-style).
- [ ] Conformal v3: top-K miss-rate CRC (Exp 05 follow-up) with valid
      AND non-vacuous bounds.
- [ ] Paper outline draft (`paper/` folder) -- ICSE 2027 single-blind
      template.
- [ ] Oral storyboard (`slides/`) -- 12 min talk + 3 min Q&A, build off
      the proposal slides already in `slides/`.

## Honest weaknesses to keep in mind (so a reviewer doesn't find them first)

- **AUC and APFD diverge** in nearly every exp -- higher AUC does not
  imply higher APFD. We have to argue carefully which metric matters
  when.
- **Listwise losses (Exp 03)** did NOT raise mean APFD; they cut sigma.
  Frame as "stability contribution" not "headline contribution".
- **Naive geometric SSL (Exp 07)** transferred poorly -- the foundation
  model story needs a physics-informed pretext (Exp 07b in progress).
- **Conformal v1 valid-but-vacuous, v2 informative-but-invalid** -- the
  safety section needs a v3 to ship; don't oversell.
- **IRM / TENT did not close the SensoDat -> Competition gap.** The
  distribution-shift section is a known-negative for now.
- **APFD on RF_2 = 0.52 is a CEILING**, not a defeat -- but reviewers will
  need that explained explicitly (95% FAIL rate dominates).
