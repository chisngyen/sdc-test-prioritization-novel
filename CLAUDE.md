# Project CLAUDE.md — SDC Test Prioritization (target: ICSE 2027 oral)

## Target venue & narrative

- **Venue**: ICSE 2027 main track (research track), oral.
  - CFP / abstract typically opens **~Aug 2026**, full paper **~late Sep
    2026**. Confirm exact dates when CFP drops.
  - Backup: ICSE 2027 SEIP / NIER if the empirical story slips.
- **Oral pitch (one sentence)**: *"One simple recipe, eight benchmarks: a
  theory-driven Transformer baseline for SDC test prioritization that is
  exactly rotation-invariant, resolution-invariant, and audit-readable
  (curvature-monotone) -- and wins or ties on every public benchmark."*
- **Why this could be an oral (10%-ish)**: it spans
  (a) **empirical**: per-bench APFD across 8+ benchmarks beating or
  matching prior work without per-bench tuning,
  (b) **theoretical**: provable invariance / monotonicity probes,
  (c) **interpretability + safety**: conformal lower bounds,
  curvature-violation audit numbers.

## Story arc (storytelling, not a feature list)

1. **Problem framing**: SDC test prioritization is brittle to road
   rotations, sampling-rate shifts, and unphysical predictions. Existing
   recipes are tuned per-benchmark.
2. **Recipe**: a single Transformer (10ch road features) + SWA + Focal
   loss, trained once per benchmark with **identical hyperparams**.
3. **Theoretical contributions** (Exp 01, 02, 04 in `exps/tracker.md`):
   - **Resolution invariance** (FNO probe, `Delta = 0.0012`).
   - **Exact rotation invariance** (SE(2) probe, `Delta = 0.0000`).
   - **Curvature-monotonicity** (PINN probe, **5.6x** violation rate drop).
4. **Empirical contributions** (this folder's `exps/{oob,scissor,
   travel,best_all}/`):
   - APFD across **8 public benchmarks** with no per-task search.
   - **Cross-threshold transfer matrix** on OOB (severity shift).
   - **Per-bench rotation-Delta probe** (extends headline figure).
5. **Safety / audit angle**:
   - Conformal lower bound on prefix APFD (Exp 05; v1 valid-but-vacuous,
     v3 work-in-progress).
   - Violation-rate audit numbers per bench (Exp 04 PINN).

## What lives where (don't move without updating tracker.md)

```
exps/
  exp00..exp14*.py       Theory-driven experiments on SensoDat (main).
  tracker.md             Headline scoreboard for SensoDat (the canonical
                         leaderboard; keep it ASCII).
  best.md                Recipe specification of the SensoDat winner.
  oob/                   OOB-Regression benchmarks (within + transfer).
  scissor/               SDC-Scissor sample_tests (5-fold CV).
  travel/                sdc-travel competition (imbalanced, multi-gen).
  best_all/              ONE script, EVERY benchmark (the oral headline).
sensodat/                SensoDat dataset module / loaders.
paper/                   LaTeX paper draft (ICSE 2027 target).
slides/                  Beamer slides (proposal + oral build-up).
data/                    Local mirrors of all public datasets.
```

Each folder under `exps/{oob,scissor,travel,best_all}/` has its own
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

## Active todos for the ICSE 2027 angle

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
