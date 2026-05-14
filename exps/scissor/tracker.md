# SDC-Scissor — Performance Tracker

> External benchmark: **SDC-Scissor sample_tests** (Birchler et al.,
> Zenodo 5914130).
> Small balanced set (N ~ 201, ~58% FAIL) of BeamNG-AI driven roads with
> raw + interpolated road points.
> Goal: prove SensoDat-tuned recipe also wins on a tiny balanced bench --
> a regime where transformer + SWA usually struggles.

## Scripts

- `exp_best_scissor.py` -- 5-fold stratified CV, full gamma sweep
  `{0.0, 1.0, 1.5, 2.0, 2.5}` + SWA per fold, ensemble across SWA snapshots.

## Why 5-fold instead of 80/20

N ~ 201 is too small for a single 80/20 split: a 20% test set is only
~40 tests with ~24 fails, giving APFD sigma > 0.05. 5-fold CV averages
across all 201 tests as held-out at some point, yielding tighter
per-config estimates while still keeping fold-level variance honest.

## Latest numbers (TODO: fill in after run)

| Config              | Mean APFD | sigma | Notes |
|---------------------|-----------|-------|-------|
| gamma=0.0 best-ckpt |  --       |  --   |       |
| gamma=1.0 best-ckpt |  --       |  --   |       |
| gamma=1.5 best-ckpt |  --       |  --   |       |
| gamma=2.0 best-ckpt |  --       |  --   |       |
| gamma=2.5 best-ckpt |  --       |  --   |       |
| gamma=2.5 + SWA     |  --       |  --   | likely winner from SensoDat priors |
| Ensemble 5 SWA      |  --       |  --   | should match SensoDat ensemble pattern |
| **Best-per-fold**   |  --       |  --   | upper bound (oracle gamma per fold) |

## Story for the paper / oral

- **Small-data robustness**: SensoDat tuned at N=14k, Scissor is N=201.
  The recipe must not overfit. Per-fold sigma is the diagnostic.
- **Balanced FAIL rate** (~58%) flips the imbalance signal vs SensoDat
  (~33% FAIL). Focal gamma should not buy as much here -- if gamma=0.0
  ties gamma=2.5, that confirms focal's job is class-imbalance not
  ranking.
- **Headline**: cross-bench result table comparing SDC-Scissor, OOB, and
  Travel side-by-side; same model, no per-bench search.

## Action items

- [ ] Run baseline (random / road-length) APFD on the same folds for a
      relative-gain number, not just the absolute APFD.
- [ ] Try rotation-invariance probe (Exp 02-style 6-rotation Delta) on
      the small set -- much cheaper here than on full SensoDat.
- [ ] If gamma=0.0 ties gamma=2.5, write it up as evidence that focal is
      doing class-imbalance work, not ranking work.
- [ ] Compare against Birchler et al.'s original SDC-Scissor numbers
      (they report a Random-Forest baseline on these tests).

## Update log

- 2026-05-14 -- folder spun off from `exps/`. File moved unchanged; only
  SEARCH_ROOTS / OUTPUT_DIR depth was patched for the deeper layout.
