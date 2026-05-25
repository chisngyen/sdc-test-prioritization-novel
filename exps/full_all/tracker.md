# Full-Run ALL Benchmarks -- Tracker

> Companion to `exps/best_all/`. Same recipe, but every dataset under
> `/kaggle/input/datasets/` runs end-to-end with **no subsampling**, and
> APFD is reported on the entire held-out test set (in addition to the
> 30%-trial protocol for sigma).

## Scope

One file: `exp_full_all.py`. Five benchmarks, hardcoded paths.

| Tag      | Path                                                                  | Protocol      |
|----------|-----------------------------------------------------------------------|---------------|
| sensodat | chinguyeen/sdc-sensodat (sensodat_full.json)                          | 80/20         |
| scissor  | chinguyeen/sdc-scissor/.../sample_tests                               | 5-fold CV     |
| its4sdc  | chiboiz/its4sdc/executed-10000                                        | 80/20         |
| travel   | chiboiz/sdc-travel/competition (66 generator campaigns pooled)        | 80/20         |
| rp       | chiboiz/sdc-pririotizer-rp/SDC-Pririotizer-RP/datasets/fullroad/...   | 5-fold CV     |

## Recipe

| Family       | Recipe                                                | Eval                                    |
|--------------|-------------------------------------------------------|-----------------------------------------|
| Geometry     | Transformer (10ch, d=128, 4L) + SWA + Focal gamma=2.5 | APFD on FULL test set + 30-trial sigma  |
| Tabular (RP) | LightGBM (`num_leaves=63, lr=0.05`), fallback HGB     | 5-fold APFD on FULL fold-test set       |

Identical hyperparameters to `best_all/` and to the SensoDat winner in
`exps/tracker.md`. The point is consistency across benchmarks, not
per-bench tuning.

## What "full" means here

1. **Full data**  -- training uses the entire 80% split; no row sampling.
2. **Full test**  -- `apfd_full` is single-pass APFD on the entire test
   set. No subsampling. This is the headline number per dataset.
3. **Full coverage** -- every dataset present under
   `/kaggle/input/datasets/` is attempted; missing folders print `[SKIP]`
   and the run continues.
4. **Full metrics** -- per-dataset report carries:
   - `N`, `FAIL%`, `n_train`, `n_test`, `n_fail_train`, `n_fail_test`
   - `AUC` (held-out validation)
   - `apfd_full`         single-pass APFD on the entire held-out set
   - `apfd_trial_mean`   mean of 30-trial APFD at sample_size = 30% (>=50)
   - `apfd_trial_std`    sigma across the 30 trials (publication number)

`apfd_full` and `apfd_trial_mean` should agree to within a few thousandths
when the test set is large; the gap grows when |test| is small or the
FAIL rate is extreme.

## Latest numbers (TODO: fill after a full run)

### Geometry benches (Transformer + SWA + Focal gamma=2.5)

| Bench       |   N  | FAIL% | n_train | n_test | AUC | APFD_full | APFD_trial (30) |
|-------------|------|-------|---------|--------|-----|-----------|------------------|
| SensoDat    | --   | --    | --      | --     | --  | --        | --               |
| Scissor (5fold) | -- | --  | --      | --     | --  | --        | --               |
| its4sdc     | --   | --    | --      | --     | --  | --        | --               |
| Travel      | --   | --    | --      | --     | --  | --        | --               |

### Tabular (LightGBM, 5-fold CV) -- expected from prior runs

| Dataset       |  N   | FAIL% | AUC    | APFD (mean +/- sigma) |
|---------------|------|-------|--------|------------------------|
| BeamNG_RF_1   | 1178 | 26.5% | 0.9669 | 0.8433 +/- 0.0109      |
| BeamNG_RF_1_5 | 5638 | 45.1% | 0.9783 | 0.7625 +/- 0.0015      |
| BeamNG_RF_2   | 1729 | 95.7% | 0.9893 | 0.5209 +/- 0.0006      |
| DriverAI      | 5630 | 18.6% | 0.9733 | 0.8855 +/- 0.0061      |

(RF_2 is a 95.7%-FAIL benchmark: APFD ceiling ~0.52 -- frame as an
APFD-degeneracy probe, not a defeat.)

## Relationship to `best_all/`

| Aspect                       | `best_all/`                          | `full_all/`                                  |
|------------------------------|--------------------------------------|----------------------------------------------|
| Path discovery               | Walks SEARCH_ROOTS, fallbacks        | Hardcoded `/kaggle/input/datasets/...`       |
| its4sdc                      | OOB-Regression fallback only         | First-class benchmark                        |
| OOB-Regression (0-1,0-3,0-5) | Yes (if present)                     | Not included (use `exps/oob/` for that)      |
| Eval                         | Multi-trial APFD only (30% sample)   | Full-test APFD + multi-trial sigma           |
| Output JSON                  | `best_all_results.json`              | `full_all_results.json`                      |

Use `best_all/` for the headline "one recipe, many benches" oral figure
(it includes the OOB severity-split story). Use this folder when you
want **APFD on the entire held-out set** without any trial sampling, or
when you want its4sdc reported as its own row instead of folded into an
OOB fallback.

## Action items

- [ ] Run `exp_full_all.py` end-to-end on Kaggle (T4, ~60-90 min).
- [ ] Fill the geometry table above; cross-check `apfd_full` vs
      `apfd_trial_mean` per row (gap should be < 0.01 for N_test > 1000).
- [ ] If `apfd_full` and `apfd_trial_mean` diverge by > 0.02 on any
      bench, investigate whether the 30% subsample size is masking a
      tail-FAIL effect.

## Update log

- 2026-05-25 -- folder created. `exp_full_all.py` written with hardcoded
  paths under `/kaggle/input/datasets/`. its4sdc promoted to first-class
  benchmark; full-test APFD added alongside multi-trial.
