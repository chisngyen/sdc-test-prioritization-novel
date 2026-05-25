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

## Novel methods targeting SOTA (K, L, N, O)

Four self-contained scripts that each ablate one mechanism on top of
the winner recipe. All run on the SAME 5 benchmarks and save their
own `exp_<X>_*_results.json` next to `full_all_results.json`.

### Exp K -- `exp_K_geom_tta.py` (Geometric Test-Time Augmentation)

At inference time, average sigmoids over `K_R x flip x reverse` = 24
label-preserving geometric views of each test road. The model is
already nearly SE(2)-invariant (Exp 02 Delta=0.0000 for an explicitly
E2-equivariant net); the practical Transformer is only *approximately*
invariant, so view averaging captures the residual.

- TTA family: rotations in {0, 60, 120, 180, 240, 300} deg x {flip, no-flip} x {reverse, forward}
- Ablations reported: `no_tta` (1 view), `rot6`, `full24`
- Headline: Delta_APFD = full24 - no_tta per benchmark
- Free 0.005-0.015 APFD gain expected (typical TTA range in CV/SDC)
- Distinct from Exp C (`best_all/`): C varies *resolution* N, not pose. They stack.

### Exp L -- `exp_L_apfd_direct.py` (APFD-Direct Loss via Soft-Rank)

Optimises APFD *literally*: `rank_soft(s_i) = sum_j sigmoid((s_j - s_i)/tau)`,
`APFD_soft = 1 - (1/(n*m)) * sum (rank_soft+1)*y + 1/(2n)`. Loss = 1 - APFD_soft.
Trained with a 3-phase schedule: Focal warmup (ep 0-14) -> Focal+APFD blend
(ep 15-44) -> pure APFD-direct (ep 45-74), with tau annealed 5.0 -> 0.5.

- Tau annealing: TAU_INIT=5.0 (smooth gradient) -> TAU_FINAL=0.5 (closer to true rank)
- Batch-wise APFD (each batch is its own "ranking universe", drop_last=True)
- Forced fp32 for the APFD pathway (pairwise sigmoids need precision)
- For RP/tabular: cannot use SoftRank in LightGBM; we report a `lambdarank`
  objective side-by-side (closest tabular analogue) with `delta_rank_vs_binary`
- Distinct from Exp 03 PL and Exp G prefix-PL: those optimise the *ranking
  permutation*; we optimise the APFD *formula*.

### Exp N -- `exp_N_distill.py` (Ensemble Distillation: 5 teachers -> 1 student)

Captures the SensoDat 5-config ensemble lift (+0.001 over single best)
in a single-inference-cost student via KL distillation.

- Phase 1: train 5 teachers, gammas in {1.5, 2.0, 2.5, 3.0, 3.5}
- Soft targets: mean of teacher sigmoids on the train set
- Phase 2: student loss = 0.3 * BCE(hard) + 0.7 * Bernoulli-KL(student, teacher_avg)
- Headline: `delta_student_vs_best_teacher`, `delta_student_vs_ensemble`
- For RP: 5 LightGBM teachers (different seeds) -> average; single vs avg5 reported
- Total wall-clock: ~5x training cost, 1x inference -- the deploy-time pitch

### Exp O -- `exp_O_swag.py` (SWAG -- Stochastic Weight Averaging-Gaussian)

Drop-in upgrade of vanilla SWA in the winner recipe. Tracks the SWA mean
plus a low-rank + diagonal Gaussian over the late-epoch weights (Maddox
NeurIPS 2019), then samples K posterior weight configurations at inference
and averages sigmoids. Strict generalisation of SWA (K=1 mean-only).

- SWAG_RANK=20, SWAG_SCALE=0.5
- K_LIST=[10, 30] -- compares baseline `swa_mean`, `swag_K10`, `swag_K30`
- Headline: `delta_swag30_vs_swa`
- Cost: same training, K extra forward passes at inference (~30s per bench at K=30)
- For RP: SWAG does not apply to trees; we report bagged-5 LightGBM as the
  Bayesian analogue (`bag5` vs `single`, `delta_bag_vs_single`)

## Results table (TODO: fill after running each exp on Kaggle)

| Bench (Test APFD)  | full_all baseline | + K TTA | + L APFD-Direct | + N Distill | + O SWAG |
|--------------------|-------------------|---------|-----------------|-------------|----------|
| SensoDat           |  --               | --      | --              | --          | --       |
| Scissor (5fold)    |  --               | --      | --              | --          | --       |
| its4sdc            |  --               | --      | --              | --          | --       |
| Travel             |  --               | --      | --              | --          | --       |
| RP/BeamNG_RF_1     |  --               | n/a     | --              | --          | --       |
| RP/BeamNG_RF_1_5   |  --               | n/a     | --              | --          | --       |
| RP/BeamNG_RF_2     |  --               | n/a     | --              | --          | --       |
| RP/DriverAI        |  --               | n/a     | --              | --          | --       |

(K does not apply to RP because the LightGBM features are tabular --
geometric TTA on raw points only makes sense for the Transformer.)

## Suggested combo runs (future work)

Each of K/L/N/O is *additive at inference time* with the others (except
L, which changes training). The natural follow-ups:

- **K + O** -- SWAG samples each scored with TTA. Pure inference upgrade,
  no extra training. Expected: K and O capture different uncertainty
  (epistemic vs symmetry-residual); they should compound.
- **L training + K inference** -- train with APFD-Direct, infer with
  TTA. The strongest single-recipe candidate for the oral headline.
- **N teachers each trained with O SWAG** -- distill an ensemble of
  SWAG-posterior averages. Heavy but potentially top of leaderboard.

These combos are NOT yet implemented -- they are clean ablation slots
once K, L, N, O headlines are in. Add a `exp_combo_KLO.py` etc. when
ready.

## Action items

- [ ] Run `exp_full_all.py` end-to-end on Kaggle (T4, ~60-90 min) -- baseline row
- [ ] Run `exp_K_geom_tta.py` -- expect 80-120 min (24 views x test set)
- [ ] Run `exp_L_apfd_direct.py` -- expect 90-130 min (fp32 APFD pathway is heavier)
- [ ] Run `exp_N_distill.py` -- ~5x baseline = 5-8 hours; consider running
      overnight or splitting per-bench
- [ ] Run `exp_O_swag.py` -- ~baseline + 30 extra forward passes per bench
      = ~75-100 min
- [ ] Fill the SOTA table above; cross-check Delta-APFD vs full_all baseline
- [ ] For any (K, L, N, O) that beats baseline by > 0.005 on >= 3
      benchmarks, write the combo follow-up (K+O is cheapest)
- [ ] Cross-check `apfd_full` vs `apfd_trial_mean` on each bench
      (gap should be < 0.01 for N_test > 1000)

## Update log

- 2026-05-25 -- folder created. `exp_full_all.py` written with hardcoded
  paths under `/kaggle/input/datasets/`. its4sdc promoted to first-class
  benchmark; full-test APFD added alongside multi-trial.
- 2026-05-25 -- added 4 novel SOTA-targeting exps: K (Geometric TTA),
  L (APFD-Direct via SoftRank), N (5-teacher distillation),
  O (SWAG Bayesian inference). Each self-contained, full 5 datasets,
  75 ep / batch 256.
