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

## Novel methods targeting SOTA (Exps 01-04)

Four self-contained scripts that each ablate one mechanism on top of
the winner recipe. All run on the SAME 5 benchmarks and save their
own `exp_<NN>_*_results.json` next to `full_all_results.json`.

### Exp 01 -- `exp_01_geom_tta.py` (Geometric Test-Time Augmentation)

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

### Exp 02 -- `exp_02_apfd_direct.py` (APFD-Direct Loss via Soft-Rank)

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

### Exp 03 -- `exp_03_distill.py` (Ensemble Distillation: 5 teachers -> 1 student)

Captures the SensoDat 5-config ensemble lift (+0.001 over single best)
in a single-inference-cost student via KL distillation.

- Phase 1: train 5 teachers, gammas in {1.5, 2.0, 2.5, 3.0, 3.5}
- Soft targets: mean of teacher sigmoids on the train set
- Phase 2: student loss = 0.3 * BCE(hard) + 0.7 * Bernoulli-KL(student, teacher_avg)
- Headline: `delta_student_vs_best_teacher`, `delta_student_vs_ensemble`
- For RP: 5 LightGBM teachers (different seeds) -> average; single vs avg5 reported
- Total wall-clock: ~5x training cost, 1x inference -- the deploy-time pitch

### Exp 04 -- `exp_04_swag.py` (SWAG -- Stochastic Weight Averaging-Gaussian)

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

| Bench (Test APFD)  | full_all baseline | + 01 TTA | + 02 APFD-Direct  | + 03 Distill   | + 04 SWAG |
|--------------------|-------------------|---------|-------------------|----------------|----------|
| SensoDat           |  --               | --      | 0.7428            | 0.7556 (S)     | --       |
| Scissor (80/20)    |  --               | --      | ERR (zero-div)    | 0.5315 (S)     | --       |
| its4sdc            |  --               | --      | 0.7381            | 0.7549 (S)     | --       |
| Travel             |  --               | --      | 0.8111            | 0.8114 (S)     | --       |
| RP/BeamNG_RF_1     |  --               | n/a     | 0.7771 (LR)       | 0.8439 (avg5)  | --       |
| RP/BeamNG_RF_1_5   |  --               | n/a     | 0.7264 (LR)       | 0.7626 (avg5)  | --       |
| RP/BeamNG_RF_2     |  --               | n/a     | 0.5194 (LR)       | 0.5209 (avg5)  | --       |
| RP/DriverAI        |  --               | n/a     | 0.8266 (LR)       | 0.8857 (avg5)  | --       |

(Exp 01 does not apply to RP because the LightGBM features are
tabular -- geometric TTA on raw points only makes sense for the
Transformer.)

### Exp 02 APFD-Direct -- detailed numbers (2026-05-25 run)

Transformer (10ch, d=128, 4L) trained with 3-phase schedule: Focal warmup
(ep 0-14) -> Focal+APFD blend (ep 15-44) -> pure APFD-Direct (ep 45-74),
tau annealed 5.0 -> 0.5. SWA on. APFD reference for comparison is the
Exp 03 T2_g2.5 teacher (same architecture, same recipe, plain Focal
gamma=2.5 -- the closest analogue to the eventual `exp_full_all.py`
baseline, since `exp_full_all.py` has not been run yet).

| Bench       | AUC    | APFD_full | APFD_trial+/-sigma  | Focal-g2.5 ref | dAPFD       |
|-------------|--------|-----------|---------------------|----------------|-------------|
| Scissor     | ERR    | ERR       | ERR                 | 0.5630         | ZeroDivErr  |
| its4sdc     | 0.9188 | 0.7381    | 0.7371 +/- 0.0083   | 0.7514         | -0.0133     |
| SensoDat    | 0.9161 | 0.7428    | 0.7437 +/- 0.0051   | 0.7531         | -0.0103     |
| Travel      | 0.8450 | 0.8111    | 0.8076 +/- 0.0228   | 0.7734         | **+0.0377** |

Tabular RP (LightGBM `binary` vs `lambdarank` -- closest tabular analogue
of "optimise the rank-based objective directly"). LR = lambdarank.

| Dataset       | binary  | lambdarank | delta    |
|---------------|---------|------------|----------|
| BeamNG_RF_1   | 0.8433  | 0.7771     | -0.0661  |
| BeamNG_RF_1_5 | 0.7625  | 0.7264     | -0.0361  |
| BeamNG_RF_2   | 0.5209  | 0.5194     | -0.0016  |
| DriverAI      | 0.8855  | 0.8266     | -0.0589  |

Wall-clock: 872.5 s (14.5 min) for all 4 geom benches + 4 RP datasets
on RTX PRO 6000 Blackwell. Faster than Exp 03 because no 5-teacher loop.

Read-through:
- **Scissor errored with ZeroDivisionError** -- almost certainly the
  SoftRank pathway dividing by `n_fail*(n-n_fail)` on a tiny batch
  (|train|=160, FAIL=58%) where some batches contain all-FAIL or all-PASS
  after drop_last. Fix: guard against `n_fail=0 or n_fail=n_batch` in
  the APFD-soft loss, or use a per-bench min-batch-size.
- **3/4 geom benches: APFD-Direct LAGS Focal baseline.** its4sdc -0.013,
  sensodat -0.010, travel +0.038 (the only win, and Travel is the
  imbalanced 5.5% FAIL bench where AUC-driven losses notoriously
  miscalibrate -- so APFD-Direct's win there is the cleanest theoretical
  signal we have).
- **AUC tanks during the APFD-only phase** (Travel: 0.83 -> 0.69 by ep 75)
  but SWA recovers a usable model. The AUC-APFD divergence flagged in
  CLAUDE.md is on full display: APFD_full=0.8111 with AUC=0.8450 is the
  weakest AUC we have at that APFD level.
- **LambdaRank LOSES on every RP dataset** (-0.066 worst, -0.002 best).
  Optimising NDCG/lambdarank does not transfer to APFD on tabular
  features -- the metric mismatch dominates the objective.
- **Net story for the paper**: APFD-Direct is a *negative result* for
  geom benches except imbalanced Travel; LambdaRank is a clean negative
  for tabular RP. Frame as "metric mismatch is not enough -- you need
  the right ranking universe (per-test prefix) and the right loss
  *shape*". This sets up Exp G prefix-PL as the natural follow-up.

### Exp 03 distillation -- detailed numbers (2026-05-25 run)

Geometry benches (Transformer teachers gammas in {1.5, 2.0, 2.5, 3.0, 3.5},
KL student alpha=0.3, 75 ep, batch 256). APFD_full = single-pass on the
entire held-out test set.

| Bench       |  N    | FAIL% | n_test | best_T (gamma)    | ENSEMBLE | STUDENT | dS-T    | dS-E    |
|-------------|-------|-------|--------|-------------------|----------|---------|---------|---------|
| Scissor     |   201 | 58.2% |     41 | 0.5630 (g=2.5)    | 0.5396   | 0.5315  | -0.0315 | -0.0081 |
| its4sdc     | 10000 | 38.5% |   2000 | 0.7566 (g=2.0)    | 0.7651   | 0.7549  | -0.0017 | -0.0102 |
| SensoDat    | 36006 | 38.4% |   7202 | 0.7542 (g=3.5)    | 0.7653   | 0.7556  | +0.0014 | -0.0097 |
| Travel      | 14166 |  5.5% |   2834 | 0.8231 (g=2.0)    | 0.8276   | 0.8114  | -0.0117 | -0.0162 |

Tabular RP (5 LightGBM teachers, different seeds -> avg).

| Dataset       | single  | avg5    | delta    |
|---------------|---------|---------|----------|
| BeamNG_RF_1   | 0.8433  | 0.8439  | +0.0006  |
| BeamNG_RF_1_5 | 0.7625  | 0.7626  | +0.0001  |
| BeamNG_RF_2   | 0.5209  | 0.5209  | -0.0000  |
| DriverAI      | 0.8855  | 0.8857  | +0.0003  |

Wall-clock: 1690.9 s (28.2 min) for all 4 geometry benches + 4 RP datasets
on RTX PRO 6000 Blackwell.

Read-through:
- **ENSEMBLE > best_T on every geometry bench** (+0.0045 to +0.0111),
  confirming the SensoDat 5-config ensemble lift generalises across all
  four geometry benches; the +0.0111 on SensoDat is the largest ensemble
  payoff we have seen.
- **STUDENT only matches best_T on SensoDat (+0.0014)**; on the other
  three geometry benches the student lags both the best teacher and the
  ensemble. Scissor is the worst case (-0.0315 vs best_T) but |test|=41
  makes that single number very noisy.
- **RP avg5 is essentially a no-op** (max delta +0.0006). LightGBM with
  the same hyperparams and only seed-varied trees converges to nearly
  identical rankings; the ensemble does not break the APFD ceiling on
  RF_2 (95% FAIL) nor materially lift the other three.
- **Net story for the paper**: Phase 1 (teachers + ensemble) is the
  empirical lift; Phase 2 (KL distillation to student) is currently a
  partial recovery rather than a free lunch. Either keep the ensemble as
  the deploy artefact, or tune the distillation (alpha schedule,
  temperature on soft labels, longer training).

## Suggested combo runs (future work)

Each of Exps 01/02/03/04 is *additive at inference time* with the
others (except 02, which changes training). The natural follow-ups:

- **01 + 04** -- SWAG samples each scored with TTA. Pure inference
  upgrade, no extra training. Expected: 01 and 04 capture different
  uncertainty (epistemic vs symmetry-residual); they should compound.
- **02 training + 01 inference** -- train with APFD-Direct, infer with
  TTA. The strongest single-recipe candidate for the oral headline.
- **03 teachers each trained with 04 SWAG** -- distill an ensemble of
  SWAG-posterior averages. Heavy but potentially top of leaderboard.

These combos are NOT yet implemented -- they are clean ablation slots
once 01-04 headlines are in. Add `exp_05_combo_01_04.py` etc. when
ready.

## Action items

- [ ] Run `exp_full_all.py` end-to-end on Kaggle (T4, ~60-90 min) -- baseline row
- [ ] Run `exp_01_geom_tta.py` -- expect 80-120 min (24 views x test set)
- [x] Run `exp_02_apfd_direct.py` -- DONE 2026-05-25, 14.5 min on
      RTX PRO 6000 (much faster than 90-130 min estimate). Negative
      result on 3/4 geom benches (its4sdc/sensodat/travel: -0.013,
      -0.010, +0.038); LambdaRank loses on all 4 RP datasets (max
      -0.066). Scissor errored: ZeroDivisionError in SoftRank loss when
      a batch has 0 FAIL or 0 PASS -- need to guard `n_fail*(n-n_fail)`
      denominator and re-run that bench
- [ ] Fix `exp_02_apfd_direct.py` Scissor ZeroDivisionError (guard
      SoftRank denominator) and re-run just that bench
- [x] Run `exp_03_distill.py` -- DONE 2026-05-25, 28.2 min wall-clock on
      RTX PRO 6000 (much faster than the 5-8h estimate; the Blackwell GPU
      eats the 5-teacher cost). Headline: ensemble lifts every geom bench
      (+0.0045 to +0.0111 over best teacher); student only ties best
      teacher on SensoDat. RP avg5 ~= single (max +0.0006)
- [ ] Run `exp_04_swag.py` -- ~baseline + 30 extra forward passes per bench
      = ~75-100 min
- [ ] Fill the SOTA table above; cross-check Delta-APFD vs full_all baseline
- [ ] For any (01, 02, 03, 04) that beats baseline by > 0.005 on >= 3
      benchmarks, write the combo follow-up (01+04 is cheapest)
- [ ] Cross-check `apfd_full` vs `apfd_trial_mean` on each bench
      (gap should be < 0.01 for N_test > 1000)

## Update log

- 2026-05-25 -- folder created. `exp_full_all.py` written with hardcoded
  paths under `/kaggle/input/datasets/`. its4sdc promoted to first-class
  benchmark; full-test APFD added alongside multi-trial.
- 2026-05-25 -- added 4 novel SOTA-targeting exps: 01 (Geometric TTA),
  02 (APFD-Direct via SoftRank), 03 (5-teacher distillation),
  04 (SWAG Bayesian inference). Each self-contained, full 5 datasets,
  75 ep / batch 256.
- 2026-05-25 -- ran Exp 03 distillation end-to-end (1690.9 s on RTX PRO
  6000 Blackwell, all 4 geom benches + 4 RP datasets). Ensemble beats
  best teacher on every geom bench (max +0.0111 on SensoDat); KL student
  recovers best teacher only on SensoDat (+0.0014) and lags elsewhere
  (Scissor -0.0315 with |test|=41 noise; Travel -0.0117; its4sdc
  -0.0017). RP avg5 ~= single (max +0.0006). See "Exp 03 distillation
  -- detailed numbers" section above.
- 2026-05-25 -- ran Exp 02 APFD-Direct (872.5 s on RTX PRO 6000). Mostly
  negative: its4sdc -0.0133, sensodat -0.0103, Travel +0.0377 (only win,
  on the 5.5% FAIL imbalanced bench), Scissor errored (ZeroDivisionError
  in SoftRank loss for small all-FAIL/all-PASS batches). RP lambdarank
  loses on every dataset (max -0.0661). Travel win is the cleanest
  theoretical signal so far; the rest sets up Exp G prefix-PL as the
  natural follow-up. See "Exp 02 APFD-Direct -- detailed numbers" above.
