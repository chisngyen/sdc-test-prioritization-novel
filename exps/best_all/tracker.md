# Best-of-Recipe ALL Benchmarks — Tracker

> One recipe, every benchmark. This folder hosts the "ICSE 2027 headline"
> configuration: SensoDat-tuned Transformer + SWA + Focal(gamma=2.5),
> re-applied without modification to every benchmark we have data for, plus
> a LightGBM equivalent for the pre-tabulated Birchler RP datasets.
>
> The script is **deliberately self-contained**: one `.py` file, no project
> imports. Paste into Kaggle or run locally as-is.

## Scripts in this folder

### Headline runner
- `exp_best_all.py` -- THE unified runner. Loops over all benchmarks,
  trains/evaluates with one recipe each, saves `best_all_results.json`.
- `exp_rp_external_bench.py` -- standalone Birchler RP bench (LightGBM,
  pre-tabulated features). Kept separately because the recipe is
  fundamentally different (gradient boosting, not Transformer).

### Ten novel ICSE-2027 experiments (A-J)
Each is self-contained, paste-into-Kaggle, and ties to a specific
insight in `exps/tracker.md`. Save `exp_<ID>_*.json` artifacts.

- `exp_A_cross_bench_transfer.py` -- k x k transfer matrix across
  {SensoDat, OOB-0-3, Scissor, Travel}. The "one recipe, many benches"
  headline figure. **Novelty: SDC literature has no cross-bench
  transfer matrix.**
- `exp_B_auc_apfd_identity.py` -- proves and empirically verifies the
  exact identity **APFD = (1-p) * AUC + p/2** (no ties, same split).
  Re-frames the tracker's recurring "AUC up, APFD down" as an
  *evaluation-split mismatch*, not a metric pathology. **Novelty:
  closed-form connection between AUC and APFD is new for SE.**
- `exp_C_multi_resolution_tta.py` -- Multi-resolution test-time
  ensembling via FNO modes. Runs the SAME model at N in {64..197},
  averages sigmoids, builds an *agreement filter* as a free safety
  layer. **Novelty: resolution-TTA never tried in SDC.**
- `exp_D_conformal_abstain.py` -- selective prediction with conformal
  abstention (Geifman & El-Yaniv 2017 style). Provable mis-coverage
  bound on the non-abstained subset. **Novelty: SDC test prio has no
  selective-prediction story; this is the statistical-safety twin of
  Exp 04 PINN's physical safety.**
- `exp_E_violation_aware_scoring.py` -- inverts Exp 04: instead of
  *minimising* curvature violations, *uses* them as a second-channel
  ranking signal. Linear blend with predicted prob; sweep alpha.
  **Novelty: violations have always been a diagnostic, never a score.**
- `exp_F_leave_one_generator_out.py` -- 30-fold LOGO-CV on sdc-travel
  (66 generator campaigns). Replaces Exp 11's *synthetic* k-means
  environments with **real** generator distributions. Spearman
  correlations of APFD with campaign size/FAIL-rate/curvature.
  **Novelty: real-OOD probe on Travel never published.**
- `exp_G_prefix_weighted_listwise.py` -- new listwise loss: prefix-
  weighted Plackett-Luce (weight comparisons by APFD's `1 - r/n` or
  `1/log(1+r)` prefix kernel). Compares BCE-focal / vanilla PL /
  linear-prefix PL / NDCG-prefix PL on the same split. **Novelty:
  APFD-specific weighting of PL has not been proposed.**
- `exp_H_severity_conditional_ddpm.py` -- severity-conditional 1D
  DDPM on curvature trajectories. Generates synthetic OOB-0-1-like
  roads to close the 0-5 -> 0-1 transfer asymmetry observed in
  `exps/oob/tracker.md`. **Novelty: continuous-severity-conditioned
  generative augmentation for SDC prioritization.**
- `exp_I_universal_temperature.py` -- one universal model on the
  union train pool, per-bench temperature + bias as a cheap
  *adaptation* layer. Compared to per-bench retraining (oracle).
  **Novelty: positions temperature scaling as deployment-time
  adaptation for SDC, not calibration.**
- `exp_J_trustworthiness_scorecard.py` -- single scalar TWS that
  fuses rotation-Delta, resolution-Delta, and curvature-violation
  rate (the three audit axes already in the tracker). With baseline
  references, weight w = (1/3, 1/3, 1/3). **Novelty: unified
  trustworthiness metric for SDC ranker auditing.**

### Insight -> Exp mapping (which tracker pattern motivates which exp)

| Tracker insight (`exps/tracker.md`)                           | Exp |
|---------------------------------------------------------------|-----|
| AUC up APFD down across Exp 01/02/03/04/10                    |  B  |
| Exp 01 FNO resolution invariance Delta = 0.0012               |  C  |
| Exp 02 SE(2) rotation invariance Delta = 0.0000               |  J  |
| Exp 04 violation rate 17.57% -> 3.14% (5.6x reduction)        | E, J |
| Exp 05/12 conformal v1 vacuous / v2 invalid                   |  D  |
| Exp 11 IRM fails on synthetic k-means environments            |  F  |
| Exp 03 PL fails to translate AUC into APFD                    |  G  |
| Exp 08 DDPM boundary hit-rate 8.8% (under-targeted)           |  H  |
| OOB transfer asymmetry 0-5 -> 0-1 drops 0.08 APFD             |  H  |
| One-recipe-many-benches premise                               |  I  |
| OOB-0-3 universal-source hypothesis                           |  A  |

### Story arc for the oral

1. **Recipe slide**: "One Transformer, every bench" (Exp_best_all).
2. **Theoretical slide**: APFD = (1-p) AUC + p/2 (Exp B) -- closes
   the AUC-vs-APFD debate.
3. **Empirical slide**: cross-bench transfer matrix (Exp A) -- the
   "one recipe" claim. Followed by real-OOD (Exp F) on Travel.
4. **Auditability slide**: TWS scorecard (Exp J) per model, plus
   conformal abstention curves (Exp D). The trustworthiness pitch.
5. **Operational slide**: severity-conditional DDPM (Exp H) and
   prefix-weighted PL (Exp G) -- two cheap operational levers.
6. **Adaptation slide**: universal + temperature (Exp I) -- deploy
   one model, adapt with a scalar.

## Recipe per family

| Family       | Recipe                                                | Eval                        |
|--------------|-------------------------------------------------------|-----------------------------|
| Geometry     | Transformer (10ch, d=128, 4L) + SWA + Focal gamma=2.5 | 30-trial APFD on held-out   |
| Tabular (RP) | LightGBM (`num_leaves=63, lr=0.05`), fallback HGB     | 5-fold stratified CV APFD   |

Hyperparameters are intentionally identical to the SensoDat winner
(`tracker.md` in `exps/` -- best-single 0.8066, ensemble 0.8077). No
per-benchmark search; the point is to show the recipe ports cleanly.

## Benchmarks targeted by `exp_best_all.py`

1. **SDC-Scissor** (`sample_tests`, 5-fold CV)            -- cheap, runs first
2. **SDC-Pririotizer-RP** (4 tabular datasets, 5-fold CV) -- LightGBM
3. **OOB-Regression** within-threshold {0-1, 0-3, 0-5}    -- per-tag 80/20
4. **SensoDat** pooled (8 corpora)                        -- 80/20
5. **sdc-travel competition** (66 campaigns pooled)       -- 80/20

The runner is **resilient**: a missing data folder prints `[SKIP]` and
continues. After every benchmark finishes, `best_all_results.json` is
overwritten so partial runs are recoverable.

## Latest numbers (TODO: fill after a full run)

### Geometry benches (Transformer + SWA + Focal gamma=2.5)

| Bench       |  N  | FAIL% | AUC | APFD (30 trials) |
|-------------|-----|-------|-----|------------------|
| SensoDat    | --  | --    | --  | --               |
| OOB-0-1     | --  | --    | --  | --               |
| OOB-0-3     | --  | --    | --  | --               |
| OOB-0-5     | --  | --    | --  | --               |
| Scissor (5-fold) | -- | -- | --  | --               |
| Travel      | --  | --    | --  | --               |

### Tabular (LightGBM, 5-fold CV) -- from prior Kaggle run

| Dataset       |  N   | FAIL% | AUC    | APFD             |
|---------------|------|-------|--------|------------------|
| BeamNG_RF_1   | 1178 | 26.5% | 0.9669 | 0.8433 +/- 0.0109 |
| BeamNG_RF_1_5 | 5638 | 45.1% | 0.9783 | 0.7625 +/- 0.0015 |
| BeamNG_RF_2   | 1729 | 95.7% | 0.9893 | 0.5209 +/- 0.0006 |
| DriverAI      | 5630 | 18.6% | 0.9733 | 0.8855 +/- 0.0061 |

### Reading the table for the oral

- BeamNG_RF_2 is **95.7% FAIL** -- ceiling of APFD is ~0.52 even with a
  perfect ranker (because virtually every test fails near the front).
  Don't compare its APFD to the others; it's an APFD-degeneracy probe.
- DriverAI / RF_1 / Scissor sit in the **healthy 20-60% FAIL regime** --
  these are the "honest" APFD numbers for ranker quality.
- High-FAIL outliers (Travel ~5.5%, RF_2 ~95%) are useful to **anchor the
  imbalance figure** for the oral: "APFD vs FAIL%" curve has a known
  inverted-U shape; we want the recipe to track that shape consistently.

## Story for the oral

1. **Slide N**: "Same model, no per-bench tuning." Show the row of APFD
   numbers across 8+ benchmarks. The point is *consistency*, not
   per-bench wins.
2. **Slide N+1**: APFD vs FAIL% scatter (this folder's numbers fill the
   x-axis). Add the prior SDC-Prioritizer baselines from Birchler et al.
   as a reference cloud.
3. **Slide N+2 (theory hook)**: link back to SensoDat folder's
   invariance / monotonicity / listwise figures. "These benchmarks also
   inherit the rotation-Delta = 0 property because nothing in the recipe
   changes."

## Action items

- [ ] Run `exp_best_all.py` end-to-end on Kaggle T4; record wall-clock per
      bench (target < 90 min total).
- [ ] Add ensemble-of-5-gammas variant as `--mode ensemble` (current
      script is single gamma=2.5). Slightly more compute, ~+0.001 APFD.
- [ ] Add rotation-invariance probe per benchmark: 6 rotations, report
      Delta-APFD. Per-bench evidence that the model is bit-identical.
- [ ] Cross-bench transfer matrix (k x k): train on bench i, eval on
      bench j. Even more elastic than OOB transfer (which is severity-
      only). This is the centrepiece of an ICSE 2027 "universal recipe"
      pitch.

## Update log

- 2026-05-14 -- `exp_best_all.py` written, `exp_rp_external_bench.py`
  moved here from `exps/`. Folder split out of `exps/`. SEARCH_ROOTS
  patched for `exps/best_all/` depth.
