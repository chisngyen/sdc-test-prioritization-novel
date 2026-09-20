# SDC-Travel — Performance Tracker

> External benchmark: **sdc-travel competition** -- 66 generator
> campaigns, 24,968 raw tests, ~14,168 valid, FAIL ~ 5.5% among valid.
> Largest external benchmark by volume; very imbalanced.
> Goal: stress-test SensoDat-tuned recipe under heavy class imbalance and
> aggregate across many generators.

## Scripts

- `exp_best_travel.py` -- pool all valid tests across campaigns, stratified
  80/20 by outcome (`random_state=42`), full gamma sweep + SWA + 5-config
  ensemble. 30-trial multi-trial APFD on the test split.

## Why this benchmark matters

- **Severe imbalance** (~5.5% FAIL): a near-worst-case for focal weighting.
  Tells us whether `pos_weight = n_neg / n_pos` blows up gradients at this
  scale.
- **Multi-generator**: 66 distinct test-generator campaigns. A model that
  overfits to one generator family will be exposed by the pooled test set.
- **Comparable scale to SensoDat** (14k valid vs SensoDat 14k+) but very
  different prior, so SensoDat -> Travel is a fair "same-N, different-shift"
  cross-domain probe.

## Latest numbers (TODO: fill in after run)

| Config              | AUC    | APFD (30-trial) | sigma | Notes |
|---------------------|--------|-----------------|-------|-------|
| gamma=0.0 best-ckpt |  --    |  --             |  --   | no focal -- baseline |
| gamma=0.0 + SWA     |  --    |  --             |  --   |       |
| gamma=1.5 + SWA     |  --    |  --             |  --   |       |
| gamma=2.5 + SWA     |  --    |  --             |  --   | SensoDat winner -- compare |
| Ensemble 5 SWA      |  --    |  --             |  --   |       |

## Story for the paper / oral

- **Imbalance scaling figure**: APFD vs FAIL-rate across {SensoDat 33%,
  Scissor 58%, OOB-0-1 low%, OOB-0-3, OOB-0-5, Travel 5.5%, RP datasets
  (5-95%)}. Travel pins the low-FAIL-rate corner.
- **Per-generator decomposition**: keep `campaign` in each test's id so we
  can recompute APFD per-campaign. If one campaign collapses APFD, that's
  evidence for the OOD / IRM section (Exp 11) on real generator shift,
  not synthetic k-means environments.
- **Recipe stability claim**: same Transformer + SWA + Focal gamma=2.5
  hits APFD within X of SensoDat best, without per-bench tuning. Repeated
  across OOB / Scissor / Travel / RP, this is the headline ICSE 2027
  cross-bench result.

## Action items

- [ ] Per-campaign APFD breakdown (groupby `campaign`, compute mean +
      sigma across campaigns -- not pooled).
- [ ] **Confirm interpolated_points vs road_points choice** is right
      (loader currently prefers `interpolated_points`); ablate.
- [ ] Test `pos_weight` clamp (`pw <= 50`) -- with FAIL ~ 5.5%,
      raw pos_weight ~ 17 which is already large; check gradient
      diagnostics if AUC collapses early.
- [ ] Generator-leave-one-out CV: train on 65 campaigns, test on the
      held-out one. Real OOD on real shift -- the strongest version of
      Exp 11 with non-synthetic environments.

## Update log

- 2026-05-14 -- folder spun off from `exps/`. File moved unchanged; only
  SEARCH_ROOTS / OUTPUT_DIR depth was patched.
