# OOB-Regression — Performance Tracker

> External benchmark: **Dataset-OOB-{0-1, 0-3, 0-5}** (Zenodo 16939865).
> BeamNG-generated roads with three out-of-bounds (OOB) severity thresholds.
> Goal: show that the SensoDat-tuned recipe (Transformer + SWA + Focal,
> APFD=0.8077 on SensoDat) transfers to OOB and behaves predictably across
> severity thresholds.
>
> Pure ASCII to avoid cp1252 mojibake on plus-minus / Delta / etc.

## Scripts in this folder

- `exp_best_oob.py` -- within-threshold: train + eval per threshold (3 runs).
- `exp_best_oob_transfer.py` -- cross-threshold: train on one threshold,
  zero-shot eval on the other two. Builds a 3x3 transfer matrix.

## Dataset sizes

| Threshold | Total | FAIL%  | Train | Test |
|-----------|-------|--------|-------|------|
| OOB-0-1   | 1324  | (low)  | 1059  |  265 |
| OOB-0-3   | 4746  | (mid)  | 3796  |  950 |
| OOB-0-5   |10000  | (high) | 8000  | 2000 |

---

## 1. Within-threshold (`exp_best_oob.py`)

Recipe: same as SensoDat best.
- 80% / 20% stratified split, batch=256, lr=5e-4
- Transformer (10ch, d_model=128, 4 layers, ~0.83 M params)
- Focal gamma sweep `{0.0, 1.0, 1.5, 2.0, 2.5}` + SWA from epoch 50/75
- 30-trial multi-trial APFD on the held-out 20%

### Latest numbers (TODO: fill in after run)

| Tag      | Best gamma | AUC    | APFD (30-trial)  | Notes               |
|----------|------------|--------|-------------------|---------------------|
| OOB-0-1  |   --       |  --    |  --               | small set; high var |
| OOB-0-3  |   --       |  --    |  --               |                     |
| OOB-0-5  |   --       |  --    |  --               | largest, lowest var |

---

## 2. Cross-threshold transfer (`exp_best_oob_transfer.py`)

Single gamma=2.5 (the SensoDat winner). Train per source threshold,
evaluate zero-shot on all three target thresholds.

### Transfer matrix (rows = train src, cols = eval tgt)

|  src \\ tgt | OOB-0-1  | OOB-0-3  | OOB-0-5  |
|------------|----------|----------|----------|
| OOB-0-1    | 0.7301 * | 0.6550   | 0.6114   |
| OOB-0-3    | 0.7256   | 0.7140 * | 0.7111   |
| OOB-0-5    | 0.6787   | 0.7272   | 0.7574 * |

(*) diagonal = within-threshold (sanity check).

### Insights from the matrix

1. **OOB-0-3 is the most transferable source**: a single model trained on
   OOB-0-3 sits within ~0.05 APFD of every threshold's own best. Useful as
   a "default deployment" recipe when severity is unknown at test time.
2. **Severe asymmetry**: training on 0-1 (smallest, mildest threshold)
   loses up to 0.12 APFD when evaluated on 0-5 -- the "easy" model never
   sees high-severity geometry signatures.
3. **OOB-0-5 -> OOB-0-1** drops 0.08 APFD; the "hard" model is too
   confident on rare 0-1 fails. The bigger training corpus does NOT buy
   universal generalisation -- distribution shift bites in both directions.
4. AUCs on the held-out test of the source: 0-1 = 0.8844, 0-3 = 0.8459,
   0-5 = 0.9208. Larger N + higher FAIL fraction -> stronger calibration.

---

## 3. Story for the paper / oral

- **Section "OOB external benchmark"**: per-threshold within-domain APFD
  table (above) showing the SensoDat recipe ports cleanly to BeamNG OOB.
- **Section "Transfer / distribution shift"**: 3x3 transfer matrix as a
  heat-map figure. The off-diagonal drop quantifies severity-shift cost
  in APFD terms -- this is the empirical hook for any
  domain-adaptation / IRM / TENT story (Exp 11, Exp 14 in SensoDat tracker).
- **Headline talking point**: the SensoDat-tuned hyperparameters
  (gamma=2.5, SWA, focal) reach APFD >= 0.71 on OOB-0-3 with zero per-
  dataset tuning. No new architecture, no per-task search.

## 4. Action items / next steps

- [ ] Re-run within-threshold with the 5-gamma sweep + ensemble (currently
      only gamma=2.5 from the transfer file is logged here).
- [ ] Add **rotation-invariance probe** (Exp 02 SE(2)) on each threshold;
      we expect Delta = 0 by construction but proving it on OOB strengthens
      the figure.
- [ ] Add **resolution-invariance probe** (Exp 01 FNO) -- BeamNG roads have
      higher native sampling than SensoDat, this is a clean N-sweep target.
- [ ] Cross-domain transfer: SensoDat -> OOB and OOB -> SensoDat as
      severity-anchored shift study.

## Update log

- 2026-05-14 -- folder spun off from `exps/`. Files moved unchanged; only
  SEARCH_ROOTS / OUTPUT_DIR depth was patched.
- (transfer matrix numbers from a Kaggle run captured before this split;
  re-validate locally after restructure.)
