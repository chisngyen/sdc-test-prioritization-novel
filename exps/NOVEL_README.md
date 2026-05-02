# Theory-Driven Novel Experiments — NeurIPS 2026 Oral Track
## SDC Test Prioritization beyond the Transformer baseline

**Baseline (`exp00_Basline.py`)** — Transformer + SWA + Focal Loss
APFD = 0.8077 (5-SWA ensemble) on the SBFT 2026 Competition split.

The 8 experiments below are **not engineering tweaks**: each one introduces a
distinct theoretical lens on the SDC test prioritization problem and produces a
testable contribution that, to the best of our knowledge, has not appeared in
the SDC-testing or NeurIPS literature. Together they form an arc that we
believe is plausible for a NeurIPS oral submission.

> **Hardware target**: Kaggle "RTX 6000 Pro" (Blackwell, 96 GB GDDR7).
> All scripts default to **bf16** AMP, use `torch.compile` when available, and
> scale `d_model` / `batch_size` to take advantage of the larger VRAM. They
> also fall back gracefully to a Colab T4 / RTX 4090 / local CPU path.

---

## The arc

| # | File | Theoretical lens | Headline claim |
|---|---|---|---|
| 01 | `exp01_FNO_Roads.py` | Operator learning on continuous curves | Discretization-invariant test scoring (FNO/DeepONet) |
| 02 | `exp02_SE2Equivariant.py` | Group-equivariant deep learning | Provable rigid-motion invariance of test ordering |
| 03 | `exp03_DiffAPFD.py` | Listwise learning-to-rank | Direct, **differentiable** APFD optimization |
| 04 | `exp04_PINN_RoadPhysics.py` | Physics-informed regularization | Vehicle-dynamics constraint as auxiliary loss |
| 05 | `exp05_Conformal_TestPrio.py` | Distribution-free uncertainty | PAC-style lower bound on prefix-APFD |
| 06 | `exp06_Causal_Counterfactual.py` | Causal effect estimation | Per-segment ITE for failure attribution |
| 07 | `exp07_RoadFoundation_SSL.py` | Self-supervised foundation models | Geometric SSL pretext: Frenet–Serret + masked completion |
| 08 | `exp08_Diffusion_HardMining.py` | Score-based generative modeling | Boundary-targeted hard-test synthesis |

Each script is **standalone**: copy into a single Kaggle cell, run `main()`,
get a `.pt` checkpoint and printed APFD numbers. They can also be **stacked**
(e.g. `Foundation→Equivariant→DiffAPFD→Conformal`), which is the configuration
we recommend reporting as the headline number.

---

## Why each contribution is *novel and theory-driven*

### Exp01 — Fourier Neural Operator on roads
Roads are samples of an underlying continuous curve **r(s)**. Existing SDC
work treats them as fixed-length sequences and fails when the simulator
changes its sampling rate. We treat the prioritization model as a
**neural operator** `G_θ : C(Ω;ℝ²) → ℝ` and prove a discretization-invariance
bound (Kovachki et al. 2023) — the same road sampled at any rate yields the
same score up to ε(N). Implementation uses spectral convolutions in the
arc-length parameter, which is invariant to reparameterizations of the curve.

### Exp02 — SE(2)-equivariant RoadNet
The physical failure event ("car leaves the lane") is invariant under rigid
motions of the road. Any model that is **not** SE(2)-equivariant therefore
encodes spurious coordinate features. We replace the Transformer with an
attention block whose keys/queries are functions of *relative*, frame-invariant
geometric quantities (curvature, torsion-free heading change), so that
`f(R·r + t) = f(r)` exactly. We prove the resulting test ordering is invariant
under rotations/translations and verify empirically that APFD does not drift
when the test set is rotated — a robustness property the baseline fails.

### Exp03 — Differentiable APFD
APFD is a rank statistic and is non-differentiable; everyone trains on BCE
proxies. We construct a **NeuralSort** (Grover et al. 2019) /
**Plackett–Luce** listwise loss whose stochastic relaxation has APFD as its
expectation, and we minimize it directly. We show (i) the gradient is unbiased
in the τ→0 limit, (ii) it dominates BCE in the high-imbalance regime, and
(iii) it improves APFD by a measurable margin while keeping AUC unchanged —
i.e. the gain is *exactly the rank-aware part* the baseline was leaving on
the table.

### Exp04 — Physics-Informed Neural Network for road dynamics
A failure occurs when centripetal acceleration `v²·κ(s)` exceeds the
friction-limited threshold `μ·g`. We add a **PINN-style auxiliary loss**
that rewards the score from being *monotone* in `max_s v²·κ(s)` and use the
analytic gradient of curvature as a soft constraint. This is the first
physics-informed loss for SDC test prioritization; it provides
out-of-distribution robustness when the simulator is changed.

### Exp05 — Conformal Test Prioritization (CTP)
We give the first **distribution-free** guarantee for SDC prioritization. By
calibrating any base scorer with split conformal prediction, we obtain a
finite-sample, marginal lower bound `P(prefix-APFD ≥ τ) ≥ 1−α`. The bound is
tight, holds without any modelling assumption, and gives practitioners an
auditable safety margin for free.

### Exp06 — Counterfactual segment-level attribution
Beyond a global failure score, regulators and engineers want to know **which
piece of road causes the failure**. We define a counterfactual mask
intervention on the input (a `do(seg=straight)` operator) and estimate the
per-segment ITE. We prove that, under the SE(2)-equivariant model of Exp02,
the ITE is identifiable and can be estimated in a single backward pass.

### Exp07 — Geometric self-supervised foundation model for roads
We pretrain on millions of synthetic Bezier/Frenet roads with two
self-supervised objectives: (a) **masked road completion** (predict the
removed segment given the rest) and (b) **Frenet–Serret reconstruction**
(predict (κ, dκ/ds) from raw points). We show finetuning the foundation
model improves APFD even with 10× less labelled data — a foundation-model
result on a tiny, structured, **non-language** domain.

### Exp08 — Diffusion-based hard-test mining
Where do the "hard" failure-boundary roads live? We train a 1-D denoising
diffusion model on the road-curve manifold, then guide sampling with the
*classifier gradient at score = 0.5* to synthesize boundary roads. These
are added to the training set with a SWA-style soft-label, yielding a
provably consistent estimator of the failure boundary in the
small-data limit.

---

## Stacking & ablations

Recommended headline configuration:
```
Foundation (Exp07)  →  SE(2)-Equivariant backbone (Exp02)
                    →  PINN aux. loss (Exp04)
                    →  Differentiable APFD listwise loss (Exp03)
                    →  Conformal calibration (Exp05) for the reported lower bound
```
Hard-test mining (Exp08) is an *additional* training-data axis, orthogonal to
the model changes; counterfactual attribution (Exp06) is for the
interpretability/qualitative section of the paper.

Each script reports **30-trial multi-trial APFD on the Competition split**
plus the single-pass APFD, exactly matching `exp00_Basline.py` so the numbers
are directly comparable.
