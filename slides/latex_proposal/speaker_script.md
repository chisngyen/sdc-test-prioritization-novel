# Speaker Script — Cost-Effective Test Prioritization for Simulation-Based SDC Regression Testing

**Proposal Deck | April 2026**

---

## Slide 1: Title Slide

*(Duration: ~30 seconds)*

> Good morning/afternoon everyone. Thank you for joining.
>
> Today I'm presenting our research proposal on **cost-effective test prioritization for self-driving car simulation testing** — a collaboration between Chi-Nguyen Tran, Dao Sy Duy Minh, and Huynh Trung Kiet from the University of Science, Ho Chi Minh City.
>
> This work builds directly on the inaugural ICST 2025 Self-Driving Car Testing Track, and we're proposing a new method called **GeoRiskRank** to tackle the 2026 competition challenge.

---

## Slide 2: Problem Context

*(Duration: ~90 seconds)*

**Key message to deliver:** Self-driving car testing via simulation (BeamNG) is computationally expensive. Each test case is a sequence of road points. We want failures to surface as early as possible in the evaluation order.

**Speak slowly here.** Emphasize:
- *"expensive and long-running"* — this is the core pain point.
- *"order tests so failures appear as early as possible"* — this is the prioritization task, distinct from test selection.
- *"APFDR/APFD"* — these are the official competition metrics. If your audience is unfamiliar, briefly define them: Average Percentage of Faults Detected per Rank measures how quickly failures rank near the top of the test order.

**Talking points:**
- Modern SDC systems are validated using simulation environments like BeamNG. These simulations are realistic but computationally heavy.
- In the 2026 competition, tests are represented as **road-point sequences** — ordered lists of coordinates and geometry features.
- Our job is not to *select* which tests to run (that's test selection). Our job is to **prioritize** — to order all tests so that the failing ones appear as early as possible.
- The target metric is **APFDR/APFD** under realistic BeamNG constraints.

**Anticipated Q — "What's the difference between test selection and test prioritization?"**
> Selection = which tests to run. Prioritization = in what order to run them. We do prioritization.

**Transition:** *"Let me show you why this matters in practice."*

---

## Slide 3: Motivation

*(Duration: ~60 seconds)*

**Left column — Why this matters:**

- **Limited test execution budget per CI cycle** — In practice, you may only afford to run 30–40% of your full test suite within a CI window. When time runs out, you want the most informative tests already done.
- **Safety-critical bugs must be surfaced quickly** — SDC failures can be catastrophic. A one-hour delay in bug discovery is not just inconvenient; in a safety-critical system, it could cost lives.
- **Regression growth makes random order ineffective** — As test suites scale, random ordering leaves too much to chance. Failing tests can end up at the very bottom, wasting the entire CI budget.

**Right column — Goal:**

Pause here and read the large text aloud. This is your thesis in one sentence:

> *"Maximize early fault detection while keeping runtime and integration cost low."*

Repeat it verbatim so the audience remembers it.

**Pacing tip:** This is a 30–60 second slide. Don't rush it, but don't linger either — the audience is still building context.

**Transition:** *"To understand where we stand, let's look at the 2025 competition that inspired this work."*

---

## Slide 4: ICST 2025 Competition Snapshot

*(Duration: ~60 seconds)*

**Context slide — establish the competition baseline and give proper credit to prior work.**

**Key facts to verbalize:**

- *"First ICST SDC tool competition"* — This is a brand-new track. We're building on a real venue with a real problem.
- *"5 submitted tools"* — Modest uptake so far, but there's significant room to grow and improve.
- *"32,580 test cases"* — This is non-trivial scale. Emphasize the challenge this presents.
- *"gRPC + Docker"* — Standardized evaluation means our results are reproducible and directly comparable across teams.

**Read the citation block aloud if time permits:**
> C. Birchler et al., "ICST Tool Competition 2025 -- Self-Driving Car Testing Track", arXiv:2502.09982 (Feb 2025).

This gives credibility to our framing and shows we are building on established competition infrastructure.

**Transition cue:** *"With this foundation, let's look at what actually worked in 2025."*

---

## Slide 5: Key Findings from ICST 2025

*(Duration: ~90 seconds)*

**This is a data slide — let the numbers speak, then explain their meaning.**

**Numbers to highlight verbally:**

- **"156.23 vs 65.39"** — The best tool (ITS4SDC) found faults roughly **2.4 times faster** than the random baseline. Time-to-fault ratio: lower is better.
- **"0.38 vs 0.80"** — ITS4SDC selected failing tests at **twice the rate** of random. Fault/selection ratio: higher is better.
- Both are competition-reported metrics. They validate that learning-based approaches genuinely work in this domain.

**Three key messages from the bullets (speak them, don't just display):**

1. **Learned selectors dramatically outperform random** — Our assumption that data-driven approaches can help is validated by real competition data.
2. **Tool behavior varies in initialization and selection latency** — Runtime performance matters for CI integration. A theoretically perfect tool that takes 10 minutes to rank tests is impractical.
3. **Curvature diversity alone was NOT clearly discriminative** — We need a richer signal than simple geometric features to tell failing tests apart.

**Transition cue:** *"These findings directly inform our research hypothesis for 2026."*

---

## Slide 6: Implications for Our 2026 Proposal

*(Duration: ~45 seconds)*

**Key messages:**

- ICST 2025 validates that **data-driven selection works**. In 2026, we extend this to **test prioritization** — a natural next step.
- We should optimize not only fault yield, but also **ranking stability** and **runtime**.
- Our proposed **GeoRiskRank** method aligns with all three lessons from 2025:
  - Strong geometry representation → captures richer signals than curvature alone.
  - Calibrated confidence → addresses tool behavior variability.
  - Lightweight reranking under budget → keeps runtime practical for CI.

**Positioning statement (read aloud):**

> From "which tests to run" (selection) to "which order to run first" (prioritization) with measurable APFD gains.

**Transition:** *"Before we formalize the hypothesis, let me show you where we stand today."*

---

## Slide 7: Current Baseline and Gap

*(Duration: ~60 seconds)*

**This slide establishes the starting point.**

- Existing baselines already use geometry-derived features and Transformer scoring — so we're not starting from zero.
- Our preliminary experiments in this repository show **strong potential but unstable gains** across variants.

**Table walkthrough:**

| Variant | APFD | Observation |
|---|---|---|
| Base Transformer | 0.7899 ± 0.0140 | Solid baseline |
| + SWA | 0.8042 ± 0.0120 | Better generalization |
| + Focal + SWA (best single) | 0.8066 ± 0.0124 | Best single setup |
| SWA Ensemble | 0.8077 ± 0.0115 | Highest reported |

**Key takeaway:** The SWA (Stochastic Weight Averaging) technique is our most consistent single improvement. Ensembling helps further, but at added complexity.

**Transition:** *"To understand which components actually matter, let's look at the full ablation picture."*

---

## Slide 8: Full Ablation from exps/01..07

*(Duration: ~60 seconds)*

**This is a detailed ablation table — use it to tell a story of discovery.**

Walk through the table top to bottom:

- **Base (0.7899)** — Our starting point: a Transformer with geometry-derived features.
- **01–03** (MultiScaleStem, DropPath, TriplePool): All slightly *worse* than base. Interesting — more complex representations hurt here.
- **04** (Mixup): Marginal improvement to 0.7733. Data augmentation helps slightly.
- **05** (FocalLoss): Improved to 0.7820. Addressing class imbalance helps.
- **06** (TTA): Dropped to 0.7700. Test-time augmentation was not effective.
- **07** (SWA): Jumped to **0.8042** — the only single-change variant that clearly improves over base.

**Key takeaway (speak this line):**

> SWA is the only single-change variant that clearly improves over the base Transformer. Every other modification either hurt or marginally helped.

**Transition:** *"Now let's look at what happened when we combined these components."*

---

## Slide 9: Focal + SWA Sweep from best.md

*(Duration: ~60 seconds)*

**Fine-grained hyperparameter tuning story.**

- We took the SWA baseline (0.8042) and added Focal Loss with different gamma values.
- The sweep shows that **gamma = 2.5** with SWA gave our best single-model result: **0.8066 ± 0.0124**.
- The **ensemble of 5 SWA models over 50 trials** reached the highest: **0.8077 ± 0.0115** — better mean and tighter standard deviation.

**Decision to communicate:**

> Our proposal will use the SWA baseline as the core model, with Focal+SWA and ensemble variants as high-performance options.

**Transition:** *"With this empirical foundation, let me state our formal research hypothesis."*

---

## Slide 10: Research Hypothesis

*(Duration: ~45 seconds)*

**Read the hypothesis block verbatim:**

> A **hybrid ranking model** that combines geometric sequence learning, uncertainty-aware calibration, and lightweight diversity control will improve APFD/APFDR under fixed evaluation budgets.

**Break it down into three testable sub-hypotheses:**

- **H1 — Representation:** Better representation of road geometry increases fail-probability ranking quality. More channels, better dynamics modeling.
- **H2 — Calibration:** Post-hoc calibration reduces over-confident false positives. We don't want to rank a test as highly likely to fail and be wrong every time.
- **H3 — Diversity control:** Diversity-aware re-ranking avoids redundant similar tests occupying top positions. If two tests are nearly identical and one fails, we want the other near the top too — not clustered together at the bottom.

**Transition:** *"Let me show you how GeoRiskRank operationalizes this hypothesis."*

---

## Slide 11: Proposed Method: GeoRiskRank

*(Duration: ~90 seconds)*

**This is the core of the talk. Walk through each component deliberately.**

**Component 1 — Feature Encoder:**

> We encode each road-point sequence into multiple geometry channels: **curvature**, **angle change**, **local variation**, and **cumulative distance**. These channels capture different aspects of road shape that correlate with failure risk.

**Component 2 — Sequence Scorer:**

> A **Transformer-based model** takes these multi-channel sequences and predicts a **fail probability** for each test case. The Transformer architecture is well-suited for modeling dependencies across the sequence of road points.

**Component 3 — Calibration Layer:**

> Raw probability outputs from neural networks are often over-confident. We apply **post-hoc calibration** to produce well-calibrated confidence scores — so when the model says "90% likely to fail," it is right about 90% of the time.

**Component 4 — Diversity Re-ranker:**

> After scoring, we apply a **diversity-aware reranker** that penalizes near-duplicate trajectories occupying top positions. If two tests have nearly identical road geometry, we don't want them both ranked in the top 5.

**Output block (read aloud):**

> The final output is a **prioritized list of test IDs**, sorted by descending risk score, ready for the evaluator RPC response.

**Transition:** *"Now let's look at how this fits into the competition infrastructure."*

---

## Slide 12: System Integration (Competition-Compatible)

*(Duration: ~45 seconds)*

**Key reassurance for the audience:**

> We made a deliberate choice to **keep the existing protocol unchanged**: the same `interface_2026.proto`, the same gRPC contract, and the same Docker-first evaluation flow.

**What this means in practice:**

- Our GeoRiskRank model slots in as the ranking component.
- The rest of the pipeline — data ingestion, RPC interface, evaluator — remains exactly as the competition specifies.
- **No protocol change required.** This makes adoption straightforward and our results directly comparable to other tools.

**Transition:** *"Now let's look at how we plan to validate this method."*

---

## Slide 13: Experimental Plan

*(Duration: ~45 seconds)*

**Datasets and splits:**

> We'll use the **provided training and test datasets** with competition-style repeated sub-sampling protocol — this ensures our results are comparable to the 2025 baseline and reproducible by other teams.

**Evaluation — primary metric:**

> **APFD and APFDR** — the official competition metrics. These measure how early in the ranking we detect faults.

**Evaluation — secondary metrics:**

- **Inference latency** — critical for CI integration. We need to rank all tests within the available budget window.
- **Memory footprint** — important for containerized deployment.
- **Robustness across random trials** — we report mean, standard deviation, and run paired significance tests to ensure improvements are not due to chance.

**Transition:** *"Let me break down exactly what we'll ablate."*

---

## Slide 14: Ablation Matrix

*(Duration: ~45 seconds)*

**Four ablation axes — each tests one aspect of the hypothesis:**

| Axis | What we test |
|---|---|
| **Representation** | 6-channel vs 10-channel features; with/without curvature dynamics |
| **Training strategy** | BCE baseline vs focal loss vs SWA vs focal+SWA |
| **Inference strategy** | Single model vs calibrated model vs calibrated+diversity rerank |
| **Ranking policy** | Pure probability ranking vs hybrid score with diversity penalty |

**Purpose of ablation:**

- Isolate the contribution of each component.
- Identify which elements are essential vs. optional.
- Provide a clear recommendation for the final submission configuration.

**Transition:** *"Of course, every project has risks. Let me address ours honestly."*

---

## Slide 15: Risk and Mitigation

*(Duration: ~45 seconds)*

**Be transparent about risks — this builds credibility.**

**Risk 1 — Overfitting to one benchmark distribution:**

> If our model learns benchmark-specific quirks rather than generalizable geometry signals, it won't transfer to unseen test suites.
>
> **Mitigation:** Strong cross-trial validation, held-out test set checks, and comparison against multiple data generators (AmbieGen, Frenetic, FreneticV).

**Risk 2 — Improvements not significant across repeated trials:**

> With stochastic training and evaluation, small APFD differences might not be statistically significant.
>
> **Mitigation:** Simplicity-first fallback. If ablations don't show significance, we fall back to the best stable baseline (SWA alone).

**Risk 3 — Runtime overhead from reranking:**

> Diversity-aware reranking could add unacceptable latency in CI pipelines.
>
> **Mitigation:** O(N log N) ranking with bounded pairwise checks. We set a hard budget cap on reranking time.

**Transition:** *"Let me show you our roadmap to get there."*

---

## Slide 16: Timeline (6 Weeks)

*(Duration: ~30 seconds)*

**Week-by-week overview — read the table concisely:**

| Week | Deliverable |
|---|---|
| **1** | Reproduce baseline and verify evaluator pipeline |
| **2** | Implement calibrated Transformer scorer |
| **3** | Add diversity-aware reranker and optimize runtime |
| **4** | Run ablations and robustness experiments |
| **5** | Final model lock, Docker packaging, reproducibility check |
| **6** | Paper/report figures, slide finalization, dry-run presentation |

**Highlight:** The first two weeks are foundation. Week 4 is where we test our hypotheses. Weeks 5–6 are for polish and submission.

**Transition:** *"Finally, let me summarize what we expect to contribute."*

---

## Slide 17: Expected Contributions

*(Duration: ~30 seconds)*

**Three concrete contributions:**

1. **A practical, reproducible prioritization method** for SDC simulation testing — grounded in real competition infrastructure, not toy problems.
2. **A competition-ready tool** with **no interface changes required** — it slots into the existing pipeline.
3. **Empirical evidence on which improvements are stable** under repeated evaluation — we won't just report one cherry-picked number.

**Success Criteria (read aloud verbatim):**

> Consistent APFD/APFDR gain over the strong Transformer baseline with acceptable inference cost.

**Closing line:**

> We believe GeoRiskRank represents a principled next step from ICST 2025's selection-based approaches to a full prioritization framework.

---

## Slide 18: Thank You

*(Duration: ~30 seconds + Q&A)*

> Thank you all for your attention.
>
> I'm happy to take questions on any aspect of the proposal — the hypothesis, the method, the experiments, or the timeline.
>
> *(Pause for questions.)*

---

## Appendix: Quick Reference

**Glossary of key terms:**

| Term | Definition |
|---|---|
| **APFD** | Average Percentage of Faults Detected per Rank — primary metric; higher is better |
| **APFDR** | Average Percentage of Faults Detected Rate — variant metric accounting for detection rate |
| **BeamNG** | Realistic open-source vehicle simulation platform used for SDC testing |
| **GeoRiskRank** | Our proposed hybrid ranking method combining geometry encoding, Transformer scoring, calibration, and diversity reranking |
| **SWA** | Stochastic Weight Averaging — training strategy that improves generalization and calibration |
| **Focal Loss** | Loss function that down-weights easy examples to focus on hard cases |
| **gRPC** | Google Remote Procedure Call — used for evaluator communication in the competition |
| **ICST** | International Conference on Software Testing — hosts the SDC testing track |

**Citation for competition reference:**

> C. Birchler et al., "ICST Tool Competition 2025 -- Self-Driving Car Testing Track", arXiv:2502.09982 (Feb 2025).
