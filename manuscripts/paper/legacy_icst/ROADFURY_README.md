# RoadFury — ICST 2026 Tool Competition entry

> **Status note (2026-09-21).** This folder was merged into the main repository
> from the standalone `sdc-roadfury-icst2026` project. The main repository now
> targets **SOICT 2026** (see the root `CLAUDE.md`), not ICSE 2027 as written
> below, and the SOICT draft lives in `manuscripts/paper/soict/`. The text below
> is kept as originally written; read "the live repository" as the repository
> root and ignore the relative link to `../sdc-test-prioritization-novel`.

This is a **finished, already-public paper**, kept on its own so it stops being
mistaken for the ICSE 2027 manuscript that has not been written yet.

## What it is

> *RoadFury at the ICST 2026 Tool Competition — Self-Driving Car Testing Track*

Five authors. Reported APFD 0.804. Submitted to the SBFT/ICST 2026 tool
competition and public. `paper/paper.pdf` is the built article;
`paper/RoadFury_Source.zip` is the submitted source.

## Why it was separated

The live project next door,
[`sdc-test-prioritization-novel`](../sdc-test-prioritization-novel), targets
**ICSE 2027** with a different paper: a Transformer that reads a test's whole
road shape rather than three summary statistics, with exact rotation and
resolution invariance, evaluated across every public benchmark.

That ICSE manuscript **does not exist yet**. The only `.tex` in the repository
was this one, so anybody opening `paper/` looking for the ICSE draft found a
competition entry with a different title, a different author list, a different
claim, and a page budget that does not apply. Nothing but the feature table and
the related work is reusable between them.

## What can be reused for the ICSE paper

- **The related-work section and the bibliography** — same field, same
  baselines.
- **The feature table** describing the road representation.
- **Nothing else.** Different claim, different evaluation, different venue
  format. The ICSE paper has to be written from scratch.

## What the ICSE paper still needs

For the record, since the two are easy to confuse. The live repo's headline
figure is an empty table: `exps/best_all/tracker.md`, `full_all`, `oob`,
`scissor` and `travel` all read *"## Latest numbers (TODO: fill in after run)"*
with `--` in every cell. `exp_best_all.py` was written and never run; the repo's
own estimate is under ninety minutes per benchmark on a Kaggle T4.

Worth knowing before that run: on SensoDat, where the numbers *are* filled in,
all fourteen "novel" methods tie or lose to a plain 5-model SWA ensemble
(0.8077). The honest contribution is not a better architecture. It is that exact
rotation invariance (Δ = 0.0000) and a 5.6× drop in unphysical predictions come
essentially for free, at a small cost in APFD. The ICSE paper should be written
as that, not as a win.

## Contents

| Path | What it holds |
|---|---|
| `paper/` | the built PDF, LaTeX source, figures, Manim assets, submitted zip |
| `competition_2026.md` | the competition's own notes from the live repo |

Copied, not moved: everything here also remains in the live repository's git
history.
