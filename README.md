# RoadFury & SE2RoadNet: Deep Learning for SDC Test Prioritization

Official repository for **RoadFury** and its geometric extension **SE2RoadNet** for simulation-based regression test prioritization of Self-Driving Cars (SDCs).

## Overview

Testing self-driving cars in high-fidelity simulators (e.g. BeamNG.tech) is computationally intensive. Prioritizing tests that are most likely to reveal autonomous driving failures (e.g. off-lane departures) maximizes early fault detection within restricted execution budgets.

### 1. RoadFury (ICST 2026 Tool Competition)
- **Architecture**: Pre-LN Transformer Encoder (4 layers, 8 heads, $d=128$) with learnable `[CLS]` token and 10-channel road geometry features.
- **Optimization**: Stochastic Weight Averaging (SWA) finding flatter loss minima.
- **Performance**: APFD = **0.8042 ± 0.0120** on SensoDat competition benchmark.
- **Implementation**: Located at `tools/prioritizers/road_fury/` (containerized gRPC service).

### 2. SE2RoadNet (Geometric & Physics-Grounded Extension)
- **$SE(2)$ Equivariance**: Intrinsic coordinate-free geometric features and relative-arclength attention bias ensuring exact rotation invariance ($\Delta\text{APFD} = 0.0000$).
- **Physical Regularization**: PINN auxiliary loss constraining centrifugal acceleration $v^2 \kappa(s)$ to eliminate unphysical predictions.
- **Manuscripts**:
  - ICST 2026 Tool paper: `manuscripts/paper/legacy_icst/icst2026_roadfury.tex`
  - SOICT 2026 Manuscript: `manuscripts/paper/soict/main.tex`

---

## Repository Structure

```
.
├── tools/
│   └── prioritizers/
│       └── road_fury/               # RoadFury tool (Dockerfile, gRPC server, weights)
├── manuscripts/
│   └── paper/
│       ├── legacy_icst/             # RoadFury ICST 2026 paper & presentation
│       ├── figures/                 # Architecture figures
│       └── soict/                   # SOICT 2026 paper draft & sections
├── exps/
│   ├── core/                        # Core experiments (exp00 RoadTransformer, exp02 SE2RoadNet, etc.)
│   ├── results/                     # Experimental result JSONs
│   ├── benchmarks/                  # Cross-benchmark evaluation scripts
│   └── tracker.md                   # Experiment scoreboard
├── data/                            # Dataset loaders & local mirrors
├── docs/                            # Mathematical notes & DATA.md
├── scripts/                         # Utility scripts
└── README.md
```

---

## Running RoadFury

### With Docker
```bash
cd tools/prioritizers/road_fury
docker build -t road-fury .
docker run --rm -t -p 4545:4545 road-fury -p 4545
```

### Local Execution
```bash
cd tools/prioritizers/road_fury
pip install -r requirements.txt
python main.py -p 4545
```

---

## Compiling Manuscripts

```bash
# SOICT 2026 paper
cd manuscripts/paper/soict
latexmk -pdf -interaction=nonstopmode main.tex
```
