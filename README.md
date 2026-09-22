# CliffordTrajNet: Geometric Clifford Representations & Continuous State-Space Dynamics for SDC and UAV Test Prioritization

This repository contains the official codebase and LaTeX manuscript for **CliffordTrajNet**, targeting **SOICT 2026**.

## Overview

Testing safety-critical autonomous systems—including Self-Driving Cars (SDCs) and Unmanned Aerial Vehicles (UAVs)—in high-fidelity simulation is computationally expensive. **CliffordTrajNet** introduces a novel geometric deep learning approach for test case prioritization:

1. **Geometric Clifford Multivector Embeddings**: Trajectories are embedded into Clifford geometric algebras ($\mathcal{G}(2,0)$ for SE(2) ground vehicles and $\mathcal{G}(3,0)$ for 3D aerial trajectories), guaranteeing exact roto-translational equivariance/invariance.
2. **Continuous State-Space Dynamics**: Employs continuous-time selective state-space layers (Continuous TrajMamba ODE) to handle irregular sampling, temporal curvature, and varying trajectory lengths.
3. **Conformal Risk Control**: Implements distribution-free risk bounds to select test subsets with formal statistical guarantees on failure recall.

```
                  ┌──────────────────────┐
                  │ 3D/2D Test Trajectory│
                  └──────────┬───────────┘
                             │
                  ▼──────────────────────▼
                  │ Clifford Embeddings  │
                  │  Cl(3,0) / Cl(2,0)   │
                  └──────────┬───────────┘
                             │
                  ▼──────────────────────▼
                  │ Continuous TrajMamba │
                  │  Selective SSM ODE   │
                  └──────────┬───────────┘
                             │
                  ▼──────────────────────▼
                  │ Conformal Calibrator │
                  │   Risk Bound (CRC)   │
                  └──────────┬───────────┘
                             │
                  ▼──────────────────────▼
                  │ Prioritized Schedule │
                  └──────────────────────┘
```

---

## Repository Structure

```
.
├── manuscripts/
│   └── paper/
│       ├── figures/
│       │   └── pipeline.png         # Main architecture and pipeline diagram
│       └── soict/
│           ├── sections/            # Modular paper sections (01 to 07)
│           ├── main.tex             # Main Springer LNCS LaTeX manuscript
│           ├── references.bib       # Clean bibtex citations
│           └── llncs.cls            # Official Springer LNCS document class
├── exps/
│   ├── core/                        # Core model implementations & experiment scripts
│   │   ├── exp00_Basline.py         # Baseline trajectory Transformer
│   │   ├── exp02_SE2Equivariant.py  # SE(2) Clifford geometric network
│   │   ├── exp20_ConformalRiskControl_SafetyBound.py
│   │   └── exp21_UAV_Clifford3D_TrajMamba.py
│   └── results/                     # Experimental results & verified JSON logs
├── data/                            # Dataset specifications & loaders
├── docs/                            # Mathematical derivations & architecture notes
└── scripts/                         # Helper scripts
```

---

## Getting Started

### 1. Compiling the Manuscript

The paper is formatted according to the Springer LNCS / CCIS conference style.

```bash
cd manuscripts/paper/soict
latexmk -pdf -interaction=nonstopmode main.tex
```

This generates `main.pdf` (14 pages).

### 2. Running Experiments

Install dependencies:
```bash
pip install torch numpy scipy scikit-learn
```

Execute the 3D UAV Clifford TrajMamba experiment:
```bash
python exps/core/exp21_UAV_Clifford3D_TrajMamba.py
```

Results will be logged directly to `exps/results/exp21_UAV_Clifford3D_results.json`.

---

## Citation & Contact

If you use this work or findings in your research, please cite:

```bibtex
@inproceedings{cliffordtrajnet2026,
  title     = {CliffordTrajNet: Geometric Clifford Representations and Continuous State-Space Dynamics for Autonomous Driving and UAV Test Prioritization},
  author    = {Chis Nguyen and Collaborators},
  booktitle = {Proceedings of the International Symposium on Information and Communication Technology (SOICT)},
  year      = {2026}
}
```
