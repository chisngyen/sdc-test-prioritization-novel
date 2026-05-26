# UAV-Testing-Competition tracker

Benchmark: https://github.com/skhatiri/UAV-Testing-Competition
External clone: `external/UAV-Testing-Competition/`

## Pipeline

1. `gen_uav_dataset.py` -- sample obstacle placements (same ranges as the
   official `RandomGenerator`) and label each test.
   - `--mode surrogate` (no Docker): label by 3D path-to-box min distance.
   - `--mode sim`: label by Aerialist `min_distance_to_obstacles` (Docker).
2. `exp_uav_prio.py` -- Transformer + SWA + Focal (gamma=2.5), 10-channel
   per-waypoint obstacle context, multi-trial APFD.

## Surrogate run (mission1+2+3, budget=900, safe_dist=2.0m, epochs=15, CPU)

| metric              | value             |
| ------------------- | ----------------- |
| n_tests             | 900               |
| n_fail              | 161 (17.9%)       |
| val AUC             | 0.9926            |
| APFD best (30 tr)   | 0.8985 +/- 0.0187 |
| APFD SWA  (30 tr)   | 0.8951 +/- 0.0201 |
| APFD all-data       | 0.9050            |
| wall                | 5956 s (CPU)      |

Honest caveat: the surrogate label is a closed-form function of the same
geometric inputs the model sees, so high APFD here is largely "model
learns the analytic surrogate". Real-claim numbers require `--mode sim`.

## Sim run (TODO -- needs Docker + Aerialist)

| metric              | value |
| ------------------- | ----- |
| n_tests             | TBD   |
| n_fail              | TBD   |
| val AUC             | TBD   |
| APFD best (30 tr)   | TBD   |
| APFD SWA  (30 tr)   | TBD   |

## Repro

```
python exps/uav/gen_uav_dataset.py --mode surrogate --budget 3000 --safe_dist 1.5
python exps/uav/exp_uav_prio.py --mode surrogate

# Once Docker + aerialist are installed:
python exps/uav/gen_uav_dataset.py --mode sim --budget 500
python exps/uav/exp_uav_prio.py --mode sim
```
