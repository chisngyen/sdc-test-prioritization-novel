# Data manifest

None of the datasets below are stored in this Git repository. They total about
11.5 GB across 65,000 files, five archives exceed GitHub's 100 MB per-file limit,
and every one of them is already hosted by its original authors. This file
records where each one comes from, how big it is, where it goes on disk, and how
to get it back. All local paths are relative to the repository root and are
excluded by `.gitignore`.

Measured on 2026-09-21 from the working copy at commit `2b49121`.

## Summary

| Local path | Source | Files | Size |
|---|---|---:|---:|
| `data/kaggle/dataset-oob/` | Kaggle `chiboiz/dataset-oob` | 16,070 | 4,636.8 MB |
| `data/kaggle/sdc-travel/` | Kaggle `chiboiz/sdc-travel` | 45,399 | 4,336.7 MB |
| `data/kaggle/sdc-pririotizer-rp/` | Kaggle `chiboiz/sdc-pririotizer-rp` | 3,302 | 1,230.6 MB |
| `data/kaggle/sdc-scissor/` | Kaggle `chinguyeen/sdc-scissor` | 456 | 33.2 MB |
| `data/zenodo/16939865-oob-regression/` | Zenodo record 16939865 | 3 | 514.7 MB |
| `data/zenodo/5914130-sdc-scissor/` | Zenodo record 5914130 | 5 | 745.6 MB |
| `external/DeepScenario/` | GitHub `Simula-COMPLEX/DeepScenario` | 33,818 | 1,247.3 MB |
| `external/UAV-Testing-Competition/` | GitHub `skhatiri/UAV-Testing-Competition` | 71 | 176.2 MB |

Small files that are tracked in Git: `data/sensodat_features.csv` (13.5 MB) and
`data/uav/uav_dataset_surrogate.json` (0.5 MB).

## Kaggle datasets

Download all four with `python scripts/download_kaggle_datasets.py`. It needs
Kaggle API credentials and the `kagglehub` package, and writes into
`data/kaggle/<folder>/`.

| Folder | Kaggle handle | Layout | Dominant file types |
|---|---|---|---|
| `dataset-oob` | `chiboiz/dataset-oob` | `Dataset-OOB-0-1/`, `Dataset-OOB-0-3/`, `Dataset-OOB-0-5/` (out-of-bound tolerance 0.1, 0.3, 0.5) | 16,070 `.json` |
| `sdc-travel` | `chiboiz/sdc-travel` | `competition/`, `sdc-prioritizer/`, `sdc-scissor/`, `README.pdf` | 34,498 `.json`, 3,559 each of `.jpg`, `.svg`, `.tsv`, 222 `.csv` |
| `sdc-pririotizer-rp` | `chiboiz/sdc-pririotizer-rp` | `SDC-Pririotizer-RP/` (has `datasets/fullroad/` and its own `README.md`) | 2,446 `.csv`, 813 `.png`, 14 `.m`, 9 `.pdf`, 4 `.R` |
| `sdc-scissor` | `chinguyeen/sdc-scissor` | `christianbirchler-org-sdc-scissor-faf11b2/` (a snapshot of the SDC-Scissor tool repository) | 215 `.json`, 111 `.py`, 30 `.md`, 14 `.m`, 13 `.png` |

`sdc-pririotizer-rp` keeps the misspelling "pririotizer" because that is the
Kaggle slug; do not correct it or the download script will fetch nothing.

## Zenodo archives

Both records are stored under `data/zenodo/<record-id>-<name>/`. The record page
is `https://zenodo.org/records/<record-id>`, and the DOI is
`10.5281/zenodo.<record-id>`. The licence and citation text are on the record
page and were not copied here.

Example download:

```bash
mkdir -p data/zenodo/16939865-oob-regression
curl -L -o data/zenodo/16939865-oob-regression/Dataset-OOB-0-3.zip \
  "https://zenodo.org/records/16939865/files/Dataset-OOB-0-3.zip?download=1"
```

Verify with the SHA-256 values below (computed locally, so a mismatch means the
copy is damaged or the record was revised).

### Record 16939865, `oob-regression`

| File | Bytes | SHA-256 |
|---|---:|---|
| `Dataset-OOB-0-1.zip` | 38,893,706 | `e4eb7f57f2f1a3a8547afa8700888736a90f27a236f013b638dd80a887cee8b6` |
| `Dataset-OOB-0-3.zip` | 154,330,964 | `6dafbfebd09b2537d1746707e65a15e7ccfc0fe865c6b17d81ab58ca23e9ea49` |
| `Dataset-OOB-0-5.zip` | 346,489,603 | `db3bb6c5343ee37ace440c528ddc53775991e1ed7f11769d42fa67895c1cebaf` |

### Record 5914130, `sdc-scissor`

| File | Bytes | SHA-256 |
|---|---:|---|
| `data-for-demo.zip` | 88,697,849 | `0e83eac4e0524397079ab2c6635f3eb697b293f8a54dd80df5068b16f720f49f` |
| `README.pdf` | 461,463 | `f2eee44a8abca4af9a05a60efb83794a4053cfdfbc21369b004ace2c1cb7642b` |
| `RF_1-5_OOB_0-5_SPEED_120.zip` | 215,657,035 | `342023be26dce9babb854edb85c70edff57e8c5c91197f7ce255749c5ac8a1e6` |
| `RF_1_OOB_0-5_SPEED_120.zip` | 256,649,184 | `01ddeccc0bc4206c05d157b951211c4a8c5336e8f3a2f8e3a087ebac758fdbdd` |
| `RF_2_OOB_0-5_SPEED_120.zip` | 220,311,319 | `6717353431435b5ed20e5676576e2c0e695eed6385e5629941d5b3402a382916` |

Five of these archives are over 100 MB (`Dataset-OOB-0-3`, `Dataset-OOB-0-5`, and
the three `RF_*` files), which is the concrete reason none of this can go
through plain `git push`.

## Third-party repositories in `external/`

Cloned for reference only, each with its own `.git`, and not vendored into this
repository. Re-create with:

```bash
git clone https://github.com/Simula-COMPLEX/DeepScenario.git external/DeepScenario
git -C external/DeepScenario checkout 7eff06fc21309269483c8ffca0c13dc7d1b1c827

git clone https://github.com/skhatiri/UAV-Testing-Competition.git external/UAV-Testing-Competition
git -C external/UAV-Testing-Competition checkout adec8b5af37f03e5426f746b7fcdd5d84628b64f
```

| Repository | Pinned commit | Commit date |
|---|---|---|
| `Simula-COMPLEX/DeepScenario` | `7eff06fc21309269483c8ffca0c13dc7d1b1c827` | 2024-01-26 |
| `skhatiri/UAV-Testing-Competition` | `adec8b5af37f03e5426f746b7fcdd5d84628b64f` | 2026-02-20 |

## Which code reads which data

Found by searching the repository for each name (checked 2026-09-21):

| Data | Read by |
|---|---|
| `dataset-oob` | `exps/benchmarks/oob/exp_best_oob.py` (1 script) |
| `sdc-scissor` | 7 scripts under `exps/`, including `exps/benchmarks/full_all/exp_01_geom_tta.py` |
| `sdc-travel` | 9 scripts under `exps/`, including `exps/benchmarks/best_all/exp_best_all.py` |
| `sdc-pririotizer-rp` | 8 scripts under `exps/`, including `exps/benchmarks/best_all/exp_se2_rp_bench.py` |
| `data/uav/uav_dataset_surrogate.json`, `external/UAV-Testing-Competition/` | `exps/benchmarks/uav/exp_uav_prio.py`, `gen_uav_dataset.py` |
| `data/sensodat_features.csv` | `manuscripts/presentation/make_figs.py` |
| `data/zenodo/*`, `external/DeepScenario/` | no script in the repository refers to them; they are local reference copies |

Experiment scripts find their inputs through the `SEARCH_ROOTS` pattern
described in the root `CLAUDE.md`, so they also run on Kaggle without arguments.
