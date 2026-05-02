"""Download SDC-related Kaggle datasets via kagglehub. Requires Kaggle API credentials."""
from __future__ import annotations

from pathlib import Path

import kagglehub

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DATA = REPO_ROOT / "data" / "kaggle"

# (kaggle handle, folder name under data/kaggle/)
DATASETS: list[tuple[str, str]] = [
    ("chinguyeen/sdc-scissor", "sdc-scissor"),
    ("chiboiz/sdc-travel", "sdc-travel"),
    ("chiboiz/dataset-oob", "dataset-oob"),
    ("chiboiz/sdc-pririotizer-rp", "sdc-pririotizer-rp"),
]


def main() -> None:
    KAGGLE_DATA.mkdir(parents=True, exist_ok=True)
    for handle, folder in DATASETS:
        out = str(KAGGLE_DATA / folder)
        print(f"Downloading: {handle} -> {out} ...")
        path = kagglehub.dataset_download(handle, output_dir=out)
        print(f"  -> {path}\n")


if __name__ == "__main__":
    main()
