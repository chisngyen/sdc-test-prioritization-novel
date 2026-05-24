"""Concatenate the 7 rendered scenes into one final video.

Usage:
    python concat_all.py              # use 480p15 (default -ql)
    python concat_all.py --qm         # 720p30
    python concat_all.py --hq         # 1080p60
    python concat_all.py --4k         # 2160p60

Output:  full_video_<quality>.mp4  next to this script.

Requires ffmpeg on PATH (Manim ships one; system install also fine).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


QUALITY_DIR = {
    "-ql": "480p15",
    "--qm": "720p30",
    "-qm": "720p30",
    "--hq": "1080p60",
    "-qh": "1080p60",
    "--4k": "2160p60",
    "-qk": "2160p60",
}

SCENES = [
    ("scene_00_context",      "Context"),
    ("scene_01_intro",        "Intro"),
    ("scene_02_input",        "InputPoints"),
    ("scene_03_features",     "FeatureExtract"),
    ("scene_04_invariance",   "RotationProof"),
    ("scene_05_architecture", "Architecture"),
    ("scene_06_attention",    "AttentionBlock"),
    ("scene_06b_compute",     "ComputeWalkthrough"),
    ("scene_07_results",      "Results"),
]


def main() -> int:
    quality = "480p15"
    for arg in sys.argv[1:]:
        if arg in QUALITY_DIR:
            quality = QUALITY_DIR[arg]

    here = Path(__file__).resolve().parent
    media = here / "media" / "videos"

    parts: list[Path] = []
    missing: list[str] = []
    for stem, scene in SCENES:
        p = media / stem / quality / f"{scene}.mp4"
        if p.exists():
            parts.append(p)
        else:
            missing.append(str(p.relative_to(here)))

    if missing:
        print("[concat_all] missing partials -- render them first:")
        for m in missing:
            print(f"   {m}")
        return 2

    # Write concat list
    list_path = here / f"_concat_{quality}.txt"
    with list_path.open("w", encoding="utf-8") as f:
        for p in parts:
            # ffmpeg concat demuxer wants forward slashes or escaped paths;
            # use absolute paths quoted.
            f.write(f"file '{p.as_posix()}'\n")

    out_path = here / f"full_video_{quality}.mp4"
    print(f"[concat_all] joining {len(parts)} files -> {out_path.name}")
    for p in parts:
        print(f"   + {p.name}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(out_path),
    ]
    ret = subprocess.call(cmd)
    list_path.unlink(missing_ok=True)

    if ret != 0:
        print(f"[concat_all] ffmpeg failed ({ret}). "
              "Falling back to re-encode (slower, but rescues mismatched streams)...")
        cmd2 = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        # need to rewrite the list file since we removed it
        with list_path.open("w", encoding="utf-8") as f:
            for p in parts:
                f.write(f"file '{p.as_posix()}'\n")
        ret = subprocess.call(cmd2)
        list_path.unlink(missing_ok=True)

    if ret == 0:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"[concat_all] OK -- {out_path.name} ({size_mb:.1f} MB)")
    return ret


if __name__ == "__main__":
    sys.exit(main())
