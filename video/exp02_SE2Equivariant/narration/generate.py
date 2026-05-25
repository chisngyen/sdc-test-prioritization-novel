"""
Generate per-scene narration MP3s with Edge-TTS.

Usage:
    python narration/generate.py            # generates all scenes
    python narration/generate.py scene_03   # only scene_03

Output: narration/scene_<NN>.mp3 next to this file.

After generation, the script prints each scene's audio duration so we
can pad the scene's final wait() if needed.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import edge_tts

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scripts import NARRATION, VOICE, RATE, PITCH


async def _gen_one(key: str, text: str, max_retries: int = 4) -> None:
    out = HERE / f"{key}.mp3"
    print(f"[generate] {key}: {len(text)} chars -> {out.name}")
    last_err = None
    for attempt in range(max_retries):
        try:
            comm = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
            await comm.save(str(out))
            return
        except Exception as e:
            last_err = e
            print(f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e}")
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed after {max_retries} attempts: {last_err}")


async def _gen_all(keys: list[str]) -> None:
    failed: list[str] = []
    for k in keys:
        if k not in NARRATION:
            print(f"[generate] WARN: unknown scene {k}")
            continue
        try:
            await _gen_one(k, NARRATION[k])
        except Exception as e:
            print(f"[generate] FAILED {k}: {e}")
            failed.append(k)
    if failed:
        print(f"\n[generate] {len(failed)} scenes failed: {failed}")


def _duration(mp3_path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp3_path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        keys = argv[1:]
    else:
        keys = list(NARRATION.keys())

    asyncio.run(_gen_all(keys))

    # Report durations as JSON so other scripts (audio_durations.json
    # consumed by scenes) can pad waits if needed.
    durations: dict[str, float] = {}
    for k in NARRATION.keys():
        mp3 = HERE / f"{k}.mp3"
        if mp3.exists():
            durations[k] = _duration(mp3)
    durations_path = HERE / "audio_durations.json"
    durations_path.write_text(json.dumps(durations, indent=2), encoding="utf-8")

    print("\n[generate] durations (seconds):")
    for k, d in durations.items():
        print(f"    {k:<10}  {d:6.2f}s")
    print(f"\n[generate] total: {sum(durations.values()):.1f}s "
          f"(saved to {durations_path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
