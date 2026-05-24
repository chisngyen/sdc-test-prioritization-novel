"""Batch render all 7 scenes for the Exp 02 video.

Usage (from this folder):
    python render_all.py          # preview quality (-ql)
    python render_all.py --hq     # 1080p60 (-qh)
    python render_all.py --4k     # 2160p60 (-qk)

Output goes to media/videos/<scene_file>/<quality>/<SceneName>.mp4 next
to this script.
"""
import subprocess, sys, os, time

SCENES = [
    ("scene_00_context.py",      "Context"),
    ("scene_01_intro.py",        "Intro"),
    ("scene_02_input.py",        "InputPoints"),
    ("scene_03_features.py",     "FeatureExtract"),
    ("scene_04_invariance.py",   "RotationProof"),
    ("scene_05_architecture.py", "Architecture"),
    ("scene_06_attention.py",    "AttentionBlock"),
    ("scene_06b_compute.py",     "ComputeWalkthrough"),
    ("scene_07_results.py",      "Results"),
]


def main():
    flag = "-ql"
    if "--hq" in sys.argv:
        flag = "-qh"
    if "--4k" in sys.argv:
        flag = "-qk"

    here = os.path.dirname(os.path.abspath(__file__))
    print(f"[render_all] rendering {len(SCENES)} scenes with {flag} into {here}")
    t0 = time.time()

    for fname, scene_class in SCENES:
        print("\n" + "=" * 64)
        print(f"  >>> {fname} :: {scene_class}")
        print("=" * 64)
        ret = subprocess.call(
            ["python", "-m", "manim", flag, fname, scene_class],
            cwd=here,
        )
        if ret != 0:
            print(f"[render_all] FAILED on {fname} :: {scene_class} "
                  f"(exit {ret}).  Stopping.")
            sys.exit(ret)

    print(f"\n[render_all] done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
