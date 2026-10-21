"""
Extra Adam H=16 seeds so the comparison is 3 seeds vs 3 seeds.

For each seed (default 43 and 44):
  1. train_adam_h16.py --seed S               -> outputs\\stage_adam_h16_sS_best.npy   (skipped if it exists)
  2. threshold.py --phase val  --tag adam16_sS (validation-only sweep, freezes thresholds)
  3. threshold.py --phase test --tag adam16_sS (test-once; refused automatically if already done)
Finally: make_pv_tables.py  (tags adam16_s43 / adam16_s44 are grouped with the existing 'adam16' row).

Run:
  python run_adam_seeds.py                 # train + val + test + tables
  python run_adam_seeds.py --no-test       # stop after validation
  python run_adam_seeds.py --dry-run       # print the commands only
Safe to re-run: existing vectors are not retrained and the test-once guard blocks a second test run.
"""
from pathlib import Path
import argparse
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SWEEP = HERE / "threshold.py" if (HERE / "threshold.py").exists() else HERE / "threshold_sweep.py"


def run(cmd, dry):
    print("\n>>>", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    return 0 if dry else subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for s in a.seeds:
        vec = OUT / f"stage_adam_h16_s{s}_best.npy"
        tag = f"adam16_s{s}"
        if vec.exists():
            print(f"[skip training] {vec.name} exists")
        elif run([sys.executable, str(HERE / "train_adam_h16.py"), "--seed", str(s)], a.dry_run) != 0:
            print(f"[ERROR] training failed for seed {s}; skipping its sweeps")
            continue
        if run([sys.executable, str(SWEEP), "--phase", "val", "--tag", tag, "--vector", str(vec)], a.dry_run) != 0:
            print(f"[ERROR] validation sweep failed for {tag}")
            continue
        if not a.no_test:
            rc = run([sys.executable, str(SWEEP), "--phase", "test", "--tag", tag, "--vector", str(vec)], a.dry_run)
            if rc != 0:
                print(f"[NOTE] test phase for {tag} returned {rc} (already evaluated, or an error above)")

    run([sys.executable, str(HERE / "make_pv_tables.py")], a.dry_run)
    print("\nDone. Send me outputs\\pv_results_summary.md")


if __name__ == "__main__":
    main()