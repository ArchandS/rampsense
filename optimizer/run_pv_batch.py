"""
Batch driver for the PV optimizer comparison. Calls existing scripts via subprocess.

  python run_pv_batch.py                  -> optimize (adam warm-start + random control, seeds 42,43,44)
                                             then VALIDATION-only threshold sweeps. Test is NOT touched.
  python run_pv_batch.py --stage test     -> test-once evaluation for every tag (each can run only once;
                                             re-runs are refused by threshold.py). Run only after reviewing val.
Options: --evals 5000 --seeds 42 43 44 --configs adam random --init-noise 0.02
"""
from pathlib import Path
import argparse
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SWEEP = HERE / "threshold.py" if (HERE / "threshold.py").exists() else HERE / "threshold_sweep.py"


def run(cmd):
    print("\n>>>", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["optval", "test"], default="optval")
    ap.add_argument("--evals", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--configs", nargs="+", default=["adam", "random"])
    ap.add_argument("--init-noise", type=float, default=0.02)
    a = ap.parse_args()

    for cfg in a.configs:
        for s in a.seeds:
            tag = f"pso_{cfg}_s{s}"
            vec = OUT / f"stage_{tag}_best.npy"
            if a.stage == "optval":
                if not vec.exists():
                    rc = run([sys.executable, str(HERE / "pso_pv.py"), "--init", cfg, "--seed", str(s),
                              "--evals", str(a.evals), "--init-noise", str(a.init_noise), "--tag", tag])
                    if rc != 0:
                        print(f"[WARN] optimizer failed for {tag}"); continue
                else:
                    print(f"[skip] {vec.name} exists")
                run([sys.executable, str(SWEEP), "--phase", "val", "--tag", tag, "--vector", str(vec)])
            else:
                run([sys.executable, str(SWEEP), "--phase", "test", "--tag", tag, "--vector", str(vec)])
    print("\nDone. Next:", "python make_pv_tables.py" if a.stage == "test" else
          "review val results, then python run_pv_batch.py --stage test")


if __name__ == "__main__":
    main()
