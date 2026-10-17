"""
RampSense - Parameter-light reference baselines for ramp-event detection
========================================================================
Purpose: calibrate what F1 a model must beat to be doing real ramp forecasting.
Ground truth is identical to the model pipeline (12.25 W/min, 8-point t+6..t+13 target,
target-scaler inverse, np.diff after inversion, >=).

Baselines (no neural network, no optimizer):
  B0  day-flag      : flag every window whose last-step G_h > GH_MIN        (no tuning)
  B1  recent-ramp   : score = max |diff| of PV_Power over the last K minutes of the input
                      history (inverse feature scaling). Flag if score >= thr.
                      K in K_GRID and thr on a grid are chosen on VALIDATION only
                      (max val F1), then frozen.
  B2  recent-ramp restricted to daylight (B1 AND day-flag)

Run:
  python baselines.py --phase val
  python baselines.py --phase test      (refuses to run twice)
All outputs go to outputs\\baselines_*.json / csv.
Labels: [ENGINEERING] baselines; K_GRID/threshold grid are engineering choices.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

OPTIMIZER_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = OPTIMIZER_DIR / "outputs"
DATA_DIR = OPTIMIZER_DIR.parent / "multistep_8" / "split_scaled"

LABEL_THRESHOLD = 12.25
GH_INDEX, PV_INDEX, GH_MIN = 1, 0, 5.0
K_GRID = (2, 3, 5, 10, 15)                     # minutes of recent history
THR_GRID = np.round(np.arange(0.0, 60.0 + 1e-9, 0.25), 6)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return p, r, f


def counts(actual, flag):
    return (int((actual & flag).sum()), int((~actual & flag).sum()),
            int((actual & ~flag).sum()), int((~actual & ~flag).sum()))


def report(actual, flag):
    tp, fp, fn, tn = counts(actual, flag)
    p, r, f = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f,
            "predicted_rate": float(flag.mean()), "actual_rate": float(actual.mean())}


def recent_ramp_scores(pv_w, k):
    """pv_w: (N,60) watts. max|diff| over last k+1 points (= k differences)."""
    return np.max(np.abs(np.diff(pv_w[:, -(k + 1):], axis=1)), axis=1)


def best_on_val(actual, pv_w, day):
    best = None
    table = []
    for k in K_GRID:
        s = recent_ramp_scores(pv_w, k)
        for restrict in (False, True):
            for th in THR_GRID:
                flag = (s >= th) & (day if restrict else True)
                tp, fp, fn, _ = counts(actual, flag)
                f = prf(tp, fp, fn)[2]
                if best is None or f > best["f1"]:
                    best = {"k": k, "daylight_restricted": restrict, "threshold": float(th), "f1": f}
        # table row: best F1 per K (unrestricted) for the log
        fs = [prf(*counts(actual, s >= th)[:3])[2] for th in THR_GRID]
        table.append({"k": k, "best_f1_unrestricted": float(max(fs)),
                      "at_threshold": float(THR_GRID[int(np.argmax(fs))])})
    return best, table


def load_split(split):
    import joblib
    X = np.load(DATA_DIR / f"X_{split}.npy", mmap_mode="r")
    Y = np.load(DATA_DIR / f"Y_{split}.npy", mmap_mode="r")
    ts = joblib.load(DATA_DIR / "target_scaler.pkl")
    fs = joblib.load(DATA_DIR / "feature_scaler.pkl")
    act_w = ts.inverse_transform(np.asarray(Y, dtype=np.float64))
    actual = np.max(np.abs(np.diff(act_w, axis=1)), axis=1) >= LABEL_THRESHOLD
    pv_w = np.asarray(X[:, :, PV_INDEX], dtype=np.float64) * fs.scale_[PV_INDEX] + fs.mean_[PV_INDEX]
    gh = np.asarray(X[:, -1, GH_INDEX], dtype=np.float64) * fs.scale_[GH_INDEX] + fs.mean_[GH_INDEX]
    return actual, pv_w, gh > GH_MIN


def phase_val(loader=load_split):
    actual, pv_w, day = loader("val")
    b0 = report(actual, day)
    best, table = best_on_val(actual, pv_w, day)
    s = recent_ramp_scores(pv_w, best["k"])
    flag = (s >= best["threshold"]) & (day if best["daylight_restricted"] else True)
    b1 = report(actual, flag)
    out = {"stage": "baselines_VAL_ONLY", "label_threshold": LABEL_THRESHOLD, "n": int(len(actual)),
           "B0_day_flag": b0, "B1_recent_ramp_frozen_params": best, "B1_val_result": b1,
           "per_K_best_f1_unrestricted": table, "frozen": True}
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "baselines_frozen.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nTest NOT touched. Saved outputs\\baselines_frozen.json")


def phase_test(loader=load_split):
    fz_path = OUTPUT_DIR / "baselines_frozen.json"
    out_path = OUTPUT_DIR / "baselines_test_once_report.json"
    if not fz_path.exists():
        raise FileNotFoundError("run --phase val first")
    if out_path.exists():
        raise RuntimeError("baselines already evaluated on test (test-once rule).")
    fz = json.loads(fz_path.read_text())
    actual, pv_w, day = loader("test")
    p = fz["B1_recent_ramp_frozen_params"]
    s = recent_ramp_scores(pv_w, p["k"])
    flag = (s >= p["threshold"]) & (day if p["daylight_restricted"] else True)
    out = {"stage": "baselines_TEST_ONCE", "n": int(len(actual)), "actual_rate": float(actual.mean()),
           "B0_day_flag": report(actual, day), "B1_recent_ramp": report(actual, flag),
           "params_frozen_on_val": p}
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["val", "test"], required=True)
    a = ap.parse_args()
    (phase_val if a.phase == "val" else phase_test)()