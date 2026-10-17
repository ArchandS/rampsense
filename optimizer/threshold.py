"""
RampSense - Decision-threshold analysis (validation-only selection, test-once evaluation)
=========================================================================================

Two DIFFERENT thresholds are involved. They are kept separate everywhere:

  LABEL_THRESHOLD     = 12.25 W/min  -> defines ground-truth ramp events (ORE). FIXED. Never swept.
  decision threshold  = swept        -> applied to the predicted max|diff| (PRE score).
                                        Chosen on VALIDATION only, then frozen.

Phase 1 (val):  python threshold_sweep.py --phase val  --tag p1
    - predicts validation windows, sweeps decision thresholds, selects two operating points
      by criteria fixed IN THIS FILE (below), writes outputs/threshold_<tag>_frozen.json
Phase 2 (test): python threshold_sweep.py --phase test --tag p1
    - reads the frozen JSON, evaluates the TEST set at the frozen thresholds, and refuses to
      run a second time (test-once guard). No sweep is ever performed on test labels.

Other vector (e.g. C11):  --vector outputs\\stage_c11_samurai_best.npy --tag c11

Labels used in the report:
  [PAPER]              ramp = max |P_s - P_(s-1)| over the window, compared with a threshold (Eq. 5/6)
  [OUR ADAPTATION]     12.25 W/min label threshold, 8-output t+6..t+13 target
  [ENGINEERING]        decision-threshold sweep, selection criteria, daylight PROXY, recall floor
"""

from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np

# ------------------------------------------------------------------ configuration
OPTIMIZER_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = OPTIMIZER_DIR / "outputs"
DATA_DIR = OPTIMIZER_DIR.parent / "multistep_8" / "split_scaled"

INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, INPUT_LENGTH = 6, 16, 8, 60
BATCH_SIZE = 512

LABEL_THRESHOLD = 12.25            # W/min, ground-truth definition. DO NOT sweep.
SWEEP_MIN, SWEEP_MAX, SWEEP_STEP = 0.0, 40.0, 0.25   # decision-threshold grid (W/min)

# ---- selection criteria, fixed BEFORE looking at any result ------------------------------
# A: decision threshold maximising VALIDATION F1 (first/lowest argmax).
# B: highest decision threshold whose VALIDATION recall >= RECALL_FLOOR (= max precision
#    under a recall floor; motivated by the paper's argument that misses cost more than
#    false alarms). The floor value itself is an [ENGINEERING] choice.
RECALL_FLOOR = 0.80

# daylight PROXY: G_h at the last input step (feature index 1) > GH_MIN_W_M2 (after inverse
# feature scaling). This is NOT the dataset's is_daylight column; replace once aligned.
GH_INDEX = 1
GH_MIN = 5.0

FEATURE_NAMES = ["PV_Power", "G_h", "G_tilt", "T_air", "W_s", "W_d"]


# ------------------------------------------------------------------ pure-numpy helpers
def max_abs_diff(y_w):
    """y_w: (N,8) watts for t+6..t+13 -> (N,) max |consecutive difference| (7 diffs)."""
    return np.max(np.abs(np.diff(y_w, axis=1)), axis=1)


def confusion(actual, predicted):
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return p, r, f


def sweep(actual_flags, pred_score, thresholds):
    """Ground-truth flags fixed; decision threshold varies on pred_score. Returns list of dicts."""
    rows = []
    n_act = int(actual_flags.sum())
    prev = n_act / len(actual_flags)
    for th in thresholds:
        pf = pred_score >= th
        tp, fp, fn, tn = confusion(actual_flags, pf)
        p, r, f = prf(tp, fp, fn)
        rows.append({
            "decision_threshold": float(th), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f,
            "predicted_rate": float(pf.mean()),
            "lift": (p / prev) if prev > 0 else float("nan"),
        })
    return rows


def select_operating_points(rows, recall_floor):
    """Apply criteria A and B on VALIDATION rows only."""
    f1s = np.array([r["f1"] for r in rows])
    ia = int(np.argmax(f1s))                       # first (lowest-threshold) argmax
    plateau = [r["decision_threshold"] for r in rows if r["f1"] >= f1s[ia] - 0.01]
    A = {"criterion": "max validation F1", "decision_threshold": rows[ia]["decision_threshold"],
         "val_f1": rows[ia]["f1"], "val_precision": rows[ia]["precision"], "val_recall": rows[ia]["recall"],
         "thresholds_within_0.01_F1_of_max": [min(plateau), max(plateau)]}
    ok = [r for r in rows if r["recall"] >= recall_floor]
    if ok:
        rb = max(ok, key=lambda r: (r["precision"], r["decision_threshold"]))
        B = {"criterion": f"max validation precision s.t. recall >= {recall_floor}",
             "decision_threshold": rb["decision_threshold"], "val_f1": rb["f1"],
             "val_precision": rb["precision"], "val_recall": rb["recall"]}
    else:
        B = {"criterion": f"max validation precision s.t. recall >= {recall_floor}",
             "decision_threshold": None, "note": "no threshold on the grid reaches the recall floor"}
    return A, B


def threshold_free(actual_flags, pred_score):
    """Average precision and ROC-AUC of the score vs fixed labels (no decision threshold)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if actual_flags.sum() == 0 or actual_flags.all():
        return {"average_precision": None, "roc_auc": None}
    return {"average_precision": float(average_precision_score(actual_flags, pred_score)),
            "roc_auc": float(roc_auc_score(actual_flags, pred_score)),
            "prevalence_(AP_of_random_score)": float(actual_flags.mean())}


def eval_at(actual_flags, pred_score, th):
    tp, fp, fn, tn = confusion(actual_flags, pred_score >= th)
    p, r, f = prf(tp, fp, fn)
    prev = actual_flags.mean()
    return {"decision_threshold": th, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f,
            "predicted_rate": float((pred_score >= th).mean()),
            "actual_rate": float(prev), "lift": (p / prev) if prev > 0 else None}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


# ------------------------------------------------------------------ torch part (isolated)
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def predict_scores(vector_path, split):
    """Returns dict with scaled preds, actual/pred max-ramp (W/min), daylight proxy mask."""
    import torch
    import joblib

    PVLSTM = _load_module("rs_model", OPTIMIZER_DIR / "model.py").PVLSTM
    set_vector = _load_module("rs_pv", OPTIMIZER_DIR / "param_vector.py").set_vector

    vec = np.asarray(np.load(vector_path), dtype=np.float64)
    if vec.shape != (1608,) or not np.all(np.isfinite(vec)):
        raise ValueError(f"bad vector {vec.shape}")
    model = PVLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    set_vector(model, vec)
    model.eval()

    X = np.load(DATA_DIR / f"X_{split}.npy", mmap_mode="r")
    Y = np.load(DATA_DIR / f"Y_{split}.npy", mmap_mode="r")
    tscaler = joblib.load(DATA_DIR / "target_scaler.pkl")
    fscaler = joblib.load(DATA_DIR / "feature_scaler.pkl")
    if X.shape[1:] != (INPUT_LENGTH, INPUT_SIZE) or Y.shape[1] != OUTPUT_SIZE or len(X) != len(Y):
        raise ValueError(f"unexpected shapes {X.shape} {Y.shape}")

    preds = []
    with torch.inference_mode():
        for s in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(np.asarray(X[s:s + BATCH_SIZE], dtype=np.float32))
            preds.append(model(xb).numpy())
    preds = np.concatenate(preds, 0)

    pred_w = tscaler.inverse_transform(preds)                       # per-column inverse
    act_w = tscaler.inverse_transform(np.asarray(Y, dtype=np.float64))
    gh = np.asarray(X[:, -1, GH_INDEX], dtype=np.float64) * fscaler.scale_[GH_INDEX] + fscaler.mean_[GH_INDEX]
    return {
        "pred_scaled": preds, "pred_w": pred_w, "act_w": act_w,
        "pred_score": max_abs_diff(pred_w),            # diff AFTER inverse transform
        "act_max_ramp": max_abs_diff(act_w),
        "daylight_proxy": gh > GH_MIN,
        "mse_scaled": float(np.mean((preds.astype(np.float64) - np.asarray(Y, dtype=np.float64)) ** 2)),
        "mae_w": float(np.mean(np.abs(pred_w - act_w))),
        "rmse_w": float(np.sqrt(np.mean((pred_w - act_w) ** 2))),
    }


# ------------------------------------------------------------------ phases
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def phase_val(vector_path, tag):
    t0 = time.perf_counter()
    d = predict_scores(vector_path, "val")
    actual = d["act_max_ramp"] >= LABEL_THRESHOLD            # fixed ground truth
    score = d["pred_score"]
    grid = np.round(np.arange(SWEEP_MIN, SWEEP_MAX + 1e-9, SWEEP_STEP), 6)

    # consistency check: at the label threshold the confusion must match the stored P1 report
    ref = OUTPUT_DIR / "stage_p1_pso_validation_report.json"
    check = None
    if tag == "p1" and ref.exists():
        r = json.loads(ref.read_text())["splits"]["val"]
        got = confusion(actual, score >= LABEL_THRESHOLD)
        exp = (r["tp"], r["fp"], r["fn"], r["tn"])
        check = {"expected_from_report": exp, "recomputed": got, "match": got == exp}
        print("Consistency vs stage_p1_pso_validation_report.json (val @12.25):", check)
        if got != exp:
            raise RuntimeError("Recomputed confusion differs from stored report; stop and investigate.")

    rows = sweep(actual, score, grid)
    A, B = select_operating_points(rows, RECALL_FLOOR)
    tf = threshold_free(actual, score)

    day = d["daylight_proxy"]
    rows_day = sweep(actual[day], score[day], grid) if day.any() else []
    tf_day = threshold_free(actual[day], score[day]) if day.any() else {}

    print(f"\nVAL windows: {len(actual)} | label ramp rate: {actual.mean():.4%} | "
          f"daylight-proxy fraction: {day.mean():.3f} | ramp rate in daylight proxy: "
          f"{(actual[day].mean() if day.any() else float('nan')):.4%}")
    print(f"Predicted max-ramp score: median {np.median(score):.3f}  p95 {np.percentile(score,95):.3f}  max {score.max():.3f}")
    print(f"Threshold-free (val): AP={tf.get('average_precision')}  ROC-AUC={tf.get('roc_auc')}  "
          f"(AP of a random score = prevalence {actual.mean():.4f})")
    print(f"\n{'dec.thr':>8} {'prec':>8} {'recall':>8} {'F1':>8} {'pred%':>8} {'lift':>6}")
    for r in rows:
        if abs(r["decision_threshold"] * 4 - round(r["decision_threshold"] * 4)) < 1e-9 and \
           (r["decision_threshold"] % 2 == 0 or abs(r["decision_threshold"] - LABEL_THRESHOLD) < 1e-9):
            print(f"{r['decision_threshold']:8.2f} {r['precision']:8.4f} {r['recall']:8.4f} "
                  f"{r['f1']:8.4f} {r['predicted_rate']*100:8.2f} {r['lift']:6.2f}")
    print("\nCriterion A:", A)
    print("Criterion B:", B)

    OUTPUT_DIR.mkdir(exist_ok=True)
    write_csv(OUTPUT_DIR / f"threshold_{tag}_val_sweep.csv", rows)
    if rows_day:
        write_csv(OUTPUT_DIR / f"threshold_{tag}_val_sweep_daylight_proxy.csv", rows_day)
    np.savez_compressed(OUTPUT_DIR / f"threshold_{tag}_val_scores.npz",
                        pred_score=score, act_max_ramp=d["act_max_ramp"], daylight_proxy=day)

    frozen = {
        "stage": "decision_threshold_analysis_VAL_ONLY", "tag": tag,
        "vector_path": str(vector_path), "vector_sha256": sha256_of(vector_path),
        "label_threshold_w_per_min": LABEL_THRESHOLD,
        "sweep_grid": [SWEEP_MIN, SWEEP_MAX, SWEEP_STEP],
        "recall_floor": RECALL_FLOOR,
        "daylight_proxy": {"feature": FEATURE_NAMES[GH_INDEX], "min": GH_MIN, "note": "PROXY, not dataset is_daylight"},
        "consistency_check_vs_stored_report": check,
        "val": {"n": int(len(actual)), "actual_rate": float(actual.mean()),
                "mse_scaled": d["mse_scaled"], "mae_w": d["mae_w"], "rmse_w": d["rmse_w"],
                "threshold_free": tf, "threshold_free_daylight_proxy": tf_day,
                "at_label_threshold": eval_at(actual, score, LABEL_THRESHOLD)},
        "criterion_A": A, "criterion_B": B,
        "frozen": True,
    }
    (OUTPUT_DIR / f"threshold_{tag}_frozen.json").write_text(json.dumps(frozen, indent=2))
    print(f"\nSaved frozen thresholds -> {OUTPUT_DIR / f'threshold_{tag}_frozen.json'}")
    print(f"Elapsed {time.perf_counter()-t0:.1f}s.  Test has NOT been touched.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        th = [r["decision_threshold"] for r in rows]
        ax[0].plot(th, [r["precision"] for r in rows], label="precision")
        ax[0].plot(th, [r["recall"] for r in rows], label="recall")
        ax[0].plot(th, [r["f1"] for r in rows], label="F1")
        ax[0].axvline(LABEL_THRESHOLD, ls=":", c="gray", label="label thr (12.25)")
        if A["decision_threshold"] is not None:
            ax[0].axvline(A["decision_threshold"], ls="--", c="k", label="A: max F1")
        ax[0].set_xlabel("decision threshold on predicted max|diff| (W/min)"); ax[0].legend(); ax[0].set_title(f"{tag}: VALIDATION sweep")
        ax[1].plot([r["recall"] for r in rows], [r["precision"] for r in rows])
        ax[1].axhline(actual.mean(), ls=":", c="gray", label="prevalence")
        ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].legend(); ax[1].set_title("PR curve (val)")
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"threshold_{tag}_val_sweep.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("(plot skipped:", e, ")")


def phase_test(vector_path, tag):
    frozen_path = OUTPUT_DIR / f"threshold_{tag}_frozen.json"
    out_path = OUTPUT_DIR / f"threshold_{tag}_test_once_report.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Run --phase val first; frozen thresholds not found.")
    if out_path.exists():
        raise RuntimeError(f"{out_path.name} already exists. Test set was already evaluated for '{tag}'. "
                           "Refusing to run again (test-once rule). Use a new --tag only for a NEW, "
                           "pre-registered configuration.")
    fz = json.loads(frozen_path.read_text())
    if sha256_of(vector_path) != fz["vector_sha256"]:
        raise RuntimeError("Vector file differs from the one used to freeze thresholds.")
    if fz["label_threshold_w_per_min"] != LABEL_THRESHOLD:
        raise RuntimeError("Label threshold changed since freezing.")

    d = predict_scores(vector_path, "test")
    actual = d["act_max_ramp"] >= LABEL_THRESHOLD
    score = d["pred_score"]
    day = d["daylight_proxy"]

    pts = {"at_label_threshold_12.25 (reference, not tuned)": LABEL_THRESHOLD}
    if fz["criterion_A"]["decision_threshold"] is not None:
        pts["A_frozen_max_val_F1"] = fz["criterion_A"]["decision_threshold"]
    if fz["criterion_B"]["decision_threshold"] is not None:
        pts["B_frozen_recall_floor"] = fz["criterion_B"]["decision_threshold"]

    res = {
        "stage": "decision_threshold_analysis_TEST_ONCE", "tag": tag,
        "vector_sha256": fz["vector_sha256"],
        "n": int(len(actual)), "actual_rate": float(actual.mean()),
        "mse_scaled": d["mse_scaled"], "mae_w": d["mae_w"], "rmse_w": d["rmse_w"],
        "threshold_free": threshold_free(actual, score),
        "all_windows": {k: eval_at(actual, score, v) for k, v in pts.items()},
        "daylight_proxy": {
            "fraction": float(day.mean()),
            "actual_rate": float(actual[day].mean()) if day.any() else None,
            "threshold_free": threshold_free(actual[day], score[day]) if day.any() else None,
            "at_thresholds": {k: eval_at(actual[day], score[day], v) for k, v in pts.items()} if day.any() else None,
        },
        "note": "Thresholds were selected on validation only. Test evaluated once.",
    }
    out_path.write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUTPUT_DIR / f"threshold_{tag}_test_scores.npz",
                        pred_score=score, act_max_ramp=d["act_max_ramp"], daylight_proxy=day)
    print(json.dumps(res, indent=2))
    print(f"\nSaved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["val", "test"], required=True)
    ap.add_argument("--tag", default="p1")
    ap.add_argument("--vector", default=str(OUTPUT_DIR / "stage_p1_pso_best.npy"))
    a = ap.parse_args()
    vp = Path(a.vector)
    if not vp.exists():
        sys.exit(f"vector not found: {vp}")
    (phase_val if a.phase == "val" else phase_test)(vp, a.tag)


if __name__ == "__main__":
    main()