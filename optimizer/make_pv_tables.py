"""
Aggregate every PV result into final tables (CSV + Markdown). Reads only saved JSONs; runs nothing heavy.
Sources: outputs\\threshold_<tag>_frozen.json (val), threshold_<tag>_test_once_report.json (test),
         baselines_frozen.json, baselines_test_once_report.json.
Seeds of the same configuration (tags ending _s<N>) are summarised as mean +/- std.
Outputs: outputs\\pv_results_per_run.csv, pv_results_summary.md
"""
from pathlib import Path
import csv
import json
import re

import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"
COLS = ["val_mse_scaled", "val_mae_w", "val_f1_at_12.25", "val_f1_at_A", "A_threshold", "val_AP", "val_AUC",
        "val_day_AUC", "test_mse_scaled", "test_mae_w", "test_f1_at_12.25", "test_f1_at_A", "test_AP",
        "test_AUC", "test_day_AUC"]


def g(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def collect():
    rows = []
    for fz in sorted(OUT.glob("threshold_*_frozen.json")):
        tag = fz.name[len("threshold_"):-len("_frozen.json")]
        j = json.loads(fz.read_text())
        r = {"tag": tag,
             "val_mse_scaled": g(j, "val", "mse_scaled"), "val_mae_w": g(j, "val", "mae_w"),
             "val_f1_at_12.25": g(j, "val", "at_label_threshold", "f1"),
             "val_f1_at_A": g(j, "criterion_A", "val_f1"), "A_threshold": g(j, "criterion_A", "decision_threshold"),
             "val_AP": g(j, "val", "threshold_free", "average_precision"),
             "val_AUC": g(j, "val", "threshold_free", "roc_auc"),
             "val_day_AUC": g(j, "val", "threshold_free_daylight_proxy", "roc_auc")}
        tp = OUT / f"threshold_{tag}_test_once_report.json"
        if tp.exists():
            t = json.loads(tp.read_text())
            r.update({"test_mse_scaled": t.get("mse_scaled"), "test_mae_w": t.get("mae_w"),
                      "test_f1_at_12.25": g(t, "all_windows", "at_label_threshold_12.25 (reference, not tuned)", "f1"),
                      "test_f1_at_A": g(t, "all_windows", "A_frozen_max_val_F1", "f1"),
                      "test_AP": g(t, "threshold_free", "average_precision"),
                      "test_AUC": g(t, "threshold_free", "roc_auc"),
                      "test_day_AUC": g(t, "daylight_proxy", "threshold_free", "roc_auc")})
        rows.append(r)
    return rows


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def md_table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def main():
    rows = collect()
    if not rows:
        raise SystemExit("no threshold_*_frozen.json found")
    with open(OUT / "pv_results_per_run.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["tag"] + COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ["tag"] + COLS})

    groups = {}
    for r in rows:
        groups.setdefault(re.sub(r"_s\d+$", "", r["tag"]), []).append(r)

    show = ["val_mse_scaled", "val_f1_at_A", "val_AP", "val_AUC", "val_day_AUC",
            "test_mse_scaled", "test_mae_w", "test_f1_at_A", "test_AP", "test_AUC", "test_day_AUC"]
    body = []
    for name, rs in groups.items():
        cells = [f"{name} (n={len(rs)})"]
        for c in show:
            vals = [r[c] for r in rs if r.get(c) is not None]
            if not vals:
                cells.append("-")
            elif len(vals) == 1:
                cells.append(fmt(vals[0]))
            else:
                cells.append(f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
        body.append(cells)

    md = ["# PV results summary", "",
          "Ground-truth ramp threshold 12.25 W/min (fixed). 'F1 at A' = F1 at the decision threshold "
          "frozen on VALIDATION (max val F1). Daylight AUC uses a G_h>5 proxy.", "",
          md_table(["model"] + show, body), ""]

    bf, bt = OUT / "baselines_frozen.json", OUT / "baselines_test_once_report.json"
    if bf.exists():
        b = json.loads(bf.read_text())
        t = json.loads(bt.read_text()) if bt.exists() else {}
        p = b["B1_recent_ramp_frozen_params"]
        brow = [["flag all daylight (no model)", fmt(g(b, "B0_day_flag", "f1")), fmt(g(t, "B0_day_flag", "f1"))],
                [f"recent-ramp rule (K={p['k']} min, thr {p['threshold']})",
                 fmt(g(b, "B1_val_result", "f1")), fmt(g(t, "B1_recent_ramp", "f1"))]]
        md += ["## Parameter-light baselines (no network; detectors, not forecasters)", "",
               md_table(["baseline", "val F1", "test F1"], brow), ""]
    (OUT / "pv_results_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("\nSaved outputs\\pv_results_per_run.csv and outputs\\pv_results_summary.md")


if __name__ == "__main__":
    main()
