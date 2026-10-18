"""
RampSense PV - PSO on the combined objective (0.75*MSE_scaled + 0.25*(1-F1)) with the unmodified
Stage-B fitness (fitness.evaluate_fitness, fixed 10,000-sample subset).

Modes
  --probe                     landscape probe around the Adam vector (no optimization)
  --init adam   (default)     WARM START: particle 0 = Adam H=16 vector, others = Adam + N(0, init_noise)
                              [OUR ADAPTATION - the paper's start point is UNKNOWN]
  --init random               CONTROL: all particles ~ N(0, 0.1) (same budget)  [ENGINEERING]

PSO constants: inertia 0.7298, c1 = c2 = 1.49618 (Clerc-Kennedy constriction values,
[ENGINEERING] - NOT disclosed in the paper). Velocity clip and init noise are engineering knobs.
No position bounds. Particle 0 of a warm start is the Adam vector, so best objective can never
be worse than the starting point.

Examples (cmd):
  python pso_pv.py --probe
  python pso_pv.py --init adam --seed 42 --evals 5000
  python pso_pv.py --init random --seed 42 --evals 5000
Outputs (outputs\\): stage_<tag>_best.npy, stage_<tag>_history.csv, stage_<tag>_report.json
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import fitness  # noqa: E402  (Stage B, unmodified; prints its banner on import)

D = 1608
ADAM_PATH = OUT / "stage_adam_h16_s42_best.npy"
W, C1, C2 = 0.7298, 1.49618, 1.49618


def f(vec):
    r = fitness.evaluate_fitness(np.ascontiguousarray(vec, dtype=np.float64))
    o = float(r["objective"])
    if not np.isfinite(o):
        raise RuntimeError("non-finite objective")
    return o, r


def load_adam():
    v = np.asarray(np.load(ADAM_PATH), dtype=np.float64)
    if v.shape != (D,) or not np.all(np.isfinite(v)):
        raise ValueError("bad Adam vector")
    return v


def probe(args):
    base = load_adam()
    fb, rb = f(base)
    rng = np.random.default_rng(args.seed)
    print(f"\nAdam vector objective {fb:.6f} | F1 {rb['f1']:.4f} | predicted ramps {rb['predicted_ramps']}")
    print(f"{'scale':>7} | {'median dObj':>12} | {'min dObj':>10} | {'frac F1>0':>9} | {'median pred%':>12} | {'best F1':>7}")
    rows = []
    for s in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        d, f1, pr = [], [], []
        for _ in range(args.probe_n):
            o, r = f(base + rng.normal(0.0, s, D))
            d.append(o - fb); f1.append(r["f1"]); pr.append(r["predicted_ramp_rate"])
        d, f1, pr = np.array(d), np.array(f1), np.array(pr)
        row = {"scale": s, "median_dObj": float(np.median(d)), "min_dObj": float(d.min()),
               "frac_f1_positive": float((f1 > 0).mean()), "median_pred_rate": float(np.median(pr)),
               "best_f1": float(f1.max())}
        rows.append(row)
        print(f"{s:7.3f} | {row['median_dObj']:12.5f} | {row['min_dObj']:10.5f} | "
              f"{row['frac_f1_positive']:9.2f} | {100*row['median_pred_rate']:11.2f}% | {row['best_f1']:7.3f}")
    (OUT / "stage_probe_report.json").write_text(json.dumps(
        {"base_objective": fb, "base_f1": rb["f1"], "n_per_scale": args.probe_n, "rows": rows}, indent=2))
    print("\nReading: pick init_noise near the smallest scale where frac F1>0 becomes non-trivial "
          "while median dObj is still small. Saved outputs\\stage_probe_report.json")


def run_pso(args):
    tag = args.tag or f"pso_{args.init}_s{args.seed}"
    N = args.particles
    rng = np.random.default_rng(args.seed)
    if args.init == "adam":
        base = load_adam()
        pos = base + rng.normal(0.0, args.init_noise, (N, D))
        pos[0] = base
        vel = rng.normal(0.0, args.init_noise * 0.1, (N, D))
        vel[0] = 0.0
        vmax = args.vmax if args.vmax is not None else 0.02
    else:
        pos = rng.normal(0.0, 0.1, (N, D))
        vel = rng.normal(0.0, 0.01, (N, D))
        vmax = args.vmax if args.vmax is not None else 0.05

    pbest, pbest_val = pos.copy(), np.full(N, np.inf)
    gbest, gval, gres = None, np.inf, None
    evals, it, hist = 0, 0, []
    first_obj = None
    t0 = time.perf_counter()
    print(f"\nPSO tag={tag} init={args.init} N={N} budget={args.evals} vmax={vmax} "
          f"init_noise={args.init_noise} seed={args.seed}")
    while evals < args.evals:
        it += 1
        for i in range(N):
            if evals >= args.evals:
                break
            o, r = f(pos[i])
            evals += 1
            if first_obj is None:
                first_obj = o                      # warm start: Adam vector (particle 0)
            if o < pbest_val[i]:
                pbest_val[i], pbest[i] = o, pos[i].copy()
            if o < gval:
                gval, gbest, gres = o, pos[i].copy(), r
        hist.append({"iteration": it, "evals": evals, "gbest_objective": gval,
                     "gbest_mse_scaled": gres["mse_scaled"], "gbest_f1": gres["f1"],
                     "swarm_median_objective": float(np.median(pbest_val[np.isfinite(pbest_val)]))})
        if it == 1 or it % 10 == 0:
            print(f"it {it:4d} | evals {evals:5d} | best {gval:.5f} | mse {gres['mse_scaled']:.4f} | "
                  f"F1 {gres['f1']:.4f} | {time.perf_counter()-t0:.0f}s")
        if evals >= args.evals:
            break
        r1, r2 = rng.random((N, D)), rng.random((N, D))
        vel = W * vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = pos + vel
        if it % 25 == 0:
            np.save(OUT / f"stage_{tag}_checkpoint_best.npy", gbest)

    np.save(OUT / f"stage_{tag}_best.npy", gbest)
    with open(OUT / f"stage_{tag}_history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)
    rep = {"stage": "PV_PSO", "tag": tag, "init": args.init, "seed": args.seed, "particles": N,
           "evaluation_budget": args.evals, "actual_evaluations": evals,
           "pso": {"inertia": W, "c1": C1, "c2": C2, "vmax": vmax, "init_noise": args.init_noise,
                   "note": "constants are [ENGINEERING], not paper values"},
           "initial_objective_particle0_or_first_eval": first_obj, "best_objective": gval,
           "improvement_vs_first": (first_obj - gval) if first_obj is not None else None,
           "best_metrics_subset": {k: gres[k] for k in ("objective", "mse_scaled", "f1", "tp", "fp", "fn", "tn",
                                                        "predicted_ramp_rate", "mae_w", "rmse_w")},
           "elapsed_seconds": time.perf_counter() - t0}
    (OUT / f"stage_{tag}_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["best_metrics_subset"], indent=2))
    print(f"initial {first_obj:.5f} -> best {gval:.5f}  saved stage_{tag}_best.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-n", type=int, default=30)
    ap.add_argument("--init", choices=["adam", "random"], default="adam")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evals", type=int, default=5000)
    ap.add_argument("--particles", type=int, default=20)
    ap.add_argument("--init-noise", type=float, default=0.02)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--tag", type=str, default=None)
    a = ap.parse_args()
    probe(a) if a.probe else run_pso(a)
