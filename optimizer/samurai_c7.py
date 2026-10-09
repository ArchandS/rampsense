"""
Stage C7 - ACTUAL PROPOSAL CALIBRATION (diagnostic only; no optimization).

Measures the real Delta = f(x+step) - f(x) distribution for the C6-style proposal
    step = FIXED_MOVE_SCALE * (L @ z),   L = chol(cov_init),  z ~ N(0, I)
from ONE fixed point, without changing optimizer state (no cov/temperature adaptation).

Because the exact C6 covariance initialization was not available to this script,
several conventions are tested side-by-side with common random numbers:
    identity   cov = 1.0 * I            (L = I)
    std2       cov = param_std^2 * I
    eps        cov = 1e-6 * I           (paper's epsilon regulariser as the whole cov)
    diagvar=V  cov = V * I              (custom)
Use --primary to choose which mode feeds the canonical output files.

Labels: [PAPER] structure only; every numeric constant here is an
[ENGINEERING DECISION]/diagnostic, not a paper value.
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from fitness import evaluate_fitness  # noqa: E402  (Stage B, unmodified)

D_EXPECTED = 1608
C6_INIT_OBJ = 0.957063972950      # from your C5/C6 logs
C6_PARAM_STD = 0.0010019844 / 0.01
C4_BASELINE_OBJ = 1.0394908190    # from your C4 log
T_GRID = np.array([1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6,
                   1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
TARGETS = (0.20, 0.30, 0.40)


# --------------------------------------------------------------------------
def fail(msg):
    print(f"\n[FATAL] {msg}")
    sys.exit(1)


def f_eval(vec):
    """Evaluate fitness; returns (objective, result_dict). Fails loudly on NaN/inf."""
    vec = np.ascontiguousarray(vec, dtype=np.float64)
    if vec.shape != (D_EXPECTED,):
        fail(f"vector shape {vec.shape} != ({D_EXPECTED},)")
    if not np.all(np.isfinite(vec)):
        fail("proposal contains NaN/inf")
    res = evaluate_fitness(vec.copy())
    o = float(res["objective"])
    if not np.isfinite(o):
        fail(f"objective is not finite: {o}")
    return o, res


def pick(res, keys):
    for k in keys:
        if k in res:
            try:
                return float(res[k])
            except Exception:
                return None
    return None


def find_start_vector(args):
    """Try to reproduce C6's start vector by matching its logged initial objective."""
    if args.start_npy:
        v = np.load(args.start_npy).astype(np.float64).ravel()
        o, _ = f_eval(v)
        return v, "start_npy", o, abs(o - C6_INIT_OBJ) < 1e-8, []

    cands = {}

    def torch_default():
        import torch
        from model import PVLSTM
        from param_vector import get_vector
        torch.manual_seed(args.seed)
        return get_vector(PVLSTM())
    cands["torch_default_init_seed42"] = torch_default
    cands["np_default_rng_normal0.1"] = lambda: np.random.default_rng(args.seed).normal(0, 0.1, D_EXPECTED)
    cands["np_RandomState_normal0.1"] = lambda: np.random.RandomState(args.seed).normal(0, 0.1, D_EXPECTED)

    def torch_randn():
        import torch
        torch.manual_seed(args.seed)
        return torch.randn(D_EXPECTED).numpy().astype(np.float64) * 0.1
    cands["torch_randn*0.1"] = torch_randn

    table, match = [], None
    for name, fn in cands.items():
        try:
            v = np.asarray(fn(), dtype=np.float64).ravel()
            if v.shape != (D_EXPECTED,):
                continue
            o, _ = f_eval(v)
        except Exception as e:  # candidate failed; keep going
            print(f"   candidate {name}: failed ({e})")
            continue
        ok = abs(o - C6_INIT_OBJ) < 1e-8
        table.append({"name": name, "std": float(v.std()), "objective": o, "matches_c6": bool(ok)})
        print(f"   candidate {name:28s} std={v.std():.10f} obj={o:.12f} "
              f"{'<== MATCHES C6' if ok else ''}"
              f"{'  (matches C4 baseline)' if abs(o - C4_BASELINE_OBJ) < 1e-8 else ''}")
        if ok and match is None:
            match = (v, name, o)
    if match:
        return match[0], match[1], match[2], True, table
    # fallback: candidate whose std is closest to C6's implied std
    best = min(table, key=lambda r: abs(r["std"] - C6_PARAM_STD))
    print("\n[WARNING] No candidate reproduced C6's initial objective 0.957063972950.")
    print(f"          Falling back to '{best['name']}' (std closest to C6's implied {C6_PARAM_STD:.8f}).")
    print("          Results describe THIS start point, not necessarily C6's. Use --start-npy to supply C6's vector.")
    v = cands[best["name"]]()
    v = np.asarray(v, dtype=np.float64).ravel()
    o, _ = f_eval(v)
    return v, best["name"] + " (FALLBACK, not matching C6)", o, False, table


def make_L(var, d):
    cov = np.eye(d, dtype=np.float64) * var
    if not np.all(np.isfinite(cov)):
        fail("covariance contains NaN/inf")
    L = np.linalg.cholesky(cov)
    if not np.all(np.isfinite(L)):
        fail("Cholesky factor contains NaN/inf")
    return cov, L


def expected_acc(pos, T):
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))
    if pos.size == 0:
        return np.full(T.shape, np.nan)
    return np.exp(-pos[None, :] / T[:, None]).mean(axis=1)


def solve_T(pos, target):
    """T such that mean(exp(-pos/T)) == target (monotone in T). Log-space bisection."""
    if pos.size < 5:
        return None
    lo, hi = np.log(1e-14), np.log(1e3)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected_acc(pos, np.exp(mid))[0] < target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def summarize_delta(d):
    return {"mean": float(d.mean()), "median": float(np.median(d)), "std": float(d.std()),
            "min": float(d.min()), "max": float(d.max())}


def summarize_pos(p):
    if p.size == 0:
        return {"count": 0}
    q = np.percentile(p, [50, 75, 90, 95, 99])
    return {"count": int(p.size), "median": float(q[0]), "p75": float(q[1]), "p90": float(q[2]),
            "p95": float(q[3]), "p99": float(q[4]), "max": float(p.max())}


# --------------------------------------------------------------------------
def run_mode(mode, var, x0, f0, res0, Z, scale, T0, param_std, rng_acc):
    d = x0.size
    cov, L = make_L(var, d)
    diag_cov = np.diag(cov)
    offmax = float(np.abs(cov - np.diag(diag_cov)).max())
    steps = scale * (Z @ L.T)                       # (n, d)  == scale * (L @ z) per row
    if not np.all(np.isfinite(steps)):
        fail("proposal step contains NaN/inf")

    norms = np.linalg.norm(steps, axis=1)
    mean_abs = np.abs(steps).mean(axis=1)           # per-proposal mean |dx|
    max_abs = np.abs(steps).max(axis=1)             # per-proposal max |dx|
    step_rms = float(np.sqrt((steps ** 2).mean()))  # actual per-coordinate step std

    n = steps.shape[0]
    f_prop = np.empty(n)
    f1_prop = np.full(n, np.nan)
    f1_0 = pick(res0, ("f1", "F1", "f1_score"))
    t0 = time.perf_counter()
    for i in range(n):
        o, r = f_eval(x0 + steps[i])
        f_prop[i] = o
        v = pick(r, ("f1", "F1", "f1_score"))
        if v is not None:
            f1_prop[i] = v
        if (i + 1) % 50 == 0:
            print(f"   [{mode}] {i + 1}/{n} proposals  ({time.perf_counter() - t0:.1f}s)")
    delta = f_prop - f0
    if not np.all(np.isfinite(delta)):
        fail("delta contains NaN/inf")

    pos = delta[delta > 0]
    neg = delta[delta < 0]
    n_zero = int((delta == 0).sum())
    pos_frac, neg_frac, zero_frac = pos.size / n, neg.size / n, n_zero / n

    # hypothetical Metropolis decision at T0 (state is NOT changed)
    p_acc = np.where(delta <= 0, 1.0, np.exp(-np.maximum(delta, 0) / T0))
    u = rng_acc.random(n)
    hyp_accept = u < p_acc
    up = delta > 0
    hyp_uphill_acc = float(hyp_accept[up].mean()) if up.any() else float("nan")

    A = expected_acc(pos, T_GRID)
    table = np.column_stack([A, pos_frac * A, pos_frac * A + neg_frac + zero_frac])
    rec = {f"T_for_{int(t * 100)}pct_uphill": solve_T(pos, t) for t in TARGETS}

    f1_changed = None
    if f1_0 is not None and np.isfinite(f1_prop).any():
        f1_changed = int(np.sum(np.abs(f1_prop - f1_0) > 1e-12))

    report = {
        "mode": mode, "cov_variance": float(var), "scale": scale,
        "cov": {"diag_mean": float(diag_cov.mean()), "diag_min": float(diag_cov.min()),
                "diag_max": float(diag_cov.max()), "max_offdiag": offmax},
        "chol": {"diag_mean": float(np.diag(L).mean()), "diag_max": float(np.diag(L).max())},
        "param_std_start": float(param_std),
        "nominal_FIXED_MOVE_SCALE": scale,
        "actual_step_rms_per_coord": step_rms,
        "actual_step_rms_over_nominal_scale": step_rms / scale,
        "actual_step_rms_over_param_std": step_rms / param_std,
        "proposal_norm": {"mean": float(norms.mean()), "median": float(np.median(norms)),
                          "p95": float(np.percentile(norms, 95))},
        "mean_abs_param_change": float(mean_abs.mean()),
        "max_abs_param_change": float(max_abs.max()),
        "delta": summarize_delta(delta),
        "positive_fraction": pos_frac, "negative_fraction": neg_frac, "zero_fraction": zero_frac,
        "positive_delta": summarize_pos(pos),
        "T0_used_for_hypothetical_acceptance": T0,
        "mean_metropolis_prob_at_T0_uphill": float(p_acc[up].mean()) if up.any() else None,
        "hypothetical_uphill_acceptance_at_T0": hyp_uphill_acc,
        "f1_baseline": f1_0,
        "num_proposals_with_changed_f1": f1_changed,
        "temperature_grid": T_GRID.tolist(),
        "expected_uphill_acceptance_conditional": A.tolist(),
        "expected_uphill_acceptance_weighted_by_pos_fraction": (pos_frac * A).tolist(),
        "expected_overall_acceptance": table[:, 2].tolist(),
        "recommended_T": rec,
        "recommended_T_over_median_positive_delta":
            {k: (v / float(np.median(pos)) if (v and pos.size) else None) for k, v in rec.items()},
    }
    arrays = {"delta": delta, "norms": norms, "pos": pos, "table": table,
              "f_prop": f_prop, "p_acc_T0": p_acc, "hyp_accept": hyp_accept}
    return report, arrays


def print_mode(rp, f0):
    print("\n" + "-" * 60)
    print(f"MODE: {rp['mode']}   (cov variance = {rp['cov_variance']:.3e})")
    print("-" * 60)
    print(f"cov diag mean {rp['cov']['diag_mean']:.3e} | chol diag mean {rp['chol']['diag_mean']:.3e} "
          f"| max offdiag {rp['cov']['max_offdiag']:.1e}")
    print(f"nominal FIXED_MOVE_SCALE        : {rp['nominal_FIXED_MOVE_SCALE']:.10f}")
    print(f"ACTUAL per-coord step RMS       : {rp['actual_step_rms_per_coord']:.4e}  "
          f"(= {rp['actual_step_rms_over_nominal_scale']:.3e} x nominal, "
          f"{rp['actual_step_rms_over_param_std']:.3e} x param_std)")
    print("Proposal statistics:")
    print(f"    mean L2 norm              : {rp['proposal_norm']['mean']:.4e}")
    print(f"    median L2 norm            : {rp['proposal_norm']['median']:.4e}")
    print(f"    p95 L2 norm               : {rp['proposal_norm']['p95']:.4e}")
    print(f"    mean absolute param change: {rp['mean_abs_param_change']:.4e}")
    print(f"    max absolute param change : {rp['max_abs_param_change']:.4e}")
    d = rp["delta"]
    print("Delta statistics:")
    print(f"    mean {d['mean']:.4e} | median {d['median']:.4e} | std {d['std']:.4e}")
    print(f"    min  {d['min']:.4e} | max    {d['max']:.4e}")
    print(f"    positive fraction {rp['positive_fraction']:.4f} | negative {rp['negative_fraction']:.4f} "
          f"| zero {rp['zero_fraction']:.4f}")
    if rp["f1_baseline"] is not None:
        print(f"    baseline F1 {rp['f1_baseline']:.6f} | proposals with changed F1: {rp['num_proposals_with_changed_f1']}")
    p = rp["positive_delta"]
    print("Positive delta:")
    if p["count"] == 0:
        print("    none")
    else:
        print(f"    count {p['count']} | median {p['median']:.4e} | p75 {p['p75']:.4e} | p90 {p['p90']:.4e}")
        print(f"    p95 {p['p95']:.4e} | p99 {p['p99']:.4e} | max {p['max']:.4e}")
    print(f"Hypothetical uphill acceptance at T0={rp['T0_used_for_hypothetical_acceptance']:.4e}: "
          f"{rp['hypothetical_uphill_acceptance_at_T0']:.4f}  "
          f"(mean Metropolis prob {rp['mean_metropolis_prob_at_T0_uphill']})")
    print("\nTemperature table:")
    print(f"{'T':>10} | {'E[acc | uphill]':>16} | {'x uphill frac':>14} | {'overall acc':>12}")
    for i, T in enumerate(T_GRID):
        print(f"{T:10.1e} | {rp['expected_uphill_acceptance_conditional'][i]:16.4f} | "
              f"{rp['expected_uphill_acceptance_weighted_by_pos_fraction'][i]:14.4f} | "
              f"{rp['expected_overall_acceptance'][i]:12.4f}")
    for t in TARGETS:
        v = rp["recommended_T"][f"T_for_{int(t * 100)}pct_uphill"]
        print(f"Recommended T for ~{int(t * 100)}% uphill acceptance: "
              f"{('%.4e' % v) if v else 'n/a (too few positive deltas)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="proposals per covariance mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.0010019844)
    ap.add_argument("--T0", type=float, default=0.0002658506)
    ap.add_argument("--modes", type=str, default="identity,std2,eps",
                    help="comma list: identity,std2,eps,diagvar=<float>")
    ap.add_argument("--primary", type=str, default="identity")
    ap.add_argument("--start-npy", type=str, default=None)
    args = ap.parse_args()

    print("=" * 60)
    print("STAGE C7 — ACTUAL PROPOSAL CALIBRATION")
    print("=" * 60)

    print("\nLocating start vector (matching C6's logged initial objective):")
    x0, start_name, f0, matched, cand_table = find_start_vector(args)
    if x0.size != D_EXPECTED:
        fail(f"dimension {x0.size} != {D_EXPECTED}")
    _, res0 = f_eval(x0)
    param_std = float(x0.std())
    print(f"\nStart vector      : {start_name}")
    print(f"Matches C6 start  : {matched}")
    print(f"Initial objective : {f0:.12f}")
    print(f"Parameter std     : {param_std:.10f}   (C6-implied {C6_PARAM_STD:.10f}; C4 baseline std was 0.1447741929)")
    print(f"Number of proposals per mode: {args.n}   scale={args.scale}   seed={args.seed}")

    modes = []
    for m in [s.strip() for s in args.modes.split(",") if s.strip()]:
        if m == "identity":
            modes.append((m, 1.0))
        elif m == "std2":
            modes.append((m, param_std ** 2))
        elif m == "eps":
            modes.append((m, 1e-6))
        elif m.startswith("diagvar="):
            modes.append((m, float(m.split("=")[1])))
        else:
            fail(f"unknown mode '{m}'")
    if args.primary not in [m for m, _ in modes]:
        fail(f"--primary '{args.primary}' not among modes")

    # common random numbers: same z for every mode
    Z = np.random.default_rng(args.seed).standard_normal((args.n, D_EXPECTED))
    reports, store = {}, {}
    for idx, (mode, var) in enumerate(modes):
        print(f"\n>>> Running mode '{mode}' ...")
        rng_acc = np.random.default_rng(args.seed + 1000 + idx)
        rp, arr = run_mode(mode, var, x0, f0, res0, Z, args.scale, args.T0, param_std, rng_acc)
        reports[mode], store[mode] = rp, arr
        print_mode(rp, f0)

    # save raw data
    for mode, arr in store.items():
        np.save(OUT / f"stage_c7_delta_samples_{mode}.npy", arr["delta"])
        np.save(OUT / f"stage_c7_proposal_norms_{mode}.npy", arr["norms"])
        np.save(OUT / f"stage_c7_positive_delta_{mode}.npy", arr["pos"])
    P = store[args.primary]
    np.save(OUT / "stage_c7_delta_samples.npy", P["delta"])
    np.save(OUT / "stage_c7_proposal_norms.npy", P["norms"])
    np.save(OUT / "stage_c7_positive_delta.npy", P["pos"])
    np.save(OUT / "stage_c7_candidate_temperatures.npy", T_GRID)
    np.save(OUT / "stage_c7_expected_acceptance.npy", P["table"])  # cols: cond, x uphill frac, overall

    summary = []
    for mode, rp in reports.items():
        summary.append({"mode": mode, "step_rms": rp["actual_step_rms_per_coord"],
                        "pos_frac": rp["positive_fraction"],
                        "T20": rp["recommended_T"]["T_for_20pct_uphill"],
                        "T30": rp["recommended_T"]["T_for_30pct_uphill"],
                        "T40": rp["recommended_T"]["T_for_40pct_uphill"],
                        "hyp_acc_at_T0": rp["hypothetical_uphill_acceptance_at_T0"]})
    out = {
        "stage": "C7", "seed": args.seed, "n_proposals_per_mode": args.n,
        "scale": args.scale, "T0": args.T0, "dimension": D_EXPECTED,
        "start_vector": {"name": start_name, "matches_c6_initial_objective": bool(matched),
                         "initial_objective": f0, "param_std": param_std,
                         "candidates_tried": cand_table},
        "primary_mode": args.primary,
        "assumptions": [
            "C6 covariance initialization unknown to this script; several conventions tested with common random numbers.",
            "Proposal form assumed: step = FIXED_MOVE_SCALE * (L @ z), L = chol(cov).",
            "Temperatures/recommendations are diagnostic engineering values, NOT paper values.",
        ],
        "modes": reports, "mode_summary": summary,
    }
    (OUT / "stage_c7_calibration_report.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 60)
    print("MODE COMPARISON (same z, same start point)")
    print("=" * 60)
    print(f"{'mode':>10} | {'step RMS':>10} | {'pos frac':>8} | {'T20':>10} | {'T30':>10} | {'T40':>10} | {'acc@T0':>7}")
    for s in summary:
        f = lambda v: f"{v:10.3e}" if v else "       n/a"
        print(f"{s['mode']:>10} | {s['step_rms']:10.3e} | {s['pos_frac']:8.3f} | {f(s['T20'])} | "
              f"{f(s['T30'])} | {f(s['T40'])} | {s['hyp_acc_at_T0']:7.3f}")
    print(f"\nReference: your T0 = {args.T0:.4e}; C4 chose it from 'scale 0.01' (T30), "
          f"but C6's nominal step is {args.scale}.")
    print(f"Saved report -> {OUT / 'stage_c7_calibration_report.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
