"""Step 3: 10-seed comparison on WT15 (paper Table III style) + Wilcoxon test + validation-tuned ramp threshold.
Run after 01 and 02 (same folder). Needs: torch, numpy, pandas, scipy."""
import importlib.util, os, sys
import numpy as np, pandas as pd, torch
from scipy.stats import wilcoxon

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m02", os.path.join(here, "02_model_wt15.py"))
m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)   # reuses data, LSTM, loss, train()

SEEDS = list(range(10))
METHODS = ["mse", "phi_soft"]          # LSTM-Adam-MSE (paper baseline)  vs  LSTM-Adam-phi(soft-F1) (our improvement)
EPOCHS = 60
EPS, RATED = m02.EPS, m02.RATED
ymu, ysd = m02.ymu, m02.ysd
a, b = m02.a, m02.b
prev_va = m02.T(m02.yprev[a:b])


def ramp_metrics(pred_s, true_s, prev_kw, k=1.0):
    """k scales the predicted-ramp threshold: PRE if |y_hat - y_prev| > k*EPS. k=1 is the paper's definition."""
    p = pred_s.numpy() * ysd + ymu
    t = true_s.numpy() * ysd + ymu
    pv = prev_kw.numpy()
    e = p - t
    ore = np.abs(t - pv) > EPS
    pre = np.abs(p - pv) > k * EPS
    tp, fp = (pre & ore).sum(), (pre & ~ore).sum()
    fn, tn = (~pre & ore).sum(), (~pre & ~ore).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(
        MAE=100 * np.abs(e).mean() / RATED,
        RMSE=100 * np.sqrt((e ** 2).mean()) / RATED,
        MBE=100 * e.mean() / RATED,
        MAAPE=np.arctan(np.abs(e) / np.maximum(np.abs(t), 1e-6)).mean(),
        F1=100 * 2 * pr * rc / max(pr + rc, 1e-8),
        Recall=100 * rc, Precision=100 * pr,
        Accuracy=100 * (tp + tn) / len(t),
    )


def best_k(model):
    """Pick the threshold multiplier that maximises F1 on the VALIDATION set (no test leakage)."""
    with torch.no_grad():
        vo = model(m02.Xva)
    ks = np.arange(0.3, 1.51, 0.05)
    f1s = [ramp_metrics(vo, m02.yva, prev_va, k)["F1"] for k in ks]
    return float(ks[int(np.argmax(f1s))])


rows = []
for method in METHODS:
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        model = m02.train(method, epochs=EPOCHS)
        model.eval()
        with torch.no_grad():
            out = model(m02.Xte)
        r1 = ramp_metrics(out, m02.yte, m02.prev_te, 1.0)
        k = best_k(model)
        rk = ramp_metrics(out, m02.yte, m02.prev_te, k)
        rows.append(dict(method=method, seed=s, k=1.0, **r1))
        rows.append(dict(method=method + "+kTuned", seed=s, k=k, **rk))
        print(f"{method:9s} seed {s}: F1={r1['F1']:.2f}  RMSE={r1['RMSE']:.2f}  | tuned k={k:.2f} F1={rk['F1']:.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(here, "results_per_seed.csv"), index=False)

# ---- persistence reference (deterministic) ----
with torch.no_grad():
    pers = ramp_metrics((m02.prev_te - ymu) / ysd, m02.yte, m02.prev_te)
print("\nPersistence:", {k: round(float(v), 3) for k, v in pers.items()})

# ---- summary table: mean (std) over seeds ----
cols = ["MAE", "RMSE", "MAAPE", "MBE", "F1", "Recall", "Precision", "Accuracy"]
summ = res.groupby("method")[cols].agg(["mean", "std"])
table = pd.DataFrame({c: summ[c]["mean"].round(3).astype(str) + " (±" + summ[c]["std"].round(3).astype(str) + ")" for c in cols})
print("\n=== Mean (std) over", len(SEEDS), "seeds ===")
print(table.to_string())
table.to_csv(os.path.join(here, "results_summary.csv"))

# ---- Wilcoxon signed-rank (paired by seed), two-sided, alpha=0.05 ----
print("\n=== Wilcoxon: phi_soft vs mse ===")
for met in ["RMSE", "MAAPE", "Precision", "Recall", "F1"]:
    x = res[res.method == "phi_soft"].sort_values("seed")[met].values
    y = res[res.method == "mse"].sort_values("seed")[met].values
    try:
        p = wilcoxon(x, y).pvalue
    except ValueError:
        p = float("nan")
    print(f"{met:10s} p = {p:.4f}  {'significant' if p < 0.05 else 'n.s.'}")