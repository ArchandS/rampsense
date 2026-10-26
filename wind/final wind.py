"""Step 7 (final wind table): frozen config = own inputs, lag 1, hidden 10, W1=0.7, W2=0.3, tau=0.05.
Models: Persistence, ARIMA (if statsmodels installed), LSTM-Adam-MSE, LSTM-Adam-phi_soft, LSTM-Adam-phi_soft + bias correction.
10 seeds, all 8 paper metrics + ramp-up / ramp-down MBE, Wilcoxon (Holm-corrected), timing, and a trade-off figure.
Needs: torch, numpy, pandas, scipy; optional: statsmodels, matplotlib. Put in the folder with WT15_2017_2018_correct.csv."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
CSV = HERE / "WT15_2017_2018_correct.csv"
EPS, RATED = 205.0, 2050.0
W1, W2, TAU, HIDDEN, EPOCHS = 0.7, 0.3, 0.05, 10, 60
SEEDS = list(range(10))

# ---------------------------------------------------------------- data (own inputs, lag 1)
df = pd.read_csv(CSV, index_col=0, parse_dates=True)[["wind_speed", "wind_direction", "power"]].dropna()
ok = (df.index[1:] - df.index[:-1]) == pd.Timedelta("10min")
idx = np.where(ok)[0]
V = df.values
X = V[idx][:, None, :]                                   # (n, 1, 3) lag 1
y_all, yprev_all = V[idx + 1, 2], V[idx, 2]
n = len(idx); a, b = int(.5 * n), int(.75 * n)           # chronological 50/25/25
mu, sd = X[:a].reshape(-1, 3).mean(0), X[:a].reshape(-1, 3).std(0)
X = (X - mu) / sd
ymu, ysd = y_all[:a].mean(), y_all[:a].std()
yS = (y_all - ymu) / ysd
print("samples", n, "| train/val/test", a, b - a, n - b)

T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
Xtr, Xva, Xte = T(X[:a]), T(X[a:b]), T(X[b:])
ytr, yva, yte = T(yS[:a]), T(yS[a:b]), T(yS[b:])
prev_tr, prev_va, prev_te = T(yprev_all[:a]), T(yprev_all[a:b]), T(yprev_all[b:])

# ---------------------------------------------------------------- model / loss / metrics
class LSTMNet(nn.Module):
    def __init__(s, nin, h=HIDDEN):
        super().__init__(); s.l = nn.LSTM(nin, h, batch_first=True); s.o = nn.Linear(h, 1)
    def forward(s, x): return s.o(s.l(x)[0][:, -1]).squeeze(-1)

def soft_f1(pred_s, true_s, prev_kw, tau=TAU):
    pk, tk = pred_s * ysd + ymu, true_s * ysd + ymu
    p = torch.sigmoid((torch.abs(pk - prev_kw) - EPS) / (tau * EPS))
    o = (torch.abs(tk - prev_kw) > EPS).float()
    tp, fp, fn = (p * o).sum(), (p * (1 - o)).sum(), ((1 - p) * o).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-8)

def metrics(pred_s, true_s, prev_kw):
    p = np.asarray(pred_s) * ysd + ymu
    t = np.asarray(true_s) * ysd + ymu
    pv = np.asarray(prev_kw)
    e = p - t
    ore, pre = np.abs(t - pv) > EPS, np.abs(p - pv) > EPS
    up, dn = (t - pv) > EPS, (t - pv) < -EPS
    tp, fp, fn, tn = (pre & ore).sum(), (pre & ~ore).sum(), (~pre & ore).sum(), (~pre & ~ore).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(MAE=100 * np.abs(e).mean() / RATED, RMSE=100 * np.sqrt((e ** 2).mean()) / RATED,
                MAAPE=np.arctan(np.abs(e) / np.maximum(np.abs(t), 1e-6)).mean(), MBE=100 * e.mean() / RATED,
                F1=100 * 2 * pr * rc / max(pr + rc, 1e-8), Recall=100 * rc, Precision=100 * pr,
                Accuracy=100 * (tp + tn) / len(t),
                MBE_rampup=100 * e[up].mean() / RATED, MBE_rampdown=100 * e[dn].mean() / RATED)

def train(loss_name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = LSTMNet(3); opt = torch.optim.Adam(m.parameters(), lr=1e-2); best, bs = 1e9, None
    def L(out, yy, pv):
        mse = ((out - yy) ** 2).mean()
        return mse if loss_name == "mse" else W1 * mse + W2 * (1 - soft_f1(out, yy, pv))
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 1024):
            j = perm[i:i + 1024]; loss = L(m(Xtr[j]), ytr[j], prev_tr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = L(m(Xva), yva, prev_va).item()
        if vl < best: best, bs = vl, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    return m, time.perf_counter() - t0

# ---------------------------------------------------------------- run models
rows = []
def add(name, seed, r, sec=np.nan): rows.append(dict(model=name, seed=seed, train_sec=sec, **r))

add("Persistence", 0, metrics(((prev_te.numpy() - ymu) / ysd), yte.numpy(), prev_te.numpy()))

try:                                                      # ARIMA baseline (optional)
    from statsmodels.tsa.arima.model import ARIMA
    ytrain = y_all[:a][-5000:]; best_aic, best_res = 1e18, None
    for p in range(3):
        for q in range(3):
            try:
                res = ARIMA(ytrain, order=(p, 1, q)).fit()
                if res.aic < best_aic: best_aic, best_res, order = res.aic, res, (p, 1, q)
            except Exception:
                pass
    fitted = best_res.apply(y_all[b:], refit=False).fittedvalues       # one-step-ahead, parameters fixed
    add("ARIMA", 0, metrics((fitted - ymu) / ysd, yte.numpy(), prev_te.numpy()))
    print("ARIMA order", order)
except ImportError:
    print("statsmodels not installed -> ARIMA skipped (pip install statsmodels)")

for name, loss in [("LSTM-Adam-MSE", "mse"), ("LSTM-Adam-phi_soft", "phi_soft")]:
    for s in SEEDS:
        m, sec = train(loss, s)
        with torch.no_grad():
            po, vo = m(Xte).numpy(), m(Xva).numpy()
        add(name, s, metrics(po, yte.numpy(), prev_te.numpy()), sec)
        if loss == "phi_soft":                            # post-hoc bias correction using VALIDATION mean error only
            shift = (vo - yva.numpy()).mean()
            add("LSTM-Adam-phi_soft+biasfix", s, metrics(po - shift, yte.numpy(), prev_te.numpy()), sec)
        print(f"{name:20s} seed {s}: F1={rows[-1]['F1']:.2f} RMSE={rows[-1]['RMSE']:.2f} MBE={rows[-1]['MBE']:.2f} ({sec:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(HERE / "final_wind_per_seed.csv", index=False)

# ---------------------------------------------------------------- tables
cols = ["MAE", "RMSE", "MAAPE", "MBE", "F1", "Recall", "Precision", "Accuracy"]
order_ = ["Persistence", "ARIMA", "LSTM-Adam-MSE", "LSTM-Adam-phi_soft", "LSTM-Adam-phi_soft+biasfix"]
g = res.groupby("model")[cols + ["MBE_rampup", "MBE_rampdown", "train_sec"]].agg(["mean", "std"]).reindex([o for o in order_ if o in res.model.values])
fmt = lambda c: g[c]["mean"].round(3).astype(str) + " (±" + g[c]["std"].fillna(0).round(3).astype(str) + ")"
table = pd.DataFrame({c: fmt(c) for c in cols})
pd.set_option("display.width", 250)
print("\n=== FINAL WIND TABLE (mean ± std over 10 seeds, test set) ===")
print(table.to_string())
print("\n=== Bias split (mean ± std) and training time ===")
print(pd.DataFrame({c: fmt(c) for c in ["MBE_rampup", "MBE_rampdown", "train_sec"]}).to_string())
table.to_csv(HERE / "final_wind_table.csv")

# ---------------------------------------------------------------- Wilcoxon with Holm correction
pairs = [("LSTM-Adam-phi_soft", "LSTM-Adam-MSE")]
tests = []
for A, B in pairs:
    xa, xb = res[res.model == A].sort_values("seed"), res[res.model == B].sort_values("seed")
    for met in ["RMSE", "MAAPE", "F1", "Recall", "Precision"]:
        try: p = wilcoxon(xa[met].values, xb[met].values).pvalue
        except ValueError: p = float("nan")
        tests.append((f"{A} vs {B}", met, p))
ps = np.array([t[2] for t in tests]); order = np.argsort(ps); holm = np.empty_like(ps); run = 0
for rank, i in enumerate(order):
    run = max(run, min(1, (len(ps) - rank) * ps[i])); holm[i] = run
print("\n=== Wilcoxon (two-sided, paired by seed), Holm-adjusted ===")
for (name, met, p), h in zip(tests, holm):
    print(f"{name:45s} {met:9s} p={p:.4f}  Holm p={h:.4f}  {'significant' if h < 0.05 else 'n.s.'}")

# ---------------------------------------------------------------- trade-off figure
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sw = HERE / "sweep_summary.csv"
    if sw.exists():
        s_ = pd.read_csv(sw); s_ = s_[(s_.sweep == "weights") & (s_.w1 >= 0.1)]
        ax.plot(s_.test_RMSE, s_.test_F1, "o-", color="tab:blue", label="W1 sweep (0.1 to 1.0)")
        for _, r in s_[s_.w1.isin([0.1, 0.5, 0.7, 0.8, 0.9, 1.0])].iterrows():
            ax.annotate(f"W1={r.w1:.1f}", (r.test_RMSE, r.test_F1), textcoords="offset points", xytext=(5, 4), fontsize=7)
    gs = HERE / "grid_summary.csv"
    if gs.exists():
        g_ = pd.read_csv(gs); ax.scatter(g_.test_RMSE, g_.test_F1, c="lightgray", edgecolors="gray", s=25, label="lag/hidden/feature grid")
    for nm, c in [("LSTM-Adam-MSE", "tab:red"), ("LSTM-Adam-phi_soft", "tab:green"), ("Persistence", "black")]:
        r = res[res.model == nm]; ax.scatter(r.RMSE.mean(), r.F1.mean(), c=c, s=70, marker="*", label=nm, zorder=5)
    ax.set_xlim(7, 12.5); ax.set_xlabel("Test RMSE (% of rated power)"); ax.set_ylabel("Test F1 (%)")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(HERE / "final_wind_tradeoff.png", dpi=200); print("Saved final_wind_tradeoff.png")
except ImportError:
    print("matplotlib not installed -> plot skipped")