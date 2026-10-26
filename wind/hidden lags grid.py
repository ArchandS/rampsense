"""Step 6: tune LSTM hidden size and lag, and test ramp-history features, for WT15 (phi_soft loss, W1=0.7, tau=0.05).
Feature sets: own | own+diff | own+neigh_ws | own+diff+neigh_ws   (diff = 1-step change of power and wind speed)
Stage A: grid over hidden x lag x featset with few seeds, chosen on VALIDATION only.
Stage B: best config per featset re-run with 10 seeds on the TEST set + Wilcoxon vs the best 'own' config.
Put in the same folder as WT11-15_2017_2018_merged.csv (created by 05_neighbours.py). Needs torch, numpy, pandas, scipy."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
MERGED = HERE / "WT11-15_2017_2018_merged.csv"
EPS, RATED = 205.0, 2050.0
W1, W2, TAU, EPOCHS = 0.7, 0.3, 0.05, 60
HIDDEN_GRID, LAG_GRID = [10, 20, 40], [1, 6]
GRID_SEEDS, FINAL_SEEDS = [0, 1, 2], list(range(10))
RMSE_TOL = 1.25                       # accept configs with val RMSE <= 1.25 x best val RMSE, then max val F1
MAXLAG = max(LAG_GRID)

# ---------------------------------------------------------------- data (same targets/samples for every config)
df = pd.read_csv(MERGED, index_col=0, parse_dates=True)
df["dP"] = df["power"].diff()
df["dws"] = df["wind_speed"].diff()
OWN = ["wind_speed", "wind_direction", "power"]
DIFF = ["dP", "dws"]
WS = [f"ws_{t}" for t in (11, 12, 13, 14)]
FEATSETS = {"own": OWN, "own+diff": OWN + DIFF, "own+neigh_ws": OWN + WS, "own+diff+neigh_ws": OWN + DIFF + WS}
df = df[OWN + DIFF + WS].dropna()

N = len(df)
ok = ((df.index[1:] - df.index[:-1]) == pd.Timedelta("10min")).astype(int)
okc = np.concatenate([[0], np.cumsum(ok)])
t_all = np.arange(MAXLAG - 1, N - 1)                                  # t = last input row, target = t+1
ts = t_all[(okc[t_all + 1] - okc[t_all - (MAXLAG - 1)]) == MAXLAG]    # contiguous window incl. target
n = len(ts); a, b = int(.5 * n), int(.75 * n)                         # chronological 50/25/25
power = df["power"].values
y_all, yprev_all = power[ts + 1], power[ts]
ymu, ysd = y_all[:a].mean(), y_all[:a].std()
yS = (y_all - ymu) / ysd
print("samples", n, "| train/val/test", a, b - a, n - b)

T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
ytr, yva, yte = T(yS[:a]), T(yS[a:b]), T(yS[b:])
prev_tr, prev_va, prev_te = T(yprev_all[:a]), T(yprev_all[a:b]), T(yprev_all[b:])

def make_split(cols, lag):
    V = df[cols].values
    X = np.stack([V[ts - k] for k in range(lag - 1, -1, -1)], axis=1)   # (n, lag, F)
    flat = X[:a].reshape(-1, X.shape[-1])
    X = (X - flat.mean(0)) / (flat.std(0) + 1e-8)                        # scaler fitted on train only
    return T(X[:a]), T(X[a:b]), T(X[b:])

# ---------------------------------------------------------------- model / loss / metrics
class LSTMNet(nn.Module):
    def __init__(s, nin, h):
        super().__init__(); s.l = nn.LSTM(nin, h, batch_first=True); s.o = nn.Linear(h, 1)
    def forward(s, x): return s.o(s.l(x)[0][:, -1]).squeeze(-1)

def soft_f1(pred_s, true_s, prev_kw, tau=TAU):
    pk, tk = pred_s * ysd + ymu, true_s * ysd + ymu
    p = torch.sigmoid((torch.abs(pk - prev_kw) - EPS) / (tau * EPS))
    o = (torch.abs(tk - prev_kw) > EPS).float()
    tp, fp, fn = (p * o).sum(), (p * (1 - o)).sum(), ((1 - p) * o).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-8)

def metrics(pred_s, true_s, prev_kw):
    p, t, pv = pred_s.numpy() * ysd + ymu, true_s.numpy() * ysd + ymu, prev_kw.numpy()
    e = p - t
    ore, pre = np.abs(t - pv) > EPS, np.abs(p - pv) > EPS
    tp, fp, fn = (pre & ore).sum(), (pre & ~ore).sum(), (~pre & ore).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(MAE=100 * np.abs(e).mean() / RATED, RMSE=100 * np.sqrt((e ** 2).mean()) / RATED,
                MBE=100 * e.mean() / RATED, F1=100 * 2 * pr * rc / max(pr + rc, 1e-8),
                Recall=100 * rc, Precision=100 * pr)

def train_eval(Xtr, Xva, Xte, hidden, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = LSTMNet(Xtr.shape[-1], hidden); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    best, bs = 1e9, None
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 1024):
            j = perm[i:i + 1024]; out = m(Xtr[j])
            loss = W1 * ((out - ytr[j]) ** 2).mean() + W2 * (1 - soft_f1(out, ytr[j], prev_tr[j]))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vo = m(Xva); vl = (W1 * ((vo - yva) ** 2).mean() + W2 * (1 - soft_f1(vo, yva, prev_va))).item()
        if vl < best: best, bs = vl, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        return metrics(m(Xva), yva, prev_va), metrics(m(Xte), yte, prev_te)

# ---------------------------------------------------------------- Stage A: grid (selection on validation)
rows = []
for fs, cols in FEATSETS.items():
    for lag in LAG_GRID:
        Xtr, Xva, Xte = make_split(cols, lag)
        for h in HIDDEN_GRID:
            for s in GRID_SEEDS:
                t0 = time.perf_counter()
                va, te = train_eval(Xtr, Xva, Xte, h, s)
                rows.append(dict(featset=fs, lag=lag, hidden=h, seed=s, sec=time.perf_counter() - t0,
                                 **{f"val_{k}": v for k, v in va.items()}, **{f"test_{k}": v for k, v in te.items()}))
                print(f"[grid] {fs:18s} lag={lag} h={h:2d} seed={s}  valF1={va['F1']:.2f} valRMSE={va['RMSE']:.2f}", flush=True)
grid = pd.DataFrame(rows)
grid.to_csv(HERE / "grid_per_seed.csv", index=False)
agg = grid.groupby(["featset", "lag", "hidden"]).mean(numeric_only=True).reset_index().drop(columns="seed")
agg.to_csv(HERE / "grid_summary.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== Grid (mean over seeds) ===")
print(agg[["featset", "lag", "hidden", "val_F1", "val_RMSE", "test_F1", "test_RMSE", "sec"]].round(2).to_string(index=False))

best_val_rmse = agg["val_RMSE"].min()
chosen = {}
for fs in FEATSETS:
    sub = agg[(agg.featset == fs) & (agg.val_RMSE <= RMSE_TOL * best_val_rmse)]
    r = sub.loc[sub["val_F1"].idxmax()]
    chosen[fs] = (int(r.lag), int(r.hidden))
print("\nChosen on validation:", chosen)

# ---------------------------------------------------------------- Stage B: 10 seeds on the chosen configs
final = []
for fs, (lag, h) in chosen.items():
    Xtr, Xva, Xte = make_split(FEATSETS[fs], lag)
    for s in FINAL_SEEDS:
        _, te = train_eval(Xtr, Xva, Xte, h, s)
        final.append(dict(featset=fs, lag=lag, hidden=h, seed=s, **te))
        print(f"[final] {fs:18s} lag={lag} h={h} seed={s}  F1={te['F1']:.2f} RMSE={te['RMSE']:.2f}", flush=True)
fin = pd.DataFrame(final)
fin.to_csv(HERE / "final_per_seed.csv", index=False)
cols = ["MAE", "RMSE", "MBE", "F1", "Recall", "Precision"]
g = fin.groupby("featset")[cols].agg(["mean", "std"])
table = pd.DataFrame({c: g[c]["mean"].round(3).astype(str) + " (±" + g[c]["std"].round(3).astype(str) + ")" for c in cols})
table.insert(0, "config(lag,hidden)", [str(chosen[i]) for i in table.index])
print("\n=== FINAL: mean (std) over", len(FINAL_SEEDS), "seeds, test set ===")
print(table.to_string())
table.to_csv(HERE / "final_summary.csv")

print("\n=== Wilcoxon vs best 'own' config (two-sided, paired by seed) ===")
base = fin[fin.featset == "own"].sort_values("seed")
for fs in FEATSETS:
    if fs == "own":
        continue
    cur = fin[fin.featset == fs].sort_values("seed")
    for met in ["F1", "Recall", "Precision", "RMSE"]:
        try:
            p = wilcoxon(cur[met].values, base[met].values).pvalue
        except ValueError:
            p = float("nan")
        print(f"{fs:18s} {met:9s} p = {p:.4f}  {'significant' if p < 0.05 else 'n.s.'}")