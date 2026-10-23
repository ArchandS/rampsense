"""Step 5: does adding neighbouring turbines (WT11-14) improve ramp detection for WT15?
Standalone: reads the two SCADA zips in this folder, builds a merged CSV, trains LSTM with phi_soft loss
(W1=0.7, tau=0.05, lag 1) for three input sets, 10 seeds each, and runs Wilcoxon vs the WT15-only baseline.
Needs: torch, numpy, pandas, scipy."""
import zipfile, glob, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
MERGED = HERE / "WT11-15_2017_2018_merged.csv"
EPS, RATED = 205.0, 2050.0
W1, W2, TAU = 0.7, 0.3, 0.05
EPOCHS, SEEDS, HIDDEN = 60, list(range(10)), 10
NEIGH = [11, 12, 13, 14]

# ---------------------------------------------------------------- 1) build merged dataset
def read_turbine(tid):
    parts = []
    for z in sorted(glob.glob(str(HERE / "*SCADA_201[78]*.zip"))):
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.startswith(f"Turbine_Data_Penmanshiel_{tid}_")][0]
            d = pd.read_csv(zf.open(name), skiprows=9, low_memory=False,
                            usecols=["# Date and time", "Wind speed (m/s)", "Wind direction (°)", "Power (kW)"])
        parts.append(d)
    d = pd.concat(parts)
    d["# Date and time"] = pd.to_datetime(d["# Date and time"])
    d = d.drop_duplicates("# Date and time").set_index("# Date and time").sort_index()
    d = d.reindex(pd.date_range("2017-01-01", "2018-12-31 23:50", freq="10min"))
    d = d.interpolate(limit=3, limit_area="inside")
    d["Power (kW)"] = d["Power (kW)"].clip(0, RATED)
    return d

if not MERGED.exists():
    t15 = read_turbine(15)
    df = pd.DataFrame({"wind_speed": t15["Wind speed (m/s)"], "wind_direction": t15["Wind direction (°)"],
                       "power": t15["Power (kW)"]})
    for tid in NEIGH:
        t = read_turbine(tid)
        df[f"ws_{tid}"] = t["Wind speed (m/s)"]
        df[f"pw_{tid}"] = t["Power (kW)"]
    df.index.name = "timestamp"
    df.to_csv(MERGED)
    print("saved", MERGED)
df = pd.read_csv(MERGED, index_col=0, parse_dates=True)

OWN = ["wind_speed", "wind_direction", "power"]
WS = [f"ws_{t}" for t in NEIGH]
PW = [f"pw_{t}" for t in NEIGH]
FEATSETS = {"own": OWN, "own+neigh_ws": OWN + WS, "own+neigh_ws+pw": OWN + WS + PW}

# same samples for every input set: drop rows where ANY column is missing
df = df[OWN + WS + PW].dropna()
ok = (df.index[1:] - df.index[:-1]) == pd.Timedelta("10min")       # lag 1: need t and t+1 contiguous
idx = np.where(ok)[0]
y_all = df["power"].values[idx + 1]
yprev_all = df["power"].values[idx]
n = len(idx); a, b = int(.5 * n), int(.75 * n)                      # chronological 50/25/25
ymu, ysd = y_all[:a].mean(), y_all[:a].std()
print("samples", n, "| train/val/test", a, b - a, n - b)

T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
yS = (y_all - ymu) / ysd

# ---------------------------------------------------------------- 2) model, loss, metrics
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
    p, t, pv = pred_s.numpy() * ysd + ymu, true_s.numpy() * ysd + ymu, prev_kw.numpy()
    e = p - t
    ore, pre = np.abs(t - pv) > EPS, np.abs(p - pv) > EPS
    tp, fp, fn = (pre & ore).sum(), (pre & ~ore).sum(), (~pre & ore).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(MAE=100 * np.abs(e).mean() / RATED, RMSE=100 * np.sqrt((e ** 2).mean()) / RATED,
                MBE=100 * e.mean() / RATED, F1=100 * 2 * pr * rc / max(pr + rc, 1e-8),
                Recall=100 * rc, Precision=100 * pr)

def make_split(cols):
    X = df[cols].values[idx]                                        # (n, F), lag 1
    mu, sd = X[:a].mean(0), X[:a].std(0) + 1e-8                     # scaler fitted on train only
    X = ((X - mu) / sd)[:, None, :]
    return T(X[:a]), T(X[a:b]), T(X[b:])

ytr, yva, yte = T(yS[:a]), T(yS[a:b]), T(yS[b:])
prev_tr, prev_va, prev_te = T(yprev_all[:a]), T(yprev_all[a:b]), T(yprev_all[b:])

def train_eval(Xtr, Xva, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = LSTMNet(Xtr.shape[-1]); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
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
        return metrics(m(Xte), yte, prev_te)

# ---------------------------------------------------------------- 3) run experiment
rows = []
for name, cols in FEATSETS.items():
    Xtr, Xva, Xte = make_split(cols)
    for s in SEEDS:
        t0 = time.perf_counter()
        r = train_eval(Xtr, Xva, Xte, s)
        rows.append(dict(featset=name, seed=s, train_sec=time.perf_counter() - t0, **r))
        print(f"{name:16s} seed {s}: F1={r['F1']:.2f} RMSE={r['RMSE']:.2f} Recall={r['Recall']:.1f} Prec={r['Precision']:.1f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(HERE / "neighbours_per_seed.csv", index=False)
cols = ["MAE", "RMSE", "MBE", "F1", "Recall", "Precision", "train_sec"]
g = res.groupby("featset")[cols].agg(["mean", "std"])
table = pd.DataFrame({c: g[c]["mean"].round(3).astype(str) + " (±" + g[c]["std"].round(3).astype(str) + ")" for c in cols})
print("\n=== Mean (std) over", len(SEEDS), "seeds ===")
print(table.to_string())
table.to_csv(HERE / "neighbours_summary.csv")

print("\n=== Wilcoxon vs 'own' (two-sided, paired by seed) ===")
base = res[res.featset == "own"].sort_values("seed")
for name in ["own+neigh_ws", "own+neigh_ws+pw"]:
    cur = res[res.featset == name].sort_values("seed")
    for met in ["F1", "Recall", "Precision", "RMSE"]:
        try:
            p = wilcoxon(cur[met].values, base[met].values).pvalue
        except ValueError:
            p = float("nan")
        print(f"{name:16s} {met:9s} p = {p:.4f}  {'significant' if p < 0.05 else 'n.s.'}") 