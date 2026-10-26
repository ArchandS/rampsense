"""Step 8: does the model predict the DIRECTION of ramps correctly, and does a signed-ramp loss fix it?
Compares, over 10 seeds (own inputs, lag 1, hidden 10, W1=0.7, tau=0.05):
  mse        : plain MSE
  phi_soft   : 0.7*MSE + 0.3*(1 - soft-F1 on |y_hat - y_prev| > eps)      (the paper's ramp definition, unsigned)
  phi_signed : 0.7*MSE + 0.3*(1 - mean(soft-F1 ramp-up, soft-F1 ramp-down))  (new: direction-aware)
Reports unsigned F1 (paper), signed F1 (a hit needs the right direction), up/down F1, direction accuracy, MBE on up/down ramps.
Needs: torch, numpy, pandas, scipy. Put next to WT15_2017_2018_correct.csv."""
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
METHODS = ["mse", "phi_soft", "phi_signed"]

# ---------------------------------------------------------------- data (identical to 07)
df = pd.read_csv(CSV, index_col=0, parse_dates=True)[["wind_speed", "wind_direction", "power"]].dropna()
ok = (df.index[1:] - df.index[:-1]) == pd.Timedelta("10min")
idx = np.where(ok)[0]
V = df.values
X = V[idx][:, None, :]
y_all, yprev_all = V[idx + 1, 2], V[idx, 2]
n = len(idx); a, b = int(.5 * n), int(.75 * n)
mu, sd = X[:a].reshape(-1, 3).mean(0), X[:a].reshape(-1, 3).std(0)
X = (X - mu) / sd
ymu, ysd = y_all[:a].mean(), y_all[:a].std()
yS = (y_all - ymu) / ysd
print("samples", n, "| train/val/test", a, b - a, n - b)

T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
Xtr, Xva, Xte = T(X[:a]), T(X[a:b]), T(X[b:])
ytr, yva, yte = T(yS[:a]), T(yS[a:b]), T(yS[b:])
prev_tr, prev_va, prev_te = T(yprev_all[:a]), T(yprev_all[a:b]), T(yprev_all[b:])

# ---------------------------------------------------------------- model and losses
class LSTMNet(nn.Module):
    def __init__(s, nin, h=HIDDEN):
        super().__init__(); s.l = nn.LSTM(nin, h, batch_first=True); s.o = nn.Linear(h, 1)
    def forward(s, x): return s.o(s.l(x)[0][:, -1]).squeeze(-1)

def _f1(p, o):
    tp, fp, fn = (p * o).sum(), (p * (1 - o)).sum(), ((1 - p) * o).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-8)

def soft_f1_unsigned(pred_s, true_s, prev_kw, tau=TAU):
    pk, tk = pred_s * ysd + ymu, true_s * ysd + ymu
    p = torch.sigmoid((torch.abs(pk - prev_kw) - EPS) / (tau * EPS))
    o = (torch.abs(tk - prev_kw) > EPS).float()
    return _f1(p, o)

def soft_f1_signed(pred_s, true_s, prev_kw, tau=TAU):
    pk, tk = pred_s * ysd + ymu, true_s * ysd + ymu
    d_p, d_t = pk - prev_kw, tk - prev_kw
    p_up = torch.sigmoid((d_p - EPS) / (tau * EPS)); p_dn = torch.sigmoid((-d_p - EPS) / (tau * EPS))
    o_up, o_dn = (d_t > EPS).float(), (d_t < -EPS).float()
    return 0.5 * (_f1(p_up, o_up) + _f1(p_dn, o_dn))

def loss_fn(name, out, yy, pv):
    mse = ((out - yy) ** 2).mean()
    if name == "mse": return mse
    if name == "phi_soft": return W1 * mse + W2 * (1 - soft_f1_unsigned(out, yy, pv))
    return W1 * mse + W2 * (1 - soft_f1_signed(out, yy, pv))

# ---------------------------------------------------------------- metrics
def f1_of(pred, true):
    tp, fp, fn = (pred & true).sum(), (pred & ~true).sum(), (~pred & true).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 100 * 2 * pr * rc / max(pr + rc, 1e-8), 100 * rc, 100 * pr

def metrics(pred_s, true_s, prev_kw):
    p = np.asarray(pred_s) * ysd + ymu
    t = np.asarray(true_s) * ysd + ymu
    pv = np.asarray(prev_kw)
    e = p - t
    d_t, d_p = t - pv, p - pv
    up_t, dn_t, up_p, dn_p = d_t > EPS, d_t < -EPS, d_p > EPS, d_p < -EPS
    ore, pre = up_t | dn_t, up_p | dn_p
    f1, rc, pr = f1_of(pre, ore)                                   # paper's (unsigned) ramp metrics
    hit = (up_t & up_p) | (dn_t & dn_p)                            # flagged AND same direction
    sp = hit.sum() / max(pre.sum(), 1); sr = hit.sum() / max(ore.sum(), 1)
    both = ore & pre
    return dict(MAE=100 * np.abs(e).mean() / RATED, RMSE=100 * np.sqrt((e ** 2).mean()) / RATED,
                MBE=100 * e.mean() / RATED, F1=f1, Recall=rc, Precision=pr,
                F1_signed=100 * 2 * sp * sr / max(sp + sr, 1e-8), Recall_signed=100 * sr, Precision_signed=100 * sp,
                F1_up=f1_of(up_p, up_t)[0], F1_down=f1_of(dn_p, dn_t)[0],
                DirAcc=100 * hit.sum() / max(both.sum(), 1),       # of ramps flagged by BOTH observed and predicted: right sign
                MBE_up=100 * e[up_t].mean() / RATED, MBE_down=100 * e[dn_t].mean() / RATED)

def train(method, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = LSTMNet(3); opt = torch.optim.Adam(m.parameters(), lr=1e-2); best, bs = 1e9, None
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 1024):
            j = perm[i:i + 1024]; loss = loss_fn(method, m(Xtr[j]), ytr[j], prev_tr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = loss_fn(method, m(Xva), yva, prev_va).item()
        if vl < best: best, bs = vl, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    return m, time.perf_counter() - t0

# ---------------------------------------------------------------- run
rows = []
with torch.no_grad():
    rows.append(dict(model="Persistence", seed=0, sec=np.nan,
                     **metrics((prev_te.numpy() - ymu) / ysd, yte.numpy(), prev_te.numpy())))
for method in METHODS:
    for s in SEEDS:
        m, sec = train(method, s)
        with torch.no_grad(): r = metrics(m(Xte).numpy(), yte.numpy(), prev_te.numpy())
        rows.append(dict(model=method, seed=s, sec=sec, **r))
        print(f"{method:10s} seed {s}: F1={r['F1']:.2f} F1_signed={r['F1_signed']:.2f} DirAcc={r['DirAcc']:.1f} "
              f"RMSE={r['RMSE']:.2f} ({sec:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(HERE / "direction_per_seed.csv", index=False)
cols = ["MAE", "RMSE", "MBE", "F1", "Recall", "Precision", "F1_signed", "Recall_signed", "Precision_signed",
        "F1_up", "F1_down", "DirAcc", "MBE_up", "MBE_down", "sec"]
g = res.groupby("model")[cols].agg(["mean", "std"]).reindex(["Persistence"] + METHODS)
fmt = lambda c: g[c]["mean"].round(2).astype(str) + " (±" + g[c]["std"].fillna(0).round(2).astype(str) + ")"
pd.set_option("display.width", 250)
print("\n=== Accuracy and unsigned ramp metrics ===")
print(pd.DataFrame({c: fmt(c) for c in ["MAE", "RMSE", "MBE", "F1", "Recall", "Precision"]}).to_string())
print("\n=== Direction-aware metrics ===")
print(pd.DataFrame({c: fmt(c) for c in ["F1_signed", "Recall_signed", "Precision_signed", "F1_up", "F1_down", "DirAcc"]}).to_string())
print("\n=== Bias on observed ramps (% of rated) and training seconds ===")
print(pd.DataFrame({c: fmt(c) for c in ["MBE_up", "MBE_down", "sec"]}).to_string())
pd.DataFrame({c: fmt(c) for c in cols}).to_csv(HERE / "direction_summary.csv")

# ---------------------------------------------------------------- Wilcoxon (Holm over all tests)
def get(m, met): return res[res.model == m].sort_values("seed")[met].values
tests = []
for A, B, mets in [("phi_signed", "phi_soft", ["F1_signed", "DirAcc", "F1", "Recall", "Precision", "RMSE"]),
                   ("phi_signed", "mse", ["F1_signed", "RMSE"]),
                   ("phi_soft", "mse", ["F1_signed", "DirAcc"])]:
    for met in mets:
        try: p = wilcoxon(get(A, met), get(B, met)).pvalue
        except ValueError: p = float("nan")
        tests.append((f"{A} vs {B}", met, p))
ps = np.array([t[2] for t in tests]); order = np.argsort(ps); holm = np.empty_like(ps); run = 0
for rank, i in enumerate(order):
    run = max(run, min(1, (len(ps) - rank) * ps[i])); holm[i] = run
print("\n=== Wilcoxon (two-sided, paired by seed), Holm-adjusted over all", len(tests), "tests ===")
for (name, met, p), h in zip(tests, holm):
    print(f"{name:25s} {met:10s} p={p:.4f}  Holm p={h:.4f}  {'significant' if h < 0.05 else 'n.s.'}")