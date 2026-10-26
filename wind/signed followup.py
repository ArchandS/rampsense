"""Step 9: signed-ramp follow-up on WT15.
 1) Momentum baseline (no training): y_hat = y_t + alpha*(y_t - y_{t-1}), alpha chosen on VALIDATION signed F1.
 2) phi_signed loss with W1 in {0.3, 0.5, 0.7} (own inputs, lag 1, hidden 10).
 3) phi_signed with ramp-history features (dP, dws) and lag 6, hidden 20.
 Reference models: Persistence, LSTM-MSE, LSTM-phi_soft (the paper's unsigned ramp loss).
All models use the SAME samples/targets (windows built for lag 6, so numbers differ slightly from 08), 10 seeds, test set.
Needs: torch, numpy, pandas, scipy; optional matplotlib. Put next to WT15_2017_2018_correct.csv."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
CSV = HERE / "WT15_2017_2018_correct.csv"
EPS, RATED, TAU, EPOCHS = 205.0, 2050.0, 0.05, 60
SEEDS = list(range(10))
MAXLAG = 6
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# name: (loss, featset, lag, hidden, w1)
CONFIGS = {
    "LSTM-MSE":             ("mse",        "own",      1, 10, 1.0),
    "LSTM-phi_soft":        ("phi_soft",   "own",      1, 10, 0.7),
    "signed W1=0.7":        ("phi_signed", "own",      1, 10, 0.7),
    "signed W1=0.5":        ("phi_signed", "own",      1, 10, 0.5),
    "signed W1=0.3":        ("phi_signed", "own",      1, 10, 0.3),
    "signed+diff lag6 W1=0.7": ("phi_signed", "own+diff", 6, 20, 0.7),
    "signed+diff lag6 W1=0.5": ("phi_signed", "own+diff", 6, 20, 0.5),
}
FEATS = {"own": ["wind_speed", "wind_direction", "power"],
         "own+diff": ["wind_speed", "wind_direction", "power", "dP", "dws"]}

# ---------------------------------------------------------------- data: identical samples for all configs
df = pd.read_csv(CSV, index_col=0, parse_dates=True)[["wind_speed", "wind_direction", "power"]]
df["dP"] = df["power"].diff(); df["dws"] = df["wind_speed"].diff()
df = df.dropna()
N = len(df)
ok = ((df.index[1:] - df.index[:-1]) == pd.Timedelta("10min")).astype(int)
okc = np.concatenate([[0], np.cumsum(ok)])
t_all = np.arange(MAXLAG - 1, N - 1)
ts = t_all[(okc[t_all + 1] - okc[t_all - (MAXLAG - 1)]) == MAXLAG]
n = len(ts); a, b = int(.5 * n), int(.75 * n)
power = df["power"].values
y_all, prev_all, prev2_all = power[ts + 1], power[ts], power[ts - 1]
ymu, ysd = y_all[:a].mean(), y_all[:a].std()
yS = (y_all - ymu) / ysd
print("samples", n, "| train/val/test", a, b - a, n - b)

T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
ytr, yva, yte = T(yS[:a]), T(yS[a:b]), T(yS[b:])
prev_tr, prev_va, prev_te = T(prev_all[:a]), T(prev_all[a:b]), T(prev_all[b:])

def make_split(cols, lag):
    Vv = df[cols].values
    X = np.stack([Vv[ts - k] for k in range(lag - 1, -1, -1)], axis=1)
    flat = X[:a].reshape(-1, X.shape[-1])
    X = (X - flat.mean(0)) / (flat.std(0) + 1e-8)           # scaler fitted on train only
    return T(X[:a]), T(X[a:b]), T(X[b:])

# ---------------------------------------------------------------- model, losses, metrics
class LSTMNet(nn.Module):
    def __init__(s, nin, h):
        super().__init__(); s.l = nn.LSTM(nin, h, batch_first=True); s.o = nn.Linear(h, 1)
    def forward(s, x): return s.o(s.l(x)[0][:, -1]).squeeze(-1)

def _f1(p, o):
    tp, fp, fn = (p * o).sum(), (p * (1 - o)).sum(), ((1 - p) * o).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-8)

def soft_unsigned(pred_s, true_s, prev_kw):
    pk, tk = pred_s * ysd + ymu, true_s * ysd + ymu
    p = torch.sigmoid((torch.abs(pk - prev_kw) - EPS) / (TAU * EPS))
    return _f1(p, (torch.abs(tk - prev_kw) > EPS).float())

def soft_signed(pred_s, true_s, prev_kw):
    pk, tk = pred_s * ysd + ymu, true_s * ysd + ymu
    dp, dt = pk - prev_kw, tk - prev_kw
    return 0.5 * (_f1(torch.sigmoid((dp - EPS) / (TAU * EPS)), (dt > EPS).float())
                  + _f1(torch.sigmoid((-dp - EPS) / (TAU * EPS)), (dt < -EPS).float()))

def loss_fn(kind, w1, out, yy, pv):
    mse = ((out - yy) ** 2).mean()
    if kind == "mse": return mse
    f = soft_unsigned if kind == "phi_soft" else soft_signed
    return w1 * mse + (1 - w1) * (1 - f(out, yy, pv))

def f1_of(pred, true):
    tp, fp, fn = (pred & true).sum(), (pred & ~true).sum(), (~pred & true).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 100 * 2 * pr * rc / max(pr + rc, 1e-8), 100 * rc, 100 * pr

def metrics(pred_kw, true_kw, prev_kw):
    p, t, pv = np.asarray(pred_kw), np.asarray(true_kw), np.asarray(prev_kw)
    e = p - t; d_t, d_p = t - pv, p - pv
    up_t, dn_t, up_p, dn_p = d_t > EPS, d_t < -EPS, d_p > EPS, d_p < -EPS
    ore, pre = up_t | dn_t, up_p | dn_p
    f1, rc, pr = f1_of(pre, ore)
    hit = (up_t & up_p) | (dn_t & dn_p)
    sp, sr = hit.sum() / max(pre.sum(), 1), hit.sum() / max(ore.sum(), 1)
    return dict(MAE=100 * np.abs(e).mean() / RATED, RMSE=100 * np.sqrt((e ** 2).mean()) / RATED,
                MBE=100 * e.mean() / RATED, F1=f1, Recall=rc, Precision=pr,
                F1_signed=100 * 2 * sp * sr / max(sp + sr, 1e-8), F1_up=f1_of(up_p, up_t)[0], F1_down=f1_of(dn_p, dn_t)[0],
                DirAcc=100 * hit.sum() / max((ore & pre).sum(), 1),
                MBE_up=100 * e[up_t].mean() / RATED, MBE_down=100 * e[dn_t].mean() / RATED)

kw = lambda s: np.asarray(s) * ysd + ymu            # scaled -> kW
true_te_kw, true_va_kw = kw(yte.numpy()), kw(yva.numpy())

def train_run(cfg, seed, splits):
    kind, _, lag, hidden, w1 = cfg
    Xtr, Xva, Xte = splits
    torch.manual_seed(seed); np.random.seed(seed)
    m = LSTMNet(Xtr.shape[-1], hidden); opt = torch.optim.Adam(m.parameters(), lr=1e-2); best, bs = 1e9, None
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 1024):
            j = perm[i:i + 1024]; loss = loss_fn(kind, w1, m(Xtr[j]), ytr[j], prev_tr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = loss_fn(kind, w1, m(Xva), yva, prev_va).item()
        if vl < best: best, bs = vl, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): pred = m(Xte).numpy()
    return metrics(kw(pred), true_te_kw, prev_te.numpy()), time.perf_counter() - t0

# ---------------------------------------------------------------- run
rows = []
rows.append(dict(model="Persistence", seed=0, sec=np.nan, **metrics(prev_te.numpy(), true_te_kw, prev_te.numpy())))

# momentum: alpha selected on validation signed F1 (no test leakage)
va_sl, te_sl = slice(a, b), slice(b, n)
best_alpha, best_s = None, -1
for al in ALPHAS:
    pv = prev_all[va_sl] + al * (prev_all[va_sl] - prev2_all[va_sl])
    s = metrics(pv, true_va_kw, prev_all[va_sl])["F1_signed"]
    print(f"[momentum] alpha={al}: val signed F1={s:.2f}")
    if s > best_s: best_alpha, best_s = al, s
mom = prev_all[te_sl] + best_alpha * (prev_all[te_sl] - prev2_all[te_sl])
mom_name = f"Momentum (alpha={best_alpha})"
mom_res = metrics(mom, true_te_kw, prev_te.numpy())
rows.append(dict(model=mom_name, seed=0, sec=np.nan, **mom_res))
print("Momentum test:", {k: round(float(v), 2) for k, v in mom_res.items()})

cache = {}
for name, cfg in CONFIGS.items():
    key = (cfg[1], cfg[2])
    if key not in cache: cache[key] = make_split(FEATS[cfg[1]], cfg[2])
    for s in SEEDS:
        r, sec = train_run(cfg, s, cache[key])
        rows.append(dict(model=name, seed=s, sec=sec, **r))
        print(f"{name:26s} seed {s}: F1={r['F1']:.2f} F1_signed={r['F1_signed']:.2f} DirAcc={r['DirAcc']:.1f} RMSE={r['RMSE']:.2f} ({sec:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(HERE / "signed_followup_per_seed.csv", index=False)
order = ["Persistence", mom_name] + list(CONFIGS)
cols = ["MAE", "RMSE", "MBE", "F1", "Recall", "Precision", "F1_signed", "F1_up", "F1_down", "DirAcc", "MBE_up", "MBE_down", "sec"]
g = res.groupby("model")[cols].agg(["mean", "std"]).reindex(order)
fmt = lambda c: g[c]["mean"].round(2).astype(str) + " (±" + g[c]["std"].fillna(0).round(2).astype(str) + ")"
pd.set_option("display.width", 260)
print("\n=== Accuracy and unsigned ramp metrics ===")
print(pd.DataFrame({c: fmt(c) for c in ["MAE", "RMSE", "MBE", "F1", "Recall", "Precision"]}).to_string())
print("\n=== Direction-aware metrics ===")
print(pd.DataFrame({c: fmt(c) for c in ["F1_signed", "F1_up", "F1_down", "DirAcc", "MBE_up", "MBE_down"]}).to_string())
pd.DataFrame({c: fmt(c) for c in cols}).to_csv(HERE / "signed_followup_summary.csv")

# ---------------------------------------------------------------- statistics (Holm over all tests)
def vals(m, met): return res[res.model == m].sort_values("seed")[met].values
tests = []
ref = "LSTM-phi_soft"
for name in CONFIGS:
    if name in (ref,): continue
    for met in ["F1_signed", "RMSE"]:
        try: p = wilcoxon(vals(name, met), vals(ref, met)).pvalue
        except ValueError: p = float("nan")
        tests.append((f"{name} vs {ref}", met, p))
mom_val = mom_res["F1_signed"]                     # deterministic baseline -> one-sample signed-rank on (seed value - baseline)
for name in CONFIGS:
    try: p = wilcoxon(vals(name, "F1_signed") - mom_val).pvalue
    except ValueError: p = float("nan")
    tests.append((f"{name} vs {mom_name}", "F1_signed", p))
ps = np.array([t[2] for t in tests]); o = np.argsort(ps); holm = np.empty_like(ps); run = 0
for rank, i in enumerate(o):
    run = max(run, min(1, (len(ps) - rank) * ps[i])); holm[i] = run
print("\n=== Wilcoxon, two-sided, Holm-adjusted over", len(tests), "tests (sign of the difference: see tables) ===")
for (nm, met, p), h in zip(tests, holm):
    print(f"{nm:55s} {met:10s} p={p:.4f}  Holm p={h:.4f}  {'significant' if h < 0.05 else 'n.s.'}")

# ---------------------------------------------------------------- figure: signed F1 vs RMSE
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    mk = {"Persistence": ("black", "*"), mom_name: ("tab:orange", "D")}
    for nm in order:
        r = res[res.model == nm]
        c, m_ = mk.get(nm, (None, "o"))
        ax.errorbar(r.RMSE.mean(), r.F1_signed.mean(), xerr=r.RMSE.std() if len(r) > 1 else 0,
                    yerr=r.F1_signed.std() if len(r) > 1 else 0, fmt=m_, color=c, capsize=2, label=nm, markersize=7)
    ax.set_xlabel("Test RMSE (% of rated power)"); ax.set_ylabel("Signed F1 (%)  [direction must match]")
    ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper right"); fig.tight_layout()
    fig.savefig(HERE / "signed_followup_figure.png", dpi=200); print("Saved signed_followup_figure.png")
except ImportError:
    print("matplotlib not installed -> figure skipped")