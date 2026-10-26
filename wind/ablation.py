"""Step 10: ablation of the signed-ramp model, with selection on VALIDATION data only.
Stage A (selection): full factorial of the signed loss over lag {1,6} x diff-features {off,on} x hidden {10,20} x W1 {0.5,0.7}
                      = 16 configs, 3 seeds each (seeds 0-2). Pick the config with the highest validation signed F1
                      among those whose validation RMSE <= 1.25 x the best validation RMSE.
Stage B (final, FRESH seeds 10-19): the selected config, its three one-at-a-time ablations (lag->1, diff->off, hidden->10),
                      and two references (LSTM-MSE, LSTM-phi_soft). Test set used only here.
Same samples/targets as 09 (windows built for lag 6). Needs torch, numpy, pandas, scipy. Put next to WT15_2017_2018_correct.csv."""
import itertools, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
CSV = HERE / "WT15_2017_2018_correct.csv"
EPS, RATED, TAU, EPOCHS = 205.0, 2050.0, 0.05, 60
GRID_SEEDS, FINAL_SEEDS = [0, 1, 2], list(range(10, 20))
RMSE_TOL, MAXLAG = 1.25, 6
LAGS, DIFFS, HIDDENS, W1S = [1, 6], [False, True], [10, 20], [0.5, 0.7]
BASE = ["wind_speed", "wind_direction", "power"]

# ---------------------------------------------------------------- data
df = pd.read_csv(CSV, index_col=0, parse_dates=True)[BASE]
df["dP"] = df["power"].diff(); df["dws"] = df["wind_speed"].diff()
df = df.dropna()
N = len(df)
ok = ((df.index[1:] - df.index[:-1]) == pd.Timedelta("10min")).astype(int)
okc = np.concatenate([[0], np.cumsum(ok)])
t_all = np.arange(MAXLAG - 1, N - 1)
ts = t_all[(okc[t_all + 1] - okc[t_all - (MAXLAG - 1)]) == MAXLAG]
n = len(ts); a, b = int(.5 * n), int(.75 * n)
power = df["power"].values
y_all, prev_all = power[ts + 1], power[ts]
ymu, ysd = y_all[:a].mean(), y_all[:a].std()
yS = (y_all - ymu) / ysd
print("samples", n, "| train/val/test", a, b - a, n - b)

T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
ytr, yva, yte = T(yS[:a]), T(yS[a:b]), T(yS[b:])
prev_tr, prev_va, prev_te = T(prev_all[:a]), T(prev_all[a:b]), T(prev_all[b:])
kw = lambda s: np.asarray(s) * ysd + ymu
true_va_kw, true_te_kw = kw(yva.numpy()), kw(yte.numpy())

_cache = {}
def get_split(diff, lag):
    if (diff, lag) not in _cache:
        cols = BASE + (["dP", "dws"] if diff else [])
        Vv = df[cols].values
        X = np.stack([Vv[ts - k] for k in range(lag - 1, -1, -1)], axis=1)
        flat = X[:a].reshape(-1, X.shape[-1])
        X = (X - flat.mean(0)) / (flat.std(0) + 1e-8)                 # scaler fitted on train only
        _cache[(diff, lag)] = (T(X[:a]), T(X[a:b]), T(X[b:]))
    return _cache[(diff, lag)]

# ---------------------------------------------------------------- model, losses, metrics
class LSTMNet(nn.Module):
    def __init__(s, nin, h):
        super().__init__(); s.l = nn.LSTM(nin, h, batch_first=True); s.o = nn.Linear(h, 1)
    def forward(s, x): return s.o(s.l(x)[0][:, -1]).squeeze(-1)

def _f1(p, o):
    tp, fp, fn = (p * o).sum(), (p * (1 - o)).sum(), ((1 - p) * o).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-8)

def soft_unsigned(pred_s, true_s, prev_kw):
    pk, tk = kw_t(pred_s), kw_t(true_s)
    return _f1(torch.sigmoid((torch.abs(pk - prev_kw) - EPS) / (TAU * EPS)), (torch.abs(tk - prev_kw) > EPS).float())

def soft_signed(pred_s, true_s, prev_kw):
    dp, dt = kw_t(pred_s) - prev_kw, kw_t(true_s) - prev_kw
    return 0.5 * (_f1(torch.sigmoid((dp - EPS) / (TAU * EPS)), (dt > EPS).float())
                  + _f1(torch.sigmoid((-dp - EPS) / (TAU * EPS)), (dt < -EPS).float()))

kw_t = lambda s: s * ysd + ymu

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
                F1_signed=100 * 2 * sp * sr / max(sp + sr, 1e-8),
                F1_up=f1_of(up_p, up_t)[0], F1_down=f1_of(dn_p, dn_t)[0],
                DirAcc=100 * hit.sum() / max((ore & pre).sum(), 1))

def run(cfg, seed):
    kind, diff, lag, hidden, w1 = cfg
    Xtr, Xva, Xte = get_split(diff, lag)
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
    with torch.no_grad(): pv_, pt_ = m(Xva).numpy(), m(Xte).numpy()
    return (metrics(kw(pv_), true_va_kw, prev_va.numpy()), metrics(kw(pt_), true_te_kw, prev_te.numpy()),
            time.perf_counter() - t0)

# ---------------------------------------------------------------- Stage A: selection on validation
rows = []
for lag, diff, hidden, w1 in itertools.product(LAGS, DIFFS, HIDDENS, W1S):
    cfg = ("phi_signed", diff, lag, hidden, w1)
    for s in GRID_SEEDS:
        va, te, sec = run(cfg, s)
        rows.append(dict(lag=lag, diff=diff, hidden=hidden, w1=w1, seed=s, sec=sec,
                         **{f"val_{k}": v for k, v in va.items()}, **{f"test_{k}": v for k, v in te.items()}))
        print(f"[grid] lag={lag} diff={int(diff)} h={hidden} W1={w1} seed={s}  valF1s={va['F1_signed']:.2f} valRMSE={va['RMSE']:.2f}", flush=True)
grid = pd.DataFrame(rows)
grid.to_csv(HERE / "ablation_grid_per_seed.csv", index=False)       # test columns saved for the record, NOT used for selection
agg = grid.groupby(["lag", "diff", "hidden", "w1"]).mean(numeric_only=True).reset_index()
pd.set_option("display.width", 220)
print("\n=== Grid, VALIDATION metrics (mean over 3 seeds) ===")
print(agg[["lag", "diff", "hidden", "w1", "val_F1_signed", "val_DirAcc", "val_RMSE"]].round(2).to_string(index=False))
print("\n=== Factor main effects on validation (mean over all other factors) ===")
for f in ["lag", "diff", "hidden", "w1"]:
    print(grid.groupby(f)[["val_F1_signed", "val_DirAcc", "val_RMSE"]].mean().round(2).to_string(), "\n")

ok_ = agg[agg.val_RMSE <= RMSE_TOL * agg.val_RMSE.min()]
sel = ok_.loc[ok_.val_F1_signed.idxmax()]
sel_cfg = ("phi_signed", bool(sel["diff"]), int(sel["lag"]), int(sel["hidden"]), float(sel["w1"]))
print("SELECTED on validation:", dict(lag=sel_cfg[2], diff=sel_cfg[1], hidden=sel_cfg[3], W1=sel_cfg[4]))

# ---------------------------------------------------------------- Stage B: final, fresh seeds, test set
variants = {"selected": sel_cfg}
if sel_cfg[2] != 1:   variants["ablate: lag->1"] = (sel_cfg[0], sel_cfg[1], 1, sel_cfg[3], sel_cfg[4])
if sel_cfg[1]:        variants["ablate: diff->off"] = (sel_cfg[0], False, sel_cfg[2], sel_cfg[3], sel_cfg[4])
if sel_cfg[3] != 10:  variants["ablate: hidden->10"] = (sel_cfg[0], sel_cfg[1], sel_cfg[2], 10, sel_cfg[4])
variants["ref: LSTM-phi_soft"] = ("phi_soft", False, 1, 10, 0.7)
variants["ref: LSTM-MSE"] = ("mse", False, 1, 10, 1.0)

fin = []
for name, cfg in variants.items():
    for s in FINAL_SEEDS:
        _, te, sec = run(cfg, s)
        fin.append(dict(model=name, seed=s, sec=sec, **te))
        print(f"[final] {name:22s} seed {s}: F1s={te['F1_signed']:.2f} Dir={te['DirAcc']:.1f} RMSE={te['RMSE']:.2f} F1={te['F1']:.2f}", flush=True)
res = pd.DataFrame(fin)
res.to_csv(HERE / "ablation_final_per_seed.csv", index=False)
cols = ["MAE", "RMSE", "MBE", "F1", "Recall", "Precision", "F1_signed", "F1_up", "F1_down", "DirAcc", "sec"]
g = res.groupby("model")[cols].agg(["mean", "std"]).reindex(list(variants))
fmt = lambda c: g[c]["mean"].round(2).astype(str) + " (±" + g[c]["std"].round(2).astype(str) + ")"
print("\n=== FINAL (fresh seeds 10-19, test set): mean (std) ===")
print(pd.DataFrame({c: fmt(c) for c in cols}).to_string())
pd.DataFrame({c: fmt(c) for c in cols}).to_csv(HERE / "ablation_final_summary.csv")

# ---------------------------------------------------------------- Wilcoxon vs selected, Holm-corrected
def v(m, met): return res[res.model == m].sort_values("seed")[met].values
tests = []
for name in variants:
    if name == "selected": continue
    for met in ["F1_signed", "DirAcc", "RMSE"]:
        try: p = wilcoxon(v(name, met), v("selected", met)).pvalue
        except ValueError: p = float("nan")
        tests.append((name, met, p))
ps = np.array([t[2] for t in tests]); o = np.argsort(ps); holm = np.empty_like(ps); run_ = 0
for rank, i in enumerate(o):
    run_ = max(run_, min(1, (len(ps) - rank) * ps[i])); holm[i] = run_
print("\n=== Wilcoxon vs 'selected' (two-sided, paired by seed), Holm over", len(tests), "tests ===")
for (nm, met, p), h in zip(tests, holm):
    d = v(nm, met).mean() - v("selected", met).mean()
    print(f"{nm:22s} {met:10s} mean diff vs selected = {d:+.2f}   p={p:.4f}  Holm p={h:.4f}  {'significant' if h < 0.05 else 'n.s.'}")