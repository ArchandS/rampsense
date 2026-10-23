import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
torch.manual_seed(0); np.random.seed(0)
CSV = str(Path(__file__).resolve().parent / "WT15_2017_2018_correct.csv")
RATED, EPS, LAG = 2050.0, 205.0, 1          # paper: lag n=1, eps = 10% of rated
W1, W2 = 0.7, 0.3                            # paper's chosen weights (wind)
USE_SINCOS = False
LOSS = "mse"                                 # "mse" | "phi_soft"  (phi_soft = W1*MSE + W2*(1-softF1))

df = pd.read_csv(CSV, index_col=0, parse_dates=True)
feats = ["wind_speed", "wd_sin", "wd_cos", "power"] if USE_SINCOS else ["wind_speed", "wind_direction", "power"]
df = df[feats].dropna()

# windows (only where the 10-min grid is unbroken)
X, y, yprev = [], [], []
v = df.values; pi = feats.index("power"); ok = (df.index[1:] - df.index[:-1] == pd.Timedelta("10min"))
for t in range(LAG - 1, len(df) - 1):
    if ok[t - LAG + 1:t + 1].all():
        X.append(v[t - LAG + 1:t + 1]); y.append(v[t + 1, pi]); yprev.append(v[t, pi])
X, y, yprev = map(np.array, (X, y, yprev))

# chronological 50/25/25, scaler fitted on train only
n = len(X); a, b = int(.5 * n), int(.75 * n)
mu, sd = X[:a].reshape(-1, X.shape[-1]).mean(0), X[:a].reshape(-1, X.shape[-1]).std(0)
ymu, ysd = y[:a].mean(), y[:a].std()
Xs = (X - mu) / sd; ys = (y - ymu) / ysd
T = lambda z: torch.tensor(z, dtype=torch.float32)
Xtr, ytr, Xva, yva, Xte, yte = T(Xs[:a]), T(ys[:a]), T(Xs[a:b]), T(ys[a:b]), T(Xs[b:]), T(ys[b:])
prev_tr, prev_te = T(yprev[:a]), T(yprev[b:])

class LSTMNet(nn.Module):
    def __init__(s, nin, h=10):
        super().__init__(); s.l = nn.LSTM(nin, h, batch_first=True); s.o = nn.Linear(h, 1)
    def forward(s, x): return s.o(s.l(x)[0][:, -1]).squeeze(-1)

def soft_f1(pred_s, true_s, prev_kw, tau=0.05):
    """Differentiable F1: ramp = sigmoid((|y_hat-y_prev|-EPS)/tau) on kW scale (tau in fraction of EPS)."""
    pred_kw = pred_s * ysd + ymu; true_kw = true_s * ysd + ymu
    p = torch.sigmoid((torch.abs(pred_kw - prev_kw) - EPS) / (tau * EPS))
    o = (torch.abs(true_kw - prev_kw) > EPS).float()
    tp, fp, fn = (p * o).sum(), (p * (1 - o)).sum(), ((1 - p) * o).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-8)

def metrics(pred_s, true_s, prev_kw):
    p = pred_s.numpy() * ysd + ymu; t = true_s.numpy() * ysd + ymu; pv = prev_kw.numpy()
    e = p - t; ore = np.abs(t - pv) > EPS; pre = np.abs(p - pv) > EPS
    tp, fp, fn, tn = (pre & ore).sum(), (pre & ~ore).sum(), (~pre & ore).sum(), (~pre & ~ore).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(MAE_pct=100 * np.abs(e).mean() / RATED, RMSE_pct=100 * np.sqrt((e**2).mean()) / RATED,
                MBE_pct=100 * e.mean() / RATED, MAAPE=np.arctan(np.abs(e) / np.maximum(np.abs(t), 1e-6)).mean(),
                F1=100 * 2 * pr * rc / max(pr + rc, 1e-8), Recall=100 * rc, Precision=100 * pr,
                Accuracy=100 * (tp + tn) / len(t))

def train(loss_name, epochs=60, lr=1e-2):
    m = LSTMNet(Xtr.shape[-1]); opt = torch.optim.Adam(m.parameters(), lr=lr); best, bs = 1e9, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 1024):
            idx = perm[i:i + 1024]; out = m(Xtr[idx]); mse = ((out - ytr[idx])**2).mean()
            loss = mse if loss_name == "mse" else W1 * mse + W2 * (1 - soft_f1(out, ytr[idx], prev_tr[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vo = m(Xva); vm = ((vo - yva)**2).mean()
            vl = vm if loss_name == "mse" else W1 * vm + W2 * (1 - soft_f1(vo, yva, T(yprev[a:b])))
        if vl < best: best, bs = vl.item(), {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); return m

if __name__ == "__main__":
    print("samples", n, "| train/val/test", a, b - a, n - b)
    with torch.no_grad():                                   # persistence baseline
        print("Persistence", {k: round(v, 3) for k, v in metrics(((prev_te - ymu) / ysd), yte, prev_te).items()})
    for name in ["mse", "phi_soft"]:
        m = train(name)
        with torch.no_grad(): r = metrics(m(Xte), yte, prev_te)
        print(f"LSTM-Adam-{name}", {k: round(float(v), 3) for k, v in r.items()})
