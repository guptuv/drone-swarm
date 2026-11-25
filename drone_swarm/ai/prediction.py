"""Trajectory prediction of intruders (birds, other aircraft) with sequence models,
and prediction-aware collision avoidance.

Intruders follow a manoeuvring unicycle model (turn rate and speed change randomly).
Given the last H = 10 positions (1 s), predict the next F = 20 (2 s).

Predictors: constant velocity (CV), constant turn rate & velocity (CTRV, physics model),
LSTM encoder-decoder, Transformer encoder.  Metrics: ADE / FDE (average / final displacement).
"""
import numpy as np
import torch
import torch.nn as nn

H, F, DT = 10, 20, 0.1


def simulate_intruders(n, steps, rng, speed=(4, 9), turn=(-0.9, 0.9), switch_p=0.03):
    th = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(*speed, n)
    w = rng.uniform(*turn, n)
    p = rng.uniform(-20, 20, (n, 2))
    out = np.zeros((n, steps, 2))
    for t in range(steps):
        sw = rng.random(n) < switch_p
        w[sw] = rng.uniform(*turn, sw.sum())
        v = np.clip(v + rng.normal(0, 0.15, n), speed[0], speed[1])
        th = th + w * DT
        p = p + np.c_[np.cos(th), np.sin(th)] * v[:, None] * DT
        out[:, t] = p
    return out + rng.normal(0, 0.03, out.shape)


def make_windows(traj):
    X, Y = [], []
    for s in range(0, traj.shape[1] - H - F, 3):
        X.append(traj[:, s:s + H]); Y.append(traj[:, s + H:s + H + F])
    X, Y = np.concatenate(X), np.concatenate(Y)
    origin = X[:, -1:].copy()
    return (X - origin).astype(np.float32), (Y - origin).astype(np.float32)        # relative to last seen


def predict_cv(X):
    v = (X[:, -1] - X[:, -3]) / 2
    return X[:, -1:] + v[:, None] * np.arange(1, F + 1)[None, :, None]


def predict_ctrv(X):
    d1, d2 = X[:, -1] - X[:, -2], X[:, -4] - X[:, -5]
    a1, a2 = np.arctan2(d1[:, 1], d1[:, 0]), np.arctan2(d2[:, 1], d2[:, 0])
    w = np.angle(np.exp(1j * (a1 - a2))) / 3
    sp = np.linalg.norm(X[:, -1] - X[:, -4], axis=1) / 3
    out = np.zeros((len(X), F, 2)); p = X[:, -1].copy(); th = a1.copy()
    for k in range(F):
        th = th + w; p = p + np.c_[np.cos(th), np.sin(th)] * sp[:, None]; out[:, k] = p
    return out


class LSTMPredictor(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.enc = nn.LSTM(2, hidden, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, F * 2))

    def forward(self, x):
        dx = torch.diff(x, dim=1, prepend=x[:, :1])
        _, (h, _) = self.enc(dx * 5)
        steps = self.head(h[-1]).reshape(-1, F, 2) / 5
        return x[:, -1:] + torch.cumsum(steps, 1)


class TransformerPredictor(nn.Module):
    def __init__(self, d=64, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(4, d)
        self.pos = nn.Parameter(torch.randn(1, H, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d * H, F * 2)

    def forward(self, x):
        dx = torch.diff(x, dim=1, prepend=x[:, :1])
        h = self.enc(self.inp(torch.cat([x / 5, dx * 5], -1)) + self.pos)
        steps = self.head(h.flatten(1)).reshape(-1, F, 2) / 5
        return x[:, -1:] + torch.cumsum(steps, 1)


def ade_fde(pred, Y):
    e = np.linalg.norm(pred - Y, axis=-1)
    return float(e.mean()), float(e[:, -1].mean())


def train_predictor(model, X, Y, epochs=15, lr=2e-3, verbose=True, name='model'):
    X, Y = torch.tensor(X), torch.tensor(Y)
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X)); tot = 0
        for s in range(0, len(X), 256):
            i = perm[s:s + 256]
            loss = (model(X[i]) - Y[i]).norm(dim=-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(i)
        sched.step()
        if verbose and (ep + 1) % 5 == 0:
            print(f"{name} epoch {ep + 1} | train ADE {tot / len(X):.3f} m")
    return model


def avoidance_trial(predict_fn, n_trials=200, seed=0, horizon=F, safe=3.0, gain=12.0, t_cross=2.5):
    """Head-on style encounter: the drone's straight path is set up to meet a manoeuvring
    intruder about `t_cross` seconds later.  The drone dodges using the *predicted* intruder
    path (closest point of approach over the horizon).  Returns the loss-of-separation rate
    (min distance < 1.5 m) and the mean minimum distance."""
    rng = np.random.default_rng(seed)
    viol, mins = 0, []
    for _ in range(n_trials):
        tr = simulate_intruders(1, 400, rng)[0]
        s = rng.integers(60, 200)
        k_meet = int(t_cross / DT)
        heading = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(heading), np.sin(heading)])
        meet = tr[s + H + k_meet]
        drone = meet - u * 6.0 * t_cross + rng.normal(0, 0.5, 2)
        goal = drone + u * 40
        v = u * 6.0; md = np.inf
        for t in range(60):
            hist = tr[s + t:s + t + H]
            pred = predict_fn(hist[None] - hist[-1])[0] + hist[-1]
            dg = goal - drone
            a = (dg / (np.linalg.norm(dg) + 1e-9) * 6 - v) * 2.0
            my = drone + v[None] * DT * np.arange(1, horizon + 1)[:, None]
            gap = my - pred[:horizon]
            dist = np.linalg.norm(gap, axis=1)
            k = dist.argmin()
            if dist[k] < safe:
                a += gap[k] / (dist[k] + 1e-6) * gain * (1 - k / horizon)
            a = a * min(1, 10 / (np.linalg.norm(a) + 1e-9))
            v = v + a * DT; v = v * min(1, 8 / (np.linalg.norm(v) + 1e-9))
            drone = drone + v * DT
            md = min(md, np.linalg.norm(drone - tr[s + t + H]))
        mins.append(md); viol += md < 1.5
    return dict(loss_of_separation=viol / n_trials, mean_min_dist=float(np.mean(mins)))
