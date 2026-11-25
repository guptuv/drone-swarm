"""Model-based control: learn the quadrotor's dynamics with a neural network, then plan
with Model Predictive Path Integral control (MPPI, Williams et al. 2017).

1. Collect random-action flight data from QuadrotorDynamics (+ wind).
2. Train an MLP f(v, z_b, a_cmd) -> (v', z_b')  (residual / delta prediction).
3. MPPI samples K action sequences, rolls them out through the model (true or learned),
   scores them (goal distance, obstacle cost, effort) and takes the softmin-weighted mean.

Task: fly a single quadrotor through a slalom of pillars to a goal, in wind.
Compared: PD controller (no look-ahead), MPPI with the learned model, MPPI with the true model.
"""
import numpy as np
import torch
import torch.nn as nn
from ..dynamics import QuadrotorDynamics, Wind
from ..core import clip_norm

DT = 0.1


class _Quad:
    """Single-drone state wrapper around QuadrotorDynamics (works on batches)."""

    @staticmethod
    def step(v, zb, T, a_cmd, dyn, wind=None):
        dyn.zb, dyn.T = zb.copy(), T.copy()
        a = dyn.step(a_cmd, v, DT, wind)
        return v + a * DT, dyn.zb.copy(), dyn.T.copy()


def collect_data(n_samples=40000, seed=0):
    rng = np.random.default_rng(seed)
    B = 200
    dyn = QuadrotorDynamics(B)
    v = rng.normal(0, 3, (B, 3)); zb = dyn.zb.copy(); T = dyn.T.copy()
    X, Y = [], []
    while len(X) * B < n_samples:
        a = rng.uniform(-8, 8, (B, 3))
        v2, zb2, T2 = _Quad.step(v, zb, T, a, dyn)
        X.append(np.c_[v, zb, a]); Y.append(np.c_[v2 - v, zb2 - zb])
        v, zb, T = clip_norm(v2, 10), zb2, T2
        reset = rng.random(B) < 0.05
        v[reset] = rng.normal(0, 3, (reset.sum(), 3))
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)


class DynamicsMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 6))
        self.register_buffer('xm', torch.zeros(9)); self.register_buffer('xs', torch.ones(9))
        self.register_buffer('ym', torch.zeros(6)); self.register_buffer('ys', torch.ones(6))

    def forward(self, x):
        return self.net((x - self.xm) / self.xs) * self.ys + self.ym

    def predict(self, v, zb, a):
        with torch.no_grad():
            d = self(torch.tensor(np.c_[v, zb, a], dtype=torch.float32)).numpy()
        zb2 = zb + d[:, 3:]
        return v + d[:, :3], zb2 / np.linalg.norm(zb2, axis=1, keepdims=True)


def train_dynamics(X, Y, epochs=30, seed=0, verbose=True):
    torch.manual_seed(seed)
    m = DynamicsMLP()
    m.xm, m.xs = torch.tensor(X.mean(0)), torch.tensor(X.std(0) + 1e-6)
    m.ym, m.ys = torch.tensor(Y.mean(0)), torch.tensor(Y.std(0) + 1e-6)
    n_val = len(X) // 10
    Xt, Yt, Xv, Yv = map(torch.tensor, (X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]))
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 512):
            i = perm[s:s + 512]
            loss = (((m(Xt[i]) - Yt[i]) / m.ys) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val = (m(Xv) - Yv)[:, :3].norm(dim=1).mean().item() / DT       # accel error m/s^2
        hist.append(dict(epoch=ep + 1, val_accel_err=val))
        if verbose and (ep + 1) % 10 == 0:
            print(f"dynamics epoch {ep + 1} | val acceleration error {val:.3f} m/s^2")
    return m, hist


PILLARS = np.array([[15, 2, 2.5], [28, -3, 2.5], [41, 3, 2.5], [54, -2, 2.5]])   # x, y, radius (vertical cylinders)
START, GOAL = np.array([0.0, 0.0, 10.0]), np.array([68.0, 0.0, 10.0])


def cost_fn(P, A, goal=GOAL, pillars=PILLARS):
    """P [K,H,3] positions, A [K,H,3] actions -> cost [K]."""
    c = np.linalg.norm(P - goal, axis=-1).sum(1) * 0.1 + np.linalg.norm(P[:, -1] - goal, axis=-1)
    for x, y, r in pillars:
        d = np.linalg.norm(P[..., :2] - [x, y], axis=-1) - r
        c += (np.clip(2.0 - d, 0, None) ** 2).sum(1) * 40 + (d < 0.6).sum(1) * 1000
    c += (A ** 2).sum((1, 2)) * 0.002
    return c


class MPPI:
    def __init__(self, model='true', K=256, H=20, sigma=4.0, lam=1.0, seed=0):
        self.model, self.K, self.H, self.sigma, self.lam = model, K, H, sigma, lam
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros((H, 3))
        self.dyn = QuadrotorDynamics(K)

    def _rollout(self, p, v, zb, T, A):
        K, H = A.shape[:2]
        P = np.zeros((K, H, 3)); pp = np.repeat(p[None], K, 0); vv = np.repeat(v[None], K, 0)
        zz = np.repeat(zb[None], K, 0); TT = np.repeat(T, K)
        for h in range(H):
            if self.model == 'true':
                vv, zz, TT = _Quad.step(vv, zz, TT, A[:, h], self.dyn)
            else:
                vv, zz = self.model.predict(vv, zz, A[:, h])
            pp = pp + vv * DT; P[:, h] = pp
        return P

    def act(self, p, v, zb, T):
        eps = self.rng.normal(0, self.sigma, (self.K, self.H, 3))
        A = np.clip(self.U[None] + eps, -8, 8)
        c = cost_fn(self._rollout(p, v, zb, T, A), A)
        w = np.exp(-(c - c.min()) / self.lam); w /= w.sum()
        self.U = (w[:, None, None] * A).sum(0)
        a = self.U[0].copy()
        self.U = np.r_[self.U[1:], self.U[-1:]]
        return a


def pd_controller(p, v, *_):
    return clip_norm((GOAL - p) * 0.8 - v * 1.2, 8)


def fly_slalom(controller, seed=0, steps=250, wind=True):
    rng = np.random.default_rng(seed)
    dyn = QuadrotorDynamics(1)
    w = Wind(mean=(rng.uniform(-2, 2), rng.uniform(-3, 3), 0), gust_std=1.5) if wind else None
    p, v = START.copy(), np.zeros(3)
    path, min_clear, t_goal = [p.copy()], np.inf, None
    for k in range(steps):
        a = controller(p, v, dyn.zb[0], dyn.T)
        wv = w.step(DT, rng) if w else None
        v, zb, T = _Quad.step(v[None], dyn.zb, dyn.T, np.asarray(a)[None], dyn, wv)
        v = v[0]; dyn.zb, dyn.T = zb, T
        p = p + v * DT; path.append(p.copy())
        clear = min(np.linalg.norm(p[:2] - [x, y]) - r for x, y, r in PILLARS)
        min_clear = min(min_clear, clear)
        if np.linalg.norm(p - GOAL) < 1.5:
            t_goal = (k + 1) * DT; break
    return dict(time_s=t_goal if t_goal else np.nan, reached=t_goal is not None,
                min_clearance=float(min_clear), crashed=bool(min_clear < 0), path=np.array(path))
