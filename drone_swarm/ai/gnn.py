"""Graph Neural Network that learns a *decentralised* flocking controller by imitating a
*centralised* expert (in the spirit of Tolstaya et al., "Learning Decentralized Controllers
for Robot Swarms with Graph Neural Networks", CoRL 2019).

* Expert (centralised): u_i = -sum_{all j} (v_i - v_j) - sum_{j} grad U(r_ij)
  -> needs every drone's velocity: not possible on a real swarm with short-range radios.
* Local controller: same law but only with 1-hop neighbours (what a drone can sense).
* GNN: drones exchange learned messages with radio neighbours for K rounds per control step,
  so information propagates K hops; trained with behaviour cloning + DAgger.

Metric: velocity-alignment cost  C = mean_i ||v_i - mean(v)||^2  (lower = better flock).
"""
import numpy as np
import torch
import torch.nn as nn


class FlockEnv:
    def __init__(self, n=40, comm_r=1.0, dt=0.01, a_max=30.0, v_init=3.0, v_bias=3.0, device='cpu'):
        self.n, self.R, self.dt, self.amax = n, comm_r, dt, a_max
        self.v_init, self.v_bias = v_init, v_bias

    def reset(self, rng):
        n = self.n
        # random connected-ish initial placement: points in a disc whose size grows with n
        rad = 0.35 * np.sqrt(n)
        while True:
            p = rng.uniform(-rad, rad, (n, 2))
            d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(n) * 9
            if d.min() > 0.1:
                break
        v = rng.uniform(-self.v_init, self.v_init, (n, 2)) + rng.uniform(-self.v_bias, self.v_bias, (1, 2))
        self.p, self.v = p, v
        return self

    def adjacency(self):
        d = np.linalg.norm(self.p[:, None] - self.p[None], axis=-1)
        A = (d < self.R).astype(float); np.fill_diagonal(A, 0)
        return A

    def expert(self):
        """Centralised controller: global velocity consensus + local collision potential."""
        n = self.n
        u = -(n * self.v - self.v.sum(0)) / n * 3.0
        r = self.p[:, None] - self.p[None]                       # i - j
        d2 = (r ** 2).sum(-1) + np.eye(n)
        A = self.adjacency()
        u += (A[..., None] * r * (2 / d2 ** 2 + 2 / d2)[..., None]).sum(1) * 0.05
        return np.clip(u, -self.amax, self.amax)

    def local(self):
        """Same law with neighbours only."""
        A = self.adjacency()
        deg = A.sum(1, keepdims=True)
        u = -(deg * self.v - A @ self.v) / np.maximum(deg, 1) * 3.0
        r = self.p[:, None] - self.p[None]
        d2 = (r ** 2).sum(-1) + np.eye(self.n)
        u += (A[..., None] * r * (2 / d2 ** 2 + 2 / d2)[..., None]).sum(1) * 0.05
        return np.clip(u, -self.amax, self.amax)

    def features(self):
        """Per-drone local observations (only from neighbours)."""
        A = self.adjacency()
        r = self.p[:, None] - self.p[None]
        d2 = (r ** 2).sum(-1) + np.eye(self.n)
        f1 = (A[..., None] * (self.v[:, None] - self.v[None])).sum(1)
        f2 = (A[..., None] * r / d2[..., None] ** 2).sum(1)
        f3 = (A[..., None] * r / d2[..., None]).sum(1)
        return np.c_[f1, f2 * 0.01, f3 * 0.1, self.v], A

    def step(self, u):
        self.v = self.v + np.clip(u, -self.amax, self.amax) * self.dt
        self.p = self.p + self.v * self.dt

    def cost(self):
        return float(((self.v - self.v.mean(0)) ** 2).sum(1).mean())


class FlockGNN(nn.Module):
    """K rounds of message passing over the radio graph (shared weights per drone)."""

    def __init__(self, in_dim=8, hidden=64, K=3):
        super().__init__()
        self.K = K
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.msg = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(K)])
        self.upd = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(K)])
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, x, A):
        """x [B,n,F], A [B,n,n] (0/1) -> u [B,n,2]"""
        h = self.enc(x)
        deg = A.sum(-1, keepdim=True).clamp(min=1)
        for k in range(self.K):
            m = (A @ torch.tanh(self.msg[k](h))) / deg              # mean of neighbour messages
            h = torch.tanh(self.upd[k](torch.cat([h, m], -1)))
        return self.dec(h)


def _to_t(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def rollout(env, controller, rng, steps=200, model=None, record=False):
    env.reset(rng)
    costs, traj, X, Y, As = [], [], [], [], []
    for _ in range(steps):
        x, A = env.features()
        if controller == 'expert':
            u = env.expert()
        elif controller == 'local':
            u = env.local()
        else:
            with torch.no_grad():
                u = model(_to_t(x)[None], _to_t(A)[None])[0].numpy() * 10
        if record:
            X.append(x); As.append(A); Y.append(env.expert() / 10)
            traj.append(env.p.copy())
        env.step(u)
        costs.append(env.cost())
    return np.array(costs), dict(X=X, A=As, Y=Y, traj=np.array(traj) if traj else None)


def train_gnn(n=40, epochs=30, episodes_per_epoch=6, steps=200, K=3, lr=1e-3, seed=0, verbose=True):
    """Behaviour cloning, then DAgger: roll out the learner, label states with the expert."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    env = FlockEnv(n=n)
    model = FlockGNN(K=K)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = []
    hist = []
    for ep in range(epochs):
        beta = max(0.0, 1.0 - ep / (epochs * 0.5))                   # expert mixing (DAgger)
        for _ in range(episodes_per_epoch):
            env.reset(rng)
            for t in range(steps):
                x, A = env.features()
                y = env.expert()
                data.append((x, A, y / 10))
                if rng.random() < beta:
                    u = y
                else:
                    with torch.no_grad():
                        u = model(_to_t(x)[None], _to_t(A)[None])[0].numpy() * 10
                env.step(u)
        data = data[-20000:]
        idx = rng.permutation(len(data))
        tot = 0
        for s in range(0, len(idx), 64):
            b = [data[i] for i in idx[s:s + 64]]
            X = _to_t(np.stack([d[0] for d in b])); A = _to_t(np.stack([d[1] for d in b])); Y = _to_t(np.stack([d[2] for d in b]))
            loss = ((model(X, A) - Y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        c, _ = rollout(env, 'gnn', np.random.default_rng(999), steps=steps, model=model)
        hist.append(dict(epoch=ep + 1, loss=tot / len(idx), beta=beta, gnn_cost=float(c.mean())))
        if verbose and ((ep + 1) % 5 == 0 or ep == 0):
            print(f"epoch {ep + 1:3d} | loss {tot / len(idx):.5f} | beta {beta:.2f} | GNN flock cost {c.mean():.3f}")
    return model, hist


def compare(model, sizes=(20, 40, 80), episodes=10, steps=200, seed=7):
    rows = []
    for n in sizes:
        env = FlockEnv(n=n)
        for name in ['expert', 'local', 'gnn']:
            cs = [rollout(env, name, np.random.default_rng(seed + e), steps=steps, model=model)[0].mean() for e in range(episodes)]
            rows.append(dict(n=n, controller=name, cost_mean=float(np.mean(cs)), cost_std=float(np.std(cs))))
    return rows
