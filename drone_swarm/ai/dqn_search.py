"""Deep Q-Network for probabilistic search (search theory meets deep RL).

A survivor is somewhere on a G x G grid.  Rescuers give a prior belief map (e.g. from the
last known position / terrain) - here a random mixture of Gaussian blobs every episode.
Each step the drone moves (N/E/S/W) and looks at its cell: if the survivor is there it is
detected with probability p_d.  The belief is updated with Bayes' rule.

Reward = probability of detecting the survivor in this step = p_d * b(cell),
so the return equals the (discounted) cumulative probability of success.

Agents compared: Double + Dueling DQN (CNN on the egocentric belief map), greedy (best neighbour),
lawnmower sweep, random walk.
"""
import numpy as np
import torch
import torch.nn as nn

MOVES = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])


class SearchGrid:
    def __init__(self, batch=64, G=16, T=60, p_d=0.8, seed=0):
        self.B, self.G, self.T, self.pd = batch, G, T, p_d
        self.rng = np.random.default_rng(seed)
        xs = np.arange(G) + 0.5
        self.gx, self.gy = np.meshgrid(xs, xs, indexing='ij')

    def _prior(self):
        b = np.zeros((self.B, self.G, self.G))
        for k in range(2):
            c = self.rng.uniform(2, self.G - 2, (self.B, 2))
            s = self.rng.uniform(1.0, 3.0, (self.B, 1, 1))
            w = self.rng.uniform(0.3, 1.0, (self.B, 1, 1))
            b += w * np.exp(-((self.gx - c[:, 0, None, None]) ** 2 + (self.gy - c[:, 1, None, None]) ** 2) / (2 * s ** 2))
        b += 0.002
        return b / b.sum((1, 2), keepdims=True)

    def reset(self):
        self.b = self._prior()
        self.b0 = self.b.copy()
        self.pos = np.zeros((self.B, 2), int)
        self.t = 0
        self.found = np.zeros(self.B)
        self._sense()
        return self.obs()

    def _sense(self):
        i = np.arange(self.B)
        p = self.b[i, self.pos[:, 0], self.pos[:, 1]] * self.pd
        self.b[i, self.pos[:, 0], self.pos[:, 1]] *= (1 - self.pd)
        self.found += p
        return p

    def obs(self):
        """Egocentric view: belief map shifted so the drone is at the centre (2G-1 x 2G-1),
        plus a channel marking cells inside the search area."""
        G, B = self.G, self.B
        S = 2 * G - 1
        o = np.zeros((B, 2, S, S), np.float32)
        for k in range(B):
            x0, y0 = G - 1 - self.pos[k, 0], G - 1 - self.pos[k, 1]
            o[k, 0, x0:x0 + G, y0:y0 + G] = self.b[k] * G * G / 4
            o[k, 1, x0:x0 + G, y0:y0 + G] = 1
        return o

    def step(self, a):
        self.pos = np.clip(self.pos + MOVES[a], 0, self.G - 1)
        r = self._sense()
        self.t += 1
        return self.obs(), r, self.t >= self.T


class QNet(nn.Module):
    """Dueling CNN Q-network on the egocentric belief map."""

    def __init__(self, G=16):
        super().__init__()
        S = 2 * G - 1
        self.f = nn.Sequential(nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.ReLU(),
                               nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        n = self.f(torch.zeros(1, 2, S, S)).shape[1]
        self.v = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 1))
        self.a = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, x):
        h = self.f(x)
        a = self.a(h)
        return self.v(h) + a - a.mean(1, keepdim=True)


def greedy_policy(env):
    best = np.full(env.B, -1.0); act = np.zeros(env.B, int)
    for k, m in enumerate(MOVES):
        p = np.clip(env.pos + m, 0, env.G - 1)
        v = env.b[np.arange(env.B), p[:, 0], p[:, 1]]
        moved = (p != env.pos).any(1)
        v = np.where(moved, v, -1)
        better = v > best
        best[better], act[better] = v[better], k
    return act


def lawnmower_policy(env):
    x, y = env.pos[:, 0], env.pos[:, 1]
    up = (x % 2 == 0)
    act = np.where(up, 0, 2)
    at_end = np.where(up, y == env.G - 1, y == 0)
    return np.where(at_end, 1, act)


def run_policy(policy, episodes=256, seed=1, G=16, T=60, model=None):
    env = SearchGrid(batch=episodes, G=G, T=T, seed=seed)
    o = env.reset(); done = False
    curve = [env.found.mean()]
    paths = [env.pos[0].copy()]
    while not done:
        if policy == 'dqn':
            with torch.no_grad():
                a = model(torch.tensor(o)).argmax(1).numpy()
        elif policy == 'greedy':
            a = greedy_policy(env)
        elif policy == 'lawnmower':
            a = lawnmower_policy(env)
        else:
            a = env.rng.integers(0, 4, env.B)
        o, r, done = env.step(a)
        curve.append(env.found.mean()); paths.append(env.pos[0].copy())
    return dict(p_detect=float(env.found.mean()), curve=np.array(curve), path=np.array(paths), prior=env.b0[0])


def train_dqn(steps=1500, batch_envs=64, G=16, T=60, gamma=0.97, lr=5e-4, buffer=60000, seed=0, verbose=True):
    """steps = vectorised env steps (each adds batch_envs transitions)."""
    torch.manual_seed(seed)
    env = SearchGrid(batch=batch_envs, G=G, T=T, seed=seed)
    q, qt = QNet(G), QNet(G)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    S = np.zeros((buffer, 2, 2 * G - 1, 2 * G - 1), np.float32); S2 = np.zeros_like(S)
    Aa = np.zeros(buffer, int); Rr = np.zeros(buffer, np.float32); Dd = np.zeros(buffer, np.float32)
    ptr = full = 0
    o = env.reset(); hist = []
    best_score, best_state = -1.0, None
    for it in range(1, steps + 1):
        eps = max(0.05, 1 - it / (steps * 0.5))
        with torch.no_grad():
            a = q(torch.tensor(o)).argmax(1).numpy()
        rand = env.rng.random(env.B) < eps
        a = np.where(rand, env.rng.integers(0, 4, env.B), a)
        o2, r, done = env.step(a)
        idx = (ptr + np.arange(env.B)) % buffer
        S[idx], Aa[idx], Rr[idx], S2[idx], Dd[idx] = o, a, r * 10, o2, float(done)
        ptr = (ptr + env.B) % buffer; full = min(full + env.B, buffer)
        o = env.reset() if done else o2
        if full > 2000:
            for _ in range(2):
                bi = env.rng.integers(0, full, 256)
                s, s2 = torch.tensor(S[bi]), torch.tensor(S2[bi])
                with torch.no_grad():
                    a2 = q(s2).argmax(1, keepdim=True)                       # Double DQN
                    tgt = torch.tensor(Rr[bi]) + gamma * (1 - torch.tensor(Dd[bi])) * qt(s2).gather(1, a2).squeeze(1)
                qv = q(s).gather(1, torch.tensor(Aa[bi])[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(qv, tgt)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10); opt.step()
            with torch.no_grad():                                         # Polyak target update
                for p, pt in zip(q.parameters(), qt.parameters()):
                    pt.mul_(0.995).add_(0.005 * p)
        if it % 100 == 0:
            ev = run_policy('dqn', episodes=128, seed=999, G=G, T=T, model=q)['p_detect']
            hist.append(dict(step=it, transitions=it * env.B, eps=eps, p_detect=ev))
            if ev > best_score:                                           # keep the best checkpoint
                best_score = ev
                best_state = {k: v.clone() for k, v in q.state_dict().items()}
            if verbose:
                print(f"step {it:5d} | eps {eps:.2f} | eval P(detect) {ev:.3f}")
    if best_state is not None:
        q.load_state_dict(best_state)
    return q, hist
