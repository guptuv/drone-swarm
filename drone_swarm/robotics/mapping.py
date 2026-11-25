"""Multi-drone occupancy-grid mapping with simulated 2D lidar and frontier-based exploration.

* Each drone carries a 360 deg lidar (36 beams, 15 m range, noisy).
* Log-odds occupancy grid update (Thrun, Probabilistic Robotics) shared by the team.
* Frontier exploration: drones go to the nearest unclaimed frontier cluster
  (boundary between known-free and unknown space); A* on the current map.
Baseline: random-walk exploration.
"""
import numpy as np
from .planning import astar

L_OCC, L_FREE, L_MAX = 0.85, -0.4, 4.0


def make_world(seed=0, size=80, n_walls=10):
    rng = np.random.default_rng(seed)
    W = np.zeros((size, size), bool)
    W[0, :] = W[-1, :] = W[:, 0] = W[:, -1] = True
    for _ in range(n_walls):
        x, y = rng.integers(8, size - 8, 2)
        if rng.random() < 0.5:
            W[x, y:y + rng.integers(8, 25)] = True
        else:
            W[x:x + rng.integers(8, 25), y] = True
    for _ in range(6):
        x, y = rng.integers(8, size - 12, 2); s = rng.integers(3, 7)
        W[x:x + s, y:y + s] = True
    W[2:8, 2:8] = False
    return W


def lidar(world, pos, n_beams=36, max_r=15.0, rng=None):
    angs = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
    hits = []
    for a in angs:
        d = np.array([np.cos(a), np.sin(a)])
        r_hit = max_r; hit = False
        for r in np.arange(0.5, max_r, 0.5):
            c = (pos + d * r).astype(int)
            if not (0 <= c[0] < world.shape[0] and 0 <= c[1] < world.shape[1]) or world[c[0], c[1]]:
                r_hit, hit = r, True; break
        if rng is not None:
            r_hit = r_hit + rng.normal(0, 0.1)
        hits.append((a, r_hit, hit))
    return hits


def update_map(logodds, pos, scan):
    for a, r, hit in scan:
        d = np.array([np.cos(a), np.sin(a)])
        for s in np.arange(0, max(r - 0.5, 0), 0.5):
            c = (pos + d * s).astype(int)
            if 0 <= c[0] < logodds.shape[0] and 0 <= c[1] < logodds.shape[1]:
                logodds[c[0], c[1]] = max(-L_MAX, logodds[c[0], c[1]] + L_FREE)
        if hit:
            c = (pos + d * r).astype(int)
            if 0 <= c[0] < logodds.shape[0] and 0 <= c[1] < logodds.shape[1]:
                logodds[c[0], c[1]] = min(L_MAX, logodds[c[0], c[1]] + L_OCC)


def frontiers(logodds):
    free = logodds < -0.5; unknown = np.abs(logodds) < 0.2
    nb = np.zeros_like(unknown)
    nb[1:] |= unknown[:-1]; nb[:-1] |= unknown[1:]; nb[:, 1:] |= unknown[:, :-1]; nb[:, :-1] |= unknown[:, 1:]
    return np.argwhere(free & nb)


def explore(world, n_drones=3, steps=250, strategy='frontier', seed=0, speed=1.0):
    rng = np.random.default_rng(seed)
    lo = np.zeros(world.shape)
    pos = np.array([[3.5 + i, 3.5] for i in range(n_drones)], float)
    paths = [[] for _ in range(n_drones)]; plans = [None] * n_drones
    known_hist = []
    free_total = (~world).sum()
    for t in range(steps):
        for i in range(n_drones):
            update_map(lo, pos[i], lidar(world, pos[i], rng=rng))
        for i in range(n_drones):
            if strategy == 'frontier':
                if plans[i] is None or len(plans[i]) == 0 or t % 15 == 0:
                    F = frontiers(lo)
                    if len(F):
                        claimed = [p[-1] for j, p in enumerate(plans) if j != i and p is not None and len(p)]
                        d = np.linalg.norm(F + 0.5 - pos[i], axis=1)
                        for c in claimed:
                            d = d + 25 * (np.linalg.norm(F + 0.5 - c, axis=1) < 8)
                        tgt = F[d.argmin()] + 0.5
                        occ = lo > 0.5
                        path, _ = astar(occ, pos[i], tgt)
                        plans[i] = list(path[1:]) if path is not None and len(path) > 1 else None
                if plans[i]:
                    step = plans[i].pop(0)
                    if not world[int(step[0]), int(step[1])]:
                        pos[i] = step
            else:                                             # random walk
                for _ in range(10):
                    cand = pos[i] + rng.normal(0, 1, 2) * speed
                    c = cand.astype(int)
                    if 0 <= c[0] < world.shape[0] and 0 <= c[1] < world.shape[1] and not world[c[0], c[1]]:
                        pos[i] = cand; break
            paths[i].append(pos[i].copy())
        known = ((np.abs(lo) > 0.2) & ~world).sum() / free_total
        known_hist.append(known)
    return dict(logodds=lo, paths=[np.array(p) for p in paths], known=np.array(known_hist))
