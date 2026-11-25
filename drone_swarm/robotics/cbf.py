"""Control Barrier Function (CBF) safety filter for multi-drone collision avoidance.

For single-integrator drones (velocity commands u_i) and pairwise safety sets
    h_ij(x) = ||p_i - p_j||^2 - D^2 >= 0
the CBF condition  dh/dt + alpha h >= 0  is linear in (u_i, u_j):
    2 (p_i - p_j)^T (u_i - u_j) + alpha h_ij >= 0.
Each drone solves  min ||u_i - u_nom_i||^2  subject to its share (half) of every constraint
with its neighbours - a small QP solved here exactly by Hildreth's dual coordinate ascent.
This is fully decentralised and guarantees safety (up to discretisation / speed limits).
Symmetric encounters can deadlock a pure CBF; a right-hand bias on the nominal command
when neighbours are close resolves it (cf. Wang, Ames & Egerstedt 2017).
"""
import numpy as np


def qp_project(u_nom, A, b, iters=200):
    """min ||u - u_nom||^2  s.t.  A u >= b   (Hildreth dual coordinate ascent)."""
    if len(b) == 0:
        return u_nom
    lam = np.zeros(len(b)); u = u_nom.copy()
    AA = (A ** 2).sum(1) + 1e-12
    for _ in range(iters):
        changed = False
        for k in range(len(b)):
            viol = b[k] - A[k] @ u
            new = max(0.0, lam[k] + viol / AA[k])
            if abs(new - lam[k]) > 1e-10:
                u = u + (new - lam[k]) * A[k]; lam[k] = new; changed = True
        if not changed:
            break
    return u


def cbf_filter(p, u_nom, D=2.0, alpha=1.0, sense_r=10.0, v_max=None):
    n = len(p); u = u_nom.copy()
    for i in range(n):
        A, b = [], []
        for j in range(n):
            if i == j:
                continue
            d = p[i] - p[j]
            if d @ d > sense_r ** 2:
                continue
            h = d @ d - D ** 2
            # reciprocal split: i enforces 2 d^T u_i >= -alpha h / 2, j enforces the mirror half,
            # so together 2 d^T (u_i - u_j) >= -alpha h  (the full CBF condition)
            A.append(2 * d); b.append(-0.5 * alpha * h)
        if A:
            u[i] = qp_project(u_nom[i], np.array(A), np.array(b))
    if v_max is not None:
        s = np.linalg.norm(u, axis=1, keepdims=True)
        u = u * np.minimum(1, v_max / (s + 1e-9))
    return u


def antipodal_swap(n=16, radius=20.0, use_cbf=True, steps=500, dt=0.05, v_max=4.0, D=2.0, seed=0):
    """Drones on a circle swap to the opposite point: nominal controller = go straight."""
    rng = np.random.default_rng(seed)
    a = 2 * np.pi * np.arange(n) / n + rng.normal(0, 0.02, n)
    p = radius * np.c_[np.cos(a), np.sin(a)]; goal = -p.copy()
    traj, min_d = [p.copy()], []
    for _ in range(steps):
        e = goal - p
        u_nom = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9) * np.minimum(v_max, np.linalg.norm(e, axis=1, keepdims=True))
        u_nom += rng.normal(0, 0.05, u_nom.shape)                       # breaks perfect symmetry
        if use_cbf:
            # deadlock resolution: when neighbours are close, bend the nominal command to the right
            dd = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(n) * 1e9
            crowd = np.clip((3 * D - dd.min(1)) / (2 * D), 0, 1)[:, None]
            right = np.c_[u_nom[:, 1], -u_nom[:, 0]]
            u_nom = u_nom + 1.5 * crowd * right
            u = cbf_filter(p, u_nom, D=D, v_max=v_max)
        else:
            u = u_nom
        p = p + u * dt; traj.append(p.copy())
        dd = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(n) * 1e9
        min_d.append(dd.min())
    arrived = np.linalg.norm(p - goal, axis=1) < 0.5
    return dict(traj=np.array(traj), min_dist=np.array(min_d), arrived=float(arrived.mean()),
                violations=int((np.array(min_d) < D * 0.95).sum()))
