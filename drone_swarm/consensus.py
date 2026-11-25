"""Distributed consensus on a proximity communication graph.

    x_i <- x_i + eps * sum_{j in N_i} (x_j - x_i + (d_i - d_j))

With all offsets d = 0 the drones rendezvous; with formation offsets d they reach a
formation without any central coordinator.  Convergence speed depends on the algebraic
connectivity lambda_2 of the graph Laplacian (Olfati-Saber & Murray 2004).
"""
import numpy as np
from scipy.spatial.distance import cdist


def laplacian(pos, comm_range):
    A = (cdist(pos, pos) < comm_range).astype(float)
    np.fill_diagonal(A, 0)
    return np.diag(A.sum(1)) - A, A


def algebraic_connectivity(L):
    return float(np.sort(np.linalg.eigvalsh(L))[1])


def run_consensus(pos0, offsets=None, comm_range=30.0, eps=0.05, steps=300, max_step=1.0):
    """Returns trajectory (steps+1, n, d), disagreement history and lambda_2 history."""
    x = np.array(pos0, float)
    d = np.zeros_like(x) if offsets is None else np.array(offsets, float)
    traj, dis, lam = [x.copy()], [], []
    for _ in range(steps):
        L, A = laplacian(x, comm_range)
        lam.append(algebraic_connectivity(L))
        z = x - d                                   # consensus variable
        dis.append(float(np.linalg.norm(z - z.mean(0), axis=1).mean()))
        u = -eps * (L @ z)
        n = np.linalg.norm(u, axis=1, keepdims=True)
        u = u * np.minimum(1, max_step / (n + 1e-9))   # actuator limit
        x = x + u
        traj.append(x.copy())
    return np.array(traj), np.array(dis), np.array(lam)


def leader_follower(pos0, leader_path, comm_range=30.0, eps=0.08, steps=300, max_step=1.5):
    """Drone 0 follows a given path; others only see neighbours (pinning consensus)."""
    x = np.array(pos0, float)
    traj, err = [x.copy()], []
    for k in range(steps):
        x[0] = leader_path[min(k, len(leader_path) - 1)]
        L, A = laplacian(x, comm_range)
        u = -eps * (L @ x)
        u[0] = 0
        n = np.linalg.norm(u, axis=1, keepdims=True)
        x = x + u * np.minimum(1, max_step / (n + 1e-9))
        traj.append(x.copy())
        err.append(float(np.linalg.norm(x[1:] - x[0], axis=1).mean()))
    return np.array(traj), np.array(err)
