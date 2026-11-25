"""Learning swarm controller gains with an Evolution Strategy (separable CMA-style ES).

The controller parameters theta = (w_sep, w_ali, w_coh, w_goal, w_obs, separation_radius)
are optimised on the city-navigation task with quadrotor dynamics + wind.  Fitness
(lower is better) trades off speed against safety:

    J = time_to_goal + 40 * near_misses + 300 * building_hits + 2 * mean_cohesion
        + 30 * max(0, 2 m - min_separation)          (safety margin)
        + 200 if the goal is not reached

Each candidate is evaluated on several random seeds so the result generalises.
"""
import numpy as np
from .core import SwarmConfig, make_city
from .swarm import DroneSwarm
from .dynamics import QuadrotorDynamics, Wind

PARAMS = ['w_sep', 'w_ali', 'w_coh', 'w_goal', 'w_obs', 'separation_radius']
LOW = np.array([0.2, 0.0, 0.0, 0.3, 1.0, 3.0])
HIGH = np.array([5.0, 3.0, 3.0, 4.0, 10.0, 8.0])


def evaluate(theta, seeds=(0, 1, 2, 3), n=15, max_steps=500, wind=True, return_details=False):
    kw = dict(zip(PARAMS, np.clip(theta, LOW, HIGH)))
    goal = np.array([80.0, 75.0, 20.0])
    J, det = [], []
    for s in seeds:
        sw = DroneSwarm(SwarmConfig(n_drones=n, **kw), start_center=(12, 12, 10), seed=100 + s)
        sw.buildings, sw.goal = make_city(seed=3, n=7), goal
        sw.dynamics = QuadrotorDynamics(n)
        if wind:
            sw.wind = Wind(mean=(3, 1, 0), gust_std=1.5)
        sw.run(max_steps, until=lambda q: np.linalg.norm(q.pos.mean(0) - goal) < 6)
        d = sw.df().iloc[20:]
        reached = np.linalg.norm(sw.pos.mean(0) - goal) < 6
        j = sw.t + 40 * d.near_misses.sum() + 300 * sw.building_hits + 2 * d.cohesion.mean() \
            + 30 * max(0.0, 2.0 - d.min_separation.min()) + (0 if reached else 200)
        J.append(j)
        det.append(dict(time=sw.t, near_misses=int(d.near_misses.sum()), building_hits=sw.building_hits,
                        min_sep=float(d.min_separation.min()), reached=bool(reached)))
    return (float(np.mean(J)), det) if return_details else float(np.mean(J))


def evolve(theta0, generations=12, popsize=10, sigma0=0.25, seed=0, verbose=True, **eval_kw):
    """(mu, lambda)-ES with per-parameter step sizes in normalised [0, 1] space."""
    rng = np.random.default_rng(seed)
    span = HIGH - LOW
    m = (np.array(theta0, float) - LOW) / span
    sig = np.full(len(m), sigma0)
    mu = popsize // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    hist = []
    best = (evaluate(theta0, **eval_kw), np.array(theta0, float))
    hist.append(dict(gen=0, best=best[0], mean=best[0]))
    for g in range(1, generations + 1):
        z = rng.normal(size=(popsize, len(m)))
        X = np.clip(m + sig * z, 0, 1)
        F = np.array([evaluate(LOW + x * span, **eval_kw) for x in X])
        order = np.argsort(F)
        m_new = (w[:, None] * X[order[:mu]]).sum(0)
        sig = np.clip(0.8 * sig + 0.2 * np.abs(m_new - m) * 2 + 0.02, 0.03, 0.4)
        m = m_new
        if F[order[0]] < best[0]:
            best = (F[order[0]], LOW + X[order[0]] * span)
        hist.append(dict(gen=g, best=best[0], mean=float(F.mean())))
        if verbose:
            print(f"gen {g:2d}  best J = {best[0]:7.2f}   pop mean = {F.mean():7.2f}")
    return best[1], hist
