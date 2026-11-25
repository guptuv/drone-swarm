"""Search & Rescue: PSO vs lawnmower vs random search strategies."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .core import clip_norm, unit

# ==============================================================
#  SEARCH & RESCUE  (2D, drones at fixed altitude)
#  Survivors emit a weak noisy signal (e.g. phone RF).  Strategies:
#   'pso'       : PSO on signal + frontier exploration when signal is weak
#   'lawnmower' : classic boustrophedon sweep, strips split between drones
#   'random'    : random waypoint
# ==============================================================
class SearchAndRescue:
    def __init__(self, n_drones=12, area=100, n_survivors=5, strategy='pso', seed=0,
                 sensor_r=5.0, sigma=14.0, noise=0.03, max_speed=5.0, dt=0.5, comm_r=25.0):
        self.rng = np.random.default_rng(seed)
        self.n, self.A, self.strategy = n_drones, area, strategy
        self.sensor_r, self.detect_r, self.sigma, self.noise = sensor_r, sensor_r, sigma, noise
        self.comm_r = comm_r
        self.vmax, self.dt, self.t = max_speed, dt, 0.0
        self.pos = np.array([5.0, 5.0]) + self.rng.uniform(0, 4, (n_drones, 2))
        self.vel = np.zeros((n_drones, 2))
        self.surv = self.rng.uniform(10, area - 10, (n_survivors, 2))
        self.found = np.zeros(n_survivors, bool)
        self.found_t = np.full(n_survivors, np.nan)
        self.cov = np.zeros((area, area), bool)       # seen by camera (sensor_r)
        self.heard = np.zeros((area, area), bool)     # listened by RF receiver (hear_r) -> PSO frontier map
        self.hear_r = 0.7 * sigma * np.sqrt(2 * np.log(1 / 0.15))   # radius where signal > threshold
        gx, gy = np.meshgrid(np.arange(area) + .5, np.arange(area) + .5)
        self.cells = np.c_[gx.ravel(), gy.ravel()]
        # PSO memory
        self.pb_pos, self.pb_val = self.pos.copy(), self.signal(self.pos)
        self.explore_tgt = np.full((n_drones, 2), np.nan)
        if strategy == 'lawnmower':
            self.paths = self._lawnmower_paths(); self.wp = np.zeros(n_drones, int)
        self.traj, self.cov_hist, self.found_hist = [], [], []
        self._update_coverage(); self._save()

    # ---- sensing ----
    def true_signal(self, P):
        act = self.surv[~self.found]
        if len(act) == 0:
            return np.zeros(len(P))
        return np.exp(-cdist(P, act) ** 2 / (2 * self.sigma ** 2)).max(1)

    def signal(self, P):
        return self.true_signal(P) + self.rng.normal(0, self.noise, len(P))

    def _update_coverage(self):
        d = cdist(self.cells, self.pos).min(1)
        self.cov.ravel()[d < self.sensor_r] = True
        self.heard.ravel()[d < self.hear_r] = True

    # ---- strategies ----
    def _lawnmower_paths(self):
        w = 2 * self.sensor_r * 0.95
        xs = np.arange(w / 2, self.A, w)
        strips = [np.array([[x, 3], [x, self.A - 3]] if k % 2 == 0 else [[x, self.A - 3], [x, 3]]) for k, x in enumerate(xs)]
        paths = [[] for _ in range(self.n)]
        for k, s in enumerate(strips):
            paths[k % self.n].extend(s)
        return [np.array(p) for p in paths]

    def _pick_frontier(self, i):
        free = self.cells[~self.heard.ravel()]
        if len(free) == 0:
            return self.rng.uniform(0, self.A, 2)
        cand = free[self.rng.choice(len(free), min(150, len(free)), replace=False)]
        others = np.delete(np.r_[self.pos, np.nan_to_num(self.explore_tgt, nan=-1e3)], [i, self.n + i], 0)
        score = -np.linalg.norm(cand - self.pos[i], axis=1) + 0.6 * cdist(cand, others).min(1)
        return cand[score.argmax()]

    def _desired_velocity(self):
        v = np.zeros_like(self.pos)
        if self.strategy == 'pso':
            s = self.signal(self.pos)
            better = s > self.pb_val
            self.pb_pos[better], self.pb_val[better] = self.pos[better], s[better]
            D = cdist(self.pos, self.pos)
            for i in range(self.n):
                nb = np.where(D[i] < self.comm_r)[0]            # only drones in radio range share info
                j = nb[self.pb_val[nb].argmax()]
                if self.pb_val[j] > 0.15:                       # signal heard -> PSO converge on source
                    r1, r2 = self.rng.random(2)
                    v[i] = 0.5 * self.vel[i] + 1.2 * r1 * (self.pb_pos[i] - self.pos[i]) \
                           + 1.5 * r2 * (self.pb_pos[j] - self.pos[i])
                    # turbulence: keep probing around the best point (bigger when signal is weak)
                    v[i] += self.rng.normal(0, 1, 2) * self.vmax * (1.0 - min(self.pb_val[j], 1.0)) * 1.5
                else:                                           # silence -> explore unvisited frontier
                    t = self.explore_tgt[i]
                    if np.isnan(t).any() or np.linalg.norm(t - self.pos[i]) < 2 or self.heard[int(t[1]), int(t[0])]:
                        self.explore_tgt[i] = t = self._pick_frontier(i)
                    v[i] = unit(t - self.pos[i]) * self.vmax
        elif self.strategy == 'lawnmower':
            for i in range(self.n):
                p = self.paths[i]
                if len(p) == 0 or self.wp[i] >= len(p):        # sweep done -> random patrol
                    t = self.explore_tgt[i]
                    if np.isnan(t).any() or np.linalg.norm(t - self.pos[i]) < 2:
                        self.explore_tgt[i] = t = self.rng.uniform(0, self.A, 2)
                    v[i] = unit(t - self.pos[i]) * self.vmax
                    continue
                d = p[self.wp[i]] - self.pos[i]
                if np.linalg.norm(d) < 1.5: self.wp[i] += 1
                v[i] = unit(d) * self.vmax
        else:  # random waypoint
            for i in range(self.n):
                t = self.explore_tgt[i]
                if np.isnan(t).any() or np.linalg.norm(t - self.pos[i]) < 2:
                    self.explore_tgt[i] = t = self.rng.uniform(0, self.A, 2)
                v[i] = unit(t - self.pos[i]) * self.vmax
        # inter-drone repulsion (collision avoidance)
        diff = self.pos[:, None] - self.pos[None]
        d = np.linalg.norm(diff, axis=-1); np.fill_diagonal(d, np.inf)
        v += (diff * np.where(d < 3, 4 / (d ** 2 + .1), 0)[..., None]).sum(1)
        return clip_norm(v, self.vmax)

    # ---- simulation ----
    def step(self):
        self.vel = self._desired_velocity()
        self.pos = np.clip(self.pos + self.vel * self.dt, 0, self.A - 1e-3)
        self.t += self.dt
        self._update_coverage()
        hit = (cdist(self.surv, self.pos).min(1) < self.detect_r) & ~self.found
        if hit.any():
            self.found[hit], self.found_t[hit] = True, self.t
            self.pb_pos, self.pb_val = self.pos.copy(), self.signal(self.pos)   # field changed -> reset memory
        self._save()

    def _save(self):
        self.traj.append(self.pos.copy()); self.cov_hist.append(self.cov.mean() * 100)
        self.found_hist.append(self.found.copy())

    def run(self, max_steps=1200):
        while not self.found.all() and len(self.traj) <= max_steps:
            self.step()
        return dict(strategy=self.strategy, all_found=bool(self.found.all()),
                    time_all_s=self.t if self.found.all() else np.nan,
                    mean_find_time_s=np.nanmean(self.found_t), coverage_pct=self.cov.mean() * 100)

