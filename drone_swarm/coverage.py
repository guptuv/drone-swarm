"""Coverage control (Cortes et al. 2004): drones move to the centroids of their Voronoi
cells, weighted by an importance density, which minimises the locational cost

    H(p) = sum_cells  phi(q) * min_i ||q - p_i||^2

Includes battery-aware persistent coverage: drones below a threshold fly back to a
charging station, recharge, and rejoin; the remaining drones re-partition the area.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .core import clip_norm


def gaussian_density(area=100, centers=((70, 70), (25, 30)), sigmas=(15, 12), weights=(1.0, 0.7), base=0.05, res=1.0):
    xs = np.arange(0, area, res) + res / 2
    gx, gy = np.meshgrid(xs, xs)
    phi = np.full(gx.shape, base)
    for (cx, cy), s, w in zip(centers, sigmas, weights):
        phi += w * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s ** 2))
    return gx, gy, phi


class CoverageSwarm:
    def __init__(self, n=10, area=100, density=None, seed=0, max_speed=4.0, dt=0.5,
                 battery=False, charger=(5.0, 5.0), drain_pct_s=0.35, charge_pct_s=4.0, rth_pct=25.0):
        self.rng = np.random.default_rng(seed)
        self.n, self.A, self.vmax, self.dt = n, area, max_speed, dt
        self.gx, self.gy, self.phi = density if density is not None else gaussian_density(area)
        self.cells = np.c_[self.gx.ravel(), self.gy.ravel()]
        self.w = self.phi.ravel()
        self.pos = np.array(charger) + self.rng.uniform(0, 10, (n, 2))
        self.battery_on = battery
        self.charger = np.array(charger, float)
        self.batt = self.rng.uniform(60, 100, n) if battery else np.full(n, 100.0)
        self.drain, self.charge, self.rth = drain_pct_s, charge_pct_s, rth_pct
        self.state = np.array(['cover'] * n, dtype=object)          # cover / return / charge
        self.t = 0.0
        self.traj, self.cost, self.active_hist, self.batt_hist, self.owner_hist = [], [], [], [], []
        self._log(self._owners())

    def _owners(self):
        act = np.where(self.state == 'cover')[0]
        if len(act) == 0:
            return act, np.zeros(len(self.cells), int)
        return act, cdist(self.cells, self.pos[act]).argmin(1)

    def locational_cost(self, act=None, owner=None):
        if act is None:
            act, owner = self._owners()
        if len(act) == 0:
            return np.nan
        d2 = ((self.cells - self.pos[act][owner]) ** 2).sum(1)
        return float((self.w * d2).sum() / self.w.sum())

    def step(self, gain=1.0):
        act, owner = self._owners()
        v = np.zeros_like(self.pos)
        for k, i in enumerate(act):                                  # move to weighted centroid
            m = owner == k
            mass = self.w[m].sum()
            if mass > 0:
                c = (self.cells[m] * self.w[m, None]).sum(0) / mass
                v[i] = gain * (c - self.pos[i])
        if self.battery_on:
            for i in range(self.n):
                if self.state[i] == 'cover' and self.batt[i] < self.rth:
                    self.state[i] = 'return'
                if self.state[i] == 'return':
                    d = self.charger - self.pos[i]
                    if np.linalg.norm(d) < 1.5:
                        self.state[i] = 'charge'
                    v[i] = d / (np.linalg.norm(d) + 1e-9) * self.vmax
                if self.state[i] == 'charge':
                    v[i] = 0
                    self.batt[i] = min(100, self.batt[i] + self.charge * self.dt)
                    if self.batt[i] >= 99:
                        self.state[i] = 'cover'
            self.batt -= np.where(self.state == 'charge', 0, self.drain * self.dt)
        self.pos = np.clip(self.pos + clip_norm(v, self.vmax) * self.dt, 0, self.A)
        self.t += self.dt
        self._log(self._owners())

    def _log(self, ao):
        act, owner = ao
        self.traj.append(self.pos.copy())
        self.cost.append(self.locational_cost(act, owner))
        self.active_hist.append(len(act))
        self.batt_hist.append(self.batt.copy())
        full = np.full(len(self.cells), -1)
        if len(act):
            full = act[owner]
        self.owner_hist.append(full)

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self
