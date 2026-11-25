"""Config, geometry helpers and building obstacles."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
# ==============================================================
#  CORE: config, helpers, buildings (obstacles)
# ==============================================================
@dataclass
class SwarmConfig:
    n_drones: int = 25
    world: tuple = (100.0, 100.0, 50.0)   # x, y, z limits (metres)
    dt: float = 0.1                       # time step (s)
    max_speed: float = 8.0                # m/s
    min_speed: float = 1.5                # flocking drones keep moving
    max_accel: float = 7.0                # m/s^2 (motor limit)
    # Boids (Reynolds 1987)
    neighbor_radius: float = 15.0
    separation_radius: float = 4.5
    w_sep: float = 2.0
    w_ali: float = 1.0
    w_coh: float = 0.7
    # Goal / obstacles / boundaries
    w_goal: float = 1.3
    w_obs: float = 5.0
    obs_margin: float = 7.0
    w_bound: float = 3.0
    bound_margin: float = 6.0
    min_altitude: float = 3.0
    # Formation PD gains
    kp: float = 1.0
    kd: float = 2.0
    formation_speed: float = 4.0          # speed of the formation centre (m/s)


@dataclass
class Building:
    """Vertical cylinder obstacle (tower / tree / pole)."""
    x: float
    y: float
    r: float
    h: float


def clip_norm(v, max_norm):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, max_norm / (n + 1e-9))


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def make_city(seed=3, n=7, world=(100, 100, 50), keep_clear=((12, 12), (80, 75), (20, 85))):
    """Random 'city' of buildings, keeping start / goal zones free."""
    rng = np.random.default_rng(seed)
    blds = []
    while len(blds) < n:
        x, y = rng.uniform(20, world[0] - 15, 2)
        r, h = rng.uniform(4, 7), rng.uniform(20, 42)
        if any(np.hypot(x - cx, y - cy) < r + 14 for cx, cy in keep_clear):
            continue
        if any(np.hypot(x - b.x, y - b.y) < r + b.r + 8 for b in blds):
            continue
        blds.append(Building(x, y, r, h))
    return blds

