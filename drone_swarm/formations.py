"""Formation shapes: each returns n x 3 offsets around a centre."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
# ==============================================================
#  FORMATION SHAPES  (each returns n x 3 offsets around a centre)
# ==============================================================
def formation_circle(n, radius=14.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[radius * np.cos(a), radius * np.sin(a), np.zeros(n)]

def formation_v(n, spacing=4.0):
    off = np.zeros((n, 3))
    for i in range(1, n):
        k, side = (i + 1) // 2, (1 if i % 2 else -1)
        off[i] = [-k * spacing, side * k * spacing, 0]
    return off - off.mean(0)

def formation_grid(n, spacing=5.0):
    c = int(np.ceil(np.sqrt(n)))
    pts = np.array([[i % c, i // c, 0] for i in range(n)], float) * spacing
    return pts - pts.mean(0)

def formation_helix(n, radius=10.0, turns=2.0, height=18.0):
    t = np.linspace(0, 2 * np.pi * turns, n)
    off = np.c_[radius * np.cos(t), radius * np.sin(t), np.linspace(-height / 2, height / 2, n)]
    return off

def formation_sphere(n, radius=12.0):
    i = np.arange(n) + 0.5                      # Fibonacci sphere
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return radius * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]

FORMATIONS = {'circle': formation_circle, 'v': formation_v, 'grid': formation_grid,
              'helix': formation_helix, 'sphere': formation_sphere}

