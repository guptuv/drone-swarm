"""Moving obstacles (other aircraft, birds, balloons) modelled as spheres bouncing in the world."""
import numpy as np


class MovingObstacle:
    def __init__(self, pos, vel, r=3.0):
        self.pos = np.array(pos, float)
        self.vel = np.array(vel, float)
        self.r = float(r)

    def step(self, dt, world):
        self.pos += self.vel * dt
        for k in range(3):
            lo, hi = (5.0 if k == 2 else self.r), world[k] - self.r
            if self.pos[k] < lo or self.pos[k] > hi:
                self.vel[k] *= -1
                self.pos[k] = np.clip(self.pos[k], lo, hi)


def random_obstacles(k=4, world=(100, 100, 50), speed=5.0, r=3.5, seed=0):
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(k):
        p = rng.uniform([15, 15, 10], [world[0] - 15, world[1] - 15, world[2] - 10])
        v = rng.normal(size=3); v[2] *= 0.3
        obs.append(MovingObstacle(p, v / np.linalg.norm(v) * speed, r))
    return obs
