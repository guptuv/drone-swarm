"""Inter-drone communication model: limited range, latency and packet loss.

Every drone broadcasts (position, velocity) each step.  Drone i receives drone j's
message only if they are within `range_m` and the packet is not dropped (prob. `p_loss`).
Messages arrive `delay_steps` steps late.  Each drone keeps the last state it received
from every neighbour, so stale information is used when packets are lost.
"""
import numpy as np
from collections import deque


class CommModel:
    def __init__(self, n, range_m=30.0, p_loss=0.0, delay_steps=0, max_age=30):
        self.n, self.range, self.p_loss, self.delay = n, range_m, p_loss, int(delay_steps)
        self.max_age = max_age
        self.buf = deque(maxlen=self.delay + 1)
        self.bpos = np.zeros((n, n, 3))
        self.bvel = np.zeros((n, n, 3))
        self.stamp = np.full((n, n), -np.inf)          # time step of last received message
        self.k = 0
        self.sent = self.received = 0

    def update(self, swarm, est_pos):
        self.buf.append((est_pos.copy(), swarm.vel.copy(), self.k))
        if len(self.buf) > self.delay:
            p_old, v_old, k_old = self.buf[0]
            d = np.linalg.norm(swarm.pos[:, None] - swarm.pos[None], axis=-1)
            ok = (d < self.range) & swarm.alive[None, :] & swarm.alive[:, None]
            np.fill_diagonal(ok, False)
            self.sent += int(ok.sum())
            if self.p_loss > 0:
                ok &= swarm.rng.random((self.n, self.n)) >= self.p_loss
            self.received += int(ok.sum())
            ii, jj = np.nonzero(ok)
            self.bpos[ii, jj] = p_old[jj]
            self.bvel[ii, jj] = v_old[jj]
            self.stamp[ii, jj] = k_old
        self.k += 1
        known = np.isfinite(self.stamp) & ((self.k - self.stamp) < self.delay + self.max_age)   # forget stale info
        return self.bpos, self.bvel, known

    def mean_age(self):
        s = self.stamp[np.isfinite(self.stamp)]
        return float((self.k - s).mean()) if s.size else np.nan

    def delivery_ratio(self):
        return self.received / max(self.sent, 1)
