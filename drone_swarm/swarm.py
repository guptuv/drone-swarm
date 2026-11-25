"""DroneSwarm: Boids flocking, obstacle avoidance, formation control, fault tolerance.

Optional extensions (all off by default, so the base behaviour is unchanged):
    sw.dynamics           = QuadrotorDynamics(n)      # thrust / attitude model instead of point mass
    sw.wind               = Wind(...)                 # mean wind + gusts
    sw.comms              = CommModel(...)            # range, latency, packet loss for neighbour info
    sw.gps_noise          = 0.5                       # std of position measurement noise (m)
    sw.moving_obstacles   = [MovingObstacle(...)]     # dynamic obstacles, predictive avoidance
    sw.team               = np.array([...])           # several swarms sharing the airspace
    sw.goal               = (n, 3) array              # per-drone goals (e.g. team goals)
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .core import SwarmConfig, Building, clip_norm, unit
from .formations import FORMATIONS


class DroneSwarm:
    def __init__(self, cfg: SwarmConfig, start_center=(12, 12, 8), spread=6.0, seed=0):
        self.cfg, self.rng = cfg, np.random.default_rng(seed)
        n = cfg.n_drones
        self.n = n
        self.W = np.array(cfg.world, float)
        self.pos = np.array(start_center, float) + self.rng.normal(0, spread, (n, 3))
        self.pos[:, 2] = np.clip(self.pos[:, 2], cfg.min_altitude, None)
        self.vel = self.rng.normal(0, 1.0, (n, 3))
        self.battery = np.full(n, 100.0)
        self.alive = np.ones(n, bool)
        self.buildings, self.goal = [], None
        self.mode, self.phase = 'flock', 'start'
        self.form_fn = self.form_off = self.form_center = self.form_goal = None
        # ---- extensions (off by default) ----
        self.dynamics = None
        self.wind = None
        self.comms = None
        self.gps_noise = 0.0
        self.moving_obstacles = []
        self.team = np.zeros(n, int)
        self.predictive_avoidance = False                  # reciprocal CPA avoidance between all drones
        # ---- bookkeeping ----
        self.t = 0.0
        self.traj, self.alive_traj, self.target_traj, self.log = [], [], [], []
        self.mob_traj = []
        self.building_hits = 0
        self.obstacle_hits = 0
        self._record(self._pairwise()[1])

    # ---------- neighbourhood ----------
    def _pairwise(self, pos=None):
        pos = self.pos if pos is None else pos
        diff = pos[:, None, :] - pos[None, :, :]          # i - j
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        dist[~self.alive, :] = np.inf
        dist[:, ~self.alive] = np.inf
        return diff, dist

    def _perceived(self):
        """What each drone believes about its neighbours.

        Returns (diff, dist, nvel) where nvel[i, j] is drone j's velocity as known by drone i.
        Separation uses on-board sensing (true relative position + GPS noise);
        alignment / cohesion use communicated states (possibly delayed / dropped).
        """
        est = self.pos if self.gps_noise == 0 else self.pos + self.rng.normal(0, self.gps_noise, self.pos.shape)
        sdiff, sdist = self._pairwise(est)
        if self.comms is None:
            return (sdiff, sdist), (sdiff, sdist, np.broadcast_to(self.vel[None], (self.n, self.n, 3)))
        bpos, bvel, known = self.comms.update(self, est)
        cdiff = est[:, None, :] - bpos
        cdist_ = np.linalg.norm(cdiff, axis=-1)
        cdist_[~known] = np.inf
        np.fill_diagonal(cdist_, np.inf)
        cdist_[~self.alive, :] = np.inf
        cdist_[:, ~self.alive] = np.inf
        return (sdiff, sdist), (cdiff, cdist_, bvel)

    # ---------- Boids rules ----------
    def separation(self, diff, dist, radius=None):
        r = radius or self.cfg.separation_radius
        w = np.where(dist < r, 1.0 / (dist ** 2 + 1e-6), 0.0)
        push = (diff * w[..., None]).sum(1)
        has = (dist < r).any(1)[:, None]
        steer = unit(push) * self.cfg.max_speed - self.vel
        return np.where(has, clip_norm(steer, self.cfg.max_accel), 0.0)

    def _same_team(self, dist):
        return (dist < self.cfg.neighbor_radius) & (self.team[:, None] == self.team[None, :])

    def alignment(self, dist, nvel=None):
        nvel = np.broadcast_to(self.vel[None], (self.n, self.n, 3)) if nvel is None else nvel
        nb = self._same_team(dist)
        cnt = nb.sum(1, keepdims=True)
        avg = (nb[..., None] * nvel).sum(1) / np.maximum(cnt, 1)
        steer = unit(avg) * self.cfg.max_speed - self.vel
        return np.where(cnt > 0, clip_norm(steer, self.cfg.max_accel), 0.0)

    def cohesion(self, dist, diff=None):
        nb = self._same_team(dist)
        cnt = nb.sum(1, keepdims=True)
        if diff is None:
            center = (nb[..., None] * self.pos[None]).sum(1) / np.maximum(cnt, 1)
            to_c = center - self.pos
        else:                                           # relative form works with perceived states
            to_c = -(nb[..., None] * diff).sum(1) / np.maximum(cnt, 1)
        steer = unit(to_c) * self.cfg.max_speed - self.vel
        return np.where(cnt > 0, clip_norm(steer, self.cfg.max_accel), 0.0)

    def seek(self, target, slow_radius=20.0):
        d = target - self.pos
        dn = np.linalg.norm(d, axis=-1, keepdims=True)
        desired = unit(d) * self.cfg.max_speed * np.minimum(1.0, dn / slow_radius)
        return clip_norm(desired - self.vel, self.cfg.max_accel)

    # ---------- obstacle avoidance (potential field + tangential slide) ----------
    def avoid_buildings(self):
        acc = np.zeros_like(self.pos)
        m = self.cfg.obs_margin
        for b in self.buildings:
            h = self.pos[:, :2] - [b.x, b.y]
            d = np.linalg.norm(h, axis=1)
            surf = d - b.r
            below = self.pos[:, 2] < b.h + 3
            near = (surf < m) & below
            if not near.any():
                continue
            strength = np.where(near, ((m - np.maximum(surf, 0.2)) / m) ** 2, 0.0)
            radial = unit(h)
            tang = np.c_[-radial[:, 1], radial[:, 0]]
            sgn = np.sign((tang * self.vel[:, :2]).sum(1, keepdims=True) + 1e-6)
            acc[:, :2] += (radial * 3.0 + tang * sgn * 1.5) * strength[:, None] * self.cfg.max_accel
            self.building_hits += int(((surf < 0) & below & self.alive).sum())
        return acc

    def avoid_moving(self, horizon=3.0):
        """Predictive (velocity-obstacle style) avoidance of moving spheres."""
        acc = np.zeros_like(self.pos)
        for ob in self.moving_obstacles:
            rel_p = self.pos - ob.pos                       # obstacle -> drone
            rel_v = self.vel - ob.vel
            tca = np.clip(-(rel_p * rel_v).sum(1) / ((rel_v ** 2).sum(1) + 1e-9), 0, horizon)
            miss = rel_p + rel_v * tca[:, None]             # separation at closest approach
            md = np.linalg.norm(miss, axis=1)
            safe = ob.r + self.cfg.obs_margin
            danger = (md < safe) & self.alive
            urgency = np.where(danger, (safe - md) / safe * (1 - tca / horizon) + 0.3, 0.0)
            acc += unit(miss) * urgency[:, None] * self.cfg.max_accel * 3
            cur = np.linalg.norm(rel_p, axis=1)
            self.obstacle_hits += int(((cur < ob.r) & self.alive).sum())
        return acc

    def avoid_agents(self, horizon=2.5, safe=4.0, other_team_only=True):
        """Reciprocal predictive avoidance between drones (closest point of approach)
        with a right-hand rule so head-on encounters always pass on the same side."""
        rel_p = self.pos[:, None] - self.pos[None]               # j -> i
        rel_v = self.vel[:, None] - self.vel[None]
        tca = np.clip(-(rel_p * rel_v).sum(-1) / ((rel_v ** 2).sum(-1) + 1e-9), 0, horizon)
        miss = rel_p + rel_v * tca[..., None]
        md = np.linalg.norm(miss, axis=-1)
        mask = self.alive[:, None] & self.alive[None, :]
        np.fill_diagonal(mask, False)
        if other_team_only:
            mask &= self.team[:, None] != self.team[None, :]
        danger = mask & (md < safe) & ((rel_p * rel_v).sum(-1) < 0)   # approaching
        if not danger.any():
            return np.zeros_like(self.pos)
        urg = np.where(danger, (safe - md) / safe * (1 - tca / horizon) + 0.2, 0.0)
        right = np.cross(self.vel, [0, 0, 1.0])                  # each drone's right-hand side
        push = unit(miss) + 0.8 * unit(right)[:, None, :]
        return (push * urg[..., None]).sum(1) * self.cfg.max_accel * 1.5

    def boundaries(self):
        m, acc = self.cfg.bound_margin, np.zeros_like(self.pos)
        lo = np.array([0, 0, self.cfg.min_altitude])
        acc += np.clip((lo + m - self.pos) / m, 0, None)
        acc -= np.clip((self.pos - (self.W - m)) / m, 0, None)
        return acc * self.cfg.max_accel

    # ---------- formation control ----------
    def set_formation(self, shape, center=None, goal=None, **kw):
        """shape: name in FORMATIONS or callable(n)->offsets. Uses Hungarian assignment."""
        self.form_fn = (lambda n: FORMATIONS[shape](n, **kw)) if isinstance(shape, str) else shape
        self.form_center = np.array(center if center is not None else self.pos[self.alive].mean(0), float)
        self.form_goal = None if goal is None else np.array(goal, float)
        self._assign()
        self.mode = 'formation'

    def _assign(self):
        idx = np.where(self.alive)[0]
        offs = self.form_fn(len(idx))
        R = self._heading_rot()                     # assign slots in the rotated frame
        cost = cdist(self.pos[idx], self.form_center + offs @ R.T)
        r, c = linear_sum_assignment(cost)          # minimise total travel distance
        self.form_off = np.zeros_like(self.pos)
        self.form_off[idx[r]] = offs[c]

    def _heading_rot(self):
        if self.form_goal is None:
            return np.eye(3)
        d = self.form_goal[:2] - self.form_center[:2]
        if np.linalg.norm(d) < 1e-3:
            return getattr(self, '_last_R', np.eye(3))
        a = np.arctan2(d[1], d[0])
        self._last_R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        return self._last_R

    def targets(self):
        return self.form_center + self.form_off @ self._heading_rot().T

    def formation_control(self):
        c = self.cfg
        if self.form_goal is not None:                       # move the whole formation
            d = self.form_goal - self.form_center
            dn = np.linalg.norm(d)
            if dn > 1e-3:
                self.form_center = self.form_center + d / dn * min(c.formation_speed * c.dt, dn)
        e = self.targets() - self.pos
        desired = clip_norm(c.kp * e, c.max_speed)
        return clip_norm(c.kd * (desired - self.vel), c.max_accel)

    # ---------- fault tolerance ----------
    def fail_drones(self, k):
        alive_idx = np.where(self.alive)[0]
        dead = self.rng.choice(alive_idx, size=min(k, len(alive_idx) - 1), replace=False)
        self.alive[dead] = False
        if self.mode == 'formation':
            self._assign()                                  # re-plan with the survivors
        return dead

    # ---------- simulation step ----------
    def desired_acceleration(self):
        c = self.cfg
        (sdiff, sdist), (cdiff, cdist_, nvel) = self._perceived()
        acc = np.zeros_like(self.pos)
        if self.mode == 'flock':
            if self.comms is None:
                acc += c.w_sep * self.separation(sdiff, sdist) + c.w_ali * self.alignment(sdist) \
                     + c.w_coh * self.cohesion(sdist)
            else:
                acc += c.w_sep * self.separation(sdiff, sdist) + c.w_ali * self.alignment(cdist_, nvel) \
                     + c.w_coh * self.cohesion(cdist_, cdiff)
            if self.goal is not None:
                acc += c.w_goal * self.seek(self.goal)
        else:
            acc += self.formation_control() + 1.5 * self.separation(sdiff, sdist, radius=2.5)
        acc += c.w_obs * self.avoid_buildings() + c.w_bound * self.boundaries()
        if self.moving_obstacles:
            acc += self.avoid_moving()
        if self.predictive_avoidance or len(np.unique(self.team)) > 1:
            acc += self.avoid_agents(other_team_only=not self.predictive_avoidance)
        return clip_norm(acc, c.max_accel * 1.5)

    def step(self):
        c = self.cfg
        true_dist = self._pairwise()[1]
        a_cmd = self.desired_acceleration()
        wind = self.wind.step(c.dt, self.rng) if self.wind is not None else None

        if self.dynamics is not None:                        # thrust / attitude model
            a = self.dynamics.step(a_cmd, self.vel, c.dt, wind)
            self.vel = self.vel + a * c.dt
            self.vel = clip_norm(self.vel, c.max_speed * 1.3)
        else:                                               # point mass (original model)
            if wind is not None:
                a_cmd = a_cmd + 0.35 * (wind - self.vel)    # aerodynamic drag towards air velocity
            self.vel = clip_norm(self.vel + a_cmd * c.dt, c.max_speed)
            if self.mode == 'flock' and self.goal is None:      # birds never hover
                sp = np.linalg.norm(self.vel, axis=1, keepdims=True)
                self.vel = np.where(sp < c.min_speed, unit(self.vel) * c.min_speed, self.vel)
        # failed drones: controlled emergency landing
        self.vel[~self.alive] = [0, 0, -3.0]
        self.pos += self.vel * c.dt
        self.pos[:, 2] = np.maximum(self.pos[:, 2], 0)
        self.vel[~self.alive & (self.pos[:, 2] <= 0)] = 0

        for ob in self.moving_obstacles:
            ob.step(c.dt, self.W)

        if self.dynamics is not None:
            self.battery -= np.where(self.alive, self.dynamics.power_pct(c.dt), 0)
        else:
            speed = np.linalg.norm(self.vel, axis=1)
            self.battery -= np.where(self.alive, (0.08 + 0.002 * speed ** 2) * c.dt, 0)  # % per step
        self.t += c.dt
        self._record(true_dist)

    def run(self, steps, phase=None, until=None):
        if phase: self.phase = phase
        for _ in range(steps):
            self.step()
            if until is not None and until(self):
                break
        return self

    # ---------- telemetry ----------
    def _record(self, dist):
        a = self.alive
        P, V = self.pos[a], self.vel[a]
        sp = np.linalg.norm(V, axis=1)
        cen = P.mean(0)
        d = dist[np.ix_(a, a)]
        goal_dist = np.nan
        if self.goal is not None and self.mode == 'flock':
            g = np.asarray(self.goal)
            goal_dist = float(np.linalg.norm(cen - g)) if g.ndim == 1 else float(np.linalg.norm(self.pos[a] - g[a], axis=1).mean())
        row = dict(t=round(self.t, 2), phase=self.phase, alive=int(a.sum()),
                   polarization=float(np.linalg.norm(unit(V).mean(0))),
                   cohesion=float(np.linalg.norm(P - cen, axis=1).mean()),
                   min_separation=float(d.min()) if d.size > 1 else np.nan,
                   near_misses=int((d[np.triu_indices(len(d), 1)] < 1.0).sum()),
                   mean_speed=float(sp.mean()), battery=float(self.battery[a].mean()),
                   goal_dist=goal_dist,
                   formation_error=float(np.linalg.norm(self.targets()[a] - P, axis=1).mean()) if self.mode == 'formation' else np.nan)
        if self.moving_obstacles:
            row['min_obstacle_dist'] = float(min(
                (np.linalg.norm(self.pos[a] - ob.pos, axis=1) - ob.r).min() for ob in self.moving_obstacles))
            self.mob_traj.append(np.array([ob.pos.copy() for ob in self.moving_obstacles]))
        if len(np.unique(self.team)) > 1:
            cross = dist[np.ix_(a, a)][self.team[a][:, None] != self.team[a][None, :]]
            row['min_cross_team'] = float(cross.min()) if cross.size else np.nan
        if self.comms is not None:
            row['msg_age'] = float(self.comms.mean_age())
        self.log.append(row)
        self.traj.append(self.pos.copy())
        self.alive_traj.append(self.alive.copy())
        self.target_traj.append(self.targets().copy() if self.mode == 'formation' else None)

    def df(self):
        return pd.DataFrame(self.log)
