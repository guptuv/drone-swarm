"""Fault detection from telemetry with an autoencoder (unsupervised anomaly detection).

A swarm of quadrotors (thrust/attitude model + wind) flies a moving circle formation.
Each drone streams telemetry; one drone may get a fault at a random time:

  * motor   - one motor degrades: maximum thrust drops to ~45 % (cannot follow commands)
  * gps     - GPS position estimate starts drifting (bias grows 0.4 m/s)
  * imu     - accelerometer gets a constant bias (2 m/s^2)
  * none    - healthy flight

Features (per step): commanded accel (3), measured accel (3), tracking error (3), tilt,
thrust ratio, GPS-vs-integrated-velocity residual (3)  -> windows of 20 steps.
A dense autoencoder is trained on healthy windows only; reconstruction error above the
99.5th percentile of healthy data raises an alarm.  Baseline: per-feature z-score rule.
"""
import numpy as np
import torch
import torch.nn as nn
from ..core import SwarmConfig
from ..swarm import DroneSwarm
from ..dynamics import QuadrotorDynamics, Wind

WIN = 20
FAULTS = ['none', 'motor', 'gps', 'imu']


def fly(fault='none', n=8, steps=400, seed=0, fault_drone=0, t_fault=None):
    rng = np.random.default_rng(seed)
    sw = DroneSwarm(SwarmConfig(n_drones=n, world=(120, 120, 60)), start_center=(60, 60, 20), spread=3, seed=seed)
    sw.dynamics = QuadrotorDynamics(n)
    sw.wind = Wind(mean=(rng.uniform(-3, 3), rng.uniform(-3, 3), 0), gust_std=1.5)
    sw.set_formation('circle', center=(60, 60, 25), radius=10)
    t_fault = t_fault if t_fault is not None else int(rng.integers(120, 300))
    feats, labels = [], []
    prev_v = sw.vel.copy(); gps_bias = np.zeros(3); v_int = sw.vel.copy()
    ang = 0.0
    for k in range(steps):
        ang += 0.01                                              # formation centre moves on a circle
        sw.form_center = np.array([60 + 15 * np.cos(ang), 60 + 15 * np.sin(ang), 25 + 3 * np.sin(2 * ang)])
        active = k >= t_fault and fault != 'none'
        if active and fault == 'motor':
            sw.dynamics.T_max[fault_drone] = 0.45 * 2.2 * sw.dynamics.m * 9.81
        a_cmd = sw.desired_acceleration()
        sw.step()
        a_true = (sw.vel - prev_v) / sw.cfg.dt; prev_v = sw.vel.copy()
        a_meas = a_true + rng.normal(0, 0.3, a_true.shape)
        if active and fault == 'imu':
            a_meas[fault_drone] += np.array([2.0, -1.0, 1.0])
        v_int = v_int + a_meas * sw.cfg.dt
        gps = sw.pos + rng.normal(0, 0.3, sw.pos.shape)
        if active and fault == 'gps':
            gps_bias += np.array([0.4, 0.25, 0.0]) * sw.cfg.dt
            gps[fault_drone] += gps_bias
        gps_v = (gps - getattr(sw, '_prev_gps', gps)) / sw.cfg.dt; sw._prev_gps = gps
        v_int = 0.97 * v_int + 0.03 * gps_v                      # complementary filter
        err = sw.targets() - gps
        f = np.c_[a_cmd, a_meas, err, sw.dynamics.tilt_deg()[:, None] / 30,
                  sw.dynamics.T[:, None] / (sw.dynamics.m * 9.81), (gps_v - v_int)]
        feats.append(f)
        lab = np.zeros(n, int)
        if active: lab[fault_drone] = 1
        labels.append(lab)
    return np.array(feats), np.array(labels), t_fault        # [T, n, 14], [T, n]


def windows(feats, labels, start=60):
    X, y = [], []
    T, n, _ = feats.shape
    for t in range(start, T - WIN, 2):
        X.append(feats[t:t + WIN].transpose(1, 0, 2).reshape(n, -1))
        y.append(labels[t + WIN - 1])
    return np.concatenate(X).astype(np.float32), np.concatenate(y)


class AutoEncoder(nn.Module):
    def __init__(self, d, z=12):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 48), nn.ReLU(), nn.Linear(48, z))
        self.d = nn.Sequential(nn.Linear(z, 48), nn.ReLU(), nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, d))

    def forward(self, x):
        return self.d(self.e(x))


class FaultDetector:
    def __init__(self):
        self.mu = self.sd = self.model = None
        self.thr = None

    def fit(self, X, epochs=40, seed=0, verbose=True):
        torch.manual_seed(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = torch.tensor((X - self.mu) / self.sd)
        self.model = AutoEncoder(Z.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), 1e-3)
        for ep in range(epochs):
            perm = torch.randperm(len(Z)); tot = 0
            for s in range(0, len(Z), 256):
                b = Z[perm[s:s + 256]]
                loss = ((self.model(b) - b) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
            if verbose and (ep + 1) % 10 == 0:
                print(f"AE epoch {ep + 1} | recon loss {tot / len(Z):.4f}")
        self.thr = np.percentile(self.score(X), 99.5)
        return self

    @torch.no_grad()
    def score(self, X):
        Z = torch.tensor((X - self.mu) / self.sd)
        return ((self.model(Z) - Z) ** 2).mean(1).numpy()


class ZScoreRule:
    """Baseline: alarm if any feature of the latest step is > k sigma from healthy mean."""

    def fit(self, X, k=None):
        last = X[:, -14:]
        self.mu, self.sd = last.mean(0), last.std(0) + 1e-6
        self.thr = np.percentile(np.abs((last - self.mu) / self.sd).max(1), 99.5)
        return self

    def score(self, X):
        return np.abs((X[:, -14:] - self.mu) / self.sd).max(1)


def detection_report(detector, runs=6, seed=1000, n=8, steps=400):
    rows = []
    for fault in FAULTS:
        tp = fn = fp = tn = 0; delays = []
        for r in range(runs):
            feats, labels, tf = fly(fault, n=n, steps=steps, seed=seed + r * 7 + FAULTS.index(fault))
            T = feats.shape[0]
            for t in range(60, T - WIN, 2):
                X = feats[t:t + WIN].transpose(1, 0, 2).reshape(n, -1).astype(np.float32)
                alarm = detector.score(X) > detector.thr
                lab = labels[t + WIN - 1].astype(bool)
                tp += (alarm & lab).sum(); fn += (~alarm & lab).sum(); fp += (alarm & ~lab).sum(); tn += (~alarm & ~lab).sum()
            if fault != 'none':                                    # detection delay for the faulty drone
                for t in range(max(60, tf - WIN), T - WIN):
                    X = feats[t:t + WIN].transpose(1, 0, 2).reshape(n, -1).astype(np.float32)
                    if detector.score(X[:1])[0] > detector.thr and t + WIN - 1 >= tf:
                        delays.append((t + WIN - 1 - tf) * 0.1); break
                else:
                    delays.append(np.nan)
        rows.append(dict(fault=fault, recall=tp / max(tp + fn, 1) if fault != 'none' else np.nan,
                         false_alarm_rate=fp / max(fp + tn, 1),
                         detected_runs=np.mean(~np.isnan(delays)) if delays else np.nan,
                         median_delay_s=float(np.nanmedian(delays)) if delays and not np.all(np.isnan(delays)) else np.nan))
    return rows


def healthy_training_set(runs=12, seed=0, n=8, steps=400):
    X = [windows(*fly('none', n=n, steps=steps, seed=seed + r)[:2])[0] for r in range(runs)]
    return np.concatenate(X)
