"""State estimation.

* EKF / Kalman filter fusing a 100 Hz IMU (accelerometer with bias + noise) and a 5 Hz GPS
  (noise + occasional outages) into position / velocity / accelerometer-bias estimates.
* Particle filter that localises a survivor's phone from noisy RSSI (log-distance path
  loss) measurements taken by a moving drone.
"""
import numpy as np


# ------------------------------------------------------------------ Kalman filter (GPS + IMU)
def simulate_flight(T=60.0, dt=0.01, seed=0, gps_dt=0.2, gps_std=2.0, acc_std=0.3, acc_bias=(0.4, -0.3, 0.2),
                    outage=(30.0, 38.0)):
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    t = np.arange(n) * dt
    pos = np.c_[30 * np.sin(0.15 * t), 20 * np.sin(0.3 * t), 10 + 3 * np.sin(0.1 * t)]
    vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
    acc_meas = acc + np.array(acc_bias) + rng.normal(0, acc_std, acc.shape)
    gps_mask = (np.arange(n) % int(gps_dt / dt) == 0) & ~((t >= outage[0]) & (t < outage[1]))
    gps = pos + rng.normal(0, gps_std, pos.shape)
    return dict(t=t, pos=pos, vel=vel, acc_meas=acc_meas, gps=gps, gps_mask=gps_mask, dt=dt,
                gps_std=gps_std, acc_std=acc_std)


class GpsImuKF:
    """State x = [p(3), v(3), b(3)]; IMU drives the prediction, GPS corrects."""

    def __init__(self, dt, acc_std=0.3, gps_std=2.0, bias_rw=0.002):
        self.dt = dt
        I = np.eye(3)
        self.F = np.block([[I, dt * I, -0.5 * dt ** 2 * I], [0 * I, I, -dt * I], [0 * I, 0 * I, I]])
        self.B = np.vstack([0.5 * dt ** 2 * I, dt * I, 0 * I])
        q = acc_std ** 2
        self.Q = np.block([[0.25 * dt ** 4 * q * I, 0.5 * dt ** 3 * q * I, 0 * I],
                           [0.5 * dt ** 3 * q * I, dt ** 2 * q * I, 0 * I],
                           [0 * I, 0 * I, bias_rw ** 2 * dt * I]])
        self.H = np.hstack([I, 0 * I, 0 * I])
        self.R = gps_std ** 2 * I
        self.x = np.zeros(9); self.P = np.eye(9) * 10

    def predict(self, acc):
        self.x = self.F @ self.x + self.B @ acc
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P


def run_kf(sim):
    kf = GpsImuKF(sim['dt'], sim['acc_std'], sim['gps_std'])
    kf.x[:3] = sim['gps'][0]
    est, sig = [], []
    dead = sim['pos'][0].copy(); dv = sim['vel'][0].copy(); dr = []
    last_gps = sim['gps'][0].copy(); hold = []
    for k in range(len(sim['t'])):
        kf.predict(sim['acc_meas'][k])
        dv = dv + sim['acc_meas'][k] * sim['dt']; dead = dead + dv * sim['dt']; dr.append(dead.copy())
        if sim['gps_mask'][k]:
            kf.update(sim['gps'][k]); last_gps = sim['gps'][k]
        hold.append(last_gps.copy())
        est.append(kf.x.copy()); sig.append(np.sqrt(np.diag(kf.P)[:3]))
    est, sig = np.array(est), np.array(sig)
    err = lambda e: float(np.sqrt(((e - sim['pos']) ** 2).sum(1).mean()))
    return dict(est=est, sigma=sig, dead_reckoning=np.array(dr), gps_hold=np.array(hold),
                rmse_kf=err(est[:, :3]), rmse_gps=err(np.array(hold)), rmse_dead=err(np.array(dr)),
                bias_est=est[-1, 6:])


# ------------------------------------------------------------------ particle filter (RSSI)
def rssi(d, rng=None, p0=-40.0, n=2.2, std=4.0):
    v = p0 - 10 * n * np.log10(np.maximum(d, 1.0))
    return v + (rng.normal(0, std, np.shape(d)) if rng is not None else 0)


class RssiParticleFilter:
    def __init__(self, n=3000, area=200.0, seed=0, std=4.0):
        self.rng = np.random.default_rng(seed)
        self.p = self.rng.uniform(0, area, (n, 2)); self.w = np.full(n, 1 / n)
        self.std, self.area = std, area

    def update(self, drone_xy, z):
        d = np.linalg.norm(self.p - drone_xy, axis=1)
        self.w *= np.exp(-0.5 * ((z - rssi(d)) / self.std) ** 2) + 1e-300
        self.w /= self.w.sum()
        if 1 / (self.w ** 2).sum() < len(self.p) / 2:            # systematic resampling
            c = np.cumsum(self.w); u = (self.rng.random() + np.arange(len(self.p))) / len(self.p)
            self.p = self.p[np.searchsorted(c, u)] + self.rng.normal(0, 1.0, self.p.shape)
            self.w = np.full(len(self.p), 1 / len(self.p))

    def estimate(self):
        m = (self.p * self.w[:, None]).sum(0)
        cov = np.cov(self.p.T, aweights=self.w)
        return m, cov


def localise_survivor(seed=0, steps=120, area=200.0, info_gain=True):
    """Drone flies an expanding spiral (or moves toward the estimate when info_gain=True)."""
    rng = np.random.default_rng(seed)
    target = rng.uniform(30, area - 30, 2)
    pf = RssiParticleFilter(area=area, seed=seed)
    drone = np.array([area / 2, area / 2]); path, errs, spread = [], [], []
    for k in range(steps):
        if info_gain and k > 10:
            m, cov = pf.estimate()
            ev, evec = np.linalg.eigh(cov)
            # fly around the estimate, across the most uncertain axis
            orbit = m + 25 * (np.cos(k * 0.3) * evec[:, 1] + np.sin(k * 0.3) * evec[:, 0])
            drone = drone + np.clip(orbit - drone, -8, 8)
        else:
            r = 10 + 1.2 * k
            drone = np.array([area / 2, area / 2]) + r * np.array([np.cos(k * 0.35), np.sin(k * 0.35)])
        drone = np.clip(drone, 0, area)
        pf.update(drone, rssi(np.linalg.norm(drone - target), rng))
        m, cov = pf.estimate()
        path.append(drone.copy()); errs.append(np.linalg.norm(m - target)); spread.append(np.sqrt(np.trace(cov)))
    return dict(target=target, path=np.array(path), err=np.array(errs), spread=np.array(spread), particles=pf.p.copy())
