"""Quadrotor translational dynamics with thrust / attitude limits, plus a wind model.

The swarm controller outputs a desired acceleration a_cmd.  A real quadrotor can only
produce acceleration along its body z-axis, so the cascaded controller

    a_cmd  ->  thrust vector f = m (a_cmd + g e_z)  ->  tilt-limited attitude  ->  motors

is modelled as:
  * thrust magnitude limited to [0, T_max]   (thrust-to-weight ratio)
  * tilt angle limited to tilt_max
  * attitude (body z-axis) follows the commanded one with a first-order lag tau
  * aerodynamic drag  -k_drag (v - v_wind)
  * electrical power ~ T^1.5 (momentum theory) used for battery drain
"""
import numpy as np
from .core import unit

G = 9.81


class QuadrotorDynamics:
    def __init__(self, n, mass=1.2, thrust_to_weight=2.2, tilt_max_deg=35.0, tau_att=0.15,
                 k_drag=0.25, battery_wh=60.0, hover_power_w=180.0):
        self.n, self.m = n, mass
        self.T_max = np.full(n, thrust_to_weight * mass * G)      # per drone (fault injection)
        self.tilt_max = np.deg2rad(tilt_max_deg)
        self.tau, self.k_drag = tau_att, k_drag
        self.zb = np.tile([0.0, 0.0, 1.0], (n, 1))       # body z-axis (thrust direction)
        self.T = np.full(n, mass * G)                      # current thrust (N)
        self.battery_wh, self.hover_power = battery_wh, hover_power_w
        self.hover_T = mass * G

    def _limit_tilt(self, z):
        z = unit(z)
        cos_t = np.clip(z[:, 2], -1, 1)
        too = np.arccos(cos_t) > self.tilt_max
        if too.any():
            h = unit(z[too, :2] + 1e-12)
            z[too, :2] = h * np.sin(self.tilt_max)
            z[too, 2] = np.cos(self.tilt_max)
        return z

    def step(self, a_cmd, vel, dt, wind=None):
        v_air = vel - (wind if wind is not None else 0.0)
        drag = -self.k_drag * v_air
        f_des = self.m * (a_cmd - drag / self.m + np.array([0, 0, G]))
        z_des = self._limit_tilt(f_des.copy())
        # first-order attitude response
        alpha = 1 - np.exp(-dt / self.tau)
        self.zb = unit(self.zb + alpha * (z_des - self.zb))
        # thrust: project desired force on the actual body axis, saturate
        self.T = np.clip((f_des * self.zb).sum(1), 0.0, self.T_max)
        return self.T[:, None] * self.zb / self.m - np.array([0, 0, G]) + drag / self.m

    def tilt_deg(self):
        return np.degrees(np.arccos(np.clip(self.zb[:, 2], -1, 1)))

    def power_pct(self, dt):
        """Battery drop in % for this step (power ~ T^1.5)."""
        p = self.hover_power * (self.T / self.hover_T) ** 1.5
        return p * dt / 3600.0 / self.battery_wh * 100.0


class Wind:
    """Mean wind + Ornstein-Uhlenbeck gusts (same for the whole airspace)."""

    def __init__(self, mean=(3.0, 0.0, 0.0), gust_std=1.5, corr_time=2.0):
        self.mean = np.array(mean, float)
        self.gust_std, self.tc = gust_std, corr_time
        self.g = np.zeros(3)
        self.hist = []

    def step(self, dt, rng):
        a = np.exp(-dt / self.tc)
        self.g = a * self.g + np.sqrt(1 - a ** 2) * self.gust_std * rng.normal(size=3)
        self.g[2] *= 0.3
        w = self.mean + self.g
        self.hist.append(w.copy())
        return w
