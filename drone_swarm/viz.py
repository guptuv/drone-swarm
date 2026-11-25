"""3D swarm animation, telemetry dashboard and search-and-rescue animation.
Animation functions return a matplotlib FuncAnimation (use .to_jshtml() in notebooks or .save() for GIF)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .core import Building

# ==============================================================
#  VISUALISATION: 3D animation + telemetry dashboard
# ==============================================================
def draw_building(ax, b, alpha=0.30):
    th, z = np.linspace(0, 2 * np.pi, 20), np.array([0, b.h])
    TH, Z = np.meshgrid(th, z)
    ax.plot_surface(b.x + b.r * np.cos(TH), b.y + b.r * np.sin(TH), Z,
                    color='dimgray', alpha=alpha, linewidth=0)

def animate_swarm(sw, every=4, trail=10, title='Drone Swarm', rotate=True, goals=()):
    traj = np.array(sw.traj)[::every]
    alive = np.array(sw.alive_traj)[::every]
    tgts = sw.target_traj[::every]
    phases = [r['phase'] for r in sw.log][::every]
    n = traj.shape[1]
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, sw.W[0]); ax.set_ylim(0, sw.W[1]); ax.set_zlim(0, sw.W[2])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    for b in sw.buildings:
        draw_building(ax, b)
    for g in goals:
        ax.scatter(*g, marker='*', s=250, c='gold', edgecolors='k', zorder=5)
    sc = ax.scatter(*traj[0].T, s=22, c='tab:blue', depthshade=False)
    tg = ax.scatter([], [], [], s=12, marker='x', c='tab:orange', alpha=0.6)
    lines = [ax.plot([], [], [], lw=0.8, alpha=0.5, c='tab:blue')[0] for _ in range(n)]
    txt = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                    bbox=dict(fc='white', alpha=0.8))
    ax.set_title(title)

    def update(f):
        P = traj[f]
        sc._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        sc.set_color(np.where(alive[f], 'tab:blue', 'red'))
        T = tgts[f]
        tg._offsets3d = (T[:, 0], T[:, 1], T[:, 2]) if T is not None else ([], [], [])
        for i, l in enumerate(lines):
            seg = traj[max(0, f - trail):f + 1, i]
            l.set_data(seg[:, 0], seg[:, 1]); l.set_3d_properties(seg[:, 2])
        txt.set_text(f"t = {f * every * sw.cfg.dt:5.1f}s | {phases[f]} | alive {alive[f].sum()}/{n}")
        if rotate:
            ax.view_init(elev=28, azim=-60 + 0.4 * f)
        return [sc]

    anim = animation.FuncAnimation(fig, update, frames=len(traj), interval=60)
    plt.close(fig)
    return anim

def plot_metrics(sw, title='Swarm telemetry'):
    df = sw.df().iloc[1:]          # skip t=0 snapshot
    cols = [('polarization', 'Polarization (1 = all aligned)'), ('cohesion', 'Mean dist. to centroid (m)'),
            ('min_separation', 'Min drone-drone distance (m)'), ('mean_speed', 'Mean speed (m/s)'),
            ('formation_error', 'Formation error (m)'), ('battery', 'Mean battery (%)')]
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.5))
    for ax, (c, lab) in zip(axs.flat, cols):
        for ph, g in df.groupby('phase', sort=False):
            ax.plot(g.t, g[c], lw=1.5, label=ph)
        ax.set_title(lab, fontsize=10); ax.set_xlabel('t (s)'); ax.grid(alpha=0.3)
    axs[0, 2].axhline(1.0, c='r', ls='--', lw=1, label='collision limit')
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13); plt.tight_layout()
    summary = df.groupby('phase', sort=False).agg(
        duration_s=('t', lambda s: s.max() - s.min()), min_sep_m=('min_separation', 'min'),
        near_misses=('near_misses', 'sum'), avg_polarization=('polarization', 'mean'),
        final_form_err_m=('formation_error', 'last'), battery_end=('battery', 'last')).round(2)
    print(f"Building collisions: {sw.building_hits}")
    return fig, summary


def animate_sar(sar, every=3, trail=12):
    idx = list(range(0, len(sar.traj), every))
    if idx[-1] != len(sar.traj) - 1: idx.append(len(sar.traj) - 1)     # always show final frame
    traj, fh = np.array(sar.traj)[idx], [sar.found_hist[k] for k in idx]
    gx, gy = np.meshgrid(np.linspace(0, sar.A, 80), np.linspace(0, sar.A, 80))
    act = sar.surv
    field = np.exp(-cdist(np.c_[gx.ravel(), gy.ravel()], act) ** 2 / (2 * sar.sigma ** 2)).max(1).reshape(gx.shape)
    # rebuild coverage frame by frame
    covs, cov = [], np.zeros((sar.A, sar.A), bool)
    full = np.array(sar.traj)
    for k in range(len(full)):
        cov.ravel()[cdist(sar.cells, full[k]).min(1) < sar.sensor_r] = True
        if k in idx: covs.append(cov.copy())

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(covs[0], origin='lower', extent=(0, sar.A, 0, sar.A), cmap='Greens', vmin=0, vmax=2.5)
    ax.contour(gx, gy, field, levels=6, cmap='Reds', alpha=0.35)
    sv = ax.scatter(*act.T, marker='X', s=140, c='red', edgecolors='k', zorder=4)
    dr = ax.scatter(*traj[0].T, s=35, c='tab:blue', edgecolors='k', zorder=5)
    lines = [ax.plot([], [], lw=0.8, c='tab:blue', alpha=.5)[0] for _ in range(sar.n)]
    txt = ax.set_title('')
    def update(f):
        dr.set_offsets(traj[f]); im.set_data(covs[f])
        sv.set_color(np.where(fh[f], 'lime', 'red'))
        for i, l in enumerate(lines):
            seg = traj[max(0, f - trail):f + 1, i]; l.set_data(seg[:, 0], seg[:, 1])
        txt.set_text(f"{sar.strategy.upper()} | t={idx[f] * sar.dt:.0f}s | found {fh[f].sum()}/{len(fh[f])} | coverage {sar.cov_hist[idx[f]]:.0f}%")
        return [dr]
    anim = animation.FuncAnimation(fig, update, frames=len(traj), interval=70)
    plt.close(fig)
    return anim

