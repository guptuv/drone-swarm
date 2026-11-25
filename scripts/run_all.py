"""Run every experiment and save plots (PNG), animations (GIF) and tables (CSV) to results/.

Usage:  python scripts/run_all.py            (from the repository root)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from drone_swarm import (SwarmConfig, DroneSwarm, SearchAndRescue, FORMATIONS, make_city,
                         formation_grid, animate_swarm, plot_metrics, animate_sar)

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)
plt.rcParams['figure.dpi'] = 85


def save_gif(anim, name, fps=15, dpi=60):
    anim.save(os.path.join(OUT, name), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close('all')
    print('saved', name)


def save_metrics(sw, title, name):
    fig, summary = plot_metrics(sw, title)
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight')
    summary.to_csv(os.path.join(OUT, name + '.csv'))
    plt.close('all')
    print(summary.to_string(), '\n')
    return summary


# ---------- formation shapes ----------
fig = plt.figure(figsize=(14, 3))
for k, (name, fn) in enumerate(FORMATIONS.items()):
    ax = fig.add_subplot(1, 5, k + 1, projection='3d')
    ax.scatter(*fn(25).T, s=18, c='tab:blue')
    ax.set_title(name); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'formations.png')); plt.close(fig)

# ---------- Demo 1: Boids flocking ----------
cfg = SwarmConfig(n_drones=30)
flock = DroneSwarm(cfg, start_center=(50, 50, 25), spread=15, seed=1)
flock.vel = flock.rng.normal(0, 4, (cfg.n_drones, 3))
flock.run(350, phase='boids flocking')
save_metrics(flock, 'Demo 1 - Boids flocking', 'demo1_flocking_metrics')
save_gif(animate_swarm(flock, every=5, title='Demo 1: Boids flocking'), 'demo1_flocking.gif')

# ---------- Demo 2: obstacle avoidance ----------
city = make_city(seed=3, n=7)
GOAL = np.array([80.0, 75.0, 20.0])
nav = DroneSwarm(SwarmConfig(n_drones=25), start_center=(12, 12, 10), seed=2)
nav.buildings, nav.goal = city, GOAL
nav.run(700, phase='navigate + avoid', until=lambda s: np.linalg.norm(s.pos[s.alive].mean(0) - GOAL) < 6)
print(f'Demo 2: goal reached in {nav.t:.1f}s')
save_metrics(nav, 'Demo 2 - Obstacle avoidance', 'demo2_obstacles_metrics')
save_gif(animate_swarm(nav, every=5, title='Demo 2: Obstacle avoidance', goals=[GOAL]), 'demo2_obstacles.gif')

# ---------- Demo 3: formation control + fault tolerance ----------
fm = DroneSwarm(SwarmConfig(n_drones=25), start_center=(20, 20, 5), spread=5, seed=4)
C = (45, 50, 25)
fm.set_formation('grid', center=C);   fm.run(150, phase='grid')
fm.set_formation('circle', center=C); fm.run(150, phase='circle')
fm.set_formation('sphere', center=C); fm.run(150, phase='sphere')
fm.fail_drones(5);                    fm.run(120, phase='sphere (5 failed)')
fm.set_formation('v', center=fm.pos[fm.alive].mean(0), goal=(85, 85, 30)); fm.run(250, phase='V moving')
save_metrics(fm, 'Demo 3 - Formation control', 'demo3_formation_metrics')
save_gif(animate_swarm(fm, every=5, title='Demo 3: Formation control'), 'demo3_formation.gif')

# ---------- Experiment: separation weight vs safety ----------
rows = []
for n in [15, 30, 50]:
    for w_sep in [0.0, 0.5, 1.0, 2.0, 4.0]:
        for seed in range(3):
            s = DroneSwarm(SwarmConfig(n_drones=n, w_sep=w_sep), start_center=(50, 50, 25), spread=12, seed=seed)
            s.run(250)
            d = s.df().iloc[50:]
            rows.append(dict(n=n, w_sep=w_sep, min_sep=d.min_separation.min(), near_misses=d.near_misses.sum(),
                             polarization=d.polarization.mean(), cohesion=d.cohesion.mean()))
exp = pd.DataFrame(rows).groupby(['n', 'w_sep']).mean().round(2)
exp.to_csv(os.path.join(OUT, 'experiment_separation.csv')); print(exp.to_string(), '\n')
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for n, g in exp.reset_index().groupby('n'):
    axs[0].plot(g.w_sep, g.min_sep, 'o-', label=f'N={n}')
    axs[1].plot(g.w_sep, g.near_misses, 'o-', label=f'N={n}')
    axs[2].plot(g.w_sep, g.polarization, 'o-', label=f'N={n}')
for ax, t in zip(axs, ['Min separation (m)', 'Near-misses (<1 m)', 'Polarization']):
    ax.set_title(t); ax.set_xlabel('separation weight'); ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'experiment_separation.png')); plt.close(fig)

# ---------- Demo 4: Search & Rescue ----------
sar = SearchAndRescue(n_drones=10, area=150, n_survivors=5, strategy='pso', seed=7)
print('Demo 4:', sar.run())
save_gif(animate_sar(sar), 'demo4_search_rescue.gif', fps=12)

# ---------- Benchmark ----------
SCENARIOS = {'A: 100x100 m, 12 drones': dict(area=100, n_drones=12, n_survivors=5),
             'B: 200x200 m, 10 drones': dict(area=200, n_drones=10, n_survivors=6)}
results, curves = [], {}
for sc, kw in SCENARIOS.items():
    for strat in ['pso', 'lawnmower', 'random']:
        curves[(sc, strat)] = []
        for seed in range(8):
            s = SearchAndRescue(strategy=strat, seed=seed, **kw)
            results.append(dict(scenario=sc, **s.run(max_steps=3000)))
            curves[(sc, strat)].append(s.cov_hist)
bench = pd.DataFrame(results)
summary = bench.groupby(['scenario', 'strategy']).agg(
    success_rate=('all_found', 'mean'), time_all_mean_s=('time_all_s', 'mean'),
    time_all_std_s=('time_all_s', 'std'), mean_find_time_s=('mean_find_time_s', 'mean'),
    coverage_pct=('coverage_pct', 'mean')).round(1)
summary.to_csv(os.path.join(OUT, 'benchmark_sar.csv')); print(summary.to_string(), '\n')
fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
summary['time_all_mean_s'].unstack().plot.bar(ax=axs[0], rot=0, yerr=summary['time_all_std_s'].unstack(), capsize=4)
axs[0].set_title('Time to find ALL survivors (lower = better)'); axs[0].set_ylabel('s'); axs[0].set_xlabel('')
for ax, sc in zip(axs[1:], SCENARIOS):
    for strat in ['pso', 'lawnmower', 'random']:
        cs = curves[(sc, strat)]; L = max(map(len, cs))
        M = np.array([np.pad(c, (0, L - len(c)), mode='edge') for c in cs])
        ax.plot(np.arange(L) * 0.5, M.mean(0), label=strat)
    ax.set_title(f'Camera coverage - {sc}', fontsize=10); ax.set_xlabel('t (s)'); ax.set_ylabel('% area seen')
    ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'benchmark_sar.png')); plt.close(fig)

# ---------- Full mission ----------
WP1, WP2 = np.array([80.0, 75.0, 22.0]), np.array([30.0, 60.0, 30.0])
m = DroneSwarm(SwarmConfig(n_drones=24), seed=11)
m.pos = np.c_[formation_grid(24, 4.0)[:, :2] + 14, np.zeros(24)]
m.vel[:] = 0; m.traj[0] = m.pos.copy()
m.buildings = city
m.set_formation('grid', center=(12, 12, 12), spacing=3.5); m.run(80, phase='1 takeoff (grid)')
m.mode, m.goal = 'flock', WP1
m.run(700, phase='2 flock to WP1', until=lambda s: np.linalg.norm(s.pos[s.alive].mean(0) - WP1) < 6)
m.goal = None
m.set_formation('circle', center=WP1, radius=13); m.run(150, phase='3 circle survey')
dead = m.fail_drones(4);                          m.run(100, phase='4 failure + re-plan')
m.set_formation('helix', center=WP1);             m.run(150, phase='5 helix')
m.set_formation('v', center=m.pos[m.alive].mean(0), goal=WP2, spacing=2.5)
m.run(500, phase='6 V to landing zone', until=lambda s: np.linalg.norm(s.form_center - WP2) < 0.5)
m.run(60, phase='6 V to landing zone')
print(f'Mission: {m.t:.1f}s | lost {len(dead)} | building collisions {m.building_hits}')
save_metrics(m, 'Full mission telemetry', 'mission_metrics')
save_gif(animate_swarm(m, every=6, trail=8, title='Full drone-swarm mission', goals=[WP1, WP2]), 'mission.gif')
print('All results saved to', os.path.abspath(OUT))
