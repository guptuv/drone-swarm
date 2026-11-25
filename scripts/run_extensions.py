"""Experiments for the v2 extensions.  Saves PNG / GIF / CSV into results/extensions/.

Usage:  python scripts/run_extensions.py   (from the repository root, ~3-4 minutes)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from scipy.sparse.csgraph import connected_components

from drone_swarm import (SwarmConfig, DroneSwarm, make_city, QuadrotorDynamics, Wind, CommModel,
                         random_obstacles, animate_swarm, formation_circle)
from drone_swarm.coverage import CoverageSwarm
from drone_swarm.consensus import run_consensus, leader_follower
from drone_swarm.allocation import auction_allocation, greedy_nearest, random_allocation, evaluate
from drone_swarm.learning import evolve, evaluate as es_evaluate, PARAMS

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'extensions')
os.makedirs(OUT, exist_ok=True)
plt.rcParams['figure.dpi'] = 85
GOAL = np.array([80.0, 75.0, 20.0])
CITY = make_city(seed=3, n=7)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight'); plt.close(fig); print('saved', name)


def gif(anim, name, fps=15, dpi=60):
    anim.save(os.path.join(OUT, name), writer=PillowWriter(fps=fps), dpi=dpi); plt.close('all'); print('saved', name)


# =================================================================
# 1. Quadrotor dynamics + wind + GPS noise
# =================================================================
def nav_run(seed, dyn=False, wind=False, gps=0.0, pa=False, **cfg):
    n = 25
    sw = DroneSwarm(SwarmConfig(n_drones=n, **cfg), start_center=(12, 12, 10), seed=seed)
    sw.buildings, sw.goal = CITY, GOAL
    if dyn: sw.dynamics = QuadrotorDynamics(n)
    if wind: sw.wind = Wind(mean=(4, 2, 0), gust_std=2.0)
    sw.gps_noise, sw.predictive_avoidance = gps, pa
    sw.run(900, phase='navigate through city', until=lambda s: np.linalg.norm(s.pos[s.alive].mean(0) - GOAL) < 6)
    return sw

cases = [('point mass (v1)', {}), ('quadrotor', dict(dyn=True)), ('quadrotor + wind', dict(dyn=True, wind=True)),
         ('quadrotor + wind + GPS noise 0.5 m', dict(dyn=True, wind=True, gps=0.5)),
         ('quadrotor + slower (6.5 m/s) + 5 m separation', dict(dyn=True, max_speed=6.5, separation_radius=5.0)),
         ('quadrotor + predictive avoidance', dict(dyn=True, pa=True))]
rows = []
for name, kw in cases:
    for seed in range(4):
        sw = nav_run(seed, **kw)
        d = sw.df().iloc[20:]
        rows.append(dict(case=name, time_s=sw.t, min_sep_m=d.min_separation.min(), near_misses=d.near_misses.sum(),
                         building_hits=sw.building_hits, battery_used_pct=100 - d.battery.iloc[-1]))
quad = pd.DataFrame(rows).groupby('case', sort=False).agg(
    time_s=('time_s', 'mean'), min_sep_m=('min_sep_m', 'min'), near_misses=('near_misses', 'sum'),
    building_hits=('building_hits', 'sum'), battery_used_pct=('battery_used_pct', 'mean')).round(2)
quad.to_csv(os.path.join(OUT, 'quadrotor_comparison.csv')); print(quad.to_string(), '\n')

sw = nav_run(0, dyn=True, wind=True)
tilt, thrust = [], []
sw2 = DroneSwarm(SwarmConfig(n_drones=25), start_center=(12, 12, 10), seed=0)
sw2.buildings, sw2.goal = CITY, GOAL
sw2.dynamics, sw2.wind = QuadrotorDynamics(25), Wind(mean=(4, 2, 0), gust_std=2.0)
for _ in range(220):
    sw2.step(); tilt.append(sw2.dynamics.tilt_deg()); thrust.append(sw2.dynamics.T / (sw2.dynamics.m * 9.81))
tilt, thrust, tt = np.array(tilt), np.array(thrust), np.arange(220) * 0.1
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(tt, tilt.mean(1), label='mean'); ax[0].plot(tt, tilt.max(1), label='max'); ax[0].axhline(35, c='r', ls='--', label='tilt limit')
ax[0].set_title('Tilt angle (deg)'); ax[0].legend()
ax[1].plot(tt, thrust.mean(1), label='mean'); ax[1].plot(tt, thrust.max(1), label='max'); ax[1].axhline(2.2, c='r', ls='--', label='T/W limit')
ax[1].set_title('Thrust / weight'); ax[1].legend()
W_ = np.array(sw2.wind.hist); ax[2].plot(tt, W_[:, 0], label='wind x'); ax[2].plot(tt, W_[:, 1], label='wind y'); ax[2].set_title('Wind (m/s)'); ax[2].legend()
for a in ax: a.set_xlabel('t (s)'); a.grid(alpha=.3)
save(fig, 'quadrotor_states.png')
gif(animate_swarm(sw, every=5, title='Quadrotor dynamics + wind gusts', goals=[GOAL]), 'quadrotor_wind.gif')

# =================================================================
# 2. Communication: packet loss x latency
# =================================================================
def comm_flock(pl, dl, seed):
    n = 30
    sw = DroneSwarm(SwarmConfig(n_drones=n), start_center=(50, 50, 25), spread=15, seed=seed)
    sw.vel = sw.rng.normal(0, 4, (n, 3))
    sw.comms = CommModel(n, range_m=20, p_loss=pl, delay_steps=dl)
    groups = []
    for k in range(400):
        sw.step()
        if k % 20 == 0 and k >= 100:
            groups.append(connected_components(sw._pairwise()[1] < 15, directed=False)[0])
    d = sw.df().iloc[100:]
    return dict(p_loss=pl, delay_s=dl * 0.1, polarization=d.polarization.mean(), cohesion_m=d.cohesion.mean(),
                groups=np.mean(groups), near_misses=d.near_misses.sum())

rows = [comm_flock(pl, dl, s) for pl in [0, 0.3, 0.6, 0.9] for dl in [0, 5, 10, 20] for s in range(3)]
comm = pd.DataFrame(rows).groupby(['p_loss', 'delay_s']).mean().round(2)
comm.to_csv(os.path.join(OUT, 'comms_experiment.csv')); print(comm.to_string(), '\n')
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, col, ttl in zip(ax, ['polarization', 'cohesion_m', 'groups'], ['Polarization (higher = better)', 'Spread around centroid (m)', 'Number of separate groups']):
    M = comm[col].unstack()
    im = a.imshow(M.values, cmap='viridis' if col == 'polarization' else 'magma_r', aspect='auto')
    a.set_xticks(range(M.shape[1])); a.set_xticklabels(M.columns); a.set_yticks(range(M.shape[0])); a.set_yticklabels(M.index)
    a.set_xlabel('latency (s)'); a.set_ylabel('packet loss'); a.set_title(ttl)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f'{M.values[i, j]:.2f}', ha='center', va='center', color='w', fontsize=8)
save(fig, 'comms_heatmap.png')

# =================================================================
# 3. Moving obstacles + two swarms crossing
# =================================================================
rows = []
for avoid in [True, False]:
    for seed in range(5):
        sw = DroneSwarm(SwarmConfig(n_drones=25), start_center=(12, 12, 20), seed=seed)
        sw.goal = np.array([85.0, 85, 25]); sw.moving_obstacles = random_obstacles(5, speed=6, seed=seed + 10)
        if not avoid:
            sw.avoid_moving = lambda horizon=3.0, sw=sw: (setattr(sw, 'obstacle_hits', sw.obstacle_hits + sum(
                int((np.linalg.norm(sw.pos - o.pos, axis=1) < o.r).sum()) for o in sw.moving_obstacles)) or np.zeros_like(sw.pos))
        sw.run(600, phase='fly to goal', until=lambda s: np.linalg.norm(s.pos.mean(0) - s.goal) < 6)
        rows.append(dict(avoidance='predictive' if avoid else 'none', time_s=sw.t,
                         min_clearance_m=sw.df().iloc[1:].min_obstacle_dist.min(), obstacle_hits=sw.obstacle_hits))
        if avoid and seed == 3:
            demo_mob = sw
mob = pd.DataFrame(rows).groupby('avoidance').agg(time_s=('time_s', 'mean'), min_clearance_m=('min_clearance_m', 'min'),
                                                  obstacle_hits=('obstacle_hits', 'sum')).round(2)
mob.to_csv(os.path.join(OUT, 'moving_obstacles.csv')); print(mob.to_string(), '\n')


def animate_with_mobs(sw, every=4, title=''):
    anim = animate_swarm(sw, every=every, title=title, goals=[sw.goal])
    fig = anim._fig; ax = fig.axes[0]
    mt = np.array(sw.mob_traj)[::every]
    balls = ax.scatter(*mt[0].T, s=[ob.r ** 2 * 18 for ob in sw.moving_obstacles], c='tab:red', alpha=0.45)
    upd = anim._func
    def update(f):
        upd(f); M = mt[min(f, len(mt) - 1)]; balls._offsets3d = (M[:, 0], M[:, 1], M[:, 2]); return []
    anim._func = update
    return anim

gif(animate_with_mobs(demo_mob, title='Moving obstacles (red) - predictive avoidance'), 'moving_obstacles.gif')

n = 30
cross = DroneSwarm(SwarmConfig(n_drones=n), seed=1)
rng = np.random.default_rng(0)
cross.team = np.r_[np.zeros(15, int), np.ones(15, int)]
cross.pos[:15] = np.c_[rng.uniform(8, 20, 15), rng.uniform(35, 65, 15), rng.uniform(20, 30, 15)]
cross.pos[15:] = np.c_[rng.uniform(80, 92, 15), rng.uniform(35, 65, 15), rng.uniform(20, 30, 15)]
cross.vel[:] = 0; cross.traj[0] = cross.pos.copy()
G = np.zeros((n, 3)); G[:15] = [88, 50, 25]; G[15:] = [12, 50, 25]; cross.goal = G
cross.run(220, phase='two swarms swap sides')
d = cross.df().iloc[1:]
print(f"crossing: min cross-team distance {d.min_cross_team.min():.2f} m, near-misses {d.near_misses.sum()}\n")
pd.DataFrame([dict(min_cross_team_m=round(d.min_cross_team.min(), 2), min_sep_m=round(d.min_separation.min(), 2),
                   near_misses=int(d.near_misses.sum()))]).to_csv(os.path.join(OUT, 'two_swarms.csv'), index=False)
anim = animate_swarm(cross, every=4, title='Two swarms crossing (right-hand rule)', rotate=False)
sc = anim._fig.axes[0].collections
upd = anim._func
cols = np.where(cross.team == 0, 'tab:blue', 'tab:orange')
def upd2(f, sc_=None):
    upd(f); anim._fig.axes[0].collections[-2].set_color(cols); return []
anim._func = upd2
gif(anim, 'two_swarms.gif')

# =================================================================
# 4. Coverage control (Lloyd) + battery-aware persistent coverage
# =================================================================
cov = CoverageSwarm(n=10, seed=0).run(120)
bat = CoverageSwarm(n=10, seed=0, battery=True).run(600)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
O = cov.owner_hist[-1].reshape(cov.gx.shape)
ax[0].imshow(O, origin='lower', extent=(0, 100, 0, 100), cmap='tab10', alpha=0.35)
ax[0].contour(cov.gx, cov.gy, cov.phi, levels=6, colors='k', alpha=0.3)
T = np.array(cov.traj)
for i in range(cov.n): ax[0].plot(T[:, i, 0], T[:, i, 1], lw=0.8, c='k', alpha=0.5)
ax[0].scatter(*T[-1].T, c='tab:blue', edgecolors='k', zorder=3); ax[0].set_title('Lloyd coverage: final Voronoi cells')
ax[1].semilogy(np.arange(len(cov.cost)) * cov.dt, cov.cost); ax[1].set_title('Locational cost H (log)'); ax[1].set_xlabel('t (s)'); ax[1].grid(alpha=.3)
tb = np.arange(len(bat.cost)) * bat.dt
ax[2].plot(tb, bat.active_hist, label='drones covering'); ax2 = ax[2].twinx()
ax2.plot(tb, np.array(bat.batt_hist).min(1), c='tab:red', alpha=0.7, label='lowest battery %')
ax[2].set_title('Battery-aware persistent coverage'); ax[2].set_xlabel('t (s)'); ax[2].set_ylabel('active drones'); ax2.set_ylabel('min battery %')
ax[2].legend(loc='lower left'); ax2.legend(loc='lower right')
save(fig, 'coverage.png')
pd.DataFrame([dict(lloyd_cost_start=round(cov.cost[0], 1), lloyd_cost_end=round(cov.cost[-1], 1),
                   battery_mean_active=round(np.mean(bat.active_hist[50:]), 2), battery_min_active=min(bat.active_hist[50:]),
                   battery_lowest_pct=round(float(np.min(bat.batt_hist)), 1))]).to_csv(os.path.join(OUT, 'coverage.csv'), index=False)

fig, axc = plt.subplots(figsize=(6, 6))
idx = list(range(0, len(bat.traj), 6))
im = axc.imshow(bat.owner_hist[0].reshape(bat.gx.shape), origin='lower', extent=(0, 100, 0, 100), cmap='tab10', vmin=-1, vmax=9, alpha=0.35)
axc.contour(bat.gx, bat.gy, bat.phi, levels=6, colors='k', alpha=0.25)
axc.scatter(*bat.charger, marker='s', s=160, c='gold', edgecolors='k', zorder=4)
dots = axc.scatter(*bat.traj[0].T, s=40, c='tab:blue', edgecolors='k', zorder=5)
ttl = axc.set_title('')
def upd_c(f):
    k = idx[f]; dots.set_offsets(bat.traj[k]); im.set_data(np.ma.masked_less(bat.owner_hist[k].reshape(bat.gx.shape), 0))
    b = bat.batt_hist[k]; dots.set_color(plt.cm.RdYlGn(b / 100)); dots.set_edgecolor('k')
    ttl.set_text(f't={k * bat.dt:.0f}s | covering {bat.active_hist[k]}/{bat.n} | min battery {b.min():.0f}%'); return [dots]
gif(animation.FuncAnimation(fig, upd_c, frames=len(idx), interval=70), 'coverage_battery.gif', fps=12)

# =================================================================
# 5. Consensus + leader-follower
# =================================================================
rng = np.random.default_rng(1); p0 = rng.uniform(0, 60, (12, 2))
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
for R, c in [(15, 'tab:red'), (25, 'tab:orange'), (35, 'tab:green'), (45, 'tab:blue')]:
    _, dis, lam = run_consensus(p0, comm_range=R)
    ax[0].semilogy(np.maximum(dis, 1e-4), c=c, label=f'range {R} m (lambda2={max(lam[0], 0):.2f})')
ax[0].set_title('Rendezvous: disagreement'); ax[0].set_xlabel('iteration'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
tr, _, _ = run_consensus(p0, comm_range=45)
for i in range(12): ax[1].plot(tr[:, i, 0], tr[:, i, 1], lw=.8)
ax[1].scatter(*p0.T, c='k', s=12); ax[1].set_title('Rendezvous paths (range 45 m)')
tr, _, _ = run_consensus(p0, offsets=formation_circle(12, 12)[:, :2], comm_range=45)
for i in range(12): ax[2].plot(tr[:, i, 0], tr[:, i, 1], lw=.8)
ax[2].scatter(*tr[-1].T, c='tab:blue', s=25); ax[2].set_title('Distributed circle formation'); ax[2].set_aspect('equal')
path = np.c_[np.linspace(30, 90, 300), 30 + 20 * np.sin(np.linspace(0, 2 * np.pi, 300))]
tr, err = leader_follower(p0, path)
for i in range(1, 12): ax[3].plot(tr[:, i, 0], tr[:, i, 1], lw=.6, alpha=.6)
ax[3].plot(path[:, 0], path[:, 1], 'k--', lw=1.5, label='leader path'); ax[3].legend(); ax[3].set_title('Leader-follower (pinning)')
save(fig, 'consensus.png')

# =================================================================
# 6. Task allocation (auction)
# =================================================================
rows = []
for seed in range(10):
    r = np.random.default_rng(seed); starts, tasks = r.uniform(0, 10, (5, 2)), r.uniform(0, 100, (30, 2))
    for name, fn in [('auction', auction_allocation), ('greedy nearest', greedy_nearest), ('random', random_allocation)]:
        rows.append(dict(method=name, **evaluate(starts, tasks, fn(starts, tasks))))
alloc = pd.DataFrame(rows).groupby('method')[['makespan', 'total']].mean().round(1)
alloc.to_csv(os.path.join(OUT, 'task_allocation.csv')); print(alloc.to_string(), '\n')
r = np.random.default_rng(3); starts, tasks = r.uniform(0, 10, (5, 2)), r.uniform(0, 100, (30, 2))
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, (name, fn) in zip(ax, [('auction', auction_allocation), ('greedy nearest', greedy_nearest), ('random', random_allocation)]):
    routes = fn(starts, tasks); ev = evaluate(starts, tasks, routes)
    for i, rt in enumerate(routes):
        P = np.vstack([starts[i], tasks[rt]]) if rt else starts[i:i + 1]
        a.plot(P[:, 0], P[:, 1], '-o', ms=3, lw=1.2)
    a.scatter(*starts.T, marker='^', s=90, c='k', zorder=4)
    a.set_title(f"{name}\nmakespan {ev['makespan']:.0f} m | total {ev['total']:.0f} m"); a.set_aspect('equal')
save(fig, 'task_allocation.png')

# =================================================================
# 7. Learning: evolution strategy tunes the controller gains
# =================================================================
c = SwarmConfig()
theta0 = [c.w_sep, c.w_ali, c.w_coh, c.w_goal, c.w_obs, c.separation_radius]
best, hist = evolve(theta0, generations=10, popsize=8, verbose=True)
H = pd.DataFrame(hist)
test = tuple(range(10, 18))
res = []
for name, th in [('default (hand-tuned)', theta0), ('learned (ES)', best)]:
    J, det = es_evaluate(th, seeds=test, return_details=True)
    D = pd.DataFrame(det)
    res.append(dict(controller=name, J=round(J, 1), time_s=round(D.time.mean(), 2), near_misses=int(D.near_misses.sum()),
                    building_hits=int(D.building_hits.sum()), min_sep_m=round(D.min_sep.min(), 2), success=D.reached.mean(),
                    **{p: round(v, 2) for p, v in zip(PARAMS, th)}))
learn = pd.DataFrame(res); learn.to_csv(os.path.join(OUT, 'learning.csv'), index=False); print(learn.to_string(), '\n')
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(H.gen, H.best, 'o-'); ax[0].set_title('ES: best fitness J (lower = better)'); ax[0].set_xlabel('generation'); ax[0].grid(alpha=.3)
x = np.arange(len(PARAMS))
ax[1].bar(x - 0.2, theta0, 0.4, label='default'); ax[1].bar(x + 0.2, best, 0.4, label='learned')
ax[1].set_xticks(x); ax[1].set_xticklabels(PARAMS, rotation=20); ax[1].legend(); ax[1].set_title('Controller gains')
save(fig, 'learning.png')
print('Done ->', os.path.abspath(OUT))
