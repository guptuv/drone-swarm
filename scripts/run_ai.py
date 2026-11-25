"""v3: deep learning, reinforcement learning and classical robotics experiments.

Usage (from the repository root):
    python scripts/run_ai.py              # loads pretrained weights from models/ when present
    python scripts/run_ai.py --retrain    # trains every model from scratch (~40 min on 2 CPU cores,
                                          #  much faster on a Colab GPU for the CNN parts)
Outputs go to results/ai/ (PNG, GIF, CSV) and models/ (PyTorch weights).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import torch

from drone_swarm import make_city
from drone_swarm.ai import marl, gnn, dqn_search, vision, prediction, anomaly, mbrl
from drone_swarm.robotics import estimation, planning, cbf, mapping

ap = argparse.ArgumentParser()
ap.add_argument('--retrain', action='store_true')
args = ap.parse_args()

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'ai'); MOD = os.path.join(ROOT, 'models')
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)
plt.rcParams['figure.dpi'] = 85
torch.set_num_threads(max(1, os.cpu_count() or 1))
SUMMARY = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight'); plt.close(fig); print('  saved', name)


def csv(df, name):
    df.to_csv(os.path.join(OUT, name), index=False); print(df.to_string(index=False), '\n')


def have(name):
    return os.path.exists(os.path.join(MOD, name)) and not args.retrain


def section(title):
    print('\n' + '=' * 70 + f'\n{title}\n' + '=' * 70)
    return time.time()


# ======================================================================= 1. MAPPO
t0 = section('1. Multi-agent RL (MAPPO, attention policy) + CBF safety shield')
pol = marl.AttentionPolicy(5, 3)
if have('mappo_swarm_nav.pt'):
    pol.load_state_dict(torch.load(os.path.join(MOD, 'mappo_swarm_nav.pt')))
    hist = json.load(open(os.path.join(MOD, 'mappo_history.json')))
else:
    pol, hist = marl.train_mappo(updates=400, log_every=20, env_kw=dict(w_hit_obs=10.0), eval_every=20)
    torch.save(pol.state_dict(), os.path.join(MOD, 'mappo_swarm_nav.pt'))
    json.dump(hist, open(os.path.join(MOD, 'mappo_history.json'), 'w'))
H = pd.DataFrame(hist)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(H['update'], H.ep_return.rolling(5, min_periods=1).mean()); ax[0].set_title('Mean episode return per drone')
ax[1].plot(H['update'], H.success.rolling(5, min_periods=1).mean()); ax[1].set_title('Fraction of drones reaching goal')
ax[2].plot(H['update'], H.collisions_per_ep.rolling(5, min_periods=1).mean()); ax[2].set_title('Collision steps per episode')
for a in ax:
    a.set_xlabel('PPO update'); a.grid(alpha=.3)
    if 'phase' in H and (H.phase != H.phase.iloc[0]).any():
        a.axvline(H['update'][(H.phase != H.phase.iloc[0]).idxmax()], c='k', ls=':', lw=1)
save(fig, 'mappo_training.png')
rows = []
ctrls = [('Potential field (aggressive)', marl.PotentialFieldController()),
         ('Potential field (safe)', marl.PotentialFieldController(k_rep=50, rep_range=5)),
         ('MAPPO policy', pol), ('MAPPO + CBF shield', marl.CBFShield(pol))]
for n in [8, 16]:
    for name, c in ctrls:
        r = marl.evaluate(c, n_agents=n, episodes=64)
        rows.append(dict(drones=n, controller=name, success=r['success_rate'], collision_free_eps=r['collision_free_eps'],
                         agent_collisions=r['agent_collisions_per_ep'], obstacle_collisions=r['obstacle_collisions_per_ep'],
                         time_s=r['mean_time_s']))
M = pd.DataFrame(rows).round(3); csv(M, 'mappo_comparison.csv'); SUMMARY['mappo'] = M.to_dict('records')
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
for a, (name, c) in zip(ax, [('Potential field (aggressive)', ctrls[0][1]), ('MAPPO + CBF shield', ctrls[3][1])]):
    r = marl.evaluate(c, n_agents=8, episodes=4, seed=5, record=True)
    T = r['traj']
    for k, (x, y) in enumerate(r['obs_c']):
        if r['obs_r'][k] > 0: a.add_patch(plt.Circle((x, y), r['obs_r'][k], color='gray', alpha=.5))
    for i in range(T.shape[1]):
        a.plot(T[:, i, 0], T[:, i, 1], lw=1.5); a.scatter(*T[0, i], marker='o', c='k', s=15); a.scatter(*r['goal'][i], marker='*', s=80, c='gold', edgecolors='k')
    a.set_title(f"{name}\nsuccess {r['success_rate']:.0%} | collision steps {r['collision_steps_per_ep']:.1f}"); a.set_aspect('equal'); a.set_xlim(0, 40); a.set_ylim(0, 40)
save(fig, 'mappo_trajectories.png')
r = marl.evaluate(marl.CBFShield(pol), n_agents=12, episodes=1, seed=9, record=True)
T = r['traj']
fig, a = plt.subplots(figsize=(6, 6))
for k, (x, y) in enumerate(r['obs_c']):
    if r['obs_r'][k] > 0: a.add_patch(plt.Circle((x, y), r['obs_r'][k], color='gray', alpha=.5))
a.scatter(*r['goal'].T, marker='*', s=90, c='gold', edgecolors='k'); a.set_xlim(0, 40); a.set_ylim(0, 40); a.set_aspect('equal')
dots = a.scatter(*T[0].T, s=60, c=np.arange(T.shape[1]), cmap='tab20', edgecolors='k', zorder=3)
lines = [a.plot([], [], lw=1, alpha=.6)[0] for _ in range(T.shape[1])]
ttl = a.set_title('')
def upd(f):
    dots.set_offsets(T[f])
    for i, l in enumerate(lines): l.set_data(T[max(0, f - 15):f + 1, i, 0], T[max(0, f - 15):f + 1, i, 1])
    ttl.set_text(f'MAPPO + CBF shield, 12 drones (trained on 8) | t={f * 0.2:.1f}s'); return [dots]
animation.FuncAnimation(fig, upd, frames=len(T), interval=80).save(os.path.join(OUT, 'mappo_swap.gif'), writer=PillowWriter(fps=12), dpi=65)
plt.close('all'); print(f'  section time {time.time() - t0:.0f}s')

# ======================================================================= 2. GNN
t0 = section('2. GNN imitation learning (decentralised flocking)')
if have('gnn_flocking.pt'):
    g = gnn.FlockGNN(); g.load_state_dict(torch.load(os.path.join(MOD, 'gnn_flocking.pt')))
    ghist = json.load(open(os.path.join(MOD, 'gnn_history.json')))
else:
    g, ghist = gnn.train_gnn(epochs=30, episodes_per_epoch=6)
    torch.save(g.state_dict(), os.path.join(MOD, 'gnn_flocking.pt')); json.dump(ghist, open(os.path.join(MOD, 'gnn_history.json'), 'w'))
G = pd.DataFrame(gnn.compare(g, sizes=(20, 40, 80), episodes=8)).round(3); csv(G, 'gnn_comparison.csv'); SUMMARY['gnn'] = G.to_dict('records')
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
gh = pd.DataFrame(ghist); ax[0].plot(gh.epoch, gh.gnn_cost, 'o-', label='GNN (rollout)'); ax[0].set_title('GNN flocking cost during DAgger'); ax[0].set_xlabel('epoch'); ax[0].grid(alpha=.3)
piv = G.pivot(index='n', columns='controller', values='cost_mean')[['expert', 'gnn', 'local']]
piv.plot.bar(ax=ax[1], rot=0); ax[1].set_title('Velocity-alignment cost (lower = better)'); ax[1].set_xlabel('swarm size (trained on 40)')
env = gnn.FlockEnv(n=40)
for name, c in [('local', 'tab:red'), ('gnn', 'tab:green'), ('expert', 'tab:blue')]:
    cs = np.mean([gnn.rollout(env, name, np.random.default_rng(50 + e), model=g)[0] for e in range(5)], 0)
    ax[2].semilogy(np.arange(len(cs)) * env.dt, cs, c=c, label=name)
ax[2].set_title('Cost over time (40 drones)'); ax[2].set_xlabel('t (s)'); ax[2].legend(); ax[2].grid(alpha=.3)
save(fig, 'gnn_flocking.png'); print(f'  section time {time.time() - t0:.0f}s')

# ======================================================================= 3. DQN
t0 = section('3. Deep Q-Network for probabilistic search')
q = dqn_search.QNet()
if have('dqn_search.pt'):
    q.load_state_dict(torch.load(os.path.join(MOD, 'dqn_search.pt'))); qh = json.load(open(os.path.join(MOD, 'dqn_history.json')))
else:
    q, qh = dqn_search.train_dqn(steps=3000)
    torch.save(q.state_dict(), os.path.join(MOD, 'dqn_search.pt')); json.dump(qh, open(os.path.join(MOD, 'dqn_history.json'), 'w'))
res = {p: dqn_search.run_policy(p, episodes=512, seed=2024, model=q) for p in ['random', 'lawnmower', 'greedy', 'dqn']}
D = pd.DataFrame([dict(policy=p, p_detect_60_steps=round(r['p_detect'], 3), p_detect_20_steps=round(r['curve'][20], 3)) for p, r in res.items()])
csv(D, 'dqn_search.csv'); SUMMARY['dqn'] = D.to_dict('records')
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
qd = pd.DataFrame(qh); ax[0].plot(qd.transitions / 1e3, qd.p_detect); ax[0].set_title('DQN training: P(detect) on eval maps'); ax[0].set_xlabel('transitions (k)'); ax[0].grid(alpha=.3)
for p, r in res.items(): ax[1].plot(r['curve'], label=p)
ax[1].set_title('Cumulative detection probability'); ax[1].set_xlabel('step'); ax[1].legend(); ax[1].grid(alpha=.3)
for a, p in zip(ax[2:], ['greedy', 'dqn']):
    r = res[p]; a.imshow(r['prior'].T, origin='lower', cmap='magma'); a.plot(r['path'][:, 0], r['path'][:, 1], 'c-o', ms=3)
    a.set_title(f'{p} path over prior belief (episode 1)')
save(fig, 'dqn_search.png'); print(f'  section time {time.time() - t0:.0f}s')

# ======================================================================= 4. Vision
t0 = section('4. CNN survivor detection from aerial images')
det = vision.DetectorCNN()
if have('detector_cnn.pt'):
    det.load_state_dict(torch.load(os.path.join(MOD, 'detector_cnn.pt'))); vh = json.load(open(os.path.join(MOD, 'detector_history.json')))
else:
    det, vh = vision.train_detector(n_train=16000, epochs=10)
    torch.save(det.state_dict(), os.path.join(MOD, 'detector_cnn.pt')); json.dump(vh, open(os.path.join(MOD, 'detector_history.json'), 'w'))
VA = pd.DataFrame(vision.performance_vs_altitude(det, n=600)).round(3); csv(VA, 'vision_altitude.csv')
TR = pd.DataFrame(vision.search_tradeoff(VA.to_dict('records'))).round(3); csv(TR, 'vision_search_tradeoff.csv')
fpr, tpr, auc = vision.roc_curve(det); SUMMARY['vision'] = dict(auc=auc, by_altitude=VA.to_dict('records'))
print(f'  ROC AUC = {auc:.3f}')
rng = np.random.default_rng(3); X, y, alt = vision.make_dataset(24, rng)
with torch.no_grad(): ps = torch.sigmoid(det(torch.tensor(X))).numpy()
fig, ax = plt.subplots(3, 8, figsize=(15, 6.3))
for k, a in enumerate(ax.flat):
    a.imshow(X[k].transpose(1, 2, 0)); a.axis('off')
    a.set_title(f"{'person' if y[k] else 'empty'} {alt[k]:.0f}m\np={ps[k]:.2f}", fontsize=8, color='g' if (ps[k] > .5) == y[k] else 'r')
save(fig, 'vision_samples.png')
fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], 'k--', lw=.8); ax[0].set_title(f'ROC (AUC = {auc:.3f})'); ax[0].set_xlabel('false positive rate'); ax[0].set_ylabel('true positive rate')
ax[1].plot(VA.altitude_m, VA.recall, 'o-', label='recall'); ax[1].plot(VA.altitude_m, VA.precision, 's-', label='precision'); ax[1].plot(VA.altitude_m, VA.accuracy, '^-', label='accuracy')
ax[1].set_title('Detection vs flight altitude'); ax[1].set_xlabel('altitude (m)'); ax[1].legend(); ax[1].grid(alpha=.3)
ax[2].plot(TR.altitude_m, TR.detections_per_min, 'o-'); ax[2].set_title('Search efficiency: expected detections / min'); ax[2].set_xlabel('altitude (m)'); ax[2].grid(alpha=.3)
best_alt = TR.loc[TR.detections_per_min.idxmax(), 'altitude_m']; ax[2].axvline(best_alt, c='r', ls='--', label=f'best {best_alt:.0f} m'); ax[2].legend()
SUMMARY['vision']['best_altitude'] = float(best_alt)
save(fig, 'vision_performance.png'); print(f'  section time {time.time() - t0:.0f}s')

# ======================================================================= 5. Trajectory prediction
t0 = section('5. LSTM / Transformer trajectory prediction + prediction-aware avoidance')
X, Y = prediction.make_windows(prediction.simulate_intruders(400, 300, np.random.default_rng(0)))
Xt, Yt = prediction.make_windows(prediction.simulate_intruders(120, 300, np.random.default_rng(1)))
lstm, tfm = prediction.LSTMPredictor(), prediction.TransformerPredictor()
if have('lstm_predictor.pt') and have('transformer_predictor.pt'):
    lstm.load_state_dict(torch.load(os.path.join(MOD, 'lstm_predictor.pt'))); tfm.load_state_dict(torch.load(os.path.join(MOD, 'transformer_predictor.pt')))
else:
    prediction.train_predictor(lstm, X, Y, epochs=20, name='LSTM'); prediction.train_predictor(tfm, X, Y, epochs=20, name='Transformer')
    torch.save(lstm.state_dict(), os.path.join(MOD, 'lstm_predictor.pt')); torch.save(tfm.state_dict(), os.path.join(MOD, 'transformer_predictor.pt'))
with torch.no_grad():
    pl = lstm(torch.tensor(Xt)).numpy(); pt = tfm(torch.tensor(Xt)).numpy()
preds = {'constant velocity': prediction.predict_cv(Xt), 'constant turn rate (CTRV)': prediction.predict_ctrv(Xt), 'LSTM': pl, 'Transformer': pt}
P = pd.DataFrame([dict(model=k, ADE_m=prediction.ade_fde(v, Yt)[0], FDE_m=prediction.ade_fde(v, Yt)[1]) for k, v in preds.items()]).round(3)
lf = lambda h: lstm(torch.tensor(h, dtype=torch.float32)).detach().numpy()
av = []
for name, fn in [('no avoidance', lambda h: np.full((1, prediction.F, 2), 1e4)), ('constant velocity', prediction.predict_cv),
                 ('CTRV', prediction.predict_ctrv), ('LSTM', lf)]:
    av.append(dict(predictor=name, **prediction.avoidance_trial(fn, n_trials=400)))
AV = pd.DataFrame(av).round(3)
csv(P, 'prediction_accuracy.csv'); csv(AV, 'prediction_avoidance.csv'); SUMMARY['prediction'] = dict(acc=P.to_dict('records'), avoid=AV.to_dict('records'))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for k, i in enumerate([3, 250, 777]):
    a = ax[k]; a.plot(Xt[i, :, 0], Xt[i, :, 1], 'k.-', label='observed (1 s)'); a.plot(Yt[i, :, 0], Yt[i, :, 1], 'k--', lw=2, label='true future (2 s)')
    for (nm, v), c in zip(preds.items(), ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']): a.plot(v[i, :, 0], v[i, :, 1], c=c, label=nm)
    a.set_aspect('equal'); a.set_title(f'Manoeuvring intruder #{k + 1}')
ax[0].legend(fontsize=7)
save(fig, 'prediction_examples.png'); print(f'  section time {time.time() - t0:.0f}s')

# ======================================================================= 6. Fault detection
t0 = section('6. Autoencoder fault detection from telemetry')
Xh = anomaly.healthy_training_set(runs=12)
fd = anomaly.FaultDetector()
if have('fault_autoencoder.pt'):
    ck = torch.load(os.path.join(MOD, 'fault_autoencoder.pt'), weights_only=False)
    fd.mu, fd.sd, fd.thr = ck['mu'], ck['sd'], ck['thr']; fd.model = anomaly.AutoEncoder(len(fd.mu)); fd.model.load_state_dict(ck['state'])
else:
    fd.fit(Xh, epochs=40)
    torch.save(dict(state=fd.model.state_dict(), mu=fd.mu, sd=fd.sd, thr=fd.thr), os.path.join(MOD, 'fault_autoencoder.pt'))
zr = anomaly.ZScoreRule().fit(Xh)
FA = pd.concat([pd.DataFrame(anomaly.detection_report(fd, runs=5)).assign(detector='autoencoder'),
                pd.DataFrame(anomaly.detection_report(zr, runs=5)).assign(detector='z-score rule')]).round(3)
csv(FA, 'fault_detection.csv'); SUMMARY['faults'] = FA.to_dict('records')
fig, ax = plt.subplots(1, 4, figsize=(20, 3.8), sharey=True)
for a, f in zip(ax, anomaly.FAULTS):
    feats, labels, tf = anomaly.fly(f, seed=4242)
    sc = np.array([fd.score(feats[t:t + anomaly.WIN].transpose(1, 0, 2).reshape(8, -1).astype(np.float32)) for t in range(60, len(feats) - anomaly.WIN)])
    tt = (np.arange(60, len(feats) - anomaly.WIN) + anomaly.WIN - 1) * 0.1
    a.semilogy(tt, sc[:, 1:], c='gray', alpha=.4, lw=.8); a.semilogy(tt, sc[:, 0], c='tab:red', lw=1.6, label='drone 0')
    a.axhline(fd.thr, c='k', ls='--', label='threshold')
    if f != 'none': a.axvline(tf * 0.1, c='tab:red', ls=':', label='fault starts')
    a.set_title(f'fault: {f}'); a.set_xlabel('t (s)')
ax[0].set_ylabel('reconstruction error'); ax[0].legend(fontsize=7)
save(fig, 'fault_detection.png'); print(f'  section time {time.time() - t0:.0f}s')

# ======================================================================= 7. Model-based RL
t0 = section('7. Learned dynamics + MPPI')
Xd, Yd = mbrl.collect_data(60000)
dm = mbrl.DynamicsMLP()
if have('dynamics_mlp.pt'):
    dm.load_state_dict(torch.load(os.path.join(MOD, 'dynamics_mlp.pt'))); dh = json.load(open(os.path.join(MOD, 'dynamics_history.json')))
else:
    dm, dh = mbrl.train_dynamics(Xd, Yd, epochs=40)
    torch.save(dm.state_dict(), os.path.join(MOD, 'dynamics_mlp.pt')); json.dump(dh, open(os.path.join(MOD, 'dynamics_history.json'), 'w'))
rows, paths = [], {}
for name, mk in [('PD controller', lambda: mbrl.pd_controller), ('MPPI + learned model', lambda: mbrl.MPPI(dm).act),
                 ('MPPI + true model', lambda: mbrl.MPPI('true').act)]:
    rs = [mbrl.fly_slalom(mk(), seed=s) for s in range(6)]
    paths[name] = rs[0]['path']
    rows.append(dict(controller=name, reached=np.mean([r['reached'] for r in rs]), crashed=np.mean([r['crashed'] for r in rs]),
                     time_s=np.nanmean([r['time_s'] for r in rs]), min_clearance_m=np.min([r['min_clearance'] for r in rs])))
MB = pd.DataFrame(rows).round(3); csv(MB, 'mbrl_slalom.csv'); SUMMARY['mbrl'] = dict(val_err=dh[-1]['val_accel_err'], rows=MB.to_dict('records'))
fig, ax = plt.subplots(1, 2, figsize=(16, 4.3), gridspec_kw=dict(width_ratios=[1, 2.5]))
ax[0].plot([d['epoch'] for d in dh], [d['val_accel_err'] for d in dh]); ax[0].set_title('Learned dynamics: validation error'); ax[0].set_ylabel('acceleration error (m/s^2)'); ax[0].set_xlabel('epoch'); ax[0].grid(alpha=.3)
for x, y, r in mbrl.PILLARS: ax[1].add_patch(plt.Circle((x, y), r, color='gray', alpha=.6))
for (nm, p), c in zip(paths.items(), ['tab:red', 'tab:green', 'tab:blue']): ax[1].plot(p[:, 0], p[:, 1], c=c, label=nm, lw=2)
ax[1].scatter(*mbrl.GOAL[:2], marker='*', s=200, c='gold', edgecolors='k'); ax[1].set_aspect('equal'); ax[1].legend(); ax[1].set_title('Pillar slalom in wind (top view)')
save(fig, 'mbrl_slalom.png'); print(f'  section time {time.time() - t0:.0f}s')

# ======================================================================= 8. Classical robotics
t0 = section('8. Classical robotics: Kalman filter, particle filter, A*/RRT*, CBF, mapping')
sim = estimation.simulate_flight(); kf = estimation.run_kf(sim)
fig, ax = plt.subplots(1, 3, figsize=(18, 4.3))
e_kf = np.linalg.norm(kf['est'][:, :3] - sim['pos'], axis=1); e_g = np.linalg.norm(kf['gps_hold'] - sim['pos'], axis=1)
ax[0].plot(sim['t'], e_g, c='tab:orange', alpha=.6, label=f"GPS only (RMSE {kf['rmse_gps']:.2f} m)")
ax[0].plot(sim['t'], e_kf, c='tab:blue', label=f"Kalman GPS+IMU (RMSE {kf['rmse_kf']:.2f} m)")
ax[0].plot(sim['t'], 2 * np.linalg.norm(kf['sigma'], axis=1), 'k--', lw=.8, label='2-sigma'); ax[0].axvspan(30, 38, color='r', alpha=.1, label='GPS outage')
ax[0].set_ylim(0, 15); ax[0].set_title('Position error'); ax[0].set_xlabel('t (s)'); ax[0].legend(fontsize=7)
pfa = estimation.localise_survivor(0); pfb = estimation.localise_survivor(0, info_gain=False)
ax[1].plot(pfa['err'], label='orbit the estimate (active)'); ax[1].plot(pfb['err'], label='expanding spiral'); ax[1].set_yscale('log')
ax[1].set_title('Particle filter: survivor localisation error (RSSI)'); ax[1].set_xlabel('measurement'); ax[1].legend()
ax[2].scatter(*pfa['particles'].T, s=1, alpha=.3); ax[2].plot(*pfa['path'].T, 'c-', lw=.8); ax[2].scatter(*pfa['target'], marker='*', s=200, c='r')
ax[2].set_title('Particles + drone path'); ax[2].set_aspect('equal')
save(fig, 'estimation.png')
pf_rows = []
for sd in range(8):
    a_, b_ = estimation.localise_survivor(sd), estimation.localise_survivor(sd, info_gain=False)
    pf_rows += [dict(strategy='active (orbit estimate)', final_err_m=a_['err'][-1], mean_err_after_40=a_['err'][40:].mean()),
                dict(strategy='expanding spiral', final_err_m=b_['err'][-1], mean_err_after_40=b_['err'][40:].mean())]
PF = pd.DataFrame(pf_rows).groupby('strategy').mean().reset_index().round(2)
EST = pd.DataFrame([dict(method='GPS only (hold last fix)', rmse_m=kf['rmse_gps']), dict(method='IMU dead reckoning', rmse_m=kf['rmse_dead']),
                    dict(method='Kalman filter GPS+IMU', rmse_m=kf['rmse_kf'])]).round(2)
csv(EST, 'kalman.csv'); csv(PF, 'particle_filter.csv')
SUMMARY['estimation'] = dict(kf=EST.to_dict('records'), bias_true=[0.4, -0.3, 0.2], bias_est=kf['bias_est'].round(3).tolist(), pf=PF.to_dict('records'))

occ, res_ = planning.occupancy_from_buildings(make_city(seed=3, n=7))
pa, exp_ = planning.astar(occ, (12, 12), (80, 75)); t1 = time.time(); pr, nodes, par = planning.rrt_star(occ, (12, 12), (80, 75), iters=3000); t_rrt = time.time() - t1
t1 = time.time(); planning.astar(occ, (12, 12), (80, 75)); t_a = time.time() - t1
PL = pd.DataFrame([dict(planner='straight line', length_m=np.hypot(68, 63), collision_free=planning._free_segment(occ, 1.0, np.array([12., 12]), np.array([80., 75]))),
                   dict(planner='A* (8-connected grid)', length_m=planning.path_length(pa), collision_free=True, time_ms=t_a * 1e3),
                   dict(planner='A* + shortcut smoothing', length_m=planning.path_length(planning.shortcut(list(pa), occ)), collision_free=True),
                   dict(planner='RRT* (3000 samples)', length_m=planning.path_length(pr), collision_free=True, time_ms=t_rrt * 1e3),
                   dict(planner='RRT* + shortcut smoothing', length_m=planning.path_length(planning.shortcut(list(pr), occ)), collision_free=True)]).round(2)
csv(PL, 'planning.csv'); SUMMARY['planning'] = PL.to_dict('records')
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
for a, (nm, p) in zip(ax, [('A* + smoothing', planning.shortcut(list(pa), occ)), ('RRT*', pr)]):
    a.imshow(occ.T, origin='lower', cmap='Greys', alpha=.6)
    if nm == 'RRT*':
        for k, pp in enumerate(par):
            if pp >= 0: a.plot([nodes[k, 0], nodes[pp, 0]], [nodes[k, 1], nodes[pp, 1]], c='tab:blue', lw=.3, alpha=.4)
    else:
        a.plot(pa[:, 0], pa[:, 1], c='tab:orange', lw=1, alpha=.7, label='raw A*')
    a.plot(p[:, 0], p[:, 1], 'r-', lw=2.5, label=nm); a.scatter([12, 80], [12, 75], c=['g', 'r'], s=80, zorder=5); a.legend(); a.set_title(f'{nm}: {planning.path_length(p):.1f} m')
save(fig, 'planning.png')

cb = {c: cbf.antipodal_swap(use_cbf=c) for c in [False, True]}
CB = pd.DataFrame([dict(controller='nominal (straight to goal)', min_distance_m=cb[False]['min_dist'].min(), arrived=cb[False]['arrived'], violation_steps=cb[False]['violations']),
                   dict(controller='CBF safety filter', min_distance_m=cb[True]['min_dist'].min(), arrived=cb[True]['arrived'], violation_steps=cb[True]['violations'])]).round(3)
csv(CB, 'cbf.csv'); SUMMARY['cbf'] = CB.to_dict('records')
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, c in zip(ax[:2], [False, True]):
    T = cb[c]['traj']
    for i in range(T.shape[1]): a.plot(T[:, i, 0], T[:, i, 1], lw=1)
    a.set_aspect('equal'); a.set_title('CBF safety filter' if c else 'Nominal controller')
ax[2].plot(np.arange(len(cb[False]['min_dist'])) * 0.05, cb[False]['min_dist'], label='nominal'); ax[2].plot(np.arange(len(cb[True]['min_dist'])) * 0.05, cb[True]['min_dist'], label='CBF')
ax[2].axhline(2.0, c='r', ls='--', label='safety distance'); ax[2].set_title('Minimum inter-drone distance'); ax[2].set_xlabel('t (s)'); ax[2].legend()
save(fig, 'cbf.png')

world = mapping.make_world()
exf = mapping.explore(world, strategy='frontier', steps=250); exr = mapping.explore(world, strategy='random', steps=250)
MP = pd.DataFrame([dict(strategy='frontier + A*', known_after_100=exf['known'][99], known_after_250=exf['known'][-1]),
                   dict(strategy='random walk', known_after_100=exr['known'][99], known_after_250=exr['known'][-1])]).round(3)
csv(MP, 'mapping.csv'); SUMMARY['mapping'] = MP.to_dict('records')
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].imshow(world.T, origin='lower', cmap='Greys'); ax[0].set_title('True world')
prob = 1 / (1 + np.exp(-exf['logodds']))
ax[1].imshow(prob.T, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1)
for p in exf['paths']: ax[1].plot(p[:, 0], p[:, 1], lw=1)
ax[1].set_title('Map built by 3 drones (frontier exploration)')
ax[2].plot(exf['known'] * 100, label='frontier + A*'); ax[2].plot(exr['known'] * 100, label='random walk'); ax[2].set_title('% of free space mapped'); ax[2].set_xlabel('step'); ax[2].legend(); ax[2].grid(alpha=.3)
save(fig, 'mapping.png'); print(f'  section time {time.time() - t0:.0f}s')

json.dump(SUMMARY, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
print('\nAll AI / robotics results saved to', os.path.abspath(OUT))
