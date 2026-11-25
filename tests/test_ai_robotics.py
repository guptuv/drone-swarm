import os
import numpy as np
import pytest

from drone_swarm import make_city
from drone_swarm.robotics import estimation, planning, cbf, mapping

torch = pytest.importorskip("torch")
from drone_swarm.ai import marl, gnn, dqn_search, vision, prediction, anomaly, mbrl  # noqa: E402

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')


# ---------------------------------------------------------------- classical robotics
def test_kalman_filter_beats_gps_and_finds_bias():
    sim = estimation.simulate_flight(T=40.0)
    r = estimation.run_kf(sim)
    assert r['rmse_kf'] < 0.5 * r['rmse_gps']
    assert np.allclose(r['bias_est'], [0.4, -0.3, 0.2], atol=0.1)


def test_particle_filter_localises_survivor():
    r = estimation.localise_survivor(seed=1, steps=100)
    assert r['err'][-1] < 6.0 and r['err'][-1] < r['err'][5]


def test_astar_and_rrtstar_find_collision_free_paths():
    occ, res = planning.occupancy_from_buildings(make_city(seed=3, n=7))
    pa, _ = planning.astar(occ, (12, 12), (80, 75))
    pr, _, _ = planning.rrt_star(occ, (12, 12), (80, 75), iters=1500)
    for p in (pa, pr):
        assert p is not None
        assert all(planning._free_segment(occ, res, p[k], p[k + 1]) for k in range(len(p) - 1))
    assert planning.path_length(pa) < 1.2 * np.hypot(68, 63)


def test_cbf_keeps_distance_and_avoids_deadlock():
    nominal = cbf.antipodal_swap(n=10, use_cbf=False)
    safe = cbf.antipodal_swap(n=10, use_cbf=True)
    assert nominal['violations'] > 0
    assert safe['violations'] == 0 and safe['min_dist'].min() >= 1.9
    assert safe['arrived'] == 1.0


def test_frontier_exploration_beats_random():
    w = mapping.make_world(seed=1)
    f = mapping.explore(w, n_drones=2, steps=80, strategy='frontier')
    r = mapping.explore(w, n_drones=2, steps=80, strategy='random')
    assert f['known'][-1] > r['known'][-1]


# ---------------------------------------------------------------- learning-based
def test_marl_env_and_short_training():
    env = marl.SwarmNavEnv(n_agents=6, batch=4)
    obs = env.reset()
    assert obs.shape == (4, 6, env.obs_dim)
    pol, hist = marl.train_mappo(updates=2, n_agents=6, batch=4, verbose=False)
    assert len(hist) == 2 and np.isfinite(hist[-1]['ep_return'])


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS, 'mappo_swarm_nav.pt')), reason='pretrained weights not present')
def test_pretrained_mappo_with_shield_is_safe():
    pol = marl.AttentionPolicy(5, 3)
    pol.load_state_dict(torch.load(os.path.join(MODELS, 'mappo_swarm_nav.pt')))
    r = marl.evaluate(marl.CBFShield(pol), n_agents=8, episodes=32)
    assert r['arrived'] >= 0.95 and r['agent_collisions_per_ep'] == 0


def test_gnn_forward_is_permutation_equivariant():
    m = gnn.FlockGNN()
    env = gnn.FlockEnv(n=12).reset(np.random.default_rng(0))
    x, A = env.features()
    x, A = torch.tensor(x, dtype=torch.float32)[None], torch.tensor(A, dtype=torch.float32)[None]
    perm = torch.randperm(12)
    out = m(x, A)[0]
    out_p = m(x[:, perm], A[:, perm][:, :, perm])[0]
    assert torch.allclose(out[perm], out_p, atol=1e-5)


def test_dqn_search_env_and_baselines():
    r_g = dqn_search.run_policy('greedy', episodes=32)
    r_r = dqn_search.run_policy('random', episodes=32)
    assert 0 <= r_r['p_detect'] < r_g['p_detect'] <= 1


def test_vision_renderer_and_cnn_shapes():
    X, y, alt = vision.make_dataset(16, np.random.default_rng(0))
    assert X.shape == (16, 3, 32, 32) and X.min() >= 0 and X.max() <= 1
    assert vision.DetectorCNN()(torch.tensor(X)).shape == (16,)


def test_prediction_models_shapes_and_baselines():
    X, Y = prediction.make_windows(prediction.simulate_intruders(5, 100, np.random.default_rng(0)))
    for m in (prediction.LSTMPredictor(), prediction.TransformerPredictor()):
        assert m(torch.tensor(X[:7])).shape == (7, prediction.F, 2)
    assert prediction.ade_fde(prediction.predict_cv(X), Y)[0] > 0


def test_fault_injection_changes_telemetry():
    f0, l0, _ = anomaly.fly('none', steps=200, seed=3, t_fault=100)
    f1, l1, _ = anomaly.fly('imu', steps=200, seed=3, t_fault=100)
    assert l0.sum() == 0 and l1[150, 0] == 1
    assert np.abs(f1[150:, 0, 3:6] - f0[150:, 0, 3:6]).mean() > 0.5


def test_mppi_true_model_completes_slalom():
    r = mbrl.fly_slalom(mbrl.MPPI('true', K=128).act, seed=0)
    assert r['reached'] and not r['crashed']
