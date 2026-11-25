import json
import numpy as np
import pytest

from drone_swarm import (SwarmConfig, DroneSwarm, FORMATIONS, make_city, SearchAndRescue,
                         QuadrotorDynamics, Wind, CommModel, random_obstacles)
from drone_swarm.coverage import CoverageSwarm
from drone_swarm.consensus import run_consensus
from drone_swarm.allocation import auction_allocation, random_allocation, evaluate
from drone_swarm.export import to_csv, to_qgc_plans, to_waypoint_json, enu_to_geodetic
from drone_swarm.formations import formation_circle


@pytest.mark.parametrize("name", list(FORMATIONS))
def test_formation_shapes(name):
    off = FORMATIONS[name](20)
    assert off.shape == (20, 3)
    d = np.linalg.norm(off[:, None] - off[None], axis=-1) + np.eye(20) * 1e9
    assert d.min() > 1.0                       # slots are not on top of each other


def test_flocking_aligns_without_collisions():
    sw = DroneSwarm(SwarmConfig(n_drones=20), start_center=(50, 50, 25), spread=12, seed=0)
    sw.run(250)
    d = sw.df().iloc[50:]
    assert d.polarization.mean() > 0.6
    assert d.near_misses.sum() == 0


def test_city_navigation_reaches_goal_safely():
    goal = np.array([80.0, 75.0, 20.0])
    sw = DroneSwarm(SwarmConfig(n_drones=20), start_center=(12, 12, 10), seed=2)
    sw.buildings, sw.goal = make_city(seed=3, n=7), goal
    sw.run(700, until=lambda s: np.linalg.norm(s.pos.mean(0) - goal) < 6)
    assert np.linalg.norm(sw.pos.mean(0) - goal) < 6
    assert sw.building_hits == 0


def test_formation_converges_and_replans_after_failure():
    sw = DroneSwarm(SwarmConfig(n_drones=16), start_center=(30, 30, 10), spread=5, seed=1)
    sw.set_formation('circle', center=(50, 50, 25))
    sw.run(200)
    assert sw.df().formation_error.iloc[-1] < 0.5
    dead = sw.fail_drones(4)
    assert (~sw.alive).sum() == 4 and len(dead) == 4
    sw.run(150)
    assert sw.df().formation_error.iloc[-1] < 0.5
    assert sw.pos[~sw.alive, 2].max() < 1e-6     # failed drones have landed


def test_quadrotor_respects_tilt_and_thrust_limits():
    n = 10
    sw = DroneSwarm(SwarmConfig(n_drones=n), start_center=(50, 50, 20), seed=0)
    sw.dynamics = QuadrotorDynamics(n, tilt_max_deg=30)
    sw.wind = Wind(mean=(4, 0, 0))
    sw.goal = np.array([90.0, 90.0, 30.0])
    for _ in range(200):
        sw.step()
        assert sw.dynamics.tilt_deg().max() <= 30.0 + 1e-6
        assert (sw.dynamics.T <= sw.dynamics.T_max + 1e-9).all()
    assert np.isfinite(sw.pos).all()


def test_comm_model_packet_loss_ratio():
    n = 15
    sw = DroneSwarm(SwarmConfig(n_drones=n), start_center=(50, 50, 25), seed=0)
    sw.comms = CommModel(n, range_m=100, p_loss=0.5, delay_steps=3)
    sw.run(100)
    assert 0.4 < sw.comms.delivery_ratio() < 0.6
    assert sw.comms.mean_age() >= 3


def test_moving_obstacles_are_avoided():
    sw = DroneSwarm(SwarmConfig(n_drones=20), start_center=(12, 12, 20), seed=0)
    sw.goal = np.array([85.0, 85.0, 25.0])
    sw.moving_obstacles = random_obstacles(5, speed=6, seed=10)
    sw.run(600, until=lambda s: np.linalg.norm(s.pos.mean(0) - s.goal) < 6)
    assert sw.obstacle_hits == 0


def test_two_teams_cross_safely():
    n = 20
    sw = DroneSwarm(SwarmConfig(n_drones=n), seed=1)
    rng = np.random.default_rng(0)
    sw.team = np.r_[np.zeros(10, int), np.ones(10, int)]
    sw.pos[:10] = np.c_[rng.uniform(8, 20, 10), rng.uniform(35, 65, 10), rng.uniform(20, 30, 10)]
    sw.pos[10:] = np.c_[rng.uniform(80, 92, 10), rng.uniform(35, 65, 10), rng.uniform(20, 30, 10)]
    sw.vel[:] = 0
    g = np.zeros((n, 3)); g[:10] = [88, 50, 25]; g[10:] = [12, 50, 25]
    sw.goal = g
    sw.run(300)
    assert sw.df().iloc[1:].min_cross_team.min() > 1.0


def test_search_and_rescue_finds_everyone():
    for strat in ['pso', 'lawnmower', 'random']:
        r = SearchAndRescue(n_drones=10, area=100, n_survivors=4, strategy=strat, seed=0).run(max_steps=3000)
        assert r['all_found'], strat


def test_coverage_cost_decreases():
    c = CoverageSwarm(n=8, seed=0).run(80)
    assert c.cost[-1] < 0.2 * c.cost[0]


def test_battery_coverage_never_runs_empty():
    c = CoverageSwarm(n=8, seed=0, battery=True).run(400)
    assert np.min(c.batt_hist) > 0


def test_consensus_needs_connectivity():
    rng = np.random.default_rng(1)
    p0 = rng.uniform(0, 60, (12, 2))
    _, dis_conn, lam = run_consensus(p0, comm_range=45)
    assert lam[0] > 0 and dis_conn[-1] < 1e-2
    _, dis_disc, lam = run_consensus(p0, comm_range=15)
    assert lam[0] < 1e-9 and dis_disc[-1] > 1.0
    _, dis_f, _ = run_consensus(p0, offsets=formation_circle(12, 12)[:, :2], comm_range=45)
    assert dis_f[-1] < 1e-2


def test_auction_beats_random_allocation():
    rng = np.random.default_rng(3)
    starts, tasks = rng.uniform(0, 10, (5, 2)), rng.uniform(0, 100, (25, 2))
    a = evaluate(starts, tasks, auction_allocation(starts, tasks))
    r = evaluate(starts, tasks, random_allocation(starts, tasks))
    assert a['tasks_done'] == 25
    assert a['total'] < r['total'] and a['makespan'] < r['makespan']


def test_exports(tmp_path):
    sw = DroneSwarm(SwarmConfig(n_drones=4))
    sw.set_formation('circle', center=(50, 50, 20))
    sw.run(60)
    df = to_csv(sw, tmp_path / 't.csv')
    assert set(df.drone) == {0, 1, 2, 3}
    files = to_qgc_plans(sw, tmp_path / 'plans')
    plan = json.load(open(files[0]))
    assert plan['fileType'] == 'Plan' and len(plan['mission']['items']) >= 3
    out = to_waypoint_json(sw, tmp_path / 'w.json')
    assert len(out['drones']) == 4
    ll = enu_to_geodetic([[0, 111.32, 0]], 0.0, 0.0)
    assert abs(ll[0, 0] - 0.001) < 1e-4
