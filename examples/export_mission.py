"""Simulate a formation flight and export it for real drones.

Writes to exports/: trajectories.csv, waypoints.json and one QGroundControl .plan per drone.
Change HOME_LAT / HOME_LON to your take-off location before flying anything.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from drone_swarm import SwarmConfig, DroneSwarm, formation_grid
from drone_swarm.export import to_csv, to_qgc_plans, to_waypoint_json

HOME_LAT, HOME_LON = 28.6139, 77.2090          # example location - replace with yours
OUT = 'exports'

n = 6
sw = DroneSwarm(SwarmConfig(n_drones=n, max_speed=4.0), seed=0)
sw.pos = np.c_[formation_grid(n, 4.0)[:, :2] + 10, np.zeros(n)]
sw.vel[:] = 0; sw.traj[0] = sw.pos.copy()
sw.set_formation('grid', center=(10, 10, 10), spacing=4); sw.run(80, phase='takeoff')
sw.set_formation('circle', center=(40, 40, 15), radius=10); sw.run(200, phase='circle')
sw.set_formation('v', center=sw.pos.mean(0), goal=(70, 20, 15), spacing=4); sw.run(250, phase='V transit')

os.makedirs(OUT, exist_ok=True)
to_csv(sw, os.path.join(OUT, 'trajectories.csv'), every=5)
to_waypoint_json(sw, os.path.join(OUT, 'waypoints.json'))
files = to_qgc_plans(sw, os.path.join(OUT, 'plans'), home_lat=HOME_LAT, home_lon=HOME_LON)
print(f"exported {len(files)} QGroundControl plans + CSV + JSON to {OUT}/")
