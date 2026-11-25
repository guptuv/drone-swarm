"""Drone Swarm Robotics Simulator."""
from .core import SwarmConfig, Building, make_city, clip_norm, unit
from .formations import (FORMATIONS, formation_circle, formation_v, formation_grid,
                         formation_helix, formation_sphere)
from .swarm import DroneSwarm
from .sar import SearchAndRescue
from .dynamics import QuadrotorDynamics, Wind
from .comms import CommModel
from .obstacles import MovingObstacle, random_obstacles
from .coverage import CoverageSwarm, gaussian_density
from .consensus import run_consensus, leader_follower, algebraic_connectivity
from .allocation import auction_allocation, greedy_nearest, random_allocation
from .viz import animate_swarm, plot_metrics, animate_sar, draw_building

__version__ = "2.0.0"
