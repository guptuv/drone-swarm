"""Drone Swarm Robotics Simulator."""
from .core import SwarmConfig, Building, make_city, clip_norm, unit
from .formations import (FORMATIONS, formation_circle, formation_v, formation_grid,
                         formation_helix, formation_sphere)
from .swarm import DroneSwarm
from .sar import SearchAndRescue
from .viz import animate_swarm, plot_metrics, animate_sar, draw_building

__version__ = "1.0.0"
