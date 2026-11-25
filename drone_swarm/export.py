"""Export simulated swarm trajectories for real drones / ground stations.

* to_csv           : t, drone, x, y, z, vx, vy, vz  (local ENU metres)
* to_qgc_plans     : one QGroundControl .plan mission per drone (lat/lon/alt waypoints)
* to_waypoint_json : compact JSON waypoint lists (e.g. for Crazyswarm / custom firmware)
"""
import json
import os
import numpy as np
import pandas as pd

EARTH_R = 6378137.0


def enu_to_geodetic(xyz, home_lat, home_lon, home_alt=0.0):
    """Flat-earth conversion (fine for areas of a few km)."""
    xyz = np.atleast_2d(xyz)
    lat = home_lat + np.degrees(xyz[:, 1] / EARTH_R)
    lon = home_lon + np.degrees(xyz[:, 0] / (EARTH_R * np.cos(np.radians(home_lat))))
    return np.c_[lat, lon, home_alt + xyz[:, 2]]


def to_csv(swarm, path, every=1):
    traj = np.array(swarm.traj)[::every]
    dt = swarm.cfg.dt * every
    rows = []
    for k, P in enumerate(traj):
        for i, p in enumerate(P):
            rows.append((round(k * dt, 3), i, *np.round(p, 3)))
    df = pd.DataFrame(rows, columns=['t', 'drone', 'x', 'y', 'z'])
    df[['vx', 'vy', 'vz']] = df.groupby('drone')[['x', 'y', 'z']].diff().fillna(0) / dt
    df.to_csv(path, index=False)
    return df


def _downsample(path_xyz, min_dist=5.0):
    keep = [path_xyz[0]]
    for p in path_xyz[1:]:
        if np.linalg.norm(p - keep[-1]) >= min_dist:
            keep.append(p)
    if np.linalg.norm(path_xyz[-1] - keep[-1]) > 1e-6:
        keep.append(path_xyz[-1])
    return np.array(keep)


def to_qgc_plans(swarm, out_dir, home_lat=28.6139, home_lon=77.2090, min_dist=5.0, speed=5.0):
    """Writes drone_XX.plan files (QGroundControl mission format, version 1)."""
    os.makedirs(out_dir, exist_ok=True)
    traj = np.array(swarm.traj)
    files = []
    for i in range(traj.shape[1]):
        wps = _downsample(traj[:, i], min_dist)
        geo = enu_to_geodetic(wps, home_lat, home_lon)
        items = [{"autoContinue": True, "command": 22, "doJumpId": 1, "frame": 3, "type": "SimpleItem",
                  "params": [0, 0, 0, None, float(geo[0, 0]), float(geo[0, 1]), max(float(geo[0, 2]), 3.0)]}]
        items.append({"autoContinue": True, "command": 178, "doJumpId": 2, "frame": 2, "type": "SimpleItem",
                      "params": [1, speed, -1, 0, 0, 0, 0]})
        for k, (lat, lon, alt) in enumerate(geo[1:], start=3):
            items.append({"autoContinue": True, "command": 16, "doJumpId": k, "frame": 3, "type": "SimpleItem",
                          "params": [0, 0, 0, None, float(lat), float(lon), max(float(alt), 3.0)]})
        items.append({"autoContinue": True, "command": 21, "doJumpId": len(items) + 1, "frame": 3,
                      "type": "SimpleItem", "params": [0, 0, 0, None, float(geo[-1, 0]), float(geo[-1, 1]), 0]})
        plan = {"fileType": "Plan", "groundStation": "QGroundControl", "version": 1,
                "geoFence": {"circles": [], "polygons": [], "version": 2},
                "rallyPoints": {"points": [], "version": 2},
                "mission": {"cruiseSpeed": speed, "hoverSpeed": speed, "firmwareType": 12, "vehicleType": 2,
                            "globalPlanAltitudeMode": 1, "version": 2, "items": items,
                            "plannedHomePosition": [home_lat, home_lon, 0]}}
        fn = os.path.join(out_dir, f"drone_{i:02d}.plan")
        with open(fn, 'w') as f:
            json.dump(plan, f, indent=1)
        files.append(fn)
    return files


def to_waypoint_json(swarm, path, min_dist=2.0):
    traj = np.array(swarm.traj)
    dt = swarm.cfg.dt
    out = {"dt": dt, "frame": "ENU", "units": "m", "drones": []}
    for i in range(traj.shape[1]):
        out["drones"].append({"id": i, "waypoints": _downsample(traj[:, i], min_dist).round(3).tolist()})
    with open(path, 'w') as f:
        json.dump(out, f)
    return out
