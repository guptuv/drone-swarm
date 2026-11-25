"""Fly a swarm of PX4 drones (SITL or real) in OFFBOARD velocity mode with MAVSDK.

Each drone gets its own MAVSDK System on its own UDP port (PX4 SITL multi-vehicle
default: 14540, 14541, ...).  The drone_swarm controller computes velocity setpoints
(NED frame) from the drones' telemetry at `RATE_HZ`.

Requirements: pip install mavsdk ; PX4 SITL multi-vehicle, e.g.
    Tools/simulation/gazebo-classic/sitl_multiple_run.sh -n 5
Run:
    python mavsdk_swarm_offboard.py --n 5 --shape circle --alt 10

Note: this is an integration template.  It is not exercised by the test-suite of this
repository (no PX4 in CI).  Always test in SITL before flying real hardware.
"""
import argparse
import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError

from drone_swarm import SwarmConfig, DroneSwarm

RATE_HZ = 10.0


async def connect(i, base_port):
    drone = System(port=50051 + i)
    await drone.connect(system_address=f"udp://:{base_port + i}")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"drone {i} connected")
            return drone


async def track_position(i, drone, swarm):
    async for pv in drone.telemetry.position_velocity_ned():
        p, v = pv.position, pv.velocity
        swarm.pos[i] = [p.east_m, p.north_m, -p.down_m]          # NED -> ENU
        swarm.vel[i] = [v.east_m_s, v.north_m_s, -v.down_m_s]


async def main(args):
    drones = await asyncio.gather(*[connect(i, args.base_port) for i in range(args.n)])
    swarm = DroneSwarm(SwarmConfig(n_drones=args.n, dt=1 / RATE_HZ, max_speed=args.speed,
                                   world=(400.0, 400.0, 60.0)), start_center=(0, 0, 0), spread=0)
    for i, d in enumerate(drones):
        asyncio.ensure_future(track_position(i, d, swarm))

    for d in drones:
        await d.action.arm()
        await d.action.set_takeoff_altitude(args.alt)
        await d.action.takeoff()
    await asyncio.sleep(10)

    for d in drones:
        await d.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        try:
            await d.offboard.start()
        except OffboardError as e:
            print("offboard start failed:", e)
            return

    center = swarm.pos.mean(0); center[2] = args.alt
    swarm.set_formation(args.shape, center=center + np.array([args.offset, 0, 0]))
    for _ in range(int(args.duration * RATE_HZ)):
        a = swarm.desired_acceleration()
        v = swarm.vel + a * swarm.cfg.dt
        for i, d in enumerate(drones):
            e, n_, u = v[i]
            await d.offboard.set_velocity_ned(VelocityNedYaw(float(n_), float(e), float(-u), 0.0))
        await asyncio.sleep(1 / RATE_HZ)

    for d in drones:
        await d.offboard.stop()
        await d.action.land()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=14540)
    ap.add_argument("--shape", default="circle")
    ap.add_argument("--alt", type=float, default=10.0)
    ap.add_argument("--offset", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--duration", type=float, default=60.0)
    asyncio.run(main(ap.parse_args()))
