"""ROS 2 node that runs the drone_swarm controller on real or simulated (Gazebo) drones.

For every drone i it
  subscribes  /drone{i}/odom      (nav_msgs/Odometry)      -> position + velocity
  publishes   /drone{i}/cmd_vel   (geometry_msgs/Twist)    <- velocity setpoint
and runs the same Boids / formation / avoidance logic as the simulator at `rate_hz`.

Requirements: ROS 2 (Humble or newer), rclpy, this repository on PYTHONPATH.
Run:
    ros2 run <your_pkg> swarm_controller_node --ros-args -p n_drones:=5 -p mode:=formation -p shape:=circle

Note: this is an integration template.  It is not exercised by the test-suite of this
repository (no ROS installation in CI); check topic names / frames for your setup.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from drone_swarm import SwarmConfig, DroneSwarm


class SwarmControllerNode(Node):
    def __init__(self):
        super().__init__('swarm_controller')
        self.declare_parameter('n_drones', 5)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('mode', 'flock')            # flock | formation
        self.declare_parameter('shape', 'circle')
        self.declare_parameter('goal', [20.0, 20.0, 10.0])
        n = self.get_parameter('n_drones').value
        rate = self.get_parameter('rate_hz').value

        self.swarm = DroneSwarm(SwarmConfig(n_drones=n, dt=1.0 / rate, world=(200.0, 200.0, 60.0)),
                                start_center=(0, 0, 0), spread=0.0)
        self.have_odom = np.zeros(n, bool)
        self.pubs = [self.create_publisher(Twist, f'/drone{i}/cmd_vel', 10) for i in range(n)]
        for i in range(n):
            self.create_subscription(Odometry, f'/drone{i}/odom', lambda msg, i=i: self.on_odom(i, msg), 10)

        if self.get_parameter('mode').value == 'formation':
            self.formation_pending = True
        else:
            self.formation_pending = False
            self.swarm.goal = np.array(self.get_parameter('goal').value, float)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'swarm controller running for {n} drones at {rate} Hz')

    def on_odom(self, i, msg):
        p, v = msg.pose.pose.position, msg.twist.twist.linear
        self.swarm.pos[i] = [p.x, p.y, p.z]
        self.swarm.vel[i] = [v.x, v.y, v.z]
        self.have_odom[i] = True

    def tick(self):
        if not self.have_odom.all():
            return
        if self.formation_pending:
            self.swarm.set_formation(self.get_parameter('shape').value,
                                     center=np.array(self.get_parameter('goal').value, float))
            self.formation_pending = False
        a = self.swarm.desired_acceleration()
        v_cmd = self.swarm.vel + a * self.swarm.cfg.dt          # velocity setpoint for the autopilot
        for i, pub in enumerate(self.pubs):
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.linear.z = map(float, v_cmd[i])
            pub.publish(msg)


def main():
    rclpy.init()
    node = SwarmControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
