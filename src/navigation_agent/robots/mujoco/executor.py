"""将统一短轨迹交给已有 MuJoCo 执行器，保持坐标转换和避障行为。"""

from navigation_agent.contracts.robot import CameraFrame
from vln_mujoco.mpc import (
    A_MAX_V, A_MAX_W, CONTROL_RATE_HZ, HORIZON, MPC_DT_S,
    OBJNAV_V_MAX, ODOM_MATCH_MAX_GAP_S, ODOM_TIMEOUT_S,
    Q_WEIGHTS, R_WEIGHTS, TRACK_V_MAX, W_MAX, WAYPOINT_DT_S,
)


class MujocoExecutor:
    control_period_s = 1.0 / CONTROL_RATE_HZ

    def __init__(self, simulation, tracker):
        self.simulation = simulation
        self.tracker = tracker

    def frame(self):
        frame = self.simulation.frame()
        if frame is None:
            return None
        return CameraFrame(stamp_ns=frame.stamp_ns, jpeg=frame.jpeg, pose=frame.pose)

    def accept_path(self, waypoints, capture_pose):
        self.tracker.set_body_path(waypoints, capture_pose)
        return self.tracker.command

    def poll(self):
        return self.tracker.poll()

    def submit(self):
        pose = self.simulation.snapshot()['pose']
        self.tracker.submit((pose['x'], pose['y'], pose['yaw']))

    def drive(self, command):
        self.simulation.set_velocity(*command)

    def reset(self):
        self.tracker.reset()

    def fail(self, message):
        self.tracker.error = message

    def close(self):
        self.tracker.close()

    def diagnostics(self):
        return {
            'error': self.tracker.error,
            'solve_ms': self.tracker.solve_ms,
            'parameters': {
                'control_rate_hz': CONTROL_RATE_HZ,
                'horizon': HORIZON,
                'mpc_dt_s': MPC_DT_S,
                'waypoint_dt_s': WAYPOINT_DT_S,
                'track_v_max': TRACK_V_MAX,
                'objnav_v_max': OBJNAV_V_MAX,
                'w_max': W_MAX,
                'a_max_v': A_MAX_V,
                'a_max_w': A_MAX_W,
                'q_x': Q_WEIGHTS[0], 'q_y': Q_WEIGHTS[1], 'q_yaw': Q_WEIGHTS[2],
                'r_v': R_WEIGHTS[0], 'r_w': R_WEIGHTS[1],
                'odom_match_max_gap_s': ODOM_MATCH_MAX_GAP_S,
                'odom_timeout_s': ODOM_TIMEOUT_S,
            },
        }
