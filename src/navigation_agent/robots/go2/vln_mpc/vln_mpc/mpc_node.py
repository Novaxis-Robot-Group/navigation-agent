"""Convert body-frame VLN paths and track them with MPC."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from .geometry import (
    TimedPose,
    pose_at_stamp,
    project_body_to_odom,
    project_local_to_odom,
    project_odom_to_local,
)
from .mpc import MPCController, build_pose_aligned_reference

OBJNAV_MODE = "objnav"
TRACK_MODE = "track"
VALID_TASK_MODES = {OBJNAV_MODE, TRACK_MODE}
CONTROL_RATE_HZ = 10.0
HORIZON = 5
MPC_DT_S = 0.1
WAYPOINT_DT_S = 0.1
MPC_CONFIG_NAMES = (
    "track_v_max",
    "objnav_v_max",
    "w_max",
    "a_max_v",
    "a_max_w",
    "q_x",
    "q_y",
    "q_yaw",
    "r_v",
    "r_w",
    "v_output_scale",
    "w_output_scale",
)
POSITIVE_MPC_CONFIG_NAMES = {
    "track_v_max",
    "objnav_v_max",
    "w_max",
    "a_max_v",
    "a_max_w",
}
NONNEGATIVE_MPC_CONFIG_NAMES = set(MPC_CONFIG_NAMES) - POSITIVE_MPC_CONFIG_NAMES
CONTROLLER_CONFIG_NAMES = {
    "w_max",
    "a_max_v",
    "a_max_w",
    "q_x",
    "q_y",
    "q_yaw",
    "r_v",
    "r_w",
}


@dataclass(frozen=True)
class OdomSnapshot:
    stamp_ns: int
    received_s: float
    pose: np.ndarray


@dataclass(frozen=True)
class PathSnapshot:
    episode: int
    sequence: int
    stamp_ns: int
    received_s: float
    trajectory: np.ndarray
    odom_match_ms: float


@dataclass(frozen=True)
class SolveResult:
    generation: int
    command: tuple[float, float]
    reference: np.ndarray
    prediction: np.ndarray
    solve_ms: float


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def parse_vln_response(
    raw: str,
    expected_frame: str,
) -> tuple[int, int, int, bool | None, list[tuple[float, float, float]]]:
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid VLN response JSON: {exc}") from exc
    if not isinstance(response, dict):
        raise ValueError("VLN response must be a JSON object")

    episode = response.get("episode")
    sequence = response.get("seq")
    stamp_ns = response.get("capture_stamp_ns")
    for name, value in (
        ("episode", episode),
        ("seq", sequence),
        ("capture_stamp_ns", stamp_ns),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"VLN response {name} must be a positive integer")
    if response.get("frame_id") != expected_frame:
        raise ValueError(
            f"VLN response frame is {response.get('frame_id')!r}, "
            f"expected {expected_frame!r}"
        )
    stop = response.get("stop")
    for name in ("stop", "visible"):
        value = response.get(name)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"VLN response {name} must be boolean or null")

    raw_waypoints = response.get("waypoints")
    if not isinstance(raw_waypoints, list):
        raise ValueError("VLN response waypoints must be an array")
    waypoints = []
    for point in raw_waypoints:
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError("each VLN waypoint must be [x, y, yaw]")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in point
        ):
            raise ValueError("VLN waypoint values must be numbers")
        values = tuple(float(value) for value in point)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("VLN waypoint contains non-finite values")
        waypoints.append(values)
    return episode, sequence, stamp_ns, stop, waypoints


def should_complete_task(mode: str, stop: bool | None) -> bool:
    return mode == OBJNAV_MODE and stop is True


def validate_mpc_config(config: dict[str, float]) -> str:
    if set(config) != set(MPC_CONFIG_NAMES):
        return "MPC configuration fields are incomplete"
    if not all(math.isfinite(value) for value in config.values()):
        return "MPC configuration values must be finite"
    if any(config[name] <= 0.0 for name in POSITIVE_MPC_CONFIG_NAMES):
        return "MPC limits must be positive"
    if any(config[name] < 0.0 for name in NONNEGATIVE_MPC_CONFIG_NAMES):
        return "MPC weights and output scales must be nonnegative"
    if not any(config[name] > 0.0 for name in ("q_x", "q_y", "q_yaw")):
        return "at least one MPC state weight must be positive"
    return ""


def scale_command(
    command: tuple[float, float],
    v_scale: float,
    w_scale: float,
) -> tuple[float, float]:
    return command[0] * v_scale, command[1] * w_scale


class MpcNode(Node):
    def __init__(self) -> None:
        super().__init__("vln_mpc")
        self.declare_parameter("enabled", False)
        self.declare_parameter("response_topic", "vln/response")
        self.declare_parameter("status_topic", "vln/status")
        self.declare_parameter("mode_topic", "vln/mode")
        self.declare_parameter("enable_topic", "mpc/enable")
        self.declare_parameter("mpc_status_topic", "mpc/status")
        self.declare_parameter("odom_path_topic", "vln/path_odom")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("command_topic", "mpc/cmd_vel")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("track_v_max", 1.5)
        self.declare_parameter("objnav_v_max", 0.8)
        self.declare_parameter("w_max", 3.0)
        self.declare_parameter("a_max_v", 2.0)
        self.declare_parameter("a_max_w", 5.0)
        self.declare_parameter("q_x", 10.0)
        self.declare_parameter("q_y", 10.0)
        self.declare_parameter("q_yaw", 1.0)
        self.declare_parameter("r_v", 0.1)
        self.declare_parameter("r_w", 0.1)
        self.declare_parameter("v_output_scale", 1.0)
        self.declare_parameter("w_output_scale", 1.0)
        self.declare_parameter("odom_match_max_gap_s", 0.3)
        self.declare_parameter("odom_timeout_s", 0.5)
        self.declare_parameter("path_timeout_s", 3.0)
        self.declare_parameter("capture_timeout_s", 5.0)
        self.declare_parameter("metrics_log_period_s", 2.0)

        self.response_topic = str(self.get_parameter("response_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.mode_topic = str(self.get_parameter("mode_topic").value)
        self.enable_topic = str(self.get_parameter("enable_topic").value)
        self.mpc_status_topic = str(self.get_parameter("mpc_status_topic").value)
        self.odom_path_topic = str(self.get_parameter("odom_path_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.control_rate_hz = CONTROL_RATE_HZ
        self.horizon = HORIZON
        self.mpc_dt_s = MPC_DT_S
        self.waypoint_dt_s = WAYPOINT_DT_S
        self.track_v_max = float(self.get_parameter("track_v_max").value)
        self.objnav_v_max = float(self.get_parameter("objnav_v_max").value)
        self.w_max = float(self.get_parameter("w_max").value)
        self.a_max_v = float(self.get_parameter("a_max_v").value)
        self.a_max_w = float(self.get_parameter("a_max_w").value)
        self.odom_match_max_gap_s = float(
            self.get_parameter("odom_match_max_gap_s").value
        )
        self.odom_timeout_s = float(self.get_parameter("odom_timeout_s").value)
        self.path_timeout_s = float(self.get_parameter('path_timeout_s').value)
        self.capture_timeout_s = float(self.get_parameter('capture_timeout_s').value)
        if not all(math.isfinite(v) and v > 0 for v in (self.path_timeout_s, self.capture_timeout_s)):
            raise ValueError('invalid trajectory timeout')
        self.metrics_log_period_s = float(
            self.get_parameter("metrics_log_period_s").value
        )
        self.q_weights = (
            float(self.get_parameter("q_x").value),
            float(self.get_parameter("q_y").value),
            float(self.get_parameter("q_yaw").value),
        )
        self.r_weights = (
            float(self.get_parameter("r_v").value),
            float(self.get_parameter("r_w").value),
        )
        self.v_output_scale = float(
            self.get_parameter("v_output_scale").value
        )
        self.w_output_scale = float(
            self.get_parameter("w_output_scale").value
        )
        required_names = (
            self.response_topic,
            self.status_topic,
            self.mode_topic,
            self.enable_topic,
            self.mpc_status_topic,
            self.odom_path_topic,
            self.odom_topic,
            self.command_topic,
            self.odom_frame,
            self.base_frame,
        )
        if (
            not all(required_names)
            or self.control_rate_hz <= 0.0
            or self.horizon <= 0
            or min(
                self.mpc_dt_s,
                self.waypoint_dt_s,
                self.track_v_max,
                self.objnav_v_max,
                self.w_max,
                self.a_max_v,
                self.a_max_w,
                self.odom_match_max_gap_s,
                self.odom_timeout_s,
                self.metrics_log_period_s,
            )
            <= 0.0
            or validate_mpc_config(self._mpc_config())
        ):
            raise ValueError("invalid MPC parameter")

        self._enabled = bool(self.get_parameter("enabled").value)
        self._task_mode = TRACK_MODE
        self._goal_reached = False
        self._vln_state = ""
        self._odom_history: deque[TimedPose] = deque(maxlen=1000)
        self._odom: Optional[OdomSnapshot] = None
        self._path: Optional[PathSnapshot] = None
        self._generation = 0
        self._session_started_ns = 0
        self._session_episode = None
        self._session_sequence = 0
        self._previous_command = (0.0, 0.0)
        self._last_solve_ms: Optional[float] = None
        self._last_error = ""
        self._last_status = ""
        self._closed = False
        self._metrics_started_s = time.monotonic()
        self._metrics_solve_count = 0
        self._metrics_solve_sum_ms = 0.0
        self._metrics_solve_max_ms = 0.0

        self._tracking_controller = MPCController(
            horizon=self.horizon,
            dt_s=self.mpc_dt_s,
            w_max=self.w_max,
            a_max_v=self.a_max_v,
            a_max_w=self.a_max_w,
            q_weights=self.q_weights,
            r_weights=self.r_weights,
        )
        self._solver_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vln-mpc"
        )
        self._solve_future: Optional[Future[SolveResult]] = None
        self.add_on_set_parameters_callback(self._on_set_parameters)

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        response_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        enable_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            String, self.response_topic, self._on_vln_response, response_qos
        )
        self.create_subscription(
            String, self.status_topic, self._on_vln_status, latched
        )
        self.create_subscription(String, self.mode_topic, self._on_mode, latched)
        self.create_subscription(Bool, self.enable_topic, self._on_enable, enable_qos)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 50)
        self.command_pub = self.create_publisher(TwistStamped, self.command_topic, 10)
        self.odom_path_pub = self.create_publisher(Path, self.odom_path_topic, latched)
        self.reference_pub = self.create_publisher(Path, "mpc/reference", 10)
        self.prediction_pub = self.create_publisher(Path, "mpc/prediction", 10)
        self.mpc_status_pub = self.create_publisher(
            String, self.mpc_status_topic, latched
        )
        self.create_timer(1.0 / self.control_rate_hz, self._control_tick)
        self.create_timer(0.01, self._consume_solution)
        self.create_timer(self.metrics_log_period_s, self._log_metrics)
        self._publish_status()
        self.get_logger().info(
            f"MPC ready (enabled={self._enabled}) "
            f"enable={self.enable_topic} status={self.mpc_status_topic} "
            f"response={self.response_topic} cmd={self.command_topic} "
            f"task_mode={self._task_mode} "
            f"v_max(track/objnav)="
            f"{self.track_v_max:.2f}/{self.objnav_v_max:.2f}m/s "
            f"output_scale(v/w)="
            f"{self.v_output_scale:.2f}/{self.w_output_scale:.2f}"
        )

    def _mpc_config(self) -> dict[str, float]:
        return {
            "track_v_max": self.track_v_max,
            "objnav_v_max": self.objnav_v_max,
            "w_max": self.w_max,
            "a_max_v": self.a_max_v,
            "a_max_w": self.a_max_w,
            "q_x": self.q_weights[0],
            "q_y": self.q_weights[1],
            "q_yaw": self.q_weights[2],
            "r_v": self.r_weights[0],
            "r_w": self.r_weights[1],
            "v_output_scale": self.v_output_scale,
            "w_output_scale": self.w_output_scale,
        }

    def _on_set_parameters(self, parameters) -> SetParametersResult:
        proposed = self._mpc_config()
        changed = set()
        for parameter in parameters:
            if parameter.name not in proposed:
                continue
            if isinstance(parameter.value, bool) or not isinstance(
                parameter.value, (int, float)
            ):
                return SetParametersResult(
                    successful=False,
                    reason=f"{parameter.name} must be numeric",
                )
            value = float(parameter.value)
            if value != proposed[parameter.name]:
                proposed[parameter.name] = value
                changed.add(parameter.name)
        if not changed:
            return SetParametersResult(successful=True)

        error = validate_mpc_config(proposed)
        if error:
            return SetParametersResult(successful=False, reason=error)
        if self._enabled or self._solve_future is not None:
            return SetParametersResult(
                successful=False,
                reason="disable MPC before updating its parameters",
            )

        controller = self._tracking_controller
        if changed & CONTROLLER_CONFIG_NAMES:
            try:
                controller = MPCController(
                    horizon=self.horizon,
                    dt_s=self.mpc_dt_s,
                    w_max=proposed["w_max"],
                    a_max_v=proposed["a_max_v"],
                    a_max_w=proposed["a_max_w"],
                    q_weights=(
                        proposed["q_x"],
                        proposed["q_y"],
                        proposed["q_yaw"],
                    ),
                    r_weights=(proposed["r_v"], proposed["r_w"]),
                )
            except Exception as exc:
                return SetParametersResult(successful=False, reason=str(exc))

        self.track_v_max = proposed["track_v_max"]
        self.objnav_v_max = proposed["objnav_v_max"]
        self.w_max = proposed["w_max"]
        self.a_max_v = proposed["a_max_v"]
        self.a_max_w = proposed["a_max_w"]
        self.q_weights = (
            proposed["q_x"],
            proposed["q_y"],
            proposed["q_yaw"],
        )
        self.r_weights = (proposed["r_v"], proposed["r_w"])
        self.v_output_scale = proposed["v_output_scale"]
        self.w_output_scale = proposed["w_output_scale"]
        self._tracking_controller = controller
        self.get_logger().info(
            "MPC parameters updated: "
            + " ".join(
                f"{name}={proposed[name]:g}" for name in sorted(changed)
            )
        )
        return SetParametersResult(successful=True)

    def _on_odom(self, message: Odometry) -> None:
        if (
            message.header.frame_id != self.odom_frame
            or message.child_frame_id != self.base_frame
        ):
            self._last_error = (
                f"odom frames are {message.header.frame_id!r} -> "
                f"{message.child_frame_id!r}, expected {self.odom_frame!r} -> "
                f"{self.base_frame!r}"
            )
            return
        if self._last_error.startswith("odom frames are"):
            self._last_error = ""
        pose = message.pose.pose
        values = np.asarray(
            [
                float(pose.position.x),
                float(pose.position.y),
                yaw_from_quaternion(
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ),
            ],
            dtype=np.float64,
        )
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        if stamp_ns <= 0 or not np.all(np.isfinite(values)):
            return
        sample = TimedPose(
            stamp_ns=stamp_ns,
            x=float(values[0]),
            y=float(values[1]),
            yaw=float(values[2]),
        )
        if self._odom_history and stamp_ns < self._odom_history[-1].stamp_ns:
            self._odom_history.clear()
        if self._odom_history and stamp_ns == self._odom_history[-1].stamp_ns:
            self._odom_history[-1] = sample
        else:
            self._odom_history.append(sample)
        self._odom = OdomSnapshot(
            stamp_ns=stamp_ns,
            received_s=time.monotonic(),
            pose=values,
        )

    def _on_vln_response(self, message: String) -> None:
        if not self._enabled or self._vln_state != "RUNNING":
            return
        try:
            episode, sequence, stamp_ns, stop, body_waypoints = parse_vln_response(
                message.data, self.base_frame
            )
        except ValueError as exc:
            self._last_error = str(exc)
            return
        age_s = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        if (self._goal_reached or stamp_ns < self._session_started_ns
            or not -.05 <= age_s <= self.capture_timeout_s
            or (self._session_episode is not None and episode != self._session_episode)
            or sequence <= self._session_sequence):
            return
        self._session_episode, self._session_sequence = episode, sequence
        if self._path is not None:
            if stamp_ns < self._path.stamp_ns:
                return
            if stamp_ns == self._path.stamp_ns and sequence <= self._path.sequence:
                return
        if should_complete_task(self._task_mode, stop):
            self._complete_task(episode, sequence)
            return

        matched = pose_at_stamp(
            list(self._odom_history), stamp_ns, self.odom_match_max_gap_s
        )
        if matched is None:
            self._last_error = (
                "no odom pose close enough to the image capture timestamp"
            )
            return
        capture_pose, gap_s = matched
        raw_odom_waypoints = project_body_to_odom(body_waypoints, capture_pose)
        trajectory = np.asarray(raw_odom_waypoints, dtype=np.float64).reshape((-1, 3))
        self._path = PathSnapshot(
            episode=episode,
            sequence=sequence,
            stamp_ns=stamp_ns,
            received_s=time.monotonic(),
            trajectory=trajectory,
            odom_match_ms=gap_s * 1000.0,
        )
        self._generation += 1
        self._last_error = ""
        self.odom_path_pub.publish(
            self._make_path(
                trajectory,
                start_stamp_ns=stamp_ns,
                include_now=False,
                time_step_s=self.waypoint_dt_s,
            )
        )

    def _on_mode(self, message: String) -> None:
        mode = message.data.strip().lower()
        if mode not in VALID_TASK_MODES:
            self._last_error = f"invalid VLN mode {mode!r}"
            return
        if self._last_error.startswith("invalid VLN mode"):
            self._last_error = ""
        changed = mode != self._task_mode
        self._task_mode = mode
        self._goal_reached = False
        if changed:
            self._generation += 1
            self._previous_command = (0.0, 0.0)
            self.get_logger().info(f"VLN task mode: {mode}")

    def _complete_task(self, episode: int, sequence: int) -> None:
        if self._goal_reached:
            return
        self._goal_reached = True
        self._enabled = False
        self._last_error = ""
        self._clear_trajectory()
        self._publish_status()
        self.get_logger().info(f"ObjNav goal reached: episode={episode} seq={sequence}")

    def _on_vln_status(self, message: String) -> None:
        state = message.data.strip()
        if state not in {"IDLE", "RUNNING", "ERROR", "PAUSED"}:
            self._last_error = f"invalid VLN status {state!r}"
            return
        if self._last_error.startswith("invalid VLN status"):
            self._last_error = ""
        if state == self._vln_state:
            return
        self._vln_state = state
        if state != "RUNNING":
            self._clear_trajectory()

    def _clear_trajectory(self) -> None:
        self._path = None
        self._generation += 1
        self._previous_command = (0.0, 0.0)
        self.odom_path_pub.publish(
            self._make_path(
                np.empty((0, 3), dtype=np.float64),
                start_stamp_ns=self.get_clock().now().nanoseconds,
                include_now=False,
                time_step_s=self.waypoint_dt_s,
            )
        )

    def _on_enable(self, message: Bool) -> None:
        self._enabled = bool(message.data)
        self._session_started_ns = self.get_clock().now().nanoseconds if self._enabled else 0
        self._session_episode = None
        self._session_sequence = 0
        if self._enabled:
            self._goal_reached = False
        # 关闭和重启均清除旧轨迹，不依赖其他 ROS 话题的到达顺序。
        self._clear_trajectory()
        self._last_error = ""
        self._publish_status()

    def _block_reason(self) -> str:
        if not self._enabled:
            return "disabled"
        if not self._vln_state:
            return "waiting for VLN status"
        if self._vln_state != "RUNNING":
            return f"VLN state is {self._vln_state}"
        now_s = time.monotonic()
        if self._odom is None:
            return "waiting for odom"
        if now_s - self._odom.received_s > self.odom_timeout_s:
            return "odom is stale"
        if self._path is None:
            return "waiting for VLN response"
        if now_s - self._path.received_s > self.path_timeout_s:
            return "VLN trajectory is stale"
        capture_age = (self.get_clock().now().nanoseconds - self._path.stamp_ns) / 1e9
        if capture_age < -0.05 or capture_age > self.capture_timeout_s:
            return "VLN capture timestamp is stale or invalid"
        if len(self._path.trajectory) == 0:
            return "VLN response has no waypoints"
        return ""

    def _control_tick(self) -> None:
        reason = self._block_reason()
        self._publish_status()
        if reason:
            self._previous_command = (0.0, 0.0)
            return
        if self._solve_future is not None:
            return
        assert self._odom is not None and self._path is not None
        try:
            reference = build_pose_aligned_reference(
                self._path.trajectory,
                self._odom.pose,
                horizon=self.horizon,
                weights=self.q_weights,
            )
        except ValueError as exc:
            self._last_error = str(exc)
            return
        self._solve_future = self._solver_pool.submit(
            self._solve,
            self._generation,
            self._odom.pose.copy(),
            reference,
            self._previous_command,
            self._task_v_max(),
        )

    def _solve(
        self,
        generation: int,
        pose: np.ndarray,
        reference: np.ndarray,
        previous: tuple[float, float],
        v_max: float,
    ) -> SolveResult:
        started_s = time.perf_counter()
        origin = TimedPose(
            stamp_ns=0,
            x=float(pose[0]),
            y=float(pose[1]),
            yaw=float(pose[2]),
        )
        local_reference = np.asarray(
            project_odom_to_local(reference, origin), dtype=np.float64
        )
        local_pose = np.zeros(3, dtype=np.float64)
        command, local_prediction = self._tracking_controller.solve(
            local_pose,
            local_reference,
            previous,
            v_max,
        )
        prediction = np.asarray(
            project_local_to_odom(local_prediction, origin),
            dtype=np.float64,
        )
        return SolveResult(
            generation=generation,
            command=command,
            reference=reference,
            prediction=prediction,
            solve_ms=(time.perf_counter() - started_s) * 1000.0,
        )

    def _consume_solution(self) -> None:
        future = self._solve_future
        if future is None or not future.done():
            return
        self._solve_future = None
        try:
            result = future.result()
        except Exception as exc:
            self._last_error = f"MPC solve failed: {exc}"
            self._previous_command = (0.0, 0.0)
            return
        self._last_solve_ms = result.solve_ms
        self._metrics_solve_count += 1
        self._metrics_solve_sum_ms += result.solve_ms
        self._metrics_solve_max_ms = max(self._metrics_solve_max_ms, result.solve_ms)
        if result.generation != self._generation or self._block_reason():
            return
        self._previous_command = result.command
        self._last_error = ""
        self._publish_command(
            scale_command(
                result.command,
                self.v_output_scale,
                self.w_output_scale,
            )
        )
        now_ns = self.get_clock().now().nanoseconds
        self.reference_pub.publish(
            self._make_path(
                result.reference,
                start_stamp_ns=now_ns,
                include_now=True,
                time_step_s=self.mpc_dt_s,
            )
        )
        self.prediction_pub.publish(
            self._make_path(
                result.prediction,
                start_stamp_ns=now_ns,
                include_now=True,
                time_step_s=self.mpc_dt_s,
            )
        )

    def _publish_command(self, command: tuple[float, float]) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame
        message.twist.linear.x = float(command[0])
        message.twist.angular.z = float(command[1])
        self.command_pub.publish(message)

    def _make_path(
        self,
        trajectory: np.ndarray,
        *,
        start_stamp_ns: int,
        include_now: bool,
        time_step_s: float,
    ) -> Path:
        path = Path()
        path.header.stamp.sec = start_stamp_ns // 1_000_000_000
        path.header.stamp.nanosec = start_stamp_ns % 1_000_000_000
        path.header.frame_id = self.odom_frame
        for index, values in enumerate(np.asarray(trajectory, dtype=np.float64)):
            pose = PoseStamped()
            offset_index = index if include_now else index + 1
            stamp_ns = start_stamp_ns + int(round(offset_index * time_step_s * 1e9))
            pose.header.stamp.sec = stamp_ns // 1_000_000_000
            pose.header.stamp.nanosec = stamp_ns % 1_000_000_000
            pose.header.frame_id = self.odom_frame
            pose.pose.position.x = float(values[0])
            pose.pose.position.y = float(values[1])
            yaw = float(values[2])
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        return path

    def _status(self) -> str:
        if not self._enabled:
            return "IDLE"
        if self._last_error:
            return "ERROR"
        return "RUNNING"

    def _publish_status(self) -> None:
        if not rclpy.ok():
            return
        status = self._status()
        if status == self._last_status:
            return
        self._last_status = status
        self.mpc_status_pub.publish(String(data=status))

    def _log_metrics(self) -> None:
        now_s = time.monotonic()
        elapsed_s = max(now_s - self._metrics_started_s, 1e-9)
        count = self._metrics_solve_count
        if self._enabled or count:
            average_ms = self._metrics_solve_sum_ms / count if count else 0.0
            latest_text = (
                f"{self._last_solve_ms:.2f}"
                if self._last_solve_ms is not None
                else "n/a"
            )
            self.get_logger().info(
                "MPC metrics: "
                f"solve_hz={count / elapsed_s:.2f} "
                f"solve_ms(latest/avg/max)={latest_text}/"
                f"{average_ms:.2f}/{self._metrics_solve_max_ms:.2f} "
                f"task_mode={self._task_mode} "
                f"v_max={self._task_v_max():.2f} "
                f"reason={self._last_error or self._block_reason() or 'tracking'}"
            )
        self._metrics_started_s = now_s
        self._metrics_solve_count = 0
        self._metrics_solve_sum_ms = 0.0
        self._metrics_solve_max_ms = 0.0

    def _task_v_max(self) -> float:
        return self.objnav_v_max if self._task_mode == OBJNAV_MODE else self.track_v_max

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._enabled = False
        self._generation += 1
        if rclpy.ok():
            self._publish_status()
        self._solver_pool.shutdown(wait=True, cancel_futures=True)

    def destroy_node(self):
        self.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MpcNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
