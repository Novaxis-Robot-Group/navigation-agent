"""ROS 2 node backing the VLN control page."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter, parameter_value_to_python
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

from .web_server import (
    MANUAL_LIMIT_NAMES,
    MPC_CONFIG_NAMES,
    VALID_ROBOT_ACTIONS,
    Command,
    WebServer,
)
from .wifi import WifiManager


def _boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _diagnostic_level(value: Any) -> int:
    if isinstance(value, (bytes, bytearray, memoryview)):
        if len(value) != 1:
            raise ValueError("diagnostic level must contain exactly one byte")
        return int(value[0])
    return int(value)


def _motor_temperatures(value: Any) -> Optional[list[int]]:
    """Decode the adapter's JSON array of per-motor temperatures."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in parsed
    ):
        return None
    return parsed


def _optional_int(value: Any) -> Optional[int]:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


class VlnWebNode(Node):
    def __init__(self) -> None:
        super().__init__("vln_web")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8088)
        self.declare_parameter("image_topic", "camera/color/image_raw")
        self.declare_parameter("image_transport", "raw")
        self.declare_parameter("response_topic", "vln/response")
        self.declare_parameter("status_topic", "vln/status")
        self.declare_parameter("server_url_topic", "vln/server_url")
        self.declare_parameter(
            "server_url_status_topic", "vln/server_url_status"
        )
        self.declare_parameter("mpc_status_topic", "mpc/status")
        self.declare_parameter("diagnostics_topic", "diagnostics")
        self.declare_parameter("instruction_topic", "vln/instruction")
        self.declare_parameter("mode_topic", "vln/mode")
        self.declare_parameter("mpc_enable_topic", "mpc/enable")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("manual_command_topic", "web/cmd_vel")
        self.declare_parameter("sent_command_topic", "cmd_vel")
        self.declare_parameter("control_source_topic", "control/source")
        self.declare_parameter("manual_control_service", "control/set_manual")
        self.declare_parameter("auto_control_service", "control/set_auto")
        self.declare_parameter("stop_service", "control/stop")
        self.declare_parameter("stand_service", "robot/stand")
        self.declare_parameter("walk_service", "robot/walk")
        self.declare_parameter("sit_service", "robot/sit")
        self.declare_parameter(
            "toggle_policy_service", "robot/toggle_policy"
        )
        self.declare_parameter("image_width", 480)
        self.declare_parameter("image_height", 270)
        self.declare_parameter("image_fps", 10.0)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("manual_linear_limit", 1.5)
        self.declare_parameter("manual_angular_limit", 3.0)
        self.declare_parameter("manual_linear_accel", 1.0)
        self.declare_parameter("manual_angular_accel", 2.0)
        self.declare_parameter("wifi_interface", "")
        self.declare_parameter("agent_lease_s", 3.0)
        self.declare_parameter("observation_timeout_s", 1.0)
        self.declare_parameter("observation_match_s", 0.15)

        self.host = str(self.get_parameter("host").value).strip()
        self.port = int(self.get_parameter("port").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.image_transport = str(
            self.get_parameter("image_transport").value
        ).strip().lower()
        self.response_topic = str(self.get_parameter("response_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.server_url_topic = str(
            self.get_parameter("server_url_topic").value
        )
        self.server_url_status_topic = str(
            self.get_parameter("server_url_status_topic").value
        )
        self.mpc_status_topic = str(
            self.get_parameter("mpc_status_topic").value
        )
        self.diagnostics_topic = str(
            self.get_parameter("diagnostics_topic").value
        )
        self.instruction_topic = str(
            self.get_parameter("instruction_topic").value
        )
        self.mode_topic = str(self.get_parameter("mode_topic").value)
        self.mpc_enable_topic = str(
            self.get_parameter("mpc_enable_topic").value
        )
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.manual_command_topic = str(
            self.get_parameter("manual_command_topic").value
        )
        self.sent_command_topic = str(
            self.get_parameter("sent_command_topic").value
        )
        self.control_source_topic = str(
            self.get_parameter("control_source_topic").value
        )
        self.manual_control_service = str(
            self.get_parameter("manual_control_service").value
        )
        self.auto_control_service = str(
            self.get_parameter("auto_control_service").value
        )
        self.stop_service = str(self.get_parameter("stop_service").value)
        self.mode_services = {
            action: str(self.get_parameter(f"{action}_service").value)
            for action in VALID_ROBOT_ACTIONS
        }
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.image_fps = float(self.get_parameter("image_fps").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.manual_linear_limit = float(
            self.get_parameter("manual_linear_limit").value
        )
        self.manual_angular_limit = float(
            self.get_parameter("manual_angular_limit").value
        )
        self.manual_linear_accel = float(
            self.get_parameter("manual_linear_accel").value
        )
        self.manual_angular_accel = float(
            self.get_parameter("manual_angular_accel").value
        )
        self.wifi_interface = str(
            self.get_parameter("wifi_interface").value
        ).strip()
        required_names = (
            self.image_topic,
            self.response_topic,
            self.status_topic,
            self.server_url_topic,
            self.server_url_status_topic,
            self.mpc_status_topic,
            self.diagnostics_topic,
            self.instruction_topic,
            self.mode_topic,
            self.mpc_enable_topic,
            self.odom_topic,
            self.manual_command_topic,
            self.sent_command_topic,
            self.control_source_topic,
            self.manual_control_service,
            self.auto_control_service,
            self.stop_service,
            *self.mode_services.values(),
        )
        if (
            not self.host
            or not 1 <= self.port <= 65535
            or not all(required_names)
            or self.image_transport not in {"raw", "compressed"}
            or self.image_width <= 0
            or self.image_height <= 0
            or self.image_fps <= 0.0
            or not 1 <= self.jpeg_quality <= 100
            or self.manual_linear_limit <= 0.0
            or self.manual_angular_limit <= 0.0
            or self.manual_linear_accel <= 0.0
            or self.manual_angular_accel <= 0.0
        ):
            raise ValueError("invalid VLN web parameter")

        self._commands: queue.Queue[Command] = queue.Queue(maxsize=512)
        web_dir = Path(get_package_share_directory("vln_web")) / "web"
        self.web_server = WebServer(
            host=self.host,
            port=self.port,
            web_dir=web_dir,
            image_topic=self.image_topic,
            manual_linear_limit=self.manual_linear_limit,
            manual_angular_limit=self.manual_angular_limit,
            manual_linear_accel=self.manual_linear_accel,
            manual_angular_accel=self.manual_angular_accel,
            commands=self._commands,
            logger=self.get_logger(),
            agent_lease_s=float(self.get_parameter('agent_lease_s').value),
            observation_timeout_s=float(self.get_parameter('observation_timeout_s').value),
            observation_match_s=float(self.get_parameter('observation_match_s').value),
        )

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        command_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
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
        if self.image_transport == "compressed":
            self.create_subscription(
                CompressedImage,
                self.image_topic,
                self._on_compressed_image,
                sensor_qos,
            )
        else:
            self.create_subscription(
                Image,
                self.image_topic,
                self._on_image,
                sensor_qos,
            )
        self.create_subscription(
            String, self.response_topic, self._on_vln_response, response_qos
        )
        self.create_subscription(
            String, self.status_topic, self._on_vln_status, latched
        )
        self.create_subscription(
            String,
            self.server_url_status_topic,
            self._on_server_url,
            latched,
        )
        self.create_subscription(
            String, self.mpc_status_topic, self._on_mpc_status, latched
        )
        self.create_subscription(
            DiagnosticArray,
            self.diagnostics_topic,
            self._on_diagnostics,
            10,
        )
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 20)
        self.create_subscription(
            TwistStamped,
            self.sent_command_topic,
            self._on_sent_command,
            20,
        )
        self.create_subscription(
            String,
            self.control_source_topic,
            self._on_control_source,
            latched,
        )
        self._history_commands = {}
        self.history_pub = self.create_publisher(String, 'vln/session_control', command_qos)
        self.create_subscription(String, 'vln/session_state', self._on_history_state, latched)
        self.instruction_pub = self.create_publisher(
            String, self.instruction_topic, command_qos
        )
        self.server_url_pub = self.create_publisher(
            String, self.server_url_topic, latched
        )
        self.mode_pub = self.create_publisher(
            String, self.mode_topic, latched
        )
        self.manual_command_pub = self.create_publisher(
            TwistStamped, self.manual_command_topic, 10
        )
        self.mpc_enable_pub = self.create_publisher(
            Bool, self.mpc_enable_topic, enable_qos
        )
        self.manual_control_client = self.create_client(
            SetBool, self.manual_control_service
        )
        self.auto_control_client = self.create_client(
            SetBool, self.auto_control_service
        )
        self.stop_client = self.create_client(Trigger, self.stop_service)
        self.mode_clients = {
            action: self.create_client(Trigger, service)
            for action, service in self.mode_services.items()
        }
        self.mpc_get_parameters_client = self.create_client(
            GetParameters, "/vln_mpc/get_parameters"
        )
        self.mpc_set_parameters_client = self.create_client(
            SetParametersAtomically,
            "/vln_mpc/set_parameters_atomically",
        )
        self.body_set_parameters_client = self.create_client(
            SetParametersAtomically, '/go2_adapter/set_parameters_atomically')

        self._last_encode_s = 0.0
        self._wifi_manager = WifiManager(self.wifi_interface)
        self._wifi_jobs: queue.Queue[tuple[str, Any, str]] = queue.Queue(
            maxsize=1
        )
        self._wifi_busy_lock = threading.Lock()
        self._wifi_busy = False
        self._wifi_stop = threading.Event()
        self._wifi_thread = threading.Thread(
            target=self._wifi_worker,
            name="vln-web-wifi",
            daemon=True,
        )
        self._wifi_thread.start()
        self._queue_wifi_job("scan", None, "")
        self._mpc_request_generation = 0
        self._mpc_desired_enabled = False
        self._mpc_config_generation = 0
        self._mpc_config_future = None
        self._closed = False
        self.create_timer(0.02, self._flush_commands)
        self.create_timer(0.5, self.web_server.broadcast_runtime)
        self.create_timer(1.0, self._refresh_mpc_config)

    def _on_image(self, message: Image) -> None:
        now = time.monotonic()
        if now - self._last_encode_s < 1.0 / self.image_fps:
            return
        encoding = str(message.encoding).lower()
        channels = {
            "mono8": 1,
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
        }.get(encoding)
        if channels is None:
            return
        width = int(message.width)
        height = int(message.height)
        step = int(message.step)
        row_bytes = width * channels
        raw = np.frombuffer(message.data, dtype=np.uint8)
        if width <= 0 or height <= 0 or step < row_bytes or raw.size < height * step:
            return
        pixels = raw[: height * step].reshape(height, step)[:, :row_bytes]
        image = (
            pixels.reshape(height, width)
            if channels == 1
            else pixels.reshape(height, width, channels)
        )
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif encoding == "rgba8":
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif encoding == "bgra8":
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        self._update_image(image, now, message.header)

    def _on_compressed_image(self, message: CompressedImage) -> None:
        now = time.monotonic()
        if (
            now - self._last_encode_s < 1.0 / self.image_fps
            or not message.data
        ):
            return
        image = cv2.imdecode(
            np.frombuffer(message.data, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            return
        self._update_image(image, now, message.header)

    def _update_image(self, image: np.ndarray, now: float, header) -> None:
        if (
            image.shape[1] != self.image_width
            or image.shape[0] != self.image_height
        ):
            image = cv2.resize(
                image,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_AREA,
            )
        success, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not success:
            return
        self._last_encode_s = now
        stamp_ns = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
        self.web_server.update_frame(
            encoded.tobytes(), now, stamp_ns=stamp_ns, frame_id=header.frame_id,
            capture_age_s=(self.get_clock().now().nanoseconds - stamp_ns) / 1e9,
        )

    def _on_history_state(self, message: String) -> None:
        try:
            state = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if not isinstance(state, dict):
            return
        self.web_server.update_history_capability(
            state.get('history_continuation'), state.get('current_generation', 0))
        if state.get('state') in {'IDLE', 'RUNNING', 'ERROR', 'PAUSED'}:
            self.web_server.update_vln_status(state['state'], generation=state.get('current_generation'))
        key = (state.get('generation'), state.get('operation'))
        client_id = self._history_commands.pop(key, None)
        if client_id:
            self.web_server.send_client(client_id, {
                'type': 'command_result', 'command': 'vln_history',
                'generation': state.get('generation'), 'operation': state.get('operation'),
                'ok': state.get('ok') is True, 'history_reused': state.get('history_reused', False),
                'message': state.get('message', 'navigation history updated'),
            })

    def _on_vln_status(self, message: String) -> None:
        state = message.data.strip()
        if state not in {"IDLE", "RUNNING", "ERROR", "PAUSED"}:
            self.get_logger().warning(f"invalid VLN status {state!r}")
            return
        self.web_server.update_vln_status(state)

    def _on_server_url(self, message: String) -> None:
        try:
            self.web_server.update_server_url(message.data)
        except ValueError as exc:
            self.get_logger().warning(f"invalid VLN server URL status: {exc}")

    def _on_mpc_status(self, message: String) -> None:
        state = message.data.strip()
        if state not in {"IDLE", "RUNNING", "ERROR"}:
            self.get_logger().warning(f"invalid MPC status {state!r}")
            return
        self.web_server.update_mpc_status(state)

    def _on_vln_response(self, message: String) -> None:
        if not self.web_server.vln_running():
            return
        try:
            response = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"invalid VLN response JSON: {exc}")
            return
        if not isinstance(response, dict):
            self.get_logger().warning("VLN response must be a JSON object")
            return
        episode = response.get("episode")
        sequence = response.get("seq")
        stamp_ns = response.get("capture_stamp_ns")
        frame_id = response.get("frame_id")
        waypoints = response.get("waypoints")
        stop = response.get("stop")
        visible = response.get("visible")
        apos_state = response.get("apos_state")
        opos_state = response.get("opos_state")
        apos_px = response.get("apos_px")
        opos_px = response.get("opos_px")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in (episode, sequence, stamp_ns)
            )
            or not isinstance(frame_id, str)
            or not frame_id
            or not isinstance(waypoints, list)
            or any(
                value is not None and not isinstance(value, bool)
                for value in (stop, visible)
            )
            or any(
                value is not None and not isinstance(value, str)
                for value in (apos_state, opos_state)
            )
            or any(
                value is not None
                and (
                    not isinstance(value, list)
                    or len(value) != 2
                    or any(
                        isinstance(number, bool)
                        or not isinstance(number, (int, float))
                        or not math.isfinite(number)
                        for number in value
                    )
                )
                for value in (apos_px, opos_px)
            )
        ):
            self.get_logger().warning("invalid VLN response fields")
            return
        body_waypoints = []
        for waypoint in waypoints:
            if (
                not isinstance(waypoint, list)
                or len(waypoint) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in waypoint
                )
            ):
                self.get_logger().warning("invalid VLN response waypoint")
                return
            body_waypoints.append([float(value) for value in waypoint])
        now_ns = self.get_clock().now().nanoseconds
        latency_ms = (now_ns - stamp_ns) / 1_000_000.0
        if not -50 <= latency_ms <= 5000:
            self.get_logger().warning('discarding stale VLN response')
            return
        data = {
            "generation": response.get('generation'),
            "episode": episode,
            "last_sequence": sequence,
            "last_latency_ms": latency_ms if latency_ms >= 0.0 else None,
            "waypoint_count": len(body_waypoints),
            "visible": visible,
            "stop": stop,
            "apos_state": apos_state,
            "opos_state": opos_state,
            "apos_px": (
                [float(value) for value in apos_px]
                if apos_px is not None
                else None
            ),
            "opos_px": (
                [float(value) for value in opos_px]
                if opos_px is not None
                else None
            ),
        }
        path = {
            "frame_id": frame_id,
            "stamp_s": stamp_ns / 1_000_000_000.0,
            "body_waypoints": body_waypoints,
        }
        accepted = self.web_server.update_vln_response(
            data, path, time.monotonic(), capture_stamp_ns=stamp_ns,
        )
        if accepted and stop is True and self.web_server.vln_mode() == "objnav":
            self._complete_objnav()

    def _on_diagnostics(self, message: DiagnosticArray) -> None:
        for status in message.status:
            name = str(status.name).strip("/").split("/")[-1]
            values = {str(item.key): str(item.value) for item in status.values}
            if name == "robot_adapter":
                data = self._parse_robot_diagnostics(status, values)
                self.web_server.update_robot_diagnostics(
                    data,
                    values.get("control_source", ""),
                )

    @staticmethod
    def _parse_robot_diagnostics(
        status, values: dict[str, str]
    ) -> dict[str, Any]:
        return {
            "available": True,
            "level": _diagnostic_level(status.level),
            "message": str(status.message),
            "adapter": values.get("adapter", ""),
            "connected": _boolean(values.get("connected", False)),
            "robot_id": values.get("robot_id", ""),
            "mode": values.get("mode", "UNKNOWN") or "UNKNOWN",
            "battery": _optional_float(values.get("battery")),
            "imu": values.get("imu", "UNKNOWN"),
            "motor": values.get("motor", "UNKNOWN"),
            "motor_temps": _motor_temperatures(values.get("motor_temps")),
            "motor_temp_max": _optional_int(values.get("motor_temp_max")),
            "motor_temps_age_s": _optional_float(values.get("motor_temps_age_s")),
            "policy": values.get("policy", ""),
            "vln_policy": values.get("vln_policy", ""),
            "avoidance_check": values.get("avoidance_check", "not_checked"),
            "safety_error": values.get("safety_error", ""),
            "sdk_error": values.get("sdk_error", ""),
            "linear_limit": _optional_float(values.get("linear_limit")),
            "angular_limit": _optional_float(values.get("angular_limit")),
            "custom_clearance_available": _boolean(values.get("custom_clearance_available", False)),
        }

    def _on_odom(self, message: Odometry) -> None:
        position = message.pose.pose.position
        q = message.pose.pose.orientation
        values = (position.x, position.y, position.z, q.x, q.y, q.z, q.w)
        valid = all(math.isfinite(v) for v in values)
        norm = math.sqrt(sum(v * v for v in (q.x, q.y, q.z, q.w))) if valid else 0
        valid = valid and norm > 1e-6 and bool(message.header.frame_id)
        pose = {}
        if valid:
            x, y, z, w = (v / norm for v in (q.x, q.y, q.z, q.w))
            pose = {'x': position.x, 'y': position.y, 'z': position.z,
                    'yaw': math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))}
        self.web_server.update_odom(
            {
                "stamp_ns": (
                    int(message.header.stamp.sec) * 1_000_000_000
                    + int(message.header.stamp.nanosec)
                ),
                "pose": pose, "pose_valid": valid,
                "frame_id": message.header.frame_id,
                "child_frame_id": message.child_frame_id,
                "twist_linear": [
                    float(message.twist.twist.linear.x),
                    float(message.twist.twist.linear.y),
                    float(message.twist.twist.linear.z),
                ],
                "twist_angular": [
                    float(message.twist.twist.angular.x),
                    float(message.twist.twist.angular.y),
                    float(message.twist.twist.angular.z),
                ],
            }
        )

    def _on_sent_command(self, message: TwistStamped) -> None:
        self.web_server.update_sent_command(
            {
                "linear": float(message.twist.linear.x),
                "angular": float(message.twist.angular.z),
            }
        )

    def _on_control_source(self, message: String) -> None:
        source = str(message.data).strip().lower() or "disabled"
        self.web_server.update_control_source(source)

    def _flush_commands(self) -> None:
        for _ in range(100):
            try:
                kind, payload, client_id = self._commands.get_nowait()
            except queue.Empty:
                return
            if kind == 'agent_release':
                self._request_agent_release(payload, client_id)
            elif kind == 'vln_history':
                self._request_history(payload, client_id)
            elif kind == "set_vln":
                self._request_vln(
                    bool(payload[0]),
                    str(payload[1]),
                    str(payload[2]),
                    client_id,
                )
            elif kind == "set_server_url":
                self._request_server_url(str(payload), client_id)
            elif kind == "set_mpc":
                self._request_mpc(bool(payload), client_id)
            elif kind == "set_mpc_config":
                self._request_mpc_config(payload, client_id)
            elif kind == 'set_body_limits':
                self._request_body_limits(payload, client_id)
            elif kind == "set_manual_limits":
                self._request_manual_limits(payload, client_id)
            elif kind == "twist":
                message = TwistStamped()
                message.header.stamp = self.get_clock().now().to_msg()
                message.header.frame_id = "base_link"
                message.twist.linear.x = float(payload[0])
                message.twist.angular.z = float(payload[1])
                self.manual_command_pub.publish(message)
            elif kind == "manual_control":
                if not self.manual_control_client.service_is_ready():
                    self._service_error(
                        client_id,
                        f"service {self.manual_control_service} is unavailable",
                    )
                    self.web_server.revoke_controller(
                        "robot adapter unavailable"
                    )
                    continue
                request = SetBool.Request()
                request.data = bool(payload)
                future = self.manual_control_client.call_async(request)
                future.add_done_callback(
                    lambda done, cid=client_id, enabled=request.data: (
                        self._manual_control_result(done, cid, enabled)
                    )
                )
            elif kind == "stop":
                if not self.stop_client.service_is_ready():
                    self._service_error(
                        client_id, f"service {self.stop_service} is unavailable"
                    )
                    continue
                future = self.stop_client.call_async(Trigger.Request())
                future.add_done_callback(
                    lambda done, cid=client_id: self._trigger_result(
                        done, cid, "stop"
                    )
                )
            elif kind == "mode":
                action = str(payload)
                client = self.mode_clients[action]
                if not client.service_is_ready():
                    self._service_error(
                        client_id,
                        f"service {self.mode_services[action]} is unavailable",
                    )
                    continue
                future = client.call_async(Trigger.Request())
                future.add_done_callback(
                    lambda done, cid=client_id, name=action: self._trigger_result(
                        done, cid, name
                    )
                )
            elif kind == "wifi_scan":
                self._queue_wifi_job("scan", None, client_id)
            elif kind == "wifi_connect":
                self._queue_wifi_job("connect", payload, client_id)

    def _request_agent_release(self, owned, client_id):
        # 接管屏障保持到本体控制服务确认；失败保持禁止接管，不能虚报停止。
        requests = []
        if owned['automatic']:
            self._mpc_request_generation += 1
            self._mpc_desired_enabled = False
            self.mpc_enable_pub.publish(Bool(data=False))
            if owned.get('preserve_history'):
                self._request_history({'operation': 'pause', 'generation': owned['generation']}, client_id)
            else:
                self._request_vln(False, '', self.web_server.vln_mode(), '')
            requests.append(self.auto_control_client)
        if owned['manual']:
            zero = TwistStamped()
            zero.header.stamp = self.get_clock().now().to_msg()
            zero.header.frame_id = 'base_link'
            self.manual_command_pub.publish(zero)
            requests.append(self.manual_control_client)
        pending = len(requests)
        errors = []

        def completed(future):
            nonlocal pending
            try:
                result = future.result()
                if not result.success:
                    errors.append(str(result.message))
            except Exception as exc:
                errors.append(str(exc))
            pending -= 1
            if pending == 0:
                self.web_server.agent_release_finished(
                    client_id, not errors, '; '.join(errors) or 'owned control disabled'
                )

        for client in requests:
            if not client.service_is_ready():
                from concurrent.futures import Future
                future = Future()
                future.set_exception(RuntimeError('robot control service unavailable'))
                completed(future)
            else:
                request = SetBool.Request()
                request.data = False
                client.call_async(request).add_done_callback(completed)

    def _refresh_mpc_config(self) -> None:
        if self._mpc_config_future is not None:
            return
        if not self.mpc_get_parameters_client.service_is_ready():
            self.web_server.set_mpc_config_error(
                "vln_mpc parameter service is unavailable"
            )
            return
        request = GetParameters.Request()
        request.names = list(MPC_CONFIG_NAMES)
        generation = self._mpc_config_generation
        future = self.mpc_get_parameters_client.call_async(request)
        self._mpc_config_future = future
        future.add_done_callback(
            lambda done, gen=generation: self._mpc_config_result(done, gen)
        )

    def _mpc_config_result(self, future, generation: int) -> None:
        if self._mpc_config_future is future:
            self._mpc_config_future = None
        if generation != self._mpc_config_generation:
            return
        try:
            response = future.result()
            if len(response.values) != len(MPC_CONFIG_NAMES):
                raise RuntimeError("incomplete MPC parameter response")
            config = {
                name: float(parameter_value_to_python(value))
                for name, value in zip(MPC_CONFIG_NAMES, response.values)
            }
        except Exception as exc:
            self.web_server.set_mpc_config_error(str(exc))
            return
        self.web_server.update_mpc_config(config)

    def _request_mpc_config(
        self,
        config: dict[str, float],
        client_id: str,
    ) -> None:
        if not self.mpc_set_parameters_client.service_is_ready():
            self._service_error(
                client_id,
                "vln_mpc parameter service is unavailable",
                "mpc_config",
            )
            return
        self._mpc_config_generation += 1
        generation = self._mpc_config_generation
        request = SetParametersAtomically.Request()
        request.parameters = [
            Parameter(name=name, value=config[name]).to_parameter_msg()
            for name in MPC_CONFIG_NAMES
        ]
        future = self.mpc_set_parameters_client.call_async(request)
        future.add_done_callback(
            lambda done, cid=client_id, values=dict(config), gen=generation: (
                self._mpc_config_update_result(done, cid, values, gen)
            )
        )

    def _mpc_config_update_result(
        self,
        future,
        client_id: str,
        config: dict[str, float],
        generation: int,
    ) -> None:
        try:
            result = future.result().result
            if not result.successful:
                raise RuntimeError(result.reason or "MPC parameter update failed")
        except Exception as exc:
            self._service_error(client_id, str(exc), "mpc_config")
            return
        if generation == self._mpc_config_generation:
            self.web_server.update_mpc_config(config)
        self.web_server.send_client(
            client_id,
            {
                "type": "command_result",
                "command": "mpc_config",
                "ok": True,
                "message": "MPC parameters updated",
            },
        )

    def _request_body_limits(self, config, client_id):
        client = self.body_set_parameters_client
        if not client.service_is_ready():
            self._service_error(client_id, 'go2_adapter parameter service unavailable', 'body_limits')
            return
        request = SetParametersAtomically.Request()
        request.parameters = [Parameter(name=k, value=float(v)).to_parameter_msg() for k, v in config.items()]
        def finished(future):
            try:
                result = future.result().result
                if not result.successful:
                    raise RuntimeError(result.reason)
            except Exception as exc:
                self._service_error(client_id, str(exc), 'body_limits')
                return
            self.web_server.send_client(client_id, {'type': 'command_result', 'command': 'body_limits',
                                                   'ok': True, 'config': config})
        client.call_async(request).add_done_callback(finished)

    def _request_manual_limits(
        self,
        config: dict[str, float],
        client_id: str,
    ) -> None:
        parameter_names = {
            "linear": "manual_linear_limit",
            "angular": "manual_angular_limit",
            "linear_accel": "manual_linear_accel",
            "angular_accel": "manual_angular_accel",
        }
        result = self.set_parameters_atomically(
            [
                Parameter(name=parameter_names[name], value=config[name])
                for name in MANUAL_LIMIT_NAMES
            ]
        )
        if not result.successful:
            self._service_error(
                client_id,
                result.reason or "WASD parameter update failed",
                "manual_limits",
            )
            return
        self.manual_linear_limit = config["linear"]
        self.manual_angular_limit = config["angular"]
        self.manual_linear_accel = config["linear_accel"]
        self.manual_angular_accel = config["angular_accel"]
        self.web_server.update_manual_limits(config)
        self.web_server.send_client(
            client_id,
            {
                "type": "command_result",
                "command": "manual_limits",
                "ok": True,
                "message": "WASD parameters updated",
            },
        )

    def _queue_wifi_job(self, kind: str, payload: Any, client_id: str) -> None:
        with self._wifi_busy_lock:
            if self._wifi_busy:
                self._service_error(
                    client_id, "Wi-Fi operation already in progress"
                )
                return
            self._wifi_busy = True
        try:
            self._wifi_jobs.put_nowait((kind, payload, client_id))
        except queue.Full:
            with self._wifi_busy_lock:
                self._wifi_busy = False
            self._service_error(client_id, "Wi-Fi operation already in progress")
            return
        self.web_server.set_wifi_busy(kind)

    def _wifi_worker(self) -> None:
        while not self._wifi_stop.is_set():
            try:
                kind, payload, client_id = self._wifi_jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if kind == "scan":
                    data = self._wifi_manager.scan()
                    message = "Wi-Fi scan complete"
                elif kind == "connect":
                    data = self._wifi_manager.connect(
                        str(payload["ssid"]), str(payload["password"])
                    )
                    message = f"Connected to Wi-Fi {data['current_ssid']}"
                else:
                    raise ValueError(f"unknown Wi-Fi operation {kind!r}")
            except Exception as exc:
                self.web_server.set_wifi_error(str(exc))
                self._service_error(client_id, str(exc))
            else:
                self.web_server.update_wifi(data)
                if client_id:
                    self.web_server.send_client(
                        client_id,
                        {
                            "type": "command_result",
                            "command": f"wifi_{kind}",
                            "ok": True,
                            "message": message,
                        },
                    )
            finally:
                self._wifi_jobs.task_done()
                with self._wifi_busy_lock:
                    self._wifi_busy = False

    def _request_mpc(self, enabled: bool, client_id: str) -> None:
        self._mpc_request_generation += 1
        generation = self._mpc_request_generation
        self._mpc_desired_enabled = enabled
        if not self.auto_control_client.service_is_ready():
            message = f"service {self.auto_control_service} is unavailable"
            if enabled:
                self._mpc_start_failed(client_id, message, generation)
            else:
                self.mpc_enable_pub.publish(Bool(data=False))
                self._service_error(client_id, message)
            return
        self.mpc_enable_pub.publish(Bool(data=enabled))
        request = SetBool.Request()
        request.data = enabled
        source_future = self.auto_control_client.call_async(request)
        source_future.add_done_callback(
            lambda done, cid=client_id, gen=generation, active=enabled: (
                self._auto_control_result(done, cid, gen, active)
            )
        )

    def _auto_control_result(
        self,
        future,
        client_id: str,
        generation: int,
        enabled: bool,
    ) -> None:
        error = ""
        try:
            result = future.result()
            if not result.success:
                error = str(result.message) or "auto control request failed"
        except Exception as exc:
            error = str(exc)
        if generation != self._mpc_request_generation:
            if not self._mpc_desired_enabled:
                source_rollback = SetBool.Request()
                source_rollback.data = False
                self.auto_control_client.call_async(source_rollback)
                self.mpc_enable_pub.publish(Bool(data=False))
            return
        if error:
            if enabled:
                rollback = SetBool.Request()
                rollback.data = False
                if self.auto_control_client.service_is_ready():
                    self.auto_control_client.call_async(rollback)
                self._mpc_start_failed(client_id, error, generation)
            else:
                self._service_error(client_id, error)
            return
        self.web_server.send_client(
            client_id,
            {
                "type": "command_result",
                "command": "mpc",
                "ok": True,
                "enabled": enabled,
                "message": (
                    "MPC auto control enabled"
                    if enabled
                    else "MPC auto control disabled"
                ),
            },
        )

    def _mpc_start_failed(
        self, client_id: str, message: str, generation: int
    ) -> None:
        if generation != self._mpc_request_generation:
            return
        self._mpc_desired_enabled = False
        self.mpc_enable_pub.publish(Bool(data=False))
        self.web_server.clear_auto_controller(message)
        self._request_vln(False, "", self.web_server.vln_mode(), "")
        self._service_error(client_id, message)

    def _request_history(self, payload, client_id):
        operation, generation = payload['operation'], payload['generation']
        if operation == 'resume':
            self.mode_pub.publish(String(data=payload['mode']))
            self.web_server.reset_vln(payload['instruction'], payload['mode'],
                                      started_stamp_ns=self.get_clock().now().nanoseconds)
        elif operation == 'finish':
            self.web_server.reset_vln('', self.web_server.vln_mode())
        self._history_commands[(generation, operation)] = client_id
        self.history_pub.publish(String(data=json.dumps(payload)))

    def _request_vln(
        self,
        enabled: bool,
        instruction: str,
        mode: str,
        client_id: str,
    ) -> None:
        command = instruction if enabled else ""
        self.mode_pub.publish(String(data=mode))
        self.instruction_pub.publish(String(data=command))
        self.web_server.reset_vln(command, mode, started_stamp_ns=self.get_clock().now().nanoseconds)
        self.web_server.send_client(
            client_id,
            {
                "type": "command_result",
                "command": "vln",
                "ok": True,
                "enabled": enabled,
                "message": (
                    "VLN instruction published"
                    if enabled
                    else "VLN stop published"
                ),
            },
        )

    def _request_server_url(self, server_url: str, client_id: str) -> None:
        self.server_url_pub.publish(String(data=server_url))
        self.web_server.update_server_url(server_url)
        self.web_server.send_client(
            client_id,
            {
                "type": "command_result",
                "command": "server_url",
                "ok": True,
                "server_url": server_url,
                "message": f"VLN server changed to {server_url}",
            },
        )

    def _complete_objnav(self) -> None:
        self.get_logger().info("ObjNav model stop; releasing VLN and MPC")
        self.web_server.complete_navigation()

    def _manual_control_result(
        self, future, client_id: str, enabled: bool
    ) -> None:
        try:
            result = future.result()
            success = bool(result.success)
            message = str(result.message)
        except Exception as exc:
            success = False
            message = str(exc)
        self.web_server.send_client(
            client_id,
            {
                "type": "command_result",
                "command": "manual_control",
                "ok": success,
                "enabled": enabled,
                "message": message,
            },
        )
        if enabled and not success:
            self.web_server.revoke_controller(message)

    def _trigger_result(self, future, client_id: str, command: str) -> None:
        try:
            result = future.result()
            success = bool(result.success)
            message = str(result.message)
        except Exception as exc:
            success = False
            message = str(exc)
        self.web_server.send_client(
            client_id,
            {
                "type": "command_result",
                "command": command,
                "ok": success,
                "message": message,
            },
        )

    def _service_error(
        self,
        client_id: str,
        message: str,
        command: str = "",
    ) -> None:
        payload = {"type": "command_result", "ok": False, "message": message}
        if command:
            payload["command"] = command
        self.web_server.send_client(client_id, payload)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wifi_stop.set()
        self.web_server.close()

    def destroy_node(self):
        self.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VlnWebNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin_ros() -> None:
        try:
            executor.spin()
        except ExternalShutdownException:
            pass
        except Exception:
            if rclpy.ok():
                raise

    ros_thread = threading.Thread(
        target=spin_ros, name="vln-web-ros", daemon=True
    )
    ros_thread.start()
    try:
        node.web_server.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        executor.shutdown(timeout_sec=2.0)
        ros_thread.join(timeout=2.0)
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
