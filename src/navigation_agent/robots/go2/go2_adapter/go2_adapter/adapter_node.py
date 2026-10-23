"""Bridge Unitree Go2 SDK2 to the shared ROS 2 robot interface."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from typing import Any

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster

from .safety import select_command
from .unitree_client import (
    GO2_MODE_NAMES,
    Go2Battery,
    Go2State,
    UnitreeClient,
    normalized_mode,
)

VALID_SOURCES = {"disabled", "manual", "auto"}


class Go2AdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("go2_adapter")
        self.declare_parameter("network_interface", Parameter.Type.STRING)
        self.declare_parameter("dds_domain_id", 0)
        self.declare_parameter("sdk_timeout_s", 2.0)
        self.declare_parameter("state_timeout_s", 0.5)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("input_watchdog_s", 0.35)
        self.declare_parameter("linear_limit", 0.3)
        self.declare_parameter("angular_limit", 0.6)
        self.declare_parameter("zero_frames", 5)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        self.network_interface = self._required_string_parameter(
            "network_interface"
        )
        self.dds_domain_id = int(self.get_parameter("dds_domain_id").value)
        self.sdk_timeout_s = float(self.get_parameter("sdk_timeout_s").value)
        self.state_timeout_s = float(
            self.get_parameter("state_timeout_s").value
        )
        self.control_rate_hz = float(
            self.get_parameter("control_rate_hz").value
        )
        self.input_watchdog_s = float(
            self.get_parameter("input_watchdog_s").value
        )
        self.final_zero_frames = int(self.get_parameter("zero_frames").value)
        self.linear_limit = float(self.get_parameter('linear_limit').value)
        self.angular_limit = float(self.get_parameter('angular_limit').value)
        if not all(math.isfinite(v) and v > 0 for v in (self.linear_limit, self.angular_limit)):
            raise ValueError('Go2 velocity limits must be finite and positive')
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        if (
            self.dds_domain_id < 0
            or min(
                self.sdk_timeout_s,
                self.state_timeout_s,
                self.control_rate_hz,
                self.input_watchdog_s,
            )
            <= 0.0
            or self.final_zero_frames < 0
            or not self.odom_frame
            or not self.base_frame
        ):
            raise ValueError("invalid Go2 adapter parameter")

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.diagnostics_pub = self.create_publisher(
            DiagnosticArray, "diagnostics", 10
        )
        self.odom_pub = self.create_publisher(Odometry, "odom", 20)
        self.sent_pub = self.create_publisher(TwistStamped, "cmd_vel", 20)
        self.source_pub = self.create_publisher(
            String, "control/source", latched
        )
        self.event_pub = self.create_publisher(String, "robot/events", 20)
        self.tf_broadcaster = (
            TransformBroadcaster(self) if self.publish_tf else None
        )
        self.create_subscription(
            TwistStamped, "web/cmd_vel", self._on_manual_command, 10
        )
        self.create_subscription(
            TwistStamped, "mpc/cmd_vel", self._on_auto_command, 10
        )
        # 控制切换串行等待 SDK；独立回调组让里程计与看门狗继续运行。
        self._control_callbacks = MutuallyExclusiveCallbackGroup()
        self.create_service(
            SetBool, "control/set_manual", self._set_manual_enabled,
            callback_group=self._control_callbacks,
        )
        self.create_service(
            SetBool, "control/set_auto", self._set_auto_enabled,
            callback_group=self._control_callbacks,
        )
        self.create_service(Trigger, "control/stop", self._software_stop)
        self.create_service(Trigger, "robot/stand", self._request_stand)
        self.create_service(Trigger, "robot/walk", self._request_walk)
        self.create_service(Trigger, "robot/sit", self._request_sit)
        self.create_service(
            Trigger, "robot/emergency_stop", self._emergency_stop
        )

        self._lock = threading.Lock()
        self._source = "disabled"
        self._release_future = None
        self._manual_command = (0.0, 0.0)
        self._auto_command = (0.0, 0.0)
        self._manual_received_s = float("-inf")
        self._auto_received_s = float("-inf")
        self._zero_frames_remaining = 0
        self._state: Go2State | None = None
        self._battery: Go2Battery | None = None
        self._last_sdk_error = ""
        self._avoidance_check = 'not_checked'
        self._safety_error = ''
        self._last_error_log_s = float("-inf")
        self._closed = False
        self._incoming: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=512)
        self.add_on_set_parameters_callback(self._set_motion_limits)
        self._client = UnitreeClient(
            network_interface=self.network_interface,
            domain_id=self.dds_domain_id,
            timeout_s=self.sdk_timeout_s,
            command_max_age_s=self.input_watchdog_s,
            state_callback=lambda value: self._put_incoming("state", value),
            battery_callback=lambda value: self._put_incoming("battery", value),
            sent_callback=lambda value: self._put_incoming("sent", value),
            event_callback=lambda value: self._put_incoming("event", value),
            error_callback=lambda value: self._put_incoming("error", value),
        )

        self.create_timer(0.005, self._process_incoming)
        self.create_timer(1.0 / self.control_rate_hz, self._command_tick)
        self.create_timer(1.0, self._publish_diagnostics)
        self._publish_source()
        self._publish_diagnostics()
        self.get_logger().info(
            "Go2 adapter ready "
            f"interface={self.network_interface} domain={self.dds_domain_id} "
            "state_topic=rt/lf/sportmodestate"
        )

    def _set_motion_limits(self, parameters):
        limits = {'linear_limit': self.linear_limit, 'angular_limit': self.angular_limit}
        changed = False
        for parameter in parameters:
            if parameter.name not in limits:
                continue
            value = parameter.value
            maximum = 1.5 if parameter.name == 'linear_limit' else 3.0
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= maximum):
                return SetParametersResult(successful=False, reason=f'{parameter.name} must be in (0, {maximum}]')
            limits[parameter.name] = float(value)
            changed = True
        if not changed:
            return SetParametersResult(successful=True)
        with self._lock:
            if self._source != 'disabled' or self._zero_frames_remaining > 0:
                return SetParametersResult(successful=False, reason='stop and release control before changing limits')
            self.linear_limit, self.angular_limit = limits['linear_limit'], limits['angular_limit']
        return SetParametersResult(successful=True)

    def _required_string_parameter(self, name: str) -> str:
        parameter = self.get_parameter(name)
        if parameter.type_ != Parameter.Type.STRING:
            raise ValueError(f"required string parameter {name!r} is not set")
        value = str(parameter.value).strip()
        if not value:
            raise ValueError(f"required string parameter {name!r} is empty")
        return value

    def _put_incoming(self, kind: str, value: Any) -> None:
        item = (kind, value)
        try:
            self._incoming.put_nowait(item)
        except queue.Full:
            try:
                self._incoming.get_nowait()
            except queue.Empty:
                pass
            self._incoming.put_nowait(item)

    def _process_incoming(self) -> None:
        for _ in range(200):
            try:
                kind, value = self._incoming.get_nowait()
            except queue.Empty:
                return
            if kind == "state":
                with self._lock:
                    self._state = value
                self._publish_odom(value)
            elif kind == "battery":
                with self._lock:
                    self._battery = value
            elif kind == "sent":
                # 真实 SDK 成功才能清除执行错误，普通状态帧不能覆盖失败。
                with self._lock:
                    self._last_sdk_error = ''
                self._publish_sent_twist(value)
            elif kind == "event":
                if value.get('action') == 'avoidance_check':
                    with self._lock:
                        self._avoidance_check = 'verified' if value['success'] else 'failed'
                        self._safety_error = value.get('message', '')
                message = String()
                message.data = json.dumps(value, separators=(",", ":"))
                self.event_pub.publish(message)
            elif kind == "error":
                error = str(value)
                with self._lock:
                    self._last_sdk_error = error
                now_s = time.monotonic()
                self._publish_diagnostics()
                if now_s - self._last_error_log_s >= 1.0:
                    self._last_error_log_s = now_s
                    self.get_logger().warning(error)

    def _connected_and_mode(self, now_s: float) -> tuple[bool, str]:
        state = self._state
        connected = (
            state is not None
            and now_s - state.received_s <= self.state_timeout_s
        )
        return (
            connected,
            normalized_mode(
                state.mode,
                error_code=state.error_code,
                body_height=state.body_height,
            )
            if connected
            else "UNKNOWN",
        )

    def _on_manual_command(self, message: TwistStamped) -> None:
        with self._lock:
            self._manual_command = (
                float(message.twist.linear.x),
                float(message.twist.angular.z),
            )
            self._manual_received_s = time.monotonic()

    def _on_auto_command(self, message: TwistStamped) -> None:
        with self._lock:
            self._auto_command = (
                float(message.twist.linear.x),
                float(message.twist.angular.z),
            )
            self._auto_received_s = time.monotonic()

    def _set_manual_enabled(self, request, response):
        return self._set_control_enabled('manual', request, response)

    def _set_auto_enabled(self, request, response):
        return self._set_control_enabled('auto', request, response)

    def _set_control_enabled(self, source, request, response):
        try:
            completion = self._set_source(source if request.data else 'disabled')
            if completion is not None:
                completion.result(timeout=2 * self.sdk_timeout_s + .5)
        except Exception as exc:
            response.success = False
            response.message = f'GO2 control change failed: {exc or type(exc).__name__}'
        else:
            response.success = True
            response.message = f'{source} control enabled' if request.data else 'SDK control released'
        return response

    def _set_source(self, source: str):
        if source not in VALID_SOURCES:
            raise ValueError(f"invalid control source {source!r}")
        now_s = time.monotonic()
        with self._lock:
            if source != 'disabled' and self._release_future is not None:
                if not self._release_future.done():
                    raise RuntimeError('GO2 SDK release is still pending')
                self._release_future.result()
            self._source = source
            self._manual_command = (0.0, 0.0)
            self._auto_command = (0.0, 0.0)
            self._manual_received_s = now_s
            self._auto_received_s = now_s
            self._zero_frames_remaining = (
                0
                if source == "disabled"
                else max(self._zero_frames_remaining, self.final_zero_frames)
            )
            if source == 'disabled':
                if self._release_future is None or self._release_future.done():
                    self._release_future = self._client.release_control()
                completion = self._release_future
            else:
                completion = None
        self._publish_source()
        self._publish_diagnostics()
        return completion

    def _request_stand(self, _request, response):
        return self._request_mode("stand", response)

    def _request_walk(self, _request, response):
        return self._request_mode("walk", response)

    def _request_sit(self, _request, response):
        self._set_source("disabled")
        return self._request_mode("sit", response)

    def _request_mode(self, mode: str, response):
        with self._lock:
            connected, current_mode = self._connected_and_mode(time.monotonic())
        if not connected:
            response.success = False
            response.message = "Go2 state is unavailable"
            return response
        if mode == "walk" and current_mode not in {"STAND", "WALK"}:
            response.success = False
            response.message = (
                f"current mode is {current_mode}; request stand first"
            )
            return response
        response.success = self._client.request_action(mode)
        response.message = (
            f"{mode} request queued"
            if response.success
            else "Go2 client is closed"
        )
        return response

    def _software_stop(self, _request, response):
        self._set_source("disabled")
        response.success = self._client.request_action("stop")
        response.message = (
            "control disabled and StopMove queued"
            if response.success
            else "Go2 client is closed"
        )
        return response

    def _emergency_stop(self, _request, response):
        self._set_source("disabled")
        response.success = self._client.request_action("damp")
        response.message = (
            "control disabled and Damp queued"
            if response.success
            else "Go2 client is closed"
        )
        return response

    def _command_tick(self) -> None:
        now_s = time.monotonic()
        disable_source = False
        with self._lock:
            connected, mode = self._connected_and_mode(now_s)
            if not connected and self._source != "disabled":
                disable_source = True
            command = select_command(
                source=self._source,
                robot_mode=mode,
                manual_command=self._manual_command,
                manual_age_s=now_s - self._manual_received_s,
                auto_command=self._auto_command,
                auto_age_s=now_s - self._auto_received_s,
                watchdog_s=self.input_watchdog_s,
                linear_limit=self.linear_limit,
                angular_limit=self.angular_limit,
            )
            if command is not None:
                self._zero_frames_remaining = self.final_zero_frames
            elif self._zero_frames_remaining > 0:
                command = (0.0, 0.0)
                self._zero_frames_remaining -= 1
            # 与控制源切换共用锁，禁止释放后再入队先前算出的速度。
            if not disable_source and command is not None:
                self._client.move(command[0], 0.0, command[1])
        if disable_source:
            self._set_source("disabled")

    def _publish_odom(self, state: Go2State) -> None:
        message = Odometry()
        message.header.stamp.sec = state.received_ns // 1_000_000_000
        message.header.stamp.nanosec = state.received_ns % 1_000_000_000
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame
        message.pose.pose.position.x = state.position[0]
        message.pose.pose.position.y = state.position[1]
        message.pose.pose.position.z = state.position[2]
        (
            message.pose.pose.orientation.x,
            message.pose.pose.orientation.y,
            message.pose.pose.orientation.z,
            message.pose.pose.orientation.w,
        ) = state.orientation_xyzw
        message.twist.twist.linear.x = state.velocity[0]
        message.twist.twist.linear.y = state.velocity[1]
        message.twist.twist.linear.z = state.velocity[2]
        message.twist.twist.angular.z = state.yaw_speed
        self.odom_pub.publish(message)
        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = message.header
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = state.position[0]
            transform.transform.translation.y = state.position[1]
            transform.transform.translation.z = state.position[2]
            transform.transform.rotation = message.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)

    def _publish_sent_twist(
        self, command: tuple[float, float, float]
    ) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame
        message.twist.linear.x = command[0]
        message.twist.linear.y = command[1]
        message.twist.angular.z = command[2]
        self.sent_pub.publish(message)

    def _publish_source(self) -> None:
        if not rclpy.ok():
            return
        with self._lock:
            source = self._source
        self.source_pub.publish(String(data=source))

    def _publish_diagnostics(self) -> None:
        if not rclpy.ok():
            return
        now_s = time.monotonic()
        with self._lock:
            connected, mode = self._connected_and_mode(now_s)
            state = self._state
            battery = self._battery
            source = self._source
            error = self._last_sdk_error
            avoidance_check = self._avoidance_check
            safety_error = self._safety_error
        status = DiagnosticStatus()
        status.name = "robot_adapter"
        status.hardware_id = "go2"
        status.level = (
            DiagnosticStatus.OK if connected else DiagnosticStatus.WARN
        )
        status.message = "online" if connected else "waiting for Go2 state"
        if safety_error or error:
            status.level = DiagnosticStatus.ERROR
            status.message = safety_error or error
        unitree_mode = state.mode if state is not None else -1
        values = {
            "adapter": "go2",
            "connected": connected,
            "robot_id": "go2",
            "mode": mode,
            "battery": (
                battery.percentage if battery is not None else None
            ),
            "imu": "OK" if connected else "UNKNOWN",
            "motor": (
                "OK" if battery is not None and battery.motors_ok else "UNKNOWN"
            ),
            "motor_temps": (
                json.dumps(list(battery.motor_temps), separators=(",", ":"))
                if battery is not None and battery.motor_temps is not None
                else None
            ),
            "motor_temp_max": (
                max(battery.motor_temps)
                if battery is not None and battery.motor_temps
                else None
            ),
            "motor_temps_age_s": max(0., now_s - battery.received_s) if battery is not None else None,
            "control_source": source,
            "network_interface": self.network_interface,
            "unitree_mode": GO2_MODE_NAMES.get(unitree_mode, unitree_mode),
            "gait_type": state.gait_type if state is not None else None,
            "body_height": state.body_height if state is not None else None,
            "sport_error_code": state.error_code if state is not None else None,
            "battery_voltage": battery.voltage if battery is not None else None,
            "sdk_error": error or None,
            "avoidance_check": avoidance_check,
            "safety_error": safety_error,
            "linear_limit": self.linear_limit,
            "angular_limit": self.angular_limit,
            "custom_clearance_available": False,
        }
        status.values = [
            KeyValue(key=key, value="" if value is None else str(value))
            for key, value in values.items()
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status.append(status)
        self.diagnostics_pub.publish(message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._set_source("disabled")
        self._client.close()

    def destroy_node(self):
        self.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Go2AdapterNode()
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        rclpy.spin(node, executor=executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
