"""ROS 2 node for the VLN client."""

from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from .vln_client import SERVER_URL, CameraFrame, InferenceResult, VlnClient

IMAGE_TOPIC = "camera/color/image_raw"
IMAGE_TRANSPORT = "raw"
INSTRUCTION_TOPIC = "vln/instruction"
RESPONSE_TOPIC = "vln/response"
PATH_TOPIC = "vln/path_body"
STATUS_TOPIC = "vln/status"
SERVER_URL_TOPIC = "vln/server_url"
SERVER_URL_STATUS_TOPIC = "vln/server_url_status"
BASE_FRAME = "base_link"
RESULT_POLL_PERIOD_S = 0.02
STATUS_PERIOD_S = 1.0
METRICS_PERIOD_S = 2.0


class VlnClientNode(Node):
    def __init__(self) -> None:
        super().__init__("vln_client")
        self.declare_parameter("server_url", SERVER_URL)
        self.declare_parameter("image_topic", IMAGE_TOPIC)
        self.declare_parameter("image_transport", IMAGE_TRANSPORT)

        self.image_topic = str(self.get_parameter("image_topic").value).strip()
        self.image_transport = str(
            self.get_parameter("image_transport").value
        ).strip().lower()
        if (
            not self.image_topic
            or self.image_transport not in {"raw", "compressed"}
        ):
            raise ValueError("invalid VLN client image parameter")

        sensor_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
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
        visualization_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
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
            String,
            INSTRUCTION_TOPIC,
            self._on_instruction,
            command_qos,
        )
        self.create_subscription(
            String,
            SERVER_URL_TOPIC,
            self._on_server_url,
            status_qos,
        )
        self.session_state_pub = self.create_publisher(String, 'vln/session_state', status_qos)
        self.create_subscription(String, 'vln/session_control', self._on_session_control, command_qos)
        self._session_ack = {'history_continuation': True}
        self.response_pub = self.create_publisher(
            String,
            RESPONSE_TOPIC,
            response_qos,
        )
        self.path_pub = self.create_publisher(
            Path,
            PATH_TOPIC,
            visualization_qos,
        )
        self.status_pub = self.create_publisher(
            String,
            STATUS_TOPIC,
            status_qos,
        )
        self.server_url_pub = self.create_publisher(
            String,
            SERVER_URL_STATUS_TOPIC,
            status_qos,
        )

        self._client = VlnClient(
            logger=self.get_logger(),
            server_url=str(self.get_parameter("server_url").value),
        )
        self._last_invalid_encoding = ""
        self._metrics_started_at = time.monotonic()
        self._metrics_result_count = 0
        self._metrics_latency_sum_ms = 0.0
        self._metrics_latency_max_ms = 0.0
        self._closed = False

        self.create_timer(RESULT_POLL_PERIOD_S, self._poll_client)
        self.create_timer(STATUS_PERIOD_S, self._publish_status)
        self.create_timer(METRICS_PERIOD_S, self._log_metrics)
        self._publish_status()
        self._publish_server_url()
        self.get_logger().info(
            f"VLN client ready: image={self.image_topic} "
            f"transport={self.image_transport} "
            f"instruction={INSTRUCTION_TOPIC} response={RESPONSE_TOPIC} "
            f"path={PATH_TOPIC} status={STATUS_TOPIC} "
            f"server={self._client.snapshot().server_url}"
        )

    def _on_session_control(self, message: String) -> None:
        command = {}
        try:
            command = json.loads(message.data)
            if not isinstance(command, dict):
                command = {}
                raise ValueError('session command must be an object')
            generation = command['generation']
            if type(generation) is not int or generation < 1:
                raise ValueError('invalid navigation generation')
            operation = command['operation']
            reused = False
            if operation == 'resume':
                self._client.set_server_url(command['server_url'])
                reused = self._client.resume(command['instruction'], generation)
            elif operation == 'pause':
                self._client.pause(generation)
            elif operation == 'finish':
                self._client.finish(generation)
            else:
                raise ValueError('unknown session operation')
            self._session_ack = {'history_continuation': True, 'generation': generation,
                                 'operation': operation, 'ok': True, 'history_reused': reused}
        except (ValueError, KeyError, TypeError) as exc:
            self._session_ack = {'history_continuation': True, 'generation': command.get('generation'),
                                 'operation': command.get('operation'), 'ok': False, 'message': str(exc)}
        self._publish_status()

    def _on_instruction(self, message: String) -> None:
        instruction = message.data.strip()
        if instruction:
            try:
                episode = self._client.start(instruction)
            except ValueError as exc:
                self.get_logger().warning(str(exc))
            else:
                self.get_logger().info(
                    f"VLN episode {episode} started: {instruction!r}"
                )
        else:
            self._client.stop()
            self.get_logger().info("VLN stopped by empty instruction")
        self._publish_status()

    def _on_server_url(self, message: String) -> None:
        try:
            changed = self._client.set_server_url(message.data)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
        else:
            if changed:
                self.get_logger().info(
                    "VLN server changed: "
                    f"{self._client.snapshot().server_url}"
                )
        self._publish_server_url()
        self._publish_status()

    def _on_image(self, message: Image) -> None:
        encoding = message.encoding.lower()
        if encoding != "rgb8":
            if encoding != self._last_invalid_encoding:
                self._last_invalid_encoding = encoding
                self.get_logger().error(
                    f"camera encoding must be rgb8, got {message.encoding!r}"
                )
            return
        self._last_invalid_encoding = ""
        if not message.data:
            return
        self._client.offer_frame(
            stamp_sec=message.header.stamp.sec,
            stamp_nanosec=message.header.stamp.nanosec,
            width=message.width,
            height=message.height,
            step=message.step,
            rgb=message.data,
        )

    def _on_compressed_image(self, message: CompressedImage) -> None:
        if not message.data:
            return
        self._client.offer_compressed_frame(
            stamp_sec=message.header.stamp.sec,
            stamp_nanosec=message.header.stamp.nanosec,
            jpeg=message.data,
        )

    def _poll_client(self) -> None:
        for result in self._client.take_results():
            self._publish_result(result)

    def _publish_result(self, result: InferenceResult) -> None:
        waypoints = list(result.waypoints)
        response = {
            "episode": result.episode,
            "generation": result.generation,
            "seq": result.sequence,
            "capture_stamp_ns": (
                result.frame.stamp_sec * 1_000_000_000
                + result.frame.stamp_nanosec
            ),
            "frame_id": BASE_FRAME,
            "waypoints": [list(point) for point in waypoints],
            "stop": result.stop,
            "visible": result.visible,
            "apos_state": result.apos_state,
            "opos_state": result.opos_state,
            "apos_px": result.apos_px,
            "opos_px": result.opos_px,
        }
        self.response_pub.publish(
            String(
                data=json.dumps(
                    response,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
        )
        self.path_pub.publish(self._make_path(result.frame, waypoints))

        self._metrics_result_count += 1
        self._metrics_latency_sum_ms += result.latency_ms
        self._metrics_latency_max_ms = max(
            self._metrics_latency_max_ms,
            result.latency_ms,
        )

    def _publish_status(self) -> None:
        if not rclpy.ok():
            return
        snapshot = self._client.snapshot()
        self.status_pub.publish(String(data=snapshot.state))
        self.session_state_pub.publish(String(data=json.dumps({
            **self._session_ack, 'current_generation': self._client.generation, 'state': snapshot.state,
        })))

    def _publish_server_url(self) -> None:
        if not rclpy.ok():
            return
        self.server_url_pub.publish(
            String(data=self._client.snapshot().server_url)
        )

    @staticmethod
    def _make_path(
        frame: CameraFrame,
        waypoints: list[tuple[float, float, float]],
    ) -> Path:
        path = Path()
        path.header.stamp.sec = frame.stamp_sec
        path.header.stamp.nanosec = frame.stamp_nanosec
        path.header.frame_id = BASE_FRAME
        for forward, lateral, yaw in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = forward
            pose.pose.position.y = lateral
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        return path

    def _log_metrics(self) -> None:
        now = time.monotonic()
        elapsed = max(now - self._metrics_started_at, 1e-9)
        count = self._metrics_result_count
        snapshot = self._client.snapshot()
        if snapshot.state != "IDLE" or count:
            average = self._metrics_latency_sum_ms / count if count else 0.0
            self.get_logger().info(
                "VLN metrics: "
                f"result_hz={count / elapsed:.2f} "
                f"latency_ms(avg/max)={average:.2f}/"
                f"{self._metrics_latency_max_ms:.2f} "
                f"connected={snapshot.connected} "
                f"in_flight={snapshot.in_flight} "
                f"state={snapshot.state}"
            )
        self._metrics_started_at = now
        self._metrics_result_count = 0
        self._metrics_latency_sum_ms = 0.0
        self._metrics_latency_max_ms = 0.0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._client.close()

    def destroy_node(self):
        self.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VlnClientNode()
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
