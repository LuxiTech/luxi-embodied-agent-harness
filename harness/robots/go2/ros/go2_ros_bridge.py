"""ROS 2/DDS transport for single-Go2 sensor, task and result messages."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Mapping
from uuid import uuid4

from harness.robots.go2.robot_messages import (
    GO2_ROBOT_IDS,
    RobotMessagePublisher,
)


GO2_ROS_TOPICS = {
    "robot_state": "/{robot_id}/state", "robot_local_map": "/{robot_id}/local_map",
    "robot_heartbeat": "/{robot_id}/heartbeat", "task_result": "/{robot_id}/task_result",
}


def go2_robot_topic(topic_key: str, robot_id: str) -> str:
    """Return a ROS-valid topic while preserving the hyphenated wire identity."""

    normalized = str(robot_id).strip().lower()
    if normalized not in GO2_ROBOT_IDS:
        raise ValueError(f"unsupported robot: {robot_id!r}")
    return GO2_ROS_TOPICS[topic_key].format(
        robot_id=normalized.replace("-", "_")
    )


class Go2RosBridgeUnavailable(RuntimeError):
    """Raised when ROS 2 Python bindings are not available."""


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_loads(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


class _RosJsonNode:
    """One isolated rclpy context so simulation nodes match separate processes."""

    def __init__(self, node_name: str) -> None:
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                LivelinessPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from std_msgs.msg import String
        except ImportError as error:  # pragma: no cover - host ROS dependency
            raise Go2RosBridgeUnavailable(
                "ROS 2 bridge requires rclpy and std_msgs; source a ROS 2 environment "
                "or leave LUXI_GO2_FLEET_TRANSPORT=atomic"
            ) from error

        self._rclpy = rclpy
        self._string_type = String
        self._context = Context()
        rclpy.init(args=None, context=self._context)
        self.node = rclpy.create_node(node_name, context=self._context)
        self._qos = {
            "state": QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
            "durable": QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=20,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
            "heartbeat": QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
                liveliness=LivelinessPolicy.AUTOMATIC,
            ),
            "ephemeral": QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
            "sensor": QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=2,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        }
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self.node)
        self._closed = False
        self._thread = threading.Thread(
            target=self._spin,
            name=f"{node_name}-executor",
            daemon=True,
        )
        self._thread.start()

    def _spin(self) -> None:  # pragma: no cover - requires ROS runtime
        while not self._closed and self._context.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def _publisher(self, topic: str, qos: str = "state") -> Any:
        return self.node.create_publisher(self._string_type, topic, self._qos[qos])

    def _subscription(
        self,
        topic: str,
        callback: Callable[[str], None],
        qos: str = "state",
    ) -> Any:
        return self.node.create_subscription(
            self._string_type,
            topic,
            lambda message: callback(message.data),
            self._qos[qos],
        )

    def _publish(self, publisher: Any, payload: Mapping[str, Any]) -> None:
        message = self._string_type()
        message.data = _json_dumps(payload)
        publisher.publish(message)

    def close(self) -> None:  # pragma: no cover - requires ROS runtime
        self._closed = True
        self._thread.join(timeout=1.0)
        self._executor.remove_node(self.node)
        self.node.destroy_node()
        self._context.shutdown()


class Go2RosRobotBridge(_RosJsonNode):
    """DDS publisher for one robot; no peers, scheduling or leases."""
    _PUBLISH_TOPIC = {"robot_state": "robot_state", "local_map": "robot_local_map",
                      "heartbeat": "robot_heartbeat", "task_result": "task_result"}
    _PUBLISH_QOS = {"robot_state": "state", "local_map": "durable",
                    "heartbeat": "heartbeat", "task_result": "durable"}

    def __init__(self, robot_id, *, boot_epoch=None):
        if robot_id not in GO2_ROBOT_IDS:
            raise ValueError("Only go2-01 is supported")
        self.robot_id = robot_id
        self.boot_epoch = boot_epoch or uuid4().hex
        super().__init__("luxi_go2_01")
        self._envelopes = RobotMessagePublisher(robot_id, self.boot_epoch)
        self._publishers = {kind: self._publisher(go2_robot_topic(topic, robot_id), self._PUBLISH_QOS[kind])
                            for kind, topic in self._PUBLISH_TOPIC.items()}

    def publish(self, kind, payload):
        if kind not in self._publishers:
            raise ValueError(f"Unsupported robot message: {kind}")
        envelope = self._envelopes.create(kind, payload)
        message = envelope.as_message()
        self._publish(self._publishers[kind], message)
        return message

    def status(self):
        return {"transport": "ros2_dds_robot_scoped", "robot_id": self.robot_id,
                "boot_epoch": self.boot_epoch, "node": self.node.get_name()}
