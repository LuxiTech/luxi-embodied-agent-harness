"""ROS 2 adapter between shared MuJoCo physics and robot-scoped containers.

The host simulator intentionally has no ROS installation.  It publishes
atomic, robot-namespaced hardware frames; this simulation-only process turns
them into ROS messages and carries robot-scoped skill/Twist requests back to
the matching MuJoCo runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np

from harness.robots.go2.robot_messages import GO2_ROBOT_IDS
from harness.robots.go2.go2_protocol import (
    Go2RuntimePaths,
    atomic_write_json,
    read_json,
)


GO2_ROUTE_TERMINAL_MAX_DISTANCE_M = 1.25


def route_terminal_matches_standoff(
    route_xy: list[list[float]],
    goal_xy: list[float] | None,
) -> bool:
    """Validate a route endpoint without requiring entry into target occupancy."""

    return bool(
        not route_xy
        or (
            goal_xy is not None
            and math.dist(route_xy[-1], goal_xy)
            <= GO2_ROUTE_TERMINAL_MAX_DISTANCE_M
        )
    )


def ros_robot_name(robot_id: str) -> str:
    normalized = str(robot_id).strip().lower()
    if normalized not in GO2_ROBOT_IDS:
        raise ValueError(f"unsupported robot: {robot_id!r}")
    return normalized.replace("-", "_")


def sim_topic(robot_id: str, suffix: str) -> str:
    return f"/sim/{ros_robot_name(robot_id)}/{suffix.strip('/')}"


def command_topic(robot_id: str, suffix: str) -> str:
    return f"/{ros_robot_name(robot_id)}/{suffix.strip('/')}"


class Go2RosSimAdapter:
    """One shared-physics adapter with strictly robot-scoped endpoints."""

    def __init__(self, runtime_root: Path) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import CompressedImage, Imu, PointCloud2, PointField
            from std_msgs.msg import String
        except ImportError as error:  # pragma: no cover - container dependency
            raise RuntimeError("MuJoCo ROS adapter requires a sourced ROS 2 environment") from error

        self.rclpy = rclpy
        self.String = String
        self.Twist = Twist
        self.CompressedImage = CompressedImage
        self.Imu = Imu
        self.PointCloud2 = PointCloud2
        self.PointField = PointField
        self.paths = Go2RuntimePaths(runtime_root)
        self.paths.ensure()
        self.node = rclpy.create_node("luxi_go2_mujoco_adapter")
        durable = QoSProfile(depth=10)
        durable.reliability = ReliabilityPolicy.RELIABLE
        durable.durability = DurabilityPolicy.TRANSIENT_LOCAL
        state_qos = QoSProfile(depth=10)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        sensor_qos = QoSProfile(depth=2)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._publishers: dict[tuple[str, str], Any] = {}
        self._last_state_write = {robot_id: 0.0 for robot_id in GO2_ROBOT_IDS}
        self._last_map_revision = {robot_id: -1 for robot_id in GO2_ROBOT_IDS}
        self._last_map_timestamp = {robot_id: 0.0 for robot_id in GO2_ROBOT_IDS}
        self._last_sensor_sequence = {robot_id: -1 for robot_id in GO2_ROBOT_IDS}
        self._last_sensor_timestamp = {robot_id: 0.0 for robot_id in GO2_ROBOT_IDS}
        self._last_camera_mtime = {robot_id: -1 for robot_id in GO2_ROBOT_IDS}
        self._last_result_signature = {robot_id: "" for robot_id in GO2_ROBOT_IDS}
        self._task_context: dict[str, dict[str, Any]] = {}
        self._cmd_sequence = {robot_id: 0 for robot_id in GO2_ROBOT_IDS}
        existing_task_request = read_json(self.paths.ros_task) or {}
        self._last_sequence = int(existing_task_request.get("sequence", 0))
        self._stopping = False

        self._task_request_publisher = self.node.create_publisher(
            String, command_topic("go2-01", "task_request"), durable
        )
        for robot_id in sorted(GO2_ROBOT_IDS):
            for key, message_type, qos in (
                ("state", String, state_qos),
                ("local_map", String, durable),
                ("lidar_proximity", String, sensor_qos),
                ("task_result", String, durable),
                ("mid360/points", PointCloud2, sensor_qos),
                ("mid360/imu", Imu, sensor_qos),
                ("hikrobot/image/compressed", CompressedImage, sensor_qos),
            ):
                self._publishers[(robot_id, key)] = self.node.create_publisher(
                    message_type, sim_topic(robot_id, key), qos
                )
            self.node.create_subscription(
                String,
                sim_topic(robot_id, "task_command"),
                lambda message, selected=robot_id: self._on_task_command(
                    selected, message.data
                ),
                durable,
            )
            self.node.create_subscription(
                Twist,
                command_topic(robot_id, "cmd_vel"),
                lambda message, selected=robot_id: self._on_cmd_vel(selected, message),
                state_qos,
            )
            self.node.create_subscription(
                String,
                command_topic(robot_id, "navigation_status"),
                lambda message, selected=robot_id: self._on_navigation_status(
                    selected, message.data
                ),
                state_qos,
            )
        self.node.create_timer(0.05, self._poll)

    def _publish_json(self, robot_id: str, key: str, payload: dict[str, Any]) -> None:
        message = self.String()
        message.data = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        self._publishers[(robot_id, key)].publish(message)

    def _on_task_command(self, expected_robot_id: str, raw: str) -> None:
        try:
            payload = json.loads(raw)
            robot_id = str(payload["robot_id"])
            instruction = str(payload["instruction"]).strip()
            sequence = int(payload["sequence"])
            task_id = str(payload["task_id"])
            boot_epoch = str(payload["boot_epoch"])
            expires_at = float(payload["expires_at"])
            raw_goal = payload.get("goal_xy")
            raw_route = payload.get("route_xy") or []
            control_mode = str(payload.get("control_mode", "simulator_skill"))
            navigation_timeout_s = float(
                payload.get("navigation_timeout_s", 240.0)
            )
            goal_xy = (
                [float(raw_goal[0]), float(raw_goal[1])]
                if raw_goal is not None
                else None
            )
            route_xy = [
                [float(point[0]), float(point[1])]
                for point in raw_route
            ]
        except (
            json.JSONDecodeError,
            IndexError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ):
            return
        if (
            robot_id != expected_robot_id
            or expected_robot_id != "go2-01"
            or not boot_epoch
            or not instruction
            or control_mode not in {"simulator_skill", "ros_local_planner"}
            or not math.isfinite(expires_at) or not 0 <= expires_at - time.time() <= 2.0
            or not math.isfinite(navigation_timeout_s)
            or not 120.0 <= navigation_timeout_s <= 360.0
            or (
                goal_xy is not None
                and (
                    len(goal_xy) != 2
                    or not all(math.isfinite(value) for value in goal_xy)
                )
            )
            or len(route_xy) > 512
            or any(
                len(point) != 2
                or not all(math.isfinite(value) for value in point)
                for point in route_xy
            )
            or not route_terminal_matches_standoff(route_xy, goal_xy)
        ):
            return
        paths = self.paths
        pending = read_json(paths.ros_task) or {}
        current = read_json(paths.state) or {}
        if (pending.get("runtime_boot_epoch") != current.get("boot_epoch")
                or pending.get("sequence") != sequence or pending.get("instruction") != instruction
                or sequence <= int(current.get("last_command_sequence", 0))
                or bool(current.get("busy"))):
            return
        atomic_write_json(
            paths.command,
            {
                "schema_version": 1,
                # Preserve the admitted single-robot request sequence.
                "sequence": sequence,
                "instruction": instruction,
                "runtime_boot_epoch": pending.get("runtime_boot_epoch"),
                "target_prior_xy": goal_xy,
                "route_waypoints_xy": route_xy,
                "navigation_timeout_s": navigation_timeout_s,
                "control_mode": control_mode,
                "source": "ros2_robot_scoped_agent_skill",
                "robot_id": expected_robot_id,
                "task_id": task_id,
                "written_at": time.time(),
            },
        )
        self._task_context[expected_robot_id] = dict(payload)
        previous_result = current.get("last_result")
        self._last_result_signature[expected_robot_id] = (
            json.dumps(previous_result, sort_keys=True, ensure_ascii=False)
            if isinstance(previous_result, dict)
            else ""
        )
        try:
            paths.ros_navigation_status.unlink()
        except FileNotFoundError:
            pass

    def _on_navigation_status(self, expected_robot_id: str, raw: str) -> None:
        try:
            payload = json.loads(raw)
            robot_id = str(payload["robot_id"])
            sequence = int(payload["sequence"])
            status = str(payload["status"])
            goal_xy = [float(value) for value in payload["goal_xy"]]
            written_at = float(payload["written_at"])
        except (
            json.JSONDecodeError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ):
            return
        if (
            robot_id != expected_robot_id
            or len(goal_xy) != 2
            or not all(math.isfinite(value) for value in (*goal_xy, written_at))
            or status not in {
                "tracking",
                "aligning",
                "scanning",
                "recovering",
                "recovery_waiting",
                "safety_hold",
                "path_unavailable",
                "path_invalidated",
                "motion_disabled",
                "arrived",
                "risk_blocked",
                "verification_failed",
            }
            or abs(time.time() - written_at) > 1.5
        ):
            return
        atomic_write_json(
            self.paths.ros_navigation_status,
            {
                **payload,
                "schema_version": 1,
                "sequence": sequence,
                "robot_id": expected_robot_id,
                "goal_xy": goal_xy,
            },
        )

    def _on_cmd_vel(self, robot_id: str, message: Any) -> None:
        values = (
            float(message.linear.x),
            float(message.linear.y),
            float(message.angular.z),
        )
        if not all(math.isfinite(value) for value in values):
            return
        self._cmd_sequence[robot_id] += 1
        now = time.time()
        atomic_write_json(
            self.paths.ros_cmd_vel,
            {
                "schema_version": 1,
                "sequence": self._cmd_sequence[robot_id],
                "robot_id": robot_id,
                "issued_at": now,
                "expires_at": now + 0.20,
                "twist": {
                    "linear": {"x": values[0], "y": values[1], "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": values[2]},
                },
            },
        )

    def _publish_state(self, robot_id: str) -> None:
        paths = self.paths
        payload = read_json(paths.state)
        if not isinstance(payload, dict) or payload.get("robot_id") != robot_id:
            return
        written_at = float(payload.get("written_at", 0.0))
        if written_at <= self._last_state_write[robot_id]:
            return
        self._last_state_write[robot_id] = written_at
        self._publish_json(robot_id, "state", payload)
        lidar = read_json(paths.lidar_proximity)
        if isinstance(lidar, dict):
            self._publish_json(robot_id, "lidar_proximity", lidar)
        result = payload.get("last_result")
        context = self._task_context.get(robot_id)
        if not isinstance(result, dict) or context is None:
            return
        signature = json.dumps(result, sort_keys=True, ensure_ascii=False)
        if signature == self._last_result_signature[robot_id]:
            return
        self._last_result_signature[robot_id] = signature
        self._publish_json(
            robot_id,
            "task_result",
            {
                "schema_version": 1,
                "robot_id": robot_id,
                "request_id": context.get("request_id"),
                "sequence": context.get("sequence"),
                "task_id": context.get("task_id"),
                "result": result,
                "written_at": time.time(),
            },
        )
        self._task_context.pop(robot_id, None)

    def _publish_map(self, robot_id: str) -> None:
        payload = read_json(self.paths.costmap)
        if not isinstance(payload, dict) or payload.get("robot_id") != robot_id:
            return
        revision = int(payload.get("revision", -1))
        timestamp = float(payload.get("timestamp", 0.0))
        if revision <= self._last_map_revision[robot_id] and not (
            revision < self._last_map_revision[robot_id]
            and timestamp > self._last_map_timestamp[robot_id]
        ):
            return
        self._last_map_timestamp[robot_id] = timestamp
        self._last_map_revision[robot_id] = revision
        self._publish_json(robot_id, "local_map", payload)

    def _publish_sensor_frame(self, robot_id: str) -> None:
        path = self.paths.mid360_frame
        try:
            with np.load(path, allow_pickle=False) as frame:
                sequence = int(frame["sequence"][0])
                timestamp = float(frame["timestamp"][0])
                if sequence <= self._last_sensor_sequence[robot_id] and not (
                    sequence < self._last_sensor_sequence[robot_id]
                    and timestamp > self._last_sensor_timestamp[robot_id]
                ):
                    return
                points = np.asarray(frame["points"], dtype="<f4")
                gyro = np.asarray(frame["gyro"], dtype=np.float64)
                acceleration = np.asarray(frame["acceleration"], dtype=np.float64)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return
        if points.ndim != 2 or points.shape[1] != 3 or not len(gyro):
            return
        self._last_sensor_sequence[robot_id] = sequence
        self._last_sensor_timestamp[robot_id] = timestamp
        stamp = self.node.get_clock().now().to_msg()
        cloud = self.PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = f"{ros_robot_name(robot_id)}/mid360"
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = [
            self.PointField(name="x", offset=0, datatype=self.PointField.FLOAT32, count=1),
            self.PointField(name="y", offset=4, datatype=self.PointField.FLOAT32, count=1),
            self.PointField(name="z", offset=8, datatype=self.PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = points.tobytes(order="C")
        self._publishers[(robot_id, "mid360/points")].publish(cloud)
        for angular, linear in zip(gyro, acceleration, strict=True):
            imu = self.Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = f"{ros_robot_name(robot_id)}/mid360_imu"
            imu.orientation.w = 1.0
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (
                float(value) for value in angular
            )
            imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (
                float(value) for value in linear
            )
            self._publishers[(robot_id, "mid360/imu")].publish(imu)

    def _publish_camera(self, robot_id: str) -> None:
        path = self.paths.camera
        try:
            mtime = path.stat().st_mtime_ns
            if mtime <= self._last_camera_mtime[robot_id]:
                return
            data = path.read_bytes()
        except OSError:
            return
        self._last_camera_mtime[robot_id] = mtime
        message = self.CompressedImage()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = f"{ros_robot_name(robot_id)}/hikrobot"
        message.format = "jpeg"
        message.data = data
        self._publishers[(robot_id, "hikrobot/image/compressed")].publish(message)

    def _publish_task_request(self) -> None:
        payload = read_json(self.paths.ros_task)
        if not isinstance(payload, dict) or payload.get('robot_id') != 'go2-01':
            return
        try:
            sequence = int(payload['sequence'])
            age = time.time() - float(payload['written_at'])
        except (KeyError, TypeError, ValueError):
            return
        if sequence <= self._last_sequence or not 0 <= age <= 2.0:
            return
        self._last_sequence = sequence
        message = self.String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        self._task_request_publisher.publish(message)

    def _poll(self) -> None:
        self._publish_task_request()
        for robot_id in sorted(GO2_ROBOT_IDS):
            self._publish_state(robot_id)
            self._publish_map(robot_id)
            self._publish_sensor_frame(robot_id)
            self._publish_camera(robot_id)
        Path("/tmp/luxi-go2-sim-adapter-status.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "robots": sorted(GO2_ROBOT_IDS),
                    "last_state_write": self._last_state_write,
                    "last_sensor_sequence": self._last_sensor_sequence,
                    "last_sequence": self._last_sequence,
                    "written_at": time.time(),
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def run(self) -> None:
        print(
            json.dumps(
                {
                    "event": "go2_mujoco_ros_adapter_ready",
                    "runtime_root": str(self.paths.root),
                    "robots": sorted(GO2_ROBOT_IDS),
                }
            ),
            flush=True,
        )
        while not self._stopping and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.10)

    def stop(self, *_args: Any) -> None:
        self._stopping = True

    def close(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/runtime"),
    )
    args = parser.parse_args(argv)
    import rclpy

    rclpy.init(args=None)
    adapter = Go2RosSimAdapter(args.runtime_root)
    signal.signal(signal.SIGINT, adapter.stop)
    signal.signal(signal.SIGTERM, adapter.stop)
    try:
        adapter.run()
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
