"""Robot-local Agent/FAST-LIO2 runtime for one simulated or physical Go2."""

from __future__ import annotations

import argparse
import base64
from collections import deque
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any

import numpy as np

from harness.robots.go2.go2_fastlio import FastLio2Process
from harness.robots.go2.go2_navigation import (
    Mid360Slam2D,
    RosLocalPlanner,
    SlamPose,
    coordinate_navigation_timeout_s,
)
from harness.robots.go2.ros.go2_ros_bridge import Go2RosRobotBridge
from harness.robots.go2.ros.go2_ros_sim_adapter import command_topic, sim_topic
from harness.robots.go2.go2_semantic_map import Go2WarehouseSemanticMap


DEFAULT_CAPABILITIES = frozenset(
    {"warehouse_inspection", "object_search", "follow_person"}
)
ROBOT_ANCHORS = {
    "go2-01": SlamPose(-5.8, -3.0, 0.0),
}


def _decode_json(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


class Go2RosRobotNode:
    """One robot node that executes Harness tasks with local sensor navigation."""

    def __init__(
        self,
        robot_id: str,
        *,
        status_path: Path = Path("/tmp/luxi-go2-ros-status.json"),
    ) -> None:
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Imu, PointCloud2
        from std_msgs.msg import String

        self.String = String
        self.Twist = Twist
        self.bridge = Go2RosRobotBridge(robot_id)
        self.robot_id = self.bridge.robot_id
        self.semantic_map = Go2WarehouseSemanticMap()
        self.local_slam = Mid360Slam2D(
            width=160,
            height=120,
            resolution=0.10,
            origin_x=-8.0,
            origin_y=-6.0,
        )
        self.fastlio = FastLio2Process(
            robot_id=self.robot_id,
            anchor=ROBOT_ANCHORS[self.robot_id],
        )
        self.status_path = status_path
        self.stop_requested = threading.Event()
        self.started_at = time.time()
        self.last_raw_state: dict[str, Any] | None = None
        self.last_raw_state_at = 0.0
        self.last_fastlio_error: str | None = None
        self.last_plan: dict[str, Any] | None = None
        self.last_task_result: dict[str, Any] | None = None
        self._last_task_sequence = 0
        self._active_task_id = None
        self._sensor_lock = threading.RLock()
        self._pending_cloud: tuple[int, np.ndarray[Any, Any]] | None = None
        self._imu_samples: deque[
            tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = deque(maxlen=50)
        self._sensor_frames = 0
        self._last_local_map_revision = -1
        self._latest_lidar_proximity: dict[str, Any] | None = None
        self._dynamic_obstacles: dict[tuple[int, int], float] = {}

        self._task_command_publisher = self.bridge._publisher(
            sim_topic(self.robot_id, "task_command"), "durable"
        )
        self._cmd_vel_publisher = self.bridge.node.create_publisher(
            Twist,
            command_topic(self.robot_id, "cmd_vel"),
            self.bridge._qos["state"],
        )
        self._navigation_status_publisher = self.bridge._publisher(
            command_topic(self.robot_id, "navigation_status"), "state"
        )
        self._navigation_status_sequence = 0
        self.local_planner = RosLocalPlanner()
        self._navigation_lock = threading.RLock()
        self._active_navigation: dict[str, Any] | None = None
        self._last_local_planner_status = "idle"
        self.bridge._subscription(
            command_topic(self.robot_id, "task_request"), self._on_task_request, "durable"
        )
        self.bridge._subscription(
            sim_topic(self.robot_id, "state"), self._on_sim_state, "state"
        )
        self.bridge._subscription(
            sim_topic(self.robot_id, "lidar_proximity"),
            self._on_lidar_proximity,
            "sensor",
        )
        self.bridge._subscription(
            sim_topic(self.robot_id, "task_result"),
            self._on_sim_task_result,
            "durable",
        )
        self.bridge.node.create_subscription(
            PointCloud2,
            sim_topic(self.robot_id, "mid360/points"),
            self._on_point_cloud,
            self.bridge._qos["sensor"],
        )
        self.bridge.node.create_subscription(
            Imu,
            sim_topic(self.robot_id, "mid360/imu"),
            self._on_imu,
            self.bridge._qos["sensor"],
        )

    @property
    def adapter_connected(self) -> bool:
        return time.time() - self.last_raw_state_at <= 1.5

    @property
    def motion_enabled(self) -> bool:
        state = self.last_raw_state or {}
        return self.adapter_connected and bool(state.get("healthy", False))


    def _on_sim_state(self, raw: str) -> None:
        payload = _decode_json(raw)
        if payload is None or payload.get("robot_id") != self.robot_id:
            return
        self.last_raw_state = payload
        self.last_raw_state_at = time.time()

    def _on_lidar_proximity(self, raw: str) -> None:
        payload = _decode_json(raw)
        if payload is not None:
            self._latest_lidar_proximity = payload

    def _apply_instantaneous_obstacle_layer(self) -> None:
        """Keep current lidar dynamics in the rolling local planning map."""

        now = time.time()
        payload = self._latest_lidar_proximity or {}
        pose = self.local_slam.pose
        try:
            frame_time = float(payload["frame_timestamp"])
            distance = float(payload["nearest_obstacle_distance"])
            bearing = math.radians(float(payload["nearest_obstacle_bearing_deg"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            frame_time = 0.0
            distance = math.inf
            bearing = 0.0
        if (
            pose is not None
            and now - frame_time <= 0.35
            and 0.05 < distance <= 1.50
        ):
            world_bearing = pose.yaw + bearing
            cell = self.local_slam.world_to_cell(
                pose.x + distance * math.cos(world_bearing),
                pose.y + distance * math.sin(world_bearing),
            )
            if cell is not None:
                self._dynamic_obstacles[cell] = now + 1.0
        self._dynamic_obstacles = {
            cell: expiry
            for cell, expiry in self._dynamic_obstacles.items()
            if expiry > now
        }
        for row, column in self._dynamic_obstacles:
            self.local_slam._observed[row, column] = True
            self.local_slam._occupancy_evidence[row, column] = max(
                6,
                int(self.local_slam._occupancy_evidence[row, column]),
            )
            self.local_slam.grid[row, column] = 100


    def _publish_cmd_vel(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
    ) -> None:
        message = self.Twist()
        message.linear.x = float(linear_x)
        message.linear.y = float(linear_y)
        message.angular.z = float(angular_z)
        self._cmd_vel_publisher.publish(message)

    def _publish_navigation_status(
        self,
        active: dict[str, Any],
        status: str,
        *,
        reason: str | None = None,
    ) -> None:
        self._navigation_status_sequence += 1
        pose = self.local_slam.pose
        goal = tuple(float(value) for value in active["goal_xy"])
        distance = (
            math.dist((pose.x, pose.y), goal) if pose is not None else None
        )
        heading_error_deg = (
            math.degrees(
                math.atan2(
                    math.sin(math.atan2(goal[1] - pose.y, goal[0] - pose.x) - pose.yaw),
                    math.cos(math.atan2(goal[1] - pose.y, goal[0] - pose.x) - pose.yaw),
                )
            )
            if pose is not None
            else None
        )
        self.bridge._publish(
            self._navigation_status_publisher,
            {
                "schema_version": 1,
                "sequence": self._navigation_status_sequence,
                "robot_id": self.robot_id,
                "task_id": active["task_id"],
                "goal_xy": list(goal),
                "status": status,
                "terminal": status in {"risk_blocked", "verification_failed"},
                "reason": reason,
                "distance_to_goal_m": distance,
                "heading_error_deg": heading_error_deg,
                "heading_tolerance_deg": 8.0,
                "route_waypoints_completed": (
                    min(
                        int(active.get("route_index", 0)) + 1,
                        len(active.get("route_xy") or ()),
                    )
                    if active.get("route_xy")
                    else 0
                ),
                "route_waypoints_total": len(active.get("route_xy") or ()),
                "safety_hold_count": int(active.get("safety_hold_count", 0)),
                "safety_hold_duration_s": round(
                    float(active.get("safety_hold_duration_s", 0.0)), 3
                ),
                "recovery_attempts": int(active.get("recovery_attempts", 0)),
                "post_hold_recovery_attempts": int(
                    active.get("post_hold_recovery_attempts", 0)
                ),
                "scan_attempts": int(active.get("scan_attempts", 0)),
                "alignment_duration_s": round(
                    float(active.get("alignment_duration_s", 0.0)), 3
                ),
                "alignment_locked": bool(active.get("alignment_locked", False)),
                "alignment_exit_samples": int(
                    active.get("alignment_exit_samples", 0)
                ),
                "navigation_timeout_s": float(
                    active.get("navigation_timeout_s", 240.0)
                ),
                "written_at": time.time(),
            },
        )

    def _start_retrace_recovery(
        self,
        active: dict[str, Any],
        *,
        now: float,
    ) -> bool:
        """Retrace only the latest validated translation, with no recovery yaw."""

        if int(active.get("recovery_attempts", 0)) >= 3:
            return False
        pose = self.local_slam.pose
        last = active.get("last_tracking_command")
        if pose is None or not isinstance(last, tuple) or len(last) != 2:
            return False
        last_x, last_y = (float(value) for value in last)
        norm = math.hypot(last_x, last_y)
        if norm <= 1.0e-4:
            return False
        recovery_x = -0.08 * last_x / norm
        recovery_y = -0.08 * last_y / norm
        local_dx = 0.16 * recovery_x / 0.08
        local_dy = 0.16 * recovery_y / 0.08
        world_dx = math.cos(pose.yaw) * local_dx - math.sin(pose.yaw) * local_dy
        world_dy = math.sin(pose.yaw) * local_dx + math.cos(pose.yaw) * local_dy
        if not self.local_slam.segment_is_traversable(
            (pose.x, pose.y),
            (pose.x + world_dx, pose.y + world_dy),
        ):
            return False
        active["recovery_attempts"] = int(active.get("recovery_attempts", 0)) + 1
        active["recovery_command"] = (recovery_x, recovery_y)
        active["recovery_until"] = now + 2.0
        active["planner_blocked_since"] = None
        return True

    def _start_post_hold_map_refresh(
        self,
        active: dict[str, Any],
        *,
        now: float,
    ) -> bool:
        """Wait at zero for fresh map evidence before scan/reconnection."""

        attempts = int(active.get("post_hold_recovery_attempts", 0))
        if attempts >= 3:
            return False
        active["post_hold_recovery_attempts"] = attempts + 1
        active["post_hold_phase"] = "map_wait"
        active["post_hold_wait_started_at"] = now
        active["post_hold_wait_until"] = now + 1.0
        active["post_hold_map_revision"] = int(self.local_slam.revision)
        active["post_hold_replan_pending"] = False
        active["planner_blocked_since"] = None
        return True

    @staticmethod
    def _start_unknown_scan(active: dict[str, Any], *, now: float) -> bool:
        """Start one bounded in-place scan before translating in unknown space."""

        attempts = int(active.get("scan_attempts", 0))
        if attempts >= 2:
            return False
        active["scan_attempts"] = attempts + 1
        active["scan_yaw_rate"] = 0.45 if attempts % 2 == 0 else -0.45
        active["scan_until"] = now + 3.5
        active["planner_blocked_since"] = None
        return True

    @staticmethod
    def _update_alignment_lock(
        active: dict[str, Any],
        *,
        distance_to_goal_m: float | None,
    ) -> bool:
        """Hold terminal yaw through noise; unlock only on sustained drift."""

        locked = bool(active.get("alignment_locked", False))
        if not locked or distance_to_goal_m is None:
            return locked
        if distance_to_goal_m <= 1.40:
            active["alignment_exit_samples"] = 0
            return True
        samples = int(active.get("alignment_exit_samples", 0)) + 1
        active["alignment_exit_samples"] = samples
        if samples < 3:
            return True
        active["alignment_locked"] = False
        active["alignment_exit_samples"] = 0
        active["alignment_started_at"] = None
        active["alignment_duration_s"] = 0.0
        return False

    def _terminal_navigation_block(
        self,
        active: dict[str, Any],
        reason: str,
    ) -> None:
        active["terminal_status"] = "risk_blocked"
        active["terminal_reason"] = reason
        self._last_local_planner_status = "risk_blocked"
        self._publish_cmd_vel(0.0, 0.0, 0.0)
        self._publish_navigation_status(active, "risk_blocked", reason=reason)

    def _terminal_alignment_failure(
        self,
        active: dict[str, Any],
        reason: str,
    ) -> None:
        active["terminal_status"] = "verification_failed"
        active["terminal_reason"] = reason
        self._last_local_planner_status = "verification_failed"
        self._publish_cmd_vel(0.0, 0.0, 0.0)
        self._publish_navigation_status(
            active,
            "verification_failed",
            reason=reason,
        )

    def _tick_local_navigation(self) -> None:
        with self._navigation_lock:
            active = self._active_navigation
            if active is None:
                return
            terminal_status = active.get("terminal_status")
            if terminal_status in {"risk_blocked", "verification_failed"}:
                self._publish_cmd_vel(0.0, 0.0, 0.0)
                self._publish_navigation_status(
                    active,
                    str(terminal_status),
                    reason=str(
                        active.get("terminal_reason")
                        or "local navigation terminal failure"
                    ),
                )
                return
            if not self.motion_enabled:
                self._publish_cmd_vel(0.0, 0.0, 0.0)
                self._last_local_planner_status = "motion_disabled"
                self._publish_navigation_status(active, "motion_disabled")
                return
            now = time.monotonic()
            safety = (((self.last_raw_state or {}).get("sensors") or {}).get(
                "motion_safety"
            ) or {})
            if safety.get("near_field_blind_hold") or safety.get("external_contact_hold"):
                self._publish_cmd_vel(0.0, 0.0, 0.0)
                self._last_local_planner_status = "safety_hold"
                if active.get("safety_hold_started_at") is None:
                    active["safety_hold_started_at"] = now
                    active["safety_hold_count"] = int(
                        active.get("safety_hold_count", 0)
                    ) + 1
                held_for = now - float(active["safety_hold_started_at"])
                self._publish_navigation_status(active, "safety_hold")
                if safety.get("external_contact_hold") or held_for >= 2.0:
                    active["safety_hold_duration_s"] = float(
                        active.get("safety_hold_duration_s", 0.0)
                    ) + held_for
                    active["safety_hold_started_at"] = None
                    self._terminal_navigation_block(
                        active,
                        "external contact hold" if safety.get("external_contact_hold")
                        else "near-field safety hold did not clear within 2 seconds",
                    )
                return
            hold_started = active.get("safety_hold_started_at")
            if hold_started is not None:
                active["safety_hold_duration_s"] = float(
                    active.get("safety_hold_duration_s", 0.0)
                ) + now - float(hold_started)
                active["safety_hold_started_at"] = None
                if active.get("alignment_locked"):
                    # A pure-yaw terminal pose sweeps no new circular
                    # footprint. Resume alignment after a transient hold;
                    # translation retrace would reintroduce endpoint jitter.
                    self._publish_cmd_vel(0.0, 0.0, 0.0)
                    self._publish_navigation_status(active, "aligning")
                    return
                if not self._start_retrace_recovery(active, now=now):
                    if not self._start_post_hold_map_refresh(active, now=now):
                        self._terminal_navigation_block(
                            active,
                            "transient hold recovery exhausted without a map-valid retrace or reconnectable route",
                        )
                        return
            if active.get("post_hold_phase") == "map_wait":
                wait_started = float(active["post_hold_wait_started_at"])
                wait_until = float(active["post_hold_wait_until"])
                newer_map = self.local_slam.revision > int(
                    active["post_hold_map_revision"]
                )
                enough_evidence_time = now - wait_started >= 0.35
                self._publish_cmd_vel(0.0, 0.0, 0.0)
                self._last_local_planner_status = "recovery_waiting"
                self._publish_navigation_status(active, "recovery_waiting")
                if not (
                    now >= wait_until or newer_map and enough_evidence_time
                ):
                    return
                active["post_hold_phase"] = None
                active["post_hold_replan_pending"] = True
                if self._start_unknown_scan(active, now=now):
                    self._last_local_planner_status = "scanning"
                    self._publish_navigation_status(active, "scanning")
                    return
            recovery_until = float(active.get("recovery_until", 0.0))
            if now < recovery_until:
                recovery_x, recovery_y = active["recovery_command"]
                self._publish_cmd_vel(recovery_x, recovery_y, 0.0)
                self._last_local_planner_status = "recovering"
                self._publish_navigation_status(active, "recovering")
                return
            if recovery_until:
                active["recovery_until"] = 0.0
                self._publish_cmd_vel(0.0, 0.0, 0.0)
                self._publish_navigation_status(active, "tracking")
                return
            scan_until = float(active.get("scan_until", 0.0))
            if now < scan_until:
                self._publish_cmd_vel(
                    0.0,
                    0.0,
                    float(active["scan_yaw_rate"]),
                )
                self._last_local_planner_status = "scanning"
                self._publish_navigation_status(active, "scanning")
                return
            if scan_until:
                active["scan_until"] = 0.0
                self._publish_cmd_vel(0.0, 0.0, 0.0)
                self._publish_navigation_status(active, "tracking")
                return
            goal = tuple(float(value) for value in active["goal_xy"])
            route = tuple(
                tuple(float(value) for value in point)
                for point in active.get("route_xy") or ()
            )
            pose = self.local_slam.pose
            distance_to_goal = (
                math.dist((pose.x, pose.y), goal) if pose is not None else None
            )
            alignment_locked = self._update_alignment_lock(
                active,
                distance_to_goal_m=distance_to_goal,
            )
            command = self.local_planner.command(
                self.local_slam,
                goal,
                route_xy=route,
                route_waypoint_index=int(active.get("route_index", 0)),
                alignment_locked=alignment_locked,
                semantic_obstacles=self.semantic_map.target_obstacle_regions(),
            )
            if command.route_waypoint_index is not None:
                active["route_index"] = max(
                    int(active.get("route_index", 0)),
                    command.route_waypoint_index,
                )
            self._last_local_planner_status = command.status
            if command.status in {"aligning", "arrived"}:
                active["alignment_locked"] = True
                active["alignment_exit_samples"] = 0
            if command.status == "aligning":
                alignment_started_at = active.get("alignment_started_at")
                if alignment_started_at is None:
                    active["alignment_started_at"] = now
                else:
                    alignment_duration = now - float(alignment_started_at)
                    active["alignment_duration_s"] = alignment_duration
                    if alignment_duration >= 12.0:
                        self._terminal_alignment_failure(
                            active,
                            "target heading did not converge within 12 seconds",
                        )
                        return
            elif command.status == "tracking" and not active.get(
                "alignment_locked"
            ):
                # Leaving the standoff envelope starts a fresh terminal
                # alignment phase after positional tracking recovers.
                active["alignment_started_at"] = None
                active["alignment_duration_s"] = 0.0
            if command.status in {"path_unavailable", "path_invalidated"}:
                if active.get("post_hold_replan_pending"):
                    if self._start_unknown_scan(active, now=now):
                        self._publish_cmd_vel(0.0, 0.0, 0.0)
                        self._last_local_planner_status = "scanning"
                        self._publish_navigation_status(active, "scanning")
                        return
                    self._terminal_navigation_block(
                        active,
                        "post-hold map refresh and two in-place scans could not reconnect the semantic route",
                    )
                    return
                blocked_since = active.get("planner_blocked_since")
                if blocked_since is None:
                    active["planner_blocked_since"] = now
                else:
                    blocked_for = now - float(blocked_since)
                    scanning_started = bool(
                        command.status == "path_unavailable"
                        and blocked_for >= 2.0
                        and self._start_unknown_scan(active, now=now)
                    )
                    recovery_due = bool(
                        command.status == "path_invalidated"
                        and blocked_for >= 3.0
                        or command.status == "path_unavailable"
                        and blocked_for >= 6.0
                    )
                    if scanning_started:
                        self._publish_cmd_vel(0.0, 0.0, 0.0)
                        self._last_local_planner_status = "scanning"
                        self._publish_navigation_status(active, "scanning")
                        return
                    if recovery_due and not self._start_retrace_recovery(
                        active, now=now
                    ):
                        self._terminal_navigation_block(
                            active,
                            f"{command.status} persisted after scan/reconnect and bounded recovery was unavailable",
                        )
                        return
            else:
                active["planner_blocked_since"] = None
                if command.status in {"tracking", "aligning", "arrived"}:
                    active["post_hold_replan_pending"] = False
                    active["post_hold_phase"] = None
            if math.hypot(command.linear_x, command.linear_y) > 1.0e-4:
                active["last_tracking_command"] = (
                    command.linear_x,
                    command.linear_y,
                )
            self._publish_cmd_vel(
                command.linear_x,
                command.linear_y,
                command.angular_z,
            )
            self._publish_navigation_status(active, command.status)

    def _on_imu(self, message: Any) -> None:
        sample = (
            (
                float(message.angular_velocity.x),
                float(message.angular_velocity.y),
                float(message.angular_velocity.z),
            ),
            (
                float(message.linear_acceleration.x),
                float(message.linear_acceleration.y),
                float(message.linear_acceleration.z),
            ),
        )
        with self._sensor_lock:
            self._imu_samples.append(sample)

    def _on_point_cloud(self, message: Any) -> None:
        if int(message.point_step) != 12 or len(message.data) % 12:
            return
        points = np.frombuffer(message.data, dtype="<f4").reshape((-1, 3)).copy()
        sequence = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        with self._sensor_lock:
            self._pending_cloud = (sequence, points)

    def _process_sensor_frame(self) -> None:
        with self._sensor_lock:
            pending = self._pending_cloud
            if pending is None:
                return
            self._pending_cloud = None
            samples = list(self._imu_samples)
            self._imu_samples.clear()
        sequence, points = pending
        fallback = samples[-1] if samples else (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 9.80665),
        )
        try:
            estimate = self.fastlio.observe(
                points,
                scan_start_s=1.0 + self._sensor_frames * 0.10,
                gyro_xyz=fallback[0],
                acceleration_xyz=fallback[1],
                imu_samples=samples or [fallback] * 6,
            )
            self._sensor_frames += 1
            self.last_fastlio_error = None
        except Exception as error:  # noqa: BLE001 - status fences motion on failure
            self.last_fastlio_error = str(error)[:500]
            return
        if not estimate.ready or estimate.pose is None:
            return
        registered = estimate.registered_points
        if len(registered):
            registered = registered[
                (registered[:, 2] > -0.24) & (registered[:, 2] < 1.20)
            ]
        self.local_slam.observe_fastlio(registered, pose=estimate.pose)

    def _local_map_payload(self) -> dict[str, Any] | None:
        if self.local_slam.revision <= 0:
            return None
        grid = self.local_slam.grid
        encoded = np.where(grid < 0, 255, grid).astype(np.uint8)
        details = self.local_slam.payload()
        return {
            "schema_version": 1,
            "available": True,
            "running": True,
            "source": "robot_container_fastlio2",
            "robot_id": self.robot_id,
            "timestamp": time.time(),
            "revision": self.local_slam.revision,
            "frame_id": "go2_map",
            "width": self.local_slam.width,
            "height": self.local_slam.height,
            "resolution": self.local_slam.resolution,
            "origin": {
                "x": self.local_slam.origin_x,
                "y": self.local_slam.origin_y,
                "yaw": 0.0,
            },
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "cells": {
                "total": int(grid.size),
                "known": details["known"],
                "free": details["free"],
                "occupied": details["occupied"],
            },
            "estimator": self.fastlio.status(),
        }

    def _publish_observation(self) -> None:
        raw = self.last_raw_state
        if raw is None:
            return
        state = dict(raw)
        state["robot_id"] = self.robot_id
        state["adapter_connected"] = self.adapter_connected
        state["agent_boot_epoch"] = self.bridge.boot_epoch
        state["fastlio2"] = self.fastlio.status()
        if self.local_slam.pose is not None:
            state["pose"] = {
                "position": [
                    self.local_slam.pose.x,
                    self.local_slam.pose.y,
                    float((raw.get("pose") or {}).get("position", [0, 0, 0.3])[2]),
                ],
                "yaw": self.local_slam.pose.yaw,
                "source": "container_fastlio2",
            }
        state_message = self.bridge.publish("robot_state", state)
        local_map = self._local_map_payload()
        if (
            local_map is not None
            and int(local_map["revision"]) > self._last_local_map_revision
        ):
            map_message = self.bridge.publish("local_map", local_map)
            self._last_local_map_revision = int(local_map["revision"])







    def _on_task_request(self, raw: str) -> None:
        from harness.robots.go2.go2_instructions import parse_instruction
        payload = _decode_json(raw)
        if not payload or payload.get('robot_id') != self.robot_id:
            return
        try:
            sequence = int(payload['sequence'])
            instruction = str(payload['instruction'])
            parsed = parse_instruction(instruction)
            age = time.time() - float(payload['written_at'])
        except (KeyError, TypeError, ValueError):
            return
        if (sequence <= self._last_task_sequence or not 0 <= age <= 2.0
                or not self.motion_enabled or self._active_task_id is not None):
            return
        self._last_task_sequence = sequence
        task_id = str(sequence)
        self._active_task_id = task_id
        # A single robot may use its configured semantic map; no task assignment occurs.
        target_names = {'red_cube': 'red-cube', 'blue_ball': 'blue-ball', 'bottle': 'water-bottle'}
        goal_xy = None
        route_xy = []
        if parsed.action == 'search':
            from harness.robots.go2.go2_semantic_map import GO2_SEMANTIC_TARGETS
            goal_xy = GO2_SEMANTIC_TARGETS[target_names[parsed.target]]
            pose = self.local_slam.pose
            route = self.semantic_map.route((pose.x, pose.y), goal_xy) if pose else None
            if route:
                selected = list(route.points[::5])
                if not selected or selected[-1] != route.points[-1]:
                    selected.append(route.points[-1])
                route_xy = [[float(x), float(y)] for x, y in selected]
        route_tuple = tuple(tuple(p) for p in route_xy)
        timeout_s = coordinate_navigation_timeout_s(route_tuple)
        command = {'schema_version': 1, 'robot_id': self.robot_id, 'boot_epoch': self.bridge.boot_epoch,
                   'request_id': task_id, 'sequence': sequence, 'task_id': task_id,
                   'instruction': instruction, 'goal_xy': goal_xy, 'route_xy': route_xy,
                   'navigation_timeout_s': timeout_s,
                   'control_mode': 'ros_local_planner' if goal_xy is not None else 'simulator_skill',
                   'expires_at': time.time() + 2.0, 'written_at': time.time()}
        self.bridge._publish(self._task_command_publisher, command)
        if goal_xy is not None:
            with self._navigation_lock:
                self._active_navigation = {
                    'task_id': task_id, 'goal_xy': tuple(goal_xy), 'route_xy': route_tuple,
                    'route_index': 0, 'started_at': time.time(),
                    'safety_hold_count': 0, 'safety_hold_duration_s': 0.0, 'safety_hold_started_at': None,
                    'recovery_attempts': 0, 'recovery_until': 0.0,
                    'post_hold_recovery_attempts': 0, 'post_hold_phase': None, 'post_hold_replan_pending': False,
                    'scan_attempts': 0, 'scan_until': 0.0,
                    'alignment_started_at': None, 'alignment_duration_s': 0.0,
                    'alignment_locked': False, 'alignment_exit_samples': 0,
                    'navigation_timeout_s': timeout_s, 'planner_blocked_since': None,
                }

    def _on_sim_task_result(self, raw: str) -> None:
        payload = _decode_json(raw)
        if (not payload or payload.get('robot_id') != self.robot_id
                or str(payload.get('task_id')) != self._active_task_id):
            return
        result = payload.get('result')
        if not isinstance(result, dict):
            return
        with self._navigation_lock:
            self._publish_cmd_vel(0.0, 0.0, 0.0)
            self._active_navigation = None
            self._active_task_id = None
            self._last_local_planner_status = str(result.get('task_status', 'incomplete'))
        self.last_task_result = payload
        self.bridge.publish('task_result', payload)

    def _publish_heartbeat(self) -> None:
        self.bridge.publish('heartbeat', {'robot_id': self.robot_id, 'adapter_connected': self.adapter_connected,
                                          'motion_enabled': self.motion_enabled, 'fastlio2': self.fastlio.status()})

    def _status(self) -> dict[str, Any]:
        return {'schema_version': 2, 'robot_id': self.robot_id, 'boot_epoch': self.bridge.boot_epoch,
                'ros_distro': os.environ.get('ROS_DISTRO'), 'ros_domain_id': os.environ.get('ROS_DOMAIN_ID', '0'),
                'adapter_connected': self.adapter_connected, 'motion_enabled': self.motion_enabled,
                'fastlio2': {**self.fastlio.status(), 'sensor_frames': self._sensor_frames, 'error': self.last_fastlio_error},
                'last_task_result': self.last_task_result, 'active_task_id': self._active_task_id,
                'local_planner': {'active': self._active_navigation is not None, 'status': self._last_local_planner_status},
                'written_at': time.time(), 'started_at': self.started_at}

    def _write_status(self) -> None:
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._status(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self.status_path)

    def run(self) -> None:
        print(
            json.dumps(
                {
                    "event": "go2_ros_agent_ready",
                    "robot_id": self.robot_id,
                    "ros_distro": os.environ.get("ROS_DISTRO"),
                    "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
                    "adapter_connected": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        next_observation = next_heartbeat = 0.0
        try:
            while not self.stop_requested.wait(0.05):
                now = time.monotonic()
                self._process_sensor_frame()
                self._apply_instantaneous_obstacle_layer()
                if now >= next_observation:
                    self._publish_observation()
                    next_observation = now + 0.20
                self._tick_local_navigation()
                if now >= next_heartbeat:
                    self._publish_heartbeat()
                    self._write_status()
                    next_heartbeat = now + 0.20
        finally:
            self.fastlio.close()
            self.bridge.close()

    def stop(self, *_args: Any) -> None:
        self.stop_requested.set()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-id",
        default=os.environ.get("LUXI_ROBOT_ID", ""),
        choices=("go2-01",),
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path("/tmp/luxi-go2-ros-status.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    node = Go2RosRobotNode(args.robot_id, status_path=args.status_path)
    signal.signal(signal.SIGTERM, node.stop)
    signal.signal(signal.SIGINT, node.stop)
    node.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
