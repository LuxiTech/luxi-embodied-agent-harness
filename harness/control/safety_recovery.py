"""Fail-closed local recovery from critical current-lidar proximity.

This controller sits below the language agent.  It never invents an escape
direction: recovery is permitted only by retracing recent odometry samples
whose footprint corridor is still free in a fresh online costmap. Scene truth,
third-person imagery, and model guesses are never inputs. Otherwise the robot
remains stopped with the planner locked out.
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import threading
import time
from typing import Any, Callable
from harness.robots.g1.safety_geometry import FOOTPRINT_RADIUS_M, RECOVERY_MARGIN_M


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class RecoveryConfig:
    poll_interval_seconds: float = 0.10
    recovery_speed_mps: float = 0.06
    safe_clearance_m: float = 0.55
    footprint_radius_m: float = FOOTPRINT_RADIUS_M
    footprint_margin_m: float = RECOVERY_MARGIN_M
    center_unknown_allowance_m: float = 0.10
    breadcrumb_spacing_m: float = 0.05
    max_breadcrumb_age_seconds: float = 30.0
    max_join_distance_m: float = 0.30
    max_retreat_distance_m: float = 1.20
    min_retreat_distance_m: float = 0.15
    waypoint_tolerance_m: float = 0.045
    max_costmap_age_seconds: float = 0.75
    costmap_wait_timeout_seconds: float = 2.0
    max_pose_costmap_skew_seconds: float = 3.0
    stationary_speed_mps: float = 0.025
    stationary_samples: int = 2
    stop_confirmation_timeout_seconds: float = 2.0
    task_stop_barrier_timeout_seconds: float = 75.0
    max_recovery_seconds: float = 8.0
    progress_timeout_seconds: float = 1.5
    warning_approach_guard: bool = False
    warning_approach_min_projected_speed: float = 0.15
    turning_creep_min_yaw_rate: float = 0.45
    turning_creep_max_planar_speed: float = 0.25


@dataclass(frozen=True)
class RecoveryPose:
    x: float
    y: float
    yaw: float
    timestamp: float


@dataclass(frozen=True)
class Breadcrumb:
    x: float
    y: float
    recorded_at: float
    pose_timestamp: float
    safe_destination: bool
    clearance_m: float | None
    view_quality: str


class LiveCostmapEvidence:
    """Small immutable view used only for local path safety checks."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        origin_yaw: float,
        cells: bytes,
        center_unknown_allowance_m: float,
        blocked_cost_threshold: int = 50,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_yaw = origin_yaw
        self.cells = cells
        self.center_unknown_allowance_m = center_unknown_allowance_m
        self.blocked_cost_threshold = blocked_cost_threshold

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        pose_timestamp: float,
        config: RecoveryConfig,
    ) -> LiveCostmapEvidence | None:
        try:
            source = payload.get("source")
            accepted_source = bool(
                source == "live"
                or (
                    source == "isaac_lidar_live"
                    and payload.get("recovery_eligible") is True
                    and payload.get("planning_scope")
                    in {"retrace_only", "explored_same_floor"}
                    and payload.get("identity_verified") is True
                    and payload.get("unknown_is_blocked") is True
                )
            )
            if (
                not payload.get("available")
                or not accepted_source
                or payload.get("frame_id") != "world"
            ):
                return None
            age = float(payload["age_seconds"])
            timestamp = float(payload["timestamp"])
            width = int(payload["width"])
            height = int(payload["height"])
            resolution = float(payload["resolution"])
            origin = payload["origin"]
            origin_x = float(origin["x"])
            origin_y = float(origin["y"])
            origin_yaw = float(origin.get("yaw", 0.0))
            cells = base64.b64decode(payload["data"], validate=True)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        values = (
            age,
            timestamp,
            pose_timestamp,
            resolution,
            origin_x,
            origin_y,
            origin_yaw,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or age < 0.0
            or age > config.max_costmap_age_seconds
            or abs(timestamp - pose_timestamp) > config.max_pose_costmap_skew_seconds
            or width <= 0
            or height <= 0
            or resolution <= 0.0
            or len(cells) != width * height
        ):
            return None
        return cls(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_yaw=origin_yaw,
            cells=cells,
            center_unknown_allowance_m=config.center_unknown_allowance_m,
            # Isaac's display grid already contains a 0.30 m value-60
            # inflation layer. Recovery applies the G1 footprint below, so
            # treating that layer as another solid obstacle would double
            # inflate it. Value 100 remains the measured RTX endpoint.
            blocked_cost_threshold=100 if source == "isaac_lidar_live" else 50,
        )

    def _local(self, x: float, y: float) -> tuple[float, float]:
        dx = x - self.origin_x
        dy = y - self.origin_y
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        return cosine * dx + sine * dy, -sine * dx + cosine * dy

    def footprint_is_clear(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        allow_traversed_unknown: bool = False,
    ) -> bool:
        local_x, local_y = self._local(x, y)
        if not (
            0.0 <= local_x < self.width * self.resolution
            and 0.0 <= local_y < self.height * self.resolution
        ):
            return False
        center_column = math.floor(local_x / self.resolution)
        center_row = math.floor(local_y / self.resolution)
        radius_cells = math.ceil(radius / self.resolution) + 1
        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for column in range(
                center_column - radius_cells,
                center_column + radius_cells + 1,
            ):
                if not (0 <= row < self.height and 0 <= column < self.width):
                    return False
                cell_x = (column + 0.5) * self.resolution
                cell_y = (row + 0.5) * self.resolution
                distance = math.hypot(cell_x - local_x, cell_y - local_y)
                if distance > radius + self.resolution * math.sqrt(0.5):
                    continue
                value = self.cells[row * self.width + column]
                if self.blocked_cost_threshold <= value <= 100:
                    return False
                if (
                    value == 255
                    and not allow_traversed_unknown
                    and distance > self.center_unknown_allowance_m
                ):
                    return False
        return True

    def segment_is_clear(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        radius: float,
        *,
        allow_traversed_unknown: bool = False,
    ) -> bool:
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, math.ceil(distance / max(0.01, self.resolution * 0.5)))
        for index in range(steps + 1):
            fraction = index / steps
            x = start[0] + (end[0] - start[0]) * fraction
            y = start[1] + (end[1] - start[1]) * fraction
            if not self.footprint_is_clear(
                x,
                y,
                radius,
                allow_traversed_unknown=allow_traversed_unknown,
            ):
                return False
        return True


class CriticalRecoveryController:
    """Stop and retrace to the nearest proven noncritical, observable pose."""

    ACTIVE_STATES = {"stopping", "retreating", "settling"}

    def __init__(
        self,
        events: Any,
        *,
        observation_provider: Callable[[], dict[str, Any]],
        costmap_provider: Callable[[], dict[str, Any]],
        force_stop: Callable[[], None],
        cancel_agent: Callable[[], None],
        begin_hold: Callable[[], object],
        publish_recovery: Callable[[object, float, float], bool],
        end_hold: Callable[[object], bool],
        on_recovery_ready: Callable[[], None] | None = None,
        motion_handoff_ready: Callable[[], bool] | None = None,
        enabled: bool = True,
        config: RecoveryConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.events = events
        self.observation_provider = observation_provider
        self.costmap_provider = costmap_provider
        self.force_stop = force_stop
        self.cancel_agent = cancel_agent
        self.begin_hold = begin_hold
        self.publish_recovery = publish_recovery
        self.end_hold = end_hold
        self.on_recovery_ready = on_recovery_ready
        self.motion_handoff_ready = motion_handoff_ready or (lambda: True)
        self._route_diagnostic: dict[str, Any] = {}
        self._route_end_recorded_at = None
        self.enabled = bool(enabled)
        self.config = config or RecoveryConfig()
        self.clock = clock
        self.wall_clock = wall_clock

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._breadcrumbs: deque[Breadcrumb] = deque(maxlen=600)
        self._state = "idle"
        self._reason = ""
        self._last_outcome = ""
        self._hold_token: object | None = None
        self._route: deque[Breadcrumb] = deque()
        self._started_at: float | None = None
        self._stop_started_at: float | None = None
        self._stop_command_completed_at: float | None = None
        self._stationary_confirmed_at: float | None = None
        self._recovery_stop_command_completed_at: float | None = None
        self._recovery_stationary_confirmed_at: float | None = None
        self._recovery_stop_started_at: float | None = None
        self._stationary_reference: RecoveryPose | None = None
        self._stationary_count = 0
        self._retreat_distance_m = 0.0
        self._last_motion_pose: RecoveryPose | None = None
        self._best_waypoint_distance: float | None = None
        self._last_progress_at: float | None = None
        self._last_clearance: float | None = None
        self._clearance_regressions = 0
        self._forward_view_quality = "unknown"
        self._costmap_wait_started_at: float | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="critical-safety-recovery",
                daemon=True,
            )
            self._thread.start()
        self.events.append(
            "safety",
            "lifecycle",
            "Critical recovery controller ready",
            (
                "Critical proximity will stop navigation and retrace only a fresh, "
                f"costmap-verified path at ≤{self.config.recovery_speed_mps:.2f} m/s."
            ),
        )

    def _run(self) -> None:
        while not self._stop.wait(self.config.poll_interval_seconds):
            try:
                self.step()
            except Exception as error:  # noqa: BLE001 - safety must fail closed
                self._hold("controller_error", str(error)[:500])

    @staticmethod
    def _pose(observation: dict[str, Any]) -> RecoveryPose | None:
        raw = observation.get("pose")
        if not isinstance(raw, dict):
            return None
        try:
            pose = RecoveryPose(
                x=float(raw["x"]),
                y=float(raw["y"]),
                yaw=float(raw["yaw"]),
                timestamp=float(raw["timestamp"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return pose if all(math.isfinite(value) for value in pose.__dict__.values()) else None

    @staticmethod
    def _risk(observation: dict[str, Any]) -> tuple[str, float | None]:
        metrics = observation.get("metrics")
        if not isinstance(metrics, dict):
            return "unknown", None
        risk = str(metrics.get("risk", "unknown")).lower()
        if risk not in {"clear", "warning", "critical", "unknown"}:
            risk = "unknown"
        raw_clearance = metrics.get("nearest_obstacle_distance")
        clearance = None
        if isinstance(raw_clearance, (int, float)) and not isinstance(raw_clearance, bool):
            value = float(raw_clearance)
            if math.isfinite(value) and value >= 0.0:
                clearance = value
        return risk, clearance

    @staticmethod
    def _view_quality(observation: dict[str, Any]) -> str:
        if observation.get("camera_available") is not True:
            return "unavailable"
        metrics = observation.get("metrics")
        if not isinstance(metrics, dict):
            return "unknown"
        quality = str(metrics.get("forward_view_quality", "unknown")).lower()
        return quality if quality in {"good", "limited", "unknown"} else "unknown"

    def _approaching_nearest_obstacle(self, observation: dict[str, Any]) -> bool:
        """Return true only when body-frame translation points at the hazard."""

        command = observation.get("command")
        metrics = observation.get("metrics")
        if (
            not isinstance(command, (list, tuple))
            or len(command) < 2
            or not isinstance(metrics, dict)
        ):
            return False
        cell = metrics.get("nearest_obstacle_cell")
        if not isinstance(cell, dict):
            return False
        try:
            command_x = float(command[0])
            command_y = float(command[1])
            command_yaw = float(command[5]) if len(command) > 5 else 0.0
            bearing = math.radians(float(cell["bearing_deg"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not all(
            math.isfinite(value)
            for value in (command_x, command_y, command_yaw, bearing)
        ):
            return False
        # Isaac's planner implements otherwise ineffective pure yaw with a
        # bounded alternating stepping creep.  It remains protected by the
        # critical envelope, but treating that turn-in-place gait as ordinary
        # forward travel makes every return-path initial rotation self-cancel.
        if (
            math.hypot(command_x, command_y)
            <= self.config.turning_creep_max_planar_speed
            and abs(command_yaw) >= self.config.turning_creep_min_yaw_rate
        ):
            return False
        projected_speed = (
            command_x * math.cos(bearing) + command_y * math.sin(bearing)
        )
        return projected_speed > self.config.warning_approach_min_projected_speed

    def _grid(self, pose: RecoveryPose) -> LiveCostmapEvidence | None:
        return LiveCostmapEvidence.from_payload(
            self.costmap_provider(),
            pose_timestamp=pose.timestamp,
            config=self.config,
        )

    @property
    def _footprint_clearance(self) -> float:
        return self.config.footprint_radius_m + self.config.footprint_margin_m

    def _record_breadcrumb(
        self,
        pose: RecoveryPose,
        risk: str,
        clearance: float | None,
        view_quality: str,
    ) -> None:
        if risk not in {"clear", "warning"}:
            return
        grid = self._grid(pose)
        if grid is None or not grid.footprint_is_clear(
            pose.x,
            pose.y,
            self._footprint_clearance,
            allow_traversed_unknown=True,
        ):
            return
        safe_destination = bool(
            risk in {"clear", "warning"}
            and (clearance is None or clearance >= self.config.safe_clearance_m)
            and view_quality == "good"
        )
        point = Breadcrumb(
            x=pose.x,
            y=pose.y,
            recorded_at=self.clock(),
            pose_timestamp=pose.timestamp,
            safe_destination=safe_destination,
            clearance_m=clearance,
            view_quality=view_quality,
        )
        with self._lock:
            cutoff = self.clock() - self.config.max_breadcrumb_age_seconds
            while self._breadcrumbs and self._breadcrumbs[0].recorded_at < cutoff:
                self._breadcrumbs.popleft()
            if self._breadcrumbs:
                latest = self._breadcrumbs[-1]
                separation = math.hypot(point.x - latest.x, point.y - latest.y)
                if point.pose_timestamp == latest.pose_timestamp:
                    return
                if (
                    point.pose_timestamp < latest.pose_timestamp
                    or separation > self.config.max_join_distance_m
                ):
                    self._breadcrumbs.clear()
                    self._breadcrumbs.append(point)
                    return
                if separation < self.config.breadcrumb_spacing_m:
                    if point.safe_destination:
                        self._breadcrumbs[-1] = Breadcrumb(
                            x=latest.x,
                            y=latest.y,
                            recorded_at=point.recorded_at,
                            pose_timestamp=point.pose_timestamp,
                            safe_destination=True,
                            clearance_m=point.clearance_m,
                            view_quality=point.view_quality,
                        )
                    return
            self._breadcrumbs.append(point)

    def _select_route(
        self,
        pose: RecoveryPose,
        grid: LiveCostmapEvidence,
        *, before_recorded_at: float | None = None,
    ) -> deque[Breadcrumb] | None:
        now = self.clock()
        self._route_diagnostic = {"reason": "no_eligible_destination"}
        with self._lock:
            candidates = [p for p in reversed(self._breadcrumbs)
                          if before_recorded_at is None or p.recorded_at < before_recorded_at]
        if not candidates:
            self._route_diagnostic = {"reason": "no_breadcrumbs"}
            return None
        if math.hypot(candidates[0].x - pose.x, candidates[0].y - pose.y) > (
            self.config.max_join_distance_m
        ):
            self._route_diagnostic = {"reason": "breadcrumb_join_too_far"}
            return None

        route: deque[Breadcrumb] = deque()
        previous = (pose.x, pose.y)
        distance = 0.0
        for point in candidates:
            if now - point.recorded_at > self.config.max_breadcrumb_age_seconds:
                break
            segment_length = math.hypot(point.x - previous[0], point.y - previous[1])
            if segment_length < self.config.breadcrumb_spacing_m * 0.4:
                if point.safe_destination and distance >= self.config.min_retreat_distance_m:
                    self._route_end_recorded_at = point.recorded_at
                    self._route_diagnostic = {"reason": "verified", "distance_m": distance}
                    return route
                continue
            distance += segment_length
            if distance > self.config.max_retreat_distance_m:
                break
            if not grid.segment_is_clear(
                previous,
                (point.x, point.y),
                self._footprint_clearance,
                allow_traversed_unknown=True,
            ):
                self._route_diagnostic = {"reason": "segment_blocked", "start": list(previous),
                                          "end": [point.x, point.y], "radius_m": self._footprint_clearance}
                return None
            route.append(point)
            previous = (point.x, point.y)
            if point.safe_destination and distance >= self.config.min_retreat_distance_m:
                self._route_diagnostic = {"reason": "verified", "distance_m": distance}
                self._route_end_recorded_at = point.recorded_at
                return route
        return None

    def _heartbeat_zero(self) -> bool:
        with self._lock:
            token = self._hold_token
        return token is not None and self.publish_recovery(token, 0.0, 0.0)

    def _wait_for_fresh_costmap(self) -> None:
        """Hold zero through a short map publication gap before failing closed."""

        self._heartbeat_zero()
        now = self.clock()
        with self._lock:
            if self._costmap_wait_started_at is None:
                self._costmap_wait_started_at = now
            waited = now - self._costmap_wait_started_at
            self._reason = "waiting_for_fresh_costmap"
        if waited > self.config.costmap_wait_timeout_seconds:
            self._hold("live_costmap_unavailable_timeout")

    def _costmap_is_fresh_again(self, reason: str) -> None:
        with self._lock:
            if self._costmap_wait_started_at is not None:
                # A sensor pause is not failed motion progress.  Give the next
                # verified waypoint a full progress interval after evidence
                # returns.
                self._last_progress_at = self.clock()
            self._costmap_wait_started_at = None
            self._reason = reason

    def _begin_critical(
        self,
        pose: RecoveryPose,
        clearance: float | None,
        *,
        reason: str = "critical_proximity",
    ) -> None:
        self.force_stop()
        token = self.begin_hold()
        stop_completed = self.wall_clock()
        with self._lock:
            self._hold_token = token
            self._state = "stopping"
            self._reason = reason
            self._last_outcome = ""
            self._route.clear()
            self._started_at = self.clock()
            self._stop_started_at = self.clock()
            self._stop_command_completed_at = stop_completed
            self._stationary_confirmed_at = None
            self._recovery_stop_command_completed_at = None
            self._recovery_stationary_confirmed_at = None
            self._recovery_stop_started_at = None
            self._stationary_reference = pose
            self._stationary_count = 0
            self._retreat_distance_m = 0.0
            self._last_motion_pose = pose
            self._best_waypoint_distance = None
            self._last_progress_at = self.clock()
            self._last_clearance = clearance
            self._clearance_regressions = 0
            self._forward_view_quality = "unknown"
            self._costmap_wait_started_at = None
        try:
            self.cancel_agent()
        except Exception as error:  # noqa: BLE001 - the bridge hold is already active
            self.events.append(
                "safety",
                "error",
                "Agent cancellation reported an error",
                str(error)[:500],
                level="warning",
            )
        self.events.append(
            "safety",
            "recovery",
            (
                "Approach guard: motion stopped"
                if reason == "warning_approach_guard"
                else "Critical proximity: motion stopped"
            ),
            (
                "The translating G1 entered the warning envelope while moving "
                "toward the nearest obstacle; old control is locked."
                if reason == "warning_approach_guard"
                else "Old navigation is locked; waiting for two fresh stationary odometry samples."
            ),
            level="danger",
            data={"clearance_m": clearance},
        )

    def _stationary_sample(self, pose: RecoveryPose, after_epoch: float) -> bool:
        with self._lock:
            reference = self._stationary_reference
            if (
                reference is None
                or pose.timestamp <= reference.timestamp
                or pose.timestamp <= after_epoch
            ):
                return False
            elapsed = pose.timestamp - reference.timestamp
            speed = math.hypot(pose.x - reference.x, pose.y - reference.y) / elapsed
            self._stationary_reference = pose
            if speed <= self.config.stationary_speed_mps:
                self._stationary_count += 1
            else:
                self._stationary_count = 0
            return self._stationary_count >= self.config.stationary_samples

    def _handle_stopping(self, pose: RecoveryPose) -> None:
        self._heartbeat_zero()
        with self._lock:
            stop_completed = self._stop_command_completed_at
            stop_started = self._stop_started_at
        if stop_completed is None or stop_started is None:
            self._hold("invalid_stop_state")
            return
        if self._stationary_sample(pose, stop_completed):
            with self._lock:
                self._stationary_confirmed_at = self.wall_clock()
            if not self.motion_handoff_ready():
                if self.clock() - stop_started > self.config.task_stop_barrier_timeout_seconds:
                    self._hold("task_stop_barrier_timeout")
                else:
                    with self._lock:
                        self._reason = "waiting_for_task_stop_barrier"
                return
            grid = self._grid(pose)
            if grid is None:
                self._wait_for_fresh_costmap()
                return
            self._costmap_is_fresh_again("critical_proximity")
            route = self._select_route(pose, grid)
            if route is None or not route:
                self._hold("no_fresh_verified_retreat_path")
                return
            with self._lock:
                self._state = "retreating"
                self._started_at = self.clock()
                self._reason = "retracing_verified_path"
                self._route = route
                self._last_progress_at = self.clock()
                self._best_waypoint_distance = None
            self.events.append(
                "safety",
                "recovery",
                "Stationary confirmed; safe retrace started",
                f"Following {len(route)} recent odometry waypoints at low speed.",
                level="warning",
            )
            return
        if self.clock() - stop_started > self.config.stop_confirmation_timeout_seconds:
            self._hold("stationary_not_confirmed")

    def _start_settling(self, pose: RecoveryPose) -> None:
        if not self._heartbeat_zero():
            self._hold("recovery_channel_unavailable")
            return
        with self._lock:
            self._state = "settling"
            self._reason = "confirming_recovery_stop"
            self._recovery_stop_command_completed_at = self.wall_clock()
            self._recovery_stop_started_at = self.clock()
            self._stationary_reference = pose
            self._stationary_count = 0

    def _update_retreat_distance(self, pose: RecoveryPose) -> None:
        with self._lock:
            previous = self._last_motion_pose
            if previous is None or pose.timestamp <= previous.timestamp:
                return
            self._retreat_distance_m += math.hypot(
                pose.x - previous.x,
                pose.y - previous.y,
            )
            self._last_motion_pose = pose

    def _handle_retreating(
        self,
        pose: RecoveryPose,
        risk: str,
        clearance: float | None,
        view_quality: str,
    ) -> None:
        self._update_retreat_distance(pose)
        with self._lock:
            started_at = self._started_at
            retreat_distance = self._retreat_distance_m
        if (
            started_at is None
            or self.clock() - started_at > self.config.max_recovery_seconds
            or retreat_distance > self.config.max_retreat_distance_m + 0.10
        ):
            self._hold("recovery_limit_reached")
            return
        if risk == "unknown":
            self._hold("live_proximity_became_unknown")
            return
        grid = self._grid(pose)
        if grid is None:
            self._wait_for_fresh_costmap()
            return
        self._costmap_is_fresh_again("retracing_verified_path")
        if not grid.footprint_is_clear(
            pose.x,
            pose.y,
            self._footprint_clearance,
            allow_traversed_unknown=True,
        ):
            self._hold("current_footprint_not_proven_clear")
            return
        with self._lock:
            self._forward_view_quality = view_quality
        if (
            risk in {"clear", "warning"}
            and (clearance is None or clearance >= self.config.safe_clearance_m)
            and view_quality == "good"
            and retreat_distance >= self.config.min_retreat_distance_m
        ):
            self._start_settling(pose)
            return
        with self._lock:
            while self._route:
                point = self._route[0]
                distance = math.hypot(point.x - pose.x, point.y - pose.y)
                if distance > self.config.waypoint_tolerance_m:
                    break
                self._route.popleft()
                self._best_waypoint_distance = None
                self._last_progress_at = self.clock()
            if not self._route:
                # The destination's live view may have changed. Validate the next
                # older segment only now, without invalidating a usable short route.
                route = self._select_route(pose, grid, before_recorded_at=self._route_end_recorded_at)
                if not route:
                    self._hold("verified_route_ended_before_clear")
                    return
                self._route = route
            point = self._route[0]
            distance = math.hypot(point.x - pose.x, point.y - pose.y)
            best = self._best_waypoint_distance
            if best is None or distance < best - 0.008:
                self._best_waypoint_distance = distance
                self._last_progress_at = self.clock()
            last_progress = self._last_progress_at
        if last_progress is None or (
            self.clock() - last_progress > self.config.progress_timeout_seconds
        ):
            self._hold("recovery_not_progressing")
            return
        if not grid.segment_is_clear(
            (pose.x, pose.y),
            (point.x, point.y),
            self._footprint_clearance,
            allow_traversed_unknown=True,
        ):
            self._hold("retreat_segment_no_longer_safe")
            return

        with self._lock:
            previous_clearance = self._last_clearance
            if clearance is not None and previous_clearance is not None:
                if clearance < previous_clearance - 0.04:
                    self._clearance_regressions += 1
                elif clearance >= previous_clearance:
                    self._clearance_regressions = 0
            if clearance is not None:
                self._last_clearance = clearance
            regressions = self._clearance_regressions
        if regressions >= 2:
            self._hold("clearance_decreasing_during_recovery")
            return

        world_x = point.x - pose.x
        world_y = point.y - pose.y
        cosine = math.cos(pose.yaw)
        sine = math.sin(pose.yaw)
        body_x = cosine * world_x + sine * world_y
        body_y = -sine * world_x + cosine * world_y
        speed = min(
            self.config.recovery_speed_mps,
            max(0.02, distance * 0.8),
        )
        scale = speed / max(distance, 1e-9)
        with self._lock:
            token = self._hold_token
        if token is None or not self.publish_recovery(
            token,
            body_x * scale,
            body_y * scale,
        ):
            self._hold("recovery_channel_unavailable")

    def _handle_settling(
        self,
        pose: RecoveryPose,
        risk: str,
        clearance: float | None,
        view_quality: str,
    ) -> None:
        self._heartbeat_zero()
        grid = self._grid(pose)
        if grid is None:
            self._wait_for_fresh_costmap()
            return
        self._costmap_is_fresh_again("confirming_recovery_stop")
        if not grid.footprint_is_clear(
            pose.x,
            pose.y,
            self._footprint_clearance,
            allow_traversed_unknown=True,
        ):
            self._hold("final_recovery_map_evidence_unavailable")
            return
        with self._lock:
            self._forward_view_quality = view_quality
        if (
            risk not in {"clear", "warning"}
            or (clearance is not None and clearance < self.config.safe_clearance_m)
            or view_quality != "good"
        ):
            self._hold("noncritical_good_view_not_stable")
            return
        with self._lock:
            stop_completed = self._recovery_stop_command_completed_at
            stop_started = self._recovery_stop_started_at
        if stop_completed is None or stop_started is None:
            self._hold("invalid_recovery_stop_state")
            return
        if not self._stationary_sample(pose, stop_completed):
            if (
                self.clock() - stop_started
                > self.config.stop_confirmation_timeout_seconds
            ):
                self._hold("recovery_stationary_not_confirmed")
            return
        confirmed_at = self.wall_clock()
        with self._lock:
            self._recovery_stationary_confirmed_at = confirmed_at
            self._state = "recovered_waiting_replan"
            self._reason = "safe_clearance_and_stationarity_confirmed"
            self._last_outcome = "recovered_ready_for_replan"
        self.events.append(
            "safety",
            "recovery",
            "Safe recovery completed",
            "Robot is stationary at the nearest verified noncritical pose with a good first-person view; the old plan remains cancelled.",
            data={
                "clearance_m": clearance,
                "retreat_distance_m": round(self._retreat_distance_m, 3),
                "forward_view_quality": view_quality,
            },
        )
        if self.on_recovery_ready is not None:
            try:
                self.on_recovery_ready()
            except Exception as error:  # noqa: BLE001 - motion remains safely held
                self.events.append(
                    "safety",
                    "error",
                    "Recovered task replan callback failed",
                    str(error)[:500],
                    level="danger",
                )

    def _hold(self, reason: str, detail: str = "") -> None:
        self._heartbeat_zero()
        with self._lock:
            if self._state == "held" and self._reason == reason:
                return
            self._state = "held"
            self._reason = reason
            self._last_outcome = "held"
            self._route.clear()
        self.events.append(
            "safety",
            "recovery",
            "Safety recovery held position",
            detail or reason.replace("_", " "),
            level="danger",
            data={"reason": reason, "route_diagnostic": dict(self._route_diagnostic)},
        )

    def step(self) -> None:
        if not self.enabled:
            return
        observation = self.observation_provider()
        pose = self._pose(observation)
        risk, clearance = self._risk(observation)
        view_quality = self._view_quality(observation)
        with self._lock:
            state = self._state
        if state == "idle":
            warning_approach = bool(
                risk == "warning"
                and self.config.warning_approach_guard
                and self._approaching_nearest_obstacle(observation)
            )
            if pose is not None and (risk == "critical" or warning_approach):
                self._begin_critical(
                    pose,
                    clearance,
                    reason=(
                        "warning_approach_guard"
                        if warning_approach
                        else "critical_proximity"
                    ),
                )
            elif pose is not None:
                self._record_breadcrumb(pose, risk, clearance, view_quality)
            return
        if pose is None:
            self._hold("odometry_unavailable")
            return
        if state == "stopping":
            self._handle_stopping(pose)
        elif state == "retreating":
            self._handle_retreating(pose, risk, clearance, view_quality)
        elif state == "settling":
            self._handle_settling(pose, risk, clearance, view_quality)
        else:
            self._heartbeat_zero()

    def acknowledge_replan(self) -> bool:
        """Release a successful hold only when a new instruction is accepted."""

        with self._lock:
            if self._state != "recovered_waiting_replan":
                return False
            token = self._hold_token
        if token is None or not self.end_hold(token):
            self._hold("recovery_hold_release_failed")
            return False
        with self._lock:
            self._hold_token = None
            self._state = "idle"
            self._reason = "new_plan_accepted"
            self._route.clear()
        self.events.append(
            "safety",
            "recovery",
            "Recovery hold released for a new plan",
            "The prior navigation was not resumed; a newly accepted instruction now owns control.",
        )
        return True

    def reset(self) -> None:
        with self._lock:
            token = self._hold_token
        if token is not None:
            self.end_hold(token)
        with self._lock:
            self._breadcrumbs.clear()
            self._state = "idle"
            self._reason = ""
            self._last_outcome = ""
            self._hold_token = None
            self._route.clear()
            self._route_diagnostic = {}
            self._route_end_recorded_at = None
            self._started_at = None
            self._stop_started_at = None
            self._stop_command_completed_at = None
            self._stationary_confirmed_at = None
            self._recovery_stop_command_completed_at = None
            self._recovery_stationary_confirmed_at = None
            self._recovery_stop_started_at = None
            self._stationary_reference = None
            self._stationary_count = 0
            self._retreat_distance_m = 0.0
            self._last_motion_pose = None
            self._best_waypoint_distance = None
            self._last_progress_at = None
            self._last_clearance = None
            self._clearance_regressions = 0
            self._forward_view_quality = "unknown"
            self._costmap_wait_started_at = None

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread:
            thread.join(timeout=2.0)
        self.reset()

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            return {
                "enabled": self.enabled,
                "state": state,
                "active": state in self.ACTIVE_STATES,
                "safety_hold": self._hold_token is not None,
                "route_diagnostic": dict(self._route_diagnostic),
                "reconsideration_required": state == "recovered_waiting_replan",
                "reason": self._reason,
                "last_outcome": self._last_outcome,
                "breadcrumb_count": len(self._breadcrumbs),
                "route_waypoints_remaining": len(self._route),
                "recovery_speed_limit_mps": self.config.recovery_speed_mps,
                "safe_clearance_m": self.config.safe_clearance_m,
                "forward_view_quality": self._forward_view_quality,
                "retreat_distance_m": round(self._retreat_distance_m, 3),
                "stop_command_completed_at": self._stop_command_completed_at,
                "stationary_confirmed_at": self._stationary_confirmed_at,
                "recovery_stop_command_completed_at": (
                    self._recovery_stop_command_completed_at
                ),
                "recovery_stationary_confirmed_at": (
                    self._recovery_stationary_confirmed_at
                ),
                "sampled_at": _utc_now(),
            }
