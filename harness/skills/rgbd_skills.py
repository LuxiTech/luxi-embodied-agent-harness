"""Shared aligned RGB-D navigation, object-search, and person-follow skills."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
from threading import Event, RLock, Thread
import time
from typing import Any, Callable, Protocol

import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable
from dimos_lcm.std_msgs import Bool

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.navigation.base import NavigationState
from dimos.navigation.navigation_spec import NavigationInterfaceSpec
from dimos.navigation.visual.query import get_object_bbox_from_image
from dimos.perception.spatial_memory_spec import SpatialMemorySpec

from harness.skills.task_primitives import (
    CsrtTargetTracker,
    FollowDistanceAcquisition,
    FrontierSelectionConfig,
    FrontierExplorationResult,
    KnownFreeFrontierGoal,
    PersonFollowResult,
    TrackingMeasurement,
    compute_follow_twist,
    select_known_free_frontier,
)
from harness.robots.g1.isaac.location_tagging import configured_tagged_locations_path
from harness.robots.g1.isaac.navigation_service import IsaacNavigationControlChannel
from harness.control.terminal_cancellation import TerminalCancellationChannel
from harness.skills.person_follow import (
    FollowAcquisition,
    FollowRuntimeConfig,
    FollowStopEvidence,
    PersonFollowExecutor,
)
from harness.robots.world_adapter import WORLD_STATE_SCHEMA_VERSION, read_world_json


def configured_depth_path() -> Path:
    configured = os.environ.get("LUXI_HEAD_DEPTH_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    runtime = os.environ.get(
        "DIMOS_RUNTIME_DIR",
        str(Path.home() / "work/Asset/dimos/runtime"),
    )
    return (Path(runtime) / "luxi-sim-control/head-depth.npz").resolve()


def configured_lidar_proximity_path() -> Path | None:
    configured = os.environ.get("LUXI_LIDAR_PROXIMITY_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return None


def _is_vertical_surface_target(query: str) -> bool:
    normalized = query.strip().casefold()
    return any(term in normalized for term in ("door", "gate", "doorway", "门"))


def _is_compact_bottle_target(query: str) -> bool:
    normalized = query.strip().casefold()
    return any(
        term in normalized
        for term in ("bottle", "water bottle", "水瓶", "矿泉水", "饮料瓶")
    )


def _semantic_detection_query(query: str) -> str:
    """Add appearance vocabulary without adding a location or scene prior."""

    if not _is_compact_bottle_target(query):
        return query
    return (
        f"{query}; also inspect for a small clear or transparent PET drinking "
        "bottle whose body may be nearly invisible except for water, label "
        "bands, neck, rings, or a ribbed cap"
    )


def _compact_vision_attempt(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep correlated perception evidence below the control-channel limit."""

    keys = (
        "request_id",
        "provider_request_id",
        "frame_timestamp",
        "targeted_crop",
        "targeted_crop_center",
        "semantic_crop_scale",
        "error",
        "localization_stage",
        "normalized_detection",
        "bbox",
        "depth_m",
        "valid_depth_points",
        "target_world",
    )
    return {key: metadata[key] for key in keys if key in metadata}


class FreshLidarRiskMonitor:
    """Read current self-filtered lidar evidence produced by this simulation."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        max_age_seconds: float = 1.5,
        warning_center_distance_m: float = 0.85,
        critical_center_distance_m: float = 0.45,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = configured_lidar_proximity_path() if path is None else path
        self.max_age_seconds = max(0.1, float(max_age_seconds))
        self.warning_center_distance_m = max(
            0.1,
            float(warning_center_distance_m),
        )
        self.critical_center_distance_m = max(
            0.05,
            min(float(critical_center_distance_m), self.warning_center_distance_m),
        )
        self.clock = clock

    def state(self) -> str:
        if self.path is None:
            return "unknown"
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self_filter = payload.get("self_filter", {})
            sequence = payload.get("sequence")
            written_at = float(payload["written_at"])
            age = self.clock() - written_at
            if (
                payload.get("schema_version") != 1
                or payload.get("available") is not True
                or payload.get("source") != "current_lidar"
                or payload.get("frame_id") != "world"
                or isinstance(sequence, bool)
                or int(sequence) < 1
                or not isinstance(self_filter, dict)
                or self_filter.get("identity_verified") is not True
                or not math.isfinite(age)
                or age < 0.0
                or age > self.max_age_seconds
            ):
                return "unknown"
            nearest_raw = payload.get("nearest_obstacle_distance")
            nearest = None if nearest_raw is None else float(nearest_raw)
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            return "unknown"
        if nearest is None:
            return "clear"
        if not math.isfinite(nearest) or nearest < 0.0:
            return "unknown"
        if nearest <= self.critical_center_distance_m:
            return "critical"
        if nearest < self.warning_center_distance_m:
            return "warning"
        return "clear"


@dataclass(frozen=True)
class PersonPosition:
    point: Vector3
    depth_m: float
    valid_points: int
    depth_spread_m: float


@dataclass(frozen=True)
class VisualTargetObservation:
    world_point: Vector3
    estimate: PersonPosition
    frame_timestamp: float
    robot_transform_at_frame: Transform | None = None


@dataclass(frozen=True)
class _CostmapRiskSnapshot:
    cells: np.ndarray[Any, np.dtype[np.int8]]
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    frame_id: str
    received_at: float


class LiveCostmapRiskMonitor:
    """Derive collision risk only from a fresh online occupancy costmap."""

    def __init__(
        self,
        *,
        max_age_seconds: float = 3.0,
        warning_distance_m: float = 0.85,
        critical_distance_m: float = 0.45,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_age_seconds = max(0.1, float(max_age_seconds))
        self.warning_distance_m = max(0.1, float(warning_distance_m))
        self.critical_distance_m = max(0.05, float(critical_distance_m))
        self.clock = clock
        self._lock = RLock()
        self._snapshot: _CostmapRiskSnapshot | None = None

    def update(self, message: OccupancyGrid) -> None:
        try:
            cells = np.asarray(message.grid, dtype=np.int8)
            resolution = float(message.resolution)
            origin_x = float(message.origin.position.x)
            origin_y = float(message.origin.position.y)
            orientation = message.origin.orientation
            quaternion_w = float(getattr(orientation, "w", 1.0))
            quaternion_x = float(getattr(orientation, "x", 0.0))
            quaternion_y = float(getattr(orientation, "y", 0.0))
            quaternion_z = float(getattr(orientation, "z", 0.0))
            origin_yaw = math.atan2(
                2.0 * (quaternion_w * quaternion_z + quaternion_x * quaternion_y),
                1.0 - 2.0 * (quaternion_y * quaternion_y + quaternion_z * quaternion_z),
            )
            frame_id = str(message.frame_id)
            valid = bool(
                cells.ndim == 2
                and cells.size > 0
                and math.isfinite(resolution)
                and resolution > 0.0
                and all(
                    math.isfinite(value) for value in (origin_x, origin_y, origin_yaw)
                )
                and frame_id in {"world", "map"}
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            valid = False
        snapshot = (
            _CostmapRiskSnapshot(
                cells=cells.copy(),
                resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
                origin_yaw=origin_yaw,
                frame_id=frame_id,
                received_at=self.clock(),
            )
            if valid
            else None
        )
        with self._lock:
            self._snapshot = snapshot

    def state(self, robot: Transform | None) -> str:
        with self._lock:
            snapshot = self._snapshot
        if robot is None or snapshot is None:
            return "unknown"
        age = self.clock() - snapshot.received_at
        if age < 0.0 or age > self.max_age_seconds:
            return "unknown"
        if robot.frame_id not in {"", snapshot.frame_id, "world", "map"}:
            return "unknown"

        relative_x = float(robot.translation.x) - snapshot.origin_x
        relative_y = float(robot.translation.y) - snapshot.origin_y
        cos_yaw = math.cos(snapshot.origin_yaw)
        sin_yaw = math.sin(snapshot.origin_yaw)
        local_x = cos_yaw * relative_x + sin_yaw * relative_y
        local_y = -sin_yaw * relative_x + cos_yaw * relative_y
        column = math.floor(local_x / snapshot.resolution)
        row = math.floor(local_y / snapshot.resolution)
        height, width = snapshot.cells.shape
        if not (0 <= column < width and 0 <= row < height):
            return "unknown"
        if int(snapshot.cells[row, column]) < 0:
            # The robot's current footprint is itself evidence that its exact
            # cell is traversable.  Still require nearby online map evidence so
            # an entirely unknown map cannot be reported as clear.
            radius_cells = max(
                1,
                math.ceil(self.warning_distance_m / snapshot.resolution),
            )
            neighborhood = snapshot.cells[
                max(0, row - radius_cells) : min(height, row + radius_cells + 1),
                max(0, column - radius_cells) : min(width, column + radius_cells + 1),
            ]
            if not np.any(neighborhood >= 0):
                return "unknown"

        occupied = np.argwhere(snapshot.cells >= 50)
        if occupied.size == 0:
            return "clear"
        dx = (occupied[:, 1].astype(np.float64) + 0.5) * snapshot.resolution - local_x
        dy = (occupied[:, 0].astype(np.float64) + 0.5) * snapshot.resolution - local_y
        nearest = float(np.sqrt(dx * dx + dy * dy).min())
        if nearest < self.critical_distance_m:
            return "critical"
        if nearest < self.warning_distance_m:
            return "warning"
        return "clear"

    def constrain_frontier_costmap(
        self,
        exploration: OccupancyGrid,
    ) -> OccupancyGrid | None:
        """Intersect exploration free space with the live planner costmap."""

        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return None
        age = self.clock() - snapshot.received_at
        try:
            exploration_cells = np.asarray(exploration.grid, dtype=np.int8)
            resolution = float(exploration.resolution)
            origin_x = float(exploration.origin.position.x)
            origin_y = float(exploration.origin.position.y)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if (
            age < 0.0
            or age > self.max_age_seconds
            or exploration_cells.shape != snapshot.cells.shape
            or abs(resolution - snapshot.resolution) > 1e-6
            or abs(origin_x - snapshot.origin_x) > resolution
            or abs(origin_y - snapshot.origin_y) > resolution
            or str(exploration.frame_id) not in {snapshot.frame_id, "world", "map"}
        ):
            return None
        constrained = exploration_cells.copy()
        # Keep exploration unknown as unknown so frontier extraction remains
        # valid, but a cell may be traversed only when both maps call it free.
        constrained[(exploration_cells == 0) & (snapshot.cells != 0)] = 100
        return OccupancyGrid(
            grid=constrained,
            resolution=resolution,
            origin=exploration.origin,
            frame_id=str(exploration.frame_id),
            ts=float(exploration.ts),
        )

    def goal_region_state(
        self,
        *,
        x: float,
        y: float,
        clearance_m: float,
        diagnostics: dict[str, Any] | None = None,
    ) -> str:
        """Return whether a goal disk remains free in the live planner map."""

        def report(state, reason, **details):
            if diagnostics is not None:
                diagnostics.update(state=state, reason=reason, point=[x, y],
                                   clearance_m=clearance_m, **details)
            return state

        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return report("unknown", "map_unavailable")
        age = self.clock() - snapshot.received_at
        if diagnostics is not None:
            diagnostics.update(map_age_s=age, max_map_age_s=self.max_age_seconds,
                               map_received_monotonic=snapshot.received_at,
                               frame_id=snapshot.frame_id, resolution=snapshot.resolution,
                               origin=[snapshot.origin_x, snapshot.origin_y, snapshot.origin_yaw])
        if age < 0.0 or age > self.max_age_seconds:
            return report("unknown", "map_stale")
        try:
            relative_x = float(x) - snapshot.origin_x
            relative_y = float(y) - snapshot.origin_y
            cos_yaw = math.cos(snapshot.origin_yaw)
            sin_yaw = math.sin(snapshot.origin_yaw)
            local_x = cos_yaw * relative_x + sin_yaw * relative_y
            local_y = -sin_yaw * relative_x + cos_yaw * relative_y
            column = math.floor(local_x / snapshot.resolution)
            row = math.floor(local_y / snapshot.resolution)
            radius = math.ceil(float(clearance_m) / snapshot.resolution)
        except (TypeError, ValueError, OverflowError):
            return report("unknown", "invalid_region")
        height, width = snapshot.cells.shape
        if (
            radius < 1
            or row - radius < 0
            or row + radius >= height
            or column - radius < 0
            or column + radius >= width
        ):
            return report("unknown", "map_out_of_bounds")
        view = snapshot.cells[
            row - radius : row + radius + 1,
            column - radius : column + radius + 1,
        ]
        rr, cc = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        disk = (rr * rr + cc * cc) * snapshot.resolution**2 <= float(
            clearance_m
        ) ** 2
        if np.all(view[disk] == 0):
            return report("clear", "clear")
        if diagnostics is not None:
            # Bounded crop of the SAME snapshot used for this decision.
            crop_radius = min(radius, 16)
            diagnostics.update(
                cell=[row, column],
                nonzero_cells=int(np.count_nonzero(view[disk])),
                unknown_cells=int(np.count_nonzero(view[disk] < 0)),
                positive_cost_cells=int(np.count_nonzero(view[disk] > 0)),
                max_cost=int(view[disk].max()),
                crop_row=row-crop_radius, crop_column=column-crop_radius,
                crop_truncated=radius > crop_radius,
                cells=snapshot.cells[row-crop_radius:row+crop_radius+1,
                                     column-crop_radius:column+crop_radius+1].tolist())
        return report("blocked", "nonfree_cells")


class LiveCostmapRiskFeed:
    """Keep costmap risk fresh on an LCM thread independent of skill RPC work."""

    def __init__(
        self,
        monitor: LiveCostmapRiskMonitor,
        *,
        transport_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.monitor = monitor
        self.transport_factory = transport_factory
        self._transport: Any | None = None
        self._unsubscribe: Callable[[], None] | None = None

    @staticmethod
    def _runtime_transport() -> Any:
        from dimos.core.transport import LCMTransport

        return LCMTransport("/global_costmap", OccupancyGrid)

    def start(self) -> None:
        if self._transport is not None:
            return
        factory = self.transport_factory or self._runtime_transport
        transport = factory()
        unsubscribe = transport.subscribe(self.monitor.update)
        self._transport = transport
        self._unsubscribe = unsubscribe

    def stop(self) -> None:
        unsubscribe = self._unsubscribe
        transport = self._transport
        self._unsubscribe = None
        self._transport = None
        if unsubscribe is not None:
            unsubscribe()
        if transport is not None:
            transport.stop()


@dataclass(frozen=True)
class VisualApproachResult:
    tool_ok: bool
    task_status: str
    completed: bool
    planner_goal_reached: bool
    target_distance_m: float | None
    stop_command_publish_latency_ms: float
    physical_stop_latency_ms: float | None
    elapsed_s: float
    message: str = ""
    vision_requests: tuple[dict[str, Any], ...] = ()
    stop_command_completed_at: float | None = None
    stationary_confirmed_at: float | None = None
    verification_frame_timestamp: float | None = None
    verification: dict[str, Any] | None = None
    viewpoint_search: dict[str, Any] | None = None
    phase_timings: dict[str, float] = field(default_factory=dict)
    requested_standoff_distance_m: float | None = None
    effective_standoff_distance_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_ok": self.tool_ok,
            "task_status": self.task_status,
            "completed": self.completed,
            "planner_goal_reached": self.planner_goal_reached,
            "target_distance_m": (
                round(self.target_distance_m, 3)
                if self.target_distance_m is not None
                else None
            ),
            "stop_command_publish_latency_ms": round(
                self.stop_command_publish_latency_ms, 1
            ),
            "physical_stop_latency_ms": (
                round(self.physical_stop_latency_ms, 1)
                if self.physical_stop_latency_ms is not None
                else None
            ),
            "elapsed_s": round(self.elapsed_s, 3),
            "message": self.message,
            "vision_requests": [dict(item) for item in self.vision_requests],
            "stop_command_completed_at": self.stop_command_completed_at,
            "stationary_confirmed_at": self.stationary_confirmed_at,
            "verification_frame_timestamp": self.verification_frame_timestamp,
            "verification": (
                dict(self.verification) if self.verification is not None else None
            ),
            "viewpoint_search": (
                dict(self.viewpoint_search)
                if self.viewpoint_search is not None
                else None
            ),
            "phase_timings": {
                key: round(float(value), 3) for key, value in self.phase_timings.items()
            },
            "requested_standoff_distance_m": (
                round(self.requested_standoff_distance_m, 3)
                if self.requested_standoff_distance_m is not None
                else None
            ),
            "effective_standoff_distance_m": (
                round(self.effective_standoff_distance_m, 3)
                if self.effective_standoff_distance_m is not None
                else None
            ),
        }

    def agent_encode(self) -> list[dict[str, str]]:
        return [
            {"type": "text", "text": json.dumps(self.as_dict(), ensure_ascii=False)}
        ]

    def __str__(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))


class VisualApproachIO(Protocol):
    def locate_target(
        self,
        query: str,
        *,
        after: float = 0.0,
    ) -> VisualTargetObservation | None: ...

    def robot_transform(self, timestamp: float | None = None) -> Transform | None: ...

    def set_goal(self, goal: PoseStamped) -> bool: ...

    def is_goal_reached(self) -> bool: ...

    def navigation_state(self) -> NavigationState: ...

    def risk_state(self) -> str: ...

    def is_cancelled(self) -> bool: ...

    def safe_stop(self) -> None: ...

    def wait_until_stationary(
        self, *, after: float, timeout: float
    ) -> float | None: ...


class VisualApproachExecutor:
    """Run one bounded visual navigation task and return a typed outcome."""

    def __init__(
        self,
        io: VisualApproachIO,
        *,
        standoff_tolerance_m: float = 0.3,
        max_final_angle_degrees: float = 35.0,
        activation_grace_seconds: float = 1.0,
        completion_grace_seconds: float = 0.75,
        planner_position_tolerance_m: float = 0.2,
        max_direct_goal_distance_m: float = 3.0,
        max_visibility_standoff_increase_m: float = 0.8,
        target_relocalization_tolerance_m: float = 0.5,
        final_alignment_timeout_seconds: float = 20.0,
        stationary_timeout_seconds: float = 0.5,
        max_stop_latency_seconds: float = 0.5,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.io = io
        self.standoff_tolerance_m = standoff_tolerance_m
        self.max_final_angle_degrees = max_final_angle_degrees
        self.activation_grace_seconds = max(0.1, float(activation_grace_seconds))
        self.completion_grace_seconds = max(0.1, float(completion_grace_seconds))
        self.planner_position_tolerance_m = max(
            0.0, float(planner_position_tolerance_m)
        )
        self.max_direct_goal_distance_m = max(
            0.5,
            float(max_direct_goal_distance_m),
        )
        self.max_visibility_standoff_increase_m = min(
            1.0,
            max(0.0, float(max_visibility_standoff_increase_m)),
        )
        self.target_relocalization_tolerance_m = max(
            0.1,
            float(target_relocalization_tolerance_m),
        )
        self.final_alignment_timeout_seconds = min(
            20.0,
            max(5.0, float(final_alignment_timeout_seconds)),
        )
        self.stationary_timeout_seconds = min(
            7.0,
            max(0.1, float(stationary_timeout_seconds)),
        )
        self.max_stop_latency_seconds = min(
            self.stationary_timeout_seconds,
            max(0.1, float(max_stop_latency_seconds)),
        )
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._sleep = sleep

    def _bounded_planner_goal(
        self,
        robot: Transform,
        final_goal: PoseStamped,
        target_world: Vector3,
    ) -> tuple[PoseStamped, bool]:
        """Keep each goal inside the useful horizon of the online map."""

        dx = final_goal.position.x - robot.translation.x
        dy = final_goal.position.y - robot.translation.y
        distance = math.hypot(dx, dy)
        if distance <= self.max_direct_goal_distance_m:
            return final_goal, True
        scale = self.max_direct_goal_distance_m / distance
        waypoint_x = robot.translation.x + dx * scale
        waypoint_y = robot.translation.y + dy * scale
        waypoint_yaw = math.atan2(
            target_world.y - waypoint_y,
            target_world.x - waypoint_x,
        )
        return (
            PoseStamped(
                position=Vector3(waypoint_x, waypoint_y, 0.0),
                orientation=Quaternion.from_euler(Vector3(0.0, 0.0, waypoint_yaw)),
                frame_id="world",
            ),
            False,
        )

    def _result(
        self,
        *,
        started_at: float,
        task_status: str,
        target_distance_m: float | None,
        stop_command_publish_latency_ms: float = 0.0,
        physical_stop_latency_ms: float | None = None,
        tool_ok: bool = True,
        planner_goal_reached: bool = False,
        message: str = "",
        vision_requests: tuple[dict[str, Any], ...] = (),
        stop_command_completed_at: float | None = None,
        stationary_confirmed_at: float | None = None,
        verification_frame_timestamp: float | None = None,
        verification: dict[str, Any] | None = None,
        viewpoint_search: dict[str, Any] | None = None,
        phase_timings: dict[str, float] | None = None,
        requested_standoff_distance_m: float | None = None,
        effective_standoff_distance_m: float | None = None,
    ) -> VisualApproachResult:
        return VisualApproachResult(
            tool_ok=tool_ok,
            task_status=task_status,
            completed=task_status == "arrived_verified",
            planner_goal_reached=planner_goal_reached,
            target_distance_m=target_distance_m,
            stop_command_publish_latency_ms=stop_command_publish_latency_ms,
            physical_stop_latency_ms=physical_stop_latency_ms,
            elapsed_s=max(0.0, self._monotonic() - started_at),
            message=message,
            vision_requests=vision_requests,
            stop_command_completed_at=stop_command_completed_at,
            stationary_confirmed_at=stationary_confirmed_at,
            verification_frame_timestamp=verification_frame_timestamp,
            verification=verification,
            viewpoint_search=viewpoint_search,
            phase_timings=dict(phase_timings or {}),
            requested_standoff_distance_m=requested_standoff_distance_m,
            effective_standoff_distance_m=effective_standoff_distance_m,
        )

    def approach(
        self,
        query: str,
        *,
        standoff_distance: float = 0.9,
        timeout: float = 25.0,
        initial_observation: VisualTargetObservation | None = None,
    ) -> VisualApproachResult:
        started_at = self._monotonic()
        vision_requests: list[dict[str, Any]] = []
        phase_timings: dict[str, float] = {}
        viewpoint_search: dict[str, Any] | None = None
        requested_standoff_distance = float(standoff_distance)
        effective_standoff_distance = requested_standoff_distance

        def record_vision_request() -> None:
            reader = getattr(self.io, "vision_request_metadata", None)
            if not callable(reader):
                return
            try:
                metadata = reader()
            except Exception:  # noqa: BLE001 - tracing must not alter task control
                return
            if not isinstance(metadata, dict) or not metadata.get("request_id"):
                return
            request_id = str(metadata["request_id"])
            if any(
                str(item.get("request_id")) == request_id for item in vision_requests
            ):
                return
            vision_requests.append(dict(metadata))

        def finish(**kwargs: Any) -> VisualApproachResult:
            return self._result(
                started_at=started_at,
                vision_requests=tuple(vision_requests),
                viewpoint_search=viewpoint_search,
                phase_timings=phase_timings,
                requested_standoff_distance_m=requested_standoff_distance,
                effective_standoff_distance_m=effective_standoff_distance,
                **kwargs,
            )

        def vision_request_failed() -> bool:
            return bool(
                vision_requests
                and (
                    vision_requests[-1].get("error")
                    or vision_requests[-1].get("localization_stage") == "vlm_error"
                )
            )

        query = query.strip()
        if (
            not query
            or not 0.5 <= standoff_distance <= 3.0
            or not 3.0 <= timeout <= 90.0
        ):
            return finish(
                task_status="invalid_input",
                target_distance_m=None,
                tool_ok=False,
                message="query、standoff_distance 或 timeout 超出允许范围",
            )

        located = initial_observation
        if located is None:
            phase_started = self._monotonic()
            located = self.io.locate_target(query)
            phase_timings["initial_visual_localization_s"] = max(
                0.0,
                self._monotonic() - phase_started,
            )
            record_vision_request()
        else:
            phase_timings["initial_visual_localization_s"] = 0.0
        if located is None:
            if vision_request_failed():
                self.io.safe_stop()
                return finish(
                    task_status="visual_branch_terminated",
                    target_distance_m=None,
                    message="视觉请求失败，已停车且不再重试",
                )
            search_viewpoint = getattr(self.io, "search_viewpoint", None)
            if not callable(search_viewpoint):
                return finish(
                    task_status="target_not_found",
                    target_distance_m=None,
                    message="最新 RGB-D 无法定位目标",
                )
            try:
                phase_started = self._monotonic()
                search_result = search_viewpoint()
                phase_timings["viewpoint_search_s"] = max(
                    0.0,
                    self._monotonic() - phase_started,
                )
            except Exception:  # noqa: BLE001 - bounded search must fail closed
                self.io.safe_stop()
                search_result = None
            metadata_reader = getattr(self.io, "viewpoint_search_metadata", None)
            if callable(metadata_reader):
                try:
                    metadata = metadata_reader()
                except Exception:  # noqa: BLE001 - diagnostics cannot alter control
                    metadata = None
                if isinstance(metadata, dict):
                    viewpoint_search = dict(metadata)
            search_completed_at = (
                float(search_result)
                if (
                    isinstance(search_result, (int, float))
                    and not isinstance(search_result, bool)
                    and math.isfinite(float(search_result))
                    and float(search_result) > 0.0
                )
                else 0.0
            )
            search_completed = bool(search_result)
            if not search_completed:
                return finish(
                    task_status="target_not_found",
                    target_distance_m=None,
                    message="目标不可见，受控换视角未找到安全观测位",
                )
            phase_started = self._monotonic()
            located = self.io.locate_target(query, after=search_completed_at)
            phase_timings["post_search_visual_localization_s"] = max(
                0.0,
                self._monotonic() - phase_started,
            )
            record_vision_request()
            if located is None:
                if vision_request_failed():
                    self.io.safe_stop()
                    return finish(
                        task_status="visual_branch_terminated",
                        target_distance_m=None,
                        message="换视角后的视觉请求失败，已停车且不再重试",
                    )
                return finish(
                    task_status="target_not_found",
                    target_distance_m=None,
                    message="受控换视角后仍无法从 RGB-D 定位目标",
                )
        robot = located.robot_transform_at_frame or self.io.robot_transform(
            located.frame_timestamp
        )
        if robot is None:
            return finish(
                task_status="pose_unavailable",
                target_distance_m=None,
                message="机器人世界位姿不可用",
            )
        standoff_recommender = getattr(
            self.io,
            "recommend_standoff_distance",
            None,
        )
        if callable(standoff_recommender) and not _is_vertical_surface_target(query):
            try:
                recommended_standoff = float(
                    standoff_recommender(located, requested_standoff_distance)
                )
            except Exception:  # noqa: BLE001 - invalid advice cannot alter safety
                recommended_standoff = requested_standoff_distance
            if math.isfinite(recommended_standoff):
                effective_standoff_distance = min(
                    3.0,
                    requested_standoff_distance
                    + self.max_visibility_standoff_increase_m,
                    max(requested_standoff_distance, recommended_standoff),
                )
        if _is_compact_bottle_target(query):
            # A tabletop bottle's first depth cluster can underestimate how
            # far the complete object extends toward the image bottom. Keep
            # another half metre of view before planner tolerance is applied.
            effective_standoff_distance = min(
                3.0,
                requested_standoff_distance
                + self.max_visibility_standoff_increase_m,
                max(
                    effective_standoff_distance,
                    requested_standoff_distance + 0.5,
                ),
            )
        current_distance = math.hypot(
            located.world_point.x - robot.translation.x,
            located.world_point.y - robot.translation.y,
        )
        if (
            abs(current_distance - effective_standoff_distance)
            <= self.standoff_tolerance_m
        ):
            # Do not advance toward an occluding surface when the measured
            # target is already inside the accepted standoff band.  Submit
            # the current known-safe position with target-facing yaw so the
            # planner must still provide real goal-reached evidence.
            final_goal = PoseStamped(
                position=Vector3(
                    float(robot.translation.x),
                    float(robot.translation.y),
                    0.0,
                ),
                orientation=Quaternion.from_euler(
                    Vector3(
                        0.0,
                        0.0,
                        math.atan2(
                            located.world_point.y - robot.translation.y,
                            located.world_point.x - robot.translation.x,
                        ),
                    )
                ),
                frame_id="world",
            )
        else:
            planning_standoff = max(
                0.5,
                effective_standoff_distance - self.planner_position_tolerance_m,
            )
            final_goal = make_standoff_goal(
                robot.translation,
                located.world_point,
                planning_standoff,
            )
        if final_goal is None:
            self.io.safe_stop()
            return finish(
                task_status="planner_failed",
                target_distance_m=current_distance,
                message="规划器拒绝目标",
            )

        planner_goal_reached = False
        task_status = "navigation_timeout"
        blocked_risk: str | None = None
        deadline: float | None = None
        current_stage_robot = robot
        navigation_started_at: float | None = None
        submitted_final_goal = False
        try:
            # Four bounded stages cover the supported 8 m RGB-D range while
            # preventing an unbounded sequence of planner submissions.
            for _stage_index in range(4):
                # A previous intermediate arrival is not final verification.
                # Each newly accepted stage starts as a navigation deadline.
                task_status = "navigation_timeout"
                stage_goal, is_final_goal = self._bounded_planner_goal(
                    current_stage_robot,
                    final_goal,
                    located.world_point,
                )
                if not self.io.set_goal(stage_goal):
                    task_status = "planner_failed"
                    break
                submitted_final_goal = is_final_goal
                stage_set_at = self._monotonic()
                if navigation_started_at is None:
                    navigation_started_at = stage_set_at
                if deadline is None:
                    deadline = stage_set_at + timeout
                saw_active_state = False
                saw_motion_progress = False
                idle_after_active_at: float | None = None
                stage_goal_reached = False
                while self._monotonic() < deadline:
                    if self.io.is_cancelled():
                        task_status = "cancelled"
                        break
                    risk = self.io.risk_state()
                    if risk in {"unknown", "critical"}:
                        task_status = "risk_blocked"
                        blocked_risk = risk
                        break
                    if self.io.is_goal_reached():
                        stage_goal_reached = True
                        task_status = "verification_failed"
                        break
                    current_robot = self.io.robot_transform()
                    if current_robot is not None:
                        translation_progress = math.hypot(
                            current_robot.translation.x
                            - current_stage_robot.translation.x,
                            current_robot.translation.y
                            - current_stage_robot.translation.y,
                        )
                        initial_yaw = current_stage_robot.rotation.to_euler().yaw
                        current_yaw = current_robot.rotation.to_euler().yaw
                        rotation_progress = abs(
                            math.atan2(
                                math.sin(current_yaw - initial_yaw),
                                math.cos(current_yaw - initial_yaw),
                            )
                        )
                        saw_motion_progress = bool(
                            saw_motion_progress
                            or translation_progress >= 0.03
                            or math.degrees(rotation_progress) >= 3.0
                        )
                    state = self.io.navigation_state()
                    if state != NavigationState.IDLE:
                        saw_active_state = True
                        idle_after_active_at = None
                    elif saw_active_state:
                        now = self._monotonic()
                        if idle_after_active_at is None:
                            idle_after_active_at = now
                        elif (
                            now - idle_after_active_at >= self.completion_grace_seconds
                        ):
                            task_status = "planner_failed"
                            break
                    elif (
                        not saw_motion_progress
                        and self._monotonic() - stage_set_at
                        >= self.activation_grace_seconds
                    ):
                        task_status = "planner_failed"
                        break
                    self._sleep(
                        min(
                            0.1,
                            max(0.0, deadline - self._monotonic()),
                        )
                    )
                if not stage_goal_reached:
                    break
                if is_final_goal:
                    planner_goal_reached = True
                    break
                # Stop at the intermediate waypoint before replacing the
                # planner goal.  The 550 ms pause also clears the navigation
                # relay's mandatory post-stop barrier.
                self.io.safe_stop()
                if deadline - self._monotonic() <= 0.55:
                    task_status = "navigation_timeout"
                    break
                self._sleep(0.55)
                next_stage_robot = self.io.robot_transform()
                if next_stage_robot is None:
                    task_status = "planner_failed"
                    break
                stage_progress = math.hypot(
                    next_stage_robot.translation.x - current_stage_robot.translation.x,
                    next_stage_robot.translation.y - current_stage_robot.translation.y,
                )
                if stage_progress < 0.03:
                    task_status = "planner_failed"
                    break
                current_stage_robot = next_stage_robot
            else:
                task_status = "planner_failed"
        finally:
            if navigation_started_at is not None:
                phase_timings["navigation_s"] = max(
                    0.0,
                    self._monotonic() - navigation_started_at,
                )
            stop_started = self._monotonic()
            self.io.safe_stop()
            stop_command_publish_latency_ms = max(
                0.0,
                (self._monotonic() - stop_started) * 1_000.0,
            )
            # This timestamp says only that zero/cancel publications returned.
            # It is not evidence that the physical robot has stopped.
            stop_command_completed_at = self._wall_time()

        try:
            phase_started = self._monotonic()
            stationary_confirmed_at = self.io.wait_until_stationary(
                after=stop_command_completed_at,
                timeout=self.stationary_timeout_seconds,
            )
        except Exception:  # noqa: BLE001 - missing stop evidence must fail closed
            stationary_confirmed_at = None
        phase_timings["stationary_confirmation_s"] = max(
            0.0,
            self._monotonic() - phase_started,
        )
        physical_stop_latency_ms = (
            max(
                0.0,
                (float(stationary_confirmed_at) - stop_command_completed_at) * 1_000.0,
            )
            if stationary_confirmed_at is not None
            else None
        )
        stop_evidence = {
            "stop_command_publish_latency_ms": stop_command_publish_latency_ms,
            "physical_stop_latency_ms": physical_stop_latency_ms,
            "stop_command_completed_at": stop_command_completed_at,
            "stationary_confirmed_at": stationary_confirmed_at,
        }

        position_goal_reached = False
        if (
            not planner_goal_reached
            and submitted_final_goal
            and task_status == "navigation_timeout"
            and blocked_risk is None
        ):
            final_pose = self.io.robot_transform()
            if final_pose is not None:
                position_goal_reached = (
                    math.hypot(
                        final_pose.translation.x - final_goal.position.x,
                        final_pose.translation.y - final_goal.position.y,
                    )
                    <= self.planner_position_tolerance_m + 1e-6
                )

        if not planner_goal_reached and not position_goal_reached:
            return finish(
                task_status=task_status,
                target_distance_m=current_distance,
                **stop_evidence,
                message=(
                    f"在线 costmap 风险状态为 {blocked_risk}，已停车"
                    if blocked_risk is not None
                    else "规划器未确认到达，任务未完成"
                ),
            )

        if (
            stationary_confirmed_at is None
            or physical_stop_latency_ms is None
            or physical_stop_latency_ms > self.max_stop_latency_seconds * 1_000.0
        ):
            return finish(
                task_status="verification_failed",
                target_distance_m=None,
                planner_goal_reached=planner_goal_reached,
                **stop_evidence,
                message=(
                    "停车命令已发布，但未在 "
                    f"{self.max_stop_latency_seconds:.1f}s 内取得连续 odom 静止确认"
                ),
            )

        world_target_verifier = getattr(self.io, "verify_world_target", None)
        verification: dict[str, Any] | None = None
        if callable(world_target_verifier):
            phase_started = self._monotonic()
            try:
                verified = world_target_verifier(
                    located.world_point,
                    after=stationary_confirmed_at,
                    query=query,
                )
            except Exception:  # noqa: BLE001 - verification failures are incomplete
                verified = None
            metadata_reader = getattr(self.io, "verification_metadata", None)
            if callable(metadata_reader):
                try:
                    metadata = metadata_reader()
                except Exception:  # noqa: BLE001 - diagnostics cannot change outcome
                    metadata = None
                if isinstance(metadata, dict):
                    verification = dict(metadata)
            if verified is None:
                try:
                    targeted_relocalizer = getattr(
                        self.io,
                        "locate_target_near_world",
                        None,
                    )
                    if callable(targeted_relocalizer):
                        relocalized = targeted_relocalizer(
                            query,
                            located.world_point,
                            after=stationary_confirmed_at,
                        )
                    else:
                        relocalized = self.io.locate_target(
                            query,
                            after=stationary_confirmed_at,
                        )
                except Exception:  # noqa: BLE001 - fallback remains fail closed
                    relocalized = None
                record_vision_request()
                if relocalized is not None:
                    relocalization_error = math.sqrt(
                        (relocalized.world_point.x - located.world_point.x) ** 2
                        + (relocalized.world_point.y - located.world_point.y) ** 2
                        + (relocalized.world_point.z - located.world_point.z) ** 2
                    )
                    relocalization_ok = (
                        relocalization_error <= self.target_relocalization_tolerance_m
                    )
                    verification = dict(verification or {})
                    verification["semantic_relocalization"] = {
                        "world_error_m": round(relocalization_error, 3),
                        "tolerance_m": round(
                            self.target_relocalization_tolerance_m,
                            3,
                        ),
                        "accepted": relocalization_ok,
                    }
                    if relocalization_ok:
                        verified = relocalized
            if verified is None and (planner_goal_reached or position_goal_reached):
                alignment_search = getattr(self.io, "search_full_rotation", None)
                if callable(alignment_search):
                    try:
                        verified = alignment_search(
                            query,
                            deadline=(
                                self._monotonic() + self.final_alignment_timeout_seconds
                            ),
                        )
                    except Exception:  # noqa: BLE001 - alignment remains fail closed
                        verified = None
                    metadata_reader = getattr(
                        self.io,
                        "viewpoint_search_metadata",
                        None,
                    )
                    if callable(metadata_reader):
                        try:
                            alignment_metadata = metadata_reader()
                        except Exception:  # noqa: BLE001 - diagnostics only
                            alignment_metadata = None
                        if isinstance(alignment_metadata, dict):
                            verification = dict(verification or {})
                            verification["final_alignment"] = dict(alignment_metadata)
                    record_vision_request()
        else:
            phase_started = self._monotonic()
            verified = self.io.locate_target(query, after=stationary_confirmed_at)
            record_vision_request()
        phase_timings["final_verification_s"] = max(
            0.0,
            self._monotonic() - phase_started,
        )
        if verified is None:
            return finish(
                task_status="verification_failed",
                target_distance_m=None,
                **stop_evidence,
                planner_goal_reached=planner_goal_reached,
                verification=verification,
                message="停车后没有取得新的目标 RGB-D 定位",
            )
        final_robot = verified.robot_transform_at_frame or self.io.robot_transform(
            verified.frame_timestamp
        )
        if final_robot is None:
            return finish(
                task_status="verification_failed",
                target_distance_m=None,
                **stop_evidence,
                verification_frame_timestamp=verified.frame_timestamp,
                planner_goal_reached=planner_goal_reached,
                verification=verification,
                message="停车后机器人位姿不可用",
            )
        dx = verified.world_point.x - final_robot.translation.x
        dy = verified.world_point.y - final_robot.translation.y
        final_distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        robot_yaw = final_robot.rotation.to_euler().yaw
        angle_error = abs(
            math.atan2(
                math.sin(target_yaw - robot_yaw),
                math.cos(target_yaw - robot_yaw),
            )
        )
        distance_ok = (
            abs(final_distance - effective_standoff_distance)
            <= self.standoff_tolerance_m
        )
        angle_ok = math.degrees(angle_error) <= self.max_final_angle_degrees
        if not (distance_ok and angle_ok):
            return finish(
                task_status="verification_failed",
                target_distance_m=final_distance,
                **stop_evidence,
                verification_frame_timestamp=verified.frame_timestamp,
                planner_goal_reached=planner_goal_reached,
                verification=verification,
                message=(
                    f"最终距离 {final_distance:.2f}m 或朝向误差 "
                    f"{math.degrees(angle_error):.1f}deg 未通过"
                ),
            )
        return finish(
            task_status="arrived_verified",
            target_distance_m=final_distance,
            **stop_evidence,
            verification_frame_timestamp=verified.frame_timestamp,
            planner_goal_reached=planner_goal_reached,
            verification=verification,
            message=(
                f"{'规划器到达' if planner_goal_reached else '位置到达'}后，"
                f"新 RGB-D 已验证目标距离 {final_distance:.2f}m，"
                f"朝向误差 {math.degrees(angle_error):.1f}deg"
            ),
        )


def localize_person_in_camera(
    bbox: tuple[float, float, float, float],
    depth: np.ndarray[Any, np.dtype[Any]],
    camera_info: CameraInfo,
    *,
    min_depth_m: float = 0.3,
    max_depth_m: float = 8.0,
    cluster_radius_m: float = 0.35,
    min_points: int = 30,
) -> PersonPosition | None:
    """Lift a torso-centred bbox crop into the optical frame."""

    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        return None
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    roi_x1 = max(0, round(x1 + width * 0.2))
    roi_x2 = min(values.shape[1], round(x1 + width * 0.8))
    roi_y1 = max(0, round(y1 + height * 0.15))
    roi_y2 = min(values.shape[0], round(y1 + height * 0.8))
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None
    roi = values[roi_y1:roi_y2, roi_x1:roi_x2]
    valid = np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)
    if int(np.count_nonzero(valid)) < min_points:
        return None
    valid_depths = roi[valid]
    seed = float(np.percentile(valid_depths, 30))
    cluster = valid & (np.abs(roi - seed) <= cluster_radius_m)
    rows, columns = np.nonzero(cluster)
    depths = roi[cluster]
    if depths.size < min_points:
        return None
    u = columns.astype(np.float64) + roi_x1
    v = rows.astype(np.float64) + roi_y1
    fx, fy, cx, cy = (
        float(camera_info.K[0]),
        float(camera_info.K[4]),
        float(camera_info.K[2]),
        float(camera_info.K[5]),
    )
    if fx <= 0.0 or fy <= 0.0:
        return None
    z = depths.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    point = Vector3(float(np.median(x)), float(np.median(y)), float(np.median(z)))
    return PersonPosition(
        point=point,
        depth_m=point.z,
        valid_points=int(depths.size),
        depth_spread_m=float(np.std(depths)),
    )


def point_to_parent(transform: Transform, point: Vector3) -> Vector3:
    return transform.rotation.rotate_vector(point) + transform.translation


def verify_world_target_with_depth(
    target_world: Vector3,
    depth_image: Any,
    camera_info: CameraInfo,
    world_from_camera: Transform,
    robot_at_frame: Transform,
    *,
    depth_tolerance_m: float = 0.35,
    patch_radius_pixels: int = 10,
    min_points: int = 20,
    allow_vertical_surface: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> VisualTargetObservation | None:
    """Verify a locked, static world target in one fresh depth frame.

    This is deliberately not a semantic re-detection.  The initial VLM call
    establishes identity; the post-stop RGB-D frame must contain a surface at
    the projection of that same world point.  The observed depth is then
    reprojected to produce the final sensor-derived world point.
    """

    if diagnostics is not None:
        diagnostics.clear()

    def reject(reason: str, **details: Any) -> None:
        if diagnostics is not None:
            diagnostics.update(reason=reason, **details)

    try:
        camera_from_world = world_from_camera.inverse()
        expected_camera = point_to_parent(camera_from_world, target_world)
        expected_depth = float(expected_camera.z)
        fx = float(camera_info.K[0])
        fy = float(camera_info.K[4])
        cx = float(camera_info.K[2])
        cy = float(camera_info.K[5])
        depth = np.asarray(depth_image.data, dtype=np.float32)
    except (AttributeError, TypeError, ValueError, OverflowError):
        reject("invalid_input")
        return None
    if (
        depth.ndim != 2
        or expected_depth <= 0.1
        or not all(math.isfinite(value) for value in (expected_depth, fx, fy, cx, cy))
        or fx <= 0.0
        or fy <= 0.0
    ):
        reject("invalid_projection_input")
        return None
    projected_u = fx * float(expected_camera.x) / expected_depth + cx
    projected_v = fy * float(expected_camera.y) / expected_depth + cy
    if not (
        math.isfinite(projected_u)
        and math.isfinite(projected_v)
        and 0.0 <= projected_u < depth.shape[1]
    ):
        reject(
            "horizontal_projection_out_of_frame",
            projected_u=round(float(projected_u), 2),
            projected_v=round(float(projected_v), 2),
            expected_depth_m=round(expected_depth, 3),
        )
        return None
    center_u = int(round(projected_u))
    radius = max(1, int(patch_radius_pixels))
    minimum_points = max(1, int(min_points))
    tolerance = float(depth_tolerance_m)

    def matching_surface(
        row_start: int,
        row_stop: int,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        column_start = max(0, center_u - radius)
        column_stop = min(depth.shape[1], center_u + radius + 1)
        region = depth[row_start:row_stop, column_start:column_stop]
        mask = (
            np.isfinite(region)
            & (region > 0.1)
            & (np.abs(region - expected_depth) <= tolerance)
        )
        rows, columns = np.nonzero(mask)
        return (
            rows.astype(np.float64) + row_start,
            columns.astype(np.float64) + column_start,
            region[mask],
        )

    verification_mode = "exact_anchor"
    matching_rows = np.asarray([], dtype=np.float64)
    matching_columns = np.asarray([], dtype=np.float64)
    matching_depths = np.asarray([], dtype=np.float32)
    if 0.0 <= projected_v < depth.shape[0]:
        center_v = int(round(projected_v))
        matching_rows, matching_columns, matching_depths = matching_surface(
            max(0, center_v - radius),
            min(depth.shape[0], center_v + radius + 1),
        )
    if matching_depths.size < minimum_points and allow_vertical_surface:
        # A VLM may anchor a tall door near its lintel.  At close range that
        # exact 3-D point can leave the vertical field of view even though the
        # same locked planar surface is still plainly present.  Search only
        # along the already locked horizontal bearing and expected depth; this
        # adds no new semantic inference or scene-layout prior.
        verification_mode = "vertical_surface"
        matching_rows, matching_columns, matching_depths = matching_surface(
            0,
            depth.shape[0],
        )
    if matching_depths.size < minimum_points:
        reject(
            "insufficient_matching_depth",
            verification_mode=verification_mode,
            projected_u=round(float(projected_u), 2),
            projected_v=round(float(projected_v), 2),
            expected_depth_m=round(expected_depth, 3),
            matching_points=int(matching_depths.size),
        )
        return None
    observed_depth = float(np.median(matching_depths))
    observed_u = float(np.median(matching_columns))
    observed_v = float(np.median(matching_rows))
    observed_camera = Vector3(
        (observed_u - cx) * observed_depth / fx,
        (observed_v - cy) * observed_depth / fy,
        observed_depth,
    )
    observed_world = point_to_parent(world_from_camera, observed_camera)
    if verification_mode == "vertical_surface":
        identity_error = math.hypot(
            float(observed_world.x) - float(target_world.x),
            float(observed_world.y) - float(target_world.y),
        )
    else:
        identity_error = math.sqrt(
            (float(observed_world.x) - float(target_world.x)) ** 2
            + (float(observed_world.y) - float(target_world.y)) ** 2
            + (float(observed_world.z) - float(target_world.z)) ** 2
        )
    if identity_error > depth_tolerance_m:
        reject(
            "locked_target_mismatch",
            verification_mode=verification_mode,
            identity_error_m=round(identity_error, 3),
            expected_depth_m=round(expected_depth, 3),
            observed_depth_m=round(observed_depth, 3),
        )
        return None
    if diagnostics is not None:
        diagnostics.update(
            reason="verified",
            verification_mode=verification_mode,
            projected_u=round(float(projected_u), 2),
            projected_v=round(float(projected_v), 2),
            expected_depth_m=round(expected_depth, 3),
            observed_depth_m=round(observed_depth, 3),
            identity_error_m=round(identity_error, 3),
            matching_points=int(matching_depths.size),
        )
    return VisualTargetObservation(
        world_point=observed_world,
        estimate=PersonPosition(
            point=observed_camera,
            depth_m=observed_depth,
            valid_points=int(matching_depths.size),
            depth_spread_m=float(np.std(matching_depths)),
        ),
        frame_timestamp=float(depth_image.ts),
        robot_transform_at_frame=robot_at_frame,
    )


def make_standoff_goal(
    robot_world: Vector3,
    person_world: Vector3,
    distance: float,
) -> PoseStamped | None:
    dx = robot_world.x - person_world.x
    dy = robot_world.y - person_world.y
    separation = math.hypot(dx, dy)
    if separation < 1e-6:
        return None
    goal_x = person_world.x + distance * dx / separation
    goal_y = person_world.y + distance * dy / separation
    goal_yaw = math.atan2(person_world.y - goal_y, person_world.x - goal_x)
    return PoseStamped(
        position=Vector3(goal_x, goal_y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, goal_yaw)),
        frame_id="world",
    )


class HeadDepthSourceConfig(ModuleConfig):
    path: Path = Field(default_factory=configured_depth_path)
    poll_hz: float = Field(default=10.0, gt=0.0, le=30.0)
    max_age_s: float = Field(default=1.5, gt=0.1, le=5.0)


class HeadDepthSource(Module):
    """Publish the worker's side-channel depth as a normal DimOS Image stream."""

    config: HeadDepthSourceConfig
    depth_image: Out[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_timestamp = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        self._thread = Thread(target=self._run, name="head-depth-source", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = 1.0 / self.config.poll_hz
        while not self._stop_event.wait(interval):
            try:
                with np.load(self.config.path, allow_pickle=False) as archive:
                    timestamp = float(archive["timestamp"])
                    if (
                        timestamp <= self._last_timestamp
                        or time.time() - timestamp > self.config.max_age_s
                    ):
                        continue
                    depth = np.asarray(archive["depth"], dtype=np.float32).copy()
            except (OSError, ValueError, KeyError, EOFError):
                continue
            if depth.ndim != 2 or not depth.size:
                continue
            self._last_timestamp = timestamp
            self.depth_image.publish(
                Image(
                    data=depth,
                    format=ImageFormat.DEPTH,
                    frame_id="camera_optical",
                    ts=timestamp,
                )
            )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        super().stop()


class ApproachPersonConfig(ModuleConfig):
    camera_info: CameraInfo
    sync_tolerance_s: float = Field(default=0.25, gt=0.0, le=1.0)
    min_depth_points: int = Field(default=30, ge=10)
    standoff_tolerance_m: float = Field(default=0.3, gt=0.0, le=0.6)
    max_final_angle_degrees: float = Field(default=30.0, gt=0.0, le=90.0)
    viewpoint_search_yaw_degrees: float = Field(default=180.0, ge=120.0, le=180.0)
    viewpoint_search_min_yaw_degrees: float = Field(default=160.0, ge=90.0, le=180.0)
    viewpoint_search_max_translation_m: float = Field(default=0.20, ge=0.05, le=0.4)
    viewpoint_search_yaw_rate: float = Field(default=0.45, ge=0.15, le=0.55)
    viewpoint_search_timeout_s: float = Field(default=15.0, ge=3.0, le=20.0)
    object_scan_step_degrees: float = Field(default=60.0, ge=30.0, le=90.0)
    object_scan_total_degrees: float = Field(default=360.0, ge=330.0, le=360.0)
    object_scan_timeout_s: float = Field(default=75.0, ge=20.0, le=120.0)
    object_scan_yaw_rate: float = Field(default=0.80, ge=0.40, le=0.80)
    object_scan_turn_creep_mps: float = Field(default=0.20, ge=0.05, le=0.25)
    object_scan_creep_period_s: float = Field(default=2.0, ge=0.5, le=3.0)
    object_scan_max_translation_m: float = Field(default=0.35, ge=0.20, le=0.50)
    object_scan_stationary_timeout_s: float = Field(default=3.0, ge=0.5, le=5.0)
    visual_approach_stationary_timeout_s: float = Field(
        default=3.0,
        ge=0.5,
        le=7.0,
    )
    visual_approach_max_stop_latency_s: float = Field(
        default=3.0,
        ge=0.5,
        le=7.0,
    )
    visibility_bottom_margin_pixels: int = Field(default=20, ge=0, le=90)
    visibility_depth_to_planar_scale: float = Field(default=1.1, ge=1.0, le=1.5)
    verification_depth_tolerance_m: float = Field(default=0.35, gt=0.1, le=0.6)
    stationary_speed_threshold_mps: float = Field(default=0.025, gt=0.0, le=0.1)
    stationary_sample_count: int = Field(default=2, ge=2, le=5)
    follow_control_hz: float = Field(default=10.0, ge=4.0, le=20.0)
    follow_frame_hold_s: float = Field(default=0.35, ge=0.2, le=0.5)
    # Isaac RGB-D is transported as large fragmented LCM messages. A short
    # burst may be dropped even while the source camera remains healthy.
    # Motion is already held after 0.35 s; allow recovery without reselecting
    # a person before declaring the task lost.
    follow_tracking_lost_s: float = Field(default=1.5, ge=0.5, le=3.0)
    follow_sensor_gap_timeout_s: float = Field(default=3.0, ge=1.5, le=5.0)
    follow_unknown_risk_timeout_s: float = Field(default=2.0, ge=0.5, le=3.0)
    follow_verification_timeout_s: float = Field(default=1.5, ge=0.5, le=3.0)
    follow_distance_tolerance_m: float = Field(default=0.35, ge=0.2, le=0.6)
    follow_max_final_angle_degrees: float = Field(default=30.0, ge=10.0, le=60.0)
    follow_min_tracking_coverage: float = Field(default=0.9, ge=0.7, le=1.0)
    follow_stationary_timeout_s: float = Field(default=1.5, ge=0.5, le=3.0)
    follow_max_stop_latency_s: float = Field(default=0.5, ge=0.5, le=0.5)
    frontier_map_max_age_s: float = Field(default=2.0, ge=0.5, le=5.0)
    frontier_goal_timeout_s: float = Field(default=45.0, ge=10.0, le=90.0)
    frontier_map_settle_s: float = Field(default=1.0, ge=0.2, le=3.0)
    frontier_initial_stability_s: float = Field(default=0.75, ge=0.25, le=2.0)
    frontier_stability_timeout_s: float = Field(default=6.0, ge=1.0, le=10.0)
    frontier_known_free_radius_m: float = Field(default=0.75, ge=0.75, le=1.25)
    semantic_goal_known_free_radius_m: float = Field(
        default=0.25,
        ge=0.25,
        le=0.5,
    )
    frontier_inward_offset_m: float = Field(default=0.9, ge=0.75, le=1.5)
    frontier_travel_cost_weight: float = Field(default=0.35, ge=0.1, le=1.0)
    frontier_min_goal_path_distance_m: float = Field(
        default=0.6,
        ge=0.4,
        le=1.5,
    )
    frontier_max_goal_path_distance_m: float = Field(
        default=0.7,
        ge=0.6,
        le=2.5,
    )
    frontier_stationary_timeout_s: float = Field(default=2.0, ge=0.5, le=3.0)
    frontier_checkpoint_tolerance_m: float = Field(
        default=0.25,
        ge=0.10,
        le=0.35,
    )
    frontier_arrival_grace_s: float = Field(default=0.75, ge=0.25, le=2.0)


class ApproachPersonSkillContainer(Module):
    """Approach visible targets via aligned RGB-D and global planning."""

    config: ApproachPersonConfig
    color_image: In[Image]
    depth_image: In[Image]
    odom: In[PoseStamped]
    exploration_costmap: In[OccupancyGrid]
    goal_reached: In[Bool]
    stop_movement: Out[Bool]
    nav_cmd_vel: Out[Twist]
    cmd_vel: Out[Twist]
    _navigation: NavigationInterfaceSpec
    _spatial_memory: SpatialMemorySpec | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from dimos.models.vl import qwen as qwen_module

        self._vl_model = qwen_module.QwenVlModel()
        self._color_frames: deque[Image] = deque(maxlen=20)
        self._depth_frames: deque[Image] = deque(maxlen=20)
        self._odom_frames: deque[PoseStamped] = deque(maxlen=100)
        self._lock = RLock()
        self._cancel_event = Event()
        self._follow_cancel_event = Event()
        self._follow_active = False
        self._last_vision_request: dict[str, Any] = {}
        self._last_localization_pair: tuple[Image, Image] | None = None
        self._last_verification: dict[str, Any] = {}
        self._last_viewpoint_search: dict[str, Any] = {}
        self._latest_exploration_costmap: OccupancyGrid | None = None
        self._exploration_costmap_received_at: float | None = None
        self._frontier_goal_reached_event = Event()
        self._frontier_goal_reached_at = 0.0
        self._risk_monitor = LiveCostmapRiskMonitor()
        self._lidar_risk_monitor = FreshLidarRiskMonitor()
        self._risk_feed = LiveCostmapRiskFeed(self._risk_monitor)

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color)))
        self.register_disposable(Disposable(self.depth_image.subscribe(self._on_depth)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        if self.exploration_costmap.transport is not None:
            self.register_disposable(
                Disposable(
                    self.exploration_costmap.subscribe(self._on_exploration_costmap)
                )
            )
        if self.goal_reached.transport is not None:
            self.register_disposable(
                Disposable(self.goal_reached.subscribe(self._on_goal_reached))
            )
        self._risk_feed.start()
        self.register_disposable(Disposable(self._risk_feed.stop))

    def _on_color(self, image: Image) -> None:
        with self._lock:
            self._color_frames.append(image)

    def _on_depth(self, image: Image) -> None:
        with self._lock:
            self._depth_frames.append(image)

    def _on_odom(self, pose: PoseStamped) -> None:
        with self._lock:
            self._odom_frames.append(pose)

    def _on_exploration_costmap(self, costmap: OccupancyGrid) -> None:
        try:
            cells = np.asarray(costmap.grid, dtype=np.int8)
            valid = bool(
                cells.ndim == 2
                and cells.size > 0
                and np.any(cells == -1)
                and np.any(cells == 0)
                and str(costmap.frame_id) in {"world", "map"}
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            valid = False
        with self._lock:
            self._latest_exploration_costmap = costmap if valid else None
            self._exploration_costmap_received_at = time.monotonic() if valid else None

    def _on_goal_reached(self, reached: Bool) -> None:
        if reached.data:
            with self._lock:
                self._frontier_goal_reached_at = time.monotonic()
            self._frontier_goal_reached_event.set()

    def wait_until_stationary(self, *, after: float, timeout: float) -> float | None:
        """Confirm physical stop from consecutive fresh odometry intervals."""

        deadline = time.monotonic() + max(0.0, float(timeout))
        previous: PoseStamped | None = None
        cursor = float(after)
        consecutive = 0
        required = int(self.config.stationary_sample_count)
        threshold = float(self.config.stationary_speed_threshold_mps)
        while True:
            with self._lock:
                frames = sorted(self._odom_frames, key=lambda item: float(item.ts))
            if previous is None:
                prior = [frame for frame in frames if float(frame.ts) <= after]
                if prior:
                    previous = prior[-1]
            for current in frames:
                current_ts = float(current.ts)
                if current_ts <= cursor:
                    continue
                if previous is not None:
                    elapsed = current_ts - float(previous.ts)
                    if elapsed > 0.0:
                        distance = math.hypot(
                            float(current.position.x) - float(previous.position.x),
                            float(current.position.y) - float(previous.position.y),
                        )
                        if distance / elapsed <= threshold:
                            consecutive += 1
                            if consecutive >= required:
                                return current_ts
                        else:
                            consecutive = 0
                previous = current
                cursor = current_ts
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)

    def _synced_pair(self, after: float = 0.0) -> tuple[Image, Image] | None:
        with self._lock:
            colors = [frame for frame in self._color_frames if frame.ts > after]
            depths = [frame for frame in self._depth_frames if frame.ts > after]
        if not colors or not depths:
            return None
        best = min(
            ((color, depth) for color in colors for depth in depths),
            key=lambda pair: abs(pair[0].ts - pair[1].ts),
        )
        if abs(best[0].ts - best[1].ts) > self.config.sync_tolerance_s:
            return None
        return best

    def _wait_for_pair(
        self, after: float = 0.0, timeout: float = 5.0
    ) -> tuple[Image, Image] | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pair = self._synced_pair(after)
            if pair is not None:
                return pair
            time.sleep(0.05)
        return None

    def _locate_world(
        self,
        query: str,
        pair: tuple[Image, Image],
        *,
        crop_center: tuple[float, float] | None = None,
    ) -> tuple[Vector3, PersonPosition, Transform] | None:
        color, depth = pair
        frame_timestamp = float(depth.ts)
        # Capture both transforms before the potentially slow VLM call.  The
        # pinned TF service retains only ten seconds, while a valid remote VLM
        # response can take longer than that.
        world_from_camera = self.tf.get(
            "world",
            depth.frame_id or "camera_optical",
            frame_timestamp,
            time_tolerance=self.config.sync_tolerance_s,
            forward_tolerance=0.5,
        )
        robot_at_frame = self.tf.get(
            "world",
            "base_link",
            frame_timestamp,
            time_tolerance=max(0.3, self.config.sync_tolerance_s),
            forward_tolerance=0.5,
        )
        if world_from_camera is None or robot_at_frame is None:
            with self._lock:
                self._last_vision_request = {
                    "localization_stage": (
                        "camera_tf_unavailable"
                        if world_from_camera is None
                        else "robot_tf_unavailable"
                    ),
                    "frame_timestamp": frame_timestamp,
                }
            return None
        localization_color = color
        localization_depth = np.asarray(depth.data)
        localization_camera = self.config.camera_info
        crop_metadata: dict[str, Any] = {}
        if crop_center is not None:
            pixels = np.asarray(color.data)
            if pixels.ndim != 3 or localization_depth.ndim != 2:
                return None
            crop_size = min(256, pixels.shape[0], pixels.shape[1])
            half = crop_size // 2
            center_u = int(round(float(crop_center[0])))
            center_v = int(round(float(crop_center[1])))
            left = max(0, min(pixels.shape[1] - crop_size, center_u - half))
            top = max(0, min(pixels.shape[0] - crop_size, center_v - half))
            right = left + crop_size
            bottom = top + crop_size
            cropped_pixels = np.ascontiguousarray(
                pixels[top:bottom, left:right]
            )
            # Preserve the original depth grid and intrinsics, but enlarge the
            # semantic input so compact objects do not disappear during the
            # provider's image tokenization.  Returned boxes are mapped back
            # to the measured crop before depth localization.
            crop_scale = 3
            localization_color = Image(
                data=np.ascontiguousarray(
                    np.repeat(
                        np.repeat(cropped_pixels, crop_scale, axis=0),
                        crop_scale,
                        axis=1,
                    )
                ),
                format=color.format,
                frame_id=color.frame_id,
                ts=float(color.ts),
            )
            localization_depth = np.ascontiguousarray(
                localization_depth[top:bottom, left:right]
            )
            source_camera = self.config.camera_info
            localization_camera = CameraInfo.from_intrinsics(
                float(source_camera.K[0]),
                float(source_camera.K[4]),
                float(source_camera.K[2]) - left,
                float(source_camera.K[5]) - top,
                crop_size,
                crop_size,
                frame_id=source_camera.frame_id,
            )
            crop_metadata = {
                "targeted_crop": [left, top, right, bottom],
                "targeted_crop_center": [
                    round(float(crop_center[0]), 2),
                    round(float(crop_center[1]), 2),
                ],
                "semantic_crop_scale": crop_scale,
            }
        semantic_query = _semantic_detection_query(query)
        if semantic_query != query:
            crop_metadata["semantic_query_augmented"] = True
        try:
            bbox = get_object_bbox_from_image(
                self._vl_model,
                localization_color,
                semantic_query,
            )
        finally:
            reader = getattr(self._vl_model, "last_request_metadata", None)
            try:
                metadata = reader() if callable(reader) else {}
            except Exception:  # noqa: BLE001 - tracing must not mask VLM outcome
                metadata = {}
            with self._lock:
                self._last_vision_request = (
                    dict(metadata) if isinstance(metadata, dict) else {}
                )
                self._last_vision_request.update(crop_metadata)
        if bbox is None:
            with self._lock:
                self._last_vision_request["localization_stage"] = "bbox_not_found"
            return None
        if crop_center is not None:
            bbox = tuple(float(value) / float(crop_scale) for value in bbox)
        estimate = localize_person_in_camera(
            bbox,
            localization_depth,
            localization_camera,
            min_points=self.config.min_depth_points,
        )
        if estimate is None:
            values = localization_depth
            finite = values[np.isfinite(values)] if values.size else np.asarray([])
            with self._lock:
                self._last_vision_request.update(
                    {
                        "localization_stage": "depth_localization_failed",
                        "bbox": [round(float(value), 3) for value in bbox],
                        "depth_shape": list(values.shape),
                        "finite_depth_points": int(finite.size),
                        "depth_min_m": (
                            round(float(finite.min()), 3) if finite.size else None
                        ),
                        "depth_max_m": (
                            round(float(finite.max()), 3) if finite.size else None
                        ),
                    }
                )
            return None
        world_point = point_to_parent(world_from_camera, estimate.point)
        with self._lock:
            self._last_vision_request.update(
                {
                    "localization_stage": "localized",
                    "bbox": [round(float(value), 3) for value in bbox],
                    "depth_m": round(float(estimate.depth_m), 3),
                    "valid_depth_points": int(estimate.valid_points),
                    "target_world": [
                        round(float(world_point.x), 3),
                        round(float(world_point.y), 3),
                        round(float(world_point.z), 3),
                    ],
                }
            )
        return (
            world_point,
            estimate,
            robot_at_frame,
        )

    def _robot_transform(self, timestamp: float | None = None) -> Transform | None:
        return self.tf.get(
            "world",
            "base_link",
            timestamp,
            time_tolerance=max(0.3, self.config.sync_tolerance_s),
        )

    def _safe_stop(self) -> None:
        # Publish zero velocity before requesting cancellation so safety does
        # not depend on a planner RPC returning.  In a real run the remote
        # cancel_goal() call can deadlock with the planner's own arrival
        # callback; the autoconnected stop_movement topic performs the same
        # cancellation inside the navigation process without blocking this
        # closed-loop skill.
        self.nav_cmd_vel.publish(Twist())
        self.cmd_vel.publish(Twist())
        self.stop_movement.publish(Bool(data=True))

    def locate_target(
        self,
        query: str,
        *,
        after: float = 0.0,
    ) -> VisualTargetObservation | None:
        with self._lock:
            self._last_vision_request = {}
        pair = self._wait_for_pair(after=after)
        if pair is None:
            return None
        with self._lock:
            self._last_localization_pair = pair
        try:
            located = self._locate_world(query, pair)
        except Exception as error:  # noqa: BLE001 - VLM failures end this branch
            self._safe_stop()
            with self._lock:
                self._last_vision_request.setdefault(
                    "error",
                    f"{type(error).__name__}: {error}"[:500],
                )
                self._last_vision_request["localization_stage"] = "vlm_error"
            return None
        if located is None:
            return None
        world_point, estimate, robot_at_frame = located
        return VisualTargetObservation(
            world_point=world_point,
            estimate=estimate,
            frame_timestamp=float(pair[1].ts),
            robot_transform_at_frame=robot_at_frame,
        )

    def locate_target_multiscale(
        self,
        query: str,
        *,
        after: float = 0.0,
    ) -> VisualTargetObservation | None:
        """Search one stationary RGB-D snapshot without scene-specific hints.

        Full-frame detection remains authoritative for ordinary objects.  A
        compact bottle may occupy only a few source pixels after a safe
        frontier leg, so fixed overlapping image tiles provide bounded
        semantic magnification.  Every accepted box is still localized
        against the untouched measured depth crop.
        """

        with self._lock:
            self._last_vision_request = {}
        pair = self._wait_for_pair(after=after)
        if pair is None:
            return None
        with self._lock:
            self._last_localization_pair = pair
        color, depth = pair
        centers: list[tuple[float, float] | None] = [None]
        if _is_compact_bottle_target(query):
            width, height = int(color.width), int(color.height)
            crop_size = min(256, width, height)
            half = crop_size / 2.0
            centers.extend(
                (x, max(half, float(height) - half))
                for x in (
                    half,
                    min(float(width) - half, float(width) * 0.6),
                    float(width) - half,
                )
            )
        attempts: list[dict[str, Any]] = []
        for center in centers:
            try:
                located = self._locate_world(
                    query,
                    pair,
                    crop_center=center,
                )
            except Exception as error:  # noqa: BLE001 - fail closed on VLM error
                self._safe_stop()
                with self._lock:
                    self._last_vision_request.setdefault(
                        "error",
                        f"{type(error).__name__}: {error}"[:500],
                    )
                    self._last_vision_request["localization_stage"] = "vlm_error"
                    self._last_vision_request["multiscale_attempts"] = attempts
                return None
            metadata = self.vision_request_metadata()
            compact_attempt = _compact_vision_attempt(metadata)
            attempts.append(compact_attempt)
            if located is None:
                continue
            world_point, estimate, robot_at_frame = located
            with self._lock:
                self._last_vision_request["multiscale_attempts"] = attempts
            return VisualTargetObservation(
                world_point=world_point,
                estimate=estimate,
                frame_timestamp=float(depth.ts),
                robot_transform_at_frame=robot_at_frame,
            )
        with self._lock:
            self._last_vision_request["multiscale_attempts"] = attempts
        return None

    def locate_target_near_world(
        self,
        query: str,
        target_world: Vector3,
        *,
        after: float = 0.0,
    ) -> VisualTargetObservation | None:
        """Re-detect a locked compact target in a bounded projected crop.

        The crop is derived only from the original RGB-D world observation
        and a newer measured camera transform.  It enlarges a small bottle for
        semantic verification without introducing authored object coordinates
        or accepting depth from outside the returned bounding box.
        """

        with self._lock:
            self._last_vision_request = {}
        pair = self._wait_for_pair(after=after)
        if pair is None:
            return None
        color, depth = pair
        world_from_camera = self.tf.get(
            "world",
            depth.frame_id or "camera_optical",
            float(depth.ts),
            time_tolerance=self.config.sync_tolerance_s,
            forward_tolerance=0.5,
        )
        if world_from_camera is None:
            return None
        expected = point_to_parent(world_from_camera.inverse(), target_world)
        if float(expected.z) <= 0.1:
            return None
        camera = self.config.camera_info
        projected_u = (
            float(camera.K[0]) * float(expected.x) / float(expected.z)
            + float(camera.K[2])
        )
        projected_v = (
            float(camera.K[4]) * float(expected.y) / float(expected.z)
            + float(camera.K[5])
        )
        if not (
            math.isfinite(projected_u)
            and math.isfinite(projected_v)
            and 0.0 <= projected_u < int(camera.width)
            and 0.0 <= projected_v < int(camera.height)
        ):
            return None
        with self._lock:
            self._last_localization_pair = pair
        try:
            located = self._locate_world(
                query,
                pair,
                crop_center=(projected_u, projected_v),
            )
        except Exception as error:  # noqa: BLE001 - fail closed like full-frame VLM
            self._safe_stop()
            with self._lock:
                self._last_vision_request.setdefault(
                    "error",
                    f"{type(error).__name__}: {error}"[:500],
                )
                self._last_vision_request["localization_stage"] = "vlm_error"
            return None
        if located is None:
            return None
        world_point, estimate, robot_at_frame = located
        return VisualTargetObservation(
            world_point=world_point,
            estimate=estimate,
            frame_timestamp=float(depth.ts),
            robot_transform_at_frame=robot_at_frame,
        )

    def vision_request_metadata(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_vision_request)

    def robot_transform(self, timestamp: float | None = None) -> Transform | None:
        return self._robot_transform(timestamp)

    def recommend_standoff_distance(
        self,
        observation: VisualTargetObservation,
        requested_distance: float,
    ) -> float:
        """Keep a locked low target inside the post-stop RGB-D frame."""

        try:
            point = observation.estimate.point
            optical_y = float(point.y)
            fy = float(self.config.camera_info.K[4])
            cy = float(self.config.camera_info.K[5])
            image_height = int(self.config.camera_info.height)
            bottom_limit = image_height - int(
                self.config.visibility_bottom_margin_pixels
            )
            usable_vertical = float(bottom_limit) - cy
        except (AttributeError, TypeError, ValueError, OverflowError):
            return float(requested_distance)
        if (
            optical_y <= 0.0
            or fy <= 0.0
            or usable_vertical <= 1.0
            or not all(math.isfinite(value) for value in (optical_y, fy, cy))
        ):
            return float(requested_distance)
        minimum_optical_depth = fy * optical_y / usable_vertical
        visibility_safe_distance = minimum_optical_depth * float(
            self.config.visibility_depth_to_planar_scale
        )
        return min(
            3.0,
            max(float(requested_distance), visibility_safe_distance),
        )

    def set_goal(self, goal: PoseStamped) -> bool:
        return bool(self._navigation.set_goal(goal))

    def is_goal_reached(self) -> bool:
        return bool(self._navigation.is_goal_reached())

    def navigation_state(self) -> NavigationState:
        return self._navigation.get_state()

    def risk_state(self) -> str:
        lidar_risk = self._lidar_risk_monitor.state()
        if lidar_risk != "unknown":
            return lidar_risk
        return self._risk_monitor.state(self._robot_transform())

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def safe_stop(self) -> None:
        self._safe_stop()

    def search_viewpoint(self) -> float | None:
        """Acquire an opposite first-person view with a measured in-place scan."""

        search_started = time.monotonic()
        start = self._robot_transform()
        initial_risk = self.risk_state() if start is not None else "unknown"
        last_risk = initial_risk
        reason = "pose_or_risk_unavailable"
        observed_yaw_degrees = 0.0
        max_translation_m = 0.0
        stationary_confirmed_at: float | None = None

        def record(completed: bool) -> None:
            with self._lock:
                self._last_viewpoint_search = {
                    "completed": completed,
                    "reason": reason,
                    "target_yaw_degrees": float(
                        self.config.viewpoint_search_yaw_degrees
                    ),
                    "minimum_yaw_degrees": float(
                        self.config.viewpoint_search_min_yaw_degrees
                    ),
                    "observed_yaw_degrees": round(observed_yaw_degrees, 2),
                    "translation_m": round(max_translation_m, 3),
                    "initial_risk": initial_risk,
                    "final_risk": last_risk,
                    "stationary_confirmed": stationary_confirmed_at is not None,
                    "stationary_confirmed_at": stationary_confirmed_at,
                    "elapsed_s": round(
                        max(0.0, time.monotonic() - search_started),
                        3,
                    ),
                }

        if start is None or initial_risk in {"unknown", "critical"}:
            self._safe_stop()
            record(False)
            return None

        start_yaw = float(start.rotation.to_euler().yaw)
        minimum_yaw_degrees = min(
            float(self.config.viewpoint_search_min_yaw_degrees),
            float(self.config.viewpoint_search_yaw_degrees),
        )
        deadline = search_started + float(self.config.viewpoint_search_timeout_s)
        reached = False
        reason = "scan_timeout"
        try:
            while time.monotonic() < deadline:
                if self.is_cancelled():
                    reason = "cancelled"
                    break
                risk = self.risk_state()
                last_risk = risk
                if risk in {"unknown", "critical"}:
                    reason = f"risk_{risk}"
                    break
                current = self._robot_transform()
                if current is not None:
                    current_yaw = float(current.rotation.to_euler().yaw)
                    observed_yaw_degrees = math.degrees(
                        abs(
                            math.atan2(
                                math.sin(current_yaw - start_yaw),
                                math.cos(current_yaw - start_yaw),
                            )
                        )
                    )
                    max_translation_m = max(
                        max_translation_m,
                        math.hypot(
                            float(current.translation.x) - float(start.translation.x),
                            float(current.translation.y) - float(start.translation.y),
                        ),
                    )
                    if max_translation_m > float(
                        self.config.viewpoint_search_max_translation_m
                    ):
                        reason = "scan_translation_exceeded"
                        break
                    if observed_yaw_degrees >= minimum_yaw_degrees:
                        reached = True
                        reason = "measured_yaw_reached"
                        break
                commanded_yaw_rate = float(self.config.viewpoint_search_yaw_rate)
                if risk == "warning":
                    commanded_yaw_rate = min(commanded_yaw_rate, 0.25)
                self.nav_cmd_vel.publish(
                    Twist(angular=Vector3(0.0, 0.0, commanded_yaw_rate))
                )
                time.sleep(0.05)
        finally:
            self._safe_stop()
            stop_command_completed_at = time.time()

        if reached:
            stationary_confirmed_at = self.wait_until_stationary(
                after=stop_command_completed_at,
                timeout=0.5,
            )
            if stationary_confirmed_at is None:
                reached = False
                reason = "stationary_confirmation_failed"
        record(reached)
        return stationary_confirmed_at if reached else None

    def viewpoint_search_metadata(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_viewpoint_search)

    def search_full_rotation(
        self,
        query: str,
        *,
        deadline: float,
    ) -> VisualTargetObservation | None:
        """Scan a full circle in measured steps and localize after every stop."""

        started_at = time.monotonic()
        start = self._robot_transform()
        initial_risk = self.risk_state() if start is not None else "unknown"
        step_degrees = float(self.config.object_scan_step_degrees)
        total_degrees = float(self.config.object_scan_total_degrees)
        scan_deadline = min(
            float(deadline),
            started_at + float(self.config.object_scan_timeout_s),
        )
        checkpoints: list[dict[str, Any]] = []
        cumulative_yaw_degrees = 0.0
        max_translation_m = 0.0
        last_risk = initial_risk
        reason = "pose_or_risk_unavailable"
        target_found = False
        completed = False

        def record() -> None:
            with self._lock:
                self._last_viewpoint_search = {
                    "completed": completed,
                    "reason": reason,
                    "scan_mode": "segmented_full_rotation",
                    "target_yaw_degrees": total_degrees,
                    "step_yaw_degrees": step_degrees,
                    "observed_yaw_degrees": round(cumulative_yaw_degrees, 2),
                    "translation_m": round(max_translation_m, 3),
                    "initial_risk": initial_risk,
                    "final_risk": last_risk,
                    "target_found": target_found,
                    "headings_checked": len(checkpoints),
                    "checkpoints": list(checkpoints),
                    "elapsed_s": round(
                        max(0.0, time.monotonic() - started_at),
                        3,
                    ),
                }

        if start is None or initial_risk in {"unknown", "critical"}:
            self._safe_stop()
            record()
            return None

        previous_yaw = float(start.rotation.to_euler().yaw)
        next_checkpoint = min(step_degrees, total_degrees)
        reason = "scan_timeout"
        try:
            while time.monotonic() < scan_deadline:
                if self.is_cancelled():
                    reason = "cancelled"
                    break
                risk = self.risk_state()
                last_risk = risk
                if risk in {"unknown", "critical"}:
                    reason = f"risk_{risk}"
                    break
                current = self._robot_transform()
                if current is None:
                    reason = "pose_unavailable"
                    break
                current_yaw = float(current.rotation.to_euler().yaw)
                yaw_delta = math.atan2(
                    math.sin(current_yaw - previous_yaw),
                    math.cos(current_yaw - previous_yaw),
                )
                # The scan commands only positive yaw. Ignore tiny reverse
                # odometry jitter so wraparound at +/-pi remains measurable.
                cumulative_yaw_degrees += max(0.0, math.degrees(yaw_delta))
                previous_yaw = current_yaw
                max_translation_m = max(
                    max_translation_m,
                    math.hypot(
                        float(current.translation.x) - float(start.translation.x),
                        float(current.translation.y) - float(start.translation.y),
                    ),
                )
                if max_translation_m > float(self.config.object_scan_max_translation_m):
                    reason = "scan_translation_exceeded"
                    break

                if cumulative_yaw_degrees >= next_checkpoint:
                    self._safe_stop()
                    stop_completed_at = time.time()
                    stationary_at = self.wait_until_stationary(
                        after=stop_completed_at,
                        timeout=float(self.config.object_scan_stationary_timeout_s),
                    )
                    checkpoint = {
                        "yaw_degrees": round(cumulative_yaw_degrees, 2),
                        "stationary_confirmed_at": stationary_at,
                        "target_found": False,
                    }
                    checkpoints.append(checkpoint)
                    if stationary_at is None:
                        reason = "stationary_confirmation_failed"
                        break
                    multiscale_locator = getattr(
                        self,
                        "locate_target_multiscale",
                        None,
                    )
                    locator = (
                        multiscale_locator
                        if _is_compact_bottle_target(query)
                        and callable(multiscale_locator)
                        else self.locate_target
                    )
                    observation = locator(query, after=stationary_at)
                    metadata_reader = getattr(
                        self, "vision_request_metadata", None
                    )
                    vision_metadata = (
                        metadata_reader() if callable(metadata_reader) else {}
                    )
                    if vision_metadata:
                        checkpoint["vision_request"] = vision_metadata
                    if observation is not None:
                        checkpoint["target_found"] = True
                        target_found = True
                        completed = True
                        reason = "target_localized"
                        record()
                        return observation
                    if vision_metadata.get("error"):
                        reason = "visual_branch_terminated"
                        break
                    if next_checkpoint >= total_degrees:
                        completed = True
                        reason = "full_rotation_completed"
                        break
                    next_checkpoint = min(
                        total_degrees,
                        next_checkpoint + step_degrees,
                    )
                    refreshed = self._robot_transform()
                    if refreshed is not None:
                        previous_yaw = float(refreshed.rotation.to_euler().yaw)
                    continue

                commanded_yaw_rate = float(self.config.object_scan_yaw_rate)
                commanded_creep = float(self.config.object_scan_turn_creep_mps)
                if risk == "warning":
                    commanded_yaw_rate = min(commanded_yaw_rate, 0.25)
                    commanded_creep = 0.0
                else:
                    creep_phase = int(
                        (time.monotonic() - started_at)
                        / float(self.config.object_scan_creep_period_s)
                    )
                    if creep_phase % 2:
                        commanded_creep = -commanded_creep
                # Use the same direct, repeatedly refreshed stepping-turn
                # surface as Isaac manual control. The navigation adapter
                # otherwise rewrites a mixed command to 0.35 rad/s steering,
                # which the reference gait cannot use for a bounded scan.
                self.cmd_vel.publish(
                    Twist(
                        linear=Vector3(commanded_creep, 0.0, 0.0),
                        angular=Vector3(0.0, 0.0, commanded_yaw_rate),
                    )
                )
                time.sleep(0.05)
        finally:
            self._safe_stop()
        record()
        return None

    def verify_world_target(
        self,
        target_world: Vector3,
        *,
        after: float,
        query: str = "",
    ) -> VisualTargetObservation | None:
        """Verify the locked target with a new synchronized RGB-D pair."""

        deadline = time.monotonic() + 5.0
        frame_cursor = float(after)
        attempts: list[dict[str, Any]] = []
        allow_vertical_surface = _is_vertical_surface_target(query)
        with self._lock:
            self._last_verification = {
                "allow_vertical_surface": allow_vertical_surface,
                "attempts": [],
            }
        while time.monotonic() < deadline:
            pair = self._wait_for_pair(
                after=frame_cursor,
                timeout=max(0.0, deadline - time.monotonic()),
            )
            if pair is None:
                break
            color, depth = pair
            frame_timestamp = float(depth.ts)
            # Advance past a rejected pair.  TF propagation can lag a freshly
            # published image by one cycle; repeatedly selecting that same
            # frame would turn a transient ordering race into a false failure.
            frame_cursor = max(
                frame_cursor,
                float(color.ts),
                frame_timestamp,
            )
            world_from_camera = self.tf.get(
                "world",
                depth.frame_id or "camera_optical",
                frame_timestamp,
                time_tolerance=self.config.sync_tolerance_s,
                forward_tolerance=0.5,
            )
            robot_at_frame = self.tf.get(
                "world",
                "base_link",
                frame_timestamp,
                time_tolerance=max(0.3, self.config.sync_tolerance_s),
                forward_tolerance=0.5,
            )
            if world_from_camera is None or robot_at_frame is None:
                attempts.append(
                    {
                        "frame_timestamp": frame_timestamp,
                        "reason": (
                            "camera_tf_unavailable"
                            if world_from_camera is None
                            else "robot_tf_unavailable"
                        ),
                    }
                )
                continue
            diagnostics: dict[str, Any] = {}
            verified = verify_world_target_with_depth(
                target_world,
                depth,
                self.config.camera_info,
                world_from_camera,
                robot_at_frame,
                depth_tolerance_m=self.config.verification_depth_tolerance_m,
                min_points=self.config.min_depth_points,
                allow_vertical_surface=allow_vertical_surface,
                diagnostics=diagnostics,
            )
            attempts.append(
                {
                    "frame_timestamp": frame_timestamp,
                    **diagnostics,
                }
            )
            if verified is not None:
                with self._lock:
                    self._last_verification = {
                        "allow_vertical_surface": allow_vertical_surface,
                        "attempts": list(attempts[-20:]),
                    }
                return verified
        with self._lock:
            self._last_verification = {
                "allow_vertical_surface": allow_vertical_surface,
                "attempts": list(attempts[-20:]),
            }
        return None

    def verification_metadata(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_verification)

    def _run_visual_approach(
        self,
        query: str,
        *,
        standoff_distance: float,
        timeout: float,
        initial_observation: VisualTargetObservation | None = None,
    ) -> VisualApproachResult:
        self._cancel_event.clear()
        return VisualApproachExecutor(
            self,
            standoff_tolerance_m=self.config.standoff_tolerance_m,
            max_final_angle_degrees=self.config.max_final_angle_degrees,
            stationary_timeout_seconds=(
                self.config.visual_approach_stationary_timeout_s
            ),
            max_stop_latency_seconds=(
                self.config.visual_approach_max_stop_latency_s
            ),
        ).approach(
            query,
            standoff_distance=float(standoff_distance),
            timeout=float(timeout),
            initial_observation=initial_observation,
        )

    def _visual_incomplete(
        self,
        status: str,
        message: str,
        *,
        elapsed_s: float,
        viewpoint_search: dict[str, Any] | None = None,
        tool_ok: bool = True,
    ) -> VisualApproachResult:
        return VisualApproachResult(
            tool_ok=tool_ok,
            task_status=status,
            completed=False,
            planner_goal_reached=False,
            target_distance_m=None,
            stop_command_publish_latency_ms=0.0,
            physical_stop_latency_ms=None,
            elapsed_s=elapsed_s,
            message=message,
            viewpoint_search=viewpoint_search,
        )

    def _frontier_snapshot(self) -> OccupancyGrid | None:
        with self._lock:
            costmap = self._latest_exploration_costmap
            received_at = self._exploration_costmap_received_at
        if costmap is None or received_at is None:
            return None
        age = time.monotonic() - received_at
        if age < 0.0 or age > float(self.config.frontier_map_max_age_s):
            return None
        return costmap

    @staticmethod
    def _known_cell_count(costmap: OccupancyGrid | None) -> int:
        if costmap is None:
            return 0
        try:
            return int(np.count_nonzero(np.asarray(costmap.grid) >= 0))
        except (AttributeError, TypeError, ValueError):
            return 0

    def _select_frontier(
        self,
        rejected_goals: tuple[tuple[float, float], ...],
    ) -> KnownFreeFrontierGoal | None:
        costmap = self._frontier_snapshot()
        robot = self._robot_transform()
        if costmap is None or robot is None:
            return None
        costmap = self._risk_monitor.constrain_frontier_costmap(costmap)
        if costmap is None:
            return None
        return select_known_free_frontier(
            costmap,
            robot_x=float(robot.translation.x),
            robot_y=float(robot.translation.y),
            robot_yaw=float(robot.rotation.euler[2]),
            rejected_goals=rejected_goals,
            config=FrontierSelectionConfig(
                inward_offset_m=float(self.config.frontier_inward_offset_m),
                known_free_radius_m=float(
                    self.config.frontier_known_free_radius_m
                ),
                travel_cost_weight=float(
                    self.config.frontier_travel_cost_weight
                ),
                min_goal_path_distance_m=float(
                    self.config.frontier_min_goal_path_distance_m
                ),
                max_goal_path_distance_m=float(
                    self.config.frontier_max_goal_path_distance_m
                ),
            ),
        )

    def _wait_for_stable_frontier_map(
        self,
        *,
        deadline: float,
    ) -> OccupancyGrid | None:
        """Require a bounded stable cold-start map before choosing a route."""

        wait_deadline = min(
            deadline,
            time.monotonic() + float(self.config.frontier_stability_timeout_s),
        )
        stable_since: float | None = None
        prior_timestamp: float | None = None
        prior_known = 0
        latest: OccupancyGrid | None = None
        while time.monotonic() < wait_deadline:
            candidate = self._frontier_snapshot()
            if candidate is None:
                stable_since = None
                time.sleep(0.05)
                continue
            try:
                timestamp = float(candidate.ts)
            except (AttributeError, TypeError, ValueError, OverflowError):
                stable_since = None
                time.sleep(0.05)
                continue
            known = self._known_cell_count(candidate)
            latest = candidate
            if prior_timestamp is None:
                stable_since = time.monotonic()
            elif timestamp > prior_timestamp:
                significant_change = abs(known - prior_known) > max(
                    20,
                    int(max(1, prior_known) * 0.01),
                )
                if significant_change:
                    stable_since = time.monotonic()
            prior_timestamp, prior_known = timestamp, known
            if (
                stable_since is not None
                and time.monotonic() - stable_since
                >= float(self.config.frontier_initial_stability_s)
            ):
                return latest
            time.sleep(0.05)
        return None

    def _navigate_to_frontier(
        self,
        goal: KnownFreeFrontierGoal,
        *,
        deadline: float,
        cancelled: Callable[[], bool] | None = None,
        clearance_m: float | None = None,
    ) -> tuple[str, float | None]:
        """Navigate to one known-free frontier pose with terminal evidence."""

        goal_clearance_m = (
            float(self.config.frontier_known_free_radius_m)
            if clearance_m is None
            else float(clearance_m)
        )
        if self.risk_state() not in {"clear", "warning"}:
            self._safe_stop()
            return ("risk_blocked", None)
        self._frontier_goal_reached_event.clear()
        with self._lock:
            self._frontier_goal_reached_at = 0.0
        goal_started_at = time.monotonic()
        robot = self._robot_transform()
        if robot is None:
            self._safe_stop()
            return ("odometry_unavailable", None)
        if self._risk_monitor.goal_region_state(
            x=goal.x,
            y=goal.y,
            clearance_m=goal_clearance_m,
        ) != "clear":
            self._safe_stop()
            return ("navigation_blocked", None)
        try:
            accepted = bool(
                self._navigation.set_goal(
                    goal.pose_stamped(z=float(robot.translation.z))
                )
            )
        except Exception:  # noqa: BLE001 - planner transport failure stops
            accepted = False
        if not accepted:
            self._safe_stop()
            return ("goal_rejected", None)

        goal_deadline = min(
            deadline,
            time.monotonic() + float(self.config.frontier_goal_timeout_s),
        )
        checkpoint_seen_at: float | None = None
        while time.monotonic() < goal_deadline:
            if self._cancel_event.is_set() or (
                cancelled is not None and cancelled()
            ):
                self._safe_stop()
                return ("cancelled", None)
            risk = self.risk_state()
            if risk in {"unknown", "critical"}:
                self._safe_stop()
                return (f"risk_{risk}", None)
            if self._risk_monitor.goal_region_state(
                x=goal.x,
                y=goal.y,
                clearance_m=goal_clearance_m,
            ) != "clear":
                self._safe_stop()
                return ("navigation_blocked", None)
            with self._lock:
                reached_at = self._frontier_goal_reached_at
            event_reached = bool(
                self._frontier_goal_reached_event.is_set()
                and reached_at >= goal_started_at
            )
            try:
                planner_reached = bool(self._navigation.is_goal_reached())
            except Exception:  # noqa: BLE001 - transient query is not completion
                planner_reached = False
            current_robot = self._robot_transform()
            checkpoint_reached = bool(
                current_robot is not None
                and math.hypot(
                    float(current_robot.translation.x) - float(goal.x),
                    float(current_robot.translation.y) - float(goal.y),
                )
                <= float(self.config.frontier_checkpoint_tolerance_m)
            )
            reached = event_reached or planner_reached
            if reached:
                self._safe_stop()
                stop_completed_at = time.time()
                stationary_at = self.wait_until_stationary(
                    after=stop_completed_at,
                    timeout=float(self.config.frontier_stationary_timeout_s),
                )
                return (
                    (
                        "frontier_reached"
                        if stationary_at is not None
                        else "stationary_confirmation_failed"
                    ),
                    stationary_at,
                )
            if checkpoint_reached:
                checkpoint_seen_at = checkpoint_seen_at or time.monotonic()
                if (
                    time.monotonic() - checkpoint_seen_at
                    >= float(self.config.frontier_arrival_grace_s)
                ):
                    self._safe_stop()
                    stop_completed_at = time.time()
                    stationary_at = self.wait_until_stationary(
                        after=stop_completed_at,
                        timeout=float(self.config.frontier_stationary_timeout_s),
                    )
                    return (
                        (
                            "frontier_checkpoint_unverified"
                            if stationary_at is not None
                            else "stationary_confirmation_failed"
                        ),
                        stationary_at,
                    )
            else:
                checkpoint_seen_at = None
            time.sleep(0.05)
        self._safe_stop()
        stop_completed_at = time.time()
        stationary_at = self.wait_until_stationary(
            after=stop_completed_at,
            timeout=float(self.config.frontier_stationary_timeout_s),
        )
        return (
            (
                "frontier_timeout"
                if stationary_at is not None
                else "stationary_confirmation_failed"
            ),
            stationary_at,
        )

    def _wait_for_frontier_map_change(
        self,
        *,
        after_timestamp: float,
        deadline: float,
    ) -> OccupancyGrid | None:
        settle_deadline = min(
            deadline,
            time.monotonic() + float(self.config.frontier_map_settle_s),
        )
        latest = self._frontier_snapshot()
        while time.monotonic() < settle_deadline:
            candidate = self._frontier_snapshot()
            if candidate is not None:
                latest = candidate
                try:
                    if float(candidate.ts) > after_timestamp:
                        return candidate
                except (AttributeError, TypeError, ValueError, OverflowError):
                    pass
            time.sleep(0.05)
        return latest

    @staticmethod
    def _semantic_result(
        payload: dict[str, Any],
        *,
        route: str,
        query: str,
        similarity: float | None = None,
    ) -> str:
        result = dict(payload)
        result["navigate_with_text_route"] = route
        result["query"] = query
        result["semantic_memory_similarity"] = (
            round(similarity, 4) if similarity is not None else None
        )
        result["used_scene_truth"] = False
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _semantic_metadata(result: Any) -> tuple[dict[str, Any] | None, float | None]:
        if not isinstance(result, dict):
            return None, None
        metadata = result.get("metadata")
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else None
        if not isinstance(metadata, dict):
            return None, None
        try:
            distance = float(result["distance"])
            similarity = 1.0 - distance
        except (KeyError, TypeError, ValueError, OverflowError):
            return None, None
        values = tuple(metadata.get(key) for key in ("pos_x", "pos_y", "rot_z"))
        if not math.isfinite(similarity) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            return None, None
        return metadata, similarity

    def _known_free_pose(
        self,
        *,
        x: float,
        y: float,
        yaw: float,
        clearance_m: float = 0.25,
        max_substitution_distance_m: float = 0.5,
    ) -> KnownFreeFrontierGoal | None:
        costmap = self._frontier_snapshot()
        if costmap is None:
            return None
        # Exact tags and remembered viewpoints must obey the same live-map
        # boundary as frontier goals.  A cell is traversable only when both
        # the exploration map and the current planner costmap call it free.
        costmap = self._risk_monitor.constrain_frontier_costmap(costmap)
        if costmap is None:
            return None
        try:
            center = costmap.world_to_grid(Vector3(x, y, 0.0))
            center_x, center_y = int(center.x), int(center.y)
            radius = int(math.ceil(clearance_m / float(costmap.resolution)))
            cells = np.asarray(costmap.grid, dtype=np.int8)
            def disk_is_free(grid_x: int, grid_y: int) -> bool:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if (
                            math.hypot(dx, dy) * float(costmap.resolution)
                            > clearance_m
                        ):
                            continue
                        candidate_x, candidate_y = grid_x + dx, grid_y + dy
                        if (
                            candidate_x < 0
                            or candidate_y < 0
                            or candidate_y >= cells.shape[0]
                            or candidate_x >= cells.shape[1]
                            or int(cells[candidate_y, candidate_x]) != 0
                        ):
                            return False
                return True

            selected_x, selected_y = center_x, center_y
            substitution_distance = 0.0
            if not disk_is_free(center_x, center_y):
                search_radius = int(
                    math.ceil(
                        max_substitution_distance_m / float(costmap.resolution)
                    )
                )
                robot_provider = getattr(self, "_robot_transform", None)
                robot = robot_provider() if callable(robot_provider) else None
                candidates: list[
                    tuple[float, float, int, int, float, float]
                ] = []
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        distance = math.hypot(dx, dy) * float(costmap.resolution)
                        if distance > max_substitution_distance_m:
                            continue
                        if disk_is_free(center_x + dx, center_y + dy):
                            world = costmap.grid_to_world(
                                (center_x + dx, center_y + dy, 0.0)
                            )
                            travel_distance = (
                                math.hypot(
                                    float(world.x) - float(robot.translation.x),
                                    float(world.y) - float(robot.translation.y),
                                )
                                if robot is not None
                                else math.inf
                            )
                            candidates.append(
                                (
                                    distance,
                                    travel_distance,
                                    center_x + dx,
                                    center_y + dy,
                                    float(world.x),
                                    float(world.y),
                                )
                            )
                if not candidates:
                    return None
                (
                    substitution_distance,
                    _travel_distance,
                    selected_x,
                    selected_y,
                    x,
                    y,
                ) = min(candidates)
            map_timestamp = float(costmap.ts)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        return KnownFreeFrontierGoal(
            x=float(x),
            y=float(y),
            yaw_radians=float(yaw),
            frontier_cell_count=0,
            adjacent_unknown_cell_count=0,
            path_distance_m=float(substitution_distance),
            map_timestamp=map_timestamp,
        )

    def _exact_tag_goal(self, query: str) -> KnownFreeFrontierGoal | None:
        path = configured_tagged_locations_path()
        payload = read_world_json(path, max_bytes=256 * 1024)
        if (
            payload is None
            or payload.get("schema_version") != WORLD_STATE_SCHEMA_VERSION
            or not isinstance(payload.get("locations"), dict)
        ):
            return self._legacy_semantic_tag_goal(query)
        entry = payload["locations"].get(query.casefold())
        if not isinstance(entry, dict) or entry.get("frame_id") != "world":
            return self._legacy_semantic_tag_goal(query)
        position = entry.get("position")
        quaternion = entry.get("quaternion_xyzw")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or not isinstance(quaternion, list)
            or len(quaternion) != 4
        ):
            return self._legacy_semantic_tag_goal(query)
        try:
            x, y = float(position[0]), float(position[1])
            yaw = float(Quaternion(quaternion).euler[2])
        except (TypeError, ValueError, OverflowError):
            return self._legacy_semantic_tag_goal(query)
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            return self._legacy_semantic_tag_goal(query)
        return self._known_free_pose(x=x, y=y, yaw=yaw)

    def _legacy_semantic_tag_goal(
        self,
        query: str,
    ) -> KnownFreeFrontierGoal | None:
        """Read a pinned DimOS MuJoCo tag through the shared memory Port."""

        memory = self._spatial_memory
        query_tagged_location = getattr(memory, "query_tagged_location", None)
        if not callable(query_tagged_location):
            return None
        try:
            location = query_tagged_location(query)
            if location is None:
                return None
            position = tuple(location.position)
            rotation = tuple(location.rotation)
            if len(position) != 3 or len(rotation) != 3:
                return None
            x, y = float(position[0]), float(position[1])
            # Match NavigationSkillContainer's legacy storage/read contract.
            yaw = float(Quaternion.from_euler(Vector3(*rotation)).euler[2])
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            return None
        return self._known_free_pose(x=x, y=y, yaw=yaw)

    def _semantic_memory_goal(
        self,
        query: str,
    ) -> tuple[KnownFreeFrontierGoal | None, float | None]:
        if self._spatial_memory is None:
            return None, None
        try:
            results = self._spatial_memory.query_by_text(query, limit=5)
        except Exception:  # noqa: BLE001 - unavailable memory falls back to exploration
            return None, None
        for result in results or ():
            metadata, similarity = self._semantic_metadata(result)
            if metadata is None or similarity is None or similarity < 0.23:
                continue
            goal = self._known_free_pose(
                x=float(metadata["pos_x"]),
                y=float(metadata["pos_y"]),
                yaw=float(metadata["rot_z"]),
            )
            if goal is not None:
                return goal, similarity
        return None, None

    def _navigate_with_text_impl(
        self,
        query: str,
        standoff_distance: float = 0.9,
        timeout: float = 120.0,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        """Navigate by exact tag, fresh RGB-D, persistent CLIP memory, then frontier search.

        CLIP matches are remembered first-person viewpoints.  A remembered
        viewpoint is accepted only when it is still surrounded by observed
        free cells, and arrival is followed by a new RGB-D localization before
        any target-approach success can be reported.
        """

        started = time.monotonic()
        query_value = query.strip() if isinstance(query, str) else ""
        try:
            distance = float(standoff_distance)
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            distance, budget = math.nan, math.nan
        if (
            not query_value
            or len(query_value) > 200
            or not math.isfinite(distance)
            or not 0.5 <= distance <= 3.0
            or not math.isfinite(budget)
            or not 20.0 <= budget <= 180.0
        ):
            return self._semantic_result(
                self._visual_incomplete(
                    "invalid_input",
                    "query、standoff_distance 或 timeout 超出允许范围",
                    elapsed_s=time.monotonic() - started,
                    tool_ok=False,
                ).as_dict(),
                route="rejected",
                query=query_value,
            )

        self._cancel_event.clear()
        deadline = started + budget
        exact_goal = self._exact_tag_goal(query_value)
        if exact_goal is not None:
            status, stationary_at = self._navigate_to_frontier(
                exact_goal,
                deadline=deadline,
                cancelled=cancelled,
                clearance_m=float(
                    self.config.semantic_goal_known_free_radius_m
                ),
            )
            completed = status == "frontier_reached" and stationary_at is not None
            task_status = (
                "navigation_verified"
                if completed
                else status
            )
            return self._semantic_result(
                {
                    "tool_ok": True,
                    "task_status": task_status,
                    "completed": completed,
                    "planner_goal_reached": completed,
                    "stationary_confirmed": completed,
                    "position_error_m": exact_goal.path_distance_m,
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "message": (
                        "已到达精确命名位置" if completed else "精确命名位置导航未完成"
                    ),
                },
                route="exact_tag",
                query=query_value,
            )

        multiscale_locator = getattr(self, "locate_target_multiscale", None)
        locator = (
            multiscale_locator
            if _is_compact_bottle_target(query_value)
            and callable(multiscale_locator)
            else self.locate_target
        )
        observation = locator(query_value)
        if observation is not None:
            result = self._run_visual_approach(
                query_value,
                standoff_distance=distance,
                timeout=min(60.0, budget),
                initial_observation=observation,
            )
            return self._semantic_result(
                result.as_dict(),
                route="current_rgbd",
                query=query_value,
            )

        memory_goal, similarity = self._semantic_memory_goal(query_value)
        if memory_goal is not None:
            status, stationary_at = self._navigate_to_frontier(
                memory_goal,
                deadline=deadline,
                cancelled=cancelled,
                clearance_m=float(
                    self.config.semantic_goal_known_free_radius_m
                ),
            )
            if status in {
                "cancelled",
                "risk_unknown",
                "risk_critical",
                "stationary_confirmation_failed",
            }:
                return self._semantic_result(
                    self._visual_incomplete(
                        (
                            "cancelled"
                            if status == "cancelled"
                            else (
                                "risk_blocked"
                                if status.startswith("risk_")
                                else "verification_failed"
                            )
                        ),
                        "CLIP 记忆视点导航被安全或停车证据终止",
                        elapsed_s=time.monotonic() - started,
                    ).as_dict(),
                    route="persistent_clip_memory",
                    query=query_value,
                    similarity=similarity,
                )
            if status == "frontier_reached" and stationary_at is not None:
                observation = self.locate_target(
                    query_value,
                    after=stationary_at,
                )
                remaining = deadline - time.monotonic()
                if observation is not None and remaining >= 3.0:
                    result = self._run_visual_approach(
                        query_value,
                        standoff_distance=distance,
                        timeout=min(60.0, remaining),
                        initial_observation=observation,
                    )
                    return self._semantic_result(
                        result.as_dict(),
                        route="persistent_clip_memory",
                        query=query_value,
                        similarity=similarity,
                    )

        remaining = deadline - time.monotonic()
        if remaining < 20.0:
            self._safe_stop()
            return self._semantic_result(
                self._visual_incomplete(
                    "incomplete_budget_exhausted",
                    "标签、当前视野或 CLIP 记忆未完成导航，剩余预算不足以探索",
                    elapsed_s=time.monotonic() - started,
                ).as_dict(),
                route="frontier_fallback",
                query=query_value,
                similarity=similarity,
            )
        # This is one terminal Skill. Calling the public object_search wrapper
        # here would recursively request a second Tool Pipeline owner while
        # navigate_with_text already owns the robot.
        result = self._object_search_impl(
            query_value,
            distance,
            min(180.0, remaining),
        )
        return self._semantic_result(
            result.as_dict(),
            route="frontier_fallback",
            query=query_value,
            similarity=similarity,
        )

    @skill(uses=[CAP_MOVEMENT])
    def navigate_with_text(
        self,
        query: str,
        standoff_distance: float = 0.9,
        timeout: float = 120.0,
    ) -> str:
        """Navigate by text through the unified owner when it is available."""

        query_value = query.strip() if isinstance(query, str) else ""
        try:
            distance = float(standoff_distance)
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            return self._navigate_with_text_impl(query, standoff_distance, timeout)
        if (
            not query_value
            or len(query_value) > 200
            or not math.isfinite(distance)
            or not 0.5 <= distance <= 3.0
            or not math.isfinite(budget)
            or not 20.0 <= budget <= 180.0
        ):
            return self._navigate_with_text_impl(query_value, distance, budget)
        cancellation = TerminalCancellationChannel()
        cancellation_revision = cancellation.revision()

        def process_cancelled() -> bool:
            return cancellation.changed_since(cancellation_revision)

        arguments = {
            "query": query_value,
            "standoff_distance": distance,
            "timeout": budget,
        }

        def execute_local(cancelled: Callable[[], bool]) -> str:
            watcher_done = Event()

            def cancellation_requested() -> bool:
                return cancelled() or process_cancelled()

            def watch_cancel() -> None:
                while not watcher_done.wait(0.02):
                    if cancellation_requested():
                        self._cancel_event.set()
                        return

            watcher = Thread(
                target=watch_cancel,
                name="luxi-navigate-with-text-cancel",
                daemon=True,
            )
            watcher.start()
            try:
                return self._navigate_with_text_impl(
                    query_value,
                    distance,
                    budget,
                    cancelled=cancellation_requested,
                )
            finally:
                watcher_done.set()
                watcher.join(timeout=0.1)

        unified = IsaacNavigationControlChannel().request_navigate_with_text(
            arguments,
            execute_local,
            timeout_s=budget + 15.0,
        )
        if unified is not None:
            return json.dumps(unified, ensure_ascii=False, sort_keys=True)
        return self._navigate_with_text_impl(
            query_value,
            distance,
            budget,
            cancelled=process_cancelled,
        )

    @skill(uses=[CAP_MOVEMENT])
    def explore_frontiers(
        self,
        timeout: float = 90.0,
        max_frontiers: int = 8,
    ) -> FrontierExplorationResult | str:
        """Explore through the unified owner, retaining the native closed loop."""

        try:
            budget = float(timeout)
            frontier_limit = int(max_frontiers)
        except (TypeError, ValueError, OverflowError):
            budget, frontier_limit = math.nan, 0
        if (
            not math.isfinite(budget)
            or not 20.0 <= budget <= 180.0
            or not 1 <= frontier_limit <= 20
        ):
            return self._explore_frontiers_local(budget, frontier_limit)
        cancellation = TerminalCancellationChannel()
        cancellation_revision = cancellation.revision()

        def process_cancelled() -> bool:
            return cancellation.changed_since(cancellation_revision)

        arguments = {"timeout": budget, "max_frontiers": frontier_limit}
        def execute_local(cancelled: Callable[[], bool]) -> str:
            watcher_done = Event()

            def cancellation_requested() -> bool:
                return cancelled() or process_cancelled()

            def watch_cancel() -> None:
                while not watcher_done.wait(0.02):
                    if cancellation_requested():
                        self._cancel_event.set()
                        return

            watcher = Thread(
                target=watch_cancel,
                name="luxi-explore-frontiers-cancel",
                daemon=True,
            )
            watcher.start()
            try:
                return str(
                    self._explore_frontiers_local(
                        budget,
                        frontier_limit,
                        cancelled=cancellation_requested,
                    )
                )
            finally:
                watcher_done.set()
                watcher.join(timeout=0.1)

        unified = IsaacNavigationControlChannel().request_explore_frontiers(
            arguments,
            execute_local,
            timeout_s=budget + 15.0,
        )
        if unified is not None:
            return json.dumps(unified, ensure_ascii=False, sort_keys=True)
        return self._explore_frontiers_local(
            budget,
            frontier_limit,
            cancelled=process_cancelled,
        )

    def _explore_frontiers_local(
        self,
        timeout: float,
        max_frontiers: int,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> FrontierExplorationResult:
        """Run the existing terminal Skill after unified admission."""

        started = time.monotonic()
        try:
            budget = float(timeout)
            frontier_limit = int(max_frontiers)
        except (TypeError, ValueError, OverflowError):
            budget, frontier_limit = math.nan, 0
        if (
            not math.isfinite(budget)
            or not 20.0 <= budget <= 180.0
            or not 1 <= frontier_limit <= 20
        ):
            return FrontierExplorationResult(
                tool_ok=False,
                task_status="invalid_input",
                completed=False,
                frontiers_attempted=0,
                frontiers_reached=0,
                known_cells_before=0,
                known_cells_after=0,
                elapsed_s=time.monotonic() - started,
                termination_reason="invalid_input",
                goals=(),
                message="timeout 必须为 20-180s，max_frontiers 必须为 1-20",
            )

        self._cancel_event.clear()
        deadline = started + budget
        first_map = self._wait_for_stable_frontier_map(deadline=deadline)
        known_before = self._known_cell_count(first_map)
        rejected: list[tuple[float, float]] = []
        goal_audit: list[dict[str, Any]] = []
        map_acquisition: dict[str, Any] = {}
        acquisition_attempted = False
        reached_count = 0
        reason = "no_fresh_exploration_map"
        while time.monotonic() < deadline and len(goal_audit) < frontier_limit:
            if cancelled is not None and cancelled():
                self._safe_stop()
                reason = "cancelled"
                break
            stable_map = self._wait_for_stable_frontier_map(deadline=deadline)
            if stable_map is None:
                reason = "frontier_map_unstable"
                break
            goal = self._select_frontier(tuple(rejected))
            if goal is None:
                if reached_count == 0 and not acquisition_attempted:
                    acquisition_attempted = True
                    known_before_scan = self._known_cell_count(stable_map)
                    stationary_at = self.search_viewpoint()
                    map_acquisition = self.viewpoint_search_metadata()
                    map_acquisition.update(
                        {
                            "mode": "measured_in_place_scan",
                            "known_cells_before": known_before_scan,
                            "stationary_confirmed": stationary_at is not None,
                        }
                    )
                    if stationary_at is None:
                        reason = "map_acquisition_failed"
                        break
                    refreshed = self._wait_for_stable_frontier_map(
                        deadline=deadline
                    )
                    map_acquisition["known_cells_after"] = self._known_cell_count(
                        refreshed
                    )
                    continue
                reason = (
                    "frontiers_exhausted"
                    if self._frontier_snapshot() is not None
                    else "no_fresh_exploration_map"
                )
                break
            rejected.append((goal.x, goal.y))
            status, stationary_at = self._navigate_to_frontier(
                goal,
                deadline=deadline,
                cancelled=cancelled,
            )
            audit = goal.as_dict()
            audit["navigation_status"] = status
            goal_audit.append(audit)
            if status != "frontier_reached" or stationary_at is None:
                reason = status
                if status in {
                    "cancelled",
                    "risk_unknown",
                    "risk_critical",
                    "stationary_confirmation_failed",
                }:
                    break
                continue
            reached_count += 1
            reason = "frontier_limit_reached"
            self._wait_for_frontier_map_change(
                after_timestamp=goal.map_timestamp,
                deadline=deadline,
            )
        else:
            if time.monotonic() >= deadline:
                reason = "timeout"
            elif len(goal_audit) >= frontier_limit and reached_count > 0:
                # The bounded exploration budget is satisfied once at least
                # one planner-verified viewpoint was reached. A later dynamic
                # costmap rejection remains in the goal audit, but must not
                # erase already verified physical progress.
                reason = "frontier_limit_reached"

        self._safe_stop()
        final_map = self._frontier_snapshot()
        known_after = self._known_cell_count(final_map)
        completed = bool(
            reached_count > 0
            and reason in {"frontiers_exhausted", "frontier_limit_reached", "timeout"}
        )
        return FrontierExplorationResult(
            tool_ok=True,
            task_status=(
                "exploration_complete"
                if reason == "frontiers_exhausted" and completed
                else ("exploration_budget_complete" if completed else reason)
            ),
            completed=completed,
            frontiers_attempted=len(goal_audit),
            frontiers_reached=reached_count,
            known_cells_before=known_before,
            known_cells_after=known_after,
            elapsed_s=time.monotonic() - started,
            termination_reason=reason,
            goals=tuple(goal_audit),
            message=(
                f"已到达 {reached_count}/{len(goal_audit)} 个已知自由侧 frontier，"
                f"新增 {max(0, known_after - known_before)} 个已知栅格"
            ),
            map_acquisition=map_acquisition,
        )

    @skill(uses=[CAP_MOVEMENT])
    def object_search(
        self,
        query: str,
        standoff_distance: float = 0.9,
        timeout: float = 120.0,
    ) -> VisualApproachResult | str:
        """Find an object by measured scan and safe frontier exploration.

        The task uses only current first-person RGB-D, world odometry, current
        obstacle evidence, an unknown-preserving exploration map, and the
        planner map supplied by the selected World Adapter. Frontier goals
        stay on reachable observed-free cells; scene truth and operator
        cameras are never task inputs.
        """

        query_value = query.strip() if isinstance(query, str) else ""
        try:
            distance = float(standoff_distance)
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            distance, budget = math.nan, math.nan
        if (
            not query_value
            or len(query_value) > 200
            or not math.isfinite(distance)
            or not 0.5 <= distance <= 3.0
            or not math.isfinite(budget)
            or not 20.0 <= budget <= 180.0
        ):
            self._cancel_event.clear()
            return self._object_search_impl(query, standoff_distance, timeout)
        arguments = {
            "query": query_value,
            "standoff_distance": distance,
            "timeout": budget,
        }
        unified = IsaacNavigationControlChannel().request_object_search(
            arguments,
            lambda cancelled: str(
                self._object_search_local(
                    query_value,
                    distance,
                    budget,
                    cancelled=cancelled,
                )
            ),
            timeout_s=budget + 15.0,
        )
        if unified is not None:
            return json.dumps(unified, ensure_ascii=False, sort_keys=True)
        self._cancel_event.clear()
        return self._object_search_impl(query_value, distance, budget)

    def _object_search_local(
        self,
        query: str,
        standoff_distance: float,
        timeout: float,
        *,
        cancelled: Callable[[], bool],
    ) -> VisualApproachResult:
        """Run the existing terminal Skill after unified admission."""

        self._cancel_event.clear()
        watcher_done = Event()

        def watch_cancel() -> None:
            while not watcher_done.wait(0.02):
                if cancelled():
                    self._cancel_event.set()
                    return

        watcher = Thread(
            target=watch_cancel,
            name="luxi-object-search-cancel",
            daemon=True,
        )
        watcher.start()
        try:
            return self._object_search_impl(query, standoff_distance, timeout)
        finally:
            watcher_done.set()
            watcher.join(timeout=0.1)

    def _object_search_impl(
        self,
        query: str,
        standoff_distance: float,
        timeout: float,
    ) -> VisualApproachResult:

        started = time.monotonic()
        query_value = query.strip() if isinstance(query, str) else ""
        try:
            distance = float(standoff_distance)
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            distance, budget = math.nan, math.nan
        if (
            not query_value
            or len(query_value) > 200
            or not math.isfinite(distance)
            or not 0.5 <= distance <= 3.0
            or not math.isfinite(budget)
            or not 20.0 <= budget <= 180.0
        ):
            return self._visual_incomplete(
                "invalid_input",
                "query、standoff_distance 或 timeout 超出允许范围",
                elapsed_s=time.monotonic() - started,
                tool_ok=False,
            )
        if self.risk_state() not in {"clear", "warning"}:
            self._safe_stop()
            return self._visual_incomplete(
                "risk_blocked",
                "当前 RTX lidar 风险证据不允许启动寻物",
                elapsed_s=time.monotonic() - started,
            )
        observation = self.locate_target(query_value)
        initial_vision_request = self.vision_request_metadata()
        scan_audit: dict[str, Any] = {
            "evidence_source": "robot_rgbd_odometry_online_map",
            "headings_checked": 1,
            "frontiers_attempted": 0,
            "frontiers_reached": 0,
            "frontier_goals": [],
            "frontier_goal_space": "known_free_only",
            "used_scene_truth": False,
            "initial_vision_request": initial_vision_request,
        }
        if observation is None and initial_vision_request.get("error"):
            self._safe_stop()
            return self._visual_incomplete(
                "visual_branch_terminated",
                "视觉请求失败，已停车且不再扫描、探索或重试",
                elapsed_s=time.monotonic() - started,
                viewpoint_search=scan_audit,
            )
        if observation is None:
            observation = self.search_full_rotation(
                query_value,
                deadline=started + budget - 8.0,
            )
            scan_metadata = self.viewpoint_search_metadata()
            scan_audit.update(scan_metadata)
            scan_audit["headings_checked"] = 1 + int(
                scan_metadata.get("headings_checked", 0)
            )
            if (
                observation is None
                and scan_metadata.get("reason") == "visual_branch_terminated"
            ):
                self._safe_stop()
                return self._visual_incomplete(
                    "visual_branch_terminated",
                    "环视中的视觉请求失败，已停车且不再探索或重试",
                    elapsed_s=time.monotonic() - started,
                    viewpoint_search=scan_audit,
                )
        if observation is None:
            rejected: list[tuple[float, float]] = []
            deadline = started + budget
            # Preserve a bounded perception/approach budget.  A frontier leg
            # may otherwise consume the whole task deadline after making
            # useful physical progress, leaving no chance to inspect the
            # stationary endpoint or verify an approach.
            approach_reserve_s = 20.0
            while time.monotonic() + approach_reserve_s < deadline:
                goal = self._select_frontier(tuple(rejected))
                if goal is None:
                    scan_audit["frontier_termination_reason"] = (
                        "frontiers_exhausted"
                        if self._frontier_snapshot() is not None
                        else "no_fresh_exploration_map"
                    )
                    break
                rejected.append((goal.x, goal.y))
                scan_audit["frontiers_attempted"] += 1
                status, stationary_at = self._navigate_to_frontier(
                    goal,
                    deadline=deadline - approach_reserve_s,
                )
                goal_audit = goal.as_dict()
                goal_audit["navigation_status"] = status
                scan_audit["frontier_goals"].append(goal_audit)
                if stationary_at is not None:
                    if status == "frontier_reached":
                        scan_audit["frontiers_reached"] += 1
                    self._wait_for_frontier_map_change(
                        after_timestamp=goal.map_timestamp,
                        deadline=deadline,
                    )
                    observation = self.locate_target_multiscale(
                        query_value,
                        after=stationary_at,
                    )
                    goal_audit["vision_request"] = self.vision_request_metadata()
                    if observation is not None:
                        scan_audit["frontier_termination_reason"] = (
                            "target_localized"
                        )
                        break
                    if goal_audit["vision_request"].get("error"):
                        scan_audit["frontier_termination_reason"] = (
                            "visual_branch_terminated"
                        )
                        break
                if status != "frontier_reached" or stationary_at is None:
                    scan_audit["frontier_termination_reason"] = status
                    if status in {
                        "cancelled",
                        "risk_unknown",
                        "risk_critical",
                        "stationary_confirmation_failed",
                    }:
                        break
                    continue
            if observation is None:
                self._safe_stop()
                return self._visual_incomplete(
                    "target_not_found",
                    (
                        "第一人称环视及已知自由侧 frontier 探索均未从"
                        "新鲜 RGB-D 中定位目标"
                    ),
                    elapsed_s=time.monotonic() - started,
                    viewpoint_search=scan_audit,
                )
        remaining = budget - (time.monotonic() - started)
        if remaining < 3.0:
            self._safe_stop()
            return self._visual_incomplete(
                "incomplete_budget_exhausted",
                "已找到目标，但剩余预算不足以启动闭环接近",
                elapsed_s=time.monotonic() - started,
                viewpoint_search=scan_audit,
            )
        result = self._run_visual_approach(
            query_value,
            standoff_distance=distance,
            # Isaac G1 may spend most of a short approach rotating around a
            # live obstacle before its measured forward gait converges.  Use
            # the remaining task budget, still bounded by the public 180 s
            # deadline, instead of cutting this safe planner phase at 60 s.
            timeout=min(90.0, remaining),
            initial_observation=observation,
        )
        return replace(result, viewpoint_search=scan_audit)

    def _latest_follow_pair(self, after: float) -> tuple[Image, Image] | None:
        with self._lock:
            colors = tuple(self._color_frames)
            depths = tuple(self._depth_frames)
        for depth in sorted(depths, key=lambda item: float(item.ts), reverse=True):
            if float(depth.ts) <= after:
                continue
            matches = [
                color
                for color in colors
                if float(color.ts) > after
                and abs(float(color.ts) - float(depth.ts))
                <= self.config.sync_tolerance_s
            ]
            if matches:
                return (
                    min(
                        matches,
                        key=lambda color: abs(float(color.ts) - float(depth.ts)),
                    ),
                    depth,
                )
        return None

    def _wait_for_follow_pair(
        self, *, after: float, timeout: float
    ) -> tuple[Image, Image] | None:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            pair = self._latest_follow_pair(after)
            if pair is not None:
                return pair
            time.sleep(0.01)
        return None

    def _follow_measurement(
        self,
        bbox: tuple[float, float, float, float],
        pair: tuple[Image, Image],
    ) -> TrackingMeasurement | None:
        color, depth = pair
        estimate = localize_person_in_camera(
            bbox,
            depth.data,
            self.config.camera_info,
            min_points=self.config.min_depth_points,
        )
        if estimate is None:
            return None
        distance = math.hypot(float(estimate.point.x), float(estimate.point.z))
        if not math.isfinite(distance) or distance <= 0.0:
            return None
        return TrackingMeasurement(
            bbox=bbox,
            distance_m=distance,
            bearing_radians=math.atan2(
                -float(estimate.point.x),
                float(estimate.point.z),
            ),
            frame_timestamp=float(depth.ts),
            valid_depth_points=int(estimate.valid_points),
        )

    def _legacy_follow_person(
        self,
        query: str = "person",
        follow_distance: float = 1.5,
        duration: float = 30.0,
        timeout: float = 150.0,
    ) -> PersonFollowResult:
        """Follow one VLM-locked person using CSRT, RGB-D and RTX lidar."""

        started = time.monotonic()
        query_value = query.strip() if isinstance(query, str) else ""
        try:
            target_distance = float(follow_distance)
            requested_duration = float(duration)
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            target_distance = requested_duration = budget = math.nan

        def finish(
            status: str,
            reason: str,
            message: str,
            *,
            tool_ok: bool = True,
            verified: float = 0.0,
            frames: int = 0,
            coverage: float = 0.0,
            gap: float = 0.0,
            measurement: TrackingMeasurement | None = None,
            stop_completed_at: float | None = None,
            stationary_at: float | None = None,
            publish_latency_ms: float = 0.0,
            verification_timestamp: float | None = None,
        ) -> PersonFollowResult:
            physical_latency = (
                max(0.0, (stationary_at - stop_completed_at) * 1_000.0)
                if stationary_at is not None and stop_completed_at is not None
                else None
            )
            return PersonFollowResult(
                tool_ok=tool_ok,
                task_status=status,
                completed=status == "follow_verified",
                requested_follow_duration_s=(
                    requested_duration if math.isfinite(requested_duration) else 0.0
                ),
                verified_tracking_duration_s=verified,
                elapsed_s=max(0.0, time.monotonic() - started),
                requested_follow_distance_m=(
                    target_distance if math.isfinite(target_distance) else 0.0
                ),
                target_distance_m=(
                    measurement.distance_m if measurement is not None else None
                ),
                target_bearing_degrees=(
                    math.degrees(measurement.bearing_radians)
                    if measurement is not None
                    else None
                ),
                tracked_frames=frames,
                tracking_coverage=coverage,
                max_tracking_gap_s=gap,
                stop_command_publish_latency_ms=publish_latency_ms,
                physical_stop_latency_ms=physical_latency,
                stop_command_completed_at=stop_completed_at,
                stationary_confirmed_at=stationary_at,
                verification_frame_timestamp=verification_timestamp,
                termination_reason=reason,
                message=message,
            )

        if (
            not query_value
            or len(query_value) > 200
            or not math.isfinite(target_distance)
            or not 1.2 <= target_distance <= 2.0
            or not math.isfinite(requested_duration)
            or not 5.0 <= requested_duration <= 60.0
            or not math.isfinite(budget)
            or not 65.0 <= budget <= 180.0
            or budget < requested_duration + 55.0
        ):
            return finish(
                "invalid_input",
                "invalid_input",
                "query、follow_distance、duration 或 timeout 超出允许范围",
                tool_ok=False,
            )
        with self._lock:
            if self._follow_active:
                return finish(
                    "invalid_input",
                    "follow_already_active",
                    "已有跟随任务正在运行",
                    tool_ok=False,
                )
            self._follow_active = True
        self._follow_cancel_event.clear()
        self._cancel_event.clear()
        deadline = started + budget
        try:
            if self.risk_state() not in {"clear", "warning"}:
                self._safe_stop()
                return finish(
                    "risk_blocked",
                    "unsafe_start",
                    "当前 RTX lidar 风险证据不允许启动跟随",
                )
            located: VisualTargetObservation | None = None
            detection_after = 0.0
            for _ in range(int(self.config.follow_initial_detection_attempts)):
                located = self.locate_target(
                    query_value,
                    after=detection_after,
                )
                if located is not None:
                    break
                with self._lock:
                    missed_pair = self._last_localization_pair
                if missed_pair is not None:
                    detection_after = max(
                        float(missed_pair[0].ts),
                        float(missed_pair[1].ts),
                    )
            if located is None:
                stationary_at = self.search_viewpoint()
                if stationary_at is not None:
                    located = self.locate_target(query_value, after=stationary_at)
            with self._lock:
                pair = self._last_localization_pair
                raw_bbox = self._last_vision_request.get("bbox")
            if located is None or pair is None or not isinstance(raw_bbox, list):
                self._safe_stop()
                return finish(
                    "target_not_found",
                    "initial_acquisition_failed",
                    "没有取得可用于跟随的人物 RGB-D 定位",
                )
            try:
                bbox = tuple(float(value) for value in raw_bbox)
            except (TypeError, ValueError, OverflowError):
                bbox = ()
            if len(bbox) != 4:
                self._safe_stop()
                return finish(
                    "tracking_init_failed",
                    "invalid_bbox",
                    "VLM 人物框无效",
                )
            tracker = CsrtTargetTracker()
            if not tracker.initialize(pair[0], bbox):
                self._safe_stop()
                return finish(
                    "tracking_init_failed",
                    "csrt_init_failed",
                    "CSRT 无法在 VLM 锁定帧上初始化",
                )
            last_frame_ts = float(pair[1].ts)
            tracking_started = time.monotonic()
            last_valid_at = tracking_started
            follow_started_at: float | None = None
            follow_last_valid_at: float | None = None
            distance_acquisition = FollowDistanceAcquisition(
                target_distance_m=target_distance,
                tolerance_m=float(self.config.follow_distance_tolerance_m),
                required_samples=int(self.config.follow_acquisition_samples),
            )
            supported_duration = 0.0
            tracked_frames = 0
            max_gap = 0.0
            last_measurement: TrackingMeasurement | None = None
            while time.monotonic() < deadline:
                now = time.monotonic()
                if self._follow_cancel_event.is_set() or self._cancel_event.is_set():
                    self._safe_stop()
                    return finish(
                        "cancelled",
                        "cancelled",
                        "跟随任务已取消并停车",
                        verified=supported_duration,
                        frames=tracked_frames,
                        coverage=(
                            supported_duration / max(0.001, now - tracking_started)
                        ),
                        measurement=last_measurement,
                    )
                pair = self._wait_for_follow_pair(
                    after=last_frame_ts,
                    timeout=1.0 / float(self.config.follow_control_hz),
                )
                if pair is None:
                    gap = now - last_valid_at
                    max_gap = max(max_gap, gap)
                    if gap >= float(self.config.follow_frame_hold_s):
                        self.nav_cmd_vel.publish(Twist())
                    if gap >= float(self.config.follow_tracking_lost_s):
                        self._safe_stop()
                        return finish(
                            "tracking_lost",
                            "fresh_frame_timeout",
                            "人物跟踪帧超过时限，已停车且不重新选人",
                            verified=supported_duration,
                            frames=tracked_frames,
                            coverage=(
                                supported_duration / max(0.001, now - tracking_started)
                            ),
                            gap=max_gap,
                            measurement=last_measurement,
                        )
                    continue
                last_frame_ts = max(
                    last_frame_ts,
                    float(pair[0].ts),
                    float(pair[1].ts),
                )
                bbox = tracker.update(pair[0])
                measurement = (
                    self._follow_measurement(bbox, pair) if bbox is not None else None
                )
                if measurement is None:
                    self.nav_cmd_vel.publish(Twist())
                    continue
                now = time.monotonic()
                frame_gap = max(0.0, now - last_valid_at)
                max_gap = max(max_gap, frame_gap)
                last_valid_at = now
                last_measurement = measurement
                tracked_frames += 1
                risk = self.risk_state()
                if risk not in {"clear", "warning"}:
                    self._safe_stop()
                    return finish(
                        "risk_blocked",
                        f"risk_{risk}",
                        "RTX lidar 风险阻断跟随，旧任务不会恢复",
                        verified=supported_duration,
                        frames=tracked_frames,
                        coverage=(
                            supported_duration / max(0.001, now - tracking_started)
                        ),
                        gap=max_gap,
                        measurement=measurement,
                    )
                self.nav_cmd_vel.publish(
                    compute_follow_twist(
                        measurement,
                        follow_distance_m=target_distance,
                        risk=risk,
                    )
                )
                acquired_now = distance_acquisition.observe(
                    measurement,
                    now=now,
                )
                if acquired_now and follow_started_at is None:
                    follow_started_at = distance_acquisition.acquired_at
                    follow_last_valid_at = now
                elif follow_started_at is not None and follow_last_valid_at is not None:
                    supported_duration += min(
                        max(0.0, now - follow_last_valid_at),
                        float(self.config.follow_frame_hold_s),
                    )
                    follow_last_valid_at = now
                if supported_duration >= requested_duration:
                    break
            else:
                self._safe_stop()
                return finish(
                    "follow_timeout",
                    (
                        "distance_acquisition_timeout"
                        if follow_started_at is None
                        else "whole_skill_timeout"
                    ),
                    (
                        "未在任务预算内连续进入请求跟随距离"
                        if follow_started_at is None
                        else "整段跟随预算已耗尽"
                    ),
                    verified=supported_duration,
                    frames=tracked_frames,
                    measurement=last_measurement,
                )
            stop_started = time.monotonic()
            self._safe_stop()
            publish_latency = (time.monotonic() - stop_started) * 1_000.0
            stop_completed_at = time.time()
            stationary_at = self.wait_until_stationary(
                after=stop_completed_at,
                timeout=float(self.config.follow_stationary_timeout_s),
            )
            coverage = supported_duration / max(
                0.001,
                time.monotonic() - (follow_started_at or tracking_started),
            )
            verification_pair = (
                self._wait_for_follow_pair(after=stationary_at, timeout=1.0)
                if stationary_at is not None
                else None
            )
            verification_bbox = (
                tracker.update(verification_pair[0])
                if verification_pair is not None
                else None
            )
            verification = (
                self._follow_measurement(verification_bbox, verification_pair)
                if verification_bbox is not None and verification_pair is not None
                else None
            )
            physical_latency = (
                (stationary_at - stop_completed_at) * 1_000.0
                if stationary_at is not None
                else math.inf
            )
            verified = bool(
                verification is not None
                and coverage >= 0.9
                and abs(verification.distance_m - target_distance)
                <= float(self.config.follow_distance_tolerance_m)
                and abs(math.degrees(verification.bearing_radians)) <= 30.0
                and physical_latency
                <= float(self.config.follow_max_stop_latency_s) * 1_000.0
                and verification.frame_timestamp > stationary_at
            )
            return finish(
                "follow_verified" if verified else "verification_failed",
                (
                    "duration_and_post_stop_rgbd_verified"
                    if verified
                    else "post_stop_evidence_failed"
                ),
                (
                    "跟随时长、最终距离/朝向与停车后 RGB-D 均已验证"
                    if verified
                    else "跟随结束，但最终距离、覆盖率或停车证据未通过"
                ),
                verified=supported_duration,
                frames=tracked_frames,
                coverage=coverage,
                gap=max_gap,
                measurement=verification or last_measurement,
                stop_completed_at=stop_completed_at,
                stationary_at=stationary_at,
                publish_latency_ms=publish_latency,
                verification_timestamp=(
                    verification.frame_timestamp if verification is not None else None
                ),
            )
        except Exception as error:  # noqa: BLE001 - motion failures stop
            self._safe_stop()
            return finish(
                "runtime_error",
                "runtime_error",
                f"跟随运行失败：{type(error).__name__}: {error}"[:500],
                tool_ok=False,
            )
        finally:
            with self._lock:
                self._follow_active = False

    @skill
    def stop_following(self) -> None:
        """Idempotently stop the task-scoped person-follow loop."""

        self._follow_cancel_event.set()
        self._safe_stop()

    def follow_risk_state(self, *, translating: bool) -> str:
        del translating
        return self.risk_state()

    def follow_stop_evidence(self) -> FollowStopEvidence:
        stop_started = time.monotonic()
        self._safe_stop()
        publish_latency_ms = (time.monotonic() - stop_started) * 1_000.0
        stop_completed_at = time.time()
        stationary_at = self.wait_until_stationary(
            after=stop_completed_at,
            timeout=float(self.config.follow_stationary_timeout_s),
        )
        return FollowStopEvidence(
            stop_command_publish_latency_ms=publish_latency_ms,
            physical_stop_latency_ms=(
                max(0.0, (stationary_at - stop_completed_at) * 1_000.0)
                if stationary_at is not None
                else None
            ),
            stop_command_completed_at=stop_completed_at,
            stationary_confirmed_at=stationary_at,
        )

    def acquire_follow_target(
        self,
        query: str,
        *,
        after: float,
        deadline: float,
    ) -> FollowAcquisition | None:
        if time.monotonic() >= deadline:
            with self._lock:
                self._last_vision_request = {"localization_stage": "time_budget"}
            return None
        detection_query = (
            "full body of the visible human person, including a "
            "photorealistic scanned or simulated character, matching this "
            f"description: {query.strip()}"
        )
        located = self.locate_target(detection_query, after=after)
        if located is None:
            return None
        with self._lock:
            pair = self._last_localization_pair
            raw_bbox = self._last_vision_request.get("bbox")
        if pair is None or not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            with self._lock:
                self._last_vision_request["localization_stage"] = "invalid_bbox"
            return None
        try:
            bbox = tuple(float(value) for value in raw_bbox)
        except (TypeError, ValueError, OverflowError):
            return None
        point = located.estimate.point
        distance = math.hypot(float(point.x), float(point.z))
        if not math.isfinite(distance) or distance <= 0.0:
            return None
        return FollowAcquisition(
            image=pair[0],
            bbox=bbox,  # type: ignore[arg-type]
            frame_timestamp=float(pair[1].ts),
            initial_distance_m=distance,
        )

    def follow_viewpoint_search(self, *, deadline: float) -> float | None:
        if time.monotonic() >= deadline or self._follow_cancel_event.is_set():
            return None
        return self.search_viewpoint()

    def follow_cancelled(self) -> bool:
        return self._follow_cancel_event.is_set() or self._cancel_event.is_set()

    def wait_for_follow_pair(
        self,
        *,
        after: float,
        timeout: float,
    ) -> tuple[Image, Image] | None:
        return self._wait_for_follow_pair(after=after, timeout=timeout)

    def follow_measurement(
        self,
        bbox: tuple[float, float, float, float],
        pair: tuple[Image, Image],
    ) -> TrackingMeasurement | None:
        return self._follow_measurement(bbox, pair)

    def publish_follow_command(self, command: Twist) -> None:
        self.nav_cmd_vel.publish(command)

    def _follow_person_impl(
        self,
        query: str = "person",
        follow_distance: float = 1.5,
        duration: float = 30.0,
        timeout: float = 150.0,
    ) -> PersonFollowResult:
        """Follow one visible person using the shared bounded state machine."""

        query_value = query.strip() if isinstance(query, str) else ""
        try:
            target_distance = float(follow_distance)
            requested_duration = float(duration)
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            target_distance = requested_duration = budget = math.nan

        def invalid(message: str) -> PersonFollowResult:
            return PersonFollowResult(
                tool_ok=False,
                task_status="invalid_input",
                completed=False,
                requested_follow_duration_s=(
                    requested_duration if math.isfinite(requested_duration) else 0.0
                ),
                verified_tracking_duration_s=0.0,
                elapsed_s=0.0,
                requested_follow_distance_m=(
                    target_distance if math.isfinite(target_distance) else 0.0
                ),
                target_distance_m=None,
                target_bearing_degrees=None,
                tracked_frames=0,
                tracking_coverage=0.0,
                max_tracking_gap_s=0.0,
                stop_command_publish_latency_ms=0.0,
                physical_stop_latency_ms=None,
                stop_command_completed_at=None,
                stationary_confirmed_at=None,
                verification_frame_timestamp=None,
                termination_reason="invalid_input",
                message=message,
            )

        if (
            not query_value
            or len(query_value) > 200
            or not math.isfinite(target_distance)
            or not 1.2 <= target_distance <= 2.0
            or not math.isfinite(requested_duration)
            or not 5.0 <= requested_duration <= 60.0
            or not math.isfinite(budget)
            or not 65.0 <= budget <= 180.0
            or budget < requested_duration + 55.0
        ):
            return invalid("query、follow_distance、duration 或 timeout 超出允许范围")
        with self._lock:
            if self._follow_active:
                return invalid("已有跟随任务正在运行")
            self._follow_active = True
        self._follow_cancel_event.clear()
        self._cancel_event.clear()
        started = time.monotonic()
        try:
            return PersonFollowExecutor(
                self,
                runtime=FollowRuntimeConfig(
                    control_hz=float(self.config.follow_control_hz),
                    frame_hold_s=float(self.config.follow_frame_hold_s),
                    tracking_lost_s=float(self.config.follow_tracking_lost_s),
                    sensor_gap_timeout_s=float(self.config.follow_sensor_gap_timeout_s),
                    unknown_risk_timeout_s=float(
                        self.config.follow_unknown_risk_timeout_s
                    ),
                    verification_timeout_s=float(
                        self.config.follow_verification_timeout_s
                    ),
                    distance_tolerance_m=float(self.config.follow_distance_tolerance_m),
                    max_final_angle_degrees=float(
                        self.config.follow_max_final_angle_degrees
                    ),
                    min_tracking_coverage=float(
                        self.config.follow_min_tracking_coverage
                    ),
                    max_stop_latency_s=float(self.config.follow_max_stop_latency_s),
                ),
            ).run(
                query_value,
                follow_distance=target_distance,
                duration=requested_duration,
                timeout=budget,
            )
        except Exception as error:  # noqa: BLE001 - motion failures stop
            evidence = self.follow_stop_evidence()
            return PersonFollowResult(
                tool_ok=False,
                task_status="runtime_error",
                completed=False,
                requested_follow_duration_s=requested_duration,
                verified_tracking_duration_s=0.0,
                elapsed_s=max(0.0, time.monotonic() - started),
                requested_follow_distance_m=target_distance,
                target_distance_m=None,
                target_bearing_degrees=None,
                tracked_frames=0,
                tracking_coverage=0.0,
                max_tracking_gap_s=0.0,
                stop_command_publish_latency_ms=(
                    evidence.stop_command_publish_latency_ms
                ),
                physical_stop_latency_ms=evidence.physical_stop_latency_ms,
                stop_command_completed_at=evidence.stop_command_completed_at,
                stationary_confirmed_at=evidence.stationary_confirmed_at,
                verification_frame_timestamp=None,
                termination_reason="runtime_error",
                message=f"{type(error).__name__}: {error}"[:500],
            )
        finally:
            with self._lock:
                self._follow_active = False

    @skill(uses=[CAP_MOVEMENT])
    def follow_person(
        self,
        query: str = "person",
        follow_distance: float = 1.5,
        duration: float = 30.0,
        timeout: float = 150.0,
    ) -> PersonFollowResult | str:
        """Follow one person through the unified owner when it is available."""

        query_value = query.strip() if isinstance(query, str) else ""
        try:
            target_distance = float(follow_distance)
            requested_duration = float(duration)
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            return self._follow_person_impl(
                query,
                follow_distance,
                duration,
                timeout,
            )
        valid = bool(
            query_value
            and len(query_value) <= 200
            and math.isfinite(target_distance)
            and 1.2 <= target_distance <= 2.0
            and math.isfinite(requested_duration)
            and 5.0 <= requested_duration <= 60.0
            and math.isfinite(budget)
            and 65.0 <= budget <= 180.0
            and budget >= requested_duration + 55.0
        )
        if not valid:
            return self._follow_person_impl(
                query_value,
                target_distance,
                requested_duration,
                budget,
            )
        arguments = {
            "query": query_value,
            "follow_distance": target_distance,
            "duration": requested_duration,
            "timeout": budget,
        }

        def execute_local(cancelled: Callable[[], bool]) -> str:
            watcher_done = Event()

            def watch_cancel() -> None:
                while not watcher_done.wait(0.02):
                    if cancelled():
                        self._follow_cancel_event.set()
                        return

            watcher = Thread(
                target=watch_cancel,
                name="luxi-follow-person-cancel",
                daemon=True,
            )
            watcher.start()
            try:
                return str(
                    self._follow_person_impl(
                        query_value,
                        target_distance,
                        requested_duration,
                        budget,
                    )
                )
            finally:
                watcher_done.set()
                watcher.join(timeout=0.1)

        unified = IsaacNavigationControlChannel().request_follow_person(
            arguments,
            execute_local,
            timeout_s=budget + 15.0,
        )
        if unified is not None:
            return json.dumps(unified, ensure_ascii=False, sort_keys=True)
        return self._follow_person_impl(
            query_value,
            target_distance,
            requested_duration,
            budget,
        )

    @skill(uses=[CAP_MOVEMENT])
    def approach_visual_target(
        self,
        query: str,
        standoff_distance: float = 0.9,
        timeout: float = 50.0,
    ) -> VisualApproachResult:
        """Approach a target, with at most one bounded viewpoint search.

        Args:
            query: Visible target description, for example ``door``.
            standoff_distance: Desired final distance, between 0.5 and 3 metres.
            timeout: Navigation deadline measured after the goal is accepted.
        """
        return self._run_visual_approach(
            query,
            standoff_distance=float(standoff_distance),
            timeout=float(timeout),
        )

    def _approach_person_impl(
        self,
        query: str = "person",
        standoff_distance: float = 1.2,
        timeout: float = 25.0,
    ) -> VisualApproachResult:
        """Approach one visible, stationary person and stop at a safe distance.

        Args:
            query: Visible description of the person to approach.
            standoff_distance: Desired final distance, between 1 and 2 metres.
            timeout: Closed-loop navigation timeout, between 3 and 30 seconds.
        """
        standoff_distance = float(standoff_distance)
        timeout = float(timeout)
        if not 1.0 <= standoff_distance <= 2.0:
            return VisualApproachResult(
                tool_ok=False,
                task_status="invalid_input",
                completed=False,
                planner_goal_reached=False,
                target_distance_m=None,
                stop_command_publish_latency_ms=0.0,
                physical_stop_latency_ms=None,
                elapsed_s=0.0,
                message="person standoff_distance 必须在 1.0 到 2.0m 之间",
            )
        if not 3.0 <= timeout <= 30.0:
            return VisualApproachResult(
                tool_ok=False,
                task_status="invalid_input",
                completed=False,
                planner_goal_reached=False,
                target_distance_m=None,
                stop_command_publish_latency_ms=0.0,
                physical_stop_latency_ms=None,
                elapsed_s=0.0,
                message="person timeout 必须在 3 到 30s 之间",
            )
        return self._run_visual_approach(
            query,
            standoff_distance=standoff_distance,
            timeout=timeout,
        )

    @skill(uses=[CAP_MOVEMENT])
    def approach_person(
        self,
        query: str = "person",
        standoff_distance: float = 1.2,
        timeout: float = 25.0,
    ) -> VisualApproachResult | str:
        """Approach a person through the unified owner when it is available."""

        query_value = query.strip() if isinstance(query, str) else ""
        try:
            distance = float(standoff_distance)
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            return self._approach_person_impl(query, standoff_distance, timeout)
        if (
            not query_value
            or len(query_value) > 200
            or not math.isfinite(distance)
            or not 1.0 <= distance <= 2.0
            or not math.isfinite(budget)
            or not 3.0 <= budget <= 30.0
        ):
            return self._approach_person_impl(query_value, distance, budget)
        arguments = {
            "query": query_value,
            "standoff_distance": distance,
            "timeout": budget,
        }

        def execute_local(cancelled: Callable[[], bool]) -> str:
            watcher_done = Event()

            def watch_cancel() -> None:
                while not watcher_done.wait(0.02):
                    if cancelled():
                        self._cancel_event.set()
                        return

            watcher = Thread(
                target=watch_cancel,
                name="luxi-approach-person-cancel",
                daemon=True,
            )
            watcher.start()
            try:
                return str(
                    self._approach_person_impl(
                        query_value,
                        distance,
                        budget,
                    )
                )
            finally:
                watcher_done.set()
                watcher.join(timeout=0.1)

        unified = IsaacNavigationControlChannel().request_approach_person(
            arguments,
            execute_local,
            timeout_s=budget + 75.0,
        )
        if unified is not None:
            return json.dumps(unified, ensure_ascii=False, sort_keys=True)
        return self._approach_person_impl(query_value, distance, budget)

    @rpc
    def stop(self) -> None:
        self._follow_cancel_event.set()
        self._cancel_event.set()
        self._safe_stop()
        self._vl_model.stop()
        super().stop()
