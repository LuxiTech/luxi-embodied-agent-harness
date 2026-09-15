"""Lightweight, exact-pose location tags for the Isaac G1 backend."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from threading import RLock
import time
from typing import Any
import unicodedata

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.navigation.base import NavigationState
from dimos.navigation.navigation_spec import NavigationInterfaceSpec

from harness.robots.g1.isaac.stop_service import IsaacStopControlChannel
from harness.robots.g1.isaac.navigation_service import IsaacNavigationControlChannel
from harness.robots.g1.isaac.isaac_protocol import (
    BACKEND_NAME,
    SCHEMA_VERSION,
    IsaacRuntimePaths,
    atomic_write_json,
    read_json,
)


TAGGED_LOCATIONS_PATH_ENV = "LUXI_ISAAC_TAGGED_LOCATIONS_PATH"
SHARED_TAGGED_LOCATIONS_PATH_ENV = "LUXI_TAGGED_LOCATIONS_PATH"
MAX_LOCATION_STORE_BYTES = 256 * 1024


def configured_tagged_locations_path() -> Path:
    shared = os.getenv(SHARED_TAGGED_LOCATIONS_PATH_ENV, "").strip()
    if shared:
        return Path(shared).expanduser().resolve()
    configured = os.getenv(TAGGED_LOCATIONS_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return IsaacRuntimePaths.configured().root / "tagged-locations.json"


class IsaacLocationTagConfig(ModuleConfig):
    storage_path: Path = Field(default_factory=configured_tagged_locations_path)
    max_odom_age_s: float = Field(default=1.5, gt=0.0, le=10.0)
    max_costmap_age_s: float = Field(default=1.5, gt=0.0, le=10.0)
    max_locations: int = Field(default=256, ge=1, le=4096)
    max_goal_distance_m: float = Field(default=8.0, gt=0.25, le=20.0)
    goal_clearance_m: float = Field(default=0.25, ge=0.15, le=0.60)
    # DimOS may replace an exact tag with the nearest costmap-safe cell and its
    # replanning completion branch explicitly uses a 0.5 m goal tolerance.
    # Keep the terminal verifier aligned with that upstream arrival contract;
    # yaw, stationary and fresh-map requirements remain independent.
    position_tolerance_m: float = Field(default=0.50, ge=0.10, le=0.50)
    yaw_tolerance_rad: float = Field(default=0.40, ge=0.15, le=0.80)
    stationary_seconds: float = Field(default=0.8, ge=0.3, le=3.0)
    goal_result_grace_s: float = Field(default=0.75, ge=0.2, le=2.0)


class IsaacLocationTagSkillContainer(Module):
    """Persist exact poses and execute terminal explored-area navigation."""

    config: IsaacLocationTagConfig
    odom: In[PoseStamped]
    global_costmap: In[OccupancyGrid]
    _navigation: NavigationInterfaceSpec
    _monotonic = staticmethod(time.monotonic)
    _wall_time = staticmethod(time.time)
    _sleep = staticmethod(time.sleep)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_odom: PoseStamped | None = None
        self._latest_odom_received_at = float("-inf")
        self._odom_revision = 0
        self._latest_costmap: OccupancyGrid | None = None
        self._latest_costmap_received_at = float("-inf")
        self._lock = RLock()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(
            Disposable(self.global_costmap.subscribe(self._on_costmap))
        )

    def _on_odom(self, pose: PoseStamped) -> None:
        with self._lock:
            self._latest_odom = pose
            self._latest_odom_received_at = self._monotonic()
            self._odom_revision += 1

    def _on_costmap(self, costmap: OccupancyGrid) -> None:
        with self._lock:
            self._latest_costmap = costmap
            self._latest_costmap_received_at = self._monotonic()

    @staticmethod
    def _normalized_name(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        name = unicodedata.normalize("NFKC", value).strip()
        if not name or len(name) > 80:
            return None
        if any(unicodedata.category(character).startswith("C") for character in name):
            return None
        return name

    @staticmethod
    def _finite_vector(value: Any, size: int) -> bool:
        return bool(
            isinstance(value, list)
            and len(value) == size
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in value
            )
        )

    @classmethod
    def _valid_entry(cls, key: Any, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        stored_name = cls._normalized_name(value.get("name"))
        if (
            not isinstance(key, str)
            or cls._normalized_name(key) != key
            or stored_name is None
            or key != stored_name.casefold()
        ):
            return False
        return bool(
            value.get("frame_id") == "world"
            and cls._finite_vector(value.get("position"), 3)
            and cls._finite_vector(value.get("quaternion_xyzw"), 4)
            and isinstance(value.get("pose_timestamp"), (int, float))
            and not isinstance(value.get("pose_timestamp"), bool)
            and math.isfinite(float(value["pose_timestamp"]))
            and isinstance(value.get("tagged_at"), (int, float))
            and not isinstance(value.get("tagged_at"), bool)
            and math.isfinite(float(value["tagged_at"]))
        )

    def _load_store(self, path: Path) -> tuple[dict[str, Any] | None, str | None]:
        if not path.exists():
            return {
                "schema_version": SCHEMA_VERSION,
                "backend": BACKEND_NAME,
                "locations": {},
            }, None
        if path.is_symlink() or not path.is_file():
            return None, "existing location store is invalid"
        payload = read_json(path, max_bytes=MAX_LOCATION_STORE_BYTES)
        if (
            payload is None
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("backend") != BACKEND_NAME
            or not isinstance(payload.get("locations"), dict)
        ):
            return None, "existing location store is invalid"
        locations = payload["locations"]
        if len(locations) > self.config.max_locations or not all(
            self._valid_entry(key, value) for key, value in locations.items()
        ):
            return None, "existing location store is invalid"
        return payload, None

    @skill
    def tag_location(self, location_name: str) -> str:
        """Record the robot's current fresh world pose under an exact name.

        Args:
            location_name: A unique human-readable name, from 1 through 80 characters.
        """

        name = self._normalized_name(location_name)
        if name is None:
            return "Error: tag_location_failed: location name is invalid"
        with self._lock:
            odom = self._latest_odom
            age = self._monotonic() - self._latest_odom_received_at
            if odom is None or age < 0.0 or age > self.config.max_odom_age_s:
                return "Error: tag_location_failed: fresh odometry is unavailable"

            position = [
                float(odom.position.x),
                float(odom.position.y),
                float(odom.position.z),
            ]
            quaternion = [
                float(odom.orientation.x),
                float(odom.orientation.y),
                float(odom.orientation.z),
                float(odom.orientation.w),
            ]
            pose_timestamp = float(odom.ts)
            if (
                odom.frame_id != "world"
                or not self._finite_vector(position, 3)
                or not self._finite_vector(quaternion, 4)
                or not math.isfinite(pose_timestamp)
            ):
                return "Error: tag_location_failed: odometry pose is invalid"

            path = Path(self.config.storage_path).expanduser()
            payload, error = self._load_store(path)
            if payload is None:
                return f"Error: tag_location_failed: {error}"
            locations = payload["locations"]
            key = name.casefold()
            if key not in locations and len(locations) >= self.config.max_locations:
                return "Error: tag_location_failed: location store is full"
            locations[key] = {
                "name": name,
                "frame_id": "world",
                "position": position,
                "quaternion_xyzw": quaternion,
                "pose_timestamp": pose_timestamp,
                "tagged_at": self._wall_time(),
            }
            try:
                atomic_write_json(path, payload)
            except OSError as exc:
                return (
                    "Error: tag_location_failed: could not persist location store "
                    f"({type(exc).__name__})"
                )

        return (
            f"Tagged '{name}': ({position[0]:.2f},{position[1]:.2f}). "
            "Use navigate_to_tag to return through observed free space."
        )

    @staticmethod
    def _result(
        status: str,
        completed: bool,
        **details: Any,
    ) -> str:
        return json.dumps(
            {
                "task_status": status,
                "tool_ok": completed,
                "completed": completed,
                **details,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _fresh_inputs(
        self,
    ) -> tuple[PoseStamped | None, OccupancyGrid | None, str | None]:
        now = self._monotonic()
        with self._lock:
            odom = self._latest_odom
            odom_age = now - self._latest_odom_received_at
            costmap = self._latest_costmap
            costmap_age = now - self._latest_costmap_received_at
        if (
            odom is None
            or odom.frame_id != "world"
            or odom_age < 0.0
            or odom_age > self.config.max_odom_age_s
        ):
            return None, None, "fresh world odometry is unavailable"
        if (
            costmap is None
            or costmap.frame_id != "world"
            or costmap_age < 0.0
            or costmap_age > self.config.max_costmap_age_s
        ):
            return None, None, "fresh verified global costmap is unavailable"
        return odom, costmap, None

    def _goal_has_clearance(
        self,
        costmap: OccupancyGrid,
        x: float,
        y: float,
    ) -> bool:
        radius_cells = int(
            math.ceil(self.config.goal_clearance_m / costmap.resolution)
        )
        center = costmap.world_to_grid(Vector3(x, y, 0.0))
        center_x, center_y = int(center.x), int(center.y)
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if (
                    math.hypot(dx, dy) * costmap.resolution
                    > self.config.goal_clearance_m
                ):
                    continue
                grid_x, grid_y = center_x + dx, center_y + dy
                if (
                    grid_x < 0
                    or grid_y < 0
                    or grid_x >= costmap.width
                    or grid_y >= costmap.height
                    or int(costmap.grid[grid_y, grid_x]) != 0
                ):
                    return False
        return True

    def _confirm_stationary_at(
        self,
        goal: PoseStamped,
        deadline: float,
    ) -> tuple[bool, dict[str, float | int]]:
        settle_started = self._monotonic()
        with self._lock:
            previous = self._latest_odom
            previous_revision = self._odom_revision
        sample_count = 0
        while self._monotonic() < deadline:
            with self._lock:
                current = self._latest_odom
                revision = self._odom_revision
                odom_age = self._monotonic() - self._latest_odom_received_at
            if current is None or odom_age > self.config.max_odom_age_s:
                return False, {"stationary_samples": sample_count}
            if revision != previous_revision and previous is not None:
                displacement = current.position.distance(previous.position)
                if displacement > 0.03:
                    settle_started = self._monotonic()
                    sample_count = 0
                else:
                    sample_count += 1
                previous = current
                previous_revision = revision
            position_error = current.position.distance(goal.position)
            yaw_error = abs(
                math.atan2(
                    math.sin(current.orientation.euler[2] - goal.orientation.euler[2]),
                    math.cos(current.orientation.euler[2] - goal.orientation.euler[2]),
                )
            )
            settled = self._monotonic() - settle_started
            if (
                sample_count >= 3
                and settled >= self.config.stationary_seconds
                and position_error <= self.config.position_tolerance_m
                and yaw_error <= self.config.yaw_tolerance_rad
            ):
                return True, {
                    "stationary_samples": sample_count,
                    "position_error_m": round(position_error, 3),
                    "yaw_error_rad": round(yaw_error, 3),
                }
            self._sleep(0.05)
        return False, {"stationary_samples": sample_count}

    def _navigate_to_goal(
        self,
        goal: PoseStamped,
        *,
        timeout_seconds: float,
        target_name: str | None = None,
        cancelled: Any | None = None,
    ) -> str:
        odom, costmap, error = self._fresh_inputs()
        if error is not None or odom is None or costmap is None:
            return self._result("navigation_rejected", False, error=error)
        distance = odom.position.distance(goal.position)
        if not math.isfinite(distance) or distance > self.config.max_goal_distance_m:
            return self._result(
                "navigation_rejected",
                False,
                error="goal exceeds same-floor bounded navigation range",
                distance_m=round(distance, 3),
            )
        if abs(goal.position.z - odom.position.z) > 0.35:
            return self._result(
                "navigation_rejected",
                False,
                error="goal is not on the current floor",
            )
        if not self._goal_has_clearance(costmap, goal.position.x, goal.position.y):
            return self._result(
                "navigation_rejected",
                False,
                error="goal or required clearance is unknown/occupied",
            )

        timeout = max(1.0, min(110.0, float(timeout_seconds)))
        deadline = self._monotonic() + timeout
        try:
            if not self._navigation.set_goal(goal):
                return self._result(
                    "navigation_rejected",
                    False,
                    error="planner rejected goal",
                )
            observed_active = False
            idle_after_active_since: float | None = None
            initial_idle_deadline = min(deadline, self._monotonic() + 1.0)
            while self._monotonic() < deadline:
                if callable(cancelled) and cancelled():
                    self._navigation.cancel_goal()
                    return self._result(
                        "navigation_cancelled",
                        False,
                        error="unified navigation owner cancelled the request",
                    )
                with self._lock:
                    active_costmap = self._latest_costmap
                    active_costmap_age = (
                        self._monotonic() - self._latest_costmap_received_at
                    )
                if (
                    active_costmap is None
                    or active_costmap.frame_id != "world"
                    or active_costmap_age < 0.0
                    or active_costmap_age > self.config.max_costmap_age_s
                ):
                    self._navigation.cancel_goal()
                    return self._result(
                        "navigation_blocked",
                        False,
                        error="fresh verified global costmap was lost during navigation",
                    )
                if not self._goal_has_clearance(
                    active_costmap,
                    goal.position.x,
                    goal.position.y,
                ):
                    self._navigation.cancel_goal()
                    return self._result(
                        "navigation_blocked",
                        False,
                        error="goal clearance became unknown/occupied during navigation",
                    )
                state = self._navigation.get_state()
                reached = self._navigation.is_goal_reached()
                if state != NavigationState.IDLE:
                    observed_active = True
                    idle_after_active_since = None
                if state == NavigationState.IDLE and reached:
                    stationary, details = self._confirm_stationary_at(goal, deadline)
                    if stationary:
                        return self._result(
                            "navigation_verified",
                            True,
                            planner_goal_reached=True,
                            stationary_confirmed=True,
                            distance_m=round(distance, 3),
                            target=target_name,
                            **details,
                        )
                    self._navigation.cancel_goal()
                    return self._result(
                        "navigation_unverified",
                        False,
                        error="goal signal was not followed by a stationary pose",
                        **details,
                    )
                if state == NavigationState.IDLE and observed_active and not reached:
                    if idle_after_active_since is None:
                        idle_after_active_since = self._monotonic()
                    elif (
                        self._monotonic() - idle_after_active_since
                        >= self.config.goal_result_grace_s
                    ):
                        self._navigation.cancel_goal()
                        return self._result(
                            "navigation_failed",
                            False,
                            error="planner stopped without reaching the goal",
                        )
                if (
                    state == NavigationState.IDLE
                    and not observed_active
                    and self._monotonic() >= initial_idle_deadline
                ):
                    self._navigation.cancel_goal()
                    return self._result(
                        "navigation_rejected",
                        False,
                        error="planner could not produce a path in observed free space",
                    )
                self._sleep(0.05)
        except Exception as exc:  # noqa: BLE001 - skill must fail closed
            try:
                self._navigation.cancel_goal()
            except Exception:
                pass
            return self._result(
                "navigation_failed",
                False,
                error=f"planner error: {type(exc).__name__}",
            )

        self._navigation.cancel_goal()
        return self._result(
            "navigation_timeout",
            False,
            error=f"goal not verified within {timeout:.1f}s",
        )

    @skill(uses=[CAP_MOVEMENT])
    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw_degrees: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> str:
        """Navigate to an exact current-floor world pose through observed free space.

        Args:
            x: Goal world X coordinate in metres.
            y: Goal world Y coordinate in metres.
            yaw_degrees: Final world yaw angle in degrees.
            timeout_seconds: Terminal deadline, from 1 through 110 seconds.
        """

        values = (float(x), float(y), float(yaw_degrees), float(timeout_seconds))
        if not all(math.isfinite(value) for value in values):
            return self._result(
                "navigation_rejected",
                False,
                error="goal values must be finite",
            )
        with self._lock:
            z = (
                float(self._latest_odom.position.z)
                if self._latest_odom is not None
                else 0.0
            )
        yaw = math.radians(values[2])
        goal = PoseStamped(
            ts=self._wall_time(),
            frame_id="world",
            position=Vector3(values[0], values[1], z),
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
        )
        arguments = {
            "x": values[0],
            "y": values[1],
            "yaw_degrees": values[2],
            "timeout_seconds": values[3],
        }
        unified = IsaacNavigationControlChannel().request_navigate_to_pose(
            arguments,
            lambda cancelled: self._navigate_to_goal(
                goal,
                timeout_seconds=values[3],
                cancelled=cancelled,
            ),
            timeout_s=values[3] + 15.0,
        )
        if unified is not None:
            return json.dumps(unified, ensure_ascii=False, sort_keys=True)
        return self._navigate_to_goal(
            goal,
            timeout_seconds=values[3],
        )

    @skill(uses=[CAP_MOVEMENT])
    def navigate_to_tag(
        self,
        location_name: str,
        timeout_seconds: float = 60.0,
    ) -> str:
        """Return to an exact named Isaac pose through observed free space."""

        name = self._normalized_name(location_name)
        if name is None:
            return self._result(
                "navigation_rejected",
                False,
                error="location name is invalid",
            )
        payload, error = self._load_store(Path(self.config.storage_path).expanduser())
        if payload is None:
            return self._result("navigation_rejected", False, error=error)
        entry = payload["locations"].get(name.casefold())
        if not self._valid_entry(name.casefold(), entry):
            return self._result(
                "navigation_rejected",
                False,
                error=f"no exact tagged location named '{name}'",
            )
        goal = PoseStamped(
            ts=self._wall_time(),
            frame_id="world",
            position=Vector3(*entry["position"]),
            orientation=Quaternion(entry["quaternion_xyzw"]),
        )
        arguments = {
            "location_name": name,
            "timeout_seconds": float(timeout_seconds),
        }
        unified = IsaacNavigationControlChannel().request_navigate_to_tag(
            arguments,
            lambda cancelled: self._navigate_to_goal(
                goal,
                timeout_seconds=timeout_seconds,
                target_name=entry["name"],
                cancelled=cancelled,
            ),
            timeout_s=float(timeout_seconds) + 15.0,
        )
        if unified is not None:
            return json.dumps(unified, ensure_ascii=False, sort_keys=True)
        return self._navigate_to_goal(
            goal,
            timeout_seconds=timeout_seconds,
            target_name=entry["name"],
        )

    @skill
    def stop_navigation(self) -> str:
        """Immediately cancel the current planned Isaac motion."""

        try:
            stopped = bool(self._navigation.cancel_goal())
        except Exception as exc:  # noqa: BLE001
            return self._result(
                "navigation_stop_failed",
                False,
                error=type(exc).__name__,
            )
        stop = IsaacStopControlChannel().request_stop(
            action="stop_navigation"
        )
        return self._result(
            "navigation_stopped",
            stopped,
            stop=stop,
        )
