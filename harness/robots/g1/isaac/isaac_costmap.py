"""Verified Isaac RTX lidar projection for UI, recovery, and bounded navigation."""

from __future__ import annotations

import base64
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid

from harness.robots.g1.isaac.isaac_protocol import (
    IsaacLidarFrame,
    IsaacRuntimePaths,
    configured_runtime_dir,
    read_fresh_lidar_frame,
    read_fresh_state,
)


class IsaacLidarCostmap:
    """Accumulate a bounded 2-D map from identity-verified world-frame scans."""

    VERTICAL_EVIDENCE_RESOLUTION_M = 0.05
    VERTICAL_CLEARING_RADIUS_BINS = 1

    def __init__(
        self,
        events: Any,
        paths: IsaacRuntimePaths,
        *,
        resolution: float = 0.10,
        width: int = 160,
        height: int = 160,
        inflation_radius_m: float = 0.30,
        max_range_m: float = 8.0,
        max_age_seconds: float = 1.25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.events = events
        self.paths = paths
        self.resolution = max(0.05, float(resolution))
        self.width = max(40, int(width))
        self.height = max(40, int(height))
        self.inflation_radius_m = max(0.0, float(inflation_radius_m))
        self.max_range_m = max(1.0, float(max_range_m))
        self.max_age_seconds = max(0.5, float(max_age_seconds))
        self.clock = clock

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_sequence = 0
        self._received_at: float | None = None
        self._payload: dict[str, Any] | None = None
        self._cells: dict[tuple[int, int], int] = {}
        self._occupied_height_bins: dict[tuple[int, int], set[int]] = {}
        self._root_position: np.ndarray[Any, Any] | None = None
        self._frame_timestamp: float | None = None

    def _new_runtime_epoch_locked(self, frame: IsaacLidarFrame) -> bool:
        if self._payload is None or frame.sequence > self._last_sequence:
            return False
        try:
            previous_timestamp = float(self._payload["timestamp"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        return frame.timestamp > previous_timestamp + self.max_age_seconds

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="isaac-lidar-costmap",
                daemon=True,
            )
            thread = self._thread
        thread.start()
        self.events.append(
            "mapping",
            "lifecycle",
            "Isaac lidar costmap ready",
            "Verified RTX scans feed display, recovery, and explored-area navigation.",
        )

    def _run(self) -> None:
        while not self._stop.wait(0.10):
            frame = read_fresh_lidar_frame(
                self.paths,
                max_age_s=self.max_age_seconds,
            )
            if frame is None:
                continue
            state = read_fresh_state(
                self.paths,
                max_age_s=self.max_age_seconds,
            )
            if state is None:
                continue
            try:
                self.integrate(frame, state)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                self.events.append(
                    "mapping",
                    "error",
                    "Isaac lidar costmap frame dropped",
                    str(error),
                    level="warning",
                )

    def integrate(
        self,
        frame: IsaacLidarFrame,
        state: dict[str, Any],
    ) -> bool:
        """Integrate one verified world-frame scan; exposed for regression tests."""

        with self._lock:
            if (
                frame.sequence <= self._last_sequence
                and not self._new_runtime_epoch_locked(frame)
            ):
                return False

        root_position = np.asarray(state["pose"]["position"], dtype=np.float64)
        sensor_position = np.asarray(
            state.get("lidar_metadata", {}).get(
                "sensor_position",
                root_position,
            ),
            dtype=np.float64,
        )
        if root_position.shape != (3,) or sensor_position.shape != (3,):
            raise ValueError("Isaac pose or lidar sensor position is malformed")
        if not np.all(np.isfinite(root_position)) or not np.all(
            np.isfinite(sensor_position)
        ):
            raise ValueError("Isaac pose or lidar sensor position is not finite")

        points = np.asarray(frame.points, dtype=np.float64)
        floor_z = float(root_position[2]) - 0.80
        relative = points[:, :2] - sensor_position[:2]
        ranges = np.linalg.norm(relative, axis=1)
        finite_returns = np.all(np.isfinite(points), axis=1) & np.isfinite(ranges)
        obstacle_band = (
            finite_returns
            & (points[:, 2] >= floor_z + 0.15)
            & (points[:, 2] <= floor_z + 1.60)
            & (ranges >= 0.30)
            & (ranges <= self.max_range_m)
        )
        # A cold, open scene can legitimately contain only ground-plane returns.
        # Those returns prove traversed ray cells free but must never become
        # occupied endpoints.  Unknown cells outside those measured rays remain
        # blocked by planning_grid(), so this does not manufacture free space.
        ground_band = (
            finite_returns
            & (points[:, 2] >= floor_z - 0.10)
            & (points[:, 2] <= floor_z + 0.12)
            & (ranges >= 0.30)
            & (ranges <= self.max_range_m)
        )
        navigation_returns = obstacle_band | ground_band
        if not np.any(navigation_returns):
            return False

        navigation_points = points[navigation_returns]
        navigation_obstacles = obstacle_band[navigation_returns]
        relative = relative[navigation_returns]
        ranges = ranges[navigation_returns]
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        ray_count = 720
        ray_indices = np.floor(
            ((angles + math.pi) / (2.0 * math.pi)) * ray_count
        ).astype(np.int64) % ray_count
        range_order = np.argsort(ranges)
        _unique_rays, first_positions = np.unique(
            ray_indices[range_order],
            return_index=True,
        )
        nearest_indices = range_order[first_positions]
        if not len(nearest_indices):
            return False

        ray_angles = angles[nearest_indices]
        ray_ranges = ranges[nearest_indices]
        ray_endpoint_z = navigation_points[nearest_indices, 2]
        ray_has_obstacle_endpoint = navigation_obstacles[nearest_indices]
        cosines = np.cos(ray_angles)
        sines = np.sin(ray_angles)
        vertical_slopes = (
            ray_endpoint_z - sensor_position[2]
        ) / ray_ranges

        step_distances = np.arange(
            self.resolution,
            self.max_range_m,
            self.resolution,
            dtype=np.float64,
        )
        free_mask = (
            step_distances[None, :]
            < ray_ranges[:, None] - self.resolution * 0.75
        )
        free_x = (
            sensor_position[0]
            + cosines[:, None] * step_distances[None, :]
        )[free_mask]
        free_y = (
            sensor_position[1]
            + sines[:, None] * step_distances[None, :]
        )[free_mask]
        free_z = (
            sensor_position[2]
            + vertical_slopes[:, None] * step_distances[None, :]
        )[free_mask]
        occupied_x = (
            sensor_position[0]
            + cosines[ray_has_obstacle_endpoint]
            * ray_ranges[ray_has_obstacle_endpoint]
        )
        occupied_y = (
            sensor_position[1]
            + sines[ray_has_obstacle_endpoint]
            * ray_ranges[ray_has_obstacle_endpoint]
        )

        free_ix = np.floor(free_x / self.resolution).astype(np.int64)
        free_iy = np.floor(free_y / self.resolution).astype(np.int64)
        occupied_ix = np.floor(
            occupied_x / self.resolution
        ).astype(np.int64)
        occupied_iy = np.floor(
            occupied_y / self.resolution
        ).astype(np.int64)
        free_height_bins = np.floor(
            free_z / self.VERTICAL_EVIDENCE_RESOLUTION_M
        ).astype(np.int64)
        occupied_height_bins = np.floor(
            ray_endpoint_z[ray_has_obstacle_endpoint]
            / self.VERTICAL_EVIDENCE_RESOLUTION_M
        ).astype(np.int64)

        with self._lock:
            if frame.sequence <= self._last_sequence:
                if not self._new_runtime_epoch_locked(frame):
                    return False
                self._cells.clear()
                self._occupied_height_bins.clear()
                self._payload = None
                self._received_at = None
                self._root_position = None
                self._frame_timestamp = None
                self._last_sequence = 0
            first_frame = self._payload is None
            for ix, iy, height_bin in zip(
                free_ix.tolist(),
                free_iy.tolist(),
                free_height_bins.tolist(),
                strict=True,
            ):
                key = (ix, iy)
                occupied_bins = self._occupied_height_bins.get(key)
                if occupied_bins:
                    remaining_bins = {
                        occupied_bin
                        for occupied_bin in occupied_bins
                        if abs(occupied_bin - height_bin)
                        > self.VERTICAL_CLEARING_RADIUS_BINS
                    }
                    if remaining_bins:
                        self._occupied_height_bins[key] = remaining_bins
                        self._cells[key] = 100
                        continue
                    self._occupied_height_bins.pop(key, None)
                elif self._cells.get(key) == 100:
                    # A hard cell without its vertical provenance must fail
                    # closed rather than accepting an unproven 2-D clear.
                    continue
                self._cells[key] = 0
            for ix, iy, height_bin in zip(
                occupied_ix.tolist(),
                occupied_iy.tolist(),
                occupied_height_bins.tolist(),
                strict=True,
            ):
                key = (ix, iy)
                self._cells[key] = 100
                self._occupied_height_bins.setdefault(key, set()).add(height_bin)
            self._prune(root_position)
            self._last_sequence = frame.sequence
            self._received_at = self.clock()
            self._root_position = root_position.copy()
            self._frame_timestamp = frame.timestamp
            self._payload = self._build_payload(
                root_position,
                frame,
            )

        if first_frame:
            self.events.append(
                "mapping",
                "observation",
                "Isaac live lidar costmap received",
                (
                    f"{self.width}×{self.height} cells · "
                    f"{self.resolution:.2f} m/cell"
                ),
            )
        return True

    def _prune(self, root_position: np.ndarray[Any, Any]) -> None:
        radius_cells = int(math.ceil(30.0 / self.resolution))
        center_ix = int(math.floor(root_position[0] / self.resolution))
        center_iy = int(math.floor(root_position[1] / self.resolution))
        if len(self._cells) < 250_000:
            return
        self._cells = {
            key: value
            for key, value in self._cells.items()
            if abs(key[0] - center_ix) <= radius_cells
            and abs(key[1] - center_iy) <= radius_cells
        }
        self._occupied_height_bins = {
            key: bins
            for key, bins in self._occupied_height_bins.items()
            if key in self._cells
        }

    def _build_payload(
        self,
        root_position: np.ndarray[Any, Any],
        frame: IsaacLidarFrame,
    ) -> dict[str, Any]:
        center_ix = int(math.floor(root_position[0] / self.resolution))
        center_iy = int(math.floor(root_position[1] / self.resolution))
        min_ix = center_ix - self.width // 2
        min_iy = center_iy - self.height // 2
        max_ix = min_ix + self.width
        max_iy = min_iy + self.height
        grid = np.full((self.height, self.width), 255, dtype=np.uint8)
        occupied: list[tuple[int, int]] = []
        for (ix, iy), value in self._cells.items():
            if min_ix <= ix < max_ix and min_iy <= iy < max_iy:
                row = iy - min_iy
                column = ix - min_ix
                grid[row, column] = value
                if value == 100:
                    occupied.append((row, column))

        inflation_cells = int(
            math.ceil(self.inflation_radius_m / self.resolution)
        )
        if inflation_cells and occupied:
            offsets = [
                (dy, dx)
                for dy in range(-inflation_cells, inflation_cells + 1)
                for dx in range(-inflation_cells, inflation_cells + 1)
                if dx * dx + dy * dy <= inflation_cells * inflation_cells
            ]
            for dy, dx in offsets:
                for row, column in occupied:
                    target_row = row + dy
                    target_column = column + dx
                    if (
                        0 <= target_row < self.height
                        and 0 <= target_column < self.width
                        and grid[target_row, target_column] != 100
                    ):
                        grid[target_row, target_column] = 60
            for row, column in occupied:
                grid[row, column] = 100

        unknown = int(np.count_nonzero(grid == 255))
        free = int(np.count_nonzero(grid == 0))
        occupied_count = int(np.count_nonzero(grid == 100))
        high_cost = int(np.count_nonzero((grid >= 50) & (grid < 100)))
        return {
            "available": True,
            "version": 1,
            "revision": frame.sequence,
            "source": "isaac_lidar_live",
            "recovery_eligible": True,
            "navigation_eligible": True,
            "planning_scope": "explored_same_floor",
            "identity_verified": True,
            "unknown_is_blocked": True,
            "clearing_policy": "height_consistent_ray",
            "vertical_evidence_resolution_m": (
                self.VERTICAL_EVIDENCE_RESOLUTION_M
            ),
            "topic": "isaac://verified-rtx-lidar",
            "frame_id": "world",
            "timestamp": frame.timestamp,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin": {
                "x": min_ix * self.resolution,
                "y": min_iy * self.resolution,
                "yaw": 0.0,
            },
            "encoding": "base64-int8-row-major",
            "data": base64.b64encode(grid.tobytes(order="C")).decode("ascii"),
            "cells": {
                "total": self.width * self.height,
                "unknown": unknown,
                "free": free,
                "cost": 0,
                "occupied": occupied_count,
                "saturated_height_cost": occupied_count,
                "high_cost": high_cost,
                "known": self.width * self.height - unknown,
            },
        }

    def payload(self) -> dict[str, Any]:
        with self._lock:
            payload = None if self._payload is None else dict(self._payload)
            received_at = self._received_at
            running = self._running
        if payload is None or received_at is None:
            return {"available": False, "running": running}
        age = max(0.0, self.clock() - received_at)
        payload["age_seconds"] = round(age, 3)
        payload["available"] = bool(age <= self.max_age_seconds)
        payload["running"] = running
        return payload

    def planning_grid(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        footprint_radius_m: float = 0.28,
    ) -> OccupancyGrid | None:
        """Return a planner grid that fails closed outside observed free space.

        The UI payload uses 255 for unknown and 60 for display-only inflation.
        DimOS A* assigns a finite penalty to its conventional ``-1`` unknown
        value, so the navigation copy deliberately encodes every unknown cell
        as occupied (100). Physical lidar endpoints remain occupied while the
        robot's current footprint is proven free only where no endpoint exists.
        """

        grid_width = self.width if width is None else max(40, int(width))
        grid_height = self.height if height is None else max(40, int(height))
        with self._lock:
            if self._root_position is None or self._frame_timestamp is None:
                return None
            root = self._root_position.copy()
            timestamp = self._frame_timestamp
            cells = dict(self._cells)

        center_ix = int(math.floor(root[0] / self.resolution))
        center_iy = int(math.floor(root[1] / self.resolution))
        min_ix = center_ix - grid_width // 2
        min_iy = center_iy - grid_height // 2
        max_ix = min_ix + grid_width
        max_iy = min_iy + grid_height

        # Unknown is intentionally occupied at the planner boundary.
        grid = np.full((grid_height, grid_width), 100, dtype=np.int8)
        for (ix, iy), value in cells.items():
            if min_ix <= ix < max_ix and min_iy <= iy < max_iy:
                grid[iy - min_iy, ix - min_ix] = np.int8(value)

        footprint_cells = int(
            math.ceil(max(0.0, float(footprint_radius_m)) / self.resolution)
        )
        for dy in range(-footprint_cells, footprint_cells + 1):
            for dx in range(-footprint_cells, footprint_cells + 1):
                if dx * dx + dy * dy > footprint_cells * footprint_cells:
                    continue
                world_key = (center_ix + dx, center_iy + dy)
                row = center_iy + dy - min_iy
                column = center_ix + dx - min_ix
                if (
                    0 <= row < grid_height
                    and 0 <= column < grid_width
                    and cells.get(world_key) != 100
                ):
                    grid[row, column] = 0

        return OccupancyGrid(
            grid=grid,
            resolution=self.resolution,
            origin=Pose(min_ix * self.resolution, min_iy * self.resolution, 0.0),
            frame_id="world",
            ts=timestamp,
        )

    def exploration_grid(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        footprint_radius_m: float = 0.28,
    ) -> OccupancyGrid | None:
        """Return sensor evidence with conventional unknown-space encoding.

        This copy is exclusively for frontier selection.  Unknown cells stay
        ``-1`` so they remain distinguishable from physical lidar endpoints
        at ``100``.  It must never be connected to the A* planner; navigation
        continues to consume :meth:`planning_grid`, where unknown is blocked.
        """

        grid_width = self.width if width is None else max(40, int(width))
        grid_height = self.height if height is None else max(40, int(height))
        with self._lock:
            if self._root_position is None or self._frame_timestamp is None:
                return None
            root = self._root_position.copy()
            timestamp = self._frame_timestamp
            cells = dict(self._cells)

        center_ix = int(math.floor(root[0] / self.resolution))
        center_iy = int(math.floor(root[1] / self.resolution))
        min_ix = center_ix - grid_width // 2
        min_iy = center_iy - grid_height // 2
        max_ix = min_ix + grid_width
        max_iy = min_iy + grid_height

        grid = np.full((grid_height, grid_width), -1, dtype=np.int8)
        for (ix, iy), value in cells.items():
            if min_ix <= ix < max_ix and min_iy <= iy < max_iy:
                grid[iy - min_iy, ix - min_ix] = np.int8(value)

        # Current robot occupancy is direct evidence of traversable space, but
        # it may not erase a physical endpoint retained by the lidar mapper.
        footprint_cells = int(
            math.ceil(max(0.0, float(footprint_radius_m)) / self.resolution)
        )
        for dy in range(-footprint_cells, footprint_cells + 1):
            for dx in range(-footprint_cells, footprint_cells + 1):
                if dx * dx + dy * dy > footprint_cells * footprint_cells:
                    continue
                world_key = (center_ix + dx, center_iy + dy)
                row = center_iy + dy - min_iy
                column = center_ix + dx - min_ix
                if (
                    0 <= row < grid_height
                    and 0 <= column < grid_width
                    and cells.get(world_key) != 100
                ):
                    grid[row, column] = 0

        return OccupancyGrid(
            grid=grid,
            resolution=self.resolution,
            origin=Pose(min_ix * self.resolution, min_iy * self.resolution, 0.0),
            frame_id="world",
            ts=timestamp,
        )

    def status(self) -> dict[str, Any]:
        payload = self.payload()
        payload.pop("data", None)
        return payload

    def reset(self) -> list[str]:
        with self._lock:
            self._cells.clear()
            self._occupied_height_bins.clear()
            self._payload = None
            self._received_at = None
            self._root_position = None
            self._frame_timestamp = None
            self._last_sequence = 0
        return []

    def save_now(self, *, announce: bool = True) -> tuple[bool, str]:
        del announce
        return False, "Isaac costmap 是实时规划输入，不持久化为离线导航地图"

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            thread = self._thread
        self._stop.set()
        if thread is not None:
            thread.join(timeout=1.0)


class _NullEvents:
    def append(self, *_args: Any, **_kwargs: Any) -> None:
        return


class IsaacGlobalCostmapConfig(ModuleConfig):
    runtime_dir: str = Field(default_factory=lambda: str(configured_runtime_dir()))
    resolution: float = Field(default=0.10, ge=0.05, le=0.25)
    width: int = Field(default=240, ge=80, le=400)
    height: int = Field(default=240, ge=80, le=400)
    max_range_m: float = Field(default=8.0, ge=2.0, le=12.0)
    max_age_seconds: float = Field(default=1.25, ge=0.5, le=3.0)
    footprint_radius_m: float = Field(default=0.28, ge=0.15, le=0.50)
    exploration_publish_interval_s: float = Field(
        default=0.5,
        ge=0.2,
        le=2.0,
    )


class IsaacGlobalCostmapModule(Module):
    """Publish separate planning and frontier maps from verified Isaac files."""

    config: IsaacGlobalCostmapConfig
    global_costmap: Out[OccupancyGrid]
    exploration_costmap: Out[OccupancyGrid]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        paths = IsaacRuntimePaths(
            Path(self.config.runtime_dir).expanduser().resolve()
        )
        self._mapper = IsaacLidarCostmap(
            _NullEvents(),
            paths,
            resolution=self.config.resolution,
            width=self.config.width,
            height=self.config.height,
            inflation_radius_m=0.0,
            max_range_m=self.config.max_range_m,
            max_age_seconds=self.config.max_age_seconds,
        )
        self._paths = paths
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_exploration_publish_at = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="isaac-global-costmap",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(0.05):
            frame = read_fresh_lidar_frame(
                self._paths,
                max_age_s=self.config.max_age_seconds,
            )
            state = read_fresh_state(
                self._paths,
                max_age_s=self.config.max_age_seconds,
            )
            if frame is None or state is None:
                continue
            try:
                if not self._mapper.integrate(frame, state):
                    continue
                grid = self._mapper.planning_grid(
                    width=self.config.width,
                    height=self.config.height,
                    footprint_radius_m=self.config.footprint_radius_m,
                )
                exploration_grid = self._mapper.exploration_grid(
                    width=self.config.width,
                    height=self.config.height,
                    footprint_radius_m=self.config.footprint_radius_m,
                )
                if grid is not None and exploration_grid is not None:
                    self.global_costmap.publish(grid)
                    now = time.monotonic()
                    if (
                        now - self._last_exploration_publish_at
                        >= self.config.exploration_publish_interval_s
                    ):
                        self.exploration_costmap.publish(exploration_grid)
                        self._last_exploration_publish_at = now
            except (KeyError, TypeError, ValueError, OverflowError):
                # Invalid/stale sensor evidence must result in no new map.
                continue

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        super().stop()
