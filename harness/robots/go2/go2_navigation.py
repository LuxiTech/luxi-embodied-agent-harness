"""MID-360/IMU 2-D SLAM map and known-free frontier planning for Go2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def coordinate_navigation_timeout_s(
    route_xy: tuple[tuple[float, float], ...],
    *,
    measured_speed_mps: float = 0.05,
) -> float:
    """Size a bounded deadline from route length and measured gait speed."""

    if len(route_xy) < 2:
        return 240.0
    route_length_m = sum(
        math.dist(start, end)
        for start, end in zip(route_xy, route_xy[1:], strict=False)
    )
    return float(
        np.clip(route_length_m / max(0.01, measured_speed_mps) + 60.0, 180.0, 360.0)
    )


@dataclass(frozen=True)
class SlamPose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class LocalPlanCommand:
    linear_x: float
    linear_y: float
    angular_z: float
    status: str
    distance_to_goal: float | None = None
    route_waypoint_index: int | None = None


class RosLocalPlanner:
    """Robot-local, continuously replanned costmap-to-Twist controller."""

    def __init__(
        self,
        *,
        standoff_m: float = 0.90,
        arrival_slack_m: float = 0.35,
        arrival_heading_tolerance_deg: float = 8.0,
        alignment_max_yaw_rate: float = 0.35,
    ) -> None:
        self.standoff_m = float(standoff_m)
        self.arrival_slack_m = float(arrival_slack_m)
        self.arrival_heading_tolerance_rad = math.radians(
            float(arrival_heading_tolerance_deg)
        )
        self.alignment_max_yaw_rate = float(alignment_max_yaw_rate)

    def command(
        self,
        slam: "Mid360Slam2D",
        goal_xy: tuple[float, float],
        *,
        route_xy: tuple[tuple[float, float], ...] = (),
        route_waypoint_index: int = 0,
        route_lookahead_m: float = 1.20,
        alignment_locked: bool = False,
        semantic_obstacles: tuple[tuple[float, float, float], ...] = (),
    ) -> LocalPlanCommand:
        pose = slam.pose
        if pose is None:
            return LocalPlanCommand(0.0, 0.0, 0.0, "pose_unavailable")
        distance = math.dist((pose.x, pose.y), goal_xy)
        target_bearing = wrap_angle(
            math.atan2(goal_xy[1] - pose.y, goal_xy[0] - pose.x) - pose.yaw
        )
        if alignment_locked or distance <= self.standoff_m + self.arrival_slack_m:
            if abs(target_bearing) > self.arrival_heading_tolerance_rad:
                return LocalPlanCommand(
                    0.0,
                    0.0,
                    float(
                        np.clip(
                            0.9 * target_bearing,
                            -self.alignment_max_yaw_rate,
                            self.alignment_max_yaw_rate,
                        )
                    ),
                    "aligning",
                    distance,
                )
            return LocalPlanCommand(0.0, 0.0, 0.0, "arrived", distance)
        guidance_goal = goal_xy
        route_progress_index: int | None = None
        if route_xy:
            start_index = min(max(0, int(route_waypoint_index)), len(route_xy) - 1)
            # Recover progress from pose without allowing a noisy estimate to
            # move the route cursor backwards. The short along-route lookahead
            # prevents the online frontier selector from cutting directly
            # through a fixture that the configured semantic route goes around.
            nearest = min(
                range(start_index, len(route_xy)),
                key=lambda index: math.dist(
                    (pose.x, pose.y), route_xy[index]
                ),
            )
            route_progress_index = nearest
            guidance_index = nearest
            accumulated = 0.0
            while guidance_index + 1 < len(route_xy):
                segment = math.dist(
                    route_xy[guidance_index],
                    route_xy[guidance_index + 1],
                )
                if accumulated + segment > max(0.25, float(route_lookahead_m)):
                    break
                accumulated += segment
                guidance_index += 1
            guidance_goal = route_xy[guidance_index]
        path = slam.frontier_path(
            goal_hint=guidance_goal,
            semantic_obstacles=semantic_obstacles,
        )
        if path is None or len(path) < 2:
            return LocalPlanCommand(
                0.0,
                0.0,
                0.0,
                "path_unavailable",
                distance,
                route_progress_index,
            )
        # Replan from the newest map on every call, then command only the first
        # validated short segment. No old multi-waypoint route is retained.
        waypoint = path[1]
        if not slam.segment_is_traversable((pose.x, pose.y), waypoint):
            return LocalPlanCommand(
                0.0,
                0.0,
                0.0,
                "path_invalidated",
                distance,
                route_progress_index,
            )
        dx, dy = waypoint[0] - pose.x, waypoint[1] - pose.y
        bearing = wrap_angle(math.atan2(dy, dx) - pose.yaw)
        local_x = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy
        local_y = -math.sin(pose.yaw) * dx + math.cos(pose.yaw) * dy
        angular_z = float(np.clip(0.9 * bearing, -0.45, 0.45))
        if abs(bearing) > math.radians(45.0):
            linear_x = linear_y = 0.0
        else:
            linear_x = float(np.clip(0.55 * local_x, -0.12, 0.20))
            linear_y = float(np.clip(0.35 * local_y, -0.10, 0.10))
        return LocalPlanCommand(
            linear_x,
            linear_y,
            angular_z,
            "tracking",
            distance,
            route_progress_index,
        )


class Mid360Slam2D:
    """FAST-LIO registered-cloud occupancy projection and frontier planning.

    ``observe_fastlio`` is the production Go2 path. ``observe`` remains as a
    compatibility seam for focused mapper tests and recorded legacy frames;
    it is not used by the MuJoCo runtime.
    """

    def __init__(
        self,
        *,
        width: int = 120,
        height: int = 120,
        resolution: float = 0.10,
        origin_x: float = -6.0,
        origin_y: float = -6.0,
        footprint_radius: float = 0.32,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.footprint_radius = footprint_radius
        self.grid = np.full((height, width), -1, dtype=np.int8)
        # Signed occupancy evidence avoids turning every isolated endpoint
        # into a permanent wall and prevents a single later free ray from
        # erasing a repeatedly observed obstacle.
        self._occupancy_evidence = np.zeros((height, width), dtype=np.int16)
        self._observed = np.zeros((height, width), dtype=np.bool_)
        # Lowest sensor-frame endpoint height ever supporting each occupied
        # column.  A later ray above that height is not evidence that the
        # low part of the column became free.  This is the minimum height
        # layer needed to keep a low obstacle from disappearing when the
        # MID-360 enters its near-field vertical blind region.
        self._lowest_endpoint_z = np.full(
            (height, width), np.inf, dtype=np.float64
        )
        self.pose: SlamPose | None = None
        self._last_odometry: SlamPose | None = None
        self._last_timestamp: float | None = None
        self.revision = 0
        self.last_scan_match_score: float | None = None

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        column = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        if 0 <= row < self.height and 0 <= column < self.width:
            return row, column
        return None

    def cell_to_world(self, row: int, column: int) -> tuple[float, float]:
        return (
            self.origin_x + (column + 0.5) * self.resolution,
            self.origin_y + (row + 0.5) * self.resolution,
        )

    @staticmethod
    def _ray_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        row0, column0 = start
        row1, column1 = end
        dx, dy = abs(column1 - column0), -abs(row1 - row0)
        sx, sy = (1 if column0 < column1 else -1), (1 if row0 < row1 else -1)
        error = dx + dy
        result: list[tuple[int, int]] = []
        while True:
            result.append((row0, column0))
            if row0 == row1 and column0 == column1:
                return result
            twice = 2 * error
            if twice >= dy:
                error += dy
                column0 += sx
            if twice <= dx:
                error += dx
                row0 += sy

    @staticmethod
    def _transform(points: np.ndarray[Any, Any], pose: SlamPose) -> np.ndarray[Any, Any]:
        cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        return points[:, :2] @ rotation.T + np.asarray((pose.x, pose.y))

    def _predict(self, odometry: SlamPose, gyro_z: float, timestamp: float) -> SlamPose:
        if self.pose is None or self._last_odometry is None:
            return odometry
        dx = odometry.x - self._last_odometry.x
        dy = odometry.y - self._last_odometry.y
        odom_dyaw = wrap_angle(odometry.yaw - self._last_odometry.yaw)
        dt = max(0.0, timestamp - (self._last_timestamp or timestamp))
        imu_dyaw = float(gyro_z) * dt
        fused_dyaw = 0.75 * odom_dyaw + 0.25 * imu_dyaw
        return SlamPose(
            self.pose.x + dx,
            self.pose.y + dy,
            wrap_angle(self.pose.yaw + fused_dyaw),
        )

    def _scan_match(self, local_points: np.ndarray[Any, Any], predicted: SlamPose) -> SlamPose:
        occupied = self.grid == 100
        if np.count_nonzero(occupied) < 20 or len(local_points) < 20:
            self.last_scan_match_score = None
            return predicted
        distance = distance_transform_edt(~occupied) * self.resolution
        sampled = local_points[:: max(1, len(local_points) // 180)]
        best_pose, best_score, best_objective = predicted, math.inf, math.inf
        # Evaluate the odometry/IMU prediction first. Corridor geometry often
        # gives many candidates the same zero residual; a motion prior avoids
        # systematically selecting the first (-3 deg, -12 cm, -12 cm) tie and
        # painting duplicated walls along the trajectory.
        for dyaw in np.deg2rad((0.0, -1.5, 1.5, -3.0, 3.0)):
            for dx in (0.0, -0.06, 0.06, -0.12, 0.12):
                for dy in (0.0, -0.06, 0.06, -0.12, 0.12):
                    candidate = SlamPose(
                        predicted.x + dx,
                        predicted.y + dy,
                        wrap_angle(predicted.yaw + float(dyaw)),
                    )
                    world = self._transform(sampled, candidate)
                    columns = ((world[:, 0] - self.origin_x) / self.resolution).astype(int)
                    rows = ((world[:, 1] - self.origin_y) / self.resolution).astype(int)
                    valid = (
                        (rows >= 0) & (rows < self.height) &
                        (columns >= 0) & (columns < self.width)
                    )
                    if np.count_nonzero(valid) < 12:
                        continue
                    residual = np.minimum(distance[rows[valid], columns[valid]], 0.5)
                    score = float(np.mean(residual))
                    objective = (
                        score
                        + 0.25 * math.hypot(dx, dy)
                        + 0.01 * abs(math.degrees(float(dyaw)))
                    )
                    if objective < best_objective - 1.0e-9:
                        best_pose, best_score, best_objective = candidate, score, objective
        self.last_scan_match_score = best_score if math.isfinite(best_score) else None
        # Reject a match that found no geometric support; retain odometry/IMU.
        return best_pose if best_score <= 0.28 else predicted

    def observe(
        self,
        local_points: np.ndarray[Any, Any],
        *,
        odometry: SlamPose,
        gyro_z: float,
        timestamp: float,
    ) -> SlamPose:
        predicted = self._predict(odometry, gyro_z, timestamp)
        pose = self._scan_match(local_points, predicted)
        self.pose = pose
        self._last_odometry = odometry
        self._last_timestamp = timestamp
        world_points = self._transform(local_points, pose)
        start = self.world_to_cell(pose.x, pose.y)
        if start is not None:
            free_cells: set[tuple[int, int]] = set()
            occupied_cells: set[tuple[int, int]] = set()
            for point in world_points:
                end = self.world_to_cell(float(point[0]), float(point[1]))
                if end is None:
                    continue
                ray = self._ray_cells(start, end)
                free_cells.update(ray[:-1])
                occupied_cells.add(ray[-1])
            # Endpoint evidence wins within one scan. Across scans, evidence
            # changes gradually so stale/dynamic hits clear without flicker.
            free_cells.difference_update(occupied_cells)
            for row, column in free_cells:
                self._observed[row, column] = True
                self._occupancy_evidence[row, column] = max(
                    -12,
                    int(self._occupancy_evidence[row, column]) - 1,
                )
            for row, column in occupied_cells:
                self._observed[row, column] = True
                self._occupancy_evidence[row, column] = min(
                    12,
                    int(self._occupancy_evidence[row, column]) + 2,
                )
            observed_rows, observed_columns = np.nonzero(self._observed)
            self.grid[observed_rows, observed_columns] = 0
            occupied = self._occupancy_evidence >= 2
            self.grid[occupied] = 100
            radius = int(math.ceil(self.footprint_radius / self.resolution))
            for row in range(start[0] - radius, start[0] + radius + 1):
                for column in range(start[1] - radius, start[1] + radius + 1):
                    if 0 <= row < self.height and 0 <= column < self.width:
                        if math.hypot(row - start[0], column - start[1]) * self.resolution <= self.footprint_radius:
                            self._observed[row, column] = True
                            self._occupancy_evidence[row, column] = -12
                            self.grid[row, column] = 0
        self.revision += 1
        return pose

    def observe_fastlio(
        self,
        registered_points: np.ndarray[Any, Any],
        *,
        pose: SlamPose,
    ) -> SlamPose:
        """Project a FAST-LIO2 pose and sensor-frame registered scan to 2-D."""

        self.pose = pose
        self._last_odometry = None
        self._last_timestamp = None
        self.last_scan_match_score = None
        local_points = np.asarray(registered_points, dtype=np.float64)
        if local_points.ndim != 2 or local_points.shape[1] < 2:
            raise ValueError("registered FAST-LIO cloud must have shape (N, >=2)")
        world_points = self._transform(local_points, pose)
        start = self.world_to_cell(pose.x, pose.y)
        if start is not None:
            free_heights: dict[tuple[int, int], float] = {}
            occupied_heights: dict[tuple[int, int], float] = {}
            for local_point, point in zip(local_points, world_points, strict=True):
                end = self.world_to_cell(float(point[0]), float(point[1]))
                if end is None:
                    continue
                ray = self._ray_cells(start, end)
                denominator = max(1, len(ray) - 1)
                endpoint_z = float(local_point[2]) if local_point.shape[0] >= 3 else 0.0
                for index, cell in enumerate(ray[:-1], start=1):
                    ray_z = endpoint_z * index / denominator
                    free_heights[cell] = min(free_heights.get(cell, math.inf), ray_z)
                occupied_heights[end] = min(
                    occupied_heights.get(end, math.inf), endpoint_z
                )
            for cell in occupied_heights:
                free_heights.pop(cell, None)
            for (row, column), ray_z in free_heights.items():
                self._observed[row, column] = True
                lowest_endpoint = float(self._lowest_endpoint_z[row, column])
                # A ray passing above a remembered endpoint may clear only
                # that upper height band, not the complete 2-D column.
                if (
                    math.isfinite(lowest_endpoint)
                    and ray_z > lowest_endpoint + 0.04
                ):
                    continue
                self._occupancy_evidence[row, column] = max(
                    -12, int(self._occupancy_evidence[row, column]) - 1
                )
                if self._occupancy_evidence[row, column] <= -2:
                    self._lowest_endpoint_z[row, column] = math.inf
            for (row, column), endpoint_z in occupied_heights.items():
                self._observed[row, column] = True
                self._occupancy_evidence[row, column] = min(
                    12, int(self._occupancy_evidence[row, column]) + 2
                )
                self._lowest_endpoint_z[row, column] = min(
                    float(self._lowest_endpoint_z[row, column]), endpoint_z
                )
            observed_rows, observed_columns = np.nonzero(self._observed)
            self.grid[observed_rows, observed_columns] = 0
            self.grid[self._occupancy_evidence >= 2] = 100
            radius = int(math.ceil(self.footprint_radius / self.resolution))
            for row in range(start[0] - radius, start[0] + radius + 1):
                for column in range(start[1] - radius, start[1] + radius + 1):
                    if (
                        0 <= row < self.height
                        and 0 <= column < self.width
                        and math.hypot(row - start[0], column - start[1]) * self.resolution
                        <= self.footprint_radius
                    ):
                        self._observed[row, column] = True
                        # Never erase a fresh physical endpoint merely because
                        # the estimated footprint overlaps its 2-D column.
                        if self._occupancy_evidence[row, column] < 2:
                            self._occupancy_evidence[row, column] = -12
                            self._lowest_endpoint_z[row, column] = math.inf
                            self.grid[row, column] = 0
        self.revision += 1
        return pose

    def frontier_path(
        self,
        *,
        goal_hint: tuple[float, float] | None = None,
        semantic_obstacles: tuple[tuple[float, float, float], ...] = (),
    ) -> list[tuple[float, float]] | None:
        """Return a safe frontier path, optionally biased by a ROS goal prior."""

        if self.pose is None:
            return None
        start = self.world_to_cell(self.pose.x, self.pose.y)
        if start is None:
            return None
        occupied = self.grid == 100
        if semantic_obstacles:
            occupied = occupied.copy()
            half_diagonal = self.resolution / math.sqrt(2.0)
            for center_x, center_y, radius_m in semantic_obstacles:
                radius = max(0.0, float(radius_m)) + half_diagonal
                min_column = max(
                    0, int(math.floor((center_x - radius - self.origin_x) / self.resolution))
                )
                max_column = min(
                    self.width - 1,
                    int(math.ceil((center_x + radius - self.origin_x) / self.resolution)),
                )
                min_row = max(
                    0, int(math.floor((center_y - radius - self.origin_y) / self.resolution))
                )
                max_row = min(
                    self.height - 1,
                    int(math.ceil((center_y + radius - self.origin_y) / self.resolution)),
                )
                for row in range(min_row, max_row + 1):
                    for column in range(min_column, max_column + 1):
                        if (
                            math.dist(
                                self.cell_to_world(row, column),
                                (center_x, center_y),
                            )
                            <= radius
                        ):
                            occupied[row, column] = True
        clearance = distance_transform_edt(~occupied) * self.resolution
        traversable = (self.grid == 0) & (clearance >= self.footprint_radius + 0.08)
        if not traversable[start]:
            traversable[start] = True
        queue: deque[tuple[int, int]] = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
        surrounding = tuple(
            (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc
        )
        frontiers: list[tuple[int, int]] = []
        while queue:
            cell = queue.popleft()
            row, column = cell
            if cell != start and any(
                0 <= row + dr < self.height
                and 0 <= column + dc < self.width
                and self.grid[row + dr, column + dc] == -1
                for dr, dc in surrounding
            ):
                frontiers.append(cell)
            for dr, dc in cardinal:
                neighbor = (row + dr, column + dc)
                if (
                    0 <= neighbor[0] < self.height
                    and 0 <= neighbor[1] < self.width
                    and traversable[neighbor]
                    and neighbor not in parent
                ):
                    parent[neighbor] = cell
                    queue.append(neighbor)
        if goal_hint is None:
            if not frontiers:
                return None
            goal = frontiers[0]
        else:
            # A coordinate prior targets the closest reachable observed-free
            # cell, even when the target neighborhood is already known and
            # therefore no longer qualifies as a frontier.
            candidates = [cell for cell in parent if cell != start]
            if not candidates:
                return None
            goal = min(
                candidates,
                key=lambda cell: math.dist(
                    self.cell_to_world(*cell),
                    goal_hint,
                ),
            )
        cells: list[tuple[int, int]] = []
        cursor: tuple[int, int] | None = goal
        while cursor is not None:
            cells.append(cursor)
            cursor = parent[cursor]
        cells.reverse()
        points = [self.cell_to_world(row, column) for row, column in cells]
        points[0] = (self.pose.x, self.pose.y)
        # Simplify only across a verified line of sight. Fixed-stride sampling
        # can turn a safe cardinal BFS bend into a chord through an inflated
        # obstacle, which the command boundary then correctly invalidates.
        simplified = [points[0]]
        index = 0
        maximum_skip = 5
        while index + 1 < len(points):
            selected = index + 1
            for candidate in range(
                min(len(points) - 1, index + maximum_skip),
                index,
                -1,
            ):
                if self._segment_is_traversable_with_clearance(
                    points[index],
                    points[candidate],
                    clearance=clearance,
                ):
                    selected = candidate
                    break
            simplified.append(points[selected])
            index = selected
        return simplified

    def _segment_is_traversable_with_clearance(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        clearance: np.ndarray[Any, Any],
        safety_margin: float = 0.08,
    ) -> bool:
        distance = math.dist(start_xy, end_xy)
        samples = max(2, int(math.ceil(distance / (0.5 * self.resolution))) + 1)
        required = self.footprint_radius + max(0.0, float(safety_margin))
        for fraction in np.linspace(0.0, 1.0, samples):
            x = start_xy[0] + fraction * (end_xy[0] - start_xy[0])
            y = start_xy[1] + fraction * (end_xy[1] - start_xy[1])
            cell = self.world_to_cell(x, y)
            if cell is None:
                return False
            if self.grid[cell] != 0 or clearance[cell] < required:
                return False
        return True

    def segment_is_traversable(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        safety_margin: float = 0.08,
    ) -> bool:
        """Validate a complete segment against the latest inflated local map."""

        occupied = self.grid == 100
        clearance = distance_transform_edt(~occupied) * self.resolution
        return self._segment_is_traversable_with_clearance(
            start_xy,
            end_xy,
            clearance=clearance,
            safety_margin=safety_margin,
        )

    def payload(self) -> dict[str, Any]:
        known = int(np.count_nonzero(self.grid >= 0))
        occupied = int(np.count_nonzero(self.grid == 100))
        return {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "known": known,
            "free": known - occupied,
            "occupied": occupied,
            "revision": self.revision,
            "pose": (
                {"x": self.pose.x, "y": self.pose.y, "yaw": self.pose.yaw}
                if self.pose is not None else None
            ),
            "scan_match_score_m": self.last_scan_match_score,
            "fusion": "mid360_raw_points+imu->dimos_fastlio2_native",
        }
