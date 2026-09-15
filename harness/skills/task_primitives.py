"""Backend-neutral primitives shared by object-search and person-follow skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import json
import math
from typing import Any, Iterable

import numpy as np

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.sensor_msgs.Image import Image


@dataclass(frozen=True)
class FrontierSelectionConfig:
    """Safety and ranking parameters for a known-free-side frontier goal."""

    min_frontier_perimeter_m: float = 0.5
    inward_offset_m: float = 0.55
    known_free_radius_m: float = 0.35
    candidate_search_radius_m: float = 1.25
    rejected_goal_radius_m: float = 0.75
    information_gain_weight: float = 0.08
    travel_cost_weight: float = 0.04
    min_goal_path_distance_m: float = 0.0
    max_goal_path_distance_m: float | None = None


@dataclass(frozen=True)
class KnownFreeFrontierGoal:
    """A planner-safe pose that faces unknown space without entering it."""

    x: float
    y: float
    yaw_radians: float
    frontier_cell_count: int
    adjacent_unknown_cell_count: int
    path_distance_m: float
    map_timestamp: float

    def pose_stamped(self, *, z: float = 0.0) -> PoseStamped:
        return PoseStamped(
            position=Vector3(self.x, self.y, float(z)),
            orientation=Quaternion.from_euler(
                Vector3(0.0, 0.0, self.yaw_radians)
            ),
            frame_id="world",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "yaw_degrees": round(math.degrees(self.yaw_radians), 2),
            "frontier_cell_count": self.frontier_cell_count,
            "adjacent_unknown_cell_count": self.adjacent_unknown_cell_count,
            "path_distance_m": round(self.path_distance_m, 3),
            "map_timestamp": self.map_timestamp,
            "goal_space": "known_free",
        }


@dataclass(frozen=True)
class FrontierExplorationResult:
    tool_ok: bool
    task_status: str
    completed: bool
    frontiers_attempted: int
    frontiers_reached: int
    known_cells_before: int
    known_cells_after: int
    elapsed_s: float
    termination_reason: str
    goals: tuple[dict[str, Any], ...]
    message: str
    map_acquisition: dict[str, Any] = field(default_factory=dict)

    @property
    def information_gain_cells(self) -> int:
        return max(0, self.known_cells_after - self.known_cells_before)

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["goals"] = [dict(item) for item in self.goals]
        payload["information_gain_cells"] = self.information_gain_cells
        payload["planner_goal_space"] = "known_free_only"
        return payload

    def agent_encode(self) -> list[dict[str, str]]:
        return [
            {
                "type": "text",
                "text": json.dumps(self.as_dict(), ensure_ascii=False),
            }
        ]

    def __str__(self) -> str:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )


def select_known_free_frontier(
    costmap: OccupancyGrid,
    *,
    robot_x: float,
    robot_y: float,
    robot_yaw: float | None = None,
    rejected_goals: Iterable[tuple[float, float]] = (),
    config: FrontierSelectionConfig = FrontierSelectionConfig(),
) -> KnownFreeFrontierGoal | None:
    """Select a reachable known-free pose on the robot side of a frontier.

    The input map must preserve the standard occupancy distinction
    ``unknown=-1, free=0, occupied=100``.  BFS is restricted to observed free
    cells.  A frontier is therefore used only as an information boundary; the
    returned planner goal is offset inward and surrounded by a known-free
    disk.
    """

    try:
        cells = np.asarray(costmap.grid, dtype=np.int8)
        resolution = float(costmap.resolution)
        origin_x = float(costmap.origin.position.x)
        origin_y = float(costmap.origin.position.y)
        orientation = costmap.origin.orientation
        quaternion_w = float(getattr(orientation, "w", 1.0))
        quaternion_x = float(getattr(orientation, "x", 0.0))
        quaternion_y = float(getattr(orientation, "y", 0.0))
        quaternion_z = float(getattr(orientation, "z", 0.0))
        origin_yaw = math.atan2(
            2.0
            * (
                quaternion_w * quaternion_z
                + quaternion_x * quaternion_y
            ),
            1.0
            - 2.0
            * (
                quaternion_y * quaternion_y
                + quaternion_z * quaternion_z
            ),
        )
        timestamp = float(costmap.ts)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if (
        cells.ndim != 2
        or cells.size == 0
        or not math.isfinite(resolution)
        or resolution <= 0.0
        or not all(
            math.isfinite(value)
            for value in (
                origin_x,
                origin_y,
                origin_yaw,
                float(robot_x),
                float(robot_y),
            )
        )
        or not np.any(cells == -1)
        or not np.any(cells == 0)
    ):
        return None

    cos_yaw, sin_yaw = math.cos(origin_yaw), math.sin(origin_yaw)

    def world_to_grid(x: float, y: float) -> tuple[int, int]:
        dx, dy = x - origin_x, y - origin_y
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        return (
            int(math.floor(local_y / resolution)),
            int(math.floor(local_x / resolution)),
        )

    def grid_to_world(row: float, column: float) -> tuple[float, float]:
        local_x = (column + 0.5) * resolution
        local_y = (row + 0.5) * resolution
        return (
            origin_x + cos_yaw * local_x - sin_yaw * local_y,
            origin_y + sin_yaw * local_x + cos_yaw * local_y,
        )

    height, width = cells.shape
    start_row, start_column = world_to_grid(float(robot_x), float(robot_y))
    if not (0 <= start_row < height and 0 <= start_column < width):
        return None
    if cells[start_row, start_column] != 0:
        nearby_free = np.argwhere(cells == 0)
        if nearby_free.size == 0:
            return None
        squared = (
            (nearby_free[:, 0] - start_row) ** 2
            + (nearby_free[:, 1] - start_column) ** 2
        )
        nearest_index = int(np.argmin(squared))
        if float(squared[nearest_index]) > 9.0:
            return None
        start_row, start_column = (
            int(nearby_free[nearest_index, 0]),
            int(nearby_free[nearest_index, 1]),
        )

    reachable = np.zeros((height, width), dtype=np.bool_)
    path_steps = np.full((height, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque([(start_row, start_column)])
    reachable[start_row, start_column] = True
    path_steps[start_row, start_column] = 0
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    surrounding = tuple(
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    )
    while queue:
        row, column = queue.popleft()
        for dr, dc in cardinal:
            neighbor_row, neighbor_column = row + dr, column + dc
            if (
                0 <= neighbor_row < height
                and 0 <= neighbor_column < width
                and not reachable[neighbor_row, neighbor_column]
                and cells[neighbor_row, neighbor_column] == 0
            ):
                reachable[neighbor_row, neighbor_column] = True
                path_steps[neighbor_row, neighbor_column] = (
                    path_steps[row, column] + 1
                )
                queue.append((neighbor_row, neighbor_column))

    frontier = np.zeros((height, width), dtype=np.bool_)
    for row, column in np.argwhere(reachable):
        if any(
            0 <= row + dr < height
            and 0 <= column + dc < width
            and cells[row + dr, column + dc] == -1
            for dr, dc in surrounding
        ):
            frontier[row, column] = True

    minimum_cells = max(
        1,
        int(math.ceil(config.min_frontier_perimeter_m / resolution)),
    )
    clearance_cells = max(
        1,
        int(math.ceil(config.known_free_radius_m / resolution)),
    )
    search_radius_cells = max(
        1,
        int(math.ceil(config.candidate_search_radius_m / resolution)),
    )
    inward_offset_cells = max(1.0, config.inward_offset_m / resolution)
    rejected = tuple(
        (float(x), float(y))
        for x, y in rejected_goals
        if math.isfinite(float(x)) and math.isfinite(float(y))
    )
    seen = np.zeros_like(frontier)
    ranked: list[tuple[float, KnownFreeFrontierGoal]] = []

    for seed_row, seed_column in np.argwhere(frontier):
        if seen[seed_row, seed_column]:
            continue
        cluster: list[tuple[int, int]] = []
        cluster_queue: deque[tuple[int, int]] = deque(
            [(int(seed_row), int(seed_column))]
        )
        seen[seed_row, seed_column] = True
        while cluster_queue:
            row, column = cluster_queue.popleft()
            cluster.append((row, column))
            for dr, dc in surrounding:
                neighbor_row, neighbor_column = row + dr, column + dc
                if (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_column < width
                    and frontier[neighbor_row, neighbor_column]
                    and not seen[neighbor_row, neighbor_column]
                ):
                    seen[neighbor_row, neighbor_column] = True
                    cluster_queue.append((neighbor_row, neighbor_column))
        if len(cluster) < minimum_cells:
            continue

        unknown_neighbors = {
            (row + dr, column + dc)
            for row, column in cluster
            for dr, dc in surrounding
            if (
                0 <= row + dr < height
                and 0 <= column + dc < width
                and cells[row + dr, column + dc] == -1
            )
        }
        if not unknown_neighbors:
            continue
        # A cold-start lidar map commonly produces one large, nearly closed
        # frontier ring. Its centroid is close to the robot and gives an
        # unstable outward normal, which can send the planner to the far side
        # of the ring. Anchor the candidate at the closest reachable part of
        # each cluster and derive the normal only from adjacent unknown cells.
        minimum_cluster_steps = min(
            int(path_steps[row, column]) for row, column in cluster
        )
        anchor_candidates = [
            cell
            for cell in cluster
            if int(path_steps[cell[0], cell[1]]) <= minimum_cluster_steps + 2
        ]

        # Prefer an anchor aligned with a quantized heading sector.  Raw gait
        # yaw made a symmetric cold-start ring jump between routes, while
        # ignoring heading entirely could deterministically select a 180°
        # U-turn and spend the whole bounded leg rotating in place.
        quantized_heading: float | None = None
        if robot_yaw is not None and math.isfinite(float(robot_yaw)):
            heading_sector = math.pi / 4.0
            quantized_heading = (
                round(float(robot_yaw) / heading_sector) * heading_sector
            )

        def anchor_rank(cell: tuple[int, int]) -> tuple[float, int, int, int]:
            row, column = cell
            if quantized_heading is None:
                angle_error = 0.0
            else:
                anchor_x, anchor_y = grid_to_world(row, column)
                bearing = math.atan2(
                    anchor_y - float(robot_y),
                    anchor_x - float(robot_x),
                )
                angle_error = abs(
                    math.atan2(
                        math.sin(bearing - quantized_heading),
                        math.cos(bearing - quantized_heading),
                    )
                )
            return (angle_error, int(path_steps[row, column]), row, column)

        anchor_row, anchor_column = min(
            anchor_candidates,
            key=anchor_rank,
        )
        local_unknown_neighbors = {
            (anchor_row + dr, anchor_column + dc)
            for dr, dc in surrounding
            if (
                0 <= anchor_row + dr < height
                and 0 <= anchor_column + dc < width
                and cells[anchor_row + dr, anchor_column + dc] == -1
            )
        }
        if not local_unknown_neighbors:
            continue
        frontier_row = float(anchor_row)
        frontier_column = float(anchor_column)
        unknown_row = float(
            np.mean([cell[0] for cell in local_unknown_neighbors])
        )
        unknown_column = float(
            np.mean([cell[1] for cell in local_unknown_neighbors])
        )
        inward_row = frontier_row - unknown_row
        inward_column = frontier_column - unknown_column
        inward_norm = math.hypot(inward_row, inward_column)
        if inward_norm < 1e-6:
            continue
        ideal_row = frontier_row + inward_offset_cells * inward_row / inward_norm
        ideal_column = (
            frontier_column
            + inward_offset_cells * inward_column / inward_norm
        )

        row_start = max(0, int(math.floor(ideal_row)) - search_radius_cells)
        row_stop = min(height, int(math.ceil(ideal_row)) + search_radius_cells + 1)
        column_start = max(
            0,
            int(math.floor(ideal_column)) - search_radius_cells,
        )
        column_stop = min(
            width,
            int(math.ceil(ideal_column)) + search_radius_cells + 1,
        )
        best_candidate: tuple[float, int, int] | None = None
        for row in range(row_start, row_stop):
            for column in range(column_start, column_stop):
                if not reachable[row, column]:
                    continue
                radius_view = cells[
                    max(0, row - clearance_cells) : min(
                        height, row + clearance_cells + 1
                    ),
                    max(0, column - clearance_cells) : min(
                        width, column + clearance_cells + 1
                    ),
                ]
                if radius_view.shape != (
                    2 * clearance_cells + 1,
                    2 * clearance_cells + 1,
                ):
                    continue
                rr, cc = np.ogrid[
                    -clearance_cells : clearance_cells + 1,
                    -clearance_cells : clearance_cells + 1,
                ]
                disk = rr * rr + cc * cc <= clearance_cells * clearance_cells
                if np.any(radius_view[disk] != 0):
                    continue
                candidate_x, candidate_y = grid_to_world(row, column)
                candidate_path_distance = (
                    float(path_steps[row, column]) * resolution
                )
                if candidate_path_distance < config.min_goal_path_distance_m:
                    continue
                if (
                    config.max_goal_path_distance_m is not None
                    and candidate_path_distance > config.max_goal_path_distance_m
                ):
                    continue
                if any(
                    math.hypot(candidate_x - old_x, candidate_y - old_y)
                    < config.rejected_goal_radius_m
                    for old_x, old_y in rejected
                ):
                    continue
                deviation = math.hypot(row - ideal_row, column - ideal_column)
                candidate_score = deviation + 0.02 * float(
                    path_steps[row, column]
                )
                if best_candidate is None or candidate_score < best_candidate[0]:
                    best_candidate = (candidate_score, row, column)
        if best_candidate is None and config.max_goal_path_distance_m is not None:
            bounded = np.argwhere(
                reachable
                & (path_steps >= 0)
                & (
                    path_steps
                    >= int(
                        math.ceil(
                            config.min_goal_path_distance_m / resolution
                        )
                    )
                )
                & (
                    path_steps
                    <= int(
                        math.floor(
                            config.max_goal_path_distance_m / resolution
                        )
                    )
                )
            )
            for row_value, column_value in bounded:
                row, column = int(row_value), int(column_value)
                radius_view = cells[
                    max(0, row - clearance_cells) : min(
                        height, row + clearance_cells + 1
                    ),
                    max(0, column - clearance_cells) : min(
                        width, column + clearance_cells + 1
                    ),
                ]
                if radius_view.shape != (
                    2 * clearance_cells + 1,
                    2 * clearance_cells + 1,
                ):
                    continue
                rr, cc = np.ogrid[
                    -clearance_cells : clearance_cells + 1,
                    -clearance_cells : clearance_cells + 1,
                ]
                disk = rr * rr + cc * cc <= clearance_cells * clearance_cells
                if np.any(radius_view[disk] != 0):
                    continue
                candidate_x, candidate_y = grid_to_world(row, column)
                if any(
                    math.hypot(candidate_x - old_x, candidate_y - old_y)
                    < config.rejected_goal_radius_m
                    for old_x, old_y in rejected
                ):
                    continue
                # If the true frontier-side pose is too far for one robust
                # terminal leg, choose the furthest safe point along the same
                # reachable direction. A later Step can select the next leg.
                deviation = math.hypot(row - ideal_row, column - ideal_column)
                candidate_score = deviation - 0.02 * float(
                    path_steps[row, column]
                )
                if best_candidate is None or candidate_score < best_candidate[0]:
                    best_candidate = (candidate_score, row, column)
        if best_candidate is None:
            continue

        deviation, candidate_row, candidate_column = best_candidate
        candidate_x, candidate_y = grid_to_world(
            candidate_row,
            candidate_column,
        )
        path_distance = float(path_steps[candidate_row, candidate_column]) * resolution
        travel_x = candidate_x - float(robot_x)
        travel_y = candidate_y - float(robot_y)
        # The current A* stack requires final yaw as well as position, while
        # its local follower does not perform a separate in-place terminal
        # rotation. Face along the admitted route so the global planner can
        # produce its real goal-reached evidence. The lidar remains 360° and
        # the candidate itself is still selected from the unknown boundary.
        if math.hypot(travel_x, travel_y) > resolution:
            route_yaw = math.atan2(travel_y, travel_x)
        else:
            unknown_x, unknown_y = grid_to_world(unknown_row, unknown_column)
            route_yaw = math.atan2(
                unknown_y - candidate_y,
                unknown_x - candidate_x,
            )
        goal = KnownFreeFrontierGoal(
            x=candidate_x,
            y=candidate_y,
            yaw_radians=route_yaw,
            frontier_cell_count=len(cluster),
            adjacent_unknown_cell_count=len(unknown_neighbors),
            path_distance_m=path_distance,
            map_timestamp=timestamp,
        )
        score = (
            deviation
            + config.travel_cost_weight * path_distance
            - config.information_gain_weight
            * math.log1p(len(unknown_neighbors))
        )
        ranked.append((score, goal))

    return min(ranked, key=lambda item: item[0])[1] if ranked else None


@dataclass(frozen=True)
class TrackingMeasurement:
    bbox: tuple[float, float, float, float]
    distance_m: float
    bearing_radians: float
    frame_timestamp: float
    valid_depth_points: int


@dataclass
class FollowDistanceAcquisition:
    """Latch after consecutive RGB-D samples enter the requested follow band."""

    target_distance_m: float
    tolerance_m: float
    required_samples: int = 3
    consecutive_samples: int = 0
    acquired_at: float | None = None

    def observe(self, measurement: TrackingMeasurement, *, now: float) -> bool:
        if self.acquired_at is not None:
            return True
        if (
            math.isfinite(measurement.distance_m)
            and abs(measurement.distance_m - self.target_distance_m)
            <= self.tolerance_m
        ):
            self.consecutive_samples += 1
        else:
            self.consecutive_samples = 0
        if self.consecutive_samples >= self.required_samples:
            self.acquired_at = float(now)
        return self.acquired_at is not None


@dataclass(frozen=True)
class FollowControlConfig:
    max_planar_speed_mps: float = 0.18
    max_yaw_rate_rps: float = 0.45
    warning_planar_speed_mps: float = 0.10
    warning_yaw_rate_rps: float = 0.25
    distance_deadband_m: float = 0.20
    distance_gain: float = 0.35
    yaw_gain: float = 1.4
    rotate_only_bearing_degrees: float = 20.0
    slow_bearing_degrees: float = 10.0
    angled_planar_speed_mps: float = 0.08


def compute_follow_twist(
    measurement: TrackingMeasurement,
    *,
    follow_distance_m: float,
    risk: str,
    config: FollowControlConfig = FollowControlConfig(),
) -> Twist:
    """Return a forward-only visual-servo command under the lidar limits."""

    if risk not in {"clear", "warning"}:
        return Twist()
    yaw_limit = (
        config.warning_yaw_rate_rps
        if risk == "warning"
        else config.max_yaw_rate_rps
    )
    speed_limit = (
        config.warning_planar_speed_mps
        if risk == "warning"
        else config.max_planar_speed_mps
    )
    bearing = float(measurement.bearing_radians)
    yaw = max(-yaw_limit, min(yaw_limit, config.yaw_gain * bearing))
    distance_error = float(measurement.distance_m) - float(follow_distance_m)
    speed = 0.0
    if distance_error > config.distance_deadband_m:
        speed = min(
            speed_limit,
            config.distance_gain * distance_error,
        )
    bearing_degrees = abs(math.degrees(bearing))
    if bearing_degrees >= config.rotate_only_bearing_degrees:
        speed = 0.0
    elif bearing_degrees >= config.slow_bearing_degrees:
        speed = min(speed, config.angled_planar_speed_mps)
    return Twist(
        linear=Vector3(max(0.0, speed), 0.0, 0.0),
        angular=Vector3(0.0, 0.0, yaw),
    )


def bbox_is_continuous(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> bool:
    """Reject implausible tracker jumps that could switch to another target."""

    def geometry(
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float] | None:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        width, height = x2 - x1, y2 - y1
        if (
            not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or width < 8.0
            or height < 12.0
            or x1 < 0.0
            or y1 < 0.0
            or x2 > image_width
            or y2 > image_height
        ):
            return None
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2), width * height)

    if image_width <= 0 or image_height <= 0:
        return False
    before, after = geometry(previous), geometry(current)
    if before is None or after is None:
        return False
    area_ratio = after[2] / before[2]
    jump = math.hypot(
        (after[0] - before[0]) / image_width,
        (after[1] - before[1]) / image_height,
    )
    return 0.4 <= area_ratio <= 2.5 and jump <= 0.35


def stable_person_tracking_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Prefer the torso over articulated limbs for appearance tracking."""

    x1, y1, x2, y2 = (float(value) for value in bbox)
    width, height = x2 - x1, y2 - y1
    if width < 24.0 or height < 48.0:
        return bbox
    return (
        x1 + 0.12 * width,
        y1 + 0.05 * height,
        x1 + 0.88 * width,
        y1 + 0.68 * height,
    )


class CsrtTargetTracker:
    """Task-local CSRT tracker initialized from exactly one VLM frame."""

    def __init__(self) -> None:
        self._tracker: Any | None = None
        self._bbox: tuple[float, float, float, float] | None = None

    @staticmethod
    def _frame(image: Image) -> np.ndarray[Any, np.dtype[Any]]:
        frame = np.asarray(image.to_opencv())
        if frame.ndim not in {2, 3} or not frame.size:
            raise ValueError("invalid RGB frame")
        return np.ascontiguousarray(frame)

    def initialize(
        self,
        image: Image,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        try:
            import cv2

            create = getattr(
                getattr(cv2, "legacy", None), "TrackerCSRT_create", None
            ) or getattr(cv2, "TrackerCSRT_create", None)
            frame = self._frame(image)
            if not callable(create) or not bbox_is_continuous(
                bbox,
                bbox,
                image_width=int(frame.shape[1]),
                image_height=int(frame.shape[0]),
            ):
                return False
            x1, y1, x2, y2 = bbox
            tracker = create()
            accepted = tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            if accepted is False:
                return False
        except (AttributeError, TypeError, ValueError):
            return False
        self._tracker, self._bbox = tracker, bbox
        return True

    def update(self, image: Image) -> tuple[float, float, float, float] | None:
        if self._tracker is None or self._bbox is None:
            return None
        try:
            frame = self._frame(image)
            tracked, raw = self._tracker.update(frame)
            if not tracked:
                return None
            x, y, width, height = (float(value) for value in raw)
            image_height, image_width = frame.shape[:2]
            x2, y2 = x + width, y + height
            if (
                x < -2.0
                or y < -2.0
                or x2 > image_width + 2.0
                or y2 > image_height + 2.0
            ):
                return None
            bbox = (
                max(0.0, x),
                max(0.0, y),
                min(float(image_width), x2),
                min(float(image_height), y2),
            )
            if not bbox_is_continuous(
                self._bbox,
                bbox,
                image_width=int(image_width),
                image_height=int(image_height),
            ):
                return None
        except (AttributeError, TypeError, ValueError):
            return None
        self._bbox = bbox
        return bbox


@dataclass(frozen=True)
class PersonFollowResult:
    tool_ok: bool
    task_status: str
    completed: bool
    requested_follow_duration_s: float
    verified_tracking_duration_s: float
    elapsed_s: float
    requested_follow_distance_m: float
    target_distance_m: float | None
    target_bearing_degrees: float | None
    tracked_frames: int
    tracking_coverage: float
    max_tracking_gap_s: float
    stop_command_publish_latency_ms: float
    physical_stop_latency_ms: float | None
    stop_command_completed_at: float | None
    stationary_confirmed_at: float | None
    verification_frame_timestamp: float | None
    termination_reason: str
    message: str

    @property
    def planner_goal_reached(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["planner_goal_reached"] = False
        return payload

    def agent_encode(self) -> list[dict[str, str]]:
        return [{"type": "text", "text": json.dumps(self.as_dict(), ensure_ascii=False)}]

    def __str__(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))
