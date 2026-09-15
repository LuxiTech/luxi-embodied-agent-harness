"""Authored warehouse semantic map used for bounded single-robot route planning.

The map is an explicit deployment input: it contains static fixtures and named
goal coordinates, but no dynamic actors or simulator-only scorer state.  The
same JSON-equivalent contract can therefore be installed on the robot computer.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable


GO2_WAREHOUSE_BOUNDS = (-7.0, 7.0, -5.0, 5.0)
GO2_WAREHOUSE_RESOLUTION_M = 0.10
GO2_WAREHOUSE_ROUTE_CLEARANCE_M = 0.40

# name, centre x/y/z, half-size x/y/z
GO2_WAREHOUSE_OBSTACLES = (
    ("wall_north", 0.0, 5.0, 1.1, 7.1, 0.08, 1.1),
    ("wall_south", 0.0, -5.0, 1.1, 7.1, 0.08, 1.1),
    ("wall_east", 7.0, 0.0, 1.1, 0.08, 5.1, 1.1),
    ("wall_west", -7.0, 0.0, 1.1, 0.08, 5.1, 1.1),
    ("shelf_north_west", -3.6, 1.55, 0.75, 1.25, 0.38, 0.75),
    ("shelf_north_center", -0.2, 1.55, 0.75, 1.25, 0.38, 0.75),
    ("shelf_north_east", 3.2, 1.55, 0.75, 1.25, 0.38, 0.75),
    ("shelf_south_west", -3.0, -1.55, 0.75, 1.25, 0.38, 0.75),
    ("shelf_south_center", 0.4, -1.55, 0.75, 1.25, 0.38, 0.75),
    ("shelf_south_east", 3.8, -1.55, 0.75, 1.25, 0.38, 0.75),
    ("packing_island", 0.0, 4.0, 0.55, 0.75, 0.45, 0.55),
    ("crate_stack_west", -5.0, 0.0, 0.60, 0.55, 0.55, 0.60),
)

GO2_SEMANTIC_TARGETS = {
    "red-cube": (4.7, 3.65),
    "blue-ball": (-2.4, -3.65),
    "water-bottle": (4.7, -3.65),
}

# Planar physical radii from the deployed semantic-map contract. The cube uses
# its half-diagonal; the sphere and bottle use their authored radial extent.
# Route clearance is added below, just as it is for every static fixture.
GO2_SEMANTIC_TARGET_FOOTPRINT_RADII_M = {
    "red-cube": math.hypot(0.18, 0.18),
    "blue-ball": 0.22,
    "water-bottle": 0.08,
}


@dataclass(frozen=True)
class SemanticRoute:
    points: tuple[tuple[float, float], ...]
    cost_m: float


class Go2WarehouseSemanticMap:
    """Static footprint-inflated warehouse grid with deterministic A*."""

    def __init__(
        self,
        *,
        resolution_m: float = GO2_WAREHOUSE_RESOLUTION_M,
        clearance_m: float = GO2_WAREHOUSE_ROUTE_CLEARANCE_M,
    ) -> None:
        if resolution_m <= 0.0 or clearance_m < 0.0:
            raise ValueError("semantic map resolution/clearance is invalid")
        self.resolution_m = float(resolution_m)
        self.clearance_m = float(clearance_m)
        self.min_x, self.max_x, self.min_y, self.max_y = GO2_WAREHOUSE_BOUNDS
        self.width = int(round((self.max_x - self.min_x) / self.resolution_m)) + 1
        self.height = int(round((self.max_y - self.min_y) / self.resolution_m)) + 1
        self._blocked: set[tuple[int, int]] = set()
        for _name, x, y, _z, half_x, half_y, _half_z in GO2_WAREHOUSE_OBSTACLES:
            self._block_rectangle(
                x - half_x - self.clearance_m,
                x + half_x + self.clearance_m,
                y - half_y - self.clearance_m,
                y + half_y + self.clearance_m,
            )
        for name, (x, y) in GO2_SEMANTIC_TARGETS.items():
            self._block_circle(
                (x, y),
                GO2_SEMANTIC_TARGET_FOOTPRINT_RADII_M[name] + self.clearance_m,
            )

    def _block_rectangle(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> None:
        min_column = max(0, int(math.floor((min_x - self.min_x) / self.resolution_m)))
        max_column = min(
            self.width - 1,
            int(math.ceil((max_x - self.min_x) / self.resolution_m)),
        )
        min_row = max(0, int(math.floor((min_y - self.min_y) / self.resolution_m)))
        max_row = min(
            self.height - 1,
            int(math.ceil((max_y - self.min_y) / self.resolution_m)),
        )
        for row in range(min_row, max_row + 1):
            for column in range(min_column, max_column + 1):
                self._blocked.add((row, column))

    def _block_circle(
        self,
        center_xy: tuple[float, float],
        radius_m: float,
    ) -> None:
        """Block every grid cell whose square intersects a circular keepout."""

        center_x, center_y = center_xy
        cell_half_diagonal = self.resolution_m / math.sqrt(2.0)
        conservative_radius = float(radius_m) + cell_half_diagonal
        min_cell = self.world_to_cell(
            (center_x - conservative_radius, center_y - conservative_radius)
        )
        max_cell = self.world_to_cell(
            (center_x + conservative_radius, center_y + conservative_radius)
        )
        if min_cell is None or max_cell is None:
            raise ValueError("semantic target keepout falls outside warehouse bounds")
        for row in range(min_cell[0], max_cell[0] + 1):
            for column in range(min_cell[1], max_cell[1] + 1):
                if (
                    math.dist(self.cell_to_world((row, column)), center_xy)
                    <= conservative_radius
                ):
                    self._blocked.add((row, column))

    def target_keepout_radius(self, name: str) -> float:
        """Return the configured physical radius plus route clearance."""

        return GO2_SEMANTIC_TARGET_FOOTPRINT_RADII_M[name] + self.clearance_m

    def target_obstacle_regions(self) -> tuple[tuple[float, float, float], ...]:
        """Physical target regions for overlay in the robot-local planner."""

        return tuple(
            (*GO2_SEMANTIC_TARGETS[name], radius)
            for name, radius in GO2_SEMANTIC_TARGET_FOOTPRINT_RADII_M.items()
        )

    def world_to_cell(self, point: tuple[float, float]) -> tuple[int, int] | None:
        x, y = point
        column = int(round((float(x) - self.min_x) / self.resolution_m))
        row = int(round((float(y) - self.min_y) / self.resolution_m))
        if not (0 <= row < self.height and 0 <= column < self.width):
            return None
        return row, column

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, column = cell
        return (
            self.min_x + column * self.resolution_m,
            self.min_y + row * self.resolution_m,
        )

    def _nearest_free(self, point: tuple[float, float]) -> tuple[int, int] | None:
        origin = self.world_to_cell(point)
        if origin is None:
            return None
        if origin not in self._blocked:
            return origin
        row, column = origin
        max_radius = max(self.width, self.height)
        for radius in range(1, max_radius):
            candidates: list[tuple[float, int, int]] = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    candidate = row + dr, column + dc
                    if (
                        0 <= candidate[0] < self.height
                        and 0 <= candidate[1] < self.width
                        and candidate not in self._blocked
                    ):
                        world = self.cell_to_world(candidate)
                        candidates.append((math.dist(point, world), *candidate))
            if candidates:
                _distance, selected_row, selected_column = min(candidates)
                return selected_row, selected_column
        return None

    def route(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> SemanticRoute | None:
        start = self._nearest_free(start_xy)
        goal = self._nearest_free(goal_xy)
        if start is None or goal is None:
            return None
        if start == goal:
            return SemanticRoute((self.cell_to_world(start),), 0.0)

        cardinal_cost = self.resolution_m
        diagonal_cost = math.sqrt(2.0) * self.resolution_m
        neighbors = (
            (-1, 0, cardinal_cost),
            (1, 0, cardinal_cost),
            (0, -1, cardinal_cost),
            (0, 1, cardinal_cost),
            (-1, -1, diagonal_cost),
            (-1, 1, diagonal_cost),
            (1, -1, diagonal_cost),
            (1, 1, diagonal_cost),
        )
        queue: list[tuple[float, float, int, int]] = [(0.0, 0.0, *start)]
        costs = {start: 0.0}
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue:
            _score, current_cost, row, column = heapq.heappop(queue)
            current = row, column
            if current_cost > costs.get(current, math.inf) + 1.0e-9:
                continue
            if current == goal:
                break
            for dr, dc, step_cost in neighbors:
                candidate = row + dr, column + dc
                if not (
                    0 <= candidate[0] < self.height
                    and 0 <= candidate[1] < self.width
                ) or candidate in self._blocked:
                    continue
                # A diagonal may not cut through the corner of an inflated
                # fixture even when its destination cell itself is free.
                if dr and dc and (
                    (row + dr, column) in self._blocked
                    or (row, column + dc) in self._blocked
                ):
                    continue
                candidate_cost = current_cost + step_cost
                if candidate_cost + 1.0e-9 >= costs.get(candidate, math.inf):
                    continue
                costs[candidate] = candidate_cost
                parents[candidate] = current
                heuristic = math.dist(candidate, goal) * self.resolution_m
                heapq.heappush(
                    queue,
                    (candidate_cost + heuristic, candidate_cost, *candidate),
                )
        if goal not in parents:
            return None
        cells: list[tuple[int, int]] = []
        cursor: tuple[int, int] | None = goal
        while cursor is not None:
            cells.append(cursor)
            cursor = parents[cursor]
        cells.reverse()
        return SemanticRoute(
            tuple(self.cell_to_world(cell) for cell in cells),
            costs[goal],
        )

    def route_cost(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> float:
        route = self.route(start_xy, goal_xy)
        return route.cost_m if route is not None else math.inf

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "map_id": "go2-warehouse-semantic-v2",
            "bounds": list(GO2_WAREHOUSE_BOUNDS),
            "resolution_m": self.resolution_m,
            "clearance_m": self.clearance_m,
            "static_obstacles": [item[0] for item in GO2_WAREHOUSE_OBSTACLES],
            "semantic_targets": {
                name: {
                    "position": list(position),
                    "physical_radius_m": GO2_SEMANTIC_TARGET_FOOTPRINT_RADII_M[name],
                    "route_keepout_radius_m": self.target_keepout_radius(name),
                }
                for name, position in GO2_SEMANTIC_TARGETS.items()
            },
        }
