"""Deterministic local apartment used for Isaac object-search/follow testing."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ApartmentBox:
    name: str
    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]
    collision: bool = True


@dataclass(frozen=True)
class PersonGaitPose:
    """Backend-neutral joint angles for the procedural walking actor."""

    left_arm_swing_deg: float
    right_arm_swing_deg: float
    left_thigh_swing_deg: float
    right_thigh_swing_deg: float
    left_knee_bend_deg: float
    right_knee_bend_deg: float


@dataclass(frozen=True)
class PersonRouteState:
    position: tuple[float, float, float]
    heading_radians: float
    moving: bool


ROBOT_START = (-4.8, 0.0, 0.80)
# The character root pose gives the RGB-D controller an initial distance inside
# the 1.5 +/- 0.35 m verification band, while leaving enough catch-up space for
# the moving-person follow demo.
PERSON_START = (-2.7, 0.0, 0.0)
BOTTLE_SUPPORT_NAME = "LivingBlueTable"
# Place the bottle in the blue table's west-front grasp zone as viewed from
# the robot's south-side approach.  The edge clearances leave room for a future
# wrist pre-grasp pose without putting the dynamic bottle close to falling.
WATER_BOTTLE_POSITION = (-5.60, 2.42, 1.07)
# Keep a deterministic catch-up margin below the lidar-warning follow limit
# (0.10 m/s). The person still moves continuously once the robot starts.
PERSON_SPEED_MPS = 0.06
PERSON_GAIT_PERIOD_S = 1.1
PERSON_TURN_RATE_RPS = math.radians(90.0)
PERSON_WALK_DISTANCE_M = 0.60
PERSON_CONTROL_RADIUS_M = 0.28
PERSON_MANUAL_SPEED_MPS = 0.75
PERSON_MANUAL_TURN_RATE_RPS = math.radians(120.0)
PERSON_ROUTE = (
    PERSON_START,
    (PERSON_START[0] + PERSON_WALK_DISTANCE_M, PERSON_START[1], PERSON_START[2]),
)


def person_position_is_walkable(
    position: tuple[float, float, float],
    *,
    radius_m: float = PERSON_CONTROL_RADIUS_M,
) -> bool:
    """Check the operator actor's circular footprint against authored obstacles."""

    x, y = float(position[0]), float(position[1])
    radius = max(0.0, float(radius_m))
    if not all(math.isfinite(value) for value in (x, y, radius)):
        return False
    for box in APARTMENT_BOXES:
        if not box.collision:
            continue
        half_x = box.size[0] / 2.0 + radius
        half_y = box.size[1] / 2.0 + radius
        if (
            abs(x - box.centre[0]) <= half_x
            and abs(y - box.centre[1]) <= half_y
        ):
            return False
    return True


def person_swept_position(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    radius_m: float = PERSON_CONTROL_RADIUS_M,
    sample_step_m: float = 0.04,
) -> tuple[tuple[float, float, float], bool]:
    """Return the furthest collision-free point on a planar movement segment."""

    distance = math.dist(start[:2], end[:2])
    if not math.isfinite(distance) or not person_position_is_walkable(
        start, radius_m=radius_m
    ):
        return start, True
    steps = max(1, math.ceil(distance / max(0.01, float(sample_step_m))))
    safe = start
    for index in range(1, steps + 1):
        ratio = index / steps
        candidate = (
            float(start[0] + (end[0] - start[0]) * ratio),
            float(start[1] + (end[1] - start[1]) * ratio),
            float(start[2] + (end[2] - start[2]) * ratio),
        )
        if not person_position_is_walkable(candidate, radius_m=radius_m):
            return safe, True
        safe = candidate
    return safe, False


def person_motion_should_start(
    command: tuple[float, float, float, float],
) -> bool:
    """Return whether a base command should start the moving-person route.

    A yaw-only visual alignment must not let the actor walk away before G1 has
    begun following. The actor keeps moving after this one-way trigger fires.
    """

    return any(abs(float(value)) > 1e-4 for value in command[:2])


def person_gait_pose(elapsed_s: float, *, moving: bool) -> PersonGaitPose:
    """Return a small natural walk cycle, or a neutral standing pose."""

    if not moving:
        return PersonGaitPose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    phase = 2.0 * math.pi * max(0.0, float(elapsed_s)) / PERSON_GAIT_PERIOD_S
    stride = math.sin(phase)
    left_leg = 18.0 * stride
    right_leg = -left_leg
    # Arms counter-swing against the opposite leg. Knees bend only during
    # each leg's recovery half-cycle, avoiding an unnatural straight-leg glide.
    return PersonGaitPose(
        left_arm_swing_deg=-14.0 * stride,
        right_arm_swing_deg=14.0 * stride,
        left_thigh_swing_deg=left_leg,
        right_thigh_swing_deg=right_leg,
        left_knee_bend_deg=14.0 * max(0.0, -stride),
        right_knee_bend_deg=14.0 * max(0.0, stride),
    )


def slew_person_heading(
    current_radians: float,
    target_radians: float,
    *,
    max_delta_radians: float,
) -> float:
    """Move toward a route heading through the shortest wrapped angle."""

    current = float(current_radians)
    target = float(target_radians)
    limit = max(0.0, float(max_delta_radians))
    delta = math.atan2(math.sin(target - current), math.cos(target - current))
    applied = max(-limit, min(limit, delta))
    return math.atan2(math.sin(current + applied), math.cos(current + applied))


def _wall(
    name: str,
    centre: tuple[float, float, float],
    size: tuple[float, float, float],
) -> ApartmentBox:
    return ApartmentBox(name, centre, size, (0.73, 0.75, 0.78))


def _furniture(
    name: str,
    centre: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> ApartmentBox:
    return ApartmentBox(name, centre, size, color)


# A 14 m x 10 m apartment. The x=0 partition has a 2.2 m central doorway.
# The straight start-to-person corridor remains wider than the 0.60 m
# rotation diameter after obstacle inflation.
APARTMENT_BOXES = (
    _wall("WestWall", (-7.0, 0.0, 1.25), (0.18, 10.0, 2.50)),
    _wall("EastWall", (7.0, 0.0, 1.25), (0.18, 10.0, 2.50)),
    _wall("NorthWall", (0.0, 5.0, 1.25), (14.0, 0.18, 2.50)),
    _wall("SouthWall", (0.0, -5.0, 1.25), (14.0, 0.18, 2.50)),
    _wall("PartitionNorth", (0.0, 3.05, 1.25), (0.18, 3.90, 2.50)),
    _wall("PartitionSouth", (0.0, -3.05, 1.25), (0.18, 3.90, 2.50)),
    _furniture(
        "LivingBlueTable", (-4.5, 2.8, 0.48), (2.4, 0.85, 0.96), (0.18, 0.36, 0.52)
    ),
    _furniture(
        "LivingCoffeeTable", (-3.5, 1.25, 0.28), (1.25, 0.72, 0.56), (0.48, 0.29, 0.16)
    ),
    _furniture(
        "LivingMediaUnit", (-5.9, -2.75, 0.42), (1.55, 0.48, 0.84), (0.20, 0.21, 0.23)
    ),
    _furniture(
        "LivingBookshelf", (-6.35, 3.65, 1.05), (0.46, 1.65, 2.10), (0.36, 0.22, 0.12)
    ),
    _furniture(
        "KitchenIsland", (3.35, -1.55, 0.39), (1.65, 0.82, 0.78), (0.70, 0.70, 0.66)
    ),
    _furniture(
        "KitchenBackCounter",
        (5.85, -3.85, 0.46),
        (2.05, 0.66, 0.92),
        (0.42, 0.45, 0.48),
    ),
    _furniture(
        "DiningTable", (3.75, 2.45, 0.38), (1.65, 1.05, 0.76), (0.47, 0.28, 0.14)
    ),
    _furniture(
        "DiningChairNorth", (3.75, 3.35, 0.45), (0.55, 0.55, 0.90), (0.19, 0.22, 0.25)
    ),
    _furniture(
        "DiningChairSouth", (3.75, 1.55, 0.45), (0.55, 0.55, 0.90), (0.19, 0.22, 0.25)
    ),
    _furniture(
        "KitchenTallUnit", (6.25, 2.95, 1.05), (0.72, 1.35, 2.10), (0.58, 0.60, 0.62)
    ),
)


def person_route_state(elapsed_s: float) -> PersonRouteState:
    """Walk the authored distance once, then remain stopped at its end."""

    elapsed = max(0.0, float(elapsed_s))
    segments: list[
        tuple[tuple[float, float, float], tuple[float, float, float], float]
    ] = []
    route_length = 0.0
    for start, end in zip(PERSON_ROUTE, PERSON_ROUTE[1:]):
        length = math.dist(start, end)
        segments.append((start, end, length))
        route_length += length
    if route_length <= 0.0:
        return PersonRouteState(PERSON_START, 0.0, False)
    distance = min(elapsed * PERSON_SPEED_MPS, route_length)
    for start, end, length in segments:
        if distance <= length:
            ratio = 0.0 if length == 0.0 else distance / length
            position = tuple(
                float(a + (b - a) * ratio) for a, b in zip(start, end, strict=True)
            )
            heading = (
                0.0
                if length == 0.0
                else math.atan2(end[1] - start[1], end[0] - start[0])
            )
            return PersonRouteState(
                position,
                heading,
                distance < route_length - 1e-9,
            )
        distance -= length
    final_start, final_end, _ = segments[-1]
    heading = math.atan2(
        final_end[1] - final_start[1],
        final_end[0] - final_start[0],
    )
    return PersonRouteState(PERSON_ROUTE[-1], heading, False)


def person_position(elapsed_s: float) -> tuple[float, float, float]:
    """Return the route position retained for existing callers."""

    return person_route_state(elapsed_s).position


def validate_task_apartment() -> None:
    names = [box.name for box in APARTMENT_BOXES]
    if len(names) != len(set(names)):
        raise RuntimeError("task apartment contains duplicate prim names")
    if not all(
        all(math.isfinite(value) and value > 0.0 for value in box.size)
        for box in APARTMENT_BOXES
    ):
        raise RuntimeError("task apartment contains an invalid box size")
    if math.dist(ROBOT_START[:2], PERSON_START[:2]) < 1.5:
        raise RuntimeError("task apartment person starts too close to G1")
    support = next(box for box in APARTMENT_BOXES if box.name == BOTTLE_SUPPORT_NAME)
    support_top = support.centre[2] + support.size[2] / 2.0
    bottle_bottom = WATER_BOTTLE_POSITION[2] - 0.11
    if abs(support_top - bottle_bottom) > 0.02:
        raise RuntimeError(
            "task apartment bottle is not supported by the blue living-room table"
        )
    for axis in range(2):
        half_extent = support.size[axis] / 2.0
        if abs(WATER_BOTTLE_POSITION[axis] - support.centre[axis]) > half_extent - 0.04:
            raise RuntimeError("task apartment bottle is outside its table surface")
    west_edge_clearance = WATER_BOTTLE_POSITION[0] - (
        support.centre[0] - support.size[0] / 2.0
    )
    front_edge_clearance = WATER_BOTTLE_POSITION[1] - (
        support.centre[1] - support.size[1] / 2.0
    )
    if not 0.07 <= west_edge_clearance <= 0.15:
        raise RuntimeError("task apartment bottle is outside the west grasp zone")
    if not 0.04 <= front_edge_clearance <= 0.08:
        raise RuntimeError("task apartment bottle is outside the front grasp zone")
    bottle_distance = math.dist(ROBOT_START[:2], WATER_BOTTLE_POSITION[:2])
    if not 2.4 <= bottle_distance <= 3.0:
        raise RuntimeError("task apartment bottle is not near the initial robot pose")
    bottle_bearing_degrees = abs(
        math.degrees(
            math.atan2(
                WATER_BOTTLE_POSITION[1] - ROBOT_START[1],
                WATER_BOTTLE_POSITION[0] - ROBOT_START[0],
            )
        )
    )
    if not 100.0 <= bottle_bearing_degrees <= 120.0:
        raise RuntimeError(
            "task apartment bottle is not outside the initial camera view"
        )


validate_task_apartment()
