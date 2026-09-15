"""RGB-only visual-goal compatibility for the pinned DimOS G1 simulation.

The upstream ``navigate_with_text`` object path assumes synchronized depth and
a BBoxNavigationModule.  The pinned G1 MuJoCo blueprint publishes neither, so
the VLM can find an object but the tracker cannot produce a navigation goal.
This compatibility layer keeps the upstream VLM lookup and A* navigator, while
projecting the detected bbox center into one bounded world-frame goal.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from typing import Any, Sequence


DEFAULT_VERTICAL_FOV_DEGREES = 45.0
DEFAULT_FORWARD_GOAL_DISTANCE = 1.8

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectedGoal:
    x: float
    y: float
    yaw: float
    bearing: float
    local_left: float


def _bounded_setting(name: str, default: float, lower: float, upper: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        number = float(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, value)
        return default
    if not math.isfinite(number) or not lower <= number <= upper:
        logger.warning("Ignoring out-of-range %s=%r", name, value)
        return default
    return number


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def project_bbox_goal(
    *,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    vertical_fov_degrees: float = DEFAULT_VERTICAL_FOV_DEGREES,
    forward_distance: float = DEFAULT_FORWARD_GOAL_DISTANCE,
) -> ProjectedGoal:
    """Project an RGB bbox center into a short world-frame navigation goal."""

    if len(bbox) != 4:
        raise ValueError("bbox must contain x1, y1, x2, y2")
    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox values must be finite")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 1.0 <= vertical_fov_degrees <= 179.0:
        raise ValueError("vertical FOV must be between 1 and 179 degrees")
    if not 0.1 <= forward_distance <= 3.0:
        raise ValueError("forward distance must be between 0.1 and 3.0 metres")

    x1, _y1, x2, _y2 = values
    center_x = max(0.0, min(float(image_width), (x1 + x2) / 2.0))
    vertical_fov = math.radians(vertical_fov_degrees)
    focal_pixels = image_height / (2.0 * math.tan(vertical_fov / 2.0))
    # Optical +x points right, while robot +y points left.
    local_left = -(center_x - image_width / 2.0) / focal_pixels * forward_distance
    bearing = math.atan2(local_left, forward_distance)
    world_x = (
        robot_x
        + math.cos(robot_yaw) * forward_distance
        - math.sin(robot_yaw) * local_left
    )
    world_y = (
        robot_y
        + math.sin(robot_yaw) * forward_distance
        + math.cos(robot_yaw) * local_left
    )
    return ProjectedGoal(
        x=world_x,
        y=world_y,
        yaw=_wrap_angle(robot_yaw + bearing),
        bearing=bearing,
        local_left=local_left,
    )


def navigate_to_visible_object(skill: Any, query: str) -> str | None:
    """Replace the depth-only DimOS object path with one bounded RGB goal."""

    try:
        bbox = skill._get_bbox_for_current_frame(query)
    except Exception:  # noqa: BLE001 - VLM failures should retain semantic fallback
        logger.exception("Failed to get bbox for %s", query)
        return None
    if bbox is None:
        return None

    image = getattr(skill, "_latest_image", None)
    image_data = getattr(image, "data", None)
    shape = getattr(image_data, "shape", ())
    odom = getattr(skill, "_latest_odom", None)
    if len(shape) < 2 or odom is None:
        return f"Found visible '{query}', but RGB geometry or odometry was unavailable."

    fov = _bounded_setting(
        "LUXI_OBJECT_NAV_VERTICAL_FOV",
        DEFAULT_VERTICAL_FOV_DEGREES,
        1.0,
        179.0,
    )
    distance = _bounded_setting(
        "LUXI_OBJECT_NAV_GOAL_DISTANCE",
        DEFAULT_FORWARD_GOAL_DISTANCE,
        0.1,
        3.0,
    )
    try:
        projected = project_bbox_goal(
            robot_x=float(odom.x),
            robot_y=float(odom.y),
            robot_yaw=float(odom.yaw),
            bbox=bbox,
            image_width=int(shape[1]),
            image_height=int(shape[0]),
            vertical_fov_degrees=fov,
            forward_distance=distance,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("Could not project bbox for %s: %s", query, exc)
        return f"Found visible '{query}', but its bounded navigation goal was invalid."

    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.geometry_msgs.Quaternion import Quaternion
    from dimos.msgs.geometry_msgs.Vector3 import Vector3

    goal = PoseStamped(
        frame_id=str(getattr(odom, "frame_id", "") or "world"),
        position=Vector3(projected.x, projected.y, float(getattr(odom, "z", 0.0))),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, projected.yaw)),
    )
    skill._navigation.set_goal(goal)
    return (
        f"Found visible '{query}'. Started bounded visual navigation toward a "
        f"{distance:.2f} m projected goal (bearing {math.degrees(projected.bearing):.1f} deg). "
        "The caller must enforce its motion deadline and stop the navigator."
    )


navigate_to_visible_object._luxi_rgb_only_goal = True  # type: ignore[attr-defined]


def install_object_navigation_compat() -> bool:
    """Install the RGB-only goal projection before DimOS workers are started."""

    value = os.environ.get("LUXI_OBJECT_NAVIGATION_COMPAT", "1").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False

    from dimos.agents.skills.navigation import NavigationSkillContainer

    current = NavigationSkillContainer._navigate_to_object
    if getattr(current, "_luxi_rgb_only_goal", False):
        return True
    NavigationSkillContainer._navigate_to_object = navigate_to_visible_object
    return True
