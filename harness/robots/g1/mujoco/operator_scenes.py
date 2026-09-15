"""Operator-selectable MuJoCo navigation scenes.

The catalog is intentionally separate from formal blind evaluation.  Scene
identity and seeds may be shown to the local operator, but the worker payload
contains only the physics inputs and is never included in Agent observations.
All generated scenes are self-contained MJCF so switching does not depend on
network access or on mutable third-party assets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Any
import xml.etree.ElementTree as ET

from harness.evaluation.blind_evaluation import generate_blind_scene


OPERATOR_SCENE_PAYLOAD_ENV = "LUXI_SIM_SCENE_PAYLOAD"
_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
_MAX_SEED = 2_147_483_647


@dataclass(frozen=True)
class OperatorSceneDescriptor:
    scene_id: str
    label: str
    description: str
    complexity: str
    source: str
    seedable: bool
    supports_person: bool

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorScene:
    scene_id: str
    seed: int
    scene_xml: str | None
    robot_start: tuple[float, float, float] | None
    person: dict[str, float] | None

    def worker_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"person": self.person}
        if self.scene_xml is not None:
            payload["scene_xml"] = self.scene_xml
        if self.robot_start is not None:
            payload["robot_start"] = list(self.robot_start)
        return payload


_SCENES = (
    OperatorSceneDescriptor(
        "office1",
        "DimOS Office 1",
        "固定版 DimOS 自带办公室；作为兼容基线，默认不注入人物。",
        "baseline",
        "pinned_dimos",
        False,
        False,
    ),
    OperatorSceneDescriptor(
        "corridor",
        "随机走廊",
        "带门、桌子和导向墙的可复现走廊。",
        "standard",
        "luxi_procedural",
        True,
        True,
    ),
    OperatorSceneDescriptor(
        "slalom",
        "错位通道",
        "需要绕过错位隔墙的 S 形通道。",
        "standard",
        "luxi_procedural",
        True,
        True,
    ),
    OperatorSceneDescriptor(
        "alcove",
        "凹室办公区",
        "带凹室、双桌和随机门位的办公布局。",
        "standard",
        "luxi_procedural",
        True,
        True,
    ),
    OperatorSceneDescriptor(
        "warehouse",
        "仓储货架区",
        "大空间、多排货架、托盘和交叉通道。",
        "complex",
        "luxi_procedural",
        True,
        True,
    ),
    OperatorSceneDescriptor(
        "office_maze",
        "多房间办公室",
        "多段隔墙、错位门洞和办公桌形成的迷宫。",
        "complex",
        "luxi_procedural",
        True,
        True,
    ),
    OperatorSceneDescriptor(
        "apartment",
        "公寓复合空间",
        "客厅、走廊、卧室和家具组成的多区域场景。",
        "complex",
        "luxi_procedural",
        True,
        True,
    ),
    OperatorSceneDescriptor(
        "home_complex",
        "复杂住宅",
        "包含独立客厅、卧室、餐区和厨房；厨房岛台桌面上放有一瓶小矿泉水。",
        "complex",
        "luxi_procedural",
        True,
        True,
    ),
)
_SCENE_BY_ID = {scene.scene_id: scene for scene in _SCENES}


def list_operator_scenes() -> tuple[OperatorSceneDescriptor, ...]:
    return _SCENES


def operator_scene_descriptor(scene_id: str) -> OperatorSceneDescriptor:
    try:
        return _SCENE_BY_ID[scene_id]
    except KeyError as error:
        choices = ", ".join(_SCENE_BY_ID)
        raise ValueError(f"unknown operator scene {scene_id!r}; choose one of: {choices}") from error


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("operator scene seed must be an integer")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"operator scene seed must be between 0 and {_MAX_SEED}")
    return seed


def _document(model: str) -> tuple[ET.Element, ET.Element]:
    root = ET.Element("mujoco", model=model)
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    for name, rgba in (
        ("floor_material", "0.19 0.22 0.23 1"),
        ("wall_material", "0.78 0.80 0.77 1"),
        ("accent_wall_material", "0.33 0.42 0.43 1"),
        ("wood_material", "0.42 0.23 0.11 1"),
        ("metal_material", "0.22 0.29 0.31 1"),
        ("crate_material", "0.62 0.42 0.16 1"),
        ("sofa_material", "0.16 0.34 0.43 1"),
        ("door_material", "0.12 0.43 0.72 1"),
        ("frame_material", "0.08 0.10 0.11 1"),
        ("handle_material", "0.92 0.72 0.18 1"),
        ("cabinet_material", "0.73 0.68 0.57 1"),
        ("countertop_material", "0.18 0.20 0.22 1"),
        ("appliance_material", "0.66 0.69 0.70 1"),
        ("mattress_material", "0.86 0.88 0.84 1"),
        ("bottle_material", "0.72 0.88 0.96 0.58"),
        ("bottle_label_material", "0.10 0.48 0.82 1"),
        ("bottle_cap_material", "0.10 0.36 0.72 1"),
    ):
        ET.SubElement(asset, "material", name=name, rgba=rgba)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.02",
        material="floor_material",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(world, "light", name="main_light", pos="0 0 7", dir="0 0 -1")
    ET.SubElement(world, "light", name="fill_light", pos="-4 -3 4", dir="1 1 -1")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.7 0.7 0.7",
        ambient="0.35 0.35 0.35",
        specular="0.15 0.15 0.15",
    )
    ET.SubElement(visual, "global", azimuth="135", elevation="-22")
    return root, world


def _box(
    world: ET.Element,
    name: str,
    x: float,
    y: float,
    z: float,
    half_x: float,
    half_y: float,
    half_z: float,
    material: str,
) -> None:
    ET.SubElement(
        world,
        "geom",
        name=name,
        type="box",
        pos=f"{x:.4f} {y:.4f} {z:.4f}",
        size=f"{half_x:.4f} {half_y:.4f} {half_z:.4f}",
        material=material,
        contype="1",
        conaffinity="1",
    )


def _cylinder(
    world: ET.Element,
    name: str,
    x: float,
    y: float,
    z: float,
    radius: float,
    half_height: float,
    material: str,
    *,
    collidable: bool = True,
) -> ET.Element:
    return ET.SubElement(
        world,
        "geom",
        name=name,
        type="cylinder",
        pos=f"{x:.4f} {y:.4f} {z:.4f}",
        size=f"{radius:.4f} {half_height:.4f}",
        material=material,
        contype="1" if collidable else "0",
        conaffinity="1" if collidable else "0",
    )


def _wall(
    world: ET.Element,
    name: str,
    x: float,
    y: float,
    half_x: float,
    half_y: float,
    material: str = "wall_material",
) -> None:
    _box(world, f"wall_{name}", x, y, 1.1, half_x, half_y, 1.1, material)


def _outer_room(
    world: ET.Element,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    door_y: float,
) -> tuple[float, float]:
    wall = 0.10
    _wall(world, "west", x_min, (y_min + y_max) / 2, wall, (y_max - y_min) / 2)
    _wall(world, "north", (x_min + x_max) / 2, y_max, (x_max - x_min) / 2, wall)
    _wall(world, "south", (x_min + x_max) / 2, y_min, (x_max - x_min) / 2, wall)
    opening_half = 0.88
    lower_end = door_y - opening_half
    upper_start = door_y + opening_half
    lower_half = (lower_end - y_min) / 2
    upper_half = (y_max - upper_start) / 2
    if lower_half <= 0.0 or upper_half <= 0.0:
        raise ValueError("operator scene door does not fit outer room")
    _wall(world, "east_lower", x_max, y_min + lower_half, wall, lower_half)
    _wall(world, "east_upper", x_max, upper_start + upper_half, wall, upper_half)

    door_x = x_max - 0.08
    _box(world, "target_door_panel", door_x, door_y, 1.0, 0.06, 0.54, 1.0, "door_material")
    _box(world, "target_door_frame_left", door_x, door_y - 0.69, 1.08, 0.11, 0.11, 1.08, "frame_material")
    _box(world, "target_door_frame_right", door_x, door_y + 0.69, 1.08, 0.11, 0.11, 1.08, "frame_material")
    _box(world, "target_door_frame_lintel", door_x, door_y, 2.10, 0.11, 0.80, 0.11, "frame_material")
    _box(world, "target_door_handle", door_x - 0.075, door_y - 0.36, 1.0, 0.025, 0.035, 0.035, "handle_material")
    return door_x, door_y


def _table(world: ET.Element, name: str, x: float, y: float, *, yaw90: bool = False) -> None:
    half_x, half_y = ((0.38, 0.62) if yaw90 else (0.62, 0.38))
    _box(world, f"table_{name}_top", x, y, 0.74, half_x, half_y, 0.06, "wood_material")
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        _box(
            world,
            f"table_{name}_leg_{index}",
            x + sx * (half_x - 0.08),
            y + sy * (half_y - 0.08),
            0.34,
            0.045,
            0.045,
            0.34,
            "metal_material",
        )


def _rack(world: ET.Element, name: str, x: float, y: float, half_x: float) -> None:
    for index, dx in enumerate((-half_x + 0.08, half_x - 0.08)):
        for side, dy in enumerate((-0.28, 0.28)):
            _box(
                world,
                f"rack_{name}_post_{index}_{side}",
                x + dx,
                y + dy,
                0.85,
                0.055,
                0.055,
                0.85,
                "metal_material",
            )
    for level, z in enumerate((0.28, 0.82, 1.36)):
        _box(
            world,
            f"rack_{name}_shelf_{level}",
            x,
            y,
            z,
            half_x,
            0.32,
            0.055,
            "metal_material",
        )


def _sofa(world: ET.Element, name: str, x: float, y: float) -> None:
    _box(world, f"sofa_{name}_seat", x, y, 0.34, 0.78, 0.38, 0.22, "sofa_material")
    _box(world, f"sofa_{name}_back", x, y + 0.34, 0.68, 0.78, 0.09, 0.48, "sofa_material")
    _box(world, f"sofa_{name}_left", x - 0.72, y, 0.48, 0.10, 0.40, 0.34, "sofa_material")
    _box(world, f"sofa_{name}_right", x + 0.72, y, 0.48, 0.10, 0.40, 0.34, "sofa_material")


def _water_bottle(world: ET.Element, x: float, y: float, surface_z: float) -> None:
    """Build a stable dynamic 500 ml mineral-water bottle."""
    body = ET.SubElement(
        world,
        "body",
        name="entity:water_bottle",
        pos=f"{x:.4f} {y:.4f} {surface_z + 0.095:.4f}",
    )
    ET.SubElement(
        body,
        "joint",
        name="entity:water_bottle:freejoint",
        type="free",
        damping="0.02",
    )
    bottle = _cylinder(
        body,
        "target_water_bottle_body",
        0.0,
        0.0,
        -0.015,
        0.034,
        0.080,
        "bottle_material",
        collidable=False,
    )
    bottle.set("mass", "0")
    shoulder = _cylinder(
        body,
        "target_water_bottle_shoulder",
        0.0,
        0.0,
        0.073,
        0.027,
        0.010,
        "bottle_material",
    )
    shoulder.set("mass", "0")
    shoulder.set("friction", "1.2 0.05 0.01")
    shoulder.set("solref", "0.015 1")
    shoulder.set("solimp", "0.95 0.99 0.001")
    neck = _cylinder(
        body,
        "target_water_bottle_neck",
        0.0,
        0.0,
        0.095,
        0.015,
        0.016,
        "bottle_material",
    )
    neck.set("mass", "0")
    neck.set("friction", "1.2 0.05 0.01")
    neck.set("solref", "0.015 1")
    neck.set("solimp", "0.95 0.99 0.001")
    cap = _cylinder(
        body,
        "target_water_bottle_cap",
        0.0,
        0.0,
        0.117,
        0.017,
        0.007,
        "bottle_cap_material",
    )
    cap.set("mass", "0")
    cap.set("friction", "1.2 0.05 0.01")
    cap.set("solref", "0.015 1")
    cap.set("solimp", "0.95 0.99 0.001")
    label = _cylinder(
        body,
        "target_water_bottle_label",
        0.0,
        0.0,
        -0.005,
        0.0345,
        0.025,
        "bottle_label_material",
        collidable=False,
    )
    label.set("mass", "0")
    ET.SubElement(
        body,
        "geom",
        name="target_water_bottle_collision",
        type="cylinder",
        pos="0 0 -0.0100",
        size="0.0320 0.0750",
        rgba="0 0 0 0",
        # Keep most mass in the bottle volume. Concentrating the full 500 g
        # in the 5 mm base disk produces an ill-conditioned inertia tensor and
        # persistent contact chatter in otherwise stationary simulation.
        mass="0.48",
        friction="2.0 0.05 0.01",
        solref="0.015 1",
        solimp="0.95 0.99 0.001",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(
        body,
        "geom",
        name="target_water_bottle_base_collision",
        type="cylinder",
        pos="0 0 -0.0900",
        size="0.0300 0.0050",
        rgba="0 0 0 0",
        mass="0.02",
        friction="0.3 0.01 0.001",
        solref="0.015 1",
        solimp="0.95 0.99 0.001",
        contype="1",
        conaffinity="1",
    )


def _clear(
    root: ET.Element,
    x: float,
    y: float,
    *,
    radius: float = 0.32,
    robot_height: float = 1.35,
) -> bool:
    for geom in root.findall(".//geom"):
        if geom.get("type") != "box" or geom.get("contype", "1") == "0":
            continue
        position = [float(value) for value in geom.get("pos", "0 0 0").split()]
        size = [float(value) for value in geom.get("size", "0 0 0").split()]
        if position[2] - size[2] > robot_height or position[2] + size[2] < 0:
            continue
        if (
            position[0] - size[0] - radius <= x <= position[0] + size[0] + radius
            and position[1] - size[1] - radius <= y <= position[1] + size[1] + radius
        ):
            return False
    return True


def _reachable(
    root: ET.Element,
    start: tuple[float, float, float],
    goal: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    x_min, x_max, y_min, y_max = bounds
    resolution = 0.16
    x_min += 0.40
    x_max -= 0.40
    y_min += 0.40
    y_max -= 0.40
    width = math.floor((x_max - x_min) / resolution) + 1
    height = math.floor((y_max - y_min) / resolution) + 1

    def cell(x: float, y: float) -> tuple[int, int]:
        return (
            min(width - 1, max(0, round((x - x_min) / resolution))),
            min(height - 1, max(0, round((y - y_min) / resolution))),
        )

    start_cell = cell(start[0], start[1])
    goal_cell = cell(goal[0], goal[1])
    frontier = [start_cell]
    visited = {start_cell}
    while frontier:
        cell_x, cell_y = frontier.pop()
        if (cell_x, cell_y) == goal_cell:
            return True
        for delta_x, delta_y in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (cell_x + delta_x, cell_y + delta_y)
            if (
                candidate in visited
                or not 0 <= candidate[0] < width
                or not 0 <= candidate[1] < height
            ):
                continue
            world_x = x_min + candidate[0] * resolution
            world_y = y_min + candidate[1] * resolution
            if not _clear(root, world_x, world_y):
                continue
            visited.add(candidate)
            frontier.append(candidate)
    return False


def _optional_person(
    root: ET.Element,
    rng: random.Random,
    start: tuple[float, float, float],
    bounds: tuple[float, float, float, float],
    include_person: bool,
) -> dict[str, float] | None:
    if not include_person:
        return None
    x_min, x_max, y_min, y_max = bounds
    for _attempt in range(200):
        x = rng.uniform(x_min + 0.7, x_max - 1.4)
        y = rng.uniform(y_min + 0.7, y_max - 0.7)
        if math.hypot(x - start[0], y - start[1]) < 1.5:
            continue
        if _clear(root, x, y, radius=0.28):
            return {"x": x, "y": y, "z": 0.0, "yaw": rng.uniform(-math.pi, math.pi)}
    raise ValueError("could not place an optional person with safe clearance")


def _generated_room_bounds(root: ET.Element) -> tuple[float, float, float, float]:
    positions: dict[str, list[float]] = {}
    for name in ("wall_west", "wall_east_lower", "wall_north", "wall_south"):
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None:
            raise ValueError(f"generated standard scene is missing {name}")
        positions[name] = [float(value) for value in geom.get("pos", "").split()]
    return (
        positions["wall_west"][0],
        positions["wall_east_lower"][0],
        positions["wall_south"][1],
        positions["wall_north"][1],
    )


def _complex_scene(scene_id: str, seed: int, include_person: bool) -> OperatorScene:
    rng = random.Random(f"luxi-operator:{scene_id}:{seed}")
    root, world = _document(f"luxi_{scene_id}")
    bounds = (-5.5, 5.5, -4.5, 4.5)
    # Apartment furniture occupies the east-wall corners, so its target door
    # varies within the clear central wall segment.  The other layouts can use
    # the full wall while preserving a safe 0.95 m standoff.
    door_y = (
        rng.uniform(-0.55, 0.55)
        if scene_id in {"apartment", "home_complex"}
        else rng.uniform(-2.25, 2.25)
    )
    door_x, target_y = _outer_room(
        world,
        x_min=bounds[0],
        x_max=bounds[1],
        y_min=bounds[2],
        y_max=bounds[3],
        door_y=door_y,
    )

    if scene_id == "warehouse":
        mirror = -1.0 if rng.random() < 0.5 else 1.0
        for row, y in enumerate((-2.55, -0.25, 2.05)):
            x = mirror * rng.uniform(-0.35, 0.35)
            _rack(world, f"{row}_left", x - 2.25, y, rng.uniform(0.78, 1.02))
            _rack(world, f"{row}_right", x + 1.80, y, rng.uniform(0.78, 1.02))
        for index, (x, y) in enumerate(((-4.25, 1.0), (3.85, -3.35), (3.9, 3.25))):
            _box(
                world,
                f"crate_{index}",
                x + rng.uniform(-0.2, 0.2),
                y + rng.uniform(-0.2, 0.2),
                0.38,
                0.42,
                0.42,
                0.38,
                "crate_material",
            )
        start = (-4.55, -3.35 if mirror > 0 else 3.35, rng.uniform(-0.65, 0.65))
    elif scene_id == "office_maze":
        mirror = -1.0 if rng.random() < 0.5 else 1.0
        first_gap = mirror * rng.uniform(1.35, 2.0)
        second_gap = -mirror * rng.uniform(1.2, 1.9)
        for prefix, x, gap in (("a", -1.65, first_gap), ("b", 1.65, second_gap)):
            gap_half = 0.72
            lower_half = (gap - gap_half - bounds[2]) / 2
            upper_half = (bounds[3] - gap - gap_half) / 2
            _wall(world, f"office_{prefix}_lower", x, bounds[2] + lower_half, 0.09, lower_half)
            _wall(world, f"office_{prefix}_upper", x, gap + gap_half + upper_half, 0.09, upper_half)
        _wall(world, "office_cross", 0.0, 0.0, 0.72, 0.09, "accent_wall_material")
        _table(world, "west", -3.65, -mirror * 1.35, yaw90=True)
        _table(world, "center", 0.0, mirror * 2.65)
        _table(world, "east", 3.55, -mirror * 2.55, yaw90=True)
        start = (-4.55, mirror * 3.3, rng.uniform(-0.8, 0.8))
    elif scene_id == "apartment":
        mirror = -1.0 if rng.random() < 0.5 else 1.0
        # Central hallway with two offset room openings.
        _wall(world, "apartment_spine_lower", -0.35, -2.55, 0.09, 1.95)
        _wall(world, "apartment_spine_upper", -0.35, 2.35, 0.09, 2.15)
        _wall(world, "bedroom_left", -3.65, mirror * 0.65, 0.95, 0.09)
        _wall(world, "bedroom_right", 2.45, -mirror * 1.05, 1.05, 0.09)
        _sofa(world, "living", -3.55, -mirror * 2.95)
        _table(world, "dining", 2.85, mirror * 2.55)
        _box(world, "bed_frame", 3.85, -mirror * 2.65, 0.28, 0.95, 0.70, 0.28, "wood_material")
        _box(world, "kitchen_island", 1.65, mirror * 0.45, 0.48, 0.78, 0.38, 0.48, "wood_material")
        start = (-4.55, mirror * 1.45, rng.uniform(-0.8, 0.8))
    elif scene_id == "home_complex":
        # Central and offset doorways connect four distinct living zones while
        # preserving a footprint-safe route for online mapping.
        _wall(world, "home_spine_lower", -0.70, -2.70, 0.09, 1.15)
        _wall(world, "home_spine_upper", -0.70, 2.70, 0.09, 1.80)
        _wall(world, "home_west_cross_left", -4.55, 1.35, 0.95, 0.09)
        _wall(world, "home_west_cross_right", -1.55, 1.35, 0.85, 0.09)
        _wall(world, "home_east_cross_left", 0.15, -1.10, 0.35, 0.09)
        _wall(world, "home_east_cross_right", 3.95, -1.10, 1.55, 0.09)

        # Living room.
        _sofa(world, "home_living", -3.65, -3.25)
        _box(
            world,
            "living_coffee_table",
            -2.05,
            -3.15,
            0.28,
            0.58,
            0.34,
            0.28,
            "wood_material",
        )
        _box(
            world,
            "living_tv_console",
            -4.55,
            -1.48,
            0.34,
            0.54,
            0.20,
            0.34,
            "frame_material",
        )
        _box(
            world,
            "living_tv",
            -4.55,
            -1.43,
            0.95,
            0.48,
            0.06,
            0.32,
            "frame_material",
        )

        # Bedroom.
        _box(
            world,
            "bedroom_bed_frame",
            -3.85,
            2.75,
            0.27,
            1.05,
            0.78,
            0.27,
            "wood_material",
        )
        _box(
            world,
            "bedroom_mattress",
            -3.85,
            2.75,
            0.57,
            0.98,
            0.72,
            0.16,
            "mattress_material",
        )
        _box(
            world,
            "bedroom_headboard",
            -4.82,
            2.75,
            0.82,
            0.09,
            0.78,
            0.55,
            "wood_material",
        )
        _box(
            world,
            "bedroom_nightstand",
            -2.55,
            3.30,
            0.34,
            0.30,
            0.30,
            0.34,
            "cabinet_material",
        )

        # Kitchen.
        _box(
            world,
            "kitchen_north_cabinets",
            2.85,
            3.88,
            0.46,
            1.75,
            0.38,
            0.46,
            "cabinet_material",
        )
        _box(
            world,
            "kitchen_north_countertop",
            2.85,
            3.88,
            0.95,
            1.80,
            0.42,
            0.05,
            "countertop_material",
        )
        _box(
            world,
            "kitchen_east_cabinets",
            4.82,
            2.15,
            0.46,
            0.38,
            1.05,
            0.46,
            "cabinet_material",
        )
        _box(
            world,
            "kitchen_east_countertop",
            4.82,
            2.15,
            0.95,
            0.42,
            1.10,
            0.05,
            "countertop_material",
        )
        _box(
            world,
            "kitchen_refrigerator",
            4.72,
            -2.05,
            1.02,
            0.46,
            0.48,
            1.02,
            "appliance_material",
        )
        _rack(world, "kitchen_pantry", 0.10, 3.55, 0.58)
        _box(
            world,
            "kitchen_island_base",
            2.05,
            1.65,
            0.41,
            0.88,
            0.48,
            0.41,
            "cabinet_material",
        )
        _box(
            world,
            "kitchen_island_countertop",
            2.05,
            1.65,
            0.85,
            0.94,
            0.54,
            0.05,
            "countertop_material",
        )
        bottle_x = 2.05 + rng.uniform(-0.08, 0.08)
        bottle_y = 1.15
        _water_bottle(world, bottle_x, bottle_y, 0.90)

        # Dining area and living-room spawn.
        _table(world, "home_dining", 2.85, -3.05)
        # Face east toward the central passage. Pinned DimOS has a native x/y
        # start option but no yaw option, so keeping this scene's spawn yaw at
        # zero also makes the upstream controller and observer agree exactly.
        start = (-3.00, -1.75, 0.0)
    else:  # pragma: no cover - caller validates the scene identifier
        raise ValueError(f"unsupported complex operator scene: {scene_id}")

    target_standoff = (door_x - 0.95, target_y)
    if not _clear(root, start[0], start[1]):
        raise ValueError(f"generated {scene_id} start is not footprint-safe")
    if not _clear(root, target_standoff[0], target_standoff[1]):
        raise ValueError(f"generated {scene_id} target standoff is not footprint-safe")
    if not _reachable(root, start, target_standoff, bounds):
        raise ValueError(f"generated {scene_id} has no footprint-safe route")

    person = _optional_person(root, rng, start, bounds, include_person)
    return OperatorScene(
        scene_id=scene_id,
        seed=seed,
        scene_xml=ET.tostring(root, encoding="unicode", short_empty_elements=True),
        robot_start=start,
        person=person,
    )


def build_operator_scene(
    scene_id: str,
    seed: int,
    *,
    include_person: bool = False,
) -> OperatorScene:
    descriptor = operator_scene_descriptor(scene_id)
    seed = _validate_seed(seed)
    if include_person and not descriptor.supports_person:
        raise ValueError(f"operator scene {scene_id!r} does not support person injection")
    if scene_id == "office1":
        return OperatorScene(scene_id, seed, None, None, None)
    if scene_id in {"corridor", "slalom", "alcove"}:
        generated = generate_blind_scene(
            scene_id,
            seed,
            include_person=False,
        )
        root = ET.fromstring(generated.xml)
        person = _optional_person(
            root,
            random.Random(f"luxi-operator-person:{scene_id}:{seed}"),
            generated.robot_start,
            _generated_room_bounds(root),
            include_person,
        )
        return OperatorScene(
            scene_id,
            seed,
            generated.xml,
            generated.robot_start,
            person,
        )
    return _complex_scene(scene_id, seed, include_person)


def _validated_worker_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("operator scene payload must be a JSON object")
    allowed = {"scene_xml", "robot_start", "person"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"operator scene payload has unsupported keys: {sorted(unknown)}")

    result: dict[str, Any] = {}
    scene_xml = payload.get("scene_xml")
    if scene_xml is not None:
        if not isinstance(scene_xml, str) or len(scene_xml.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise ValueError("operator scene XML is missing, invalid, or too large")
        try:
            root = ET.fromstring(scene_xml)
        except ET.ParseError as error:
            raise ValueError("operator scene XML is malformed") from error
        if root.tag != "mujoco":
            raise ValueError("operator scene XML root must be <mujoco>")
        if root.find(".//include") is not None:
            raise ValueError("operator scene XML may not include external files")
        if any(element.get("file") for element in root.iter()):
            raise ValueError("operator scene XML may not reference file-backed assets")
        result["scene_xml"] = scene_xml

    start = payload.get("robot_start")
    if start is not None:
        if not isinstance(start, (list, tuple)) or len(start) != 3:
            raise ValueError("operator scene robot_start must contain x,y,yaw")
        try:
            normalized_start = [float(value) for value in start]
        except (TypeError, ValueError) as error:
            raise ValueError("operator scene robot_start must contain numeric values") from error
        if not all(math.isfinite(value) for value in normalized_start):
            raise ValueError("operator scene robot_start must contain finite values")
        result["robot_start"] = normalized_start

    person = payload.get("person")
    if person is not None:
        if not isinstance(person, dict):
            raise ValueError("operator scene person must be an object or null")
        try:
            normalized_person = {
                key: float(person.get(key, 0.0))
                for key in ("x", "y", "z", "yaw")
            }
        except (TypeError, ValueError) as error:
            raise ValueError("operator scene person coordinates must be numeric") from error
        if not all(math.isfinite(value) for value in normalized_person.values()):
            raise ValueError("operator scene person coordinates must be finite")
        result["person"] = normalized_person
    else:
        result["person"] = None
    return result


def write_operator_scene_payload(path: Path, scene: OperatorScene) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _validated_worker_payload(scene.worker_payload())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def load_operator_scene_payload(path: Path) -> dict[str, Any]:
    path = path.expanduser()
    if path.is_symlink():
        raise ValueError("operator scene payload may not be a symbolic link")
    path = path.resolve()
    if path.stat().st_size > _MAX_PAYLOAD_BYTES:
        raise ValueError("operator scene payload is too large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("operator scene payload could not be read") from error
    return _validated_worker_payload(payload)
