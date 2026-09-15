"""Reproducible MuJoCo scenes for information-isolated blind evaluation.

This module owns scene generation and scorer-only ground truth.  Agent-facing
code must consume only live sensor products; it must never serialize a
``BlindScene`` or ``BlindTarget`` into observations or tool results.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import sys
from typing import Any, Callable, Iterable, Mapping
import xml.etree.ElementTree as ET


BLIND_SCENE_IDS = ("corridor", "slalom", "alcove")
BLIND_MODE_ENV = "LUXI_BLIND_MODE"
BLIND_RUN_TOKEN_ENV = "LUXI_BLIND_RUN_TOKEN"
_RUN_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")
_SHARED_MEMORY_NAME_PATTERN = re.compile(r"psm_[A-Za-z0-9_-]{1,128}")
_AGENT_SHARED_MEMORY_CHANNELS = ("video", "odom", "cmd")
BLIND_MODEL_PROVIDER = "qwen"
_FORBIDDEN_BLIND_CLI_OPTIONS = (
    "--mujoco-room",
    "--mujoco-room-from-occupancy",
    "--mujoco-global-costmap-from-occupancy",
    "--mujoco-global-map-from-pointcloud",
)


@dataclass(frozen=True)
class BlindTarget:
    label: str
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class BlindPerson:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class BlindScene:
    scene_id: str
    seed: int
    xml: str
    robot_start: tuple[float, float, float]
    target: BlindTarget
    person: BlindPerson | None


@dataclass(frozen=True)
class PreparedBlindRun:
    run_token: str
    run_directory: Path

    def agent_environment(self) -> dict[str, str]:
        """Return the non-semantic envelope inherited by the agent runtime."""

        return {
            BLIND_MODE_ENV: "1",
            BLIND_RUN_TOKEN_ENV: self.run_token,
            # Active Loop has not passed the separate blind-information gate.
            # Pin the accepted owner so a machine-level G1 deployment default
            # cannot silently widen the blind execution boundary.
            "LUXI_MODEL_PROVIDER": "qwen",
        }

    def runtime_paths(self) -> BlindRuntimePaths:
        return BlindRuntimePaths(
            run_directory=self.run_directory,
            costmap_path=self.run_directory / "agent/maps/latest-costmap.json.gz",
            memory_directory=self.run_directory / "agent/memory",
            control_directory=self.run_directory / "simulator/control",
            shm_manifest_path=(
                self.run_directory / "simulator/control/shared-memory.json"
            ),
            head_depth_path=self.run_directory / "agent/sensors/head-depth.npz",
            scorer_third_person_path=(
                self.run_directory / "scorer/third-person.jpg"
            ),
            scorer_telemetry_path=(
                self.run_directory / "scorer/telemetry.jsonl"
            ),
        )


@dataclass(frozen=True)
class BlindRuntimePaths:
    run_directory: Path
    costmap_path: Path
    memory_directory: Path
    control_directory: Path
    shm_manifest_path: Path
    head_depth_path: Path
    scorer_third_person_path: Path
    scorer_telemetry_path: Path


def list_blind_scenes() -> tuple[str, ...]:
    return BLIND_SCENE_IDS


def require_blind_harness(environment: Mapping[str, str]) -> None:
    """Allow only the shared Harness with its sanitized Qwen model boundary."""
    from harness.runtime.configuration import validate_runtime_environment
    validate_runtime_environment(environment)


def reject_forbidden_blind_cli_args(
    argv: Iterable[str],
    environment: Mapping[str, str],
) -> None:
    """Reject simulator or map truth injected through DimOS global options."""

    blind_value = environment.get(BLIND_MODE_ENV, "").strip().lower()
    if blind_value in {"", "0", "false", "no", "off"}:
        return
    normalized = [str(argument).strip().lower().replace("_", "-") for argument in argv]
    for argument in normalized:
        for forbidden in _FORBIDDEN_BLIND_CLI_OPTIONS:
            if argument == forbidden or argument.startswith(f"{forbidden}="):
                raise RuntimeError(
                    f"blind evaluation forbids DimOS option {forbidden}"
                )


def _random_for(scene_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{scene_id}:{seed}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _add_box(
    world: ET.Element,
    *,
    name: str,
    position: tuple[float, float, float],
    half_size: tuple[float, float, float],
    material: str,
) -> None:
    ET.SubElement(
        world,
        "geom",
        name=name,
        type="box",
        pos=" ".join(f"{value:.4f}" for value in position),
        size=" ".join(f"{value:.4f}" for value in half_size),
        material=material,
        contype="1",
        conaffinity="1",
    )


def _add_wall(
    world: ET.Element,
    name: str,
    x: float,
    y: float,
    half_x: float,
    half_y: float,
) -> None:
    _add_box(
        world,
        name=f"wall_{name}",
        position=(x, y, 1.0),
        half_size=(half_x, half_y, 1.0),
        material="wall_material",
    )


def _add_table(world: ET.Element, name: str, x: float, y: float) -> None:
    _add_box(
        world,
        name=f"table_{name}_top",
        position=(x, y, 0.72),
        half_size=(0.52, 0.38, 0.06),
        material="table_material",
    )
    for index, (dx, dy) in enumerate(
        ((-0.43, -0.29), (-0.43, 0.29), (0.43, -0.29), (0.43, 0.29))
    ):
        _add_box(
            world,
            name=f"table_{name}_leg_{index}",
            position=(x + dx, y + dy, 0.33),
            half_size=(0.045, 0.045, 0.33),
            material="table_leg_material",
        )


def _add_door(world: ET.Element, x: float, y: float) -> BlindTarget:
    _add_box(
        world,
        name="target_door_panel",
        position=(x, y, 1.0),
        half_size=(0.06, 0.52, 1.0),
        material="door_material",
    )
    for name, position, size in (
        ("left", (x - 0.01, y - 0.66, 1.08), (0.11, 0.11, 1.08)),
        ("right", (x - 0.01, y + 0.66, 1.08), (0.11, 0.11, 1.08)),
        ("lintel", (x - 0.01, y, 2.10), (0.11, 0.77, 0.11)),
    ):
        _add_box(
            world,
            name=f"target_door_frame_{name}",
            position=position,
            half_size=size,
            material="door_frame_material",
        )
    _add_box(
        world,
        name="target_door_handle",
        position=(x - 0.075, y - 0.35, 1.0),
        half_size=(0.025, 0.035, 0.035),
        material="handle_material",
    )
    return BlindTarget(label="door", x=x, y=y, z=1.0, yaw=math.pi)


def _base_document() -> tuple[ET.Element, ET.Element]:
    root = ET.Element("mujoco", model="luxi_blind_scene")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    for name, rgba in (
        ("floor_material", "0.70 0.72 0.74 1"),
        ("wall_material", "0.88 0.88 0.84 1"),
        ("table_material", "0.48 0.25 0.10 1"),
        ("table_leg_material", "0.18 0.12 0.08 1"),
        ("door_material", "0.16 0.42 0.72 1"),
        ("door_frame_material", "0.12 0.12 0.14 1"),
        ("handle_material", "0.92 0.72 0.18 1"),
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
    ET.SubElement(world, "light", name="ceiling_light", pos="1 1 6", dir="0 0 -1")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.8 0.8 0.8",
        ambient="0.35 0.35 0.35",
        specular="0.2 0.2 0.2",
    )
    ET.SubElement(visual, "global", azimuth="140", elevation="-20")
    return root, world


def _outer_room(
    world: ET.Element,
    *,
    half_height: float,
    front_x: float,
    door_y: float,
) -> None:
    west_x = -2.5
    center_x = (west_x + front_x) / 2.0
    half_width = (front_x - west_x) / 2.0
    north_y = 1.0 + half_height
    south_y = 1.0 - half_height
    _add_wall(world, "north", center_x, north_y, half_width, 0.08)
    _add_wall(world, "south", center_x, south_y, half_width, 0.08)
    _add_wall(world, "west", west_x, 1.0, 0.08, half_height)

    # Build the east wall around this seed's door segment.  The 1.7 m opening
    # leaves the 1.54 m frame visible without leaking its location to the agent.
    opening_half = 0.85
    lower_end = door_y - opening_half
    upper_start = door_y + opening_half
    lower_half = (lower_end - south_y) / 2.0
    upper_half = (north_y - upper_start) / 2.0
    if lower_half <= 0.0 or upper_half <= 0.0:
        raise ValueError("door opening does not fit generated room")
    _add_wall(
        world,
        "east_lower",
        front_x,
        south_y + lower_half,
        0.08,
        lower_half,
    )
    _add_wall(
        world,
        "east_upper",
        front_x,
        upper_start + upper_half,
        0.08,
        upper_half,
    )


def _point_has_footprint_clearance(
    root: ET.Element,
    x: float,
    y: float,
    *,
    footprint_radius: float = 0.3,
    robot_height: float = 1.35,
) -> bool:
    for geom in root.findall(".//geom"):
        if geom.get("type") != "box" or geom.get("contype", "1") == "0":
            continue
        position = [float(value) for value in geom.get("pos", "0 0 0").split()]
        half_size = [float(value) for value in geom.get("size", "0 0 0").split()]
        if position[2] - half_size[2] > robot_height or position[2] + half_size[2] < 0:
            continue
        if (
            position[0] - half_size[0] - footprint_radius
            <= x
            <= position[0] + half_size[0] + footprint_radius
            and position[1] - half_size[1] - footprint_radius
            <= y
            <= position[1] + half_size[1] + footprint_radius
        ):
            return False
    return True


def _validate_scene_reachability(
    root: ET.Element,
    *,
    robot_start: tuple[float, float, float],
    target: BlindTarget,
    front_x: float,
    south_y: float,
    north_y: float,
) -> None:
    """Reject seeded geometry that is unsafe or unsolvable for the known footprint."""

    stop_x = target.x + 0.9 * math.cos(target.yaw)
    stop_y = target.y + 0.9 * math.sin(target.yaw)
    if not _point_has_footprint_clearance(root, robot_start[0], robot_start[1]):
        raise ValueError("generated blind scene places the robot start in collision")
    if not _point_has_footprint_clearance(root, stop_x, stop_y):
        raise ValueError("generated blind scene makes the target standoff unreachable")

    resolution = 0.12
    x_min = -2.5 + 0.38
    x_max = front_x - 0.38
    y_min = south_y + 0.38
    y_max = north_y - 0.38
    width = max(1, math.floor((x_max - x_min) / resolution) + 1)
    height = max(1, math.floor((y_max - y_min) / resolution) + 1)

    def cell_for(x: float, y: float) -> tuple[int, int]:
        return (
            min(width - 1, max(0, round((x - x_min) / resolution))),
            min(height - 1, max(0, round((y - y_min) / resolution))),
        )

    start_cell = cell_for(robot_start[0], robot_start[1])
    goal_cell = cell_for(stop_x, stop_y)
    frontier = deque([start_cell])
    visited = {start_cell}
    while frontier:
        cell_x, cell_y = frontier.popleft()
        if (cell_x, cell_y) == goal_cell:
            return
        for delta_x, delta_y in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_x = cell_x + delta_x
            next_y = cell_y + delta_y
            candidate = (next_x, next_y)
            if (
                not 0 <= next_x < width
                or not 0 <= next_y < height
                or candidate in visited
            ):
                continue
            world_x = x_min + next_x * resolution
            world_y = y_min + next_y * resolution
            if not _point_has_footprint_clearance(root, world_x, world_y):
                continue
            visited.add(candidate)
            frontier.append(candidate)
    raise ValueError("generated blind scene has no footprint-safe path to target standoff")


def generate_blind_scene(
    scene_id: str,
    seed: int,
    *,
    include_person: bool = False,
) -> BlindScene:
    if scene_id not in BLIND_SCENE_IDS:
        choices = ", ".join(BLIND_SCENE_IDS)
        raise ValueError(f"unknown blind scene {scene_id!r}; choose one of: {choices}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("blind scene seed must be an integer")

    rng = _random_for(scene_id, seed)
    root, world = _base_document()
    half_height = {
        "corridor": 2.20,
        "slalom": 2.45,
        "alcove": 2.55,
    }[scene_id]
    front_x = rng.uniform(4.55, 5.55)
    south_y = 1.0 - half_height
    north_y = 1.0 + half_height
    door_y = rng.uniform(south_y + 1.05, north_y - 1.05)
    _outer_room(
        world,
        half_height=half_height,
        front_x=front_x,
        door_y=door_y,
    )

    if scene_id == "corridor":
        guide_side = -1.0 if rng.random() < 0.5 else 1.0
        guide_y = 1.0 + guide_side * (half_height - 0.45)
        _add_wall(
            world,
            "corridor_guide",
            rng.uniform(2.0, 3.2),
            guide_y,
            rng.uniform(0.55, 0.95),
            0.08,
        )
        _add_table(
            world,
            "corridor",
            rng.uniform(0.55, 2.05),
            1.0 - guide_side * rng.uniform(1.15, 1.55),
        )
    elif scene_id == "slalom":
        gap_width = rng.uniform(1.10, 1.50)
        gap_center = rng.uniform(0.65, 1.35)
        gap_lower = gap_center - gap_width / 2.0
        gap_upper = gap_center + gap_width / 2.0
        lower_half = (gap_lower - (south_y + 0.08)) / 2.0
        upper_half = ((north_y - 0.08) - gap_upper) / 2.0
        lower_x = rng.uniform(0.55, 1.25)
        upper_x = rng.uniform(2.15, 3.05)
        if rng.random() < 0.5:
            lower_x, upper_x = upper_x, lower_x
        _add_wall(
            world,
            "slalom_lower",
            lower_x,
            south_y + 0.08 + lower_half,
            0.08,
            lower_half,
        )
        _add_wall(
            world,
            "slalom_upper",
            upper_x,
            gap_upper + upper_half,
            0.08,
            upper_half,
        )
        _add_table(
            world,
            "slalom",
            rng.uniform(3.25, max(3.3, front_x - 0.85)),
            south_y + 0.62 if door_y >= 1.0 else north_y - 0.62,
        )
    else:
        # Put the alcove on the opposite half from the door so the final 0.9 m
        # standoff remains reachable while the route topology still mirrors.
        alcove_side = -1.0 if door_y >= 1.0 else 1.0
        spine_y = 1.0 + alcove_side * rng.uniform(1.0, 1.55)
        spine_x = rng.uniform(1.45, 2.15)
        spine_half = rng.uniform(0.75, 1.10)
        return_x = spine_x + spine_half - 0.08
        return_half = rng.uniform(0.48, 0.78)
        _add_wall(world, "alcove_spine", spine_x, spine_y, spine_half, 0.08)
        _add_wall(
            world,
            "alcove_return",
            return_x,
            spine_y - alcove_side * return_half,
            0.08,
            return_half,
        )
        _add_table(
            world,
            "alcove_left",
            rng.uniform(0.35, 1.05),
            1.0 + alcove_side * rng.uniform(1.60, 1.85),
        )
        _add_table(
            world,
            "alcove_right",
            rng.uniform(3.0, max(3.05, front_x - 1.0)),
            1.0 + alcove_side * rng.uniform(1.45, 1.75),
        )

    target = _add_door(world, front_x - 0.08, door_y)
    start_y = min(
        north_y - 0.55,
        max(south_y + 0.55, door_y + rng.uniform(-0.85, 0.85)),
    )
    robot_start = (
        rng.uniform(-1.35, -0.75),
        start_y,
        rng.uniform(-0.75, 0.75),
    )
    _validate_scene_reachability(
        root,
        robot_start=robot_start,
        target=target,
        front_x=front_x,
        south_y=south_y,
        north_y=north_y,
    )
    person = (
        BlindPerson(
            x=rng.uniform(-0.15, 0.75),
            y=(
                south_y + rng.uniform(0.55, 0.80)
                if robot_start[1] >= 1.0
                else north_y - rng.uniform(0.55, 0.80)
            ),
            z=0.0,
            yaw=rng.uniform(-math.pi, math.pi),
        )
        if include_person
        else None
    )
    xml = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return BlindScene(
        scene_id=scene_id,
        seed=seed,
        xml=xml,
        robot_start=robot_start,
        target=target,
        person=person,
    )


def _run_directory(runtime_root: Path, run_token: str) -> Path:
    if _RUN_TOKEN_PATTERN.fullmatch(run_token) is None:
        raise ValueError("invalid blind run token")
    return runtime_root.expanduser().resolve() / "blind-evaluation" / "runs" / run_token


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _read_private_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"blind run file must contain an object: {path.name}")
    return payload


def _scorer_collision_geometry(scene_xml: str) -> dict[str, Any]:
    """Extract private 2-D obstacle footprints from the generated scene."""

    boxes: list[dict[str, Any]] = []
    root = ET.fromstring(scene_xml)
    for geom in root.findall(".//geom"):
        if geom.get("type") != "box" or geom.get("contype", "1") == "0":
            continue
        position = [float(value) for value in geom.get("pos", "0 0 0").split()]
        half_size = [float(value) for value in geom.get("size", "0 0 0").split()]
        if len(position) != 3 or len(half_size) != 3:
            raise ValueError("generated blind scene has invalid box geometry")
        boxes.append(
            {
                "name": geom.get("name", "unnamed_box"),
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "half_x": half_size[0],
                "half_y": half_size[1],
                "half_z": half_size[2],
            }
        )
    return {
        # Matches DimOS' configured 0.6 m G1 rotation diameter.  Robot geometry
        # is permitted system knowledge; only the scene boxes remain private.
        "robot_footprint_radius_m": 0.3,
        "robot_height_m": 1.35,
        "boxes": boxes,
    }


def prepare_blind_run(
    runtime_root: Path,
    *,
    scene_id: str,
    seed: int,
    include_person: bool = False,
    scorer_video: bool = False,
) -> PreparedBlindRun:
    """Materialize one private simulator payload and scorer manifest."""

    scene = generate_blind_scene(scene_id, seed, include_person=include_person)
    run_token = secrets.token_hex(16)
    run_directory = _run_directory(runtime_root, run_token)
    run_directory.mkdir(parents=True, mode=0o700)
    run_directory.chmod(0o700)
    simulator_payload: dict[str, Any] = {
        "schema_version": 1,
        "scene_xml": scene.xml,
        "robot_start": list(scene.robot_start),
        "person": asdict(scene.person) if scene.person is not None else None,
    }
    if scorer_video:
        simulator_payload["scorer_third_person_path"] = str(
            run_directory / "scorer/third-person.jpg"
        )
    _write_private_json(run_directory / "simulator.json", simulator_payload)
    _write_private_json(
        run_directory / "scorer.json",
        {
            "schema_version": 1,
            "scene_id": scene.scene_id,
            "seed": scene.seed,
            "geometry_sha256": hashlib.sha256(scene.xml.encode()).hexdigest(),
            "robot_start": list(scene.robot_start),
            "target": asdict(scene.target),
            "person": asdict(scene.person) if scene.person is not None else None,
            "scorer_video_enabled": bool(scorer_video),
            "collision_geometry": _scorer_collision_geometry(scene.xml),
        },
    )
    return PreparedBlindRun(run_token=run_token, run_directory=run_directory)


def load_blind_simulator_payload(runtime_root: Path, run_token: str) -> dict[str, Any]:
    return _read_private_json(_run_directory(runtime_root, run_token) / "simulator.json")


def load_blind_scorer_manifest(runtime_root: Path, run_token: str) -> dict[str, Any]:
    return _read_private_json(_run_directory(runtime_root, run_token) / "scorer.json")


def open_prepared_blind_run(runtime_root: Path, run_token: str) -> PreparedBlindRun:
    run_directory = _run_directory(runtime_root, run_token)
    for name in ("simulator.json", "scorer.json"):
        if not (run_directory / name).is_file():
            raise FileNotFoundError(f"blind run is missing {name}")
    return PreparedBlindRun(run_token=run_token, run_directory=run_directory)


def write_blind_shared_memory_manifest(
    runtime_root: Path,
    run_token: str,
    names: Mapping[str, Any],
) -> Path:
    """Bind agent-visible buffers to this run's exact MuJoCo allocation.

    The opaque shared-memory names contain no scene semantics.  Persisting the
    binding prevents a blind run from discovering stale or foreign ``psm_*``
    buffers merely because they happen to have the same byte size.
    """

    prepared = open_prepared_blind_run(runtime_root, run_token)
    channels: dict[str, str] = {}
    for channel in _AGENT_SHARED_MEMORY_CHANNELS:
        name = names.get(channel)
        if not isinstance(name, str) or _SHARED_MEMORY_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"invalid shared-memory name for blind channel {channel}"
            )
        channels[channel] = name
    path = prepared.runtime_paths().shm_manifest_path
    path.parent.mkdir(parents=True, mode=0o700)
    path.parent.chmod(0o700)
    _write_private_json(
        path,
        {
            "schema_version": 1,
            "channels": channels,
        },
    )
    return path


def _read_scorer_telemetry(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(record, dict)
            and record.get("source") == "scorer_ground_truth"
        ):
            records.append(record)
    return sorted(records, key=lambda item: float(item.get("wall_time", 0.0)))


def _scorer_geometry_collisions(
    telemetry: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> set[str]:
    """Return scene boxes intersected by the robot footprint at any sample."""

    geometry = manifest.get("collision_geometry")
    if not isinstance(geometry, dict):
        return set()
    try:
        radius = float(geometry["robot_footprint_radius_m"])
        robot_height = float(geometry["robot_height_m"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return set()
    boxes = geometry.get("boxes")
    if not isinstance(boxes, list):
        return set()

    hits: set[str] = set()
    for record in telemetry:
        try:
            pose = record["robot_pose"]
            robot_x = float(pose["x"])
            robot_y = float(pose["y"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        for box in boxes:
            try:
                name = str(box["name"])
                box_x = float(box["x"])
                box_y = float(box["y"])
                box_z = float(box["z"])
                half_x = float(box["half_x"])
                half_y = float(box["half_y"])
                half_z = float(box["half_z"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if box_z - half_z >= robot_height or box_z + half_z <= 0.0:
                continue
            separation_x = max(abs(robot_x - box_x) - half_x, 0.0)
            separation_y = max(abs(robot_y - box_y) - half_y, 0.0)
            if math.hypot(separation_x, separation_y) <= radius:
                hits.add(name)
    return hits


def _segment_intersects_aabb(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    *,
    minimum_x: float,
    maximum_x: float,
    minimum_y: float,
    maximum_y: float,
) -> bool:
    """Liang-Barsky slab test for a line segment and an axis-aligned box."""

    lower = 0.0
    upper = 1.0
    for start, end, minimum, maximum in (
        (start_x, end_x, minimum_x, maximum_x),
        (start_y, end_y, minimum_y, maximum_y),
    ):
        delta = end - start
        if abs(delta) < 1e-12:
            if start < minimum or start > maximum:
                return False
            continue
        entry = (minimum - start) / delta
        exit_ = (maximum - start) / delta
        if entry > exit_:
            entry, exit_ = exit_, entry
        lower = max(lower, entry)
        upper = min(upper, exit_)
        if lower > upper:
            return False
    return True


def _scorer_swept_geometry_collisions(
    telemetry: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> set[str]:
    """Intersect adjacent robot centre segments with footprint-expanded boxes."""

    geometry = manifest.get("collision_geometry")
    if not isinstance(geometry, dict):
        return set()
    try:
        radius = float(geometry["robot_footprint_radius_m"])
        robot_height = float(geometry["robot_height_m"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return set()
    boxes = geometry.get("boxes")
    if not isinstance(boxes, list):
        return set()

    poses: list[tuple[float, float, float]] = []
    for record in telemetry:
        try:
            pose = record["robot_pose"]
            poses.append(
                (
                    float(record["wall_time"]),
                    float(pose["x"]),
                    float(pose["y"]),
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    poses.sort(key=lambda item: item[0])

    hits: set[str] = set()
    for (_, start_x, start_y), (_, end_x, end_y) in zip(poses, poses[1:]):
        for box in boxes:
            try:
                name = str(box["name"])
                box_x = float(box["x"])
                box_y = float(box["y"])
                box_z = float(box["z"])
                half_x = float(box["half_x"])
                half_y = float(box["half_y"])
                half_z = float(box["half_z"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if box_z - half_z >= robot_height or box_z + half_z <= 0.0:
                continue
            if _segment_intersects_aabb(
                start_x,
                start_y,
                end_x,
                end_y,
                minimum_x=box_x - half_x - radius,
                maximum_x=box_x + half_x + radius,
                minimum_y=box_y - half_y - radius,
                maximum_y=box_y + half_y + radius,
            ):
                hits.add(name)
    return hits


def _scorer_motion_samples(
    telemetry: Iterable[Mapping[str, Any]],
) -> list[dict[str, float]]:
    """Flatten scorer-only per-sync motion evidence in wall-clock order."""

    samples: list[dict[str, float]] = []
    for record in telemetry:
        nested = record.get("motion_samples")
        candidates = nested if isinstance(nested, list) else [record]
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            try:
                wall_time = float(candidate["wall_time"])
                speed = float(candidate["planar_speed_mps"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(wall_time) and math.isfinite(speed):
                samples.append(
                    {
                        "wall_time": wall_time,
                        "planar_speed_mps": max(0.0, speed),
                    }
                )
    return sorted(samples, key=lambda item: item["wall_time"])


def _truth_stationary_evidence(
    motion_samples: Iterable[Mapping[str, float]],
    *,
    stop_command_completed_at: float,
    verification_frame_timestamp: float | None,
    speed_threshold_mps: float = 0.025,
) -> tuple[float | None, float | None]:
    """Find physical stop evidence and, when present, verify-frame speed."""

    consecutive_stopped = 0
    stationary_confirmed_at: float | None = None
    verification_speed: float | None = None
    for sample in motion_samples:
        wall_time = float(sample["wall_time"])
        speed = float(sample["planar_speed_mps"])
        if wall_time > stop_command_completed_at and stationary_confirmed_at is None:
            if speed <= speed_threshold_mps:
                consecutive_stopped += 1
                if consecutive_stopped >= 2:
                    stationary_confirmed_at = wall_time
            else:
                consecutive_stopped = 0
        if (
            verification_frame_timestamp is not None
            and wall_time >= verification_frame_timestamp
            and verification_speed is None
        ):
            verification_speed = speed
    return stationary_confirmed_at, verification_speed


def score_blind_run(
    runtime_root: Path,
    run_token: str,
    agent_result: Mapping[str, Any],
    *,
    expected_standoff_distance_m: float = 0.9,
    distance_tolerance_m: float = 0.3,
) -> dict[str, Any]:
    """Score one run from private MuJoCo truth without feeding it to the agent."""

    prepared = open_prepared_blind_run(runtime_root, run_token)
    paths = prepared.runtime_paths()
    manifest = load_blind_scorer_manifest(runtime_root, run_token)
    telemetry = _read_scorer_telemetry(paths.scorer_telemetry_path)
    stop_command_completed_at = agent_result.get("stop_command_completed_at")
    stationary_confirmed_at = agent_result.get("stationary_confirmed_at")
    verified_at = agent_result.get("verification_frame_timestamp")
    def finite_or_nan(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return math.nan
        return number if math.isfinite(number) else math.nan

    # Parse these independently.  A failed/incomplete task legitimately has no
    # verification frame, but that must not erase its stop command or physical
    # stationary evidence from the audit report.
    stop_command_value = finite_or_nan(stop_command_completed_at)
    stationary_value = finite_or_nan(stationary_confirmed_at)
    verified_value = finite_or_nan(verified_at)
    agent_stationary_order_ok = bool(
        math.isfinite(stop_command_value)
        and math.isfinite(stationary_value)
        and math.isfinite(verified_value)
        and stationary_value > stop_command_value
        and verified_value > stationary_value
    )

    motion_samples = _scorer_motion_samples(telemetry)
    scorer_stationary_at: float | None = None
    verification_truth_speed: float | None = None
    if math.isfinite(stop_command_value):
        scorer_stationary_at, verification_truth_speed = _truth_stationary_evidence(
            motion_samples,
            stop_command_completed_at=stop_command_value,
            verification_frame_timestamp=(
                verified_value if math.isfinite(verified_value) else None
            ),
        )
    physical_stop_latency_ms = (
        max(0.0, (scorer_stationary_at - stop_command_value) * 1_000.0)
        if scorer_stationary_at is not None and math.isfinite(stop_command_value)
        else None
    )
    physical_stop_confirmed = bool(
        scorer_stationary_at is not None
        and physical_stop_latency_ms is not None
        and physical_stop_latency_ms <= 500.0
    )
    truth_robot_stopped = bool(
        physical_stop_confirmed
        and verification_truth_speed is not None
        and verification_truth_speed <= 0.025
    )

    final_truth: dict[str, Any] | None = None
    if telemetry:
        if math.isfinite(verified_value):
            final_truth = next(
                (
                    record
                    for record in telemetry
                    if float(record.get("wall_time", 0.0)) >= verified_value
                ),
                None,
            )
        else:
            final_truth = telemetry[-1]

    ground_truth_distance: float | None = None
    if final_truth is not None:
        try:
            pose = final_truth["robot_pose"]
            target = manifest["target"]
            ground_truth_distance = math.hypot(
                float(target["x"]) - float(pose["x"]),
                float(target["y"]) - float(pose["y"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            ground_truth_distance = None

    try:
        reported_sensor_distance = float(agent_result["target_distance_m"])
    except (KeyError, TypeError, ValueError, OverflowError):
        reported_sensor_distance = None
    if reported_sensor_distance is not None and not math.isfinite(
        reported_sensor_distance
    ):
        reported_sensor_distance = None
    sensor_final_distance: float | None = None
    initial_sensor_distance: float | None = None
    unverified_sensor_distance: float | None = None
    if reported_sensor_distance is None:
        sensor_distance_evidence = "unavailable"
    elif agent_stationary_order_ok:
        sensor_final_distance = reported_sensor_distance
        sensor_distance_evidence = "post_stationary_rgbd"
    elif not math.isfinite(verified_value):
        initial_sensor_distance = reported_sensor_distance
        sensor_distance_evidence = "initial_rgbd_only"
    else:
        unverified_sensor_distance = reported_sensor_distance
        sensor_distance_evidence = "verification_not_post_stationary"
    contact_collision = any(
        bool(record.get("collision_ever") or record.get("collision"))
        for record in telemetry
    )
    latched_contact_sources = {
        str(source)
        for record in telemetry
        for source in (
            record.get("collision_sources")
            if isinstance(record.get("collision_sources"), list)
            else []
        )
    }
    first_collision_candidates: list[float] = []
    for record in telemetry:
        try:
            candidate = float(record["first_collision_wall_time"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(candidate):
            first_collision_candidates.append(candidate)
    first_collision_wall_time = (
        min(first_collision_candidates) if first_collision_candidates else None
    )
    endpoint_collision_geometries = _scorer_geometry_collisions(telemetry, manifest)
    swept_collision_geometries = _scorer_swept_geometry_collisions(
        telemetry,
        manifest,
    )
    collision_geometries = (
        endpoint_collision_geometries | swept_collision_geometries
    )
    collision_sources = sorted(latched_contact_sources)
    if contact_collision:
        if not collision_sources:
            collision_sources.append("mujoco_contact")
    if endpoint_collision_geometries:
        collision_sources.append("scorer_geometry")
    if swept_collision_geometries:
        collision_sources.append("scorer_swept_geometry")
    if collision_geometries and first_collision_wall_time is None:
        ordered_telemetry = sorted(
            telemetry,
            key=lambda record: float(record.get("wall_time", 0.0)),
        )
        for index, record in enumerate(ordered_telemetry):
            endpoint_hit = bool(_scorer_geometry_collisions([record], manifest))
            swept_hit = bool(
                index > 0
                and _scorer_swept_geometry_collisions(
                    [ordered_telemetry[index - 1], record],
                    manifest,
                )
            )
            if endpoint_hit or swept_hit:
                try:
                    first_collision_wall_time = float(record["wall_time"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    pass
                break
    collision_sources = sorted(set(collision_sources))
    collision = bool(collision_sources)
    agent_completed = bool(
        agent_result.get("completed") is True
        and agent_result.get("planner_goal_reached") is True
        and agent_result.get("task_status") == "arrived_verified"
    )
    sensor_distance_ok = bool(
        sensor_final_distance is not None
        and abs(sensor_final_distance - expected_standoff_distance_m)
        <= distance_tolerance_m
    )
    ground_truth_distance_ok = bool(
        ground_truth_distance is not None
        and abs(ground_truth_distance - expected_standoff_distance_m)
        <= distance_tolerance_m
    )
    success = bool(
        agent_completed
        and agent_stationary_order_ok
        and truth_robot_stopped
        and sensor_distance_ok
        and ground_truth_distance_ok
        and not collision
    )

    if not agent_completed:
        failure_reason = str(agent_result.get("task_status") or "agent_incomplete")
    elif not agent_stationary_order_ok:
        failure_reason = "verification_before_stationary_confirmation"
    elif scorer_stationary_at is None or verification_truth_speed is None:
        failure_reason = "robot_not_stopped"
    elif physical_stop_latency_ms is None or physical_stop_latency_ms > 500.0:
        failure_reason = "physical_stop_timeout"
    elif verification_truth_speed > 0.025:
        failure_reason = "robot_not_stopped"
    elif not telemetry or ground_truth_distance is None:
        failure_reason = "scorer_telemetry_unavailable"
    elif collision:
        failure_reason = "collision"
    elif not sensor_distance_ok:
        failure_reason = "sensor_distance_out_of_tolerance"
    elif not ground_truth_distance_ok:
        failure_reason = "ground_truth_distance_out_of_tolerance"
    else:
        failure_reason = None

    report = {
        "schema_version": 1,
        "run_token": run_token,
        "scene_id": manifest["scene_id"],
        "seed": manifest["seed"],
        "task_status": agent_result.get("task_status", "agent_incomplete"),
        "success": success,
        "false_success": bool(agent_completed and not success),
        "failure_reason": failure_reason,
        "verification_after_stationary": agent_stationary_order_ok,
        "verification_frame_timestamp": (
            verified_value if math.isfinite(verified_value) else None
        ),
        "stop_command_completed_at": (
            stop_command_value if math.isfinite(stop_command_value) else None
        ),
        "agent_stationary_confirmed_at": (
            stationary_value if math.isfinite(stationary_value) else None
        ),
        "scorer_stationary_confirmed_at": scorer_stationary_at,
        "physical_stop_latency_ms": (
            round(physical_stop_latency_ms, 1)
            if physical_stop_latency_ms is not None
            else None
        ),
        "verification_truth_speed_mps": verification_truth_speed,
        "stationary_speed_threshold_mps": 0.025,
        "stationary_truth_sample_count_required": 2,
        "truth_motion_samples": len(motion_samples),
        "physical_stop_confirmed": physical_stop_confirmed,
        "robot_stopped": truth_robot_stopped,
        "sensor_final_distance_m": (
            round(sensor_final_distance, 4)
            if sensor_final_distance is not None
            else None
        ),
        "initial_sensor_target_distance_m": (
            round(initial_sensor_distance, 4)
            if initial_sensor_distance is not None
            else None
        ),
        "unverified_sensor_target_distance_m": (
            round(unverified_sensor_distance, 4)
            if unverified_sensor_distance is not None
            else None
        ),
        "sensor_distance_evidence": sensor_distance_evidence,
        "ground_truth_distance_m": (
            round(ground_truth_distance, 4)
            if ground_truth_distance is not None
            else None
        ),
        "expected_standoff_distance_m": expected_standoff_distance_m,
        "collision": collision,
        "collision_ever": collision,
        "first_collision_wall_time": first_collision_wall_time,
        "collision_sources": collision_sources,
        "collision_geometries": sorted(collision_geometries),
        "stop_command_publish_latency_ms": agent_result.get(
            "stop_command_publish_latency_ms"
        ),
        "agent_physical_stop_latency_ms": agent_result.get(
            "physical_stop_latency_ms"
        ),
        "tool_calls": agent_result.get("tool_calls"),
        "planning_steps": agent_result.get("planning_steps"),
        "elapsed_s": agent_result.get("elapsed_s"),
        "vision_requests": agent_result.get("vision_requests", []),
        "verification": agent_result.get("verification"),
        "telemetry_samples": len(telemetry),
    }
    for key in (
        "instruction_submitted_at",
        "first_rgb_frame_ready_s",
        "first_depth_frame_ready_s",
        "first_costmap_ready_s",
        "initial_first_person_frame",
        "final_first_person_frame",
        "costmap_age_at_submit_s",
        "costmap_timestamp_at_submit",
        "costmap_source",
        "target_discovery_s",
        "first_motion_command_s",
        "perception_to_first_motion_s",
        "tool_timings",
        "timed_out",
        "scorer_evidence_boundary",
        "scorer_post_evidence_sample_ready",
        "scorer_post_verification_sample_ready",
        "agent_provider",
        "information_isolation",
        "third_person_enabled",
        "scorer_video_enabled",
    ):
        report[key] = agent_result.get(key)
    _write_private_json(paths.run_directory / "scorer/report.json", report)
    return report


def build_blind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch an information-isolated MuJoCo blind evaluation"
    )
    parser.add_argument("--scene", required=True, choices=BLIND_SCENE_IDS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--with-person", action="store_true")
    parser.add_argument(
        "--scorer-video",
        action="store_true",
        help="record a scorer-only third-person frame; never exposed to the agent or UI",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--no-start-sim", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser


def launch_blind_evaluation(
    argv: Iterable[str],
    *,
    environment: dict[str, str] | None = None,
    execute: Callable[[str, list[str], dict[str, str]], Any] = os.execvpe,
) -> PreparedBlindRun:
    args = build_blind_parser().parse_args(list(argv))
    launch_environment = dict(os.environ if environment is None else environment)
    require_blind_harness(launch_environment)
    runtime_root = Path(
        launch_environment.get(
            "DIMOS_RUNTIME_DIR",
            str(Path.home() / "work/Asset/dimos/runtime"),
        )
    )
    prepared = prepare_blind_run(
        runtime_root,
        scene_id=args.scene,
        seed=args.seed,
        include_person=args.with_person,
        scorer_video=args.scorer_video,
    )
    launch_environment.update(prepared.agent_environment())
    launch_environment.pop("LUXI_THIRD_PERSON_FRAME", None)

    command = [sys.executable, "-m", "harness.app.server"]
    command.extend(["--host", args.host, "--port", str(args.port)])
    if args.asset_root is not None:
        command.extend(["--asset-root", str(args.asset_root)])
    if args.no_start_sim:
        command.append("--no-start-sim")
    if args.no_browser:
        command.append("--no-browser")
    execute(command[0], command, launch_environment)
    return prepared


def main(argv: Iterable[str] | None = None) -> int:
    launch_blind_evaluation(sys.argv[1:] if argv is None else argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
