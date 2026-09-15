#!/usr/bin/env python3
# ruff: noqa: E402
"""Isaac Sim 5.1 runtime for the Luxi Unitree G1 adapter.

This file is executed by ``/isaac-sim/python.sh`` inside the NVIDIA
container.  It deliberately imports Isaac only after constructing
``SimulationApp`` and communicates with the host through ``isaac_protocol``.

The robot USD and the proven 910 -> 12 Unitree locomotion controller are
provided by the configured, read-only G1 reference bundle.  This runtime adds
only the Luxi command/observation boundary; it does not claim to start the
official Genie Sim task, benchmark, or data-collection stacks.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import math
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any

import numpy as np

from harness.robots.g1.isaac.scenes.isaac_brownstone import (
    BrownstoneCollisionRole,
    brownstone_collision_target,
    validate_brownstone_collision_counts,
)
from harness.robots.g1.isaac.isaac_gait_control import IsaacGaitCommandAdapter
from harness.robots.g1.isaac.isaac_entities import (
    ENTITY_ROOT,
    IsaacEntitySpec,
    WATER_BOTTLE,
    selected_entity_specs,
)
from harness.robots.g1.isaac.isaac_protocol import (
    BACKEND_NAME,
    SCHEMA_VERSION,
    IsaacRuntimePaths,
    atomic_write_json,
    atomic_write_npz,
    read_json,
    validated_velocity_command,
)
from harness.robots.g1.isaac.isaac_lidar import (
    LidarIdentityAudit,
    UnverifiedLidarIdentity,
    filter_lidar_returns,
    merge_stable_id_mapping,
    spherical_returns_to_cartesian,
    summarize_lidar_proximity,
    transform_points_wxyz,
)
from harness.robots.g1.isaac.isaac_manipulation import (
    ARM_JOINTS_BY_HAND,
    IsaacManipulationJointController,
    damped_cartesian_joint_delta,
)
from harness.robots.g1.isaac.scenes.isaac_task_apartment import (
    APARTMENT_BOXES,
    PERSON_MANUAL_SPEED_MPS,
    PERSON_MANUAL_TURN_RATE_RPS,
    PERSON_ROUTE,
    PERSON_SPEED_MPS,
    PERSON_TURN_RATE_RPS,
    PERSON_START as TASK_APARTMENT_PERSON_START,
    ROBOT_START as TASK_APARTMENT_ROBOT_START,
    WATER_BOTTLE_POSITION as TASK_APARTMENT_BOTTLE_POSITION,
    person_gait_pose,
    person_motion_should_start as task_person_motion_should_start,
    person_route_state as task_apartment_person_route_state,
    person_swept_position,
    slew_person_heading,
)


ASSET_ROOT = Path("/workspace/assets")
DEFAULT_ROBOT_ASSET = (
    ASSET_ROOT
    / "assets/robots/g1-29dof_wholebody_inspire"
    / "g1_29dof_with_inspire_rev_1_0.usd"
)
DEFAULT_POLICY = ASSET_ROOT / "assets/model/policy1.onnx"
DEFAULT_POLICY_RUNTIME = ASSET_ROOT / "policy_runtime"
DEFAULT_LIDAR_CONFIG = "OS1"
DEFAULT_LIDAR_VARIANT = "OS1_REV6_32ch10hz512res"
SCENE_ASSETS = {
    # Brownstone is supplied through --scene-asset because its downloaded
    # OpenUSD tree is mounted read-only at launch time.
    "brownstone": None,
    "grid": None,
    "skill_demo": None,
    "task_apartment": None,
    "office": (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
    ),
    "warehouse": (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/"
        "warehouse_multiple_shelves.usd"
    ),
}
SCENE_START_POSITIONS = {
    # Brownstone01 is authored in centimetres. This point comes from the
    # clearest bundled level-one kitchen camera, with the robot root 0.80 m
    # above the measured 0.694944 m finished floor.
    "brownstone": (-36.613083, -12.257807, 1.494944),
    "grid": (0.0, 0.0, 0.80),
    "skill_demo": (0.0, 0.0, 0.80),
    "task_apartment": TASK_APARTMENT_ROBOT_START,
    "office": (0.0, 0.0, 0.80),
    "warehouse": (3.2, -0.8, 0.80),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Luxi Isaac G1 bridge runtime")
    parser.add_argument("--runtime-dir", type=Path, default=Path("/workspace/runtime"))
    parser.add_argument(
        "--controller-root", type=Path, default=Path("/workspace/g1ref")
    )
    parser.add_argument("--asset", type=Path, default=DEFAULT_ROBOT_ASSET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--policy-runtime", type=Path, default=DEFAULT_POLICY_RUNTIME)
    parser.add_argument("--scene", choices=sorted(SCENE_ASSETS), default="grid")
    parser.add_argument("--scene-asset")
    parser.add_argument("--person-asset", type=Path)
    parser.add_argument("--person-texture", type=Path)
    parser.add_argument(
        "--entity",
        action="append",
        default=[],
        help="enable one backend-owned dynamic entity (currently: water_bottle)",
    )
    parser.add_argument("--start-position", type=float, nargs=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--sensor-width", type=int, default=640)
    parser.add_argument("--sensor-height", type=int, default=360)
    parser.add_argument("--sensor-hz", type=float, default=10.0)
    parser.add_argument(
        "--disable-observer",
        action="store_true",
        help="skip the operator-only third-person chase camera",
    )
    parser.add_argument("--observer-width", type=int, default=320)
    parser.add_argument("--observer-height", type=int, default=180)
    parser.add_argument("--observer-hz", type=float, default=1.0)
    parser.add_argument(
        "--disable-lidar",
        action="store_true",
        help="skip the RTX lidar and stable-ID self filtering",
    )
    parser.add_argument(
        "--lidar-config",
        default=DEFAULT_LIDAR_CONFIG,
        help="Isaac-supported RTX lidar model name",
    )
    parser.add_argument(
        "--lidar-variant",
        default=DEFAULT_LIDAR_VARIANT,
        help="Isaac-supported RTX lidar model variant",
    )
    parser.add_argument(
        "--lidar-acceptance-test",
        action="store_true",
        help="add non-colliding self/external targets for RTX lidar acceptance only",
    )
    parser.add_argument(
        "--manipulation-acceptance-test",
        action="store_true",
        help="add the deterministic Grid pedestal used by manipulation acceptance",
    )
    args = parser.parse_args(argv)
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if not 1 <= args.sensor_width <= 1920 or not 1 <= args.sensor_height <= 1080:
        parser.error("sensor resolution is outside the supported range")
    if not 1.0 <= args.sensor_hz <= 30.0:
        parser.error("--sensor-hz must be between 1 and 30")
    if not 1 <= args.observer_width <= 960 or not 1 <= args.observer_height <= 540:
        parser.error("observer resolution is outside the supported range")
    if not 0.2 <= args.observer_hz <= 10.0:
        parser.error("--observer-hz must be between 0.2 and 10")
    if args.lidar_acceptance_test and (args.disable_lidar or args.scene != "grid"):
        parser.error("--lidar-acceptance-test requires grid with lidar enabled")
    if args.scene == "task_apartment":
        if args.person_asset is None or not args.person_asset.is_file():
            parser.error("task_apartment requires a readable --person-asset")
        suffix = args.person_asset.suffix.casefold()
        if suffix not in {".gltf", ".glb", ".obj"}:
            parser.error("task_apartment person asset must be GLTF, GLB, or OBJ")
        if suffix == ".obj" and (
            args.person_texture is None or not args.person_texture.is_file()
        ):
            parser.error("task_apartment OBJ requires a readable --person-texture")
    try:
        args.entity_specs = selected_entity_specs(args.entity)
    except ValueError as error:
        parser.error(str(error))
    if args.manipulation_acceptance_test and (
        args.scene != "grid"
        or not any(spec.entity_id == "water_bottle" for spec in args.entity_specs)
    ):
        parser.error(
            "--manipulation-acceptance-test requires grid and --entity water_bottle"
        )
    return args


ARGS = parse_args()

# Isaac/Omniverse imports must happen after SimulationApp exists.
from isaacsim import SimulationApp


SIMULATION_APP = SimulationApp(
    {
        "headless": bool(ARGS.headless),
        "enable_motion_bvh": True,
        "extra_args": [
            "--/rtx-transient/stableIds/enabled=true",
            # render() deliberately decouples RTX updates from physics. Isaac
            # 5.1 otherwise emits one harmless interpolation warning per
            # sensor frame and can grow simulator.log by gigabytes.
            "--/log/level=error",
        ],
        "width": ARGS.sensor_width if ARGS.headless else max(1280, ARGS.sensor_width),
        "height": ARGS.sensor_height if ARGS.headless else max(720, ARGS.sensor_height),
    }
)

import omni.usd
import omni.physx
import omni.kit.asset_converter
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.rtx import LidarRtx, get_gmo_data
from pxr import (
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    Sdf,
    Usd,
    UsdGeom,
    UsdLux,
    UsdPhysics,
    UsdShade,
    UsdSkel,
)


BROWNSTONE_SCALE = 0.01
BROWNSTONE_ROOT_PRIMS = (
    "/World/Brownstone01",
    "/World/Brownstone01_Stair",
)


@dataclass
class _EntityCartesianMotion:
    token: str
    action: str
    entity_id: str
    hand: str
    target_hand_position: np.ndarray[Any, Any]
    started_at: float
    deadline: float
    stable_frames: int = 0
    release_after_arrival: bool = False
    target_entity_position: np.ndarray[Any, Any] | None = None
    joint_targets: dict[str, float] | None = None
    start_entity_position: np.ndarray[Any, Any] | None = None


@dataclass
class _TaskPersonRig:
    translate: Any
    heading: Any | None = None
    skeleton: Any | None = None
    base_rest_transforms: tuple[Any, ...] = ()
    joint_indices: dict[str, int] | None = None
    last_pose_key: tuple[Any, ...] | None = None


def _joint_rotation(axis: tuple[float, float, float], angle_deg: float) -> Any:
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(Gf.Rotation(Gf.Vec3d(*axis), float(angle_deg)))
    return transform


def _apply_task_person_pose(
    rig: _TaskPersonRig,
    *,
    elapsed_s: float,
    moving: bool,
) -> None:
    """Apply a relaxed standing pose or a lightweight procedural walk cycle."""

    if rig.skeleton is None or not rig.base_rest_transforms or not rig.joint_indices:
        return
    pose = person_gait_pose(elapsed_s, moving=moving)
    pose_key = (
        moving,
        round(pose.left_arm_swing_deg, 3),
        round(pose.right_arm_swing_deg, 3),
        round(pose.left_thigh_swing_deg, 3),
        round(pose.right_thigh_swing_deg, 3),
        round(pose.left_knee_bend_deg, 3),
        round(pose.right_knee_bend_deg, 3),
    )
    if rig.last_pose_key == pose_key:
        return
    transforms = [Gf.Matrix4d(value) for value in rig.base_rest_transforms]

    # The source asset is authored in a T-pose. Lower both arms beside the
    # torso first, then add the smaller fore/aft counter-swing of walking.
    arm_angles = {
        "upperarm_l": (-68.0, pose.left_arm_swing_deg),
        "upperarm_r": (68.0, pose.right_arm_swing_deg),
    }
    for joint, (lowering, swing) in arm_angles.items():
        index = rig.joint_indices[joint]
        transforms[index] = (
            _joint_rotation((0.0, 0.0, 1.0), lowering)
            * _joint_rotation((1.0, 0.0, 0.0), swing)
            * transforms[index]
        )

    joint_angles = {
        "thigh_l": pose.left_thigh_swing_deg,
        "thigh_r": pose.right_thigh_swing_deg,
        "calf_l": pose.left_knee_bend_deg,
        "calf_r": -pose.right_knee_bend_deg,
    }
    for joint, angle in joint_angles.items():
        index = rig.joint_indices[joint]
        transforms[index] = (
            _joint_rotation(
                (1.0, 0.0, 0.0),
                angle,
            )
            * transforms[index]
        )
    rig.skeleton.GetRestTransformsAttr().Set(transforms)
    rig.last_pose_key = pose_key


def _load_locomotion_controller(controller_root: Path) -> type[Any]:
    root = controller_root.expanduser().resolve()
    source = root / "g1_locomotion_controller.py"
    if not source.is_file():
        raise FileNotFoundError(f"G1 reference controller is missing: {source}")
    sys.path.insert(0, str(root))
    from g1_locomotion_controller import G1LocomotionController

    return G1LocomotionController


def _wait_for_stage_assets(timeout_s: float = 600.0) -> None:
    context = omni.usd.get_context()
    started = time.monotonic()
    last_report = started
    while True:
        loaded, total, loading = context.get_stage_loading_status()
        if loading == 0:
            return
        now = time.monotonic()
        if now - started > timeout_s:
            raise TimeoutError(
                f"scene assets did not load in {timeout_s:.0f}s "
                f"(loaded={loaded}, total={total}, loading={loading})"
            )
        if now - last_report >= 10.0:
            print(
                f"Isaac scene loading: loaded={loaded} total={total} loading={loading}",
                flush=True,
            )
            last_report = now
        SIMULATION_APP.update()


def _load_brownstone_scene(stage: Any, scene_asset: str) -> None:
    """Compose one complete home and add static structure/obstacle collision."""

    # This 2022 AEC pack binds many meshes to absolute /Looks paths. A normal
    # reference would move the home under another prim and USD 5.1 would
    # correctly discard those out-of-scope targets. Sublayering retains the
    # pack's /World + /Looks namespace; we then convert both authored building
    # roots from centimetres to metres without modifying the read-only asset.
    root_layer = stage.GetRootLayer()
    root_layer.subLayerPaths.append(scene_asset)
    for prim_path in BROWNSTONE_ROOT_PRIMS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Brownstone scene is missing {prim_path}")
        xformable = UsdGeom.Xformable(prim)
        translate_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ),
            None,
        )
        scale_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeScale
            ),
            None,
        )
        if translate_op is None or scale_op is None:
            raise RuntimeError(
                f"Brownstone scene has unexpected xform ops at {prim_path}"
            )
        translate_op.Set(translate_op.Get() * BROWNSTONE_SCALE)
        scale_op.Set(scale_op.Get() * BROWNSTONE_SCALE)
    _wait_for_stage_assets()

    collision_counts: dict[str, int] = {}
    role_counts = {
        BrownstoneCollisionRole.STRUCTURE: 0,
        BrownstoneCollisionRole.NAVIGATION_OBSTACLE: 0,
    }
    for prim in stage.Traverse():
        target = brownstone_collision_target(str(prim.GetPath()))
        if not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy() or target is None:
            continue
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr().Set(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
        collision_counts[target.group] = collision_counts.get(target.group, 0) + 1
        role_counts[target.role] += 1
    validate_brownstone_collision_counts(collision_counts)
    structural_count = role_counts[BrownstoneCollisionRole.STRUCTURE]
    obstacle_count = role_counts[BrownstoneCollisionRole.NAVIGATION_OBSTACLE]
    if structural_count == 0:
        raise RuntimeError("Brownstone01 composed without structural collision meshes")
    collision_count = structural_count + obstacle_count
    print(
        f"Brownstone01 sublayered at scale={BROWNSTONE_SCALE:g} "
        f"with {collision_count} static colliders "
        f"({structural_count} structure, {obstacle_count} navigation obstacle)",
        flush=True,
    )


def _yaw_from_wxyz(quaternion: np.ndarray[Any, Any]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _atomic_error(paths: IsaacRuntimePaths, message: str) -> None:
    atomic_write_json(
        paths.state,
        {
            "schema_version": SCHEMA_VERSION,
            "backend": BACKEND_NAME,
            "ready": False,
            "written_at": time.time(),
            "error": message,
        },
    )


def _set_camera_fov(camera: Camera, horizontal_fov_degrees: float) -> None:
    aperture = float(camera.get_horizontal_aperture())
    focal_length = aperture / (
        2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    )
    camera.set_focal_length(focal_length)


def _add_lidar_acceptance_geometry(stage: Any) -> None:
    """Add render-only targets used to prove stable-ID self filtering."""

    self_target = UsdGeom.Cube.Define(stage, "/World/G1/LidarAcceptanceSelfTarget")
    self_target.CreateSizeAttr(1.0)
    self_xform = UsdGeom.Xformable(self_target.GetPrim())
    self_xform.AddTranslateOp().Set(Gf.Vec3d(-0.30, 0.0, 0.55))
    self_xform.AddScaleOp().Set(Gf.Vec3f(0.12, 0.30, 0.30))

    external = UsdGeom.Cube.Define(stage, "/World/LidarAcceptanceExternalTarget")
    external.CreateSizeAttr(1.0)
    external_xform = UsdGeom.Xformable(external.GetPrim())
    external_xform.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.50))
    external_xform.AddScaleOp().Set(Gf.Vec3f(0.40, 0.40, 1.0))


def _add_manipulation_acceptance_geometry(
    stage: Any,
    start_position: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Add a static pedestal and return its reachable bottle-centre pose."""

    pedestal = UsdGeom.Cube.Define(stage, "/World/LuxiManipulationPedestal")
    pedestal.CreateSizeAttr(1.0)
    pedestal.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.35, 0.25)])
    xform = UsdGeom.Xformable(pedestal.GetPrim())
    centre = (
        float(start_position[0]) + 0.24,
        float(start_position[1]) - 0.22,
        0.31,
    )
    xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xform.AddScaleOp().Set(Gf.Vec3f(0.10, 0.10, 0.62))
    UsdPhysics.CollisionAPI.Apply(pedestal.GetPrim())
    return centre[0], centre[1], 0.73


def _add_skill_demo_person(
    stage: Any,
    *,
    root_path: str = "/World/LuxiSkillDemoPerson",
    start: tuple[float, float, float] = (2.4, 0.0, 0.0),
) -> Any:
    """Create a local articulated-looking mannequin for visual follow tests."""

    root = UsdGeom.Xform.Define(stage, root_path)
    translate = root.AddTranslateOp()
    translate.Set(Gf.Vec3d(*start))

    def part(
        name: str,
        *,
        shape: str,
        position: tuple[float, float, float],
        scale: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> None:
        path = f"{root_path}/{name}"
        geometry = (
            UsdGeom.Sphere.Define(stage, path)
            if shape == "sphere"
            else UsdGeom.Cube.Define(stage, path)
        )
        if shape == "sphere":
            geometry.CreateRadiusAttr(0.5)
        else:
            geometry.CreateSizeAttr(1.0)
        geometry.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xform = UsdGeom.Xformable(geometry.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))

    part(
        "Torso",
        shape="cube",
        position=(0.0, 0.0, 1.25),
        scale=(0.22, 0.40, 0.55),
        color=(0.12, 0.38, 0.82),
    )
    part(
        "Head",
        shape="sphere",
        position=(0.0, 0.0, 1.82),
        scale=(0.24, 0.24, 0.24),
        color=(0.72, 0.49, 0.34),
    )
    for side, lateral in (("Left", 0.22), ("Right", -0.22)):
        part(
            f"{side}Leg",
            shape="cube",
            position=(0.0, lateral, 0.62),
            scale=(0.13, 0.13, 0.62),
            color=(0.08, 0.10, 0.16),
        )
        part(
            f"{side}Arm",
            shape="cube",
            position=(0.0, 1.75 * lateral, 1.25),
            scale=(0.11, 0.11, 0.52),
            color=(0.72, 0.49, 0.34),
        )
    return translate


def _add_scanned_person(
    stage: Any,
    *,
    obj_path: Path,
    texture_path: Path,
    root_path: str,
    start: tuple[float, float, float],
) -> Any:
    """Load a textured local human scan as a sensor-visible USD mesh."""

    vertices: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_uvs: list[tuple[int, int, int]] = []

    def obj_index(raw: str, count: int) -> int:
        value = int(raw)
        return value - 1 if value > 0 else count + value

    with obj_path.open("r", encoding="utf-8") as source:
        for raw_line in source:
            fields = raw_line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif fields[0] == "vt" and len(fields) >= 3:
                # OBJ and UsdUVTexture both use the authored ``st`` convention
                # for this scan. Flipping V here mirrors the texture atlas and
                # smears face/floor fragments across the clothes.
                texcoords.append((float(fields[1]), float(fields[2])))
            elif fields[0] == "f" and len(fields) >= 4:
                corners: list[tuple[int, int]] = []
                for token in fields[1:]:
                    parts = token.split("/")
                    vertex_index = obj_index(parts[0], len(vertices))
                    uv_index = (
                        obj_index(parts[1], len(texcoords))
                        if len(parts) > 1 and parts[1]
                        else -1
                    )
                    corners.append((vertex_index, uv_index))
                for offset in range(1, len(corners) - 1):
                    face = (corners[0], corners[offset], corners[offset + 1])
                    triangles.append(tuple(corner[0] for corner in face))
                    triangle_uvs.append(tuple(corner[1] for corner in face))

    if not vertices or not triangles:
        raise RuntimeError(f"person OBJ contains no renderable mesh: {obj_path}")
    if not texcoords or any(index < 0 for face in triangle_uvs for index in face):
        raise RuntimeError(
            f"person OBJ does not contain complete texture coordinates: {obj_path}"
        )

    source_points = np.asarray(vertices, dtype=np.float32)
    horizontal_centre = (
        (float(source_points[:, 0].min()) + float(source_points[:, 0].max())) / 2.0,
        (float(source_points[:, 2].min()) + float(source_points[:, 2].max())) / 2.0,
    )
    floor = float(source_points[:, 1].min())
    target_height = 1.76
    source_height = float(source_points[:, 1].max()) - floor
    scale = target_height / source_height
    points = (
        np.column_stack(
            (
                -(source_points[:, 2] - horizontal_centre[1]),
                -(source_points[:, 0] - horizontal_centre[0]),
                source_points[:, 1] - floor,
            )
        )
        * scale
    )

    face_indices = np.asarray(triangles, dtype=np.int32)
    first = points[face_indices[:, 0]]
    second = points[face_indices[:, 1]]
    third = points[face_indices[:, 2]]
    face_normals = np.cross(second - first, third - first)
    vertex_normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(vertex_normals, face_indices[:, corner], face_normals)
    lengths = np.linalg.norm(vertex_normals, axis=1)
    valid = lengths > 1e-8
    vertex_normals[valid] /= lengths[valid, None]
    vertex_normals[~valid] = (0.0, 0.0, 1.0)

    root = UsdGeom.Xform.Define(stage, root_path)
    translate = root.AddTranslateOp()
    translate.Set(Gf.Vec3d(*start))
    # The scan's forward axis is +Y. Face the initial G1 position along -X
    # while preserving the root translation op used by the follow route.
    root.AddRotateZOp().Set(90.0)
    mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/Body")
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(face_indices.reshape(-1).tolist())
    mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, normal)) for normal in vertex_normals])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(*map(float, points.min(axis=0))),
            Gf.Vec3f(*map(float, points.max(axis=0))),
        ]
    )

    flattened_uvs = [
        Gf.Vec2f(*map(float, texcoords[index]))
        for face in triangle_uvs
        for index in face
    ]
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set(flattened_uvs)

    material = UsdShade.Material.Define(stage, f"{root_path}/Material")
    surface = UsdShade.Shader.Define(stage, f"{root_path}/Material/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    texture = UsdShade.Shader.Define(stage, f"{root_path}/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture_path))
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    reader = UsdShade.Shader.Define(stage, f"{root_path}/Material/UVReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(),
        "result",
    )
    surface.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f,
    ).ConnectToSource(texture.ConnectableAPI(), "rgb")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(),
        "surface",
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    print(
        f"Loaded textured person scan: vertices={len(vertices)} "
        f"triangles={len(triangles)} texture={texture_path.name}",
        flush=True,
    )
    return translate


def _convert_gltf_person_to_usd(
    source_path: Path,
    output_path: Path,
) -> Path:
    """Convert a vendored glTF character with Isaac's own asset converter."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_mtime = max(
        (
            path.stat().st_mtime
            for path in source_path.parent.iterdir()
            if path.is_file()
        ),
        default=source_path.stat().st_mtime,
    )
    if output_path.is_file() and output_path.stat().st_mtime >= source_mtime:
        return output_path

    context = omni.kit.asset_converter.AssetConverterContext()
    context.export_preview_surface = True
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.use_meter_as_world_unit = True
    context.convert_stage_up_z = True
    context.embed_textures = True
    converter = omni.kit.asset_converter.get_instance()

    async def convert() -> bool:
        task = converter.create_converter_task(
            str(source_path),
            str(output_path),
            lambda _current, _total: None,
            context,
        )
        success = await task.wait_until_finished()
        if not success:
            raise RuntimeError(
                "Isaac person asset conversion failed: "
                f"{task.get_status()} {task.get_error_message()}"
            )
        return True

    future = asyncio.ensure_future(convert())
    while not future.done():
        SIMULATION_APP.update()
    future.result()
    if not output_path.is_file():
        raise RuntimeError("Isaac person asset converter produced no USD")
    print(f"Converted glTF person asset: {source_path.name}", flush=True)
    return output_path


def _add_gltf_person(
    stage: Any,
    *,
    source_path: Path,
    converted_path: Path,
    root_path: str,
    start: tuple[float, float, float],
) -> _TaskPersonRig:
    """Reference a clean PBR character converted by the Isaac runtime."""

    usd_path = _convert_gltf_person_to_usd(source_path, converted_path)
    root = UsdGeom.Xform.Define(stage, root_path)
    translate = root.AddTranslateOp()
    translate.Set(Gf.Vec3d(*start))
    heading = root.AddRotateZOp()
    heading.Set(180.0)
    character_path = f"{root_path}/Character"
    character = UsdGeom.Xform.Define(stage, character_path)
    body_path = f"{character_path}/Body"
    add_reference_to_stage(str(usd_path), body_path)
    _wait_for_stage_assets()
    body_prim = stage.GetPrimAtPath(body_path)
    bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    mesh_range = Gf.Range3d()
    mesh_count = 0
    for prim in Usd.PrimRange(body_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_range.UnionWith(bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        mesh_count += 1
    if mesh_count == 0 or mesh_range.IsEmpty():
        raise RuntimeError("converted person USD contains no bounded render mesh")
    size = mesh_range.GetSize()
    vertical_axis = max(range(3), key=lambda index: float(size[index]))
    target_height = 1.76
    character_scale = target_height / float(size[vertical_axis])
    character.AddScaleOp().Set(
        Gf.Vec3f(character_scale, character_scale, character_scale)
    )
    # Asset Converter versions differ in whether they bake glTF's up-axis
    # into skinned nodes. Infer the body's longest local axis and lift it to Z.
    if vertical_axis == 0:
        character.AddRotateYOp().Set(-90.0)
    elif vertical_axis == 1:
        character.AddRotateXOp().Set(90.0)
    # Preserve the converted asset's local upright/facing correction. The
    # outer world-Z heading adds 180 degrees plus the route tangent, changing
    # only yaw so a trailing robot sees the person's back.
    character.AddRotateZOp().Set(-90.0)
    rig: _TaskPersonRig | None = None
    gait_joint_names = {
        "upperarm_l",
        "upperarm_r",
        "thigh_l",
        "thigh_r",
        "calf_l",
        "calf_r",
    }
    for prim in Usd.PrimRange(body_prim):
        if not prim.IsA(UsdSkel.Skeleton):
            continue
        skeleton = UsdSkel.Skeleton(prim)
        joints = list(skeleton.GetJointsAttr().Get() or [])
        rest_transforms = list(skeleton.GetRestTransformsAttr().Get() or [])
        if len(joints) != len(rest_transforms):
            continue
        joint_indices: dict[str, int] = {}
        for index, joint in enumerate(joints):
            leaf = str(joint).rsplit("/", 1)[-1].casefold()
            if leaf in gait_joint_names:
                joint_indices[leaf] = index
        if gait_joint_names.issubset(joint_indices):
            rig = _TaskPersonRig(
                translate=translate,
                heading=heading,
                skeleton=skeleton,
                base_rest_transforms=tuple(rest_transforms),
                joint_indices=joint_indices,
            )
            _apply_task_person_pose(rig, elapsed_s=0.0, moving=False)
            break
    if rig is None:
        raise RuntimeError("converted person USD has no compatible walking skeleton")
    print(f"Loaded PBR person character: {source_path.name}", flush=True)
    print(
        "PBR person local extent: "
        f"{tuple(round(float(value), 3) for value in size)} "
        f"meshes={mesh_count} vertical_axis={vertical_axis} "
        f"scale={character_scale:.4f} "
        f"gait_joints={sorted(rig.joint_indices or {})}",
        flush=True,
    )
    return rig


def _add_skill_demo_exploration_room(stage: Any) -> None:
    """Add a collision-authored room and open corridor for frontier tests."""

    root = UsdGeom.Xform.Define(stage, "/World/LuxiSkillDemoRoom")
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    def wall(
        name: str,
        *,
        centre: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> None:
        geometry = UsdGeom.Cube.Define(
            stage,
            f"/World/LuxiSkillDemoRoom/{name}",
        )
        geometry.CreateSizeAttr(1.0)
        geometry.CreateDisplayColorAttr([Gf.Vec3f(0.68, 0.71, 0.75)])
        xform = UsdGeom.Xformable(geometry.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())

    # The east wall has a 1.8 m doorway.  Parallel corridor walls extend
    # beyond the lidar horizon, leaving a real unknown frontier instead of a
    # hidden map prior or an artificial free-space seed.
    wall("West", centre=(-4.0, 0.0, 1.2), size=(0.16, 8.0, 2.4))
    wall("North", centre=(0.0, 4.0, 1.2), size=(8.0, 0.16, 2.4))
    wall("South", centre=(0.0, -4.0, 1.2), size=(8.0, 0.16, 2.4))
    wall("EastNorth", centre=(4.0, 2.45, 1.2), size=(0.16, 3.1, 2.4))
    wall("EastSouth", centre=(4.0, -2.45, 1.2), size=(0.16, 3.1, 2.4))
    wall("CorridorNorth", centre=(7.5, 0.98, 1.2), size=(7.0, 0.16, 2.4))
    wall("CorridorSouth", centre=(7.5, -0.98, 1.2), size=(7.0, 0.16, 2.4))


def _add_task_apartment(
    stage: Any,
    *,
    person_asset: Path,
    person_texture: Path | None,
    person_cache: Path,
) -> _TaskPersonRig:
    """Author a furnished collision scene for search/follow acceptance."""

    root_path = "/World/LuxiTaskApartment"
    UsdGeom.Xform.Define(stage, root_path)
    for box in APARTMENT_BOXES:
        geometry = UsdGeom.Cube.Define(stage, f"{root_path}/{box.name}")
        geometry.CreateSizeAttr(1.0)
        geometry.CreateDisplayColorAttr([Gf.Vec3f(*box.color)])
        xform = UsdGeom.Xformable(geometry.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*box.centre))
        xform.AddScaleOp().Set(Gf.Vec3f(*box.size))
        if box.collision:
            collision = UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
            collision.CreateCollisionEnabledAttr().Set(True)
    root_path = "/World/LuxiTaskApartmentPerson"
    if person_asset.suffix.casefold() in {".gltf", ".glb"}:
        return _add_gltf_person(
            stage,
            source_path=person_asset,
            converted_path=person_cache / "quaternius-person.usd",
            root_path=root_path,
            start=TASK_APARTMENT_PERSON_START,
        )
    if person_texture is None:
        raise RuntimeError("OBJ person asset requires a texture")
    return _TaskPersonRig(
        translate=_add_scanned_person(
            stage,
            obj_path=person_asset,
            texture_path=person_texture,
            root_path=root_path,
            start=TASK_APARTMENT_PERSON_START,
        )
    )


def _bind_preview_material(
    stage: Any,
    geometry: Any,
    *,
    name: str,
    color: tuple[float, float, float],
    roughness: float,
    opacity: float = 1.0,
) -> None:
    """Bind one compact PBR material to a procedural bottle component."""

    material_path = f"{geometry.GetPath()}/Looks/{name}"
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, f"{material_path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(),
        "surface",
    )
    UsdShade.MaterialBindingAPI.Apply(geometry).Bind(material)


def _add_realistic_water_bottle_visuals(
    stage: Any,
    geometry: Any,
    spec: IsaacEntitySpec,
) -> None:
    """Add a transparent PET body, water, label and cap to one rigid collider."""

    root = geometry.GetPrim()
    _bind_preview_material(
        stage,
        root,
        name="ClearPet",
        color=(0.72, 0.91, 0.98),
        roughness=0.12,
        opacity=0.32,
    )

    def cylinder(
        name: str,
        *,
        radius: float,
        height: float,
        z: float,
        color: tuple[float, float, float],
        roughness: float,
        opacity: float = 1.0,
    ) -> Any:
        item = UsdGeom.Cylinder.Define(stage, f"{spec.prim_path}/{name}")
        item.CreateAxisAttr("Z")
        item.CreateRadiusAttr(radius)
        item.CreateHeightAttr(height)
        item.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        item.CreateDisplayOpacityAttr([opacity])
        UsdGeom.Xformable(item.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
        _bind_preview_material(
            stage,
            item.GetPrim(),
            name="Material",
            color=color,
            roughness=roughness,
            opacity=opacity,
        )
        return item

    cylinder(
        "Water",
        radius=spec.radius_m - 0.004,
        height=0.15,
        z=-0.025,
        color=(0.30, 0.68, 0.88),
        roughness=0.05,
        opacity=0.58,
    )
    cylinder(
        "WaterMeniscus",
        radius=spec.radius_m - 0.005,
        height=0.003,
        z=0.051,
        color=(0.62, 0.88, 0.97),
        roughness=0.02,
        opacity=0.72,
    )
    cylinder(
        "Label",
        radius=spec.radius_m + 0.0015,
        height=0.052,
        z=-0.005,
        color=(0.93, 0.96, 0.98),
        roughness=0.42,
    )
    cylinder(
        "LabelBlueBand",
        radius=spec.radius_m + 0.002,
        height=0.018,
        z=0.004,
        color=(0.02, 0.32, 0.82),
        roughness=0.32,
    )
    for name, z in (("LabelTopStripe", 0.023), ("LabelBottomStripe", -0.031)):
        cylinder(
            name,
            radius=spec.radius_m + 0.0022,
            height=0.003,
            z=z,
            color=(0.04, 0.48, 0.88),
            roughness=0.30,
        )
    for index, z in enumerate((-0.078, -0.061, 0.044, 0.059)):
        cylinder(
            f"BodyRing{index + 1}",
            radius=spec.radius_m + 0.0012,
            height=0.0025,
            z=z,
            color=(0.67, 0.89, 0.97),
            roughness=0.10,
            opacity=0.38,
        )
    cylinder(
        "BaseRing",
        radius=spec.radius_m + 0.001,
        height=0.006,
        z=-0.105,
        color=(0.65, 0.86, 0.95),
        roughness=0.18,
        opacity=0.55,
    )

    shoulder = UsdGeom.Cone.Define(stage, f"{spec.prim_path}/Shoulder")
    shoulder.CreateAxisAttr("Z")
    shoulder.CreateRadiusAttr(spec.radius_m - 0.002)
    shoulder.CreateHeightAttr(0.032)
    shoulder.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.91, 0.98)])
    shoulder.CreateDisplayOpacityAttr([0.36])
    UsdGeom.Xformable(shoulder.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.083)
    )
    _bind_preview_material(
        stage,
        shoulder.GetPrim(),
        name="Material",
        color=(0.72, 0.91, 0.98),
        roughness=0.12,
        opacity=0.36,
    )

    cylinder(
        "Neck",
        radius=0.015,
        height=0.026,
        z=0.096,
        color=(0.70, 0.90, 0.97),
        roughness=0.15,
        opacity=0.42,
    )
    cylinder(
        "Cap",
        radius=0.019,
        height=0.020,
        z=0.104,
        color=(0.02, 0.24, 0.66),
        roughness=0.58,
    )
    for index, z in enumerate((0.097, 0.103, 0.109)):
        cylinder(
            f"CapRib{index + 1}",
            radius=0.0197,
            height=0.0018,
            z=z,
            color=(0.01, 0.18, 0.54),
            roughness=0.7,
        )
    for index in range(16):
        angle = 2.0 * math.pi * index / 16.0
        rib = UsdGeom.Cube.Define(
            stage,
            f"{spec.prim_path}/CapVerticalRib{index + 1}",
        )
        rib.CreateSizeAttr(1.0)
        rib.CreateDisplayColorAttr([Gf.Vec3f(0.01, 0.18, 0.54)])
        xform = UsdGeom.Xformable(rib.GetPrim())
        xform.AddTranslateOp().Set(
            Gf.Vec3d(
                0.0196 * math.cos(angle),
                0.0196 * math.sin(angle),
                0.104,
            )
        )
        xform.AddRotateZOp().Set(math.degrees(angle))
        xform.AddScaleOp().Set(Gf.Vec3f(0.0018, 0.0030, 0.018))
        _bind_preview_material(
            stage,
            rib.GetPrim(),
            name="Material",
            color=(0.01, 0.18, 0.54),
            roughness=0.72,
        )


class SensorRig:
    """First-person RGB-D, optional lidar, and an operator-only chase camera."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.head = Camera(
            prim_path="/World/LuxiSensors/HeadCamera",
            resolution=(args.sensor_width, args.sensor_height),
        )
        self.head.initialize()
        _set_camera_fov(self.head, 90.0)
        self.head.add_distance_to_image_plane_to_frame()
        self.observer: Camera | None = None
        if not args.disable_observer:
            self.observer = Camera(
                prim_path="/World/LuxiSensors/ObserverCamera",
                resolution=(args.observer_width, args.observer_height),
            )
            self.observer.initialize()
            _set_camera_fov(self.observer, 72.0)
        self.lidar: LidarRtx | None = None
        self.lidar_identity_verified = False
        self.lidar_identity_error: str | None = None
        self.lidar_metadata: dict[str, Any] | None = None
        self._lidar_stable_id_map: dict[int, str] = {}
        self._lidar_scan_synchronized = False
        self._lidar_scan_started_at = 0.0
        self._lidar_scan_points: list[np.ndarray[Any, Any]] = []
        self._lidar_scan_audits: list[LidarIdentityAudit] = []
        if not args.disable_lidar:
            self.lidar = LidarRtx(
                prim_path="/World/LuxiSensors/RtxLidar",
                name="luxi_rtx_lidar",
                position=np.asarray([0.0, 0.0, 1.35]),
                config_file_name=args.lidar_config,
                variant=args.lidar_variant,
                **{
                    "omni:sensor:Core:auxOutputType": "FULL",
                },
            )
            self.lidar.initialize()
            self.lidar.attach_annotator("GenericModelOutput")
            self.lidar.attach_annotator("StableIdMap")

    @staticmethod
    def _basis(yaw: float) -> tuple[np.ndarray[Any, Any], ...]:
        forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        right = np.asarray([math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
        down = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        return forward, right, down

    def update_poses(
        self,
        root_position: np.ndarray[Any, Any],
        root_quaternion: np.ndarray[Any, Any],
    ) -> None:
        yaw = _yaw_from_wxyz(root_quaternion)
        root = np.asarray(root_position, dtype=np.float64)
        forward, right, _down = self._basis(yaw)
        head_eye = root + np.asarray([0.0, 0.0, 0.55]) + 0.12 * forward
        head_target = head_eye + 3.0 * forward
        set_camera_view(
            eye=head_eye.tolist(),
            target=head_target.tolist(),
            camera_prim_path=self.head.prim_path,
        )
        if self.observer is not None:
            # Rear three-quarter view keeps the whole body visible while
            # staying close enough for room-scale interiors. This channel is
            # for the local operator only and is never published through DimOS.
            observer_eye = (
                root - 2.2 * forward + 0.85 * right + np.asarray([0.0, 0.0, 1.55])
            )
            observer_target = root + 0.30 * forward + np.asarray([0.0, 0.0, 0.45])
            set_camera_view(
                eye=observer_eye.tolist(),
                target=observer_target.tolist(),
                camera_prim_path=self.observer.prim_path,
            )

        if self.lidar is not None:
            # The sensor follows the floating base but stays outside the robot
            # prim subtree.  RTX stable IDs still identify any arms/torso/legs
            # hit by the 360-degree scan as descendants of /World/G1.
            sensor_position = root + np.asarray([0.0, 0.0, 0.55]) + 0.12 * forward
            self.lidar.set_world_pose(
                position=sensor_position,
                orientation=np.asarray(root_quaternion, dtype=np.float64),
            )

    @staticmethod
    def _valid_rgb(value: Any) -> np.ndarray[Any, np.dtype[np.uint8]] | None:
        if value is None:
            return None
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[2] < 3 or not frame.size:
            return None
        rgb = frame[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    @staticmethod
    def _valid_depth(value: Any) -> np.ndarray[Any, np.dtype[np.float32]] | None:
        if value is None:
            return None
        depth = np.asarray(value, dtype=np.float32)
        if depth.ndim != 2 or not depth.size:
            return None
        # Replicator normally uses +inf for a ray with no hit. Normalize any
        # backend-specific NaN/negative sentinel to the same explicit value.
        depth = np.where(np.isfinite(depth) & (depth >= 0.0), depth, np.inf)
        return np.ascontiguousarray(depth)

    def head_frame(
        self,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
        rgb = self._valid_rgb(self.head.get_rgba())
        depth = self._valid_depth(self.head.get_depth())
        if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
            return None
        intrinsics = np.asarray(self.head.get_intrinsics_matrix(), dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            return None
        return rgb, depth, intrinsics

    def observer_frame(self) -> np.ndarray[Any, np.dtype[np.uint8]] | None:
        if self.observer is None:
            return None
        return self._valid_rgb(self.observer.get_rgba())

    def lidar_frame(
        self,
        timestamp: float,
        *,
        carried_prim_paths: tuple[str, ...] = (),
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], LidarIdentityAudit, float] | None:
        """Return one complete world-frame scan with verified RTX identity."""

        if self.lidar is None:
            return None
        try:
            current = self.lidar.get_current_frame()
            gmo_buffer = current.get("GenericModelOutput")
            stable_buffer = current.get("StableIdMap")
            if gmo_buffer is None or stable_buffer is None:
                raise UnverifiedLidarIdentity("RTX identity annotators are not ready")
            gmo = get_gmo_data(gmo_buffer)
            count = int(gmo.numElements)
            if count < 1:
                raise UnverifiedLidarIdentity(
                    "RTX lidar returned no identity-bearing points"
                )

            x = np.asarray(gmo.x, dtype=np.float64).reshape(-1).copy()
            y = np.asarray(gmo.y, dtype=np.float64).reshape(-1).copy()
            z = np.asarray(gmo.z, dtype=np.float64).reshape(-1).copy()
            flags = np.asarray(gmo.flags, dtype=np.uint8).reshape(-1).copy()
            if not all(len(values) == count for values in (x, y, z, flags)):
                raise UnverifiedLidarIdentity(
                    "GMO point arrays have inconsistent lengths"
                )

            coords_type = int(gmo.elementsCoordsType)
            sensor_pose_position, sensor_pose_orientation = self.lidar.get_world_pose()
            self.lidar_metadata = {
                "coords_type": coords_type,
                "frame_of_reference": int(gmo.frameOfReference),
                "scan_complete": int(gmo.scanComplete),
                "frame_end_position": [
                    float(value) for value in np.asarray(gmo.frameEnd.posM).reshape(-1)
                ],
                "frame_end_orientation": [
                    float(value)
                    for value in np.asarray(gmo.frameEnd.orientation).reshape(-1)
                ],
                "sensor_position": [float(value) for value in sensor_pose_position],
                "sensor_orientation_wxyz": [
                    float(value) for value in sensor_pose_orientation
                ],
                "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
                "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
                "z_range": [float(np.nanmin(z)), float(np.nanmax(z))],
            }
            if coords_type == 1:  # CoordsType.SPHERICAL
                sensor_points = spherical_returns_to_cartesian(
                    azimuth_degrees=x,
                    elevation_degrees=y,
                    ranges_m=z,
                )
                ranges = z
            elif coords_type == 0:  # CoordsType.CARTESIAN
                sensor_points = np.ascontiguousarray(
                    np.column_stack((x, y, z)), dtype=np.float32
                )
                ranges = np.linalg.norm(sensor_points, axis=1)
            else:
                raise UnverifiedLidarIdentity("unsupported GMO coordinate type")

            valid = (
                ((flags & np.uint8(1 << 6)) != 0) & (ranges >= 0.10) & (ranges <= 100.0)
            )
            frame_of_reference = int(gmo.frameOfReference)
            if frame_of_reference == 0:  # FrameOfReference.SENSOR
                sensor_position, sensor_orientation = self.lidar.get_world_pose()
                world_points = transform_points_wxyz(
                    sensor_points,
                    position=np.asarray(sensor_position, dtype=np.float64),
                    quaternion_wxyz=np.asarray(sensor_orientation, dtype=np.float64),
                )
            elif frame_of_reference == 1:  # FrameOfReference.WORLD
                if coords_type != 0:
                    raise UnverifiedLidarIdentity(
                        "world-frame GMO unexpectedly uses spherical coordinates"
                    )
                world_points = sensor_points
            else:
                raise UnverifiedLidarIdentity("unsupported GMO frame of reference")

            stable_map = merge_stable_id_mapping(
                self._lidar_stable_id_map,
                LidarRtx.decode_stable_id_mapping(np.asarray(stable_buffer).tobytes()),
            )
            self._lidar_stable_id_map = stable_map
            object_ids = np.asarray(
                LidarRtx.get_object_ids(np.asarray(gmo.objId).copy())
            )
            if len(object_ids) != count:
                raise UnverifiedLidarIdentity(
                    "GMO point and stable object ID count mismatch"
                )
            valid_points = world_points[valid]
            valid_object_ids = object_ids[valid]
            points, valid_audit = filter_lidar_returns(
                valid_points,
                valid_object_ids,
                stable_map,
                robot_prim_path="/World/G1",
                carried_prim_paths=(
                    *carried_prim_paths,
                    "/World/LuxiSensors/RtxLidar",
                ),
            )
            audit = LidarIdentityAudit(
                raw_return_count=count,
                resolved_return_count=valid_audit.resolved_return_count,
                self_return_count=valid_audit.self_return_count,
                retained_return_count=valid_audit.retained_return_count,
                invalid_return_count=count - len(valid_points),
            )
            self.lidar_identity_verified = True
            self.lidar_identity_error = None
            scan_complete = bool(int(gmo.scanComplete))
            if not self._lidar_scan_synchronized:
                if scan_complete:
                    self._lidar_scan_synchronized = True
                    self._lidar_scan_started_at = timestamp
                    self._lidar_scan_points.clear()
                    self._lidar_scan_audits.clear()
                return None

            if not self._lidar_scan_points:
                self._lidar_scan_started_at = timestamp
            self._lidar_scan_points.append(points)
            self._lidar_scan_audits.append(audit)
            if not scan_complete:
                return None

            complete_points = np.ascontiguousarray(
                np.concatenate(self._lidar_scan_points, axis=0), dtype=np.float32
            )
            complete_audit = LidarIdentityAudit(
                raw_return_count=sum(
                    item.raw_return_count for item in self._lidar_scan_audits
                ),
                resolved_return_count=sum(
                    item.resolved_return_count for item in self._lidar_scan_audits
                ),
                self_return_count=sum(
                    item.self_return_count for item in self._lidar_scan_audits
                ),
                retained_return_count=len(complete_points),
                invalid_return_count=sum(
                    item.invalid_return_count for item in self._lidar_scan_audits
                ),
            )
            scan_started_at = self._lidar_scan_started_at
            self._lidar_scan_points.clear()
            self._lidar_scan_audits.clear()
            self._lidar_scan_started_at = timestamp
            return complete_points, complete_audit, scan_started_at
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            self.lidar_identity_verified = False
            self.lidar_identity_error = str(error)
            self._lidar_scan_synchronized = False
            self._lidar_scan_points.clear()
            self._lidar_scan_audits.clear()
            return None


class BridgeRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.paths = IsaacRuntimePaths(args.runtime_dir.expanduser().resolve())
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.state_sequence = 0
        self.frame_sequence = 0
        self.lidar_sequence = 0
        self.observer_sequence = 0
        self.entity_sequence = 0
        self.last_command_sequence = 0
        self.last_reset_token: str | None = None
        self.last_entity_token: str | None = None
        self.entity_contacts: dict[tuple[str, str], float] = {}
        self.entity_attachments: dict[str, str] = {}
        self.entity_grasp_postures: dict[str, dict[str, float]] = {}
        self.entity_motion: _EntityCartesianMotion | None = None
        self._contact_report_subscription: Any | None = None
        self.last_sensor_at = 0.0
        self.last_camera_at = 0.0
        self.last_observer_at = 0.0
        self.last_state_at = 0.0
        self.gait_command = IsaacGaitCommandAdapter()

        for stale in (
            self.paths.state,
            self.paths.camera,
            self.paths.observer,
            self.paths.entities,
            self.paths.manipulation_ack,
            self.paths.entity_ack,
            self.paths.person_command,
            self.paths.lidar,
            self.paths.lidar_proximity,
            self.paths.reset_ack,
        ):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

        asset = args.asset.expanduser().resolve()
        policy = args.policy.expanduser().resolve()
        policy_runtime = args.policy_runtime.expanduser().resolve()
        if not asset.is_file():
            raise FileNotFoundError(f"G1 USD is missing: {asset}")
        if not policy.is_file():
            raise FileNotFoundError(f"G1 policy is missing: {policy}")
        if not policy_runtime.is_dir():
            raise FileNotFoundError(f"ONNX runtime is missing: {policy_runtime}")

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 200.0,
            rendering_dt=1.0 / 60.0,
        )
        scene_asset = args.scene_asset or SCENE_ASSETS[args.scene]
        if args.scene == "brownstone" and not scene_asset:
            raise RuntimeError(
                "brownstone requires --scene-asset from the read-only local scene mount"
            )
        if scene_asset:
            print(f"Loading Isaac scene: {scene_asset}", flush=True)
            stage = omni.usd.get_context().get_stage()
            if args.scene == "brownstone":
                _load_brownstone_scene(stage, scene_asset)
            else:
                add_reference_to_stage(scene_asset, "/World/Environment")
                _wait_for_stage_assets()
        else:
            self.world.scene.add_default_ground_plane()

        self.start_position = tuple(
            args.start_position or SCENE_START_POSITIONS[args.scene]
        )
        configured_specs = list(args.entity_specs)
        if args.scene in {"skill_demo", "task_apartment"} and not any(
            spec.entity_id == WATER_BOTTLE.entity_id for spec in configured_specs
        ):
            configured_specs.append(WATER_BOTTLE)
        self.entity_specs = tuple(configured_specs)
        self.entities: dict[str, SingleRigidPrim] = {}
        if args.scene == "skill_demo":
            _add_skill_demo_exploration_room(omni.usd.get_context().get_stage())
        self.task_person_rig: _TaskPersonRig | None = None
        if args.scene == "skill_demo":
            self.task_person_rig = _TaskPersonRig(
                translate=_add_skill_demo_person(omni.usd.get_context().get_stage())
            )
        elif args.scene == "task_apartment":
            assert args.person_asset is not None
            self.task_person_rig = _add_task_apartment(
                omni.usd.get_context().get_stage(),
                person_asset=args.person_asset,
                person_texture=args.person_texture,
                person_cache=self.paths.root / "person-cache",
            )
        self.task_person_translate = (
            self.task_person_rig.translate if self.task_person_rig is not None else None
        )
        self.task_person_motion_started_at: float | None = None
        self.task_person_last_pose_at = 0.0
        self.task_person_heading_radians = math.pi
        self.task_person_heading_updated_at = time.monotonic()
        self.task_person_position = (
            TASK_APARTMENT_PERSON_START
            if args.scene == "task_apartment"
            else (2.4, 0.0, 0.0)
        )
        self.task_person_control_mode = "auto"
        self.task_person_manual_forward = 0.0
        self.task_person_manual_turn = 0.0
        self.task_person_manual_expires_at = 0.0
        self.task_person_last_command_sequence = 0
        self.task_person_last_update_at = time.monotonic()
        self.task_person_collision_blocked = False
        self.robot_path = "/World/G1"
        add_reference_to_stage(str(asset), self.robot_path)
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(self.robot_path)
        if not robot_prim.IsValid():
            raise RuntimeError("G1 USD did not compose at /World/G1")
        xformable = UsdGeom.Xformable(robot_prim)
        translate_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ),
            None,
        )
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*self.start_position))
        if args.lidar_acceptance_test:
            _add_lidar_acceptance_geometry(stage)
        manipulation_spawn = (
            _add_manipulation_acceptance_geometry(stage, self.start_position)
            if args.manipulation_acceptance_test
            else None
        )

        if self.entity_specs:
            UsdGeom.Xform.Define(stage, ENTITY_ROOT)
        for spec in self.entity_specs:
            if spec.shape != "cylinder":
                raise RuntimeError(f"unsupported Isaac entity shape: {spec.shape}")
            geometry = UsdGeom.Cylinder.Define(stage, spec.prim_path)
            geometry.CreateAxisAttr("Z")
            geometry.CreateRadiusAttr(spec.radius_m)
            geometry.CreateHeightAttr(spec.height_m)
            geometry.CreateDisplayColorAttr([Gf.Vec3f(*spec.color_rgb)])
            if spec.entity_id == WATER_BOTTLE.entity_id:
                _add_realistic_water_bottle_visuals(stage, geometry, spec)
            UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
            contact_api = PhysxSchema.PhysxContactReportAPI.Apply(geometry.GetPrim())
            contact_api.CreateThresholdAttr().Set(0.0)
            self.entities[spec.entity_id] = self.world.scene.add(
                SingleRigidPrim(
                    prim_path=spec.prim_path,
                    name=f"luxi_entity_{spec.entity_id}",
                    position=np.asarray(
                        (
                            manipulation_spawn
                            if manipulation_spawn is not None
                            and spec.entity_id == "water_bottle"
                            else self._entity_spawn_position(spec)
                        ),
                        dtype=np.float32,
                    ),
                    orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    mass=spec.mass_kg,
                    reset_xform_properties=True,
                )
            )

        self.robot = self.world.scene.add(
            SingleArticulation(
                prim_path=self.robot_path,
                name="luxi_isaac_g1",
                reset_xform_properties=False,
            )
        )
        sun = UsdLux.DistantLight.Define(stage, "/World/LuxiSun")
        sun.CreateIntensityAttr(3000.0)
        sun.CreateAngleAttr(0.5)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(
            Gf.Vec3f(45.0, 30.0, 20.0)
        )
        dome = UsdLux.DomeLight.Define(stage, "/World/LuxiDome")
        dome.CreateIntensityAttr(500.0)
        UsdGeom.Xform.Define(stage, "/World/LuxiSensors")

        for _ in range(5):
            SIMULATION_APP.update()
        self.world.reset()
        controller_type = _load_locomotion_controller(args.controller_root)
        self.controller = controller_type(
            articulation=self.robot,
            policy_path=policy,
            runtime_path=policy_runtime,
            start_position=self.start_position,
        )
        self.controller.start(enable_keyboard=False)
        self.joint_names = tuple(str(name) for name in self.robot.dof_names)
        self.body_names = tuple(
            str(name) for name in self.robot._articulation_view.body_names
        )
        self.hand_body_indices = {
            side: next(
                index
                for index, name in enumerate(self.body_names)
                if name == f"{side}_wrist_yaw_link"
            )
            for side in ("left", "right")
        }
        # Cartesian entity motion owns only the selected arm. The locomotion
        # policy retains the waist so a reach cannot destabilize the floating
        # base or turn an arm request into unintended walking.
        self.arm_joint_names = dict(ARM_JOINTS_BY_HAND)
        self.arm_joint_indices = {
            side: np.asarray(
                [self.joint_names.index(name) for name in self.arm_joint_names[side]],
                dtype=np.int32,
            )
            for side in ("left", "right")
        }
        self.joint_limits = np.asarray(
            self.robot._articulation_view.get_dof_limits()[0],
            dtype=np.float64,
        )
        if self.joint_limits.shape != (len(self.joint_names), 2):
            raise RuntimeError(
                "G1 articulation joint limits do not match the runtime DOF schema"
            )
        self.manipulation = IsaacManipulationJointController(
            joint_names=self.joint_names,
            joint_limits=self.joint_limits,
            neutral_positions=np.asarray(
                self.controller._joint_targets,
                dtype=np.float64,
            ),
            request_path=self.paths.manipulation_request,
            ack_path=self.paths.manipulation_ack,
        )
        self.controller.set_command_override(self.gait_command.reset())
        self.sensors = SensorRig(args)
        self.world.play()
        self._contact_report_subscription = (
            omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report
            )
        )
        self._write_entities(time.time())

    def request_stop(self, *_args: Any) -> None:
        self.stop_requested = True

    @staticmethod
    def _contact_hand(path: str) -> str | None:
        normalized = path.casefold()
        if not normalized.startswith("/world/g1/"):
            return None
        hand_tokens = ("wrist", "hand", "thumb", "index", "middle", "ring", "pinky")
        if not any(token in normalized for token in hand_tokens):
            return None
        if any(token in normalized for token in ("/right_", "/r_", "right")):
            return "right"
        if any(token in normalized for token in ("/left_", "/l_", "left")):
            return "left"
        return None

    def _on_contact_report(self, headers: Any, _data: Any) -> None:
        observed_at = time.time()
        for header in headers:
            if int(header.num_contact_data) < 1:
                continue
            collider_paths = (
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            )
            for spec in self.entity_specs:
                entity_index = next(
                    (
                        index
                        for index, path in enumerate(collider_paths)
                        if path == spec.prim_path
                        or path.startswith(f"{spec.prim_path}/")
                    ),
                    None,
                )
                if entity_index is None:
                    continue
                hand = self._contact_hand(collider_paths[1 - entity_index])
                if hand is not None:
                    self.entity_contacts[(spec.entity_id, hand)] = observed_at

    def _hand_position(self, hand: str) -> np.ndarray[Any, Any]:
        transforms = np.asarray(
            self.robot._articulation_view._physics_view.get_link_transforms()[0],
            dtype=np.float64,
        )
        return transforms[self.hand_body_indices[hand], :3].copy()

    def _start_entity_motion(
        self,
        *,
        token: str,
        action: str,
        entity_id: str,
        hand: str,
        target_hand_position: np.ndarray[Any, Any],
        target_entity_position: np.ndarray[Any, Any] | None = None,
        joint_targets: dict[str, float] | None = None,
        release_after_arrival: bool = False,
        timeout_s: float = 7.0,
    ) -> None:
        if self.entity_motion is not None:
            self._entity_response(
                token=token,
                action=action,
                ok=False,
                entity_id=entity_id,
                error="another Isaac entity motion is active",
            )
            return
        target = np.asarray(target_hand_position, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            self._entity_response(
                token=token,
                action=action,
                ok=False,
                entity_id=entity_id,
                error="Cartesian hand target is invalid",
            )
            return
        now = time.time()
        self.entity_motion = _EntityCartesianMotion(
            token=token,
            action=action,
            entity_id=entity_id,
            hand=hand,
            target_hand_position=target,
            target_entity_position=(
                None
                if target_entity_position is None
                else np.asarray(target_entity_position, dtype=np.float64)
            ),
            release_after_arrival=release_after_arrival,
            joint_targets=joint_targets,
            start_entity_position=np.asarray(
                self.entities[entity_id].get_world_pose()[0],
                dtype=np.float64,
            ),
            started_at=now,
            deadline=now + max(0.5, float(timeout_s)),
        )

    def _update_entity_motion(self) -> None:
        motion = self.entity_motion
        if motion is None:
            return
        now = time.time()
        if now > motion.deadline:
            self._entity_response(
                token=motion.token,
                action=motion.action,
                ok=False,
                entity_id=motion.entity_id,
                error="Isaac Cartesian entity motion timed out",
                hand=motion.hand,
            )
            self.entity_motion = None
            return
        current = self._hand_position(motion.hand)
        if motion.joint_targets is not None:
            self.manipulation.set_runtime_targets(
                motion.joint_targets,
                hold_s=30.0,
                now=now,
            )
            joint_state = self.robot.get_joints_state()
            if joint_state is None:
                raise RuntimeError(
                    "G1 joint state is unavailable during posture motion"
                )
            actual = np.asarray(joint_state.positions, dtype=np.float64)
            distance = max(
                abs(float(actual[self.joint_names.index(name)]) - float(target))
                for name, target in motion.joint_targets.items()
            )
            if now - motion.started_at >= 1.0 and distance <= 0.06:
                motion.stable_frames += 1
            else:
                motion.stable_frames = 0
        else:
            error = motion.target_hand_position - current
            distance = float(np.linalg.norm(error))
            if distance <= 0.015:
                motion.stable_frames += 1
            else:
                motion.stable_frames = 0
        if motion.stable_frames >= 5:
            entity = self.entities[motion.entity_id]
            if motion.action == "approach":
                joint_state = self.robot.get_joints_state()
                if joint_state is None:
                    raise RuntimeError(
                        "G1 joint state is unavailable while holding approach"
                    )
                actual = np.asarray(joint_state.positions, dtype=np.float64)
                self.manipulation.set_runtime_targets(
                    {
                        name: float(actual[index])
                        for name, index in zip(
                            self.arm_joint_names[motion.hand],
                            self.arm_joint_indices[motion.hand],
                            strict=True,
                        )
                    },
                    hold_s=10.0,
                    now=now,
                )
            if motion.release_after_arrival:
                self._release_entity(motion.entity_id)
            position, orientation = entity.get_world_pose()
            target_error = (
                None
                if motion.target_entity_position is None
                else float(
                    np.linalg.norm(
                        np.asarray(position, dtype=np.float64)
                        - motion.target_entity_position
                    )
                )
            )
            placement_ok = bool(
                not motion.release_after_arrival
                or target_error is not None
                and target_error <= 0.05
            )
            self._entity_response(
                token=motion.token,
                action=motion.action,
                ok=placement_ok,
                entity_id=motion.entity_id,
                error=(
                    None
                    if placement_ok
                    else "released entity missed the requested placement target"
                ),
                hand=motion.hand,
                pose=[
                    *[float(value) for value in position],
                    *[float(value) for value in orientation],
                ],
                attached=motion.entity_id in self.entity_attachments,
                contact=(
                    now - self.entity_contacts.get((motion.entity_id, motion.hand), 0.0)
                    <= 0.25
                ),
                hand_position=[float(value) for value in current],
                hand_target_error_m=(
                    distance if motion.joint_targets is None else None
                ),
                joint_target_error_rad=(
                    distance if motion.joint_targets is not None else None
                ),
                entity_target_error_m=target_error,
                placed=motion.release_after_arrival,
                gait_ownership_restored=motion.action == "carry",
            )
            self.entity_motion = None
            return

        if motion.joint_targets is not None:
            return

        jacobians = np.asarray(
            self.robot._articulation_view.get_jacobians()[0],
            dtype=np.float64,
        )
        indices = self.arm_joint_indices[motion.hand]
        position_jacobian = jacobians[
            self.hand_body_indices[motion.hand],
            :3,
        ][:, 6 + indices]
        delta = damped_cartesian_joint_delta(
            position_jacobian,
            error * 0.65,
            damping=0.10,
            max_delta_rad=0.06,
        )
        joint_state = self.robot.get_joints_state()
        if joint_state is None:
            raise RuntimeError("G1 joint state is unavailable during Cartesian motion")
        actual = np.asarray(joint_state.positions, dtype=np.float64)
        targets = {
            name: float(actual[index] + delta[offset])
            for offset, (name, index) in enumerate(
                zip(
                    self.arm_joint_names[motion.hand],
                    indices,
                    strict=True,
                )
            )
        }
        self.manipulation.set_runtime_targets(targets, hold_s=0.30, now=now)

    def _poll_command(self) -> None:
        payload = read_json(self.paths.command)
        sequence = payload.get("sequence") if payload else None
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            sequence = 0
        command = validated_velocity_command(payload)
        if command is None:
            command = (0.0, 0.0, 0.0, 0.8)
        if (
            self.task_person_translate is not None
            and self.task_person_motion_started_at is None
            and task_person_motion_should_start(command)
        ):
            self.task_person_motion_started_at = time.monotonic()
        self.last_command_sequence = max(self.last_command_sequence, sequence)
        self.controller.set_command_override(self.gait_command.step(command))

    def _poll_person_command(self) -> None:
        if self.args.scene != "task_apartment" or self.task_person_translate is None:
            return
        payload = read_json(self.paths.person_command, max_bytes=8_192)
        if not isinstance(payload, dict):
            return
        sequence = payload.get("sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= self.task_person_last_command_sequence
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("backend") != BACKEND_NAME
        ):
            return
        mode = payload.get("mode")
        if mode not in {"auto", "manual", "paused"}:
            return
        try:
            written_at = float(payload["written_at"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        wall_now = time.time()
        if not math.isfinite(written_at) or wall_now - written_at > 2.0:
            return
        self.task_person_last_command_sequence = sequence
        self.task_person_control_mode = mode
        self.task_person_collision_blocked = False
        if mode == "auto":
            self.task_person_motion_started_at = time.monotonic()
            self.task_person_manual_forward = 0.0
            self.task_person_manual_turn = 0.0
            return
        if mode == "paused":
            self.task_person_manual_forward = 0.0
            self.task_person_manual_turn = 0.0
            return
        try:
            forward = float(payload.get("forward", 0.0))
            turn = float(payload.get("turn", 0.0))
            expires_at = float(payload["expires_at"])
        except (KeyError, TypeError, ValueError, OverflowError):
            self.task_person_manual_forward = 0.0
            self.task_person_manual_turn = 0.0
            return
        if (
            not all(math.isfinite(value) for value in (forward, turn, expires_at))
            or not -1.0 <= forward <= 1.0
            or not -1.0 <= turn <= 1.0
            or expires_at <= wall_now
        ):
            self.task_person_manual_forward = 0.0
            self.task_person_manual_turn = 0.0
            return
        self.task_person_manual_forward = forward
        self.task_person_manual_turn = turn
        self.task_person_manual_expires_at = time.monotonic() + min(
            0.35, expires_at - wall_now
        )

    def _poll_reset(self) -> None:
        payload = read_json(self.paths.reset_request, max_bytes=8_192)
        if not payload or payload.get("action") != "reset_pose":
            return
        token = payload.get("token")
        if not isinstance(token, str) or not token or token == self.last_reset_token:
            return
        self.last_reset_token = token
        self.gait_command.reset()
        self.controller.reset()
        self.manipulation.reset()
        self.task_person_motion_started_at = None
        self.task_person_heading_radians = math.pi
        self.task_person_heading_updated_at = time.monotonic()
        self.task_person_position = (
            TASK_APARTMENT_PERSON_START
            if self.args.scene == "task_apartment"
            else (2.4, 0.0, 0.0)
        )
        self.task_person_control_mode = "auto"
        self.task_person_manual_forward = 0.0
        self.task_person_manual_turn = 0.0
        self.task_person_manual_expires_at = 0.0
        self.task_person_last_update_at = time.monotonic()
        self.task_person_collision_blocked = False
        self._update_task_person()
        self._reset_entities()
        position, orientation = self.robot.get_world_pose()
        atomic_write_json(
            self.paths.reset_ack,
            {
                "action": "reset_pose",
                "token": token,
                "applied_at": time.time(),
                "position": [float(value) for value in position],
                "quaternion_wxyz": [float(value) for value in orientation],
                "backend": BACKEND_NAME,
            },
        )

    def _reset_entities(self) -> None:
        self.entity_motion = None
        for spec in self.entity_specs:
            self._reset_entity(spec.entity_id)

    def _entity_spawn_position(
        self, spec: IsaacEntitySpec
    ) -> tuple[float, float, float]:
        if self.args.manipulation_acceptance_test and spec.entity_id == "water_bottle":
            # The bottle starts clear of the robot; approach moves the wrist
            # into the distal hand's grasp volume before contact is accepted.
            return (
                float(self.start_position[0]) + 0.24,
                float(self.start_position[1]) - 0.22,
                0.73,
            )
        if self.args.scene == "skill_demo" and spec.entity_id == "water_bottle":
            return (
                float(self.start_position[0]) - 2.0,
                float(self.start_position[1]) - 0.35,
                0.12,
            )
        if self.args.scene == "task_apartment" and spec.entity_id == "water_bottle":
            return TASK_APARTMENT_BOTTLE_POSITION
        return spec.spawn_position(self.start_position)

    def _reset_entity(self, entity_id: str) -> None:
        if self.entity_motion is not None and self.entity_motion.entity_id == entity_id:
            self.entity_motion = None
        self._release_entity(entity_id)
        zero = np.zeros(3, dtype=np.float32)
        spec = next(
            (item for item in self.entity_specs if item.entity_id == entity_id),
            None,
        )
        if spec is None:
            raise KeyError(entity_id)
        entity = self.entities[entity_id]
        entity.set_world_pose(
            position=np.asarray(
                self._entity_spawn_position(spec),
                dtype=np.float32,
            ),
            orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        entity.set_linear_velocity(zero)
        entity.set_angular_velocity(zero)

    def _release_entity(self, entity_id: str) -> None:
        joint_path = f"/World/LuxiAttachments/{entity_id}"
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(joint_path).IsValid():
            stage.RemovePrim(joint_path)
        self.entity_attachments.pop(entity_id, None)
        self.entity_grasp_postures.pop(entity_id, None)
        entity_prim = stage.GetPrimAtPath(
            next(
                spec.prim_path
                for spec in self.entity_specs
                if spec.entity_id == entity_id
            )
        )
        if entity_prim.IsValid():
            filtered = UsdPhysics.FilteredPairsAPI.Get(stage, entity_prim.GetPath())
            if filtered:
                filtered.GetFilteredPairsRel().RemoveTarget(Sdf.Path("/World/G1"))

    def _attach_entity(self, entity_id: str, hand: str) -> None:
        stage = omni.usd.get_context().get_stage()
        hand_tokens = (
            f"{hand}_wrist_yaw_link",
            f"{hand}_hand",
        )
        hand_prim = next(
            (
                prim
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith("/World/G1/")
                and any(token in prim.GetName().casefold() for token in hand_tokens)
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ),
            None,
        )
        if hand_prim is None:
            raise RuntimeError(f"Isaac {hand} hand rigid body is unavailable")
        attachment_root = "/World/LuxiAttachments"
        if not stage.GetPrimAtPath(attachment_root).IsValid():
            UsdGeom.Xform.Define(stage, attachment_root)
        joint_path = f"{attachment_root}/{entity_id}"
        if stage.GetPrimAtPath(joint_path).IsValid():
            stage.RemovePrim(joint_path)
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([hand_prim.GetPath()])
        entity_path = next(
            spec.prim_path for spec in self.entity_specs if spec.entity_id == entity_id
        )
        joint.CreateBody1Rel().SetTargets([entity_path])
        # Keep the current relative pose instead of snapping the object to the
        # wrist origin when the contact-gated constraint is created.
        hand_world = omni.usd.get_world_transform_matrix(hand_prim)
        entity_world = omni.usd.get_world_transform_matrix(
            stage.GetPrimAtPath(entity_path)
        )
        local = entity_world * hand_world.GetInverse()
        local_translation = local.ExtractTranslation()
        local_rotation = local.ExtractRotationQuat()
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_translation))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(local_rotation.GetReal()),
                Gf.Vec3f(*local_rotation.GetImaginary()),
            )
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        entity_prim = stage.GetPrimAtPath(entity_path)
        filtered = UsdPhysics.FilteredPairsAPI.Apply(entity_prim)
        filtered.CreateFilteredPairsRel().AddTarget(Sdf.Path("/World/G1"))
        self.entity_attachments[entity_id] = hand

    def _entity_response(
        self,
        *,
        token: str,
        action: str,
        ok: bool,
        entity_id: str,
        error: str | None = None,
        **evidence: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "backend": BACKEND_NAME,
            "token": token,
            "action": action,
            "ok": bool(ok),
            "entity_id": entity_id,
            "applied_at": time.time(),
            **evidence,
        }
        if error:
            payload["error"] = error
        atomic_write_json(self.paths.entity_ack, payload)

    def _poll_entity_request(self) -> None:
        payload = read_json(self.paths.entity_request, max_bytes=16_384)
        if not payload:
            return
        token = payload.get("token")
        if not isinstance(token, str) or not token or token == self.last_entity_token:
            return
        self.last_entity_token = token
        action = payload.get("action")
        arguments = payload.get("arguments")
        entity_id = arguments.get("entity_id") if isinstance(arguments, dict) else None
        requested_at = payload.get("requested_at")
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("backend") != BACKEND_NAME
            or not isinstance(action, str)
            or not isinstance(arguments, dict)
            or not isinstance(entity_id, str)
            or not isinstance(requested_at, (int, float))
            or isinstance(requested_at, bool)
            or time.time() - float(requested_at) < -0.25
            or time.time() - float(requested_at) > 10.0
        ):
            self._entity_response(
                token=token,
                action=str(action),
                ok=False,
                entity_id=str(entity_id),
                error="invalid or stale entity request",
            )
            return
        entity = self.entities.get(entity_id)
        if entity is None:
            self._entity_response(
                token=token,
                action=action,
                ok=False,
                entity_id=entity_id,
                error="unknown or unavailable entity",
            )
            return
        position, orientation = entity.get_world_pose()
        common = {
            "pose": [
                *[float(value) for value in position],
                *[float(value) for value in orientation],
            ],
            "attached": False,
            "contact": False,
            "resting_height_m": float(
                self._entity_spawn_position(
                    next(
                        spec
                        for spec in self.entity_specs
                        if spec.entity_id == entity_id
                    )
                )[2]
            ),
        }
        hand = arguments.get("hand")
        normalized_hand = (
            str(hand).casefold().removesuffix("_hand")
            if isinstance(hand, str)
            else None
        )
        fresh_contact = bool(
            normalized_hand in {"left", "right"}
            and time.time()
            - self.entity_contacts.get((entity_id, normalized_hand), 0.0)
            <= 0.25
        )
        common["attached"] = entity_id in self.entity_attachments
        common["contact"] = fresh_contact
        if normalized_hand:
            common["hand"] = normalized_hand
        if action == "status":
            self._entity_response(
                token=token,
                action=action,
                ok=True,
                entity_id=entity_id,
                **common,
            )
        elif action == "reset":
            self._reset_entity(entity_id)
            position, orientation = entity.get_world_pose()
            self._entity_response(
                token=token,
                action=action,
                ok=True,
                entity_id=entity_id,
                pose=[
                    *[float(value) for value in position],
                    *[float(value) for value in orientation],
                ],
                attached=False,
                contact=False,
            )
        elif action == "contact":
            self._entity_response(
                token=token,
                action=action,
                ok=fresh_contact,
                entity_id=entity_id,
                error=(
                    None
                    if fresh_contact
                    else "fresh PhysX hand/entity contact is required"
                ),
                **common,
            )
        elif action == "approach":
            if normalized_hand not in {"left", "right"}:
                self._entity_response(
                    token=token,
                    action=action,
                    ok=False,
                    entity_id=entity_id,
                    error="hand must be left or right",
                    **common,
                )
            else:
                robot_position, robot_orientation = self.robot.get_world_pose()
                yaw = _yaw_from_wxyz(np.asarray(robot_orientation, dtype=np.float64))
                forward = np.asarray(
                    [math.cos(yaw), math.sin(yaw), 0.0],
                    dtype=np.float64,
                )
                # The wrist origin sits behind the Inspire Hand fingers. A
                # 9 cm standoff brings the distal collision bodies to the
                # bottle without driving the wrist through it.
                target = np.asarray(position, dtype=np.float64) - 0.09 * forward
                self._start_entity_motion(
                    token=token,
                    action=action,
                    entity_id=entity_id,
                    hand=normalized_hand,
                    target_hand_position=target,
                )
        elif action == "grasp":
            evidence_timestamp = arguments.get("evidence_timestamp")
            evidence_fresh = bool(
                isinstance(evidence_timestamp, (int, float))
                and not isinstance(evidence_timestamp, bool)
                and math.isfinite(float(evidence_timestamp))
                and 0.0 <= time.time() - float(evidence_timestamp) <= 1.5
            )
            if not evidence_fresh:
                self._entity_response(
                    token=token,
                    action=action,
                    ok=False,
                    entity_id=entity_id,
                    error="fresh aligned RGB-D evidence is required",
                    **common,
                )
            elif not fresh_contact or normalized_hand not in {"left", "right"}:
                self._entity_response(
                    token=token,
                    action=action,
                    ok=False,
                    entity_id=entity_id,
                    error="fresh PhysX hand/entity contact is required",
                    **common,
                )
            else:
                joint_state = self.robot.get_joints_state()
                if joint_state is None:
                    raise RuntimeError("G1 joint state is unavailable during grasp")
                actual = np.asarray(joint_state.positions, dtype=np.float64)
                self.entity_grasp_postures[entity_id] = {
                    name: float(actual[index])
                    for name, index in zip(
                        self.arm_joint_names[normalized_hand],
                        self.arm_joint_indices[normalized_hand],
                        strict=True,
                    )
                }
                self._attach_entity(entity_id, normalized_hand)
                common["attached"] = True
                self._entity_response(
                    token=token,
                    action=action,
                    ok=True,
                    entity_id=entity_id,
                    grasped=True,
                    contact_confirmed=True,
                    **common,
                )
        elif action == "carry":
            if self.entity_attachments.get(entity_id) != normalized_hand:
                self._entity_response(
                    token=token,
                    action=action,
                    ok=False,
                    entity_id=entity_id,
                    error="entity is not attached to the requested hand",
                    **common,
                )
            else:
                target_pose = arguments.get("target_pose")
                if target_pose is None:
                    grasp_posture = self.entity_grasp_postures.get(entity_id)
                    if grasp_posture is None:
                        self._entity_response(
                            token=token,
                            action=action,
                            ok=False,
                            entity_id=entity_id,
                            error="verified Isaac grasp posture is unavailable",
                            **common,
                        )
                        return
                    shoulder = f"{normalized_hand}_shoulder_pitch_joint"
                    index = self.joint_names.index(shoulder)
                    joint_targets = {
                        shoulder: float(
                            np.clip(
                                grasp_posture[shoulder] + 0.20,
                                self.joint_limits[index, 0],
                                self.joint_limits[index, 1],
                            )
                        )
                    }
                    target_hand = self._hand_position(normalized_hand)
                    target_entity = None
                elif (
                    isinstance(target_pose, list)
                    and len(target_pose) in {3, 7}
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        for value in target_pose
                    )
                ):
                    target_entity = np.asarray(target_pose[:3], dtype=np.float64)
                    target_hand = (
                        self._hand_position(normalized_hand)
                        + target_entity
                        - np.asarray(position, dtype=np.float64)
                    )
                    joint_targets = None
                else:
                    self._entity_response(
                        token=token,
                        action=action,
                        ok=False,
                        entity_id=entity_id,
                        error="target_pose must contain 3 or 7 finite values",
                        **common,
                    )
                    return
                self._start_entity_motion(
                    token=token,
                    action=action,
                    entity_id=entity_id,
                    hand=normalized_hand,
                    target_hand_position=target_hand,
                    target_entity_position=target_entity,
                    joint_targets=joint_targets,
                )
        elif action == "place":
            target = arguments.get("target_position")
            if self.entity_attachments.get(entity_id) != normalized_hand:
                self._entity_response(
                    token=token,
                    action=action,
                    ok=False,
                    entity_id=entity_id,
                    error="entity is not attached to the requested hand",
                    **common,
                )
            elif (
                not isinstance(target, list)
                or len(target) != 3
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in target
                )
            ):
                self._entity_response(
                    token=token,
                    action=action,
                    ok=False,
                    entity_id=entity_id,
                    error="target_position must contain 3 finite values",
                    **common,
                )
            else:
                target_entity = np.asarray(target, dtype=np.float64)
                grasp_posture = self.entity_grasp_postures.get(entity_id)
                if grasp_posture is None:
                    self._entity_response(
                        token=token,
                        action=action,
                        ok=False,
                        entity_id=entity_id,
                        error="verified Isaac grasp posture is unavailable",
                        **common,
                    )
                    return
                self._start_entity_motion(
                    token=token,
                    action=action,
                    entity_id=entity_id,
                    hand=normalized_hand,
                    target_hand_position=self._hand_position(normalized_hand),
                    target_entity_position=target_entity,
                    joint_targets=dict(grasp_posture),
                    release_after_arrival=True,
                )
        elif action == "release":
            if self.entity_attachments.get(entity_id) != normalized_hand:
                self._entity_response(
                    token=token,
                    action=action,
                    ok=False,
                    entity_id=entity_id,
                    error="entity is not attached to the requested hand",
                    **common,
                )
            else:
                self._release_entity(entity_id)
                common["attached"] = False
                self._entity_response(
                    token=token,
                    action=action,
                    ok=True,
                    entity_id=entity_id,
                    **common,
                )
        else:
            self._entity_response(
                token=token,
                action=action,
                ok=False,
                entity_id=entity_id,
                error="Isaac entity operation is not implemented",
                **common,
            )

    def _write_entities(self, timestamp: float) -> None:
        self.entity_sequence += 1
        entities: list[dict[str, Any]] = []
        for spec in self.entity_specs:
            entity = self.entities[spec.entity_id]
            position, orientation = entity.get_world_pose()
            entities.append(
                {
                    "entity_id": spec.entity_id,
                    "prim_path": spec.prim_path,
                    "shape": spec.shape,
                    "dynamic": True,
                    "graspable": bool(spec.graspable),
                    "position": [float(value) for value in position],
                    "quaternion_wxyz": [float(value) for value in orientation],
                    "linear_velocity_world": [
                        float(value) for value in entity.get_linear_velocity()
                    ],
                    "angular_velocity_world": [
                        float(value) for value in entity.get_angular_velocity()
                    ],
                    "attached_to": self.entity_attachments.get(spec.entity_id),
                }
            )
        atomic_write_json(
            self.paths.entities,
            {
                "schema_version": SCHEMA_VERSION,
                "backend": BACKEND_NAME,
                "sequence": self.entity_sequence,
                "written_at": timestamp,
                "entities": entities,
            },
        )

    def _write_state(self, timestamp: float, *, force: bool = False) -> None:
        if not force and timestamp - self.last_state_at < 0.02:
            return
        self.last_state_at = timestamp
        position, orientation = self.robot.get_world_pose()
        linear_velocity = self.robot.get_linear_velocity()
        angular_velocity = self.robot.get_angular_velocity()
        joint_state = self.robot.get_joints_state()
        if joint_state is None:
            raise RuntimeError("G1 joint state is unavailable")
        link_transforms = np.asarray(
            self.robot._articulation_view._physics_view.get_link_transforms()[0],
            dtype=np.float64,
        )
        jacobians = np.asarray(
            self.robot._articulation_view.get_jacobians()[0],
            dtype=np.float64,
        )
        hand_poses = {
            side: {
                "body_name": self.body_names[index],
                "position": [float(value) for value in link_transforms[index, :3]],
                # Tensor API uses scalar-last XYZW while the Luxi protocol
                # consistently publishes scalar-first WXYZ.
                "quaternion_wxyz": [
                    float(link_transforms[index, 6]),
                    float(link_transforms[index, 3]),
                    float(link_transforms[index, 4]),
                    float(link_transforms[index, 5]),
                ],
            }
            for side, index in self.hand_body_indices.items()
        }
        self.state_sequence += 1
        atomic_write_json(
            self.paths.state,
            {
                "schema_version": SCHEMA_VERSION,
                "backend": BACKEND_NAME,
                "ready": (
                    self.frame_sequence > 0 and timestamp - self.last_camera_at <= 1.5
                ),
                "sequence": self.state_sequence,
                "written_at": timestamp,
                "scene": self.args.scene,
                "lidar_acceptance_test": bool(self.args.lidar_acceptance_test),
                "manipulation_acceptance_test": bool(
                    self.args.manipulation_acceptance_test
                ),
                "pose": {
                    "position": [float(value) for value in position],
                    "quaternion_wxyz": [float(value) for value in orientation],
                },
                "linear_velocity_world": [float(value) for value in linear_velocity],
                "angular_velocity_world": [float(value) for value in angular_velocity],
                "command_sequence": self.last_command_sequence,
                "camera_sequence": self.frame_sequence,
                "observer_sequence": self.observer_sequence,
                "observer_enabled": self.sensors.observer is not None,
                "entities_sequence": self.entity_sequence,
                "entity_ids": [spec.entity_id for spec in self.entity_specs],
                "person_control": (
                    {
                        "available": True,
                        "mode": self.task_person_control_mode,
                        "position": [float(value) for value in self.task_person_position],
                        "heading_radians": float(
                            math.atan2(
                                math.sin(self.task_person_heading_radians - math.pi),
                                math.cos(self.task_person_heading_radians - math.pi),
                            )
                        ),
                        "moving": bool(
                            self.task_person_control_mode == "manual"
                            and time.monotonic() <= self.task_person_manual_expires_at
                            and (
                                abs(self.task_person_manual_forward) > 1e-9
                                or abs(self.task_person_manual_turn) > 1e-9
                            )
                        ),
                        "collision_blocked": self.task_person_collision_blocked,
                        "command_sequence": self.task_person_last_command_sequence,
                    }
                    if self.args.scene == "task_apartment"
                    and self.task_person_translate is not None
                    else {"available": False}
                ),
                "lidar_sequence": self.lidar_sequence,
                "lidar_enabled": self.sensors.lidar is not None,
                "lidar_identity_verified": self.sensors.lidar_identity_verified,
                "lidar_identity_error": self.sensors.lidar_identity_error,
                "lidar_metadata": self.sensors.lidar_metadata,
                "controller": "unitree_onnx_joint_position_policy",
                "policy_shape": {"input": [1, 910], "output": [1, 12]},
                "joint_names": list(self.joint_names),
                "joint_limits_rad": self.joint_limits.tolist(),
                "hand_poses": hand_poses,
                "articulation_diagnostics": {
                    "body_names": list(self.body_names),
                    "jacobian_shape": list(jacobians.shape),
                },
                "entity_motion": (
                    None
                    if self.entity_motion is None
                    else {
                        "action": self.entity_motion.action,
                        "entity_id": self.entity_motion.entity_id,
                        "hand": self.entity_motion.hand,
                        "started_at": self.entity_motion.started_at,
                        "deadline": self.entity_motion.deadline,
                    }
                ),
                "manipulation": self.manipulation.state(
                    np.asarray(joint_state.positions, dtype=np.float64)
                ),
                "physics_hz": 200,
                "policy_hz": 50,
            },
        )

    def _write_sensors(self, timestamp: float) -> bool:
        position, orientation = self.robot.get_world_pose()
        self.sensors.update_poses(
            np.asarray(position, dtype=np.float64),
            np.asarray(orientation, dtype=np.float64),
        )
        # Refresh RTX/camera outputs without advancing physics. This preserves
        # the 200/50 Hz locomotion ratio while allowing the 10 Hz rotary lidar
        # to complete a full scan inside the 1.25 s freshness gate.
        self.world.render()
        observer_interval = 1.0 / float(self.args.observer_hz)
        if (
            self.sensors.observer is not None
            and timestamp - self.last_observer_at >= observer_interval
        ):
            observer = self.sensors.observer_frame()
            if observer is not None:
                self.observer_sequence += 1
                atomic_write_npz(
                    self.paths.observer,
                    timestamp=np.asarray(timestamp, dtype=np.float64),
                    sequence=np.asarray(self.observer_sequence, dtype=np.int64),
                    rgb=observer,
                )
                self.last_observer_at = timestamp
        head = self.sensors.head_frame()
        if head is None:
            return False
        rgb, depth, intrinsics = head
        self.frame_sequence += 1
        atomic_write_npz(
            self.paths.camera,
            timestamp=np.asarray(timestamp, dtype=np.float64),
            sequence=np.asarray(self.frame_sequence, dtype=np.int64),
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            position=np.asarray(position, dtype=np.float64),
            quaternion_wxyz=np.asarray(orientation, dtype=np.float64),
        )
        self.last_camera_at = timestamp
        self._write_entities(timestamp)
        carried_prim_paths = tuple(
            spec.prim_path
            for spec in self.entity_specs
            if spec.entity_id in self.entity_attachments
        )
        lidar_frame = self.sensors.lidar_frame(
            timestamp,
            carried_prim_paths=carried_prim_paths,
        )
        if lidar_frame is not None:
            points, audit, scan_started_at = lidar_frame
            self.lidar_sequence += 1
            atomic_write_npz(
                self.paths.lidar,
                timestamp=np.asarray(timestamp, dtype=np.float64),
                scan_started_at=np.asarray(scan_started_at, dtype=np.float64),
                sequence=np.asarray(self.lidar_sequence, dtype=np.int64),
                points=points,
                frame_id=np.asarray("world"),
                identity_verified=np.asarray(audit.identity_verified),
                identity_source=np.asarray(audit.identity_source),
                robot_prim_path=np.asarray("/World/G1"),
                carried_prim_paths=np.asarray(carried_prim_paths),
                raw_return_count=np.asarray(audit.raw_return_count, dtype=np.int64),
                resolved_return_count=np.asarray(
                    audit.resolved_return_count, dtype=np.int64
                ),
                self_return_count=np.asarray(audit.self_return_count, dtype=np.int64),
                retained_return_count=np.asarray(
                    audit.retained_return_count, dtype=np.int64
                ),
                invalid_return_count=np.asarray(
                    audit.invalid_return_count, dtype=np.int64
                ),
            )
            atomic_write_json(
                self.paths.lidar_proximity,
                summarize_lidar_proximity(
                    points,
                    frame_timestamp=timestamp,
                    scan_started_at=scan_started_at,
                    pose_timestamp=timestamp,
                    position=position,
                    quaternion_wxyz=orientation,
                    sequence=self.lidar_sequence,
                    audit=audit,
                    written_at=timestamp,
                ),
            )
        elif not self.sensors.lidar_identity_verified:
            try:
                self.paths.lidar.unlink()
            except FileNotFoundError:
                pass
            try:
                self.paths.lidar_proximity.unlink()
            except FileNotFoundError:
                pass
        return True

    def _update_task_person(self) -> None:
        if self.task_person_translate is None:
            return
        now = time.monotonic()
        update_dt = min(0.05, max(0.0, now - self.task_person_last_update_at))
        self.task_person_last_update_at = now
        if self.args.scene == "task_apartment":
            moving = False
            desired_heading = self.task_person_heading_radians
            if self.task_person_control_mode == "manual":
                if now > self.task_person_manual_expires_at:
                    self.task_person_manual_forward = 0.0
                    self.task_person_manual_turn = 0.0
                self.task_person_heading_radians = math.atan2(
                    math.sin(
                        self.task_person_heading_radians
                        + self.task_person_manual_turn
                        * PERSON_MANUAL_TURN_RATE_RPS
                        * update_dt
                    ),
                    math.cos(
                        self.task_person_heading_radians
                        + self.task_person_manual_turn
                        * PERSON_MANUAL_TURN_RATE_RPS
                        * update_dt
                    ),
                )
                distance = (
                    self.task_person_manual_forward
                    * PERSON_MANUAL_SPEED_MPS
                    * update_dt
                )
                actor_heading = self.task_person_heading_radians - math.pi
                candidate = (
                    self.task_person_position[0] + math.cos(actor_heading) * distance,
                    self.task_person_position[1] + math.sin(actor_heading) * distance,
                    self.task_person_position[2],
                )
                self.task_person_position, blocked = person_swept_position(
                    self.task_person_position, candidate
                )
                self.task_person_collision_blocked = blocked
                moving = (
                    abs(distance) > 1e-9
                    or abs(self.task_person_manual_turn) > 1e-9
                ) and not blocked
            elif self.task_person_control_mode == "auto":
                if self.task_person_motion_started_at is not None:
                    target = PERSON_ROUTE[-1]
                    dx = target[0] - self.task_person_position[0]
                    dy = target[1] - self.task_person_position[1]
                    remaining = math.hypot(dx, dy)
                    if remaining > 1e-6:
                        route_heading = math.atan2(dy, dx)
                        desired_heading = route_heading + math.pi
                        travel = min(PERSON_SPEED_MPS * update_dt, remaining)
                        candidate = (
                            self.task_person_position[0] + math.cos(route_heading) * travel,
                            self.task_person_position[1] + math.sin(route_heading) * travel,
                            self.task_person_position[2],
                        )
                        self.task_person_position, blocked = person_swept_position(
                            self.task_person_position, candidate
                        )
                        self.task_person_collision_blocked = blocked
                        moving = travel > 1e-9 and not blocked
            self.task_person_translate.Set(Gf.Vec3d(*self.task_person_position))
            if self.task_person_rig is not None and self.task_person_rig.heading is not None:
                if self.task_person_control_mode != "manual":
                    self.task_person_heading_radians = slew_person_heading(
                        self.task_person_heading_radians,
                        desired_heading,
                        max_delta_radians=PERSON_TURN_RATE_RPS * update_dt,
                    )
                self.task_person_rig.heading.Set(
                    math.degrees(self.task_person_heading_radians)
                )
            if now - self.task_person_last_pose_at >= 1.0 / 10.0:
                if self.task_person_rig is not None:
                    _apply_task_person_pose(
                        self.task_person_rig,
                        elapsed_s=now,
                        moving=moving,
                    )
                self.task_person_last_pose_at = now
            return
        elapsed = (
            0.0
            if self.task_person_motion_started_at is None
            else max(0.0, time.monotonic() - self.task_person_motion_started_at)
        )
        route_state = (
            task_apartment_person_route_state(elapsed)
            if self.args.scene == "task_apartment"
            else None
        )
        position = (
            route_state.position
            if route_state is not None
            else (
                2.4 + min(2.0, 0.06 * elapsed),
                0.30 * math.sin(0.35 * elapsed),
                0.0,
            )
        )
        self.task_person_translate.Set(Gf.Vec3d(*position))
        now = time.monotonic()
        if (
            route_state is not None
            and self.task_person_rig is not None
            and self.task_person_rig.heading is not None
        ):
            heading_dt = max(0.0, now - self.task_person_heading_updated_at)
            self.task_person_heading_radians = slew_person_heading(
                self.task_person_heading_radians,
                route_state.heading_radians + math.pi,
                max_delta_radians=PERSON_TURN_RATE_RPS * heading_dt,
            )
            self.task_person_rig.heading.Set(
                math.degrees(self.task_person_heading_radians)
            )
            self.task_person_heading_updated_at = now
        if (
            self.task_person_rig is not None
            and now - self.task_person_last_pose_at >= 1.0 / 10.0
        ):
            actor_moving = self.task_person_motion_started_at is not None and (
                route_state is None or route_state.moving
            )
            _apply_task_person_pose(
                self.task_person_rig,
                elapsed_s=elapsed,
                moving=actor_moving,
            )
            self.task_person_last_pose_at = now

    def run(self) -> None:
        for _ in range(8):
            position, orientation = self.robot.get_world_pose()
            self.sensors.update_poses(
                np.asarray(position, dtype=np.float64),
                np.asarray(orientation, dtype=np.float64),
            )
            self.world.step(render=True)

        step = 0
        ready_printed = False
        sensor_interval = 1.0 / float(self.args.sensor_hz)
        physics_interval = 1.0 / 200.0
        next_physics_step = time.monotonic()
        while (
            not self.stop_requested
            and SIMULATION_APP.is_running()
            and (self.args.max_steps == 0 or step < self.args.max_steps)
        ):
            self._update_task_person()
            if step % 4 == 0:
                self._poll_command()
                self._poll_person_command()
                self._poll_reset()
                self._poll_entity_request()
                self.controller.update_policy()
                self.manipulation.poll()
                self._update_entity_motion()
                self.manipulation.update(dt=0.02)
            self.manipulation.overlay(self.controller._joint_targets)
            self.controller.apply_action()
            # Sensor capture renders without stepping physics. Other iterations
            # advance exactly one physics step, preserving the 200/50 Hz
            # locomotion contract independently of the sensor rate.
            now = time.time()
            if now - self.last_sensor_at >= sensor_interval:
                captured = self._write_sensors(now)
                self.last_sensor_at = now
                if captured and not ready_printed:
                    self._write_state(now, force=True)
                    print("ISAAC_G1_BRIDGE_READY", flush=True)
                    ready_printed = True
            else:
                self.world.step(render=False)
            if step % 4 == 0:
                self._write_state(time.time())
            step += 1
            next_physics_step += physics_interval
            delay = next_physics_step - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -0.25:
                # Rendering may temporarily run slower than real time. Never
                # catch up by executing a burst of unpaced physics steps.
                next_physics_step = time.monotonic()

    def mark_stopped(self) -> None:
        self.state_sequence += 1
        if self.stop_requested:
            reason = "signal"
        elif self.args.max_steps > 0:
            reason = "max_steps"
        else:
            reason = "application_closed"
        try:
            position, orientation = self.robot.get_world_pose()
            pose = {
                "position": [float(value) for value in position],
                "quaternion_wxyz": [float(value) for value in orientation],
            }
        except Exception:
            pose = None
        atomic_write_json(
            self.paths.state,
            {
                "schema_version": SCHEMA_VERSION,
                "backend": BACKEND_NAME,
                "ready": False,
                "sequence": self.state_sequence,
                "written_at": time.time(),
                "reason": reason,
                "pose": pose,
                "camera_sequence": self.frame_sequence,
                "observer_sequence": self.observer_sequence,
                "command_sequence": self.last_command_sequence,
            },
        )

    def close(self) -> None:
        try:
            self.controller.set_command_override(self.gait_command.reset())
            self.controller.apply_action()
        except Exception:
            pass
        try:
            self.controller.close()
        finally:
            self.world.stop()


def main() -> int:
    paths = IsaacRuntimePaths(ARGS.runtime_dir.expanduser().resolve())
    runtime: BridgeRuntime | None = None
    try:
        runtime = BridgeRuntime(ARGS)
        signal.signal(signal.SIGTERM, runtime.request_stop)
        signal.signal(signal.SIGINT, runtime.request_stop)
        runtime.run()
        runtime.mark_stopped()
        return 0
    except Exception as exc:
        _atomic_error(paths, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        if runtime is not None:
            runtime.close()
        SIMULATION_APP.close()


if __name__ == "__main__":
    raise SystemExit(main())
