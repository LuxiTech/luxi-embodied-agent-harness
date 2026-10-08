#!/usr/bin/env python3
"""Run DimOS' pinned MuJoCo worker with Luxi's local simulation fixes.

The upstream worker owns the model and physics state, so a genuine third-person
frame has to be rendered inside that process.  This launcher wraps only
``mujoco.viewer.launch_passive`` and then executes the untouched upstream
launcher.  It also installs the simulation-only G1 idle state before the model
is loaded.  No DimOS source file or shared-memory layout is modified.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable
from numbers import Integral

import mujoco
from mujoco import viewer
import numpy as np


FRAME_ENV = "LUXI_THIRD_PERSON_FRAME"
DEPTH_ENV = "LUXI_HEAD_DEPTH_PATH"
LIDAR_PROXIMITY_ENV = "LUXI_LIDAR_PROXIMITY_PATH"
PERSON_ENABLED_ENV = "LUXI_SIM_PERSON"
PERSON_POSITION_ENV = "LUXI_SIM_PERSON_POSITION"


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _bounded_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class _NoPersonPositionController:
    """Pinned-worker contract used when no person body was requested."""

    def __init__(self, _model: Any) -> None:
        pass

    def tick(self, _data: Any) -> None:
        pass

    def stop(self) -> None:
        pass


def install_g1_dynamic_qpos_policy_compat() -> None:
    """Keep the pinned 29-DoF policy isolated from appended scene free joints."""

    from dimos.simulation.mujoco import policy as policy_module

    base_controller = policy_module.OnnxController
    if not getattr(base_controller, "_luxi_extra_actuator_compat", False):
        original_get_control = base_controller.get_control

        def compatible_control(self: Any, model: Any, data: Any) -> None:
            if len(data.ctrl) == len(self._default_angles):
                original_get_control(self, model, data)
                return
            self._counter += 1
            if self._counter % self._n_substeps != 0:
                return
            observation = self.get_obs(model, data)
            prediction = self._policy.run(
                self._output_names,
                {"obs": observation.reshape(1, -1)},
            )[0][0]
            self._last_action = prediction.copy()
            count = len(self._default_angles)
            data.ctrl[:count] = (
                prediction * self._action_scale + self._default_angles
            )
            self._post_control_update()

        base_controller.get_control = compatible_control
        base_controller._luxi_extra_actuator_compat = True

    controller = policy_module.G1OnnxController
    if getattr(controller, "_luxi_dynamic_qpos_compat", False):
        return
    original_init = controller.__init__

    def compatible_init(self: Any, *args: Any, **kwargs: Any) -> None:
        positional = list(args)
        if "default_angles" in kwargs:
            kwargs["default_angles"] = np.asarray(kwargs["default_angles"])[:29]
        elif len(positional) >= 2:
            positional[1] = np.asarray(positional[1])[:29]
        original_init(self, *positional, **kwargs)

    def compatible_get_obs(self: Any, model: Any, data: Any) -> np.ndarray[Any, Any]:
        linvel = data.sensor("local_linvel_pelvis").data
        gyro = data.sensor("gyro_pelvis").data
        imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_count = len(self._default_angles)
        joint_angles = data.qpos[7 : 7 + joint_count] - self._default_angles
        joint_velocities = data.qvel[6 : 6 + joint_count]
        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
        command = self._input_controller.get_command()
        command[0] = command[0] * 2 + self._drift_compensation[0]
        command[1] = command[1] * 2 + self._drift_compensation[1]
        command[2] += self._drift_compensation[2]
        return np.hstack(
            [
                linvel,
                gyro,
                gravity,
                command,
                joint_angles,
                joint_velocities,
                self._last_action,
                phase,
            ]
        ).astype(np.float32)

    controller.__init__ = compatible_init
    controller.get_obs = compatible_get_obs
    controller._luxi_dynamic_qpos_compat = True


def install_simulation_scene_overrides(
    *,
    model_module: Any,
    person_module: Any,
    payload: dict[str, Any],
) -> None:
    """Install worker-local scene/person policy without modifying DimOS source."""

    scene_xml = payload.get("scene_xml")
    if scene_xml is not None:
        if not isinstance(scene_xml, str) or "<mujoco" not in scene_xml:
            raise ValueError("simulator payload has invalid scene XML")

        def load_private_scene(config: Any) -> str:
            if any(
                getattr(config, field, None)
                for field in (
                    "mujoco_room_from_occupancy",
                    "mujoco_global_costmap_from_occupancy",
                    "mujoco_global_map_from_pointcloud",
                )
            ):
                raise RuntimeError("selected scene forbids additional ground-truth map injection")
            return scene_xml

        model_module.load_scene_xml = load_private_scene

    person = payload.get("person")
    if person is None:
        model_module._add_person_object = lambda _root: None
        person_module.PersonPositionController = _NoPersonPositionController
        return
    if not isinstance(person, dict):
        raise ValueError("person configuration must be an object or null")
    try:
        x = float(person["x"])
        y = float(person["y"])
        z = float(person.get("z", 0.0))
        yaw = float(person.get("yaw", 0.0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("person configuration has invalid coordinates") from error
    if not all(math.isfinite(value) for value in (x, y, z, yaw)):
        raise ValueError("person configuration coordinates must be finite")

    original_add_person = model_module._add_person_object

    def add_explicit_person(root: Any) -> None:
        original_add_person(root)
        body = root.find(".//body[@name='person']")
        if body is None:
            raise RuntimeError("pinned person injection did not create a body")
        body.set("pos", f"{x:.4f} {y:.4f} {z:.4f}")
        body.set(
            "quat",
            f"{math.cos(yaw / 2.0):.6f} 0 0 {math.sin(yaw / 2.0):.6f}",
        )
        person_geom = body.find("geom")
        if person_geom is None:
            raise RuntimeError("pinned person injection did not create a geom")
        person_geom.set("name", "person_collision_geom")

    model_module._add_person_object = add_explicit_person


def _configured_regular_person() -> dict[str, float] | None:
    if not _env_enabled(PERSON_ENABLED_ENV):
        return None
    raw = os.environ.get(PERSON_POSITION_ENV, "0,0,0,0")
    try:
        values = [float(part.strip()) for part in raw.split(",")]
    except ValueError as error:
        raise ValueError(f"{PERSON_POSITION_ENV} must contain x,y,z,yaw") from error
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{PERSON_POSITION_ENV} must contain four finite values")
    return {"x": values[0], "y": values[1], "z": values[2], "yaw": values[3]}


def apply_simulation_start(data: Any, payload: dict[str, Any]) -> None:
    """Apply a selected scene start before any sensor frame is emitted."""

    start = payload.get("robot_start")
    if start is None:
        return
    try:
        x, y, yaw = (float(value) for value in start)
    except (TypeError, ValueError) as error:
        raise ValueError("robot_start must contain x,y,yaw") from error
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ValueError("robot_start must contain finite values")
    if len(data.qpos) < 7:
        raise ValueError("robot_start requires a floating-base model")
    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    data.qvel[:] = 0.0
    # The next idle callback must acquire the selected scene pose rather than
    # restoring a lock captured from upstream's default start.
    from harness.robots.g1.mujoco.g1_idle_stabilizer import reset_active_idle_stabilizers

    reset_active_idle_stabilizers()


def _robot_body_ids(model: Any) -> set[int]:
    """Identify the floating-base subtree that owns the robot sensor cameras."""

    camera_bodies: list[int] = []
    for camera_id in range(int(model.ncam)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        if name == "head_camera" or (name or "").startswith("lidar_"):
            camera_bodies.append(int(model.cam_bodyid[camera_id]))

    free_joint = int(mujoco.mjtJoint.mjJNT_FREE)
    free_roots = {
        body_id
        for body_id in range(1, int(model.nbody))
        if any(
            int(model.jnt_type[joint_id]) == free_joint
            for joint_id in range(
                int(model.body_jntadr[body_id]),
                int(model.body_jntadr[body_id]) + int(model.body_jntnum[body_id]),
            )
        )
    }
    roots: set[int] = set()
    for camera_body in camera_bodies:
        body_id = camera_body
        while body_id > 0:
            if body_id in free_roots:
                roots.add(body_id)
                break
            body_id = int(model.body_parentid[body_id])
    if not roots and len(free_roots) == 1:
        # Minimal tests and robot models may omit cameras.  Multiple free bodies
        # remain ambiguous and intentionally produce no guessed robot identity.
        roots = free_roots

    bodies: set[int] = set()
    for body_id in range(1, int(model.nbody)):
        ancestor = body_id
        while ancestor > 0:
            if ancestor in roots:
                bodies.add(body_id)
                break
            ancestor = int(model.body_parentid[ancestor])
    return bodies


class LidarSegmentationMasks:
    """Bind rendered depth arrays to robot-only masks using MuJoCo geom IDs."""

    def __init__(self, model: Any, *, max_entries: int = 12) -> None:
        self.model = model
        self.robot_body_ids = _robot_body_ids(model)
        self._carried_body_ids: Callable[[], set[int]] = set
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[int, tuple[Any, Any]] = OrderedDict()
        self.diagnostic_frames: dict[str, Any] = {}

    def set_carried_body_ids_provider(
        self,
        provider: Callable[[], set[int]],
    ) -> None:
        self._carried_body_ids = provider

    def self_body_ids(self) -> set[int]:
        bodies = set(self.robot_body_ids)
        try:
            carried = self._carried_body_ids()
        except Exception:
            carried = set()
        bodies.update(
            int(body_id)
            for body_id in carried
            if 0 < int(body_id) < int(self.model.nbody)
        )
        return bodies

    def register(self, depth: Any, segmentation: Any) -> None:
        import numpy as np

        depth_array = np.asarray(depth)
        segmentation_array = np.asarray(segmentation)
        if (
            not self.robot_body_ids
            or depth_array.ndim != 2
            or segmentation_array.shape != (*depth_array.shape, 2)
        ):
            return
        object_ids = segmentation_array[..., 0]
        object_types = segmentation_array[..., 1]
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        valid = (
            (object_types == geom_type)
            & (object_ids >= 0)
            & (object_ids < int(self.model.ngeom))
        )
        mask = np.zeros(depth_array.shape, dtype=bool)
        if np.any(valid):
            geom_ids = object_ids[valid].astype(np.int64, copy=False)
            body_ids = np.asarray(self.model.geom_bodyid)[geom_ids]
            mask[valid] = np.isin(body_ids, tuple(self.self_body_ids()))
        key = id(depth)
        self._entries[key] = (depth, mask)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def pop(self, depth: Any) -> Any | None:
        entry = self._entries.pop(id(depth), None)
        if entry is None or entry[0] is not depth:
            return None
        return entry[1]


def make_lidar_self_filter(
    original: Callable[..., Any],
    *,
    robot_pixel_mask: Callable[[Any], Any | None],
    on_filter_result: Callable[[str, int], None] | None = None,
) -> Callable[..., Any]:
    """Zero only depth pixels whose segmentation geom belongs to the robot."""

    def report(status: str, masked_pixels: int = 0) -> None:
        if on_filter_result is not None:
            on_filter_result(status, masked_pixels)

    def filtered(*args: Any, **kwargs: Any) -> Any:
        depth = args[0] if args else kwargs.get("depth_image")
        try:
            mask = robot_pixel_mask(depth)
        except Exception:
            mask = None
        if mask is None:
            # Fail safe: retain every endpoint.  This may conservatively stop on
            # self returns but cannot erase a real close obstacle.
            report("missing")
            return original(*args, **kwargs)

        import numpy as np

        depth_array = np.asarray(depth)
        mask_array = np.asarray(mask, dtype=bool)
        if depth_array.ndim != 2 or mask_array.shape != depth_array.shape:
            report("invalid")
            return original(*args, **kwargs)
        report("applied", int(np.count_nonzero(mask_array)))
        filtered_depth = depth_array.copy()
        filtered_depth[mask_array] = 0.0
        if args:
            result = original(filtered_depth, *args[1:], **kwargs)
        else:
            filtered_kwargs = dict(kwargs)
            filtered_kwargs["depth_image"] = filtered_depth
            result = original(**filtered_kwargs)
        return result

    filtered._luxi_lidar_self_filter = True  # type: ignore[attr-defined]
    return filtered


def filter_robot_lidar_points_by_ray_identity(
    points: Any,
    *,
    model: Any,
    data: Any,
    robot_body_ids: set[int],
    camera_ids: tuple[int, ...],
    max_planar_radius_m: float = 0.90,
    hit_tolerance_m: float = 0.08,
) -> tuple[Any, dict[str, Any]]:
    """Remove near-field robot returns using MuJoCo's current geom identity.

    Segmentation is the primary self filter, but a raster silhouette can still
    leave one voxel at a moving limb edge.  For only the safety-relevant near
    field, cast rays from each physical lidar camera to the downsampled point.
    A point is removed only when at least one matching ray hits a robot geom and
    no matching ray hits an external geom.  Thus a real wall at 0.4--0.6 m is
    retained by identity instead of being erased by a radial blind zone.
    """

    import numpy as np

    point_array = np.asarray(points, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "available": False,
        "candidate_points": 0,
        "ray_matched": 0,
        "robot_matched": 0,
        "external_matched": 0,
        "robot_removed": 0,
    }
    if point_array.ndim != 2 or point_array.shape[1:] != (3,):
        return point_array, diagnostics
    if not robot_body_ids or not camera_ids:
        return point_array, diagnostics
    if any(camera_id < 0 or camera_id >= int(model.ncam) for camera_id in camera_ids):
        return point_array, diagnostics

    diagnostics["available"] = True
    if len(point_array) == 0:
        return point_array.copy(), diagnostics

    base_xy = np.asarray(data.qpos[:2], dtype=np.float64)
    planar_radius = np.linalg.norm(point_array[:, :2] - base_xy, axis=1)
    candidate_indices = np.flatnonzero(
        np.isfinite(point_array).all(axis=1)
        & (planar_radius <= max(0.0, float(max_planar_radius_m)))
    )
    diagnostics["candidate_points"] = int(len(candidate_indices))
    if len(candidate_indices) == 0:
        return point_array.copy(), diagnostics

    candidate_points = point_array[candidate_indices]
    matched_robot = np.zeros(len(candidate_points), dtype=bool)
    matched_external = np.zeros(len(candidate_points), dtype=bool)
    tolerance = max(0.0, float(hit_tolerance_m))

    for camera_id in camera_ids:
        origin = np.asarray(data.cam_xpos[camera_id], dtype=np.float64).copy()
        vectors = candidate_points - origin
        expected_distances = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(expected_distances) & (expected_distances > 1e-9)
        if not np.any(valid):
            continue
        valid_indices = np.flatnonzero(valid)
        directions = np.ascontiguousarray(
            vectors[valid_indices] / expected_distances[valid_indices, None],
            dtype=np.float64,
        )
        geom_ids = np.full(len(valid_indices), -1, dtype=np.int32)
        hit_distances = np.full(len(valid_indices), -1.0, dtype=np.float64)
        mujoco.mj_multiRay(
            model,
            data,
            origin,
            directions.reshape(-1),
            None,
            1,
            -1,
            geom_ids,
            hit_distances,
            None,
            len(valid_indices),
            float(np.max(expected_distances[valid_indices]) + tolerance),
        )
        close_hit = (
            (geom_ids >= 0)
            & (hit_distances >= 0.0)
            & (
                np.abs(hit_distances - expected_distances[valid_indices])
                <= tolerance
            )
        )
        if not np.any(close_hit):
            continue
        close_indices = valid_indices[close_hit]
        close_geom_ids = geom_ids[close_hit]
        body_ids = np.asarray(model.geom_bodyid)[close_geom_ids]
        is_robot = np.isin(body_ids, tuple(robot_body_ids))
        matched_robot[close_indices[is_robot]] = True
        matched_external[close_indices[~is_robot]] = True

    ray_matched = matched_robot | matched_external
    remove_candidates = matched_robot & ~matched_external
    keep = np.ones(len(point_array), dtype=bool)
    keep[candidate_indices[remove_candidates]] = False
    diagnostics.update(
        {
            "ray_matched": int(np.count_nonzero(ray_matched)),
            "robot_matched": int(np.count_nonzero(matched_robot)),
            "external_matched": int(np.count_nonzero(matched_external)),
            "robot_removed": int(np.count_nonzero(remove_candidates)),
        }
    )
    return point_array[keep].copy(), diagnostics


def _direction_label(bearing_radians: float) -> str:
    degrees = math.degrees(bearing_radians)
    absolute = abs(degrees)
    if absolute <= 22.5:
        return "front"
    if absolute <= 67.5:
        return "front_left" if degrees > 0 else "front_right"
    if absolute <= 112.5:
        return "left" if degrees > 0 else "right"
    if absolute <= 157.5:
        return "rear_left" if degrees > 0 else "rear_right"
    return "rear"


def summarize_lidar_proximity(
    points: Any,
    *,
    frame_timestamp: float,
    pose_timestamp: float,
    position: Any,
    quaternion: Any,
    sequence: int,
    written_at: float | None = None,
) -> dict[str, Any]:
    """Summarize one current, self-filtered lidar frame around the G1 base.

    Unlike the accumulated HeightCost grid, every sequence is independent
    physical sensor evidence.  Points near the floor and above the robot body
    are excluded by height only; close horizontal points are deliberately kept.
    """

    import numpy as np

    point_array = np.asarray(points, dtype=np.float64)
    base = np.asarray(position, dtype=np.float64).reshape(-1)
    rotation = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if point_array.ndim != 2 or point_array.shape[1] < 3:
        raise ValueError("lidar points must be an Nx3 array")
    if base.size < 3 or rotation.size < 4:
        raise ValueError("lidar pose must contain position and wxyz quaternion")
    numeric = (
        float(frame_timestamp),
        float(pose_timestamp),
        float(base[0]),
        float(base[1]),
        float(base[2]),
        *(float(value) for value in rotation[:4]),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("lidar pose and timestamps must be finite")
    if isinstance(sequence, bool) or int(sequence) < 1:
        raise ValueError("lidar sequence must be a positive integer")

    xyz = point_array[:, :3]
    finite = np.all(np.isfinite(xyz), axis=1)
    # The floating base rests around z=0.75 m.  This band retains walls,
    # furniture, people, and low obstacles while excluding floor and ceiling.
    lower_z = float(base[2]) - 0.55
    upper_z = float(base[2]) + 0.75
    candidates = xyz[finite & (xyz[:, 2] >= lower_z) & (xyz[:, 2] <= upper_z)]

    qw, qx, qy, qz = (float(value) for value in rotation[:4])
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    sectors: dict[str, float] = {}
    nearest_distance: float | None = None
    nearest_bearing: float | None = None
    nearest_direction: str | None = None
    nearest_world_xyz: list[float] | None = None
    nearest_body_xyz: list[float] | None = None
    if len(candidates):
        offsets = candidates[:, :2] - base[:2]
        distances = np.linalg.norm(offsets, axis=1)
        bearings = np.arctan2(offsets[:, 1], offsets[:, 0]) - yaw
        bearings = np.arctan2(np.sin(bearings), np.cos(bearings))
        for distance_value, bearing_value in zip(distances, bearings, strict=True):
            distance = float(distance_value)
            bearing = float(bearing_value)
            direction = _direction_label(bearing)
            previous = sectors.get(direction)
            if previous is None or distance < previous:
                sectors[direction] = distance
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        nearest_bearing = float(bearings[nearest_index])
        nearest_direction = _direction_label(nearest_bearing)
        nearest_point = candidates[nearest_index]
        dx = float(nearest_point[0] - base[0])
        dy = float(nearest_point[1] - base[1])
        nearest_world_xyz = [round(float(value), 4) for value in nearest_point]
        nearest_body_xyz = [
            round(math.cos(yaw) * dx + math.sin(yaw) * dy, 4),
            round(-math.sin(yaw) * dx + math.cos(yaw) * dy, 4),
            round(float(nearest_point[2] - base[2]), 4),
        ]

    return {
        "schema_version": 1,
        "available": True,
        "source": "current_lidar",
        "frame_id": "world",
        "sequence": int(sequence),
        "frame_timestamp": float(frame_timestamp),
        "pose_timestamp": float(pose_timestamp),
        "written_at": time.time() if written_at is None else float(written_at),
        "nearest_obstacle_distance": nearest_distance,
        "nearest_obstacle_bearing_deg": (
            None if nearest_bearing is None else round(math.degrees(nearest_bearing), 3)
        ),
        "nearest_obstacle_direction": nearest_direction,
        "nearest_obstacle_world_xyz": nearest_world_xyz,
        "nearest_obstacle_body_xyz": nearest_body_xyz,
        "sectors_m": {name: round(value, 4) for name, value in sectors.items()},
        "candidate_points": int(len(candidates)),
        "height_band_world_m": [round(lower_z, 4), round(upper_z, 4)],
    }


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def make_segmentation_aware_renderer(
    original_renderer: Callable[..., Any],
    masks: LidarSegmentationMasks,
) -> type[Any]:
    """Render depth and geom IDs from the exact same MuJoCo scene/context.

    A second renderer can disagree on silhouette pixels even when both were
    updated from the same ``MjData``.  Those one-pixel disagreements are enough
    to leak a moving G1 limb into a near-field point cloud.  Toggling rendering
    mode on one renderer keeps the scene, visibility ordering, rasterization,
    and camera state identical for the paired depth/identity images.
    """

    class SegmentationAwareRenderer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._args = args
            self._kwargs = kwargs
            # Depth at thin silhouettes must not use multisample depth resolve.
            model = args[0] if args else kwargs.get("model")
            quality = getattr(getattr(model, "vis", None), "quality", None)
            previous_samples = quality.offsamples if quality is not None else None
            try:
                if quality is not None:
                    quality.offsamples = 0
                self._renderer = original_renderer(*args, **kwargs)
            finally:
                if quality is not None:
                    quality.offsamples = previous_samples
            self._depth_enabled = False
            self._capture_segmentation = False
            self._segmentation_failed = False

        def __getattr__(self, name: str) -> Any:
            return getattr(self._renderer, name)

        def enable_depth_rendering(self) -> None:
            self._depth_enabled = True
            self._renderer.enable_depth_rendering()

        def disable_depth_rendering(self) -> None:
            self._depth_enabled = False
            self._capture_segmentation = False
            self._renderer.disable_depth_rendering()

        def update_scene(self, *args: Any, **kwargs: Any) -> Any:
            result = self._renderer.update_scene(*args, **kwargs)
            camera = kwargs.get("camera")
            if camera is None and len(args) >= 2:
                camera = args[1]
            camera_name = camera if isinstance(camera, str) else None
            if isinstance(camera, Integral):
                camera_name = mujoco.mj_id2name(
                    masks.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    int(camera),
                )
            self._camera_name = camera_name
            self._capture_segmentation = bool(
                self._depth_enabled
                and not self._segmentation_failed
                and isinstance(camera_name, str)
                and camera_name.startswith("lidar_")
            )
            return result

        def render(self, *args: Any, **kwargs: Any) -> Any:
            depth = self._renderer.render(*args, **kwargs)
            if self._capture_segmentation:
                try:
                    self._renderer.enable_segmentation_rendering()
                    segmentation = self._renderer.render()
                    masks.register(depth, segmentation)
                    try:
                        if not _env_enabled("LUXI_BLIND_MODE"):
                            import numpy as np
                            # Only current lidar frames; diagnostics never affect filtering.
                            if len(masks.diagnostic_frames) < 3 or self._camera_name in masks.diagnostic_frames:
                                masks.diagnostic_frames[self._camera_name] = (
                                    np.array(depth, copy=True), np.array(segmentation, copy=True))
                    except Exception:
                        pass
                except Exception:
                    self._segmentation_failed = True
                    self._capture_segmentation = False
                finally:
                    if self._depth_enabled:
                        self._renderer.enable_depth_rendering()
                    else:
                        self._renderer.disable_segmentation_rendering()
            return depth

        def close(self) -> None:
            self._renderer.close()

    return SegmentationAwareRenderer


class _HeadlessViewerHandle:
    """Subset of MuJoCo's passive-viewer API used by the pinned worker."""

    def __init__(self) -> None:
        self.cam = SimpleNamespace(
            lookat=[0.0, 0.0, 0.0],
            distance=0.0,
            azimuth=0.0,
            elevation=0.0,
        )

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        return None


class _HeadlessViewerContext:
    def __init__(self) -> None:
        self.handle = _HeadlessViewerHandle()

    def __enter__(self) -> _HeadlessViewerHandle:
        return self.handle

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        return None


class ThirdPersonPublisher:
    """Render a free camera that follows the robot root without touching physics."""

    def __init__(self, model: Any, data: Any, frame_path: Path) -> None:
        self.data = data
        self.frame_path = frame_path
        self.frame_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = _bounded_env_int("LUXI_THIRD_PERSON_WIDTH", 640, 320, 1280)
        self.height = _bounded_env_int("LUXI_THIRD_PERSON_HEIGHT", 360, 180, 720)
        fps = _bounded_env_float("LUXI_THIRD_PERSON_FPS", 10.0, 1.0, 20.0)
        self.interval = 1.0 / fps
        self.last_frame_at = 0.0
        self.failed = False

        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.scene_option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = _bounded_env_float(
            "LUXI_THIRD_PERSON_DISTANCE", 4.2, 1.5, 12.0
        )
        self.camera.azimuth = _bounded_env_float(
            "LUXI_THIRD_PERSON_AZIMUTH", 135.0, -360.0, 360.0
        )
        self.camera.elevation = _bounded_env_float(
            "LUXI_THIRD_PERSON_ELEVATION", -22.0, -80.0, 20.0
        )
        self.lookat_height = _bounded_env_float(
            "LUXI_THIRD_PERSON_LOOKAT_HEIGHT", 0.9, 0.0, 2.5
        )

    def maybe_render(self) -> None:
        if self.failed:
            return
        now = time.monotonic()
        if now - self.last_frame_at < self.interval:
            return
        self.last_frame_at = now
        try:
            root = self.data.qpos
            self.camera.lookat[:] = (float(root[0]), float(root[1]), self.lookat_height)
            self.renderer.update_scene(
                self.data,
                camera=self.camera,
                scene_option=self.scene_option,
            )
            pixels = self.renderer.render()

            # Pillow is already pinned by the DimOS environment.  A per-process
            # temporary name plus os.replace keeps concurrent HTTP reads valid.
            from PIL import Image

            temporary = self.frame_path.with_name(
                f".{self.frame_path.name}.{os.getpid()}.tmp"
            )
            Image.fromarray(pixels, mode="RGB").save(
                temporary,
                format="JPEG",
                quality=82,
                optimize=False,
            )
            os.replace(temporary, self.frame_path)
        except Exception:
            # The observer is optional.  A renderer or filesystem failure must
            # never stop the robot simulation or fill the worker's stderr pipe.
            self.failed = True

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass
        temporary = self.frame_path.with_name(
            f".{self.frame_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.unlink()
        except OSError:
            pass


class HeadDepthPublisher:
    """Publish head-camera metric depth without changing upstream SHM fields."""

    def __init__(self, model: Any, data: Any, depth_path: Path) -> None:
        self.data = data
        self.depth_path = depth_path
        self.depth_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = _bounded_env_int("LUXI_HEAD_DEPTH_WIDTH", 640, 160, 1280)
        self.height = _bounded_env_int("LUXI_HEAD_DEPTH_HEIGHT", 360, 90, 720)
        fps = _bounded_env_float("LUXI_HEAD_DEPTH_FPS", 5.0, 1.0, 15.0)
        self.interval = 1.0 / fps
        self.last_frame_at = 0.0
        self.failed = False
        camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "head_camera",
        )
        if camera_id < 0:
            raise RuntimeError("MuJoCo model does not expose head_camera")
        self.camera_id = camera_id
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.renderer.enable_depth_rendering()
        self.scene_option = mujoco.MjvOption()
        try:
            self.depth_path.unlink()
        except OSError:
            pass

    def maybe_render(self) -> None:
        if self.failed:
            return
        now = time.monotonic()
        if now - self.last_frame_at < self.interval:
            return
        self.last_frame_at = now
        try:
            import numpy as np

            self.renderer.update_scene(
                self.data,
                camera=self.camera_id,
                scene_option=self.scene_option,
            )
            depth = np.asarray(self.renderer.render(), dtype=np.float32)
            temporary = self.depth_path.with_name(
                f".{self.depth_path.name}.{os.getpid()}.tmp"
            )
            with temporary.open("wb") as stream:
                np.savez(stream, depth=depth, timestamp=time.time())
            os.replace(temporary, self.depth_path)
        except Exception:
            self.failed = True

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass
        temporary = self.depth_path.with_name(
            f".{self.depth_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.unlink()
        except OSError:
            pass


class BlindScorerTelemetryPublisher:
    """Write MuJoCo truth to a scorer-only channel without agent feedback."""

    _OBSTACLE_PREFIXES = ("wall_", "table_", "target_", "person")

    def __init__(
        self,
        model: Any,
        data: Any,
        path: Path,
        *,
        interval_seconds: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self.model = model
        self.data = data
        self.path = path
        self.interval_seconds = max(0.02, float(interval_seconds))
        self.clock = clock
        self.wall_time = wall_time
        self._last_recorded_at: float | None = None
        self._closed = False
        self._collision_ever = False
        self._first_collision_wall_time: float | None = None
        self._collision_sources: set[str] = set()
        self._pending_motion_samples: list[dict[str, float]] = []
        self._last_state: dict[str, Any] | None = None
        self._robot_body_ids = _robot_body_ids(model)
        self._obstacle_geom_ids = {
            geom_id
            for geom_id in range(int(model.ngeom))
            if (
                (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
                and name.startswith(self._OBSTACLE_PREFIXES)
            )
        }
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.path.touch(mode=0o600)
        self.path.chmod(0o600)

    def _collision_contacts(self) -> int:
        collisions = 0
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            def is_robot_geom(geom_id: int) -> bool:
                body_id = int(self.model.geom_bodyid[geom_id])
                return body_id in self._robot_body_ids

            if (
                geom1 in self._obstacle_geom_ids
                and is_robot_geom(geom2)
            ) or (
                geom2 in self._obstacle_geom_ids
                and is_robot_geom(geom1)
            ):
                collisions += 1
        return collisions

    def _observe_sync(self) -> tuple[float, dict[str, Any]]:
        """Capture safety evidence on every simulation sync, without I/O."""

        wall_now = self.wall_time()
        qpos = self.data.qpos
        qvel = self.data.qvel
        x = float(qpos[0]) if len(qpos) >= 1 else 0.0
        y = float(qpos[1]) if len(qpos) >= 2 else 0.0
        yaw = 0.0
        if len(qpos) >= 7:
            w, qx, qy, qz = (float(value) for value in qpos[3:7])
            yaw = math.atan2(
                2.0 * (w * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
        planar_speed = (
            math.hypot(float(qvel[0]), float(qvel[1])) if len(qvel) >= 2 else 0.0
        )
        contacts = self._collision_contacts()
        if contacts > 0:
            self._collision_ever = True
            self._collision_sources.add("mujoco_contact")
            if self._first_collision_wall_time is None:
                self._first_collision_wall_time = wall_now
        self._pending_motion_samples.append(
            {
                "wall_time": wall_now,
                "planar_speed_mps": planar_speed,
            }
        )
        state = {
            "wall_time": wall_now,
            "simulation_time": float(self.data.time),
            "robot_pose": {"x": x, "y": y, "yaw": yaw},
            "planar_speed_mps": planar_speed,
            "collision": contacts > 0,
            "collision_contacts": contacts,
        }
        self._last_state = state
        return wall_now, state

    def _write_pending(self, state: dict[str, Any], *, recorded_at: float) -> None:
        record = {
            "schema_version": 1,
            "source": "scorer_ground_truth",
            **state,
            "collision_ever": self._collision_ever,
            "first_collision_wall_time": self._first_collision_wall_time,
            "collision_sources": sorted(self._collision_sources),
            "motion_samples": list(self._pending_motion_samples),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._pending_motion_samples.clear()
        self._last_recorded_at = recorded_at

    def maybe_record(self, *, force: bool = False) -> None:
        if self._closed:
            return
        now = self.clock()
        _, state = self._observe_sync()
        if (
            not force
            and self._last_recorded_at is not None
            and now - self._last_recorded_at < self.interval_seconds
        ):
            return
        self._write_pending(state, recorded_at=now)

    def close(self) -> None:
        if self._pending_motion_samples and self._last_state is not None:
            self._write_pending(self._last_state, recorded_at=self.clock())
        self._closed = True


class _ViewerProxy:
    def __init__(
        self,
        handle: Any,
        publisher: ThirdPersonPublisher | None,
        depth_publisher: HeadDepthPublisher | None,
        scorer_publisher: BlindScorerTelemetryPublisher | None,
        reset_controller: Any,
        entity_controller: Any | None = None,
        pacing_data: Any | None = None,
    ) -> None:
        self._handle = handle
        self._publisher = publisher
        self._depth_publisher = depth_publisher
        self._scorer_publisher = scorer_publisher
        self._reset_controller = reset_controller
        self._entity_controller = entity_controller
        self._pacing_data = pacing_data
        from harness.robots.g1.mujoco.realtime import RealtimePacer
        self._pacer = RealtimePacer(pacing_data.time) if pacing_data is not None else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def sync(self) -> Any:
        if self._pacer is not None:
            self._pacer.sync(self._pacing_data.time)
        lock = getattr(self._handle, "lock", None)
        guard = lock() if callable(lock) else nullcontext()
        with guard:
            reset_applied = self._reset_controller.poll()
            if reset_applied and self._entity_controller is not None:
                self._entity_controller.reset()
            if self._entity_controller is not None:
                self._entity_controller.poll()
        result = self._handle.sync()
        if self._publisher is not None:
            self._publisher.maybe_render()
        if self._depth_publisher is not None:
            self._depth_publisher.maybe_render()
        if self._scorer_publisher is not None:
            self._scorer_publisher.maybe_record()
        return result


class _ObservedViewerContext:
    def __init__(
        self,
        context: Any,
        model: Any,
        data: Any,
        frame_path: Path | None,
        depth_path: Path | None,
        scorer_telemetry_path: Path | None = None,
        lidar_masks: LidarSegmentationMasks | None = None,
    ) -> None:
        self._context = context
        self._model = model
        self._data = data
        self._frame_path = frame_path
        self._depth_path = depth_path
        self._scorer_telemetry_path = scorer_telemetry_path
        self._lidar_masks = lidar_masks
        self._publisher: ThirdPersonPublisher | None = None
        self._depth_publisher: HeadDepthPublisher | None = None
        self._scorer_publisher: BlindScorerTelemetryPublisher | None = None
        self._reset_controller: Any | None = None
        self._entity_controller: Any | None = None

    def __enter__(self) -> _ViewerProxy:
        handle = self._context.__enter__()
        if self._frame_path is not None:
            try:
                self._publisher = ThirdPersonPublisher(
                    self._model,
                    self._data,
                    self._frame_path,
                )
            except Exception:
                self._publisher = None
        if self._depth_path is not None:
            try:
                self._depth_publisher = HeadDepthPublisher(
                    self._model,
                    self._data,
                    self._depth_path,
                )
            except Exception:
                self._depth_publisher = None
        if self._scorer_telemetry_path is not None:
            self._scorer_publisher = BlindScorerTelemetryPublisher(
                self._model,
                self._data,
                self._scorer_telemetry_path,
            )
        from harness.control.simulation_control import SimulationResetController
        from harness.robots.g1.mujoco.entity_manipulation import EntityManipulationController

        self._reset_controller = SimulationResetController(self._model, self._data)
        self._entity_controller = EntityManipulationController(
            self._model,
            self._data,
        )
        if self._lidar_masks is not None:
            self._lidar_masks.set_carried_body_ids_provider(
                self._entity_controller.attached_body_ids
            )
        return _ViewerProxy(
            handle,
            self._publisher,
            self._depth_publisher,
            self._scorer_publisher,
            self._reset_controller,
            self._entity_controller,
            pacing_data=self._data,
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        if self._publisher is not None:
            self._publisher.close()
        if self._depth_publisher is not None:
            self._depth_publisher.close()
        if self._scorer_publisher is not None:
            self._scorer_publisher.close()
        return self._context.__exit__(exc_type, exc, traceback)


def main() -> None:
    from harness.integrations.dimos.mujoco_assets_compat import install_mujoco_asset_compat

    install_mujoco_asset_compat()
    from harness.robots.g1.mujoco.g1_idle_stabilizer import install_g1_idle_stabilizer

    install_g1_idle_stabilizer()
    install_g1_dynamic_qpos_policy_compat()

    from dimos.simulation.mujoco import model as model_module
    from dimos.simulation.mujoco import person_on_track as person_module
    from harness.robots.g1.mujoco.g1_native_hand import install_g1_native_hand

    install_g1_native_hand(model_module)

    candidate_manifest = os.environ.get("LUXI_COMPOSED_SHM_MANIFEST")
    if candidate_manifest and not _env_enabled("LUXI_BLIND_MODE"):
        from harness.robots.g1.mujoco.shared_memory_manifest import write_manifest
        if len(sys.argv) < 3:
            raise RuntimeError("candidate worker needs exact shared-memory channel names")
        write_manifest(candidate_manifest, json.loads(sys.argv[2]))

    scorer_telemetry_path: Path | None = None
    if _env_enabled("LUXI_BLIND_MODE"):
        from harness.evaluation.blind_evaluation import (
            load_blind_simulator_payload,
            open_prepared_blind_run,
            write_blind_shared_memory_manifest,
        )

        runtime_root = Path(os.environ["DIMOS_RUNTIME_DIR"])
        run_token = os.environ.get("LUXI_BLIND_RUN_TOKEN", "")
        simulation_payload = load_blind_simulator_payload(runtime_root, run_token)
        scorer_telemetry_path = open_prepared_blind_run(
            runtime_root, run_token
        ).runtime_paths().scorer_telemetry_path
        if len(sys.argv) < 3:
            raise RuntimeError("blind MuJoCo worker did not receive shared-memory names")
        try:
            shared_memory_names = json.loads(sys.argv[2])
        except json.JSONDecodeError as error:
            raise RuntimeError("blind MuJoCo worker received invalid shared-memory names") from error
        if not isinstance(shared_memory_names, dict):
            raise RuntimeError("blind MuJoCo shared-memory names must be an object")
        write_blind_shared_memory_manifest(
            runtime_root,
            run_token,
            shared_memory_names,
        )
    else:
        from harness.robots.g1.mujoco.operator_scenes import (
            OPERATOR_SCENE_PAYLOAD_ENV,
            load_operator_scene_payload,
        )

        operator_payload_path = os.environ.get(OPERATOR_SCENE_PAYLOAD_ENV, "").strip()
        simulation_payload = (
            load_operator_scene_payload(Path(operator_payload_path))
            if operator_payload_path
            else {"person": _configured_regular_person()}
        )
    install_simulation_scene_overrides(
        model_module=model_module,
        person_module=person_module,
        payload=simulation_payload,
    )

    frame_value = os.environ.get(FRAME_ENV, "").strip()
    if not frame_value:
        frame_value = str(simulation_payload.get("scorer_third_person_path", "")).strip()
    frame_path = Path(frame_value).expanduser().resolve() if frame_value else None
    depth_value = os.environ.get(DEPTH_ENV, "").strip()
    depth_path = Path(depth_value).expanduser().resolve() if depth_value else None
    lidar_proximity_value = os.environ.get(LIDAR_PROXIMITY_ENV, "").strip()
    lidar_proximity_path = (
        Path(lidar_proximity_value).expanduser().resolve()
        if lidar_proximity_value
        else None
    )
    if lidar_proximity_path is not None:
        try:
            lidar_proximity_path.unlink()
        except FileNotFoundError:
            pass
    lidar_masks_state: dict[str, LidarSegmentationMasks | None] = {"masks": None}
    lidar_pose_state: dict[str, Any | None] = {"data": None}
    lidar_filter_state = {
        "frames": 0,
        "applied": 0,
        "missing": 0,
        "invalid": 0,
        "masked_pixels": 0,
    }
    original_renderer = mujoco.Renderer

    from dimos.simulation.mujoco import depth_camera

    def robot_pixel_mask(depth: Any) -> Any | None:
        masks = lidar_masks_state["masks"]
        return masks.pop(depth) if masks is not None else None

    def record_filter_result(status: str, masked_pixels: int) -> None:
        lidar_filter_state["frames"] += 1
        if status in {"applied", "missing", "invalid"}:
            lidar_filter_state[status] += 1
        lidar_filter_state["masked_pixels"] += max(0, int(masked_pixels))

    from harness.robots.g1.mujoco.depth_projection import pixel_center_projection
    depth_camera.depth_image_to_point_cloud = make_lidar_self_filter(
        pixel_center_projection(depth_camera.depth_image_to_point_cloud),
        robot_pixel_mask=robot_pixel_mask,
        on_filter_result=record_filter_result,
    )
    if lidar_proximity_path is not None:
        from dimos.simulation.mujoco import shared_memory as shared_memory_module

        original_write_lidar = shared_memory_module.ShmReader.write_lidar
        local_sequence = 0
        from harness.robots.g1.mujoco.critical_diagnostics import CriticalFrameRecorder
        critical_recorder = (None if _env_enabled("LUXI_BLIND_MODE") else
                             CriticalFrameRecorder(lidar_proximity_path.parent / "diagnostics/critical-lidar"))

        def write_lidar_with_proximity(self: Any, lidar_msg: Any) -> None:
            nonlocal local_sequence
            data = lidar_pose_state["data"]
            if data is None:
                original_write_lidar(self, lidar_msg)
                return
            ray_diagnostics: dict[str, Any] = {
                "available": False,
                "candidate_points": 0,
                "ray_matched": 0,
                "robot_matched": 0,
                "external_matched": 0,
                "robot_removed": 0,
            }
            camera_ids = ()
            masks = lidar_masks_state["masks"]
            raw_points = np.asarray(lidar_msg.pointcloud.points)
            try:
                points = lidar_msg.pointcloud.points
                if masks is not None:
                    camera_ids = tuple(
                        camera_id
                        for camera_id in range(int(masks.model.ncam))
                        if (
                            mujoco.mj_id2name(
                                masks.model,
                                mujoco.mjtObj.mjOBJ_CAMERA,
                                camera_id,
                            )
                            or ""
                        ).startswith("lidar_")
                    )
                    filtered_points, ray_diagnostics = (
                        filter_robot_lidar_points_by_ray_identity(
                            points,
                            model=masks.model,
                            data=data,
                            robot_body_ids=masks.self_body_ids(),
                            camera_ids=camera_ids,
                        )
                    )
                    if ray_diagnostics["robot_removed"]:
                        lidar_msg = type(lidar_msg).from_numpy(
                            filtered_points,
                            frame_id=str(lidar_msg.frame_id),
                            timestamp=float(lidar_msg.ts),
                        )
                        points = lidar_msg.pointcloud.points
            except Exception:
                # Preserve every point on refinement failure.  The unverified
                # sidecar makes motion fail closed without hiding an obstacle.
                points = lidar_msg.pointcloud.points

            original_write_lidar(self, lidar_msg)
            try:
                local_sequence += 1
                frame_timestamp = float(lidar_msg.ts)
                payload = summarize_lidar_proximity(
                    points,
                    frame_timestamp=frame_timestamp,
                    pose_timestamp=frame_timestamp,
                    position=data.qpos[:3],
                    quaternion=data.qpos[3:7],
                    sequence=local_sequence,
                )
                diagnostics = dict(lidar_filter_state)
                diagnostics["identity_verified"] = bool(
                    diagnostics["frames"] == 3
                    and diagnostics["applied"] == 3
                    and diagnostics["missing"] == 0
                    and diagnostics["invalid"] == 0
                    and ray_diagnostics["available"]
                )
                diagnostics["mode"] = (
                    "same_renderer_geom_id+nearfield_ray_identity"
                )
                diagnostics["ray_identity"] = ray_diagnostics
                payload["self_filter"] = diagnostics
                if critical_recorder is not None:
                    try:
                        candidate = critical_recorder.capture(
                            payload, raw_points, points, model=masks.model, data=data,
                            camera_ids=camera_ids, self_body_ids=masks.self_body_ids(),
                            carried_body_ids=set(masks._carried_body_ids()),
                            camera_frames=masks.diagnostic_frames) if masks is not None else None
                        if candidate:
                            payload["critical_diagnostic_candidate"] = candidate
                    except Exception as error:
                        # A diagnostic failure cannot suppress a fresh safety observation.
                        payload["critical_diagnostic_error"] = f"{type(error).__name__}: {error}"
                if masks is not None:
                    masks.diagnostic_frames.clear()
                _write_json_atomically(lidar_proximity_path, payload)
            except Exception:
                # The sidecar is safety input.  A failed frame remains stale and
                # therefore fail-closed in the UI; it must not kill simulation.
                return
            finally:
                for key in lidar_filter_state:
                    lidar_filter_state[key] = 0

        shared_memory_module.ShmReader.write_lidar = write_lidar_with_proximity
    if _env_enabled("LUXI_MUJOCO_HEADLESS"):

        def original_launch_passive(*_args: Any, **_kwargs: Any) -> _HeadlessViewerContext:
            return _HeadlessViewerContext()
    else:
        original_launch_passive = viewer.launch_passive

    def launch_with_observer(model: Any, data: Any, **kwargs: Any) -> _ObservedViewerContext:
        lidar_pose_state["data"] = data
        if lidar_masks_state["masks"] is None:
            masks = LidarSegmentationMasks(model)
            lidar_masks_state["masks"] = masks
            mujoco.Renderer = make_segmentation_aware_renderer(  # type: ignore[misc]
                original_renderer,
                masks,
            )
        apply_simulation_start(data, simulation_payload)
        mujoco.mj_forward(model, data)
        context = original_launch_passive(model, data, **kwargs)
        return _ObservedViewerContext(
            context,
            model,
            data,
            frame_path,
            depth_path,
            scorer_telemetry_path,
            lidar_masks_state["masks"],
        )

    # The wrapper is also the worker-local simulation reset hook, so it is
    # installed even when the optional third-person feed is disabled.
    viewer.launch_passive = launch_with_observer  # type: ignore[assignment]

    # In this fresh subprocess the constant still points at the pinned upstream
    # worker, even though the parent process selected this wrapper as launcher.
    from dimos.simulation.mujoco.constants import LAUNCHER_PATH as upstream_launcher

    runpy.run_path(str(upstream_launcher), run_name="__main__")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if os.environ.get("LUXI_COMPOSED_SHM_MANIFEST"):
            # Upstream pipes stderr without draining it during startup. Surface candidate diagnostics.
            import traceback
            traceback.print_exc(file=sys.stdout)
        raise
