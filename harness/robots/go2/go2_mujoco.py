"""Dynamic Unitree Go2 MuJoCo environment with HIKROBOT RGB and MID-360.

Motion is produced by twelve bounded joint torques and physical foot contact.
Object identity comes only from HIKROBOT RGB, metric localization from current
MID-360 returns, and exploration from the MID-360/IMU SLAM map. MuJoCo segmentation and
direct floating-base pose writes are intentionally absent from the task loop.
"""

from __future__ import annotations

import argparse
import base64
import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from typing import Any, Callable, Iterable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from harness.robots.go2.go2_protocol import (
    Go2RuntimePaths,
    atomic_write_bytes,
    atomic_write_json,
    read_json,
)
from harness.robots.go2.go2_semantic_map import (
    GO2_WAREHOUSE_OBSTACLES,
    Go2WarehouseSemanticMap,
)
from harness.robots.go2.go2_locomotion import Go2TorqueTrotController
from harness.robots.go2.go2_fastlio import FastLio2Process
from harness.robots.go2.go2_navigation import Mid360Slam2D, SlamPose, wrap_angle
from harness.robots.go2.go2_perception import HikrobotRgbSemanticDetector, RgbLidarDetection
from harness.robots.go2.go2_velocity import Go2VelocityGateway


HIKROBOT_CAMERA = "hikrobot_mv_cu013_a0uc_color"
MID360_SITE = "mid360"
GO2_BODY = "base"
HIKROBOT_NATIVE_IMAGE_SIZE = (1280, 1024)
HIKROBOT_MAX_FPS = 201.4
HIKROBOT_PIXEL_SIZE_UM = 4.8
HIKROBOT_ASSUMED_FOCAL_LENGTH_MM = 6.0
HIKROBOT_SENSOR_WIDTH_MM = HIKROBOT_NATIVE_IMAGE_SIZE[0] * HIKROBOT_PIXEL_SIZE_UM / 1000.0
HIKROBOT_SENSOR_HEIGHT_MM = HIKROBOT_NATIVE_IMAGE_SIZE[1] * HIKROBOT_PIXEL_SIZE_UM / 1000.0
HIKROBOT_HORIZONTAL_FOV_DEG = math.degrees(
    2.0 * math.atan(HIKROBOT_SENSOR_WIDTH_MM / (2.0 * HIKROBOT_ASSUMED_FOCAL_LENGTH_MM))
)
HIKROBOT_VERTICAL_FOV_DEG = math.degrees(
    2.0 * math.atan(HIKROBOT_SENSOR_HEIGHT_MM / (2.0 * HIKROBOT_ASSUMED_FOCAL_LENGTH_MM))
)
DEFAULT_IMAGE_SIZE = HIKROBOT_NATIVE_IMAGE_SIZE
OBSERVER_IMAGE_SIZE = (640, 360)
GO2_FOOTPRINT_RADIUS_M = 0.32
GO2_MIN_MOTION_CLEARANCE_M = 0.40
GO2_NEAR_FIELD_BLIND_BASELINE_M = 0.70
# In the warehouse profile a 0.22 m floor obstacle disappears from the lowest
# (-7 degree) MID-360 channel while its measured surface clearance is still
# about 0.70 m.  The continuity guard must engage before that geometric blind
# transition, not only after entering the ordinary 0.40 m stop envelope.
GO2_WAREHOUSE_NEAR_FIELD_BLIND_BASELINE_M = 0.80
MID360_VERTICAL_CHANNELS_DEG = (-7, -6, -5, -4, -3, -2, -1, 0, 4, 12, 24, 40)
GO2_ROBOT_IDS = ("go2-01",)
GO2_ROBOT_PREFIXES = ("",)
GO2_SCENE_ID = "warehouse_single_robot_inspection"
GO2_HOME = (-5.8, -3.0, 0.30)
GO2_BLUE_BALL_POSITION = (-2.4, -3.65, 0.22)
GO2_WAREHOUSE_MAP_SIZE = (160, 120)
GO2_WAREHOUSE_MAP_ORIGIN = (-8.0, -6.0)
GO2_INSPECTION_ROUTES = {
    "north": (
        (-3.0, 3.0),
        (0.0, 3.0),
        (3.0, 3.0),
        (5.6, 3.0),
        (5.6, 0.8),
    ),
    "south": (
        (-3.0, -3.0),
        (0.0, -3.0),
        (3.0, -3.0),
        (5.6, -3.0),
    ),
}

@dataclass(frozen=True)
class TaskResult:
    task_status: str
    completed: bool
    stationary_confirmed: bool
    sensor: str
    target: str
    elapsed_s: float
    final_distance_m: float | None = None
    final_bearing_deg: float | None = None
    frame_coverage: float | None = None
    reason: str | None = None
    inspection_region: str | None = None
    route_waypoints_completed: int | None = None
    anomaly_detected: bool | None = None
    last_planner_status: str | None = None
    safety_hold_duration_s: float | None = None
    safety_hold_count: int | None = None
    recovery_attempts: int | None = None
    post_hold_recovery_attempts: int | None = None
    scan_attempts: int | None = None
    requested_duration_s: float | None = None
    tracking_duration_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


from harness.robots.go2.go2_instructions import ParsedInstruction


from harness.robots.go2.go2_instructions import _TARGET_ALIASES
def locate_go2_model() -> Path:
    """Find the pinned MuJoCo Menagerie model without downloading assets."""

    try:
        from mujoco_playground._src import mjx_env
    except ImportError as exc:  # pragma: no cover - exercised by deployment
        raise RuntimeError(
            "mujoco_playground is required; run scripts/dimos.sh bootstrap"
        ) from exc
    path = Path(mjx_env.MENAGERIE_PATH) / "unitree_go2" / "go2.xml"
    if not path.is_file():
        raise RuntimeError(f"pinned Unitree Go2 model is missing: {path}")
    return path


def _add_geom(
    parent: ET.Element,
    *,
    name: str,
    geom_type: str,
    pos: str,
    size: str,
    rgba: str,
    group: str = "0",
) -> ET.Element:
    return ET.SubElement(
        parent,
        "geom",
        name=name,
        type=geom_type,
        pos=pos,
        size=size,
        rgba=rgba,
        group=group,
        contype="1",
        conaffinity="1",
    )




def build_go2_scene_xml(
    go2_model: Path | None = None,
    *,
    robot_count: int = 1,
) -> str:
    """Inline the Menagerie model and add only the requested sensor suite."""

    if robot_count != 1:
        raise ValueError("Go2 MuJoCo supports only go2-01")

    model_path = go2_model or locate_go2_model()
    root = ET.parse(model_path).getroot()
    root.set("model", "luxi_go2_hikrobot_mid360")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(model_path.parent / "assets"))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.01")

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("Go2 model has no worldbody")
    base = world.find("body[@name='base']")
    if base is None:
        raise RuntimeError("Go2 model has no floating base")
    freejoint = base.find("freejoint")
    if freejoint is None:
        raise RuntimeError("Go2 model has no floating joint")
    freejoint.set("name", "freejoint")

    # MuJoCo cameras look along local -Z.  These axes make +X forward, +Z up.
    ET.SubElement(
        base,
        "camera",
        name=HIKROBOT_CAMERA,
        pos="0.355 0 0.095",
        xyaxes="0 -1 0 0 0 1",
        fovy=f"{HIKROBOT_VERTICAL_FOV_DEG:.6f}",
    )
    ET.SubElement(
        base,
        "geom",
        name="hikrobot_camera_body",
        type="box",
        pos="0.315 0 0.095",
        size="0.015 0.0145 0.0145",
        rgba="0.10 0.10 0.10 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    ET.SubElement(
        base,
        "geom",
        name="hikrobot_c_mount_lens",
        type="cylinder",
        pos="0.340 0 0.095",
        size="0.010 0.018",
        quat="0.7071068 0 0.7071068 0",
        rgba="0.04 0.04 0.04 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    ET.SubElement(
        base,
        "site",
        name=MID360_SITE,
        pos="0.02 0 0.245",
        size="0.008",
        rgba="1 0.55 0.05 1",
    )
    ET.SubElement(
        base,
        "geom",
        name="mid360_mount",
        type="cylinder",
        pos="0.02 0 0.178",
        size="0.025 0.018",
        rgba="0.08 0.09 0.10 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    ET.SubElement(
        base,
        "geom",
        name="mid360_housing",
        type="cylinder",
        pos="0.02 0 0.220",
        size="0.045 0.030",
        rgba="0.12 0.13 0.14 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    ET.SubElement(
        base,
        "geom",
        name="mid360_lens_ring",
        type="cylinder",
        pos="0.02 0 0.246",
        size="0.047 0.007",
        rgba="0.95 0.48 0.04 1",
        contype="0",
        conaffinity="0",
        group="2",
    )

    _add_geom(
        world,
        name="floor",
        geom_type="plane",
        pos="0 0 0",
        size="0 0 0.05",
        rgba="0.22 0.25 0.27 1",
    )
    obstacles = [
        (
            name,
            f"{x:g} {y:g} {z:g}",
            f"{sx:g} {sy:g} {sz:g}",
        )
        for name, x, y, z, sx, sy, sz in GO2_WAREHOUSE_OBSTACLES
    ]
    for name, pos, size in obstacles:
        _add_geom(
            world,
            name=name,
            geom_type="box",
            pos=pos,
            size=size,
            rgba="0.55 0.58 0.60 1",
        )

    _add_geom(
        world,
        name="entity_red_cube",
        geom_type="box",
        pos="4.7 3.65 0.22",
        size="0.18 0.18 0.22",
        rgba="0.9 0.05 0.04 1",
    )
    _add_geom(
        world,
        name="entity_blue_ball",
        geom_type="sphere",
        # Keep the target in the visible south aisle.  The old y=-1.6
        # position was inside shelf_south_west and hid the whole sphere.
        pos=" ".join(f"{value:g}" for value in GO2_BLUE_BALL_POSITION),
        size="0.22",
        rgba="0.03 0.18 0.92 1",
    )
    bottle = ET.SubElement(
        world,
        "body",
        name="bottle_body",
        pos="4.7 -3.65 0",
    )
    _add_geom(
        bottle,
        name="entity_bottle",
        geom_type="cylinder",
        pos="0 0 0.20",
        size="0.08 0.20",
        rgba="0.15 0.65 0.92 0.9",
    )
    _add_geom(
        bottle,
        name="entity_bottle_cap",
        geom_type="cylinder",
        pos="0 0 0.43",
        size="0.045 0.03",
        rgba="0.08 0.22 0.75 1",
    )

    _add_geom(
        world,
        name="inspection_anomaly_fallen_pallet",
        geom_type="box",
        pos="5.6 -4.18 0.24",
        size="0.65 0.34 0.24",
        rgba="0.92 0.04 0.62 1",
    )

    # A visually unambiguous humanoid prevents blue-object search from locking
    # onto the old single dark-blue leg capsule.
    person = ET.SubElement(
        world,
        "body",
        name="person",
        mocap="true",
        pos="-5.5 4.2 0",
    )
    _add_geom(
        person,
        name="person_torso",
        geom_type="box",
        pos="0 0 1.18",
        size="0.20 0.30 0.38",
        rgba="0.92 0.34 0.08 1",
    )
    _add_geom(
        person,
        name="person_head",
        geom_type="sphere",
        pos="0 0 1.72",
        size="0.18",
        rgba="0.78 0.57 0.43 1",
    )
    for side, y in (("left", 0.14), ("right", -0.14)):
        _add_geom(
            person,
            name=f"person_{side}_leg",
            geom_type="capsule",
            pos=f"0 {y} 0.48",
            size="0.09 0.38",
            rgba="0.18 0.18 0.18 1",
        )
        _add_geom(
            person,
            name=f"person_{side}_arm",
            geom_type="capsule",
            pos=f"0 {0.36 if side == 'left' else -0.36} 1.16",
            size="0.065 0.32",
            rgba="0.78 0.57 0.43 1",
        )

    sensors = root.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "gyro", name="mid360_gyro", site=MID360_SITE)
    ET.SubElement(
        sensors,
        "accelerometer",
        name="mid360_accelerometer",
        site=MID360_SITE,
    )
    ET.SubElement(sensors, "framepos", name="mid360_position", objtype="site", objname=MID360_SITE)
    ET.SubElement(sensors, "framequat", name="mid360_orientation", objtype="site", objname=MID360_SITE)

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(HIKROBOT_NATIVE_IMAGE_SIZE[0]))
    global_visual.set("offheight", str(HIKROBOT_NATIVE_IMAGE_SIZE[1]))
    ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.35 0.35 0.35")
    return ET.tostring(root, encoding="unicode")


def compile_go2_scene(
    go2_model: Path | None = None,
    *,
    robot_count: int = 1,
) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(
        build_go2_scene_xml(go2_model, robot_count=robot_count)
    )
    assert_requested_sensor_contract(model, robot_count=robot_count)
    return model


def _object_names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [
        mujoco.mj_id2name(model, object_type, index) or ""
        for index in range(count)
    ]


def assert_requested_sensor_contract(
    model: mujoco.MjModel,
    *,
    robot_count: int = 1,
) -> None:
    """Fail closed if a Go2 built-in radar/lidar leaks into the runtime."""

    cameras = _object_names(model, mujoco.mjtObj.mjOBJ_CAMERA, int(model.ncam))
    sensors = _object_names(model, mujoco.mjtObj.mjOBJ_SENSOR, int(model.nsensor))
    sites = _object_names(model, mujoco.mjtObj.mjOBJ_SITE, int(model.nsite))
    for prefix in GO2_ROBOT_PREFIXES[:robot_count]:
        if f"{prefix}{HIKROBOT_CAMERA}" not in cameras or f"{prefix}{MID360_SITE}" not in sites:
            raise RuntimeError(
                f"HIKROBOT RGB or MID-360 sensor suite is incomplete for {prefix or 'go2-01'}"
            )
    forbidden = [
        name
        for name in (*cameras, *sensors, *sites)
        if re.search(r"(?:radar|utlidar|go2_lidar|native_lidar)", name, re.IGNORECASE)
    ]
    if forbidden:
        raise RuntimeError(f"Go2 built-in radar/lidar must remain disabled: {forbidden}")


from harness.robots.go2.go2_instructions import parse_instruction


class Go2MujocoEnvironment:
    """Sensor-driven Go2 search/follow environment."""

    def __init__(
        self,
        *,
        width: int = DEFAULT_IMAGE_SIZE[0],
        height: int = DEFAULT_IMAGE_SIZE[1],
        realtime: bool = False,
        detector: HikrobotRgbSemanticDetector | None = None,
        robot_id: str = GO2_ROBOT_IDS[0],
        name_prefix: str = GO2_ROBOT_PREFIXES[0],
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
        home_pose: tuple[float, float, float] = GO2_HOME,
        auto_reset: bool = True,
    ) -> None:
        self.model = model or compile_go2_scene()
        self.data = data or mujoco.MjData(self.model)
        self._warehouse_profile = True
        self.robot_id = str(robot_id)
        self.name_prefix = str(name_prefix)
        self.home_pose = tuple(float(value) for value in home_pose)
        self.width = width
        self.height = height
        self._renderer: mujoco.Renderer | None = None
        self._observer_renderer: mujoco.Renderer | None = None
        self._render_lock = threading.RLock()
        self._viewer: Any | None = None
        self.realtime = bool(realtime)
        self.cancel_requested = False
        self.cancel_reason = "operator stop requested"
        self.stationary_confirmed = True
        self.current_command = [0.0] * 6
        self.velocity_gateway = Go2VelocityGateway()
        self._semantic_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="go2-qwen"
        )
        self._tick_callback: Callable[["Go2MujocoEnvironment"], None] | None = None
        self._last_tick_callback_time = -math.inf
        self._base_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{self.name_prefix}{GO2_BODY}",
        )
        if self._base_id < 0:
            raise RuntimeError(f"Go2 body is missing for {self.robot_id}")
        freejoint_id = int(self.model.body_jntadr[self._base_id])
        self._qpos_adr = int(self.model.jnt_qposadr[freejoint_id])
        self._qvel_adr = int(self.model.jnt_dofadr[freejoint_id])
        robot_body_ids = {self._base_id}
        for body_id in range(int(self.model.nbody)):
            parent = body_id
            while parent > 0:
                if parent == self._base_id:
                    robot_body_ids.add(body_id)
                    break
                parent = int(self.model.body_parentid[parent])
        self._robot_geom_ids = {
            geom_id
            for geom_id in range(int(self.model.ngeom))
            if int(self.model.geom_bodyid[geom_id]) in robot_body_ids
        }
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._collision_hold = False
        self._inspection_anomaly_detected = False
        self._near_field_blind_hold = False
        self.external_navigation_status: dict[str, Any] | None = None
        self._last_translation_direction: np.ndarray[Any, Any] | None = None
        self._last_directional_clearance = math.inf
        self._zero_translation_started_at: float | None = None
        self._zero_translation_epoch_reset = False
        self._safety_scan_time = -math.inf
        self._safety_scan_cloud = np.empty((0, 3), dtype=np.float64)
        self._mid360_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"{self.name_prefix}{MID360_SITE}",
        )
        self._camera_name = f"{self.name_prefix}{HIKROBOT_CAMERA}"
        self._person_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "person"
        )
        self._person_geom_ids = {
            geom_id
            for geom_id in range(int(self.model.ngeom))
            if int(self.model.geom_bodyid[geom_id]) == self._person_body_id
        }
        self._person_mocap_id = int(self.model.body_mocapid[self._person_body_id])
        self._person_x = -5.5 if self._warehouse_profile else 2.6
        self._person_y = 4.2 if self._warehouse_profile else 1.4
        self._person_yaw = math.pi
        self._person_mode = "paused" if self._warehouse_profile else "auto"
        self._person_forward = 0.0
        self._person_turn = 0.0
        self._person_command_expires_at = 0.0
        self._person_boundary_blocked = False
        self._person_auto_waypoints = (
            (
                (-5.5, 4.2),
                (-4.8, 4.2),
                (-4.8, 4.5),
                (-5.5, 4.5),
            )
            if self._warehouse_profile
            else (
                (2.6, 1.4),
                (3.5, 1.4),
                (3.5, 1.85),
                (2.6, 1.85),
            )
        )
        self._person_auto_waypoint = 1
        self._person_owner: Go2MujocoEnvironment = self
        self._observer_camera = mujoco.MjvCamera()
        self._observer_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._observer_camera.trackbodyid = self._base_id
        self._observer_camera.distance = 6.4
        self._observer_camera.azimuth = 135.0
        self._observer_camera.elevation = -32.0
        self.locomotion = Go2TorqueTrotController(
            self.model,
            name_prefix=self.name_prefix,
        )
        self.detector = detector or HikrobotRgbSemanticDetector(
            horizontal_fov_degrees=HIKROBOT_HORIZONTAL_FOV_DEG
        )
        self.slam = Mid360Slam2D(
            width=GO2_WAREHOUSE_MAP_SIZE[0] if self._warehouse_profile else 120,
            height=GO2_WAREHOUSE_MAP_SIZE[1] if self._warehouse_profile else 120,
            origin_x=GO2_WAREHOUSE_MAP_ORIGIN[0] if self._warehouse_profile else -6.0,
            origin_y=GO2_WAREHOUSE_MAP_ORIGIN[1] if self._warehouse_profile else -6.0,
            footprint_radius=GO2_FOOTPRINT_RADIUS_M,
        )
        self.fastlio: FastLio2Process | None = None
        self._fastlio_imu_samples: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ] = []
        self._last_fastlio_imu_time = -math.inf
        self._fastlio_filtered_gyro: np.ndarray[Any, np.dtype[np.float64]] | None = None
        self._fastlio_filtered_acceleration: np.ndarray[
            Any, np.dtype[np.float64]
        ] | None = None
        self._gyro_sensor_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            f"{self.name_prefix}mid360_gyro",
        )
        self._accelerometer_sensor_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            f"{self.name_prefix}mid360_accelerometer",
        )
        if auto_reset:
            self._set_home_pose()

    def _set_home_pose(self, *, reset_model: bool = True) -> None:
        if reset_model:
            if int(self.model.nkey):
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            else:
                mujoco.mj_resetData(self.model, self.data)
        qpos = self.data.qpos
        qpos[self._qpos_adr : self._qpos_adr + 3] = self.home_pose
        qpos[self._qpos_adr + 3 : self._qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        qpos[self.locomotion._qpos_indices] = self.locomotion.HOME
        self.data.qvel[self._qvel_adr : self._qvel_adr + 6] = 0.0
        self.locomotion.reset()
        self._person_x = -5.5 if self._warehouse_profile else 2.6
        self._person_y = 4.2 if self._warehouse_profile else 1.4
        self._person_yaw = math.pi
        self._person_auto_waypoint = 1
        self._apply_person_pose()
        mujoco.mj_forward(self.model, self.data)

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
        return self._renderer

    def _capture_hikrobot_on_render_thread(self) -> np.ndarray[Any, Any]:
        renderer = self.renderer
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(self.data, camera=self._camera_name)
        return renderer.render().copy()

    def _capture_operator_on_render_thread(self) -> np.ndarray[Any, Any]:
        if self._observer_renderer is None:
            self._observer_renderer = mujoco.Renderer(
                self.model,
                height=OBSERVER_IMAGE_SIZE[1],
                width=OBSERVER_IMAGE_SIZE[0],
            )
        self._observer_renderer.update_scene(
            self.data,
            camera=self._observer_camera,
        )
        return self._observer_renderer.render().copy()

    def open_viewer(self) -> None:
        if self._viewer is None:
            from mujoco import viewer

            self._viewer = viewer.launch_passive(self.model, self.data)

    def close(self) -> None:
        self._semantic_executor.shutdown(wait=False, cancel_futures=True)
        if self.fastlio is not None:
            self.fastlio.close()
            self.fastlio = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._observer_renderer is not None:
            self._observer_renderer.close()
            self._observer_renderer = None

    def _fastlio(self) -> FastLio2Process:
        if self.fastlio is None:
            self.fastlio = FastLio2Process(
                robot_id=self.robot_id,
                anchor=SlamPose(self.home_pose[0], self.home_pose[1], 0.0),
            )
        return self.fastlio

    def __enter__(self) -> "Go2MujocoEnvironment":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def base_yaw(self) -> float:
        start = self._qpos_adr + 3
        w, x, y, z = (float(value) for value in self.data.qpos[start : start + 4])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def base_pose(self) -> tuple[list[float], list[float]]:
        start = self._qpos_adr
        return (
            [float(value) for value in self.data.qpos[start : start + 3]],
            [float(value) for value in self.data.qpos[start + 3 : start + 7]],
        )

    def command_velocity(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
        *,
        source: str,
    ) -> None:
        """Publish the simulator equivalent of one ROS 2 ``/cmd_vel``."""

        self.velocity_gateway.submit(
            linear_x,
            linear_y,
            angular_z,
            source=source,
        )

    def drive_velocity(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
        *,
        source: str,
        steps: int = 1,
    ) -> None:
        self.command_velocity(linear_x, linear_y, angular_z, source=source)
        self._step(steps=steps)

    def _step(self, *, steps: int = 1) -> None:
        # _prepare_physics_step owns velocity_gateway.current and
        # locomotion.apply; keeping that torque-only boundary explicit here
        # also documents that this method never writes floating-base state.
        timestep = float(self.model.opt.timestep)
        for _ in range(steps):
            wall_step_started = time.monotonic()
            self._prepare_physics_step(timestep)
            self._update_person_motion(timestep)
            mujoco.mj_step(self.model, self.data)
            self._finish_physics_step()
            if self.realtime or self._viewer is not None:
                remaining = timestep - (time.monotonic() - wall_step_started)
                if remaining > 0.0:
                    time.sleep(remaining)

    def _prepare_physics_step(
        self,
        timestep: float,
    ) -> None:
        if self.cancel_requested:
            self.velocity_gateway.stop(source="cancellation_barrier")
        command = self.velocity_gateway.current()
        safe_x, safe_y = command.linear_x, command.linear_y
        clearance = self.mid360_directional_clearance(safe_x, safe_y)
        speed = math.hypot(safe_x, safe_y)
        blind_loss = False
        if speed <= 1.0e-4:
            if self._zero_translation_started_at is None:
                self._zero_translation_started_at = float(self.data.time)
                self._zero_translation_epoch_reset = False
            elif (
                not self._zero_translation_epoch_reset
                and float(self.data.time) - self._zero_translation_started_at >= 0.30
            ):
                # A sustained stop or pure-yaw scan breaks translational lidar
                # continuity: the next body-frame direction is a fresh route
                # segment, not evidence that an old near return disappeared.
                # Current absolute clearance and contact gates remain active.
                self._last_translation_direction = None
                self._last_directional_clearance = math.inf
                self._zero_translation_epoch_reset = True
        else:
            self._zero_translation_started_at = None
            self._zero_translation_epoch_reset = False
        if speed > 0.0:
            blind_baseline = (
                GO2_WAREHOUSE_NEAR_FIELD_BLIND_BASELINE_M
            )
            direction = np.asarray((safe_x / speed, safe_y / speed))
            same_direction = bool(
                self._last_translation_direction is not None
                and float(direction @ self._last_translation_direction) >= 0.85
            )
            blind_loss = bool(
                same_direction
                and math.isfinite(self._last_directional_clearance)
                and self._last_directional_clearance
                <= blind_baseline
                and clearance > self._last_directional_clearance + 0.35
            )
            if not blind_loss:
                self._last_translation_direction = direction
                self._last_directional_clearance = clearance
        self._near_field_blind_hold = blind_loss
        if (
            self._collision_hold
            or blind_loss
            or (speed > 0.0 and clearance <= GO2_MIN_MOTION_CLEARANCE_M)
        ):
            safe_x, safe_y = 0.0, 0.0
        safe_yaw = command.angular_z
        self.current_command = [safe_x, safe_y, 0.0, 0.0, 0.0, safe_yaw]
        self.locomotion.apply(
            self.data,
            linear_x_mps=safe_x,
            linear_y_mps=safe_y,
            yaw_rps=safe_yaw,
            timestep=timestep,
        )

    def _finish_physics_step(self) -> None:
        if float(self.data.time) - self._last_fastlio_imu_time >= 0.019:
            gyro_address = int(self.model.sensor_adr[self._gyro_sensor_id])
            acceleration_address = int(
                self.model.sensor_adr[self._accelerometer_sensor_id]
            )
            gyro = tuple(
                float(value)
                for value in self.data.sensordata[gyro_address : gyro_address + 3]
            )
            acceleration = tuple(
                float(value)
                for value in self.data.sensordata[
                    acceleration_address : acceleration_address + 3
                ]
            )
            raw_gyro = np.asarray(gyro, dtype=np.float64)
            raw_acceleration = np.asarray(acceleration, dtype=np.float64)
            if self._fastlio_filtered_gyro is None:
                self._fastlio_filtered_gyro = raw_gyro
                self._fastlio_filtered_acceleration = raw_acceleration
            else:
                # Approximate the MID-360 IMU's bandwidth at the 50 Hz
                # estimator boundary. MuJoCo's rigid site otherwise exposes
                # contact impulses above the physical sensor/filter bandwidth.
                self._fastlio_filtered_gyro += 0.25 * (
                    raw_gyro - self._fastlio_filtered_gyro
                )
                assert self._fastlio_filtered_acceleration is not None
                self._fastlio_filtered_acceleration += 0.12 * (
                    raw_acceleration - self._fastlio_filtered_acceleration
                )
            assert self._fastlio_filtered_acceleration is not None
            acceleration_norm = float(
                np.linalg.norm(self._fastlio_filtered_acceleration)
            )
            if acceleration_norm > 1.0e-6:
                # The procedural trot has no compliant sensor mount and its
                # rigid contact impulses alias into the 50 Hz site sensor.
                # Retain the measured gravity direction (roll/pitch) while
                # applying the MID-360 boundary's gravity-scale calibration.
                calibrated_acceleration = (
                    self._fastlio_filtered_acceleration
                    * (9.80665 / acceleration_norm)
                )
            else:
                calibrated_acceleration = np.asarray((0.0, 0.0, 9.80665))
            self._fastlio_imu_samples.append(
                (
                    tuple(float(value) for value in self._fastlio_filtered_gyro),
                    tuple(
                        float(value)
                        for value in calibrated_acceleration
                    ),
                )
            )
            self._fastlio_imu_samples = self._fastlio_imu_samples[-8:]
            self._last_fastlio_imu_time = float(self.data.time)
        if self._has_external_robot_contact():
            self._collision_hold = True
            self.cancel_requested = True
            self.cancel_reason = "external contact safety stop"
            self.velocity_gateway.stop(source="external_contact_stop")
            self.current_command = [0.0] * 6
        if self._viewer is not None:
            self._viewer.sync()
        if (
            self._tick_callback is not None
            and float(self.data.time) - self._last_tick_callback_time >= 0.095
        ):
            self._last_tick_callback_time = float(self.data.time)
            self._tick_callback(self)

    def stop(self) -> bool:
        """Command zero and require two distinct low-speed odometry samples."""

        self.stationary_confirmed = False
        self.velocity_gateway.stop(source="terminal_stop")
        consecutive = 0
        for _ in range(20):
            self._step(steps=10)
            planar_speed = math.hypot(
                float(self.data.qvel[self._qvel_adr]),
                float(self.data.qvel[self._qvel_adr + 1]),
            )
            yaw_speed = abs(float(self.data.qvel[self._qvel_adr + 5]))
            consecutive = consecutive + 1 if planar_speed <= 0.025 and yaw_speed <= 0.05 else 0
            if consecutive >= 2:
                self.stationary_confirmed = True
                break
        # A newly accepted stationary route segment starts a fresh directional
        # lidar continuity epoch. The old near-return comparison belongs to
        # the completed segment and must not deadlock motion after a turn or
        # waypoint stop.
        self._last_translation_direction = None
        self._last_directional_clearance = math.inf
        self._zero_translation_started_at = float(self.data.time)
        self._zero_translation_epoch_reset = True
        self._near_field_blind_hold = False
        self.current_command = [0.0] * 6
        return self.stationary_confirmed

    def set_tick_callback(
        self,
        callback: Callable[["Go2MujocoEnvironment"], None] | None,
    ) -> None:
        self._tick_callback = callback

    def request_stop(self) -> None:
        self.cancel_requested = True
        self.cancel_reason = "operator stop requested"
        self.velocity_gateway.stop(source="operator_stop")
        self.current_command = [0.0] * 6

    def _has_external_robot_contact(self) -> bool:
        """Last-resort contact stop; never used for target identity or ranging."""

        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 < 0 or geom2 < 0:
                continue
            first_robot = geom1 in self._robot_geom_ids
            second_robot = geom2 in self._robot_geom_ids
            if first_robot == second_robot:
                continue
            external = geom2 if first_robot else geom1
            if external != self._floor_geom_id:
                return True
        return False

    def set_person_control(
        self,
        mode: str,
        *,
        forward: float = 0.0,
        turn: float = 0.0,
        expires_at: float = 0.0,
    ) -> None:
        if self._person_owner is not self:
            self._person_owner.set_person_control(
                mode,
                forward=forward,
                turn=turn,
                expires_at=expires_at,
            )
            return
        if mode not in {"auto", "manual", "paused"}:
            return
        if mode == "auto" and self._person_mode != "auto":
            self._person_auto_waypoint = min(
                range(len(self._person_auto_waypoints)),
                key=lambda index: math.hypot(
                    self._person_auto_waypoints[index][0] - self._person_x,
                    self._person_auto_waypoints[index][1] - self._person_y,
                ),
            )
        self._person_mode = mode
        self._person_forward = float(np.clip(forward, -1.0, 1.0))
        self._person_turn = float(np.clip(turn, -1.0, 1.0))
        self._person_command_expires_at = float(expires_at)

    def person_control_status(self) -> dict[str, Any]:
        if self._person_owner is not self:
            return self._person_owner.person_control_status()
        command_fresh = bool(
            self._person_mode == "manual"
            and time.time() <= self._person_command_expires_at
        )
        return {
            "mode": self._person_mode,
            "command_fresh": command_fresh,
            "forward": self._person_forward if command_fresh else 0.0,
            "turn": self._person_turn if command_fresh else 0.0,
            "boundary_blocked": self._person_boundary_blocked,
            "position": [self._person_x, self._person_y, 0.0],
            "yaw": self._person_yaw,
        }

    def _apply_person_pose(self) -> None:
        if self._person_owner is not self:
            self._person_owner._apply_person_pose()
            return
        if self._person_mocap_id < 0:
            return
        self.data.mocap_pos[self._person_mocap_id] = (self._person_x, self._person_y, 0.0)
        half = 0.5 * self._person_yaw
        self.data.mocap_quat[self._person_mocap_id] = (
            math.cos(half), 0.0, 0.0, math.sin(half)
        )

    def _update_person_motion(self, timestep: float) -> None:
        if self._person_owner is not self:
            return
        forward = turn = 0.0
        if self._person_mode == "auto":
            target_x, target_y = self._person_auto_waypoints[self._person_auto_waypoint]
            delta_x, delta_y = target_x - self._person_x, target_y - self._person_y
            if math.hypot(delta_x, delta_y) < 0.10:
                self._person_auto_waypoint = (
                    self._person_auto_waypoint + 1
                ) % len(self._person_auto_waypoints)
                target_x, target_y = self._person_auto_waypoints[self._person_auto_waypoint]
                delta_x, delta_y = target_x - self._person_x, target_y - self._person_y
            heading_error = wrap_angle(math.atan2(delta_y, delta_x) - self._person_yaw)
            turn = float(np.clip(2.0 * heading_error, -0.75, 0.75))
            forward = 0.08 if abs(heading_error) < 0.75 else 0.02
        elif self._person_mode == "manual" and time.time() <= self._person_command_expires_at:
            forward = 0.40 * self._person_forward
            turn = 0.90 * self._person_turn

        self._person_yaw = wrap_angle(self._person_yaw + turn * timestep)
        next_x = self._person_x + math.cos(self._person_yaw) * forward * timestep
        next_y = self._person_y + math.sin(self._person_yaw) * forward * timestep
        clamped_x = float(np.clip(next_x, -6.6, 6.6))
        clamped_y = float(np.clip(next_y, -4.6, 4.6))
        self._person_boundary_blocked = abs(clamped_x - next_x) > 1e-9 or abs(clamped_y - next_y) > 1e-9
        self._person_x, self._person_y = clamped_x, clamped_y
        if self._person_boundary_blocked and self._person_mode == "auto":
            self._person_yaw = wrap_angle(self._person_yaw + math.pi)
        self._apply_person_pose()

    def _move_person(self) -> None:
        """Compatibility hook: motion is integrated once per physics step."""
        self._person_owner._apply_person_pose()

    def mid360_point_cloud(
        self,
        *,
        azimuth_samples: int = 180,
        elevation_deg: Iterable[float] = (-12, -6, 0, 6, 12),
        frame: str = "world",
        exclude_mapping_dynamics: bool = False,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return a MID-360 scan in the requested public sensor/world frame."""

        if frame not in {"world", "sensor"}:
            raise ValueError("MID-360 frame must be 'world' or 'sensor'")

        origin = np.asarray(self.data.site_xpos[self._mid360_site_id], dtype=np.float64)
        rotation = np.asarray(self.data.site_xmat[self._mid360_site_id], dtype=np.float64).reshape(3, 3)
        endpoints: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        for elevation in elevation_deg:
            pitch = math.radians(float(elevation))
            for index in range(azimuth_samples):
                yaw = 2.0 * math.pi * index / azimuth_samples
                local_ray = np.array(
                    [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)],
                    dtype=np.float64,
                )
                ray = rotation @ local_ray
                geom_id = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    ray,
                    None,
                    1,
                    self._base_id,
                    geom_id,
                )
                if 0.05 < distance <= 30.0:
                    hit_geom_id = int(geom_id[0])
                    dynamic_hit = bool(
                        exclude_mapping_dynamics
                        and (
                            hit_geom_id in self._person_geom_ids
                        )
                    )
                    if dynamic_hit:
                        continue
                    endpoints.append(
                        origin + ray * distance
                        if frame == "world"
                        else local_ray * distance
                    )
        if not endpoints:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(endpoints)

    def mid360_min_clearance(self, *, forward_only: bool = False) -> float:
        cloud = self.mid360_point_cloud(
            azimuth_samples=144,
            elevation_deg=MID360_VERTICAL_CHANNELS_DEG,
        )
        if cloud.size == 0:
            return math.inf
        origin = np.asarray(self.data.site_xpos[self._mid360_site_id], dtype=np.float64)
        delta = cloud - origin
        if forward_only:
            rotation = np.asarray(self.data.site_xmat[self._mid360_site_id]).reshape(3, 3)
            local = delta @ rotation
            mask = (local[:, 0] > 0.0) & (np.abs(local[:, 1]) < np.maximum(0.18, local[:, 0] * 0.55))
            delta = delta[mask]
            if not len(delta):
                return math.inf
        centre_distance = float(np.min(np.linalg.norm(delta[:, :2], axis=1)))
        return max(0.0, centre_distance - GO2_FOOTPRINT_RADIUS_M)

    def mid360_directional_clearance(self, linear_x: float, linear_y: float) -> float:
        """Return clearance in the commanded body-frame translation corridor."""

        speed = math.hypot(linear_x, linear_y)
        if speed <= 1.0e-6:
            return math.inf
        if float(self.data.time) - self._safety_scan_time >= 0.045:
            self._safety_scan_cloud = self.mid360_point_cloud(
                azimuth_samples=90,
                elevation_deg=MID360_VERTICAL_CHANNELS_DEG,
                frame="sensor",
            )
            self._safety_scan_time = float(self.data.time)
        cloud = self._safety_scan_cloud
        if not len(cloud):
            return math.inf
        direction = np.asarray((linear_x / speed, linear_y / speed))
        planar = cloud[:, :2]
        along = planar @ direction
        cross = np.abs(planar[:, 0] * direction[1] - planar[:, 1] * direction[0])
        mask = (along > 0.0) & (cross < np.maximum(0.18, along * 0.55))
        if not np.any(mask):
            return math.inf
        return max(
            0.0,
            float(np.min(np.linalg.norm(planar[mask], axis=1)))
            - GO2_FOOTPRINT_RADIUS_M,
        )

    def capture_hikrobot_rgb(self) -> np.ndarray[Any, Any]:
        with self._render_lock:
            return self._capture_hikrobot_on_render_thread()

    def capture_operator_view(self) -> np.ndarray[Any, Any]:
        """Render a tracking view for the local operator, never task evidence."""

        with self._render_lock:
            return self._capture_operator_on_render_thread()

    def detect(self, target: str, *, semantic: bool = False) -> RgbLidarDetection | None:
        color = self.capture_hikrobot_rgb()
        mid360_points = self.mid360_point_cloud(
            azimuth_samples=720,
            elevation_deg=MID360_VERTICAL_CHANNELS_DEG,
            frame="sensor",
        )
        if semantic:
            frame_time = float(self.data.time)
            # Qwen is remote I/O and must never own the MuJoCo/sensor thread.
            # Snapshot synchronized RGB/lidar evidence, run only localization
            # in the worker, and keep stepping at real-time cadence with a
            # forced zero command so camera, MID-360, SLAM and UI state remain
            # live while inference is pending.
            self.velocity_gateway.stop(source="semantic_inference_hold")
            future = self._semantic_executor.submit(
                self.detector.localize_bbox,
                target=target,
                color=color.copy(),
                frame_time=frame_time,
            )
            try:
                # Cheap localizers retain their previous zero-step behavior;
                # remote inference enters the live sensor wait loop.
                semantic_bbox = future.result(timeout=0.02)
            except FutureTimeoutError:
                while not future.done():
                    wall_started = time.monotonic()
                    self._step(steps=5)
                    remaining = 0.05 - (time.monotonic() - wall_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
                semantic_bbox = future.result()
            return self.detector.acquire_from_bbox(
                target=target,
                color=color,
                mid360_points=mid360_points,
                frame_time=frame_time,
                semantic_bbox=semantic_bbox,
            )
        return self.detector.track(
            target=target,
            color=color,
            mid360_points=mid360_points,
            frame_time=float(self.data.time),
        )

    def turn_measured(
        self,
        angle_radians: float,
        *,
        yaw_rate: float = 0.45,
        timeout_s: float = 12.0,
    ) -> bool:
        """Turn using the torque gait until measured base yaw reaches the target."""

        start = self.base_yaw()
        sign = 1.0 if angle_radians >= 0.0 else -1.0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self.cancel_requested:
            progress = sign * wrap_angle(self.base_yaw() - start)
            if progress >= abs(angle_radians) - math.radians(3.0):
                return self.stop()
            self.drive_velocity(
                0.0,
                0.0,
                sign * abs(yaw_rate),
                source="measured_viewpoint_scan",
                steps=5,
            )
        self.stop()
        return False

    def navigate_path(
        self,
        path: list[tuple[float, float]],
        *,
        deadline: float,
    ) -> bool:
        """Follow known-free SLAM waypoints with torque gait and live lidar."""

        for goal_x, goal_y in path[1:]:
            blind_recoveries = 0
            while time.monotonic() < deadline and not self.cancel_requested:
                pose = self.slam.pose
                if pose is None:
                    self.drive_velocity(
                        0.0, 0.0, 0.0, source="slam_pose_wait", steps=5
                    )
                    continue
                dx, dy = goal_x - pose.x, goal_y - pose.y
                distance = math.hypot(dx, dy)
                if distance <= 0.18:
                    break
                # Mapping continues while the robot moves.  Revalidate the
                # complete remaining segment every control cycle so a newly
                # occupied or stale/unknown cell invalidates the old path.
                if not self.slam.segment_is_traversable(
                    (pose.x, pose.y),
                    (goal_x, goal_y),
                ):
                    self.stop()
                    return False
                bearing = wrap_angle(math.atan2(dy, dx) - pose.yaw)
                local_x = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy
                local_y = -math.sin(pose.yaw) * dx + math.cos(pose.yaw) * dy
                yaw_rate = float(np.clip(0.9 * bearing, -0.45, 0.45))
                if abs(bearing) > math.radians(45.0):
                    linear_x, linear_y = 0.0, 0.0
                else:
                    linear_x = float(np.clip(0.55 * local_x, -0.12, 0.20))
                    linear_y = float(np.clip(0.35 * local_y, -0.10, 0.10))
                if self.mid360_directional_clearance(linear_x, linear_y) <= 0.32:
                    self.stop()
                    return False
                self.drive_velocity(
                    linear_x,
                    linear_y,
                    yaw_rate,
                    source="mid360_frontier_navigation",
                    steps=5,
                )
                if self._near_field_blind_hold:
                    blind_recoveries += 1
                    self.stop()
                    if blind_recoveries > 3:
                        return False
            else:
                self.stop()
                return False
        return self.stop()

    def search_object(
        self,
        target: str,
        *,
        timeout_s: float = 240.0,
        standoff_m: float = 0.9,
        target_prior_xy: tuple[float, float] | None = None,
    ) -> TaskResult:
        started = time.monotonic()
        self.cancel_requested = False
        self.cancel_reason = "operator stop requested"
        deadline = started + timeout_s
        detection: RgbLidarDetection | None = None
        frontiers_attempted = 0

        # A configured-map task may deliberately carry a coordinate prior so an
        # integration test measures ROS intent/task/actuator/result delivery,
        # rather than open-vocabulary detector recall.  The prior never comes
        # from MuJoCo scene geometry: it must arrive in the robot-scoped task
        # command and is verified against this robot's FAST-LIO pose.
        if target_prior_xy is not None:
            while time.monotonic() < deadline and frontiers_attempted < 12:
                pose = self.slam.pose
                if pose is not None:
                    dx = target_prior_xy[0] - pose.x
                    dy = target_prior_xy[1] - pose.y
                    distance = math.hypot(dx, dy)
                    bearing = wrap_angle(math.atan2(dy, dx) - pose.yaw)
                    if distance <= standoff_m + 0.35:
                        self.stop()
                        return TaskResult(
                            "coordinate_reached",
                            True,
                            self.stationary_confirmed,
                            "ros_goal_prior+fastlio2+mid360",
                            target,
                            time.monotonic() - started,
                            distance,
                            math.degrees(bearing),
                            reason="ROS task coordinate prior reached",
                        )
                path = self.slam.frontier_path(goal_hint=target_prior_xy)
                if not path:
                    break
                frontiers_attempted += 1
                if not self.navigate_path(path, deadline=deadline):
                    # The safety gate may stop on the target-side standoff
                    # boundary.  Re-enter through the FAST-LIO distance check
                    # before deciding this frontier was a failure.
                    continue
            self.stop()
            return TaskResult(
                (
                    "navigation_timeout"
                    if time.monotonic() >= deadline
                    else "risk_blocked"
                ),
                False,
                self.stationary_confirmed,
                "ros_goal_prior+fastlio2+mid360",
                target,
                time.monotonic() - started,
                reason=(
                    "coordinate-prior navigation did not reach the requested standoff "
                    f"after {frontiers_attempted} frontiers"
                ),
            )

        # First exhaust a measured 360-degree HIKROBOT RGB scan. Every semantic
        # decision is a real RGB model request; turning is measured from the
        # physically simulated base rather than assumed from step count.
        for _ in range(8):
            if self.cancel_requested:
                self.stop()
                return TaskResult(
                    "cancelled", False, self.stationary_confirmed, "hikrobot_rgb+mid360", target,
                    time.monotonic() - started, reason=self.cancel_reason,
                )
            detection = self.detect(target, semantic=True)
            if detection is not None:
                break
            if time.monotonic() >= deadline or not self.turn_measured(math.pi / 4):
                break

        # A full miss enters MID-360 frontier exploration.  Only observed-free
        # map cells are traversed; each reached viewpoint gets fresh RGB plus
        # a current MID-360 metric association.
        # semantic observation.
        while detection is None and time.monotonic() < deadline and frontiers_attempted < 6:
            path = self.slam.frontier_path(goal_hint=target_prior_xy)
            if not path:
                break
            frontiers_attempted += 1
            if not self.navigate_path(path, deadline=deadline):
                break
            detection = self.detect(target, semantic=True)
            if detection is not None:
                break
            for _ in range(3):
                if not self.turn_measured(math.pi / 2):
                    break
                detection = self.detect(target, semantic=True)
                if detection is not None:
                    break
        if detection is None:
            self.stop()
            return TaskResult(
                "target_not_found", False, self.stationary_confirmed, "hikrobot_rgb+mid360", target,
                time.monotonic() - started,
                reason=(
                    "HIKROBOT RGB semantic scan and MID-360 known-free frontier search "
                    f"found no target after {frontiers_attempted} frontiers"
                ),
            )

        transient_range_failures = 0
        while time.monotonic() < deadline:
            if self.cancel_requested:
                self.stop()
                return TaskResult(
                    "cancelled", False, self.stationary_confirmed, "hikrobot_rgb+mid360", target,
                    time.monotonic() - started, reason=self.cancel_reason,
                )
            detection = self.detect(target)
            if detection is None:
                failure = str(
                    self.detector.metadata().get("last_failure")
                    or "unknown_visual_failure"
                )
                if failure in {
                    "mid360_background_range_jump",
                    "mid360_bbox_association_missing",
                } and transient_range_failures < 3:
                    transient_range_failures += 1
                    self.velocity_gateway.stop(source="visual_range_dropout_hold")
                    self._step(steps=5)
                    continue
                self.stop()
                detection = self.detect(target, semantic=True)
                if detection is None:
                    failure = str(
                        self.detector.metadata().get("last_failure")
                        or "unknown_visual_failure"
                    )
                    return TaskResult(
                        "verification_failed", False, self.stationary_confirmed, "hikrobot_rgb+mid360", target,
                        time.monotonic() - started,
                        reason=f"HIKROBOT target continuity failed: {failure}",
                    )
            transient_range_failures = 0
            control_distance = detection.distance_m
            control_bearing = detection.bearing_rad
            prior_pose = self.slam.pose
            if target_prior_xy is not None and prior_pose is not None:
                prior_dx = target_prior_xy[0] - prior_pose.x
                prior_dy = target_prior_xy[1] - prior_pose.y
                control_distance = math.hypot(prior_dx, prior_dy)
                control_bearing = wrap_angle(
                    math.atan2(prior_dy, prior_dx) - prior_pose.yaw
                )
            distance_error = control_distance - standoff_m
            if abs(control_bearing) <= math.radians(8.0) and distance_error <= 0.12:
                self.stop()
                verified = self.detect(target, semantic=True)
                if verified is None:
                    failure = str(
                        self.detector.metadata().get("last_failure")
                        or "unknown_visual_failure"
                    )
                    return TaskResult(
                        "verification_failed", False, self.stationary_confirmed,
                        "hikrobot_rgb+mid360", target, time.monotonic() - started,
                        reason=f"stationary target verification failed: {failure}",
                    )
                prior_verified = target_prior_xy is not None and (
                    control_distance <= standoff_m + 0.25
                    and abs(control_bearing) <= math.radians(12.0)
                )
                sensor_verified = (
                    verified.distance_m <= standoff_m + 0.18
                    and abs(verified.bearing_rad) <= math.radians(12.0)
                )
                if self.stationary_confirmed and (prior_verified or sensor_verified):
                    return TaskResult(
                        "arrived_verified", True, True, "hikrobot_rgb+mid360", target,
                        time.monotonic() - started,
                        control_distance if prior_verified else verified.distance_m,
                        (
                            math.degrees(control_bearing)
                            if prior_verified
                            else math.degrees(verified.bearing_rad)
                        ),
                    )
                # A fresh semantic box can shift relative to the continuous
                # tracker.  Keep the robot stopped while reacquiring, then use
                # that verified observation to centre the target instead of
                # turning a small residual bearing error into a terminal task
                # failure.
                detection = verified
                if target_prior_xy is None:
                    control_distance = detection.distance_m
                    control_bearing = detection.bearing_rad
                    distance_error = control_distance - standoff_m
            # Camera image-right is robot -Y and needs sign inversion.  A
            # world-frame prior bearing already follows positive base yaw.
            yaw_gain = 1.4 if target_prior_xy is not None else -1.4
            yaw_rate = float(np.clip(yaw_gain * control_bearing, -0.45, 0.45))
            linear = 0.0 if abs(control_bearing) > math.radians(18.0) else float(
                np.clip(0.45 * distance_error, 0.0, 0.24)
            )
            self.drive_velocity(
                min(linear, 0.20),
                0.0,
                yaw_rate,
                source="hikrobot_mid360_object_approach",
                steps=5,
            )
        self.stop()
        return TaskResult(
            "navigation_timeout", False, self.stationary_confirmed, "hikrobot_rgb+mid360", target,
            time.monotonic() - started, reason="object approach exceeded its bounded deadline",
        )

    @staticmethod
    def _inspection_anomaly_visible(color: np.ndarray[Any, Any]) -> bool:
        """Detect the large magenta inspection marker without open-vocabulary VLM."""

        pixels = np.asarray(color, dtype=np.int16)
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            return False
        red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        mask = (red >= 170) & (blue >= 105) & (green <= 95)
        lower_half = mask[mask.shape[0] // 3 :, :]
        return int(np.count_nonzero(lower_half)) >= 180

    def inspect_warehouse_region(
        self,
        region: str,
        *,
        timeout_s: float = 480.0,
    ) -> TaskResult:
        """Follow a robot-scoped patrol route while building the shared map."""

        normalized = str(region).strip().lower()
        if normalized not in GO2_INSPECTION_ROUTES:
            raise ValueError("仓库巡检区域必须是 north 或 south")
        started = time.monotonic()
        deadline = started + timeout_s
        self.cancel_requested = False
        self.cancel_reason = "operator stop requested"
        route = GO2_INSPECTION_ROUTES[normalized]
        completed_waypoints = 0
        anomaly_detected = False
        failure_reason: str | None = None

        for goal in route:
            if self.cancel_requested:
                failure_reason = self.cancel_reason
                break
            pose = self.slam.pose
            if pose is None:
                self._step(steps=10)
                pose = self.slam.pose
            from harness.robots.go2.go2_semantic_map import Go2WarehouseSemanticMap
            semantic_route = Go2WarehouseSemanticMap().route((pose.x, pose.y), goal) if pose is not None else None
            points = list(semantic_route.points[::3]) if semantic_route is not None else []
            if points and points[-1] != semantic_route.points[-1]:
                points.append(semantic_route.points[-1])
            if not points or not self.navigate_path(points, deadline=deadline):
                cause = "timed out" if time.monotonic() >= deadline else "was blocked"
                failure_reason = (
                    f"{normalized} inspection route {cause} before waypoint "
                    f"{completed_waypoints + 1}"
                )
                break
            completed_waypoints += 1
            color = self.capture_hikrobot_rgb()
            anomaly_detected = anomaly_detected or self._inspection_anomaly_visible(color)
            self._inspection_anomaly_detected = (
                self._inspection_anomaly_detected or anomaly_detected
            )

        self.stop()
        completed = bool(
            completed_waypoints == len(route)
            and self.stationary_confirmed
            and not self._collision_hold
            and not self.cancel_requested
        )
        if not completed and failure_reason is None:
            failure_reason = "warehouse inspection terminal verification failed"
        return TaskResult(
            "inspection_route_verified"
            if completed
            else ("cancelled" if self.cancel_requested else "inspection_route_incomplete"),
            completed,
            self.stationary_confirmed,
            "hikrobot_rgb+mid360+odometry",
            f"warehouse_{normalized}",
            time.monotonic() - started,
            reason=None if completed else failure_reason,
            inspection_region=normalized,
            route_waypoints_completed=completed_waypoints,
            anomaly_detected=anomaly_detected,
        )

    def monitor_ros_coordinate_goal(
        self,
        target: str,
        target_prior_xy: tuple[float, float],
        *,
        timeout_s: float = 240.0,
        standoff_m: float = 0.9,
        heading_tolerance_deg: float = 8.0,
    ) -> TaskResult:
        """Verify a goal while the robot-local ROS planner owns ``cmd_vel``."""

        started = time.monotonic()
        deadline = started + timeout_s
        alignment_deadline: float | None = None
        self.cancel_requested = False
        self.cancel_reason = "operator stop requested"
        final_distance: float | None = None
        final_bearing: float | None = None
        safety_hold_started: float | None = None
        safety_hold_duration = 0.0
        safety_hold_count = 0
        while not self.cancel_requested:
            now = time.monotonic()
            navigation = self.external_navigation_status or {}
            if navigation.get("status") in {"aligning", "arrived"}:
                if alignment_deadline is None:
                    alignment_deadline = now + 12.0
            if now >= deadline and (
                alignment_deadline is None or now >= alignment_deadline
            ):
                break
            if self._collision_hold:
                pose = self.slam.pose
                if pose is not None:
                    dx = target_prior_xy[0] - pose.x
                    dy = target_prior_xy[1] - pose.y
                    final_distance = math.hypot(dx, dy)
                    final_bearing = math.degrees(
                        wrap_angle(math.atan2(dy, dx) - pose.yaw)
                    )
                navigation = self.external_navigation_status or {}
                self.stop()
                return TaskResult(
                    "risk_blocked",
                    False,
                    self.stationary_confirmed,
                    "ros_local_costmap+cmd_vel+mid360",
                    target,
                    time.monotonic() - started,
                    final_distance,
                    final_bearing,
                    reason="external contact safety hold",
                    route_waypoints_completed=(
                        int(navigation.get("route_waypoints_completed", 0))
                        if navigation
                        else None
                    ),
                    last_planner_status="risk_blocked",
                    safety_hold_duration_s=float(
                        navigation.get("safety_hold_duration_s", safety_hold_duration)
                    ),
                    safety_hold_count=max(
                        safety_hold_count,
                        int(navigation.get("safety_hold_count", 0)),
                    ),
                    recovery_attempts=int(navigation.get("recovery_attempts", 0)),
                    post_hold_recovery_attempts=int(
                        navigation.get("post_hold_recovery_attempts", 0)
                    ),
                    scan_attempts=int(navigation.get("scan_attempts", 0)),
                )
            pose = self.slam.pose
            if pose is not None:
                dx = target_prior_xy[0] - pose.x
                dy = target_prior_xy[1] - pose.y
                distance = math.hypot(dx, dy)
                bearing = wrap_angle(math.atan2(dy, dx) - pose.yaw)
                final_distance = distance
                final_bearing = math.degrees(bearing)
            now = time.monotonic()
            if self._near_field_blind_hold:
                if safety_hold_started is None:
                    safety_hold_started = now
                    safety_hold_count += 1
            elif safety_hold_started is not None:
                safety_hold_duration += now - safety_hold_started
                safety_hold_started = None
            navigation = self.external_navigation_status or {}
            raw_goal = navigation.get("goal_xy")
            matching_goal = bool(
                isinstance(raw_goal, list)
                and len(raw_goal) == 2
                and math.dist(
                    (float(raw_goal[0]), float(raw_goal[1])),
                    target_prior_xy,
                ) <= 0.15
            )
            navigation_distance: float | None = None
            navigation_bearing: float | None = None
            try:
                navigation_distance = float(navigation["distance_to_goal_m"])
                navigation_bearing = float(navigation["heading_error_deg"])
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
            if not (
                navigation_distance is not None
                and navigation_bearing is not None
                and math.isfinite(navigation_distance)
                and math.isfinite(navigation_bearing)
            ):
                navigation_distance = navigation_bearing = None
            if (
                matching_goal
                and navigation.get("status") == "arrived"
                and navigation_distance is not None
                # The robot-local planner latches alignment only after it has
                # entered the 1.25 m standoff envelope, and deliberately does
                # not resume translation until three samples exceed 1.40 m.
                # Use that same hysteresis envelope for terminal verification
                # so a few centimetres of post-stop FAST-LIO drift cannot
                # deadlock an otherwise stationary, correctly aligned goal.
                and navigation_distance
                <= standoff_m
                + (0.50 if navigation.get("alignment_locked") else 0.35)
                and navigation_bearing is not None
                and abs(navigation_bearing) <= heading_tolerance_deg
            ):
                self.stop()
                # Distance and heading must come from the same robot-local
                # FAST-LIO estimate that generated cmd_vel. MuJoCo runs a
                # second estimator for its own sensor display; requiring both
                # at an 8-degree edge can deadlock on normal estimator noise.
                # The physics owner independently proves stationarity here.
                final_distance = navigation_distance
                final_bearing = navigation_bearing
                if self.stationary_confirmed:
                    return TaskResult(
                        "coordinate_reached",
                        True,
                        True,
                        "ros_local_costmap+cmd_vel+fastlio2+mid360",
                        target,
                        time.monotonic() - started,
                        final_distance,
                        final_bearing,
                        reason=(
                            "ROS local planner reached the coordinate-prior "
                            "standoff and aligned its body-fixed camera"
                        ),
                        route_waypoints_completed=int(
                            navigation.get("route_waypoints_completed", 0)
                        ),
                        last_planner_status="arrived",
                        recovery_attempts=int(
                            navigation.get("recovery_attempts", 0)
                        ),
                        post_hold_recovery_attempts=int(
                            navigation.get("post_hold_recovery_attempts", 0)
                        ),
                        scan_attempts=int(navigation.get("scan_attempts", 0)),
                    )
            if matching_goal and navigation.get("status") == "risk_blocked":
                self.stop()
                return TaskResult(
                    "risk_blocked",
                    False,
                    self.stationary_confirmed,
                    "ros_local_costmap+cmd_vel+fastlio2+mid360",
                    target,
                    time.monotonic() - started,
                    final_distance,
                    final_bearing,
                    reason=str(
                        navigation.get("reason")
                        or "robot-local bounded navigation recovery was exhausted"
                    ),
                    route_waypoints_completed=int(
                        navigation.get("route_waypoints_completed", 0)
                    ),
                    last_planner_status="risk_blocked",
                    safety_hold_duration_s=float(
                        max(
                            safety_hold_duration,
                            float(navigation.get("safety_hold_duration_s", 0.0)),
                        )
                    ),
                    safety_hold_count=max(
                        safety_hold_count,
                        int(navigation.get("safety_hold_count", 0)),
                    ),
                    recovery_attempts=int(navigation.get("recovery_attempts", 0)),
                    post_hold_recovery_attempts=int(
                        navigation.get("post_hold_recovery_attempts", 0)
                    ),
                    scan_attempts=int(navigation.get("scan_attempts", 0)),
                )
            if matching_goal and navigation.get("status") == "verification_failed":
                self.stop()
                return TaskResult(
                    "verification_failed",
                    False,
                    self.stationary_confirmed,
                    "ros_local_costmap+cmd_vel+fastlio2+mid360",
                    target,
                    time.monotonic() - started,
                    final_distance,
                    final_bearing,
                    reason=str(
                        navigation.get("reason")
                        or "target heading alignment did not converge"
                    ),
                    route_waypoints_completed=int(
                        navigation.get("route_waypoints_completed", 0)
                    ),
                    last_planner_status="verification_failed",
                    recovery_attempts=int(navigation.get("recovery_attempts", 0)),
                    post_hold_recovery_attempts=int(
                        navigation.get("post_hold_recovery_attempts", 0)
                    ),
                    scan_attempts=int(navigation.get("scan_attempts", 0)),
                )
            # The tick callback ingests the latest robot-scoped ROS Twist;
            # watchdog and collision gates remain inside the physics owner.
            self._step(steps=5)
        if safety_hold_started is not None:
            safety_hold_duration += time.monotonic() - safety_hold_started
        navigation = self.external_navigation_status or {}
        self.stop()
        return TaskResult(
            "cancelled" if self.cancel_requested else "navigation_timeout",
            False,
            self.stationary_confirmed,
            "ros_local_costmap+cmd_vel+fastlio2+mid360",
            target,
            time.monotonic() - started,
            final_distance,
            final_bearing,
            reason=(
                self.cancel_reason
                if self.cancel_requested
                else "ROS local planner did not reach the coordinate-prior standoff"
            ),
            route_waypoints_completed=(
                int(navigation.get("route_waypoints_completed", 0))
                if navigation
                else None
            ),
            last_planner_status=str(navigation.get("status") or "unknown"),
            safety_hold_duration_s=float(
                max(
                    safety_hold_duration,
                    float(navigation.get("safety_hold_duration_s", 0.0)),
                )
            ),
            safety_hold_count=max(
                safety_hold_count,
                int(navigation.get("safety_hold_count", 0)),
            ),
            recovery_attempts=int(navigation.get("recovery_attempts", 0)),
            post_hold_recovery_attempts=int(
                navigation.get("post_hold_recovery_attempts", 0)
            ),
            scan_attempts=int(navigation.get("scan_attempts", 0)),
        )

    def follow_person(
        self, *, duration_s: float = 12.0, timeout_s: float = 70.0, standoff_m: float = 1.2
    ) -> TaskResult:
        started = time.monotonic()
        self.cancel_requested = False
        self.cancel_reason = "operator stop requested"
        deadline = started + timeout_s
        total_frames = 0
        tracked_frames = 0
        last: RgbLidarDetection | None = None
        while float(self.data.time) < 1.0:
            self._step(steps=5)
        acquisition: RgbLidarDetection | None = None
        for _ in range(8):
            self._move_person()
            mujoco.mj_forward(self.model, self.data)
            acquisition = self.detect("person", semantic=True)
            if acquisition is not None:
                break
            if time.monotonic() >= deadline or not self.turn_measured(math.pi / 4):
                break
        if acquisition is None:
            self.stop()
            return TaskResult(
                "tracking_init_failed",
                False,
                self.stationary_confirmed,
                "hikrobot_rgb+mid360",
                "person",
                time.monotonic() - started,
                frame_coverage=0.0,
                reason="measured HIKROBOT RGB scan could not acquire a person",
            )
        last = acquisition
        simulation_started = float(self.data.time)
        while float(self.data.time) - simulation_started < duration_s and time.monotonic() < deadline:
            if self.cancel_requested:
                self.stop()
                return TaskResult(
                    "cancelled", False, self.stationary_confirmed, "hikrobot_rgb+mid360", "person",
                    time.monotonic() - started,
                    frame_coverage=tracked_frames / max(total_frames, 1),
                    reason=self.cancel_reason,
                )
            self._move_person()
            mujoco.mj_forward(self.model, self.data)
            total_frames += 1
            detection = self.detect("person")
            if detection is None:
                self.stop()
                return TaskResult(
                    "tracking_lost", False, self.stationary_confirmed, "hikrobot_rgb+mid360", "person",
                    time.monotonic() - started,
                    frame_coverage=tracked_frames / max(total_frames, 1),
                    reason="the locked person left the HIKROBOT RGB frame",
                )
            tracked_frames += 1
            last = detection
            distance_error = detection.distance_m - standoff_m
            yaw_rate = float(np.clip(-1.5 * detection.bearing_rad, -0.45, 0.45))
            linear = 0.0 if abs(detection.bearing_rad) > math.radians(25.0) else float(
                np.clip(0.55 * distance_error, 0.0, 0.18)
            )
            self.drive_velocity(
                linear,
                0.0,
                yaw_rate,
                source="hikrobot_mid360_person_follow",
                steps=5,
            )
        tracking_duration = float(self.data.time) - simulation_started
        self.stop()
        self._move_person()
        mujoco.mj_forward(self.model, self.data)
        # Preserve the acquired person identity through the same CSRT track.
        # A second open-vocabulary request could switch identities; the newer
        # post-stop RGB frame plus current MID-360 range is the verification.
        final = self.detect("person")
        coverage = tracked_frames / max(total_frames, 1)
        completed = bool(
            final
            and self.stationary_confirmed
            and tracking_duration >= duration_s
            and coverage >= 0.90
            and abs(final.distance_m - standoff_m) <= 0.35
            and abs(final.bearing_rad) <= math.radians(30.0)
        )
        return TaskResult(
            "follow_verified" if completed else "follow_timeout" if tracking_duration < duration_s else "verification_failed",
            completed,
            self.stationary_confirmed,
            "hikrobot_rgb+mid360",
            "person",
            time.monotonic() - started,
            final.distance_m if final else (last.distance_m if last else None),
            math.degrees(final.bearing_rad) if final else None,
            coverage,
            None if completed else "requested tracking duration or final post-stop HIKROBOT RGB/MID-360 verification failed",
            requested_duration_s=duration_s, tracking_duration_s=tracking_duration,
        )

    def execute(
        self,
        instruction: str,
        *,
        target_prior_xy: tuple[float, float] | None = None,
    ) -> TaskResult:
        request = parse_instruction(instruction)
        if request.action == "search":
            return self.search_object(
                request.target,
                standoff_m=request.standoff_m,
                target_prior_xy=target_prior_xy,
            )
        if request.action == "inspection":
            return self.inspect_warehouse_region(request.target)
        return self.follow_person(
            duration_s=request.duration_s, standoff_m=request.standoff_m
        )




class Go2UiRuntime:
    """Publish Go2 observations and accept one terminal task at a time."""

    _DIRECTIONS = (
        "front",
        "left",
        "rear",
        "right",
    )
    def __init__(self, environment: Go2MujocoEnvironment, paths: Go2RuntimePaths) -> None:
        self.environment = environment
        self.paths = paths
        self.paths.ensure()
        self.current_instruction: str | None = None
        self.current_task_status = "idle"
        self.last_result: dict[str, Any] | None = None
        self.boot_epoch = uuid.uuid4().hex
        self.last_command_sequence = 0
        self.last_stop_sequence = 0
        self.last_person_control_sequence = 0
        self.last_ros_cmd_vel_sequence = 0
        self.last_ros_navigation_status_sequence = 0
        self.external_twist_control = False
        self.frame_sequence = 0
        self.environment.set_tick_callback(self.publish)

    @staticmethod
    def _direction(angle: float) -> str:
        wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
        if -math.pi / 4 <= wrapped < math.pi / 4:
            return "front"
        if math.pi / 4 <= wrapped < 3 * math.pi / 4:
            return "left"
        if -3 * math.pi / 4 <= wrapped < -math.pi / 4:
            return "right"
        return "rear"

    @staticmethod
    def _jpeg(color: np.ndarray[Any, Any]) -> bytes:
        from PIL import Image

        output = io.BytesIO()
        Image.fromarray(color, mode="RGB").save(
            output, format="JPEG", quality=84, optimize=False
        )
        return output.getvalue()

    def _publish_cameras(self, color: np.ndarray[Any, Any]) -> None:
        atomic_write_bytes(self.paths.camera, self._jpeg(color))
        observer = self.environment.capture_operator_view()
        atomic_write_bytes(self.paths.observer, self._jpeg(observer))

    def _lidar_payload(
        self,
    ) -> tuple[dict[str, Any], float | None, np.ndarray[Any, Any]]:
        # Use the MID-360 mapping band around the horizon. The separate safety
        # scan below retains the full downward/upward pattern; near-horizontal
        # FAST-LIO input avoids feeding the ideal MuJoCo floor's thousands of
        # exactly coplanar endpoints into the IESKF surface solve.
        mapping_cloud = self.environment.mid360_point_cloud(
            azimuth_samples=180,
            elevation_deg=(-7, -4, 0, 4),
            frame="sensor",
            exclude_mapping_dynamics=True,
        )
        if len(mapping_cloud):
            mapping_cloud = mapping_cloud[
                (mapping_cloud[:, 2] > -0.24) & (mapping_cloud[:, 2] < 1.20)
            ]
        proximity_cloud = self.environment.mid360_point_cloud(
            azimuth_samples=180,
            elevation_deg=MID360_VERTICAL_CHANNELS_DEG,
            frame="sensor",
        )
        now = time.time()
        self.frame_sequence += 1
        nearest: float | None = None
        nearest_bearing: float | None = None
        sectors: dict[str, float] = {}
        if len(proximity_cloud):
            distances = np.linalg.norm(proximity_cloud[:, :2], axis=1)
            bearings = np.arctan2(proximity_cloud[:, 1], proximity_cloud[:, 0])
            nearest_index = int(np.argmin(distances))
            nearest = float(distances[nearest_index])
            nearest_bearing = math.degrees(float(bearings[nearest_index]))
            for distance, bearing in zip(distances, bearings, strict=True):
                direction = self._direction(float(bearing))
                sectors[direction] = min(
                    sectors.get(direction, math.inf), float(distance)
                )
        direction = (
            self._direction(math.radians(nearest_bearing))
            if nearest_bearing is not None
            else None
        )
        payload = {
            "schema_version": 1,
            "source": "current_lidar",
            "sequence": self.frame_sequence,
            "frame_timestamp": now,
            "pose_timestamp": now,
            "written_at": now,
            "nearest_obstacle_distance": nearest,
            "nearest_obstacle_bearing_deg": nearest_bearing,
            "nearest_obstacle_direction": direction,
            "sectors_m": sectors,
            "candidate_points": int(len(proximity_cloud)),
            "self_filter": {
                "identity_verified": True,
                "frames": 1,
                "applied": 1,
                "missing": 0,
                "invalid": 0,
                "masked_pixels": 0,
                "mode": "mujoco_body_subtree_exclusion",
            },
        }
        return payload, nearest, mapping_cloud

    def _costmap_payload(self, cloud: np.ndarray[Any, Any]) -> dict[str, Any]:
        # FAST-LIO2 owns pose estimation. This layer only serializes its
        # registered-cloud occupancy projection for the operator dashboard.
        slam = self.environment.slam
        details = slam.payload()
        encoded = np.where(slam.grid < 0, 255, slam.grid).astype(np.uint8)
        return {
            "schema_version": 1,
            "available": True,
            "running": True,
            "source": "go2_mid360_live",
            "robot_id": self.environment.robot_id,
            "timestamp": time.time(),
            "revision": slam.revision,
            "frame_id": "go2_map",
            "width": slam.width,
            "height": slam.height,
            "resolution": slam.resolution,
            "origin": {"x": slam.origin_x, "y": slam.origin_y, "yaw": 0.0},
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "cells": {
                "total": slam.width * slam.height,
                "known": details["known"],
                "free": details["free"],
                "occupied": details["occupied"],
                "saturated_height_cost": details["occupied"],
            },
            "sensor": "livox_mid360_simulated_raw+imu",
            "self_filter": "mujoco_body_subtree_exclusion",
            "estimator": (
                self.environment.fastlio.status()
                if self.environment.fastlio is not None
                else {"backend": "dimos_fastlio2_native", "ready": False}
            ),
            "slam": details,
        }

    def _publish_costmap(self, cloud: np.ndarray[Any, Any]) -> dict[str, Any]:
        payload = self._costmap_payload(cloud)
        atomic_write_json(self.paths.costmap, payload)
        return payload

    def _publish_mid360_frame(
        self,
        cloud: np.ndarray[Any, Any],
        *,
        gyro: tuple[float, float, float],
        acceleration: tuple[float, float, float],
        imu_samples: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ],
    ) -> None:
        """Expose raw simulated sensor data to the ROS-only adapter process."""

        samples = imu_samples or [(gyro, acceleration)]
        output = io.BytesIO()
        np.savez_compressed(
            output,
            schema_version=np.asarray([1], dtype=np.int32),
            sequence=np.asarray([self.frame_sequence], dtype=np.int64),
            timestamp=np.asarray([time.time()], dtype=np.float64),
            simulation_time=np.asarray(
                [float(self.environment.data.time)], dtype=np.float64
            ),
            points=np.asarray(cloud[:, :3], dtype=np.float32),
            gyro=np.asarray([sample[0] for sample in samples], dtype=np.float64),
            acceleration=np.asarray(
                [sample[1] for sample in samples], dtype=np.float64
            ),
        )
        atomic_write_bytes(self.paths.mid360_frame, output.getvalue())

    def _check_ros_cmd_vel(self) -> None:
        """Apply one fresh, robot-scoped Twist written by the ROS adapter."""

        payload = read_json(self.paths.ros_cmd_vel)
        if payload is None or payload.get("schema_version") != 1:
            return
        try:
            sequence = int(payload["sequence"])
            robot_id = str(payload["robot_id"])
            expires_at = float(payload["expires_at"])
            twist = dict(payload["twist"])
            linear = dict(twist["linear"])
            angular = dict(twist["angular"])
            values = (
                float(linear.get("x", 0.0)),
                float(linear.get("y", 0.0)),
                float(angular.get("z", 0.0)),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if (
            sequence <= self.last_ros_cmd_vel_sequence
            or robot_id != self.environment.robot_id
            or not math.isfinite(expires_at)
            or expires_at < time.time()
            or expires_at - time.time() > 0.30
            or not all(math.isfinite(value) for value in values)
        ):
            return
        self.last_ros_cmd_vel_sequence = sequence
        # A terminal simulator skill owns its local controller until it exits.
        # External Twist cannot preempt it or bypass its collision checks.
        if self.current_instruction is not None and not self.external_twist_control:
            return
        self.environment.command_velocity(
            *values,
            source=f"ros2:{robot_id}",
        )

    def _check_ros_navigation_status(self) -> None:
        payload = read_json(self.paths.ros_navigation_status)
        if payload is None or payload.get("schema_version") != 1:
            return
        try:
            sequence = int(payload["sequence"])
            robot_id = str(payload["robot_id"])
            status = str(payload["status"])
            goal = tuple(float(value) for value in payload["goal_xy"])
            written_at = float(payload["written_at"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if (
            (
                sequence <= self.last_ros_navigation_status_sequence
                and self.environment.external_navigation_status is not None
            )
            or robot_id != self.environment.robot_id
            or len(goal) != 2
            or not all(math.isfinite(value) for value in (*goal, written_at))
            or abs(time.time() - written_at) > 1.5
            or status not in {
                "tracking",
                "aligning",
                "scanning",
                "recovering",
                "recovery_waiting",
                "safety_hold",
                "path_unavailable",
                "path_invalidated",
                "motion_disabled",
                "arrived",
                "risk_blocked",
                "verification_failed",
            }
        ):
            return
        self.last_ros_navigation_status_sequence = sequence
        self.environment.external_navigation_status = dict(payload)

    def _check_stop(self) -> None:
        payload = read_json(self.paths.stop)
        if payload is None or payload.get("schema_version") != 1:
            return
        try:
            sequence = int(payload["sequence"])
        except (KeyError, TypeError, ValueError):
            return
        if sequence > self.last_stop_sequence:
            self.last_stop_sequence = sequence
            self.environment.request_stop()

    def _check_person_control(self) -> None:
        payload = read_json(self.paths.person_control)
        if payload is None or payload.get("schema_version") != 1:
            return
        try:
            sequence = int(payload["sequence"])
            mode = str(payload["mode"])
            expires_at = float(payload.get("expires_at", 0.0))
            forward = float(payload.get("forward", 0.0))
            turn = float(payload.get("turn", 0.0))
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if sequence <= self.last_person_control_sequence:
            return
        if (
            mode not in {"auto", "manual", "paused"}
            or not all(math.isfinite(value) for value in (expires_at, forward, turn))
            or not (-1.0 <= forward <= 1.0 and -1.0 <= turn <= 1.0)
        ):
            return
        self.last_person_control_sequence = sequence
        self.environment.set_person_control(
            mode,
            forward=forward,
            turn=turn,
            expires_at=expires_at,
        )

    def publish(self, _environment: Go2MujocoEnvironment | None = None) -> None:
        self._check_stop()
        self._check_person_control()
        self._check_ros_cmd_vel()
        self._check_ros_navigation_status()
        color = self.environment.capture_hikrobot_rgb()
        self._publish_cameras(color)
        lidar, nearest, cloud = self._lidar_payload()
        atomic_write_json(self.paths.lidar_proximity, lidar)
        position, quaternion = self.environment.base_pose()
        gyro_address = int(
            self.environment.model.sensor_adr[self.environment._gyro_sensor_id]
        )
        acceleration_address = int(
            self.environment.model.sensor_adr[
                self.environment._accelerometer_sensor_id
            ]
        )
        gyro = tuple(
            float(value)
            for value in self.environment.data.sensordata[gyro_address : gyro_address + 3]
        )
        acceleration = tuple(
            float(value)
            for value in self.environment.data.sensordata[
                acceleration_address : acceleration_address + 3
            ]
        )
        acceleration_norm = math.sqrt(
            sum(value * value for value in acceleration)
        )
        if acceleration_norm < 1.0:
            # mj_forward has no acceleration history. A stationary physical
            # MID-360 IMU reports +g, so never initialize FAST-LIO with the
            # simulator's pre-integration near-zero placeholder.
            acceleration = (0.0, 0.0, 9.80665)
        imu_samples = self.environment._fastlio_imu_samples.copy()
        self._publish_mid360_frame(
            cloud,
            gyro=gyro,
            acceleration=acceleration,
            imu_samples=imu_samples,
        )
        fastlio = self.environment._fastlio()
        estimate = fastlio.observe(
            cloud,
            scan_start_s=1.0 + fastlio.frames * 0.10,
            gyro_xyz=gyro,
            acceleration_xyz=acceleration,
            imu_samples=imu_samples,
        )
        self.environment._fastlio_imu_samples.clear()
        if estimate.ready and estimate.pose is not None:
            registered = estimate.registered_points
            if len(registered):
                registered = registered[
                    (registered[:, 2] > -0.24) & (registered[:, 2] < 1.20)
                ]
            self.environment.slam.observe_fastlio(
                registered,
                pose=estimate.pose,
            )
        local_map = self._publish_costmap(cloud)
        state = {
                "schema_version": 1,
                "boot_epoch": self.boot_epoch,
                "backend": "mujoco-go2",
                "robot_id": self.environment.robot_id,
                "robot_model": "unitree_go2",
                "written_at": time.time(),
                "simulation_time": float(self.environment.data.time),
                "pose": {
                    "position": position,
                    "quaternion_wxyz": quaternion,
                },
                "command": list(self.environment.current_command),
                "cmd_vel": self.environment.velocity_gateway.status(),
                "sensors": {
                    "hikrobot_mv_cu013_a0uc": {
                        "rgb": True,
                        "aligned_depth": False,
                        "imu": False,
                        "resolution": list(HIKROBOT_NATIVE_IMAGE_SIZE),
                        "render_resolution": [self.environment.width, self.environment.height],
                        "max_fps": HIKROBOT_MAX_FPS,
                        "interface": "USB3.0",
                        "lens_mount": "C-mount",
                        "simulated_lens_focal_length_mm": HIKROBOT_ASSUMED_FOCAL_LENGTH_MM,
                        "horizontal_fov_degrees": HIKROBOT_HORIZONTAL_FOV_DEG,
                    },
                    "mid360_imu": {
                        "gyro": True,
                        "accelerometer": True,
                        "odometry_fused": True,
                    },
                    "mid360": {"points": lidar["candidate_points"], "native_go2_radar": False},
                    "motion_safety": {
                        "minimum_clearance_m": GO2_MIN_MOTION_CLEARANCE_M,
                        "last_directional_clearance_m": (
                            self.environment._last_directional_clearance
                            if math.isfinite(self.environment._last_directional_clearance)
                            else None
                        ),
                        "near_field_blind_hold": self.environment._near_field_blind_hold,
                        "external_contact_hold": self.environment._collision_hold,
                    },
                },
                "odometry_source": "mid360_raw_points+imu->dimos_fastlio2_native",
                "fastlio2": fastlio.status(),
                "locomotion": self.environment.locomotion.status(),
                "perception": self.environment.detector.metadata(),
                "person_control": self.environment.person_control_status(),
                "slam": self.environment.slam.payload(),
                "nearest_obstacle_distance": nearest,
                "busy": self.current_instruction is not None,
                "healthy": not self.environment._collision_hold,
                "stationary_confirmed": self.environment.stationary_confirmed,
                "workload": 1 if self.current_instruction is not None else 0,
                "last_command_sequence": self.last_command_sequence,
                "instruction": self.current_instruction,
                "task_status": self.current_task_status,
                "last_result": self.last_result,
        }
        atomic_write_json(self.paths.state, state)
        return {"state": state, "local_map": local_map}

    def _next_instruction(
        self,
    ) -> tuple[int, str, tuple[float, float] | None, str, float] | None:
        payload = read_json(self.paths.command)
        if payload is None or payload.get("schema_version") != 1:
            return None
        try:
            sequence = int(payload["sequence"])
            written_at = float(payload["written_at"])
            instruction = str(payload["instruction"]).strip()
            control_mode = str(payload.get("control_mode", "simulator_skill"))
            navigation_timeout_s = float(payload.get("navigation_timeout_s", 240.0))
            raw_prior = payload.get("target_prior_xy")
            target_prior_xy = (
                (float(raw_prior[0]), float(raw_prior[1]))
                if raw_prior is not None
                else None
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (
            sequence <= self.last_command_sequence
            or payload.get("runtime_boot_epoch") != self.boot_epoch
            or payload.get("robot_id") != "go2-01"
            or not math.isfinite(written_at) or not 0 <= time.time() - written_at <= 2.0
            or not instruction
            or control_mode not in {"simulator_skill", "ros_local_planner"}
            or not math.isfinite(navigation_timeout_s)
            or not 120.0 <= navigation_timeout_s <= 360.0
            or (
                target_prior_xy is not None
                and not all(math.isfinite(value) for value in target_prior_xy)
            )
        ):
            return None
        self.environment.external_navigation_status = None
        return (
            sequence,
            instruction,
            target_prior_xy,
            control_mode,
            navigation_timeout_s,
        )

    def prewarm_fastlio(self, *, maximum_frames: int = 8) -> None:
        """Require stationary FAST-LIO2 initialization before task motion."""

        self.environment.velocity_gateway.stop(source="fastlio2_initialization")
        # Let torque support settle before estimating gravity. The initial
        # mj_forward pose has no integrated IMU history and may still be falling.
        callback = self.environment._tick_callback
        self.environment.set_tick_callback(None)
        try:
            self.environment._step(steps=600)
            self.environment._fastlio_imu_samples.clear()
        finally:
            self.environment.set_tick_callback(callback)
        for _ in range(maximum_frames):
            self.publish()
            if (
                self.environment.fastlio is not None
                and self.environment.fastlio.status()["ready"]
            ):
                return
        raise RuntimeError(
            f"{self.environment.robot_id} FAST-LIO2 did not initialize while stationary"
        )

    def serve(self) -> None:
        for path in (self.paths.command, self.paths.stop, self.paths.person_control):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self.prewarm_fastlio()
        print("Go2 UI runtime ready", flush=True)
        while True:
            request = self._next_instruction()
            if request is None:
                self.environment._step(steps=5)
                continue
            (
                sequence,
                instruction,
                target_prior_xy,
                control_mode,
                navigation_timeout_s,
            ) = request
            self.last_command_sequence = sequence
            self.current_instruction = instruction
            self.current_task_status = "started"
            self.last_result = None
            self.publish()
            try:
                if control_mode == "ros_local_planner" and target_prior_xy is not None:
                    self.external_twist_control = True
                    parsed = parse_instruction(instruction)
                    result = self.environment.monitor_ros_coordinate_goal(
                        parsed.target,
                        target_prior_xy,
                        timeout_s=navigation_timeout_s,
                    )
                else:
                    result = self.environment.execute(
                        instruction,
                        target_prior_xy=target_prior_xy,
                    )
            except ValueError as error:
                result = TaskResult(
                    "invalid_input",
                    False,
                    True,
                    "hikrobot_rgb+mid360",
                    "",
                    0.0,
                    reason=str(error),
                )
            except Exception as error:  # noqa: BLE001 - publish safe task failure
                self.environment.stop()
                result = TaskResult(
                    "runtime_error",
                    False,
                    True,
                    "hikrobot_rgb+mid360",
                    "",
                    0.0,
                    reason=str(error)[:500],
                )
            self.last_result = result.as_dict()
            self.current_task_status = result.task_status
            self.current_instruction = None
            self.external_twist_control = False
            self.publish()




def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unitree Go2 MuJoCo demo using HIKROBOT MV-CU013-A0UC RGB and MID-360"
    )
    parser.add_argument("--command", help="例如：寻找红色方块 / 跟随这个人 10 秒")
    parser.add_argument("--headless", action="store_true", help="不打开 MuJoCo 窗口")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument(
        "--ui-runtime",
        type=Path,
        help="由网页操作台托管时使用的原子文件协议目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.width < 160 or args.height < 120:
        raise SystemExit("HIKROBOT RGB render size must be at least 160x120")
    with Go2MujocoEnvironment(width=args.width, height=args.height) as environment:
        if args.ui_runtime is not None:
            environment.realtime = True
            if not args.headless:
                environment.open_viewer()
            Go2UiRuntime(environment, Go2RuntimePaths(args.ui_runtime.resolve())).serve()
            return 0
        if not args.headless:
            environment.open_viewer()
        if args.command:
            result = environment.execute(args.command)
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
            return 0 if result.completed else 3
        print("Go2 仿真已启动。可输入：寻找红色方块 / 寻找蓝色球 / 寻找水瓶 / 跟随这个人 10秒")
        for line in sys.stdin:
            instruction = line.strip()
            if instruction.lower() in {"quit", "exit", "退出"}:
                return 0
            try:
                result = environment.execute(instruction)
                print(json.dumps(result.as_dict(), ensure_ascii=False))
            except ValueError as exc:
                print(json.dumps({"task_status": "invalid_input", "completed": False, "reason": str(exc)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
