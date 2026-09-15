"""Read-only MuJoCo Go2 ports for the unified robot runtime boundary."""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from harness.robots.go2.go2_protocol import Go2RuntimePaths, read_fresh_go2_state

from .contracts import ArtifactRef
from .ports import (
    LifecycleHealthPort,
    MappingLocalizationPort,
    ObservationPort,
    ObservationSnapshot,
    Pose2D,
    RobotPorts,
    RobotWorldAdapter,
)


GO2_READ_ONLY_CAPABILITIES = frozenset(
    {
        "robot.observation",
        "robot.localization.pose",
        "robot.mapping.snapshot",
        "robot.lifecycle.health",
    }
)


def _yaw_from_quaternion(quaternion: list[Any]) -> float:
    qw, qx, qy, qz = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _camera_freshness(path: Path, *, now: float) -> float:
    try:
        stat = path.stat()
    except OSError:
        return math.inf
    if stat.st_size <= 0:
        return math.inf
    return max(0.0, now - float(stat.st_mtime))


class Go2ObservationPort(ObservationPort):
    """Project one robot's atomic state and RGB artifacts without commands."""

    def __init__(
        self,
        paths: Go2RuntimePaths,
        *,
        robot_id: str,
        max_observation_age_s: float = 1.5,
    ) -> None:
        self.paths = paths
        self.robot_id = robot_id
        self.max_observation_age_s = max(0.1, float(max_observation_age_s))
        self._boot_epoch = "unbound"
        self._motion_command_port_bound = False

    def bind_boot_epoch(self, boot_epoch: str) -> None:
        self._boot_epoch = boot_epoch

    def set_motion_command_port_bound(self, bound: bool) -> None:
        self._motion_command_port_bound = bool(bound)

    def snapshot(self, robot_id: str) -> ObservationSnapshot:
        if robot_id != self.robot_id:
            raise PermissionError("Go2 observation robot mismatch")
        captured_at = time.monotonic()
        wall_clock = time.time()
        state = read_fresh_go2_state(
            self.paths,
            max_age_s=self.max_observation_age_s,
        )
        if state is not None and state.get("robot_id", self.robot_id) != self.robot_id:
            state = None
        camera_age = _camera_freshness(self.paths.camera, now=wall_clock)
        camera_available = camera_age <= self.max_observation_age_s
        state_age = (
            max(0.0, wall_clock - float(state["written_at"]))
            if state is not None
            else math.inf
        )
        freshness = max(state_age, camera_age)
        values: dict[str, Any] = {
            "available": state is not None or camera_available,
            "pose_available": state is not None,
            "camera_available": camera_available,
            "atomic_sources": True,
            "cross_source_atomic": False,
            "aligned_depth": False,
            "motion_command_port_bound": self._motion_command_port_bound,
        }
        if state is not None:
            values["motion_healthy"] = state.get("healthy") is True
            pose = state["pose"]
            position = pose["position"]
            quaternion = pose["quaternion_wxyz"]
            values["pose"] = {
                "x_m": float(position[0]),
                "y_m": float(position[1]),
                "z_m": float(position[2]),
                "yaw_rad": _yaw_from_quaternion(quaternion),
                "quaternion_wxyz": [float(value) for value in quaternion],
                "source_timestamp": float(state["written_at"]),
            }
            values["task"] = {
                "busy": bool(state.get("busy")),
                "task_status": state.get("task_status"),
            }
        return ObservationSnapshot(
            robot_id=robot_id,
            boot_epoch=self._boot_epoch,
            frame_id="map",
            timestamp_monotonic=captured_at,
            freshness_s=freshness,
            source="mujoco-go2-atomic-files",
            calibration_revision="go2-hikrobot-mid360-v1",
            values=values,
        )


class Go2MappingLocalizationPort(MappingLocalizationPort):
    def __init__(
        self,
        paths: Go2RuntimePaths,
        *,
        robot_id: str,
        max_observation_age_s: float = 1.5,
    ) -> None:
        self.paths = paths
        self.robot_id = robot_id
        self.max_observation_age_s = max(0.1, float(max_observation_age_s))

    def _state(self, robot_id: str) -> Mapping[str, Any]:
        if robot_id != self.robot_id:
            raise PermissionError("Go2 localization robot mismatch")
        state = read_fresh_go2_state(
            self.paths,
            max_age_s=self.max_observation_age_s,
        )
        if state is None or state.get("robot_id", self.robot_id) != self.robot_id:
            raise RuntimeError("Go2 odometry is unavailable")
        return state

    def pose(self, robot_id: str) -> Pose2D:
        state = self._state(robot_id)
        pose = state["pose"]
        position = pose["position"]
        return Pose2D(
            x_m=float(position[0]),
            y_m=float(position[1]),
            yaw_rad=_yaw_from_quaternion(pose["quaternion_wxyz"]),
            frame_id="map",
            timestamp_monotonic=time.monotonic(),
        )

    def map_artifact(self, robot_id: str) -> ArtifactRef | None:
        if robot_id != self.robot_id:
            raise PermissionError("Go2 map robot mismatch")
        try:
            payload = self.paths.costmap.read_bytes()
            stat = self.paths.costmap.stat()
        except OSError:
            return None
        return ArtifactRef(
            uri=str(self.paths.costmap.resolve()),
            sha256=sha256(payload).hexdigest(),
            media_type="application/json",
            captured_at=float(stat.st_mtime),
        )


class Go2LifecycleHealthPort(LifecycleHealthPort):
    """Attach one robot to the single-robot supervisor without motion."""

    def __init__(
        self,
        *,
        robot_id: str,
        observation: Go2ObservationPort,
        start_runtime: Callable[[], Any],
        stop_runtime: Callable[[], Any],
        status_provider: Callable[[], Mapping[str, Any]],
        owns_lifecycle: bool,
        max_observation_age_s: float = 1.5,
    ) -> None:
        self.robot_id = robot_id
        self.observation = observation
        self.start_runtime = start_runtime
        self.stop_runtime = stop_runtime
        self.status_provider = status_provider
        self.owns_lifecycle = bool(owns_lifecycle)
        self.max_observation_age_s = max(0.1, float(max_observation_age_s))
        self._attached = False
        self._start_requested = False
        self._start_rejected = False

    def start(self, robot_id: str, boot_epoch: str) -> None:
        if robot_id != self.robot_id:
            raise PermissionError("Go2 lifecycle robot mismatch")
        self._attached = True
        self.observation.bind_boot_epoch(boot_epoch)
        if self.owns_lifecycle:
            result = self.start_runtime()
            self._start_requested = result is True
            self._start_rejected = result is False

    def stop(self, robot_id: str, boot_epoch: str) -> None:
        del boot_epoch
        if robot_id != self.robot_id:
            raise PermissionError("Go2 lifecycle robot mismatch")
        if self.owns_lifecycle:
            self.stop_runtime()
        self._attached = False
        self._start_requested = False
        self._start_rejected = False

    def health(self, robot_id: str) -> Mapping[str, Any]:
        if robot_id != self.robot_id:
            raise PermissionError("Go2 health robot mismatch")
        status = dict(self.status_provider())
        observation = self.observation.snapshot(robot_id)
        observation_ready = bool(
            observation.values.get("pose_available") is True
            and observation.values.get("camera_available") is True
            and math.isfinite(observation.freshness_s)
            and observation.freshness_s <= self.max_observation_age_s
        )
        backend_valid = status.get("backend") == "mujoco-go2"
        bridge_ready = status.get("bridge_ready") is True
        startup_failed = bool(
            self._start_rejected
            or (
                self.owns_lifecycle
                and self._start_requested
                and not status.get("starting")
                and not bridge_ready
                and not status.get("process_alive")
            )
        )
        return {
            "ready": bool(
                self._attached
                and backend_valid
                and bridge_ready
                and observation_ready
                and (not observation.values.get("motion_command_port_bound")
                     or observation.values.get("motion_healthy") is True)
            ),
            "attached": self._attached,
            "backend_valid": backend_valid,
            "bridge_ready": bridge_ready,
            "observation_ready": observation_ready,
            "motion_healthy": observation.values.get("motion_healthy") is True,
            "pose_ready": observation.values.get("pose_available") is True,
            "camera_ready": observation.values.get("camera_available") is True,
            "observation_age_s": (
                round(observation.freshness_s, 3)
                if math.isfinite(observation.freshness_s)
                else None
            ),
            "startup_failed": startup_failed,
            "start_requested": self._start_requested,
            "process_alive": bool(status.get("process_alive")),
            "owned_by_ui": bool(status.get("owned_by_ui")),
            "starting": bool(status.get("starting")),
            "single_robot": True,
            "lifecycle_owner": self.owns_lifecycle,
        }


def build_go2_readonly_adapter(
    *,
    paths: Go2RuntimePaths,
    robot_id: str,
    start_runtime: Callable[[], Any],
    stop_runtime: Callable[[], Any],
    status_provider: Callable[[], Mapping[str, Any]],
    owns_lifecycle: bool,
    max_observation_age_s: float = 1.5,
) -> RobotWorldAdapter:
    observation = Go2ObservationPort(
        paths,
        robot_id=robot_id,
        max_observation_age_s=max_observation_age_s,
    )
    mapping = Go2MappingLocalizationPort(
        paths,
        robot_id=robot_id,
        max_observation_age_s=max_observation_age_s,
    )
    lifecycle = Go2LifecycleHealthPort(
        robot_id=robot_id,
        observation=observation,
        start_runtime=start_runtime,
        stop_runtime=stop_runtime,
        status_provider=status_provider,
        owns_lifecycle=owns_lifecycle,
        max_observation_age_s=max_observation_age_s,
    )
    return RobotWorldAdapter(
        adapter_id=f"mujoco-{robot_id}-readonly-v1",
        robot_type="go2",
        backend="mujoco-go2",
        ports=RobotPorts(
            observation=observation,
            mapping_localization=mapping,
            navigation=None,
            motion=None,
            manipulation_entity=None,
            lifecycle_health=lifecycle,
        ),
        capabilities=GO2_READ_ONLY_CAPABILITIES,
    )
