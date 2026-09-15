"""Read-only MuJoCo G1 ports for the unified robot runtime boundary."""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol

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


MUJOCO_G1_READ_ONLY_CAPABILITIES = frozenset(
    {
        "robot.observation",
        "robot.localization.pose",
        "robot.mapping.snapshot",
        "robot.lifecycle.health",
    }
)


class MujocoProbe(Protocol):
    def pose(self) -> Any | None: ...

    def camera_path(self) -> Path | None: ...


def _source_freshness(timestamp: float | None) -> float:
    if timestamp is None or not math.isfinite(timestamp):
        return math.inf
    # Existing MuJoCo shared-memory odometry uses Unix time. A monotonic
    # capture timestamp is still used by the versioned port envelope.
    return max(0.0, time.time() - timestamp)


class MujocoG1ObservationPort(ObservationPort):
    """Project existing shared-memory evidence without opening a command path."""

    def __init__(self, probe: MujocoProbe) -> None:
        self.probe = probe
        self._boot_epoch = "unbound"
        self._motion_command_port_bound = False

    def bind_boot_epoch(self, boot_epoch: str) -> None:
        self._boot_epoch = boot_epoch

    def set_motion_command_port_bound(self, bound: bool) -> None:
        self._motion_command_port_bound = bool(bound)

    def snapshot(self, robot_id: str) -> ObservationSnapshot:
        captured_at = time.monotonic()
        pose = self.probe.pose()
        camera_path = self.probe.camera_path()
        source_timestamp = (
            float(pose.timestamp) if pose is not None else None
        )
        values: dict[str, Any] = {
            "available": pose is not None or camera_path is not None,
            "pose_available": pose is not None,
            "camera_available": camera_path is not None,
            "atomic": False,
            "motion_command_port_bound": self._motion_command_port_bound,
        }
        if pose is not None:
            values["pose"] = {
                "x_m": float(pose.x),
                "y_m": float(pose.y),
                "z_m": float(pose.z),
                "yaw_rad": float(pose.yaw),
                "quaternion_wxyz": [
                    float(pose.qw),
                    float(pose.qx),
                    float(pose.qy),
                    float(pose.qz),
                ],
                "source_timestamp": source_timestamp,
            }
        return ObservationSnapshot(
            robot_id=robot_id,
            boot_epoch=self._boot_epoch,
            frame_id="map",
            timestamp_monotonic=captured_at,
            freshness_s=_source_freshness(source_timestamp),
            source="mujoco-g1-shared-memory",
            calibration_revision="mujoco-g1-existing-v1",
            values=values,
        )


class MujocoG1MappingLocalizationPort(MappingLocalizationPort):
    def __init__(
        self,
        probe: MujocoProbe,
        *,
        map_path_provider: Callable[[], Path | None],
    ) -> None:
        self.probe = probe
        self.map_path_provider = map_path_provider

    def pose(self, robot_id: str) -> Pose2D:
        del robot_id
        pose = self.probe.pose()
        if pose is None:
            raise RuntimeError("MuJoCo G1 odometry is unavailable")
        return Pose2D(
            x_m=float(pose.x),
            y_m=float(pose.y),
            yaw_rad=float(pose.yaw),
            frame_id="map",
            timestamp_monotonic=time.monotonic(),
        )

    def map_artifact(self, robot_id: str) -> ArtifactRef | None:
        del robot_id
        path = self.map_path_provider()
        if path is None:
            return None
        try:
            payload = path.read_bytes()
            stat = path.stat()
        except OSError:
            return None
        return ArtifactRef(
            uri=str(path.resolve()),
            sha256=sha256(payload).hexdigest(),
            media_type="application/gzip+json",
            captured_at=float(stat.st_mtime),
        )


class MujocoG1LifecycleHealthPort(LifecycleHealthPort):
    """Delegate lifecycle to the existing supervisor; never publish motion."""

    def __init__(
        self,
        *,
        start_runtime: Callable[[], Any],
        stop_runtime: Callable[[], Any],
        status_provider: Callable[[], Mapping[str, Any]],
        observation: MujocoG1ObservationPort,
        max_observation_age_s: float = 1.5,
    ) -> None:
        self.start_runtime = start_runtime
        self.stop_runtime = stop_runtime
        self.status_provider = status_provider
        self.observation = observation
        self.max_observation_age_s = max(0.1, float(max_observation_age_s))
        self._attached = False
        self._start_requested = False
        self._start_rejected = False

    def start(self, robot_id: str, boot_epoch: str) -> None:
        del robot_id
        self._attached = True
        self.observation.bind_boot_epoch(boot_epoch)
        result = self.start_runtime()
        self._start_requested = result is True
        self._start_rejected = result is False

    def stop(self, robot_id: str, boot_epoch: str) -> None:
        del robot_id, boot_epoch
        self.stop_runtime()
        self._attached = False
        self._start_requested = False
        self._start_rejected = False

    def health(self, robot_id: str) -> Mapping[str, Any]:
        status = dict(self.status_provider())
        backend_valid = status.get("backend") == "mujoco"
        bridge_ready = status.get("bridge_ready") is True
        observation = self.observation.snapshot(robot_id=robot_id)
        pose_ready = observation.values.get("pose_available") is True
        camera_ready = observation.values.get("camera_available") is True
        observation_fresh = bool(
            pose_ready
            and camera_ready
            and math.isfinite(observation.freshness_s)
            and observation.freshness_s <= self.max_observation_age_s
        )
        startup_failed = bool(
            self._start_rejected
            or (
                self._start_requested
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
                and observation_fresh
            ),
            "attached": self._attached,
            "backend_valid": backend_valid,
            "bridge_ready": bridge_ready,
            "observation_ready": observation_fresh,
            "pose_ready": pose_ready,
            "camera_ready": camera_ready,
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
            "mcp_ready": bool(status.get("mcp")),
            "command_center_ready": bool(status.get("command_center")),
        }


def build_mujoco_g1_readonly_adapter(
    *,
    probe: MujocoProbe,
    map_path_provider: Callable[[], Path | None],
    start_runtime: Callable[[], Any],
    stop_runtime: Callable[[], Any],
    status_provider: Callable[[], Mapping[str, Any]],
    max_observation_age_s: float = 1.5,
) -> RobotWorldAdapter:
    """Bind only observation/localization/lifecycle ports for this phase."""

    observation = MujocoG1ObservationPort(probe)
    mapping = MujocoG1MappingLocalizationPort(
        probe,
        map_path_provider=map_path_provider,
    )
    lifecycle = MujocoG1LifecycleHealthPort(
        start_runtime=start_runtime,
        stop_runtime=stop_runtime,
        status_provider=status_provider,
        observation=observation,
        max_observation_age_s=max_observation_age_s,
    )
    return RobotWorldAdapter(
        adapter_id="mujoco-g1-readonly-v1",
        robot_type="g1",
        backend="mujoco",
        ports=RobotPorts(
            observation=observation,
            mapping_localization=mapping,
            navigation=None,
            motion=None,
            manipulation_entity=None,
            lifecycle_health=lifecycle,
        ),
        capabilities=MUJOCO_G1_READ_ONLY_CAPABILITIES,
    )
