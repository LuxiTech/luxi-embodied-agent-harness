"""Read-only Isaac G1 ports for the unified robot runtime boundary."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping
import uuid

from harness.robots.g1.isaac.isaac_protocol import (
    IsaacRuntimePaths,
    read_fresh_camera_frame,
    read_fresh_state,
)

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


ISAAC_G1_READ_ONLY_CAPABILITIES = frozenset(
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


class IsaacG1ObservationPort(ObservationPort):
    """Read atomic Isaac state/RGB-D files without opening a command path."""

    def __init__(
        self,
        paths: IsaacRuntimePaths,
        *,
        max_observation_age_s: float = 1.5,
    ) -> None:
        self.paths = paths
        self.max_observation_age_s = max(0.1, float(max_observation_age_s))
        self._boot_epoch = "unbound"
        self._motion_command_port_bound = False

    def bind_boot_epoch(self, boot_epoch: str) -> None:
        self._boot_epoch = boot_epoch

    def set_motion_command_port_bound(self, bound: bool) -> None:
        self._motion_command_port_bound = bool(bound)

    def snapshot(self, robot_id: str) -> ObservationSnapshot:
        captured_at = time.monotonic()
        wall_clock = time.time()
        state = read_fresh_state(
            self.paths,
            now=wall_clock,
            max_age_s=self.max_observation_age_s,
        )
        camera = read_fresh_camera_frame(
            self.paths,
            now=wall_clock,
            max_age_s=self.max_observation_age_s,
        )
        state_timestamp = float(state["written_at"]) if state is not None else None
        camera_timestamp = camera.timestamp if camera is not None else None
        source_timestamps = [
            timestamp
            for timestamp in (state_timestamp, camera_timestamp)
            if timestamp is not None
        ]
        freshness = (
            max(max(0.0, wall_clock - timestamp) for timestamp in source_timestamps)
            if source_timestamps
            else math.inf
        )
        values: dict[str, Any] = {
            "available": state is not None or camera is not None,
            "pose_available": state is not None,
            "camera_available": camera is not None,
            "atomic_sources": True,
            "cross_source_atomic": False,
            "motion_command_port_bound": self._motion_command_port_bound,
        }
        if state is not None:
            pose = state["pose"]
            position = pose["position"]
            quaternion = pose["quaternion_wxyz"]
            linear_velocity = state["linear_velocity_world"]
            angular_velocity = state["angular_velocity_world"]
            values["state_sequence"] = int(state["sequence"])
            values["pose"] = {
                "x_m": float(position[0]),
                "y_m": float(position[1]),
                "z_m": float(position[2]),
                "yaw_rad": _yaw_from_quaternion(quaternion),
                "quaternion_wxyz": [float(value) for value in quaternion],
                "source_timestamp": state_timestamp,
            }
            values["motion"] = {
                "planar_speed_mps": math.hypot(
                    float(linear_velocity[0]),
                    float(linear_velocity[1]),
                ),
                "yaw_rate_rps": float(angular_velocity[2]),
                "source_timestamp": state_timestamp,
            }
        if camera is not None:
            values["camera"] = {
                "sequence": camera.sequence,
                "source_timestamp": camera_timestamp,
                "width": int(camera.rgb.shape[1]),
                "height": int(camera.rgb.shape[0]),
                "aligned_depth": True,
            }
        return ObservationSnapshot(
            robot_id=robot_id,
            boot_epoch=self._boot_epoch,
            frame_id="world",
            timestamp_monotonic=captured_at,
            freshness_s=freshness,
            source="isaac-g1-atomic-files",
            calibration_revision="isaac-g1-filesystem-v1",
            values=values,
        )


class IsaacG1MappingLocalizationPort(MappingLocalizationPort):
    """Expose measured world pose and content-addressed live-map snapshots."""

    def __init__(
        self,
        paths: IsaacRuntimePaths,
        *,
        map_payload_provider: Callable[[], Mapping[str, Any]],
        artifact_root: Path,
        max_observation_age_s: float = 1.5,
    ) -> None:
        self.paths = paths
        self.map_payload_provider = map_payload_provider
        self.artifact_root = artifact_root
        self.max_observation_age_s = max(0.1, float(max_observation_age_s))

    def pose(self, robot_id: str) -> Pose2D:
        del robot_id
        state = read_fresh_state(
            self.paths,
            max_age_s=self.max_observation_age_s,
        )
        if state is None:
            raise RuntimeError("Isaac G1 odometry is unavailable")
        pose = state["pose"]
        position = pose["position"]
        return Pose2D(
            x_m=float(position[0]),
            y_m=float(position[1]),
            yaw_rad=_yaw_from_quaternion(pose["quaternion_wxyz"]),
            frame_id="world",
            timestamp_monotonic=time.monotonic(),
        )

    def map_artifact(self, robot_id: str) -> ArtifactRef | None:
        del robot_id
        try:
            snapshot = dict(self.map_payload_provider())
        except Exception:
            return None
        if snapshot.get("available") is not True:
            return None
        payload = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest = sha256(payload).hexdigest()
        target = self.artifact_root / f"{digest}.json"
        try:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = self.artifact_root / f".{digest}.{uuid.uuid4().hex}.tmp"
                try:
                    temporary.write_bytes(payload)
                    os.replace(temporary, target)
                finally:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
        except OSError:
            return None
        try:
            captured_at = float(snapshot.get("timestamp", target.stat().st_mtime))
        except (TypeError, ValueError, OverflowError, OSError):
            captured_at = time.time()
        return ArtifactRef(
            uri=str(target.resolve()),
            sha256=digest,
            media_type="application/json",
            captured_at=captured_at,
        )


class IsaacG1LifecycleHealthPort(LifecycleHealthPort):
    """Delegate lifecycle to the existing supervisor; never publish motion."""

    def __init__(
        self,
        *,
        start_runtime: Callable[[], Any],
        stop_runtime: Callable[[], Any],
        status_provider: Callable[[], Mapping[str, Any]],
        observation: IsaacG1ObservationPort,
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
        backend_valid = status.get("backend") == "isaac-g1"
        bridge_ready = status.get("bridge_ready") is True
        mcp_ready = status.get("mcp") is True
        command_center_ready = status.get("command_center") is True
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
                and mcp_ready
                and command_center_ready
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
            "mcp_ready": mcp_ready,
            "command_center_ready": command_center_ready,
        }


def build_isaac_g1_readonly_adapter(
    *,
    paths: IsaacRuntimePaths,
    map_payload_provider: Callable[[], Mapping[str, Any]],
    artifact_root: Path,
    start_runtime: Callable[[], Any],
    stop_runtime: Callable[[], Any],
    status_provider: Callable[[], Mapping[str, Any]],
    max_observation_age_s: float = 1.5,
) -> RobotWorldAdapter:
    """Bind only observation/localization/lifecycle ports for this phase."""

    observation = IsaacG1ObservationPort(
        paths,
        max_observation_age_s=max_observation_age_s,
    )
    mapping = IsaacG1MappingLocalizationPort(
        paths,
        map_payload_provider=map_payload_provider,
        artifact_root=artifact_root,
        max_observation_age_s=max_observation_age_s,
    )
    lifecycle = IsaacG1LifecycleHealthPort(
        start_runtime=start_runtime,
        stop_runtime=stop_runtime,
        status_provider=status_provider,
        observation=observation,
        max_observation_age_s=max_observation_age_s,
    )
    return RobotWorldAdapter(
        adapter_id="isaac-g1-readonly-v1",
        robot_type="g1",
        backend="isaac-g1",
        ports=RobotPorts(
            observation=observation,
            mapping_localization=mapping,
            navigation=None,
            motion=None,
            manipulation_entity=None,
            lifecycle_health=lifecycle,
        ),
        capabilities=ISAAC_G1_READ_ONLY_CAPABILITIES,
    )
