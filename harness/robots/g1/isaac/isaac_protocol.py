"""Filesystem protocol shared by the host DimOS adapter and Isaac Sim.

The Isaac process runs in a container, so the protocol deliberately uses one
small bind-mounted runtime directory instead of importing either runtime into
the other.  JSON commands/state and atomic NumPy frame archives keep the
boundary inspectable and make stale evidence straightforward to reject.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping
import uuid

import numpy as np


SCHEMA_VERSION = 1
BACKEND_NAME = "isaac-g1"
RUNTIME_DIR_ENV = "LUXI_ISAAC_RUNTIME_DIR"
MAX_JSON_BYTES = 64 * 1024
MAX_CAMERA_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_OBSERVER_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_LIDAR_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_LIDAR_POINTS = 2_000_000


def configured_runtime_dir(environment: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environment is None else environment
    configured = values.get(RUNTIME_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    dimos_runtime = Path(
        values.get(
            "DIMOS_RUNTIME_DIR",
            str(Path.home() / "work/Asset/dimos/runtime"),
        )
    )
    return (dimos_runtime / "luxi-isaac-g1").expanduser().resolve()


@dataclass(frozen=True)
class IsaacRuntimePaths:
    root: Path

    @classmethod
    def configured(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "IsaacRuntimePaths":
        return cls(configured_runtime_dir(environment))

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def command(self) -> Path:
        return self.root / "command.json"

    @property
    def safety_hold(self) -> Path:
        return self.root / "safety-hold.json"

    @property
    def camera(self) -> Path:
        return self.root / "head-camera.npz"

    @property
    def observer(self) -> Path:
        return self.root / "observer-camera.npz"

    @property
    def entities(self) -> Path:
        return self.root / "entities.json"

    @property
    def manipulation_request(self) -> Path:
        return self.root / "manipulation-request.json"

    @property
    def manipulation_ack(self) -> Path:
        return self.root / "manipulation-ack.json"

    @property
    def entity_request(self) -> Path:
        return self.root / "entity-request.json"

    @property
    def entity_ack(self) -> Path:
        return self.root / "entity-ack.json"

    @property
    def person_command(self) -> Path:
        return self.root / "person-command.json"

    @property
    def lidar(self) -> Path:
        return self.root / "lidar.npz"

    @property
    def lidar_proximity(self) -> Path:
        return self.root / "lidar-proximity.json"

    @property
    def reset_request(self) -> Path:
        return self.root / "request.json"

    @property
    def reset_ack(self) -> Path:
        return self.root / "ack.json"

    @property
    def simulator_log(self) -> Path:
        return self.root / "simulator.log"


@dataclass(frozen=True)
class IsaacCameraFrame:
    timestamp: float
    sequence: int
    rgb: np.ndarray[Any, Any]
    depth: np.ndarray[Any, Any]
    intrinsics: np.ndarray[Any, Any]
    position: np.ndarray[Any, Any]
    quaternion_wxyz: np.ndarray[Any, Any]


@dataclass(frozen=True)
class IsaacObserverFrame:
    timestamp: float
    sequence: int
    rgb: np.ndarray[Any, Any]


@dataclass(frozen=True)
class IsaacLidarFrame:
    timestamp: float
    scan_started_at: float
    sequence: int
    points: np.ndarray[Any, Any]
    frame_id: str
    robot_prim_path: str
    raw_return_count: int
    resolved_return_count: int
    self_return_count: int
    retained_return_count: int
    invalid_return_count: int


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _finite_sequence(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        sequence = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return sequence if sequence >= 1 else None


def _finite_vector(value: Any, size: int) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != size:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def validated_state(
    payload: Mapping[str, Any] | None,
    *,
    now: float | None = None,
    max_age_s: float = 1.5,
) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("backend") != BACKEND_NAME
        or payload.get("ready") is not True
        or _finite_sequence(payload.get("sequence")) is None
    ):
        return None
    try:
        written_at = float(payload["written_at"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    clock = time.time() if now is None else float(now)
    age = clock - written_at
    if not math.isfinite(age) or age < -0.25 or age > max(0.1, float(max_age_s)):
        return None
    pose = payload.get("pose")
    if not isinstance(pose, Mapping):
        return None
    position = _finite_vector(pose.get("position"), 3)
    quaternion = _finite_vector(pose.get("quaternion_wxyz"), 4)
    linear_velocity = _finite_vector(payload.get("linear_velocity_world"), 3)
    angular_velocity = _finite_vector(payload.get("angular_velocity_world"), 3)
    if None in (position, quaternion, linear_velocity, angular_velocity):
        return None
    quaternion_norm = math.sqrt(sum(component * component for component in quaternion or ()))
    if not 0.9 <= quaternion_norm <= 1.1:
        return None
    return dict(payload)


def read_fresh_state(
    paths: IsaacRuntimePaths,
    *,
    now: float | None = None,
    max_age_s: float = 1.5,
) -> dict[str, Any] | None:
    return validated_state(read_json(paths.state), now=now, max_age_s=max_age_s)


def read_fresh_camera_frame(
    paths: IsaacRuntimePaths,
    *,
    now: float | None = None,
    max_age_s: float = 1.5,
) -> IsaacCameraFrame | None:
    """Read one complete RGB-D archive and reject stale or malformed data."""

    try:
        if paths.camera.stat().st_size > MAX_CAMERA_ARCHIVE_BYTES:
            return None
        with np.load(paths.camera, allow_pickle=False) as archive:
            required = {
                "timestamp",
                "sequence",
                "rgb",
                "depth",
                "intrinsics",
                "position",
                "quaternion_wxyz",
            }
            if not required.issubset(archive.files):
                return None
            timestamp_array = np.asarray(archive["timestamp"])
            sequence_array = np.asarray(archive["sequence"])
            if timestamp_array.size != 1 or sequence_array.size != 1:
                return None
            timestamp = float(timestamp_array.item())
            sequence = _finite_sequence(sequence_array.item())
            rgb = np.asarray(archive["rgb"]).copy()
            depth = np.asarray(archive["depth"]).copy()
            intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64).copy()
            position = np.asarray(archive["position"], dtype=np.float64).copy()
            quaternion = np.asarray(
                archive["quaternion_wxyz"], dtype=np.float64
            ).copy()
    except (OSError, ValueError, EOFError, KeyError, TypeError, OverflowError):
        return None

    clock = time.time() if now is None else float(now)
    age = clock - timestamp
    if (
        sequence is None
        or not math.isfinite(age)
        or age < -0.25
        or age > max(0.1, float(max_age_s))
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or depth.shape != rgb.shape[:2]
        or not np.issubdtype(depth.dtype, np.floating)
        or rgb.shape[0] < 1
        or rgb.shape[1] < 1
        or rgb.shape[0] > 1080
        or rgb.shape[1] > 1920
        or intrinsics.shape != (3, 3)
        or position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(intrinsics))
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quaternion))
        or float(intrinsics[0, 0]) <= 0.0
        or float(intrinsics[1, 1]) <= 0.0
        or not 0.9 <= float(np.linalg.norm(quaternion)) <= 1.1
    ):
        return None
    invalid_depth = np.isnan(depth) | (depth < 0.0)
    if np.any(invalid_depth):
        return None
    return IsaacCameraFrame(
        timestamp=timestamp,
        sequence=sequence,
        rgb=np.ascontiguousarray(rgb),
        depth=np.ascontiguousarray(depth, dtype=np.float32),
        intrinsics=intrinsics,
        position=position,
        quaternion_wxyz=quaternion,
    )


def read_fresh_observer_frame(
    paths: IsaacRuntimePaths,
    *,
    now: float | None = None,
    max_age_s: float = 2.5,
) -> IsaacObserverFrame | None:
    """Read one fresh operator-only RGB frame from the Isaac runtime."""

    try:
        if paths.observer.stat().st_size > MAX_OBSERVER_ARCHIVE_BYTES:
            return None
        with np.load(paths.observer, allow_pickle=False) as archive:
            if not {"timestamp", "sequence", "rgb"}.issubset(archive.files):
                return None
            timestamp_array = np.asarray(archive["timestamp"])
            sequence_array = np.asarray(archive["sequence"])
            if timestamp_array.size != 1 or sequence_array.size != 1:
                return None
            timestamp = float(timestamp_array.item())
            sequence = _finite_sequence(sequence_array.item())
            rgb = np.asarray(archive["rgb"]).copy()
    except (OSError, ValueError, EOFError, KeyError, TypeError, OverflowError):
        return None

    clock = time.time() if now is None else float(now)
    age = clock - timestamp
    if (
        sequence is None
        or not math.isfinite(age)
        or age < -0.25
        or age > max(0.1, float(max_age_s))
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.shape[0] < 1
        or rgb.shape[1] < 1
        or rgb.shape[0] > 540
        or rgb.shape[1] > 960
    ):
        return None
    return IsaacObserverFrame(
        timestamp=timestamp,
        sequence=sequence,
        rgb=np.ascontiguousarray(rgb),
    )


def read_fresh_lidar_frame(
    paths: IsaacRuntimePaths,
    *,
    now: float | None = None,
    max_age_s: float = 1.5,
) -> IsaacLidarFrame | None:
    """Read only an atomically complete, stable-ID-verified RTX lidar frame."""

    try:
        if paths.lidar.stat().st_size > MAX_LIDAR_ARCHIVE_BYTES:
            return None
        with np.load(paths.lidar, allow_pickle=False) as archive:
            required = {
                "timestamp",
                "scan_started_at",
                "sequence",
                "points",
                "frame_id",
                "identity_verified",
                "identity_source",
                "robot_prim_path",
                "raw_return_count",
                "resolved_return_count",
                "self_return_count",
                "retained_return_count",
                "invalid_return_count",
            }
            if not required.issubset(archive.files):
                return None

            def scalar(name: str) -> Any:
                value = np.asarray(archive[name])
                if value.size != 1:
                    raise ValueError(f"{name} must be scalar")
                return value.item()

            timestamp = float(scalar("timestamp"))
            scan_started_at = float(scalar("scan_started_at"))
            sequence = _finite_sequence(scalar("sequence"))
            points = np.asarray(archive["points"]).copy()
            frame_id = scalar("frame_id")
            identity_verified = scalar("identity_verified")
            identity_source = scalar("identity_source")
            robot_prim_path = scalar("robot_prim_path")
            counts = {
                name: scalar(name)
                for name in (
                    "raw_return_count",
                    "resolved_return_count",
                    "self_return_count",
                    "retained_return_count",
                    "invalid_return_count",
                )
            }
    except (OSError, ValueError, EOFError, KeyError, TypeError, OverflowError):
        return None

    normalized_counts: dict[str, int] = {}
    for name, value in counts.items():
        if isinstance(value, (bool, np.bool_)):
            return None
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if normalized < 0 or normalized != value:
            return None
        normalized_counts[name] = normalized

    clock = time.time() if now is None else float(now)
    age = clock - timestamp
    scan_duration = timestamp - scan_started_at
    raw = normalized_counts["raw_return_count"]
    resolved = normalized_counts["resolved_return_count"]
    self_count = normalized_counts["self_return_count"]
    retained = normalized_counts["retained_return_count"]
    invalid = normalized_counts["invalid_return_count"]
    if (
        sequence is None
        or not math.isfinite(age)
        or not math.isfinite(scan_duration)
        or age < -0.25
        or age > max(0.1, float(max_age_s))
        or scan_duration < 0.0
        or scan_duration > 1.25
        or not isinstance(identity_verified, (bool, np.bool_))
        or bool(identity_verified) is not True
        or identity_source != "rtx_stable_id"
        or frame_id != "world"
        or robot_prim_path != "/World/G1"
        or points.ndim != 2
        or points.shape[1] != 3
        or len(points) > MAX_LIDAR_POINTS
        or not np.issubdtype(points.dtype, np.floating)
        or not np.all(np.isfinite(points))
        or retained != len(points)
        or resolved + invalid != raw
        or raw != self_count + retained + invalid
    ):
        return None
    return IsaacLidarFrame(
        timestamp=timestamp,
        scan_started_at=scan_started_at,
        sequence=sequence,
        points=np.ascontiguousarray(points, dtype=np.float32),
        frame_id=frame_id,
        robot_prim_path=robot_prim_path,
        raw_return_count=raw,
        resolved_return_count=resolved,
        self_return_count=self_count,
        retained_return_count=retained,
        invalid_return_count=invalid,
    )


def velocity_command_payload(
    *,
    sequence: int,
    linear: tuple[float, float, float],
    angular: tuple[float, float, float],
    duration_s: float,
    now: float | None = None,
) -> dict[str, Any]:
    values = tuple(float(value) for value in (*linear, *angular, duration_s))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Isaac velocity command values must be finite")
    if sequence < 1:
        raise ValueError("Isaac velocity command sequence must be positive")
    if abs(linear[0]) > 0.45 or abs(linear[1]) > 0.25 or abs(linear[2]) > 1e-9:
        raise ValueError("Isaac planar velocity exceeds the adapter envelope")
    if abs(angular[0]) > 1e-9 or abs(angular[1]) > 1e-9 or abs(angular[2]) > 0.8:
        raise ValueError("Isaac angular velocity exceeds the adapter envelope")
    bounded_duration = min(20.0, max(0.0, float(duration_s)))
    issued_at = time.time() if now is None else float(now)
    ttl = bounded_duration if bounded_duration > 0.0 else 0.35
    return {
        "schema_version": SCHEMA_VERSION,
        "action": "velocity",
        "sequence": sequence,
        "issued_at": issued_at,
        "expires_at": issued_at + ttl,
        "linear": list(linear),
        "angular": list(angular),
    }


def validated_velocity_command(
    payload: Mapping[str, Any] | None,
    *,
    now: float | None = None,
) -> tuple[float, float, float, float] | None:
    if not isinstance(payload, Mapping):
        return None
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("action") != "velocity"
        or _finite_sequence(payload.get("sequence")) is None
    ):
        return None
    linear = _finite_vector(payload.get("linear"), 3)
    angular = _finite_vector(payload.get("angular"), 3)
    try:
        issued_at = float(payload["issued_at"])
        expires_at = float(payload["expires_at"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    clock = time.time() if now is None else float(now)
    if (
        linear is None
        or angular is None
        or expires_at < issued_at
        or expires_at - issued_at > 20.001
        or clock < issued_at - 0.25
        or clock > expires_at
        or abs(linear[0]) > 0.45
        or abs(linear[1]) > 0.25
        or abs(linear[2]) > 1e-9
        or abs(angular[0]) > 1e-9
        or abs(angular[1]) > 1e-9
        or abs(angular[2]) > 0.8
    ):
        return None
    return linear[0], linear[1], angular[2], 0.8


class IsaacCommandWriter:
    def __init__(self, paths: IsaacRuntimePaths) -> None:
        self.paths = paths
        self._sequence = 0
        self._lock = threading.Lock()

    def _next_sequence_locked(self) -> int:
        existing = read_json(self.paths.command)
        existing_sequence = 0
        if isinstance(existing, Mapping):
            value = existing.get("sequence")
            if isinstance(value, int) and not isinstance(value, bool):
                existing_sequence = max(0, value)
        self._sequence = max(self._sequence, existing_sequence) + 1
        return self._sequence

    def _active_hold_token_locked(self) -> tuple[bool, str | None]:
        exists = self.paths.safety_hold.exists()
        payload = read_json(self.paths.safety_hold)
        token = None
        if (
            isinstance(payload, Mapping)
            and payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("action") == "safety_hold"
            and isinstance(payload.get("token"), str)
            and payload.get("token")
        ):
            token = str(payload["token"])
        return exists, token

    def _write_velocity_locked(
        self,
        linear: tuple[float, float, float],
        angular: tuple[float, float, float],
        *,
        duration_s: float,
        now: float | None,
        safety_token: str | None,
    ) -> int:
        moving = any(abs(float(value)) > 1e-9 for value in (*linear, *angular))
        hold_exists, active_token = self._active_hold_token_locked()
        if moving and (
            hold_exists
            and (active_token is None or safety_token != active_token)
        ):
            raise PermissionError(
                "Isaac safety recovery hold blocks non-recovery motion"
            )
        sequence = self._next_sequence_locked()
        payload = velocity_command_payload(
            sequence=sequence,
            linear=linear,
            angular=angular,
            duration_s=duration_s,
            now=now,
        )
        if safety_token is not None:
            payload["command_source"] = "safety_recovery"
        atomic_write_json(self.paths.command, payload)
        return sequence

    def write_velocity(
        self,
        linear: tuple[float, float, float],
        angular: tuple[float, float, float],
        *,
        duration_s: float = 0.0,
        now: float | None = None,
        safety_token: str | None = None,
    ) -> int:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.paths.root / ".command.lock"
        with self._lock, lock_path.open("a+b") as lock_file:
            # The DimOS connection and the operator UI are independent host
            # processes.  Serialize their writes and continue after the last
            # on-disk sequence so an emergency stop can never be ignored as an
            # older command.
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                return self._write_velocity_locked(
                    linear=linear,
                    angular=angular,
                    duration_s=duration_s,
                    now=now,
                    safety_token=safety_token,
                )
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def stop(self, *, now: float | None = None) -> int:
        return self.write_velocity(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            duration_s=0.0,
            now=now,
        )

    def begin_safety_hold(self, *, now: float | None = None) -> str:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.paths.root / ".command.lock"
        with self._lock, lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                hold_exists, _active_token = self._active_hold_token_locked()
                if hold_exists:
                    raise PermissionError("Isaac safety recovery hold already exists")
                token = uuid.uuid4().hex
                issued_at = time.time() if now is None else float(now)
                atomic_write_json(
                    self.paths.safety_hold,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "action": "safety_hold",
                        "token": token,
                        "issued_at": issued_at,
                    },
                )
                self._write_velocity_locked(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    duration_s=0.0,
                    now=now,
                    safety_token=token,
                )
                return token
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def end_safety_hold(
        self,
        token: str,
        *,
        now: float | None = None,
    ) -> bool:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.paths.root / ".command.lock"
        with self._lock, lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                hold_exists, active_token = self._active_hold_token_locked()
                if not hold_exists or active_token != token:
                    return False
                self._write_velocity_locked(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    duration_s=0.0,
                    now=now,
                    safety_token=token,
                )
                self.paths.safety_hold.unlink()
                return True
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def clear_safety_hold(self, *, now: float | None = None) -> None:
        """Release a stale hold only as part of an explicit UI/reset lifecycle."""

        self.paths.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.paths.root / ".command.lock"
        with self._lock, lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._write_velocity_locked(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    duration_s=0.0,
                    now=now,
                    safety_token=None,
                )
                try:
                    self.paths.safety_hold.unlink()
                except FileNotFoundError:
                    pass
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
