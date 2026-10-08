#!/usr/bin/env python3
"""Serve the LuxiAgent local operator dashboard.

The server intentionally uses the DimOS virtual environment and Python's
standard-library HTTP server.  The only non-stdlib runtime dependency is
Pillow, which is already part of the pinned DimOS environment.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlparse

from harness.integrations.dimos.dimos_runtime import CostmapMonitor, NavigationVelocityBridge
from harness.robots.g1.mujoco.entity_manipulation import EntityControlChannel
from harness.evaluation.blind_evaluation import (
    BlindRuntimePaths,
    open_prepared_blind_run,
    require_blind_harness,
)
from harness.robots.g1.mujoco.operator_scenes import (
    OPERATOR_SCENE_PAYLOAD_ENV,
    OperatorScene,
    build_operator_scene,
    list_operator_scenes,
    operator_scene_descriptor,
    write_operator_scene_payload,
)
from harness.robots.g1.isaac.isaac_protocol import (
    IsaacCommandWriter,
    IsaacRuntimePaths,
    read_fresh_camera_frame,
    read_fresh_observer_frame,
    read_fresh_state,
    read_json,
    validated_velocity_command,
)
from harness.robots.g1.isaac.isaac_costmap import IsaacLidarCostmap
from harness.robots.g1.isaac.isaac_manual_control import IsaacManualControl, IsaacPersonControlWriter
from harness.robots.g1.isaac.isaac_recovery import IsaacSafetyCommandChannel
from harness.control.long_task_control import LongTaskControlChannel
from harness.app.mcp_jobs import MCPJobManager
from harness.robots.go2.go2_mujoco import (
    GO2_SCENE_ID,
)
from harness.robots.go2.go2_manual_control import Go2PersonControlWriter, Go2PersonManualControl
from harness.robots.go2.go2_protocol import (
    Go2CommandWriter,
    Go2RuntimePaths,
    read_fresh_go2_state,
)
from harness.robots.robot_profiles import get_backend_profile
from harness.runtime.go2_readonly import build_go2_readonly_adapter
from harness.runtime.isaac_readonly import build_isaac_g1_readonly_adapter
from harness.runtime.mujoco_readonly import build_mujoco_g1_readonly_adapter
from harness.runtime.robot_runtime import LuxiRobotRuntimeHost, RuntimeStatusProjection
from harness.control.safety_recovery import CriticalRecoveryController, RecoveryConfig
from harness.app.ui_event_projection import EventStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_ASSET_ROOT = Path(
    os.environ.get("DIMOS_ASSET_ROOT", str(Path.home() / "work/Asset/dimos"))
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_BYTES = CAMERA_WIDTH * CAMERA_HEIGHT * 3
ODOM_BYTES = 8 * 8
COMMAND_BYTES = 6 * 4


# 查找可用的 Codex CLI 可执行文件，供兼容入口启动会话。

# G1 navigation uses a 0.60 m-wide footprint.  User-facing safety thresholds
# are robot-surface clearances; current lidar still reports centre-to-endpoint
# distances, so the runtime derives the internal thresholds explicitly.
from harness.robots.g1.safety_geometry import (
    FOOTPRINT_RADIUS_M as G1_FOOTPRINT_RADIUS_M, WARNING_SURFACE_CLEARANCE_M,
    CRITICAL_SURFACE_CLEARANCE_M, IMMEDIATE_CRITICAL_SURFACE_CLEARANCE_M,
    RELEASE_SURFACE_CLEARANCE_M as CRITICAL_RELEASE_SURFACE_CLEARANCE_M,
)
WARNING_CENTER_DISTANCE_M = round(
    G1_FOOTPRINT_RADIUS_M + WARNING_SURFACE_CLEARANCE_M,
    6,
)
CRITICAL_CENTER_DISTANCE_M = round(
    G1_FOOTPRINT_RADIUS_M + CRITICAL_SURFACE_CLEARANCE_M,
    6,
)
IMMEDIATE_CRITICAL_CENTER_DISTANCE_M = round(
    G1_FOOTPRINT_RADIUS_M + IMMEDIATE_CRITICAL_SURFACE_CLEARANCE_M,
    6,
)
CRITICAL_RELEASE_CENTER_DISTANCE_M = round(
    G1_FOOTPRINT_RADIUS_M + CRITICAL_RELEASE_SURFACE_CLEARANCE_M,
    6,
)
CRITICAL_CONFIRMATION_FRAMES = 2

UI_BACKENDS = frozenset({"mujoco", "isaac-g1", "mujoco-go2"})
LCM_RECEIVE_BUFFER_BYTES = 4_194_304
DEFAULT_ISAAC_UI_LCM_URL = "udpm://239.255.76.67:17668?ttl=0"


# 为 LCM 通信地址配置接收缓冲区，支持较大的 RGB-D 分片消息。
def lcm_url_with_receive_buffer(
    url: str,
    *,
    receive_buffer_bytes: int = LCM_RECEIVE_BUFFER_BYTES,
) -> str:
    """Request a large LCM UDP socket buffer without overriding operators."""

    value = str(url).strip()
    if not value or receive_buffer_bytes < 1:
        raise ValueError("LCM URL and receive buffer size must be valid")
    query = value.partition("?")[2]
    if any(
        part.partition("=")[0] == "recv_buf_size"
        for part in query.split("&")
        if part
    ):
        return value
    separator = "&" if "?" in value else "?"
    return f"{value}{separator}recv_buf_size={int(receive_buffer_bytes)}"
ISAAC_SCENE_CATALOG = (
    {
        "scene_id": "grid",
        "label": "Isaac 校验网格",
        "description": "本地空旷网格；Unitree G1 Isaac 的基础校验场景。",
        "complexity": "baseline",
        "source": "isaac_local",
        "seedable": False,
        "supports_person": False,
    },
    {
        "scene_id": "skill_demo",
        "label": "Isaac 寻物与跟随",
        "description": "本地离线场景：背向矿泉水瓶与缓慢移动的仿真人物。",
        "complexity": "standard",
        "source": "isaac_local",
        "seedable": False,
        "supports_person": True,
    },
    {
        "scene_id": "task_apartment",
        "label": "Isaac 寻物跟随实景公寓",
        "description": (
            "本地完整客厅—门厅—厨房场景；含碰撞家具、初始视野外茶几水瓶和"
            "带正常五官、头发与 PBR 服装材质的人物角色。"
        ),
        "complexity": "complex",
        "source": "isaac_local",
        "seedable": False,
        "supports_person": True,
    },
    {
        "scene_id": "brownstone",
        "label": "NVIDIA Brownstone 住宅",
        "description": "本地只读 OpenUSD 多层建筑资产；首次启动会检查完整场景包。",
        "complexity": "very_complex",
        "source": "nvidia_openusd_pack",
        "seedable": False,
        "supports_person": False,
    },
    {
        "scene_id": "office",
        "label": "Isaac Office",
        "description": "Isaac Sim Office 资产；首次加载可能需要可用的远程资产缓存。",
        "complexity": "complex",
        "source": "isaac_asset",
        "seedable": False,
        "supports_person": False,
    },
    {
        "scene_id": "warehouse",
        "label": "Isaac Warehouse",
        "description": "Isaac Sim Warehouse 资产；只有新鲜 Stable-ID RTX lidar 可用时才开放已探索区域导航。",
        "complexity": "complex",
        "source": "isaac_asset",
        "seedable": False,
        "supports_person": False,
    },
)


# 规范化并校验面板使用的机器人仿真后端名称。
def normalize_ui_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend not in UI_BACKENDS:
        choices = ", ".join(sorted(UI_BACKENDS))
        raise ValueError(f"Unsupported UI backend {value!r}; expected one of: {choices}")
    return backend


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.08) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def quaternion_yaw(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def env_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_enabled_from_mapping(
    environment: dict[str, str],
    name: str,
    default: bool = False,
) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def third_person_enabled_for_environment(environment: dict[str, str]) -> bool:
    return env_enabled_from_mapping(
        environment, "LUXI_THIRD_PERSON_ENABLED"
    ) and not env_enabled_from_mapping(environment, "LUXI_BLIND_MODE")


def bounded_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


# 机器人位姿与运动观测的数据容器，供监控和状态展示使用。
@dataclass(frozen=True)
class PoseSnapshot:
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float
    timestamp: float
    yaw: float


# 平面矩形障碍物，提供面积和点到障碍物的距离计算。
@dataclass(frozen=True)
class RectangleObstacle:
    label: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def distance_to(self, x: float, y: float) -> float:
        dx = max(self.x_min - x, 0.0, x - self.x_max)
        dy = max(self.y_min - y, 0.0, y - self.y_max)
        return math.hypot(dx, dy)


# 读取 MuJoCo 的共享内存观测：位姿、控制指令和相机图像；不拥有这些缓冲区。
class SharedMemoryProbe:
    """Read the newest DimOS MuJoCo shared-memory buffers without owning them."""

    _SAFE_NAME = re.compile(r"psm_[A-Za-z0-9_-]{1,128}")

    def __init__(
        self,
        shm_root: Path = Path("/dev/shm"),
        *,
        manifest_path: Path | None = None,
    ) -> None:
        self.shm_root = shm_root
        self.manifest_path = manifest_path
        self._cache: dict[int, tuple[float, list[Path]]] = {}
        self._manifest_cache: tuple[float, dict[str, str]] | None = None
        self._video_progress: tuple[str, int, float] | None = None
        self._lock = threading.Lock()

    def _manifest_channels(self) -> dict[str, str]:
        now = time.monotonic()
        with self._lock:
            cached = self._manifest_cache
            if cached and now - cached[0] < 0.25:
                return dict(cached[1])

        channels: dict[str, str] = {}
        if self.manifest_path is not None:
            try:
                payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                raw_channels = payload.get("channels", {})
                if payload.get("schema_version") == 1 and isinstance(raw_channels, dict):
                    channels = {
                        channel: name
                        for channel, name in raw_channels.items()
                        if isinstance(channel, str)
                        and isinstance(name, str)
                        and self._SAFE_NAME.fullmatch(name) is not None
                    }
            except (FileNotFoundError, OSError, json.JSONDecodeError, AttributeError):
                channels = {}

        with self._lock:
            self._manifest_cache = (now, channels)
        return dict(channels)

    def _paths_for_channel(self, channel: str, size: int) -> list[Path]:
        if self.manifest_path is None:
            return self._paths_with_size(size)
        name = self._manifest_channels().get(channel)
        if name is None:
            return []
        path = self.shm_root / name
        try:
            return [path] if path.stat().st_size == size else []
        except OSError:
            return []

    def _paths_with_size(self, size: int) -> list[Path]:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(size)
            if cached and now - cached[0] < 0.75:
                return list(cached[1])

        candidates: list[tuple[int, Path]] = []
        try:
            for path in self.shm_root.glob("psm_*"):
                try:
                    stat = path.stat()
                    if stat.st_size == size:
                        candidates.append((stat.st_mtime_ns, path))
                except OSError:
                    continue
        except OSError:
            candidates = []
        paths = [path for _, path in sorted(candidates, reverse=True)]

        with self._lock:
            self._cache[size] = (now, paths)
        return paths

    def reset(self) -> None:
        """Forget paths owned by the previous MuJoCo process."""

        with self._lock:
            self._cache.clear()
            self._manifest_cache = None
            self._video_progress = None

    @staticmethod
    def _read(path: Path, expected_size: int) -> bytes | None:
        try:
            payload = path.read_bytes()
        except OSError:
            return None
        return payload if len(payload) == expected_size else None

    def pose(self) -> PoseSnapshot | None:
        for path in self._paths_for_channel("odom", ODOM_BYTES):
            payload = self._read(path, ODOM_BYTES)
            if payload is None:
                continue
            values = struct.unpack("=8d", payload)
            x, y, z, qw, qx, qy, qz, timestamp = values
            quaternion_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if not all(math.isfinite(value) for value in values):
                continue
            if timestamp < 1_000_000_000 or not 0.25 < quaternion_norm < 1.75:
                continue
            return PoseSnapshot(
                x=x,
                y=y,
                z=z,
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz,
                timestamp=timestamp,
                yaw=quaternion_yaw(qw, qx, qy, qz),
            )
        return None

    def command(self) -> list[float] | None:
        for path in self._paths_for_channel("cmd", COMMAND_BYTES):
            payload = self._read(path, COMMAND_BYTES)
            if payload is None:
                continue
            values = list(struct.unpack("=6f", payload))
            if all(math.isfinite(value) and abs(value) < 100 for value in values):
                return values
        return None

    def camera_path(self) -> Path | None:
        # The worker increments seq[0] only after publishing RGB. A newly
        # allocated video buffer is not a frame (and a real black frame is valid).
        channels = self._manifest_channels() if self.manifest_path is not None else {}
        has_sequence = "seq" in channels
        if has_sequence:
            seq_paths = self._paths_for_channel("seq", 8 * 8)
            raw = self._read(seq_paths[0], 8 * 8) if seq_paths else None
            if raw is None:
                return None
            sequence = struct.unpack_from("=q", raw)[0]
            if sequence <= 0:
                return None
            identity = channels["seq"]
            now = time.monotonic()
            with self._lock:
                previous = self._video_progress
                if previous is None or previous[:2] != (identity, sequence):
                    self._video_progress = (identity, sequence, now)
                elif now - previous[2] > 2.5:
                    return None
        for path in self._paths_for_channel("video", CAMERA_BYTES):
            if has_sequence and path.exists():
                return path
            # Older manifests have no sequence channel. Do not promote their
            # zero-filled allocation to a ready camera; never discover a foreign
            # sequence buffer by size (odom has the same byte size).
            payload = self._read(path, CAMERA_BYTES)
            if payload is not None and any(payload):
                return path
        return None

    def camera_jpeg(self, quality: int = 82) -> bytes | None:
        path = self.camera_path()
        if path is None:
            return None
        payload = self._read(path, CAMERA_BYTES)
        if payload is None:
            return None

        try:
            from PIL import Image

            image = Image.frombytes("RGB", (CAMERA_WIDTH, CAMERA_HEIGHT), payload)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=False)
            return output.getvalue()
        except Exception:
            return None


# 将 Isaac 运行目录中的新鲜状态和相机帧转换为面板统一的观测接口。
class IsaacFileProbe:
    """Expose only fresh observations from the isolated Isaac file protocol."""

    def __init__(
        self,
        paths: IsaacRuntimePaths,
        *,
        max_age_seconds: float = 1.5,
    ) -> None:
        self.paths = paths
        self.max_age_seconds = max(0.1, float(max_age_seconds))
        self._lock = threading.Lock()
        self._cached_frame_at = 0.0
        self._cached_frame: Any | None = None

    def reset(self) -> None:
        with self._lock:
            self._cached_frame_at = 0.0
            self._cached_frame = None

    def _state(self) -> dict[str, Any] | None:
        return read_fresh_state(
            self.paths,
            max_age_s=self.max_age_seconds,
        )

    def _frame(self) -> Any | None:
        now = time.monotonic()
        with self._lock:
            if now - self._cached_frame_at <= 0.08:
                return self._cached_frame
        frame = read_fresh_camera_frame(
            self.paths,
            max_age_s=self.max_age_seconds,
        )
        with self._lock:
            self._cached_frame_at = now
            self._cached_frame = frame
        return frame

    def ready(self) -> bool:
        return self._state() is not None and self._frame() is not None

    def pose(self) -> PoseSnapshot | None:
        state = self._state()
        if state is None:
            return None
        pose = state["pose"]
        position = pose["position"]
        qw, qx, qy, qz = pose["quaternion_wxyz"]
        return PoseSnapshot(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            qw=float(qw),
            qx=float(qx),
            qy=float(qy),
            qz=float(qz),
            timestamp=float(state["written_at"]),
            yaw=quaternion_yaw(float(qw), float(qx), float(qy), float(qz)),
        )

    def command(self) -> list[float] | None:
        if self._state() is None:
            return None
        command = validated_velocity_command(read_json(self.paths.command))
        if command is None:
            # Missing and expired commands both become zero inside Isaac.
            return [0.0] * 6
        x, y, yaw, _height = command
        return [float(x), float(y), 0.0, 0.0, 0.0, float(yaw)]

    def camera_path(self) -> Path | None:
        return self.paths.camera if self._frame() is not None else None

    def camera_jpeg(self, quality: int = 82) -> bytes | None:
        frame = self._frame()
        if frame is None:
            return None
        try:
            from PIL import Image

            image = Image.fromarray(frame.rgb, mode="RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=False)
            return output.getvalue()
        except Exception:
            return None


# 将 Go2 的新鲜里程计和 HIKROBOT 图像转换为面板统一的观测接口。
class Go2FileProbe:
    """Expose only fresh HIKROBOT RGB/odometry files from the Go2 runtime."""

    def __init__(
        self,
        paths: Go2RuntimePaths,
        *,
        max_age_seconds: float = 1.5,
    ) -> None:
        self.paths = paths
        self.max_age_seconds = max(0.1, float(max_age_seconds))

    def reset(self) -> None:
        self.paths.clear_observations()

    def _state(self) -> dict[str, Any] | None:
        return read_fresh_go2_state(
            self.paths,
            max_age_s=self.max_age_seconds,
        )

    def ready(self) -> bool:
        return self._state() is not None and self.camera_path() is not None

    def pose(self) -> PoseSnapshot | None:
        state = self._state()
        if state is None:
            return None
        position = state["pose"]["position"]
        qw, qx, qy, qz = state["pose"]["quaternion_wxyz"]
        return PoseSnapshot(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            qw=float(qw),
            qx=float(qx),
            qy=float(qy),
            qz=float(qz),
            timestamp=float(state["written_at"]),
            yaw=quaternion_yaw(float(qw), float(qx), float(qy), float(qz)),
        )

    def command(self) -> list[float] | None:
        state = self._state()
        return list(state["command"]) if state is not None else None

    def camera_path(self) -> Path | None:
        if self._state() is None:
            return None
        try:
            stat = self.paths.camera.stat()
        except OSError:
            return None
        age = time.time() - stat.st_mtime
        return (
            self.paths.camera
            if stat.st_size >= 128 and -0.25 <= age <= self.max_age_seconds
            else None
        )

    def camera_jpeg(self, quality: int = 82) -> bytes | None:
        path = self.camera_path()
        if path is None:
            return None
        try:
            payload = path.read_bytes()
        except OSError:
            return None
        if len(payload) < 128 or not payload.startswith(b"\xff\xd8"):
            return None
        return payload


# 读取 Go2 基于 MID-360 射线生成的在线栅格地图，并提供地图状态。
class Go2LidarCostmap:
    """Read the fresh occupancy grid built only from MID-360 raycasts."""

    def __init__(
        self,
        paths: Go2RuntimePaths,
        *,
        max_age_seconds: float = 1.5,
    ) -> None:
        self.paths = paths
        self.max_age_seconds = max(0.1, float(max_age_seconds))

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _payload(self) -> dict[str, Any] | None:
        payload = read_json(self.paths.costmap)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("source") not in {
                "go2_mid360_live",
            }
            or payload.get("available") is not True
        ):
            return None
        try:
            timestamp = float(payload["timestamp"])
            width = int(payload["width"])
            height = int(payload["height"])
            resolution = float(payload["resolution"])
            encoded = base64.b64decode(payload["data"], validate=True)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        age = time.time() - timestamp
        if (
            not math.isfinite(timestamp)
            or not math.isfinite(resolution)
            or not 1 <= width <= 512
            or not 1 <= height <= 512
            or not 0.01 <= resolution <= 1.0
            or len(encoded) != width * height
            or age < -0.25
            or age > self.max_age_seconds
        ):
            return None
        result = dict(payload)
        result["age_seconds"] = round(max(0.0, age), 3)
        return result

    def payload(self) -> dict[str, Any]:
        return self._payload() or {"available": False, "running": True}

    def status(self) -> dict[str, Any]:
        payload = self._payload()
        if payload is None:
            return {
                "available": False,
                "running": True,
                "topic": "mid360-costmap",
                "snapshot_path": str(self.paths.costmap),
            }
        return {
            key: payload.get(key)
            for key in (
                "available",
                "running",
                "revision",
                "source",
                "frame_id",
                "timestamp",
                "width",
                "height",
                "resolution",
                "origin",
                "cells",
                "age_seconds",
                "sensor",
                "inspection",
            )
        }

    def reset(self) -> list[str]:
        try:
            self.paths.costmap.unlink()
            return [str(self.paths.costmap)]
        except FileNotFoundError:
            return []

    def save_now(self, *, announce: bool = True) -> tuple[bool, str]:
        del announce
        return False, "Go2 MID-360 地图由运行时持续原子保存"


# 读取带时效检查的 JPEG 文件，避免面板把旧画面当成实时观测。
class FreshJpegFrame:
    """Read an atomically published JPEG only while its producer is live."""

    def __init__(
        self,
        path: Path,
        max_age_seconds: float = 2.5,
        *,
        enabled: bool = True,
    ) -> None:
        self.path = path
        self.max_age_seconds = max_age_seconds
        self.enabled = enabled

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"available": False, "age_seconds": None}
        try:
            stat = self.path.stat()
        except OSError:
            return {"available": False, "age_seconds": None}
        age = max(0.0, time.time() - stat.st_mtime)
        return {
            "available": stat.st_size >= 128 and age <= self.max_age_seconds,
            "age_seconds": round(age, 3),
        }

    def jpeg(self) -> bytes | None:
        if not self.status()["available"]:
            return None
        try:
            payload = self.path.read_bytes()
        except OSError:
            return None
        if len(payload) < 128 or not payload.startswith(b"\xff\xd8"):
            return None
        return payload

    def clear(self) -> bool:
        """Remove the last frame so a reset cannot display a stale simulation."""

        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False


# 适配 Isaac 第三人称观察相机，按其帧协议检查新鲜度并提供 JPEG。
class FreshIsaacObserverFrame(FreshJpegFrame):
    """Expose fresh Isaac observer archives as browser JPEGs only."""

    def __init__(
        self,
        paths: IsaacRuntimePaths,
        max_age_seconds: float = 2.5,
        *,
        enabled: bool = True,
    ) -> None:
        self.paths = paths
        super().__init__(
            paths.observer,
            max_age_seconds=max_age_seconds,
            enabled=enabled,
        )

    def _frame(self) -> Any | None:
        if not self.enabled:
            return None
        return read_fresh_observer_frame(
            self.paths,
            max_age_s=self.max_age_seconds,
        )

    def status(self) -> dict[str, Any]:
        frame = self._frame()
        if frame is None:
            return {"available": False, "age_seconds": None}
        return {
            "available": True,
            "age_seconds": round(max(0.0, time.time() - frame.timestamp), 3),
        }

    def jpeg(self) -> bytes | None:
        frame = self._frame()
        if frame is None:
            return None
        try:
            from PIL import Image

            image = Image.fromarray(frame.rgb, mode="RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=82, optimize=False)
            return output.getvalue()
        except Exception:
            return None


# 读取最新激光近距离观测，向风险监控提供障碍物距离及数据可用性。
class FreshLidarProximity:
    """Read one atomically published current-lidar proximity summary."""

    _DIRECTIONS = {
        "front",
        "front_left",
        "left",
        "rear_left",
        "rear",
        "rear_right",
        "right",
        "front_right",
    }

    def __init__(
        self,
        path: Path,
        max_age_seconds: float = 1.25,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.max_age_seconds = max(0.1, float(max_age_seconds))
        self.clock = clock

    @staticmethod
    def _unavailable(reason: str, age: float | None = None) -> dict[str, Any]:
        return {
            "available": False,
            "source": "current_lidar",
            "age_seconds": None if age is None else round(max(0.0, age), 3),
            "reason": reason,
        }

    def payload(self) -> dict[str, Any]:
        try:
            stat = self.path.stat()
            if stat.st_size <= 0 or stat.st_size > 64 * 1024:
                return self._unavailable("invalid_size")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return self._unavailable("invalid_payload")
            if raw.get("schema_version") != 1 or raw.get("source") != "current_lidar":
                return self._unavailable("invalid_schema")
            sequence = raw.get("sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                return self._unavailable("invalid_sequence")
            frame_timestamp = float(raw["frame_timestamp"])
            pose_timestamp = float(raw["pose_timestamp"])
            written_at = float(raw["written_at"])
            if not all(
                math.isfinite(value)
                for value in (frame_timestamp, pose_timestamp, written_at)
            ):
                return self._unavailable("invalid_timestamp")
            age = self.clock() - written_at
            if age < -0.25 or age > self.max_age_seconds:
                return self._unavailable("stale", age)

            self_filter = raw.get("self_filter")
            if not isinstance(self_filter, dict):
                return self._unavailable("self_filter_missing", age)
            filter_frames = int(self_filter.get("frames", -1))
            filter_applied = int(self_filter.get("applied", -1))
            filter_missing = int(self_filter.get("missing", -1))
            filter_invalid = int(self_filter.get("invalid", -1))
            if (
                self_filter.get("identity_verified") is not True
                or filter_frames < 1
                or filter_applied != filter_frames
                or filter_missing != 0
                or filter_invalid != 0
            ):
                return self._unavailable("self_filter_unverified", age)

            nearest_raw = raw.get("nearest_obstacle_distance")
            nearest = None if nearest_raw is None else float(nearest_raw)
            if nearest is not None and (not math.isfinite(nearest) or nearest < 0.0):
                return self._unavailable("invalid_distance", age)
            bearing_raw = raw.get("nearest_obstacle_bearing_deg")
            bearing = None if bearing_raw is None else float(bearing_raw)
            if bearing is not None and not math.isfinite(bearing):
                return self._unavailable("invalid_bearing", age)
            direction = raw.get("nearest_obstacle_direction")
            if direction is not None and direction not in self._DIRECTIONS:
                return self._unavailable("invalid_direction", age)
            if (nearest is None) != (direction is None):
                return self._unavailable("incomplete_nearest_obstacle", age)

            sectors_raw = raw.get("sectors_m", {})
            if not isinstance(sectors_raw, dict):
                return self._unavailable("invalid_sectors", age)
            sectors: dict[str, float] = {}
            for name, value in sectors_raw.items():
                if name not in self._DIRECTIONS:
                    return self._unavailable("invalid_sector_name", age)
                distance = float(value)
                if not math.isfinite(distance) or distance < 0.0:
                    return self._unavailable("invalid_sector_distance", age)
                sectors[name] = distance
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
            return self._unavailable("unreadable")

        return {
            "available": True,
            "source": "current_lidar",
            "frame_id": "world",
            "sequence": sequence,
            "frame_timestamp": frame_timestamp,
            "pose_timestamp": pose_timestamp,
            "written_at": written_at,
            "age_seconds": round(max(0.0, age), 3),
            "nearest_obstacle_distance": nearest,
            "nearest_obstacle_bearing_deg": bearing,
            "nearest_obstacle_direction": direction,
            "sectors_m": sectors,
            "candidate_points": raw.get("candidate_points"),
            "critical_diagnostic_candidate": raw.get("critical_diagnostic_candidate"),
            "critical_diagnostic_error": raw.get("critical_diagnostic_error"),
            "self_filter": {
                "identity_verified": True,
                "frames": filter_frames,
                "applied": filter_applied,
                "masked_pixels": self_filter.get("masked_pixels"),
                "mode": self_filter.get("mode"),
            },
        }

    def status(self) -> dict[str, Any]:
        payload = self.payload()
        return {
            key: payload.get(key)
            for key in ("available", "age_seconds", "sequence", "reason")
        }

    def clear(self) -> bool:
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False


def _obj_components(path: Path) -> list[list[tuple[float, float]]]:
    """Return connected OBJ components in MuJoCo world X/Y coordinates."""

    vertices: list[tuple[float, float]] = []
    faces: list[list[int]] = []
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return []

    for line in lines:
        if line.startswith("v "):
            fields = line.split()
            if len(fields) >= 4:
                # scene_office1 bodies rotate Blender meshes +90 degrees about X.
                vertices.append((float(fields[1]), -float(fields[3])))
        elif line.startswith("f "):
            indices: list[int] = []
            for field in line.split()[1:]:
                try:
                    indices.append(int(field.split("/", 1)[0]) - 1)
                except ValueError:
                    continue
            if len(indices) >= 2:
                faces.append(indices)

    parent = list(range(len(vertices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for face in faces:
        anchor = face[0]
        if not 0 <= anchor < len(vertices):
            continue
        for index in face[1:]:
            if 0 <= index < len(vertices):
                union(anchor, index)

    components: dict[int, list[tuple[float, float]]] = {}
    for index, vertex in enumerate(vertices):
        components.setdefault(find(index), []).append(vertex)
    return list(components.values())


def _overlap_fraction(left: RectangleObstacle, right: RectangleObstacle) -> float:
    x_overlap = max(0.0, min(left.x_max, right.x_max) - max(left.x_min, right.x_min))
    y_overlap = max(0.0, min(left.y_max, right.y_max) - max(left.y_min, right.y_min))
    intersection = x_overlap * y_overlap
    denominator = min(left.area, right.area)
    return intersection / denominator if denominator > 0 else 0.0


# 从场景资产提取桌子障碍物的平面范围，供需要场景几何的观测路径使用。
def load_table_obstacles(asset_root: Path) -> list[RectangleObstacle]:
    """Extract large tabletop footprints from the pinned office OBJ assets."""

    base = asset_root / "upstream/data/mujoco_sim/scene_office1/office_split"
    meshes = {
        "wooden desk": "Cube_007_woodenDesk.obj",
        "white table": "Cube_013_BigWhiteTable.obj",
        "small table": "Cube_017_SmallWhiteTable_low_001.obj",
        "meeting table": "Plane_029_meetingTable.obj",
    }
    candidates: list[RectangleObstacle] = []
    for label, filename in meshes.items():
        for component in _obj_components(base / filename):
            if len(component) < 4:
                continue
            xs = [point[0] for point in component]
            ys = [point[1] for point in component]
            rectangle = RectangleObstacle(label, min(xs), max(xs), min(ys), max(ys))
            width = rectangle.x_max - rectangle.x_min
            height = rectangle.y_max - rectangle.y_min
            if width >= 0.45 and height >= 0.45:
                candidates.append(rectangle)

    # Visual and collision components can describe the same tabletop.  Keep the
    # largest footprint and remove near-contained duplicates for a clean map.
    result: list[RectangleObstacle] = []
    for candidate in sorted(candidates, key=lambda item: item.area, reverse=True):
        if any(_overlap_fraction(candidate, existing) > 0.82 for existing in result):
            continue
        result.append(candidate)
    return sorted(result, key=lambda item: (item.y_min, item.x_min))


# 后台汇总位姿、视觉和激光观测，计算风险、记录运动审计并生成观测快照。
# 分别提供操作面板和 Agent 所需的视图，Agent 视图遵守环境信息边界。
class ObservationMonitor:
    """Continuously build a small, explainable world-state snapshot."""

    def __init__(
        self,
        probe: SharedMemoryProbe,
        events: EventStore,
        obstacles: list[RectangleObstacle] | None = None,
        mapping_status: Callable[[], dict[str, Any]] | None = None,
        navigation_status: Callable[[], dict[str, Any]] | None = None,
        costmap_payload: Callable[[], dict[str, Any]] | None = None,
        lidar_proximity_payload: Callable[[], dict[str, Any]] | None = None,
        recovery_status: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.probe = probe
        self.events = events
        self.mapping_status = mapping_status
        self.navigation_status = navigation_status
        self.costmap_payload = costmap_payload
        self.lidar_proximity_payload = lidar_proximity_payload
        self.recovery_status = recovery_status
        self._trail: deque[dict[str, float]] = deque(maxlen=500)
        self._motion_samples: deque[PoseSnapshot] = deque(maxlen=40)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_pose: PoseSnapshot | None = None
        self._last_risk = "unknown"
        self._pending_risk: str | None = None
        self._pending_risk_since = 0.0
        self._stable_risk = "unknown"
        self._last_risk_evidence_key: tuple[str, int | float] | None = None
        self._critical_confirmation_frames = 0
        self._critical_release_frames = 0
        self._motion_audit_generation = 0
        self._active_motion_audit: int | None = None
        self._motion_audit_payload: dict[str, Any] | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="observation-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def reset(self) -> None:
        """Clear trajectory, motion samples, and risk transition state."""

        with self._lock:
            self._trail.clear()
            self._motion_samples.clear()
            self._last_pose = None
            self._last_risk = "unknown"
            self._pending_risk = None
            self._pending_risk_since = 0.0
            self._stable_risk = "unknown"
            self._last_risk_evidence_key = None
            self._critical_confirmation_frames = 0
            self._critical_release_frames = 0
            self._motion_audit_generation += 1
            self._active_motion_audit = None
            self._motion_audit_payload = None

    # 开启一次运动审计并返回标识，用于关联后续运动期间的观测证据。
    def begin_motion_audit(self) -> int:
        """Start a distinct-frame physical-clearance audit for one closed loop."""

        with self._lock:
            self._motion_audit_generation += 1
            token = self._motion_audit_generation
            self._active_motion_audit = token
            self._motion_audit_payload = {
                "available": True,
                "started_at": utc_now(),
                "started_wall_time": time.time(),
                "started_monotonic": time.monotonic(),
                "samples": 0,
                "last_lidar_sequence": None,
                "minimum_obstacle_distance_m": None,
                "minimum_robot_surface_clearance_m": None,
                "minimum_lidar_sequence": None,
                "minimum_frame_timestamp": None,
                "minimum_direction": None,
                "minimum_risk_raw": None,
                "critical_raw_samples": 0,
                "consecutive_critical_raw_samples": 0,
                "max_consecutive_critical_raw_samples": 0,
                "warning_raw_samples": 0,
                "immediate_critical_samples": 0,
                "translating_samples": 0,
            }
            return token

    @staticmethod
    def _public_motion_audit(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key not in {"started_wall_time", "started_monotonic", "last_lidar_sequence"}
        }

    def motion_audit(self, token: int) -> dict[str, Any]:
        with self._lock:
            if token != self._active_motion_audit or self._motion_audit_payload is None:
                return {"available": False, "reason": "motion_audit_token_inactive"}
            return self._public_motion_audit(self._motion_audit_payload)

    # 结束指定运动审计并返回收集的证据。
    def finish_motion_audit(self, token: int) -> dict[str, Any]:
        with self._lock:
            if token != self._active_motion_audit or self._motion_audit_payload is None:
                return {"available": False, "reason": "motion_audit_token_inactive"}
            payload = self._motion_audit_payload
            payload["completed_at"] = utc_now()
            payload["duration_s"] = round(
                max(0.0, time.monotonic() - float(payload["started_monotonic"])),
                3,
            )
            result = self._public_motion_audit(payload)
            self._active_motion_audit = None
            self._motion_audit_payload = None
            return result

    def _record_motion_audit(self, metrics: dict[str, Any]) -> None:
        sequence = metrics.get("lidar_sequence")
        frame_timestamp = metrics.get("lidar_frame_timestamp")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or isinstance(frame_timestamp, bool)
            or not isinstance(frame_timestamp, (int, float))
        ):
            return
        with self._lock:
            payload = self._motion_audit_payload
            if self._active_motion_audit is None or payload is None:
                return
            if float(frame_timestamp) < float(payload["started_wall_time"]):
                return
            if sequence == payload["last_lidar_sequence"]:
                return
            payload["last_lidar_sequence"] = sequence
            payload["samples"] += 1
            raw_risk = str(metrics.get("risk_raw") or "unknown")
            if raw_risk == "critical":
                payload["critical_raw_samples"] += 1
                payload["consecutive_critical_raw_samples"] += 1
                payload["max_consecutive_critical_raw_samples"] = max(
                    payload["max_consecutive_critical_raw_samples"],
                    payload["consecutive_critical_raw_samples"],
                )
            elif raw_risk == "warning":
                payload["warning_raw_samples"] += 1
                payload["consecutive_critical_raw_samples"] = 0
            else:
                payload["consecutive_critical_raw_samples"] = 0
            if metrics.get("critical_immediate") is True:
                payload["immediate_critical_samples"] += 1
            if metrics.get("critical_immediate_allowed") is True:
                payload["translating_samples"] += 1
            distance = metrics.get("nearest_obstacle_distance")
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(float(distance))
            ):
                return
            minimum = payload["minimum_obstacle_distance_m"]
            if minimum is not None and float(distance) >= float(minimum):
                return
            cell = metrics.get("nearest_obstacle_cell") or {}
            payload.update(
                {
                    "minimum_obstacle_distance_m": float(distance),
                    "minimum_robot_surface_clearance_m": max(
                        0.0,
                        float(distance) - G1_FOOTPRINT_RADIUS_M,
                    ),
                    "minimum_lidar_sequence": sequence,
                    "minimum_frame_timestamp": float(frame_timestamp),
                    "minimum_direction": cell.get("direction"),
                    "minimum_risk_raw": raw_risk,
                }
            )

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            pose = self.probe.pose()
            if pose is None:
                continue
            with self._lock:
                previous = self._last_pose
                self._last_pose = pose
                self._motion_samples.append(pose)
                if (
                    previous is None
                    or math.hypot(pose.x - previous.x, pose.y - previous.y) >= 0.015
                    or not self._trail
                ):
                    self._trail.append({"x": pose.x, "y": pose.y, "t": time.time()})
            self._publish_risk_transition(pose)

    @staticmethod
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

    @staticmethod
    def _forward_view_metrics(
        raw: bytes,
        *,
        width: int,
        height: int,
        resolution: float,
        local_x: float,
        local_y: float,
        local_yaw: float,
    ) -> dict[str, Any]:
        """Estimate first-person openness using only the live online costmap."""

        ray_offsets = tuple(
            math.radians(value)
            for value in (-60, -45, -30, -15, 0, 15, 30, 45, 60)
        )
        step = max(resolution, 0.10)
        minimum = G1_FOOTPRINT_RADIUS_M + 0.05
        maximum = 1.50
        samples_per_ray = max(1, math.floor((maximum - minimum) / step) + 1)
        expected_samples = len(ray_offsets) * samples_per_ray
        known_samples = 0
        reaches: list[float] = []
        for offset in ray_offsets:
            angle = local_yaw + offset
            reach = 0.0
            for sample_index in range(samples_per_ray):
                distance = minimum + sample_index * step
                sample_x = local_x + math.cos(angle) * distance
                sample_y = local_y + math.sin(angle) * distance
                column = math.floor(sample_x / resolution)
                row = math.floor(sample_y / resolution)
                if not (0 <= column < width and 0 <= row < height):
                    break
                value = raw[row * width + column]
                if value == 255:
                    break
                known_samples += 1
                if 50 <= value <= 100:
                    break
                reach = distance
            reaches.append(reach)
        ordered = sorted(reaches)
        open_distance = ordered[len(ordered) // 2] if ordered else 0.0
        known_ratio = known_samples / expected_samples if expected_samples else 0.0
        quality = (
            "good"
            if open_distance >= 0.75 and known_ratio >= 0.45
            else "limited"
        )
        return {
            "forward_view_quality": quality,
            "forward_known_ratio": round(known_ratio, 3),
            "forward_open_distance_m": round(open_distance, 3),
        }

    def _stabilized_risk(
        self,
        *,
        raw_risk: str,
        nearest_obstacle: float | None,
        evidence_key: tuple[str, int | float],
        immediate_allowed: bool = True,
    ) -> tuple[str, int, bool]:
        """Debounce physical proximity across distinct current-lidar frames."""

        immediate = bool(
            immediate_allowed
            and raw_risk == "critical"
            and nearest_obstacle is not None
            and nearest_obstacle <= IMMEDIATE_CRITICAL_CENTER_DISTANCE_M
        )
        with self._lock:
            is_new_evidence = evidence_key != self._last_risk_evidence_key
            if is_new_evidence:
                self._last_risk_evidence_key = evidence_key

            if immediate:
                self._stable_risk = "critical"
                self._critical_confirmation_frames = CRITICAL_CONFIRMATION_FRAMES
                self._critical_release_frames = 0
            elif self._stable_risk != "critical":
                if raw_risk == "critical":
                    if is_new_evidence:
                        self._critical_confirmation_frames += 1
                    if self._critical_confirmation_frames >= CRITICAL_CONFIRMATION_FRAMES:
                        self._stable_risk = "critical"
                        self._critical_release_frames = 0
                    else:
                        # A single near return is advisory until a fresh frame agrees.
                        self._stable_risk = "warning"
                else:
                    self._critical_confirmation_frames = 0
                    self._critical_release_frames = 0
                    self._stable_risk = raw_risk
            else:
                safely_outside = bool(
                    raw_risk != "critical"
                    and (
                        nearest_obstacle is None
                        or nearest_obstacle >= CRITICAL_RELEASE_CENTER_DISTANCE_M
                    )
                )
                if safely_outside:
                    if is_new_evidence:
                        self._critical_release_frames += 1
                    if self._critical_release_frames >= CRITICAL_CONFIRMATION_FRAMES:
                        self._stable_risk = raw_risk
                        self._critical_confirmation_frames = 0
                        self._critical_release_frames = 0
                else:
                    self._critical_release_frames = 0
            return self._stable_risk, self._critical_confirmation_frames, immediate

    def _current_lidar_metrics(
        self,
        pose: PoseSnapshot,
    ) -> tuple[dict[str, Any] | None, tuple[str, int] | None]:
        if self.lidar_proximity_payload is None:
            return None, None
        try:
            payload = self.lidar_proximity_payload()
            sequence = payload.get("sequence")
            if (
                not payload.get("available")
                or payload.get("source") != "current_lidar"
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 1
            ):
                return None, None
            age = float(payload["age_seconds"])
            frame_timestamp = float(payload["frame_timestamp"])
            if (
                not math.isfinite(age)
                or not math.isfinite(frame_timestamp)
                or age < 0.0
                or age > 1.5
                or abs(pose.timestamp - frame_timestamp) > 2.0
            ):
                return None, None
            nearest_raw = payload.get("nearest_obstacle_distance")
            nearest = None if nearest_raw is None else float(nearest_raw)
            if nearest is not None and (not math.isfinite(nearest) or nearest < 0.0):
                return None, None
            bearing_raw = payload.get("nearest_obstacle_bearing_deg")
            bearing = None if bearing_raw is None else float(bearing_raw)
            direction = payload.get("nearest_obstacle_direction")
            if nearest is not None and (
                bearing is None
                or not math.isfinite(bearing)
                or not isinstance(direction, str)
            ):
                return None, None
            sectors = payload.get("sectors_m", {})
            if not isinstance(sectors, dict):
                return None, None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None, None

        if nearest is not None and nearest <= CRITICAL_CENTER_DISTANCE_M:
            raw_risk = "critical"
        elif nearest is not None and nearest < WARNING_CENTER_DISTANCE_M:
            raw_risk = "warning"
        else:
            raw_risk = "clear"
        cell = (
            None
            if nearest is None
            else {
                "source": "current_lidar",
                "bearing_deg": round(float(bearing), 1),
                "direction": direction,
            }
        )
        return (
            {
                "nearest_obstacle_distance": nearest,
                "nearest_obstacle_cell": cell,
                "robot_surface_clearance_m": (
                    None
                    if nearest is None
                    else max(0.0, nearest - G1_FOOTPRINT_RADIUS_M)
                ),
                "risk_thresholds": {
                    "basis": "estimated_robot_surface_clearance",
                    "warning_m": WARNING_SURFACE_CLEARANCE_M,
                    "critical_m": CRITICAL_SURFACE_CLEARANCE_M,
                    "immediate_critical_m": IMMEDIATE_CRITICAL_SURFACE_CLEARANCE_M,
                    "release_m": CRITICAL_RELEASE_SURFACE_CLEARANCE_M,
                    "confirmation_frames": CRITICAL_CONFIRMATION_FRAMES,
                },
                "lidar_sequence": sequence,
                "lidar_frame_timestamp": frame_timestamp,
                "lidar_age_seconds": round(age, 3),
                "lidar_sectors_m": sectors,
                "critical_diagnostic_candidate": payload.get("critical_diagnostic_candidate"),
                "critical_diagnostic_error": payload.get("critical_diagnostic_error"),
                "risk_raw": raw_risk,
            },
            ("lidar_sequence", sequence),
        )

    def _metrics(self, pose: PoseSnapshot | None) -> dict[str, Any]:
        unavailable = {
            "person_distance": None,
            "nearest_table_distance": None,
            "nearest_table": None,
            "nearest_obstacle_distance": None,
            "nearest_obstacle_cell": None,
            "nearest_costmap_hazard_distance": None,
            "nearest_costmap_cell": None,
            "robot_surface_clearance_m": None,
            "risk_thresholds": {
                "basis": "estimated_robot_surface_clearance",
                "warning_m": WARNING_SURFACE_CLEARANCE_M,
                "critical_m": CRITICAL_SURFACE_CLEARANCE_M,
                "immediate_critical_m": IMMEDIATE_CRITICAL_SURFACE_CLEARANCE_M,
                "release_m": CRITICAL_RELEASE_SURFACE_CLEARANCE_M,
                "confirmation_frames": CRITICAL_CONFIRMATION_FRAMES,
            },
            "forward_view_quality": "unknown",
            "forward_known_ratio": None,
            "forward_open_distance_m": None,
            "risk_source": "current_lidar",
            "risk_raw": "unknown",
            "risk": "unknown",
            "critical_confirmation_frames": 0,
            "critical_confirmation_required": CRITICAL_CONFIRMATION_FRAMES,
            "critical_immediate": False,
            "critical_immediate_allowed": False,
            "lidar_sequence": None,
            "lidar_frame_timestamp": None,
            "lidar_age_seconds": None,
            "lidar_sectors_m": {},
        }
        if pose is None:
            return unavailable

        lidar, evidence_key = self._current_lidar_metrics(pose)
        if lidar is None or evidence_key is None:
            with self._lock:
                latched_risk = (
                    "critical" if self._stable_risk == "critical" else "unknown"
                )
                confirmation_frames = self._critical_confirmation_frames
            physical = {
                "risk": latched_risk,
                "risk_raw": "unknown",
                "critical_confirmation_frames": confirmation_frames,
                "critical_immediate": False,
            }
        else:
            immediate_allowed = True
            try:
                command = self.probe.command()
                if command is not None and len(command) >= 2:
                    planar_command = math.hypot(float(command[0]), float(command[1]))
                    if math.isfinite(planar_command):
                        # A circular footprint does not sweep new space during
                        # pure yaw.  One ultra-close frame while only rotating
                        # must be confirmed; translating toward any direction
                        # still retains the immediate-stop path.
                        immediate_allowed = planar_command > 0.02
            except (AttributeError, TypeError, ValueError, OverflowError):
                immediate_allowed = True
            risk, confirmation_frames, immediate = self._stabilized_risk(
                raw_risk=str(lidar["risk_raw"]),
                nearest_obstacle=lidar["nearest_obstacle_distance"],
                evidence_key=evidence_key,
                immediate_allowed=immediate_allowed,
            )
            physical = {
                **lidar,
                "risk": risk,
                "critical_confirmation_frames": confirmation_frames,
                "critical_immediate": immediate,
                "critical_immediate_allowed": immediate_allowed,
            }

        result = {**unavailable, **physical}
        if self.costmap_payload is None:
            return result
        try:
            payload = self.costmap_payload()
            source = payload.get("source")
            accepted_source = bool(
                source == "live"
                or (
                    source == "isaac_lidar_live"
                    and payload.get("recovery_eligible") is True
                    and payload.get("planning_scope")
                    in {"retrace_only", "explored_same_floor"}
                    and payload.get("identity_verified") is True
                    and payload.get("unknown_is_blocked") is True
                )
            )
            if (
                not payload.get("available")
                or not accepted_source
                or payload.get("frame_id") != "world"
            ):
                return result
            width = int(payload["width"])
            height = int(payload["height"])
            resolution = float(payload["resolution"])
            timestamp = float(payload["timestamp"])
            origin = payload["origin"]
            origin_x = float(origin["x"])
            origin_y = float(origin["y"])
            origin_yaw = float(origin.get("yaw", 0.0))
            raw = base64.b64decode(payload["data"], validate=True)
            if (
                width <= 0
                or height <= 0
                or resolution <= 0.0
                or len(raw) != width * height
                or abs(pose.timestamp - timestamp) > 3.0
            ):
                return result
        except (KeyError, TypeError, ValueError, OverflowError):
            return result

        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)
        relative_x = pose.x - origin_x
        relative_y = pose.y - origin_y
        local_x = cos_yaw * relative_x + sin_yaw * relative_y
        local_y = -sin_yaw * relative_x + cos_yaw * relative_y
        if not (0.0 <= local_x < width * resolution and 0.0 <= local_y < height * resolution):
            return result

        column = math.floor(local_x / resolution)
        row = math.floor(local_y / resolution)
        if raw[row * width + column] == 255:
            # The robot footprint proves that its exact cell is traversable,
            # but an entirely unknown neighborhood is not enough evidence to
            # declare the world clear.
            radius_cells = max(1, math.ceil(0.85 / resolution))
            known_nearby = any(
                raw[nearby_row * width + nearby_column] != 255
                for nearby_row in range(
                    max(0, row - radius_cells),
                    min(height, row + radius_cells + 1),
                )
                for nearby_column in range(
                    max(0, column - radius_cells),
                    min(width, column + radius_cells + 1),
                )
            )
            if not known_nearby:
                return result

        nearest: float | None = None
        nearest_cell: dict[str, Any] | None = None
        for index, value in enumerate(raw):
            if not 50 <= value <= 100:
                continue
            row, column = divmod(index, width)
            cell_x = (column + 0.5) * resolution
            cell_y = (row + 0.5) * resolution
            dx = cell_x - local_x
            dy = cell_y - local_y
            distance = math.hypot(dx, dy)
            world_dx = cos_yaw * dx - sin_yaw * dy
            world_dy = sin_yaw * dx + cos_yaw * dy
            bearing = math.atan2(
                math.sin(math.atan2(world_dy, world_dx) - pose.yaw),
                math.cos(math.atan2(world_dy, world_dx) - pose.yaw),
            )
            cell_detail = {
                "value": value,
                "kind": "height_cost",
                "saturated": value == 100,
                "bearing_deg": round(math.degrees(bearing), 1),
                "direction": self._direction_label(bearing),
            }
            if nearest is None or distance < nearest:
                nearest = distance
                nearest_cell = cell_detail
        view = self._forward_view_metrics(
            raw,
            width=width,
            height=height,
            resolution=resolution,
            local_x=local_x,
            local_y=local_y,
            local_yaw=pose.yaw - origin_yaw,
        )
        return {
            **result,
            "nearest_costmap_hazard_distance": nearest,
            "nearest_costmap_cell": nearest_cell,
            **view,
        }

    def risk_state(self) -> str:
        return str(self._metrics(self.probe.pose())["risk"])

    def recovery_observation(self) -> dict[str, Any]:
        """Return only sensor-derived fields needed by the local safety loop."""

        pose = self.probe.pose()
        return {
            "pose": asdict(pose) if pose else None,
            "command": self.probe.command(),
            "camera_available": self.probe.camera_path() is not None,
            "metrics": self._metrics(pose),
        }

    def _motion_metrics(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._motion_samples)
        if len(samples) < 3:
            return {
                "planar_speed": None,
                "yaw_rate": None,
                "window_seconds": None,
            }

        latest = samples[-1]
        target_timestamp = latest.timestamp - 2.0
        older = [sample for sample in samples[:-1] if sample.timestamp <= target_timestamp]
        reference = older[-1] if older else samples[0]
        elapsed = latest.timestamp - reference.timestamp
        if not math.isfinite(elapsed) or elapsed < 0.5:
            return {
                "planar_speed": None,
                "yaw_rate": None,
                "window_seconds": None,
            }

        yaw_delta = math.atan2(
            math.sin(latest.yaw - reference.yaw),
            math.cos(latest.yaw - reference.yaw),
        )
        return {
            "planar_speed": math.hypot(latest.x - reference.x, latest.y - reference.y) / elapsed,
            "yaw_rate": yaw_delta / elapsed,
            "window_seconds": elapsed,
        }

    def _publish_risk_transition(self, pose: PoseSnapshot) -> None:
        metrics = self._metrics(pose)
        self._record_motion_audit(metrics)
        risk = metrics["risk"]
        if risk == self._last_risk:
            self._pending_risk = None
            return

        risk_rank = {"clear": 0, "warning": 1, "critical": 2}
        escalating = (
            self._last_risk == "unknown"
            or risk_rank.get(risk, 0) > risk_rank.get(self._last_risk, 0)
        )
        if not escalating:
            now = time.monotonic()
            if self._pending_risk != risk:
                self._pending_risk = risk
                self._pending_risk_since = now
                return
            if now - self._pending_risk_since < 1.5:
                return

        if risk == "critical" and metrics.get("critical_diagnostic_candidate"):
            from harness.robots.g1.mujoco.critical_diagnostics import pin_critical_frame
            metrics["critical_diagnostic"] = pin_critical_frame(
                metrics["critical_diagnostic_candidate"], metrics["lidar_sequence"],
                metrics["lidar_frame_timestamp"])
        self._last_risk = risk
        self._pending_risk = None
        level = "danger" if risk == "critical" else "warning" if risk == "warning" else "info"
        self.events.append(
            "perception",
            "observation",
            f"Proximity state: {risk}",
            (
                "estimated robot-surface clearance "
                f"{metrics['robot_surface_clearance_m']:.2f} m "
                "(current-lidar centre return "
                f"{metrics['nearest_obstacle_distance']:.2f} m)"
                if metrics["nearest_obstacle_distance"] is not None
                else (
                    "current lidar reports no body-height obstacle"
                    if metrics.get("lidar_sequence") is not None
                    else "current-lidar proximity unavailable"
                )
            ),
            level=level,
            data=metrics,
        )

    # 生成操作面板使用的世界观测快照。
    def snapshot(self) -> dict[str, Any]:
        pose = self.probe.pose()
        command = self.probe.command()
        metrics = self._metrics(pose)
        motion = self._motion_metrics()
        commanded_zero = command is not None and max(
            abs(command[0]), abs(command[1]), abs(command[5])
        ) < 0.01
        motion["commanded_zero"] = commanded_zero
        motion["unexpected_motion"] = bool(
            commanded_zero
            and (
                (motion["planar_speed"] or 0.0) > 0.025
                or abs(motion["yaw_rate"] or 0.0) > 0.08
            )
        )
        with self._lock:
            trail = list(self._trail)
        observations: list[dict[str, Any]] = []
        if metrics["nearest_obstacle_distance"] is not None:
            cell = metrics.get("nearest_obstacle_cell") or {}
            direction = str(cell.get("direction") or "unknown")
            observations.append(
                {
                    "label": f"nearest physical obstacle ({direction})",
                    "value": round(metrics["nearest_obstacle_distance"], 3),
                    "unit": "m",
                    "severity": (
                        "danger"
                        if metrics["risk"] == "critical"
                        else "normal"
                    ),
                }
            )
        hazard_distance = metrics.get("nearest_costmap_hazard_distance")
        nearest_cell = metrics.get("nearest_costmap_cell") or {}
        if (
            hazard_distance is not None
            and nearest_cell.get("kind") == "height_cost"
        ):
            observations.append(
                {
                    "label": f"nearest map height cost ({nearest_cell.get('direction', 'unknown')})",
                    "value": round(float(hazard_distance), 3),
                    "unit": "m",
                    "severity": "warning",
                }
            )
        if metrics.get("robot_surface_clearance_m") is not None:
            observations.append(
                {
                    "label": "estimated robot-surface clearance",
                    "value": round(float(metrics["robot_surface_clearance_m"]), 3),
                    "unit": "m",
                    "severity": (
                        "danger"
                        if metrics["risk"] == "critical"
                        else "warning"
                        if metrics["risk"] == "warning"
                        else "normal"
                    ),
                }
            )
        if metrics.get("forward_view_quality") != "unknown":
            observations.append(
                {
                    "label": (
                        "first-person forward view "
                        f"({metrics['forward_view_quality']})"
                    ),
                    "value": metrics.get("forward_open_distance_m") or 0.0,
                    "unit": "m",
                    "severity": (
                        "normal"
                        if metrics["forward_view_quality"] == "good"
                        else "warning"
                    ),
                }
            )
        if motion["planar_speed"] is not None:
            observations.append(
                {
                    "label": "measured planar speed",
                    "value": round(motion["planar_speed"], 3),
                    "unit": "m/s",
                    "severity": "danger" if motion["unexpected_motion"] else "normal",
                }
            )
        return {
            "pose": asdict(pose) if pose else None,
            "command": command,
            "camera_available": self.probe.camera_path() is not None,
            "trail": trail,
            "metrics": metrics,
            "motion": motion,
            "observations": observations,
            "sampled_at": utc_now(),
        }

    # 生成 Agent 使用的观测视图，控制可暴露的场景和传感器信息。
    def agent_snapshot(self) -> dict[str, Any]:
        """Return live observations without UI-only map geometry and pose history."""

        snapshot = self.snapshot()
        result = {
            "pose": snapshot["pose"],
            "command": snapshot["command"],
            "camera_available": snapshot["camera_available"],
            "metrics": snapshot["metrics"],
            "motion": snapshot["motion"],
            "observations": snapshot["observations"],
            "sampled_at": snapshot["sampled_at"],
        }
        if self.mapping_status is not None:
            mapping = self.mapping_status()
            result["mapping"] = {
                key: mapping.get(key)
                for key in (
                    "available",
                    "running",
                    "source",
                    "age_seconds",
                    "width",
                    "height",
                    "resolution",
                    "origin",
                    "cells",
                )
            }
        if self.navigation_status is not None:
            navigation = self.navigation_status()
            result["navigation_bridge"] = {
                key: navigation.get(key)
                for key in (
                    "running",
                    "active",
                    "source_topic",
                    "target_topic",
                    "last_input_age_seconds",
                    "last_command",
                    "watchdog_stops",
                )
            }
        if self.recovery_status is not None:
            recovery = self.recovery_status()
            result["safety_recovery"] = {
                key: recovery.get(key)
                for key in (
                    "enabled",
                    "state",
                    "active",
                    "safety_hold",
                    "reconsideration_required",
                    "reason",
                    "last_outcome",
                    "recovery_speed_limit_mps",
                    "safe_clearance_m",
                    "forward_view_quality",
                    "retreat_distance_m",
                )
            }
        return result


# 管理面板关联的模拟器：场景配置、启动参数、进程生命周期和就绪状态。
class SimulationSupervisor:
    """Reuse or own one DimOS stack for the selected simulator backend."""

    def __init__(
        self,
        events: EventStore,
        asset_root: Path,
        third_person: FreshJpegFrame,
        *,
        backend: str = "mujoco",
        isaac_paths: IsaacRuntimePaths | None = None,
        go2_paths: Go2RuntimePaths | None = None,
        third_person_enabled: bool = False,
        blind_runtime_paths: BlindRuntimePaths | None = None,
        lidar_proximity: FreshLidarProximity | None = None,
    ) -> None:
        self.events = events
        self.asset_root = asset_root
        self.backend = normalize_ui_backend(backend)
        self.isaac_paths = isaac_paths or IsaacRuntimePaths.configured()
        self.go2_paths = go2_paths or Go2RuntimePaths.configured(
            asset_root=asset_root
        )
        self.third_person = third_person
        self.third_person_enabled = bool(third_person_enabled)
        self.blind_runtime_paths = blind_runtime_paths
        if self.blind_runtime_paths is not None and self.backend != "mujoco":
            raise ValueError("Blind evaluation currently requires the MuJoCo backend")
        self.lidar_proximity = lidar_proximity
        self.runtime_dir = (
            blind_runtime_paths.run_directory / "simulator"
            if blind_runtime_paths is not None
            else asset_root
            / (
                "runtime/luxi-isaac-ui"
                if self.backend == "isaac-g1"
                else "runtime/luxi-go2-ui"
                if self.backend == "mujoco-go2"
                else "runtime/luxi-ui"
            )
        )
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.process: subprocess.Popen[str] | None = None
        self.owned = False
        # Process ownership and runtime ownership diverge when fault injection
        # kills the DimOS wrapper while its Isaac container remains alive.  Keep
        # the latter only after this supervisor observed its own READY marker so
        # close() can reap that container without ever taking over an unrelated
        # fixed-name container that made the launch command fail early.
        self._runtime_owned = False
        self.starting = False
        self._lock = threading.Lock()
        self._runner_thread: threading.Thread | None = None
        self._stop_callback: Callable[[str], Any] | None = None
        self.scene_payload_path = self.runtime_dir / "operator-scene.json"
        self._operator_scene: OperatorScene | None = None
        self._isaac_scene: str | None = None
        self._go2_scene = (
            GO2_SCENE_ID
            if self.backend == "mujoco-go2"
            else None
        )
        if self.blind_runtime_paths is None:
            if self.backend == "mujoco-go2":
                return
            configured_scene = os.environ.get(
                "LUXI_ISAAC_SCENE" if self.backend == "isaac-g1" else "LUXI_UI_SCENE",
                "grid" if self.backend == "isaac-g1" else "office1",
            ).strip()
            try:
                configured_seed = int(
                    os.environ.get("LUXI_UI_SCENE_SEED", "0")
                    if self.backend == "mujoco"
                    else "0"
                )
            except ValueError as error:
                raise ValueError("LUXI_UI_SCENE_SEED must be an integer") from error
            configured_person = bool(
                self.backend == "mujoco"
                and env_enabled("LUXI_UI_SCENE_PERSON", False)
            )
            self.configure_scene(
                configured_scene,
                configured_seed,
                include_person=configured_person,
            )

    def selected_scene(self) -> OperatorScene | str | None:
        with self._lock:
            return (
                self._isaac_scene
                if self.backend == "isaac-g1"
                else self._go2_scene
                if self.backend == "mujoco-go2"
                else self._operator_scene
            )

    def configure_scene(
        self,
        scene_id: str,
        seed: int,
        *,
        include_person: bool = False,
    ) -> OperatorScene | str:
        if self.blind_runtime_paths is not None:
            raise RuntimeError("正式盲测场景由 run token 锁定，UI 不允许切换")
        if self.backend == "mujoco-go2":
            if (
                scene_id != GO2_SCENE_ID
                or seed != 0
                or include_person
            ):
                raise ValueError("Go2 当前只有固定单 Go2 仓库巡检场景")
            return scene_id
        if self.backend == "isaac-g1":
            supported = {item["scene_id"] for item in ISAAC_SCENE_CATALOG}
            if scene_id not in supported:
                raise ValueError(f"不支持的 Isaac 场景：{scene_id}")
            if seed != 0:
                raise ValueError("Isaac 场景当前不支持 seed")
            descriptor = next(
                item for item in ISAAC_SCENE_CATALOG if item["scene_id"] == scene_id
            )
            if include_person and not descriptor["supports_person"]:
                raise ValueError("所选 Isaac 场景不支持人物")
            with self._lock:
                self._isaac_scene = scene_id
            return scene_id
        scene = build_operator_scene(
            scene_id,
            seed,
            include_person=include_person,
        )
        write_operator_scene_payload(self.scene_payload_path, scene)
        with self._lock:
            self._operator_scene = scene
        return scene

    def restore_scene(self, scene: OperatorScene | str | None) -> None:
        if self.backend == "mujoco-go2":
            return
        if self.backend == "isaac-g1":
            with self._lock:
                self._isaac_scene = scene if isinstance(scene, str) else None
            return
        if scene is None:
            try:
                self.scene_payload_path.unlink()
            except FileNotFoundError:
                pass
        else:
            write_operator_scene_payload(self.scene_payload_path, scene)
        with self._lock:
            self._operator_scene = scene

    def scene_status(self) -> dict[str, Any]:
        if self.blind_runtime_paths is not None:
            return {
                "switchable": False,
                "reason": "blind_mode_locked",
            }
        scene = self.selected_scene()
        if scene is None:
            return {"switchable": False, "reason": "scene_not_configured"}
        if self.backend == "mujoco-go2":
            return {
                "switchable": False,
                "reason": "go2_fixed_scene",
                "selected_id": GO2_SCENE_ID,
                "seed": 0,
                "include_person": True,
                "catalog": [],
                "selected": {
                    "scene_id": GO2_SCENE_ID,
                    "label": "单 Go2 仓库巡检",
                    "description": "大型遮挡仓库 · 单机区域巡检 · MID-360 在线建图",
                    "complexity": "standard",
                    "source": "luxi_go2_mujoco",
                    "seedable": False,
                    "supports_person": True,
                },
            }
        if self.backend == "isaac-g1":
            assert isinstance(scene, str)
            selected = next(
                item for item in ISAAC_SCENE_CATALOG if item["scene_id"] == scene
            )
            return {
                "switchable": True,
                "selected_id": scene,
                "seed": 0,
                "include_person": bool(selected["supports_person"]),
                "catalog": [dict(item) for item in ISAAC_SCENE_CATALOG],
                "selected": dict(selected),
            }
        assert isinstance(scene, OperatorScene)
        descriptor = operator_scene_descriptor(scene.scene_id)
        return {
            "switchable": True,
            "selected_id": scene.scene_id,
            "seed": scene.seed,
            "include_person": scene.person is not None,
            "catalog": [item.public() for item in list_operator_scenes()],
            "selected": descriptor.public(),
        }

    def capabilities(self) -> dict[str, bool]:
        isaac = self.backend == "isaac-g1"
        capabilities = get_backend_profile(self.backend).capabilities.as_dict()
        capabilities["third_person"] = bool(
            capabilities["third_person"] and self.third_person_enabled
        )
        lidar_safety = capabilities["lidar_safety"] and not isaac
        if self.backend == "mujoco-go2" and self.lidar_proximity is not None:
            lidar_safety = bool(self.lidar_proximity.status().get("available"))
        if isaac and self.lidar_proximity is not None:
            lidar_safety = bool(self.lidar_proximity.status().get("available"))
        capabilities["lidar_safety"] = lidar_safety
        return capabilities

    def third_person_profile(self) -> dict[str, int | float]:
        if self.backend == "isaac-g1":
            return {
                "width": bounded_env_int(
                    "LUXI_ISAAC_OBSERVER_WIDTH", 320, 1, 960
                ),
                "height": bounded_env_int(
                    "LUXI_ISAAC_OBSERVER_HEIGHT", 180, 1, 540
                ),
                "hz": bounded_env_float(
                    "LUXI_ISAAC_OBSERVER_HZ", 1.0, 0.2, 10.0
                ),
            }
        return {"width": 640, "height": 360, "hz": 10.0}

    def backend_ready(self) -> bool:
        if self.backend == "mujoco-go2":
            return bool(
                read_fresh_go2_state(self.go2_paths, max_age_s=1.5) is not None
                and FreshJpegFrame(self.go2_paths.camera, 1.5).jpeg() is not None
            )
        if self.backend != "isaac-g1":
            return port_open(9990) and port_open(7779)
        return bool(
            read_fresh_state(self.isaac_paths, max_age_s=1.5) is not None
            and read_fresh_camera_frame(self.isaac_paths, max_age_s=1.5) is not None
        )

    def _external_go2_processes(self) -> list[int]:
        """Return only Go2 runtimes that target this supervisor's runtime path."""
        if self.backend != "mujoco-go2":
            return []
        expected_root = str(self.go2_paths.root.resolve())
        matches: list[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                arguments = [
                    item.decode("utf-8", errors="replace")
                    for item in (entry / "cmdline").read_bytes().split(b"\0")
                    if item
                ]
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            try:
                module_index = arguments.index("-m")
                runtime_index = arguments.index("--ui-runtime")
            except ValueError:
                continue
            if module_index + 1 >= len(arguments) or runtime_index + 1 >= len(arguments):
                continue
            if arguments[module_index + 1] != "harness.robots.go2.go2_mujoco":
                continue
            try:
                runtime_root = str(Path(arguments[runtime_index + 1]).resolve())
            except OSError:
                continue
            if runtime_root == expected_root:
                matches.append(int(entry.name))
        return sorted(matches)

    def force_stop(self) -> int | None:
        if self.backend == "mujoco-go2":
            return Go2CommandWriter(self.go2_paths).request_stop("force_stop")
        if self.backend != "isaac-g1":
            return None
        if self._stop_callback is not None:
            result = self._stop_callback("simulation_supervisor_force_stop")
            evidence = (
                result.get("safety_evidence")
                if isinstance(result, dict)
                else None
            )
            if isinstance(evidence, dict) and evidence.get(
                "stop_command_completed"
            ):
                sequence = evidence.get("command_sequence")
                return sequence if isinstance(sequence, int) else None
        return IsaacCommandWriter(self.isaac_paths).stop()

    def bind_stop_callback(self, callback: Callable[[str], Any]) -> None:
        with self._lock:
            if self._stop_callback is not None and self._stop_callback is not callback:
                raise RuntimeError("simulation stop callback already has an owner")
            self._stop_callback = callback

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self.process
            owned = self.owned
            starting = self.starting
        owned_process_alive = process is not None and process.poll() is None
        external_pids = (
            self._external_go2_processes()
            if self.backend == "mujoco-go2" and not owned_process_alive
            else []
        )
        process_alive = owned_process_alive or bool(external_pids)
        bridge_ready = self.backend_ready()
        if self.backend == "isaac-g1" and process_alive and bridge_ready and starting:
            with self._lock:
                if self.process is process:
                    self.starting = False
                    starting = False
        observer = self.third_person.status()
        lidar = (
            self.lidar_proximity.status()
            if self.lidar_proximity is not None
            else {"available": False, "age_seconds": None, "sequence": None}
        )
        return {
            "backend": self.backend,
            "backend_label": (
                "Isaac Sim 5.1 / Unitree G1"
                if self.backend == "isaac-g1"
                else "MuJoCo / Unitree Go2 · 海康 MV-CU013-A0UC + MID-360"
                if self.backend == "mujoco-go2"
                else "MuJoCo / Unitree G1"
            ),
            "bridge_ready": bridge_ready,
            "capabilities": self.capabilities(),
            "mcp": bridge_ready if self.backend == "mujoco-go2" else port_open(9990),
            "command_center": bridge_ready if self.backend == "mujoco-go2" else port_open(7779),
            "process_alive": process_alive,
            "owned_by_ui": owned,
            "starting": starting,
            "agent_runtime": "harness",
            "pid": (
                process.pid
                if owned_process_alive and process is not None
                else external_pids[0]
                if external_pids
                else None
            ),
            "external_pids": external_pids,
            "third_person_enabled": self.third_person_enabled,
            "third_person_available": observer["available"],
            "third_person_age_seconds": observer["age_seconds"],
            "third_person_profile": self.third_person_profile(),
            "lidar_proximity_available": lidar["available"],
            "lidar_proximity_age_seconds": lidar["age_seconds"],
            "lidar_proximity_sequence": lidar.get("sequence"),
            "scene": self.scene_status(),
        }

    # 在后台启动仿真，避免启动耗时阻塞面板请求。
    def start_async(self) -> bool:
        with self._lock:
            if self.starting:
                return False
            if self.backend == "mujoco-go2":
                if (
                    (self.process is not None and self.process.poll() is None)
                    or self.backend_ready()
                ):
                    self.events.append(
                        "simulation",
                        "lifecycle",
                        "Reused running Go2 MuJoCo",
                        "Fresh Go2 state and HIKROBOT RGB frame are ready.",
                    )
                    return True
                self.starting = True
                runner = threading.Thread(
                    target=self._start,
                    name="go2-simulation-starter",
                    daemon=True,
                )
                self._runner_thread = runner
                runner.start()
                return True
            if port_open(9990) and port_open(7779):
                if self.backend == "isaac-g1" and not self.backend_ready():
                    self.events.append(
                        "simulation",
                        "error",
                        "Refused foreign DimOS instance",
                        "Ports 9990/7779 are occupied but no fresh Isaac G1 bridge is present.",
                        level="danger",
                    )
                    return False
                self.events.append(
                    "simulation", "lifecycle", "Reused running DimOS", "Ports 9990 and 7779 are ready."
                )
                return True
            self.starting = True
            runner = threading.Thread(
                target=self._start,
                name="simulation-starter",
                daemon=True,
            )
            self._runner_thread = runner
        runner.start()
        return True

    # 根据后端与运行模式组装模拟器子进程的环境变量。
    def launch_environment(
        self,
        inherited: dict[str, str] | None = None,
    ) -> dict[str, str]:
        environment = dict(os.environ if inherited is None else inherited)
        environment.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
        environment.setdefault("PYTEST_VERSION", "1")
        environment.setdefault("HF_HUB_OFFLINE", "1")
        environment.setdefault("TRANSFORMERS_OFFLINE", "1")
        environment.setdefault("DIMOS_VIEWER", "none")
        environment.pop("LUXI_THIRD_PERSON_FRAME", None)
        environment.pop("LUXI_LIDAR_PROXIMITY_PATH", None)
        environment.pop(OPERATOR_SCENE_PAYLOAD_ENV, None)
        if self.backend == "mujoco":
            environment.setdefault(
                "LUXI_TAGGED_LOCATIONS_PATH",
                str(self.runtime_dir / "spatial-memory/tagged-locations.json"),
            )
        if self.backend == "mujoco-go2":
            environment["LUXI_UI_BACKEND"] = "mujoco-go2"
            environment["LUXI_SIM_BACKEND"] = "mujoco-go2"
            environment["LUXI_GO2_RUNTIME_DIR"] = str(self.go2_paths.root)
            if not env_enabled_from_mapping(
                environment,
                "LUXI_GO2_NATIVE_VIEWER",
                False,
            ):
                environment["MUJOCO_GL"] = "egl"
                environment["PYOPENGL_PLATFORM"] = "egl"
        if self.backend == "isaac-g1":
            environment["LUXI_UI_BACKEND"] = "isaac-g1"
            environment["LUXI_SIM_BACKEND"] = "isaac-g1"
            environment["LUXI_ISAAC_RUNTIME_DIR"] = str(self.isaac_paths.root)
            environment["LUXI_SIM_CONTROL_PATH"] = str(self.isaac_paths.root)
            environment["LUXI_AGENT_STOP_CONTROL_PATH"] = str(
                self.isaac_paths.root / "agent-stop"
            )
            environment["LUXI_AGENT_NAVIGATION_CONTROL_PATH"] = str(
                self.isaac_paths.root / "agent-navigation"
            )
            environment.setdefault(
                "LUXI_ISAAC_TAGGED_LOCATIONS_PATH",
                str(self.runtime_dir / "spatial-memory/tagged-locations.json"),
            )
            environment["LUXI_ISAAC_SPATIAL_MEMORY_PATH"] = str(
                self.runtime_dir / "spatial-memory/clip"
            )
            # RTX lidar needs six rendered sectors per complete scan. Ten
            # sensor frames per second keeps a verified 360-degree scan below
            # the 1.25 s freshness gate while the compact RGB profile limits
            # GPU load.
            selected = self.selected_scene()
            detailed_task_scene = selected == "task_apartment"
            environment.setdefault("LUXI_ISAAC_SENSOR_HZ", "10")
            environment.setdefault(
                "LUXI_ISAAC_SENSOR_WIDTH",
                "640" if detailed_task_scene else "256",
            )
            environment.setdefault(
                "LUXI_ISAAC_SENSOR_HEIGHT",
                "360" if detailed_task_scene else "144",
            )
            environment.setdefault("LUXI_ISAAC_LIDAR_ENABLED", "1")
            environment.setdefault(
                "LUXI_ISAAC_OBSERVER_ENABLED",
                "1" if self.third_person_enabled else "0",
            )
            environment.setdefault("LUXI_ISAAC_OBSERVER_HZ", "1")
            environment.setdefault("LUXI_ISAAC_OBSERVER_WIDTH", "320")
            environment.setdefault("LUXI_ISAAC_OBSERVER_HEIGHT", "180")
            if isinstance(selected, str):
                environment["LUXI_ISAAC_SCENE"] = selected
        if (
            self.backend == "mujoco"
            and self.third_person_enabled
            and not env_enabled_from_mapping(
                environment, "LUXI_BLIND_MODE"
            )
        ):
            environment["LUXI_THIRD_PERSON_FRAME"] = str(self.third_person.path)
        if self.lidar_proximity is not None:
            environment["LUXI_LIDAR_PROXIMITY_PATH"] = str(
                self.lidar_proximity.path
            )
        if self.blind_runtime_paths is not None:
            environment["LUXI_HEAD_DEPTH_PATH"] = str(
                self.blind_runtime_paths.head_depth_path
            )
            environment["LUXI_SIM_CONTROL_PATH"] = str(
                self.blind_runtime_paths.control_directory
            )
        elif self.backend == "mujoco" and self.selected_scene() is not None:
            environment[OPERATOR_SCENE_PAYLOAD_ENV] = str(self.scene_payload_path)
        return environment

    # 根据机器人后端和 Agent 配置生成模拟器启动命令。
    def launch_command(self) -> list[str]:
        if self.backend == "mujoco-go2":
            go2_command = (
                "go2-ros-demo"
                if os.environ.get("LUXI_GO2_FLEET_TRANSPORT", "").strip().lower()
                in {"ros", "ros2", "dds"}
                else "go2-demo"
            )
            command = [
                str(PROJECT_ROOT / "scripts/dimos.sh"),
                go2_command,
                "--ui-runtime",
                str(self.go2_paths.root),
            ]
            if not env_enabled("LUXI_GO2_NATIVE_VIEWER", False):
                command.insert(2, "--headless")
            return command
        command = [str(PROJECT_ROOT / "scripts/dimos.sh"),
                   "g1-isaac-tools" if self.backend == "isaac-g1" else "g1-tools"]
        if self.blind_runtime_paths is None:
            return command
        memory = self.blind_runtime_paths.memory_directory
        options = {
            "spatialmemory.db_path": memory / "chromadb_data",
            "spatialmemory.visual_memory_path": memory / "visual_memory.pkl",
            "spatialmemory.output_dir": memory / "visual",
            "spatialmemory.collection_name": "blind_spatial_memory",
            "spatialmemory.new_memory": "true",
        }
        for name, value in options.items():
            command.extend(["--option", f"{name}={value}"])
        return command

    def _start(self) -> None:
        command = self.launch_command()
        environment = self.launch_environment()
        log_path = self.runtime_dir / "simulation.log"

        try:
            self.third_person.path.unlink()
        except OSError:
            pass
        if self.backend == "mujoco-go2":
            self.go2_paths.clear_observations()
        if self.lidar_proximity is not None:
            self.lidar_proximity.clear()

        backend_title = (
            "G1 Isaac Sim"
            if self.backend == "isaac-g1"
            else "Go2 MuJoCo"
            if self.backend == "mujoco-go2"
            else "G1 MuJoCo"
        )
        blueprint = self.launch_command()[1]
        self.events.append(
            "simulation",
            "lifecycle",
            f"Starting {backend_title}",
            f"Launching isolated {blueprint} blueprint.",
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            with self._lock:
                self.starting = False
                if self._runner_thread is threading.current_thread():
                    self._runner_thread = None
            self.events.append(
                "simulation", "error", "Failed to start DimOS", str(error), level="danger"
            )
            return

        with self._lock:
            self.process = process
            self.owned = True

        ready_announced = False
        with log_path.open("a", encoding="utf-8") as log_file:
            assert process.stdout is not None
            for raw_line in process.stdout:
                log_file.write(raw_line)
                log_file.flush()
                line = ANSI_RE.sub("", raw_line).strip()
                if not line:
                    continue
                ready_marker = (
                    "Isaac G1 bridge is ready"
                    if self.backend == "isaac-g1"
                    else "Go2 UI runtime ready"
                    if self.backend == "mujoco-go2"
                    else "MuJoCo process started successfully"
                )
                if ready_marker in line:
                    ready_announced = True
                    with self._lock:
                        if self.process is process:
                            self.starting = False
                            if self.backend == "isaac-g1":
                                self._runtime_owned = True
                    self.events.append(
                        "simulation",
                        "lifecycle",
                        f"{backend_title} ready",
                        (
                            "Go2 physics, HIKROBOT RGB and MID-360 are running."
                            if self.backend == "mujoco-go2"
                            else "G1 policy, physics, RGB and odometry are running."
                        ),
                    )
                elif "Loaded policy:" in line:
                    self.events.append("simulation", "model", "G1 walking policy loaded", line)
                elif "Traceback" in line or "[err]" in line.lower() or " error" in line.lower():
                    self.events.append("simulation", "error", "DimOS runtime message", line, level="danger")

        exit_code = process.wait()
        with self._lock:
            if self.process is process:
                self.process = None
                self.owned = False
                self.starting = False
            if self._runner_thread is threading.current_thread():
                self._runner_thread = None
        level = "warning" if exit_code in {0, -signal.SIGTERM} else "danger"
        self.events.append(
            "simulation",
            "lifecycle",
            f"{backend_title} stopped",
            f"Process exited with code {exit_code}.",
            level=level,
            data={"ready_was_announced": ready_announced},
        )

    def stop_owned(self) -> None:
        with self._lock:
            process = self.process
            owned = self.owned
            runtime_owned = self._runtime_owned
            runner = self._runner_thread
        process_alive = bool(process is not None and process.poll() is None)
        if self.backend == "isaac-g1":
            if not ((owned and process_alive) or runtime_owned):
                return
        elif not owned or not process_alive:
            return
        try:
            self.force_stop()
        except OSError as error:
            self.events.append(
                "safety",
                "error",
                "Isaac stop command failed before shutdown",
                str(error),
                level="danger",
            )
        self.events.append(
            "simulation",
            "lifecycle",
            f"Stopping owned {self.backend} DimOS instance",
        )
        if self.backend == "mujoco-go2":
            if not owned or process is None:
                return
            try:
                process.terminate()
                process.wait(timeout=5)
            except OSError:
                return
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if runner is not None and runner is not threading.current_thread():
                runner.join(timeout=5)
            return
        stop_command = (
            [str(PROJECT_ROOT / "scripts/isaac_g1.sh"), "stop"]
            if self.backend == "isaac-g1"
            and runtime_owned
            and not process_alive
            else [str(PROJECT_ROOT / "scripts/dimos.sh"), "stop"]
        )
        try:
            subprocess.run(
                stop_command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            if process_alive and process is not None:
                process.terminate()
        if process_alive and process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError("DimOS 进程在 SIGKILL 后仍未退出") from error
        if runner is not None and runner is not threading.current_thread():
            runner.join(timeout=5)
            if runner.is_alive():
                raise RuntimeError("DimOS 日志线程未能结束，拒绝并发启动新仿真")
        with self._lock:
            if self.process is process:
                self.process = None
                self.owned = False
                self.starting = False
            self._runtime_owned = False

    def stop_for_reset(self) -> None:
        """Stop an owned runtime or an exact-path Go2 runtime left by an old UI."""
        with self._lock:
            process = self.process
            owned = self.owned
        if owned and process is not None and process.poll() is None:
            self.stop_owned()
            return
        if self.backend != "mujoco-go2":
            self.stop_owned()
            return

        pids = self._external_go2_processes()
        if not pids:
            if self.backend_ready():
                raise RuntimeError(
                    "检测到新鲜 Go2 数据，但无法确认其进程身份；为避免误停其他项目，拒绝复位"
                )
            return
        self.events.append(
            "simulation",
            "lifecycle",
            "Stopping previous Go2 MuJoCo runtime",
            f"Safely taking over exact runtime {self.go2_paths.root} (pid={','.join(map(str, pids))}).",
            level="warning",
        )
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + 7.0
        remaining = set(pids)
        while remaining and time.monotonic() < deadline:
            current = set(self._external_go2_processes())
            remaining.intersection_update(current)
            if remaining:
                time.sleep(0.05)
        for pid in remaining:
            # Revalidate the exact command before escalating, guarding PID reuse.
            if pid not in self._external_go2_processes():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if remaining.intersection(self._external_go2_processes()):
            raise RuntimeError("旧 Go2 MuJoCo 进程在 SIGKILL 后仍未退出")


# 通过 Docker Compose 管理隔离的单 Go2 ROS 服务，配合 MuJoCo 一起复位。
class Go2RosComposeLifecycle:
    """Restart the isolated single-robot ROS stack together with its MuJoCo hardware side."""

    def __init__(self, *, enabled: bool, compose_file: Path | None = None) -> None:
        self.enabled = bool(enabled)
        self.compose_file = compose_file or PROJECT_ROOT / "docker/compose.go2-ros.yml"

    def _run(self, *arguments: str, timeout: float) -> None:
        if not self.enabled:
            return
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", str(self.compose_file), *arguments],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"ROS 2 容器操作失败：{error}") from error
        if result.returncode != 0:
            detail = result.stdout.strip()[-2_000:]
            raise RuntimeError(f"ROS 2 容器操作失败：{detail}")

    def stop(self) -> None:
        self._run("down", "--remove-orphans", timeout=45.0)

    def start(self) -> None:
        self._run("up", "-d", "--force-recreate", timeout=120.0)


# 协调完整实验复位：先停止任务与仿真，再清理状态并重启相关组件。
class ExperimentResetController:
    """Safely rebuild one complete UI-managed simulation experiment."""

    PHASE_MESSAGES = {
        "idle": "尚未复位",
        "stopping": "正在停车并停止旧仿真",
        "clearing": "正在清理地图、轨迹和空间记忆",
        "starting": "正在启动全新仿真与建图",
        "ready": "复位完成",
        "failed": "复位失败",
    }

    def __init__(
        self,
        events: EventStore,
        *,
        simulation: Any,
        agent: Any,
        navigation: Any,
        costmap: Any,
        monitor: Any,
        probe: Any,
        third_person: Any,
        lidar_proximity: Any | None = None,
        spatial_memory_path: Path,
        safety_recovery: Any | None = None,
        task_recovery: Any | None = None,
        auxiliary_lifecycle: Any | None = None,
        runtime_host: Any | None = None,
        cancel_long_tasks: Callable[[str], bool] | None = None,
        readiness_timeout: float = 120.0,
        agent_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.events = events
        self.simulation = simulation
        self.agent = agent
        self.navigation = navigation
        self.costmap = costmap
        self.monitor = monitor
        self.safety_recovery = safety_recovery
        self.task_recovery = task_recovery
        self.auxiliary_lifecycle = auxiliary_lifecycle
        self.runtime_host = runtime_host
        self.cancel_long_tasks = cancel_long_tasks
        self.probe = probe
        self.third_person = third_person
        self.lidar_proximity = lidar_proximity
        self.spatial_memory_path = spatial_memory_path
        self.readiness_timeout = max(1.0, float(readiness_timeout))
        self.agent_timeout = max(1.0, float(agent_timeout))
        self.clock = clock

        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._phase = "idle"
        self._generation = 0
        self._started_at: str | None = None
        self._completed_at: str | None = None
        self._last_error = ""
        self._cleared: list[str] = []

    def status(self) -> dict[str, Any]:
        with self._lock:
            phase = self._phase
            return {
                "active": self._active,
                "phase": phase,
                "message": self.PHASE_MESSAGES.get(phase, phase),
                "generation": self._generation,
                "started_at": self._started_at,
                "completed_at": self._completed_at,
                "last_error": self._last_error,
                "cleared": list(self._cleared),
            }

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def start_async(self) -> tuple[bool, str]:
        simulation = self.simulation.status()
        externally_managed = bool(
            (
                simulation.get("mcp")
                or simulation.get("command_center")
                or simulation.get("process_alive")
            )
            and not simulation.get("owned_by_ui")
        )
        exact_go2_takeover = bool(
            simulation.get("backend") == "mujoco-go2"
            and callable(getattr(self.simulation, "stop_for_reset", None))
        )
        if externally_managed and not exact_go2_takeover:
            return False, "当前 DimOS 不是由此 UI 托管；为避免停止其他项目，拒绝复位"

        with self._lock:
            if self._active:
                return False, "实验正在复位，请等待当前流程完成"
            self._active = True
            self._phase = "stopping"
            self._generation += 1
            generation = self._generation
            self._started_at = utc_now()
            self._completed_at = None
            self._last_error = ""
            self._cleared = []
            self._cancel.clear()
            thread = threading.Thread(
                target=self._run,
                args=(generation,),
                name="experiment-reset",
                daemon=True,
            )
            self._thread = thread
        self.events.append(
            "system",
            "reset",
            "Experiment reset requested",
            "Stopping motion before rebuilding simulation, mapping, and spatial memory.",
            level="warning",
            data={"generation": generation},
        )
        thread.start()
        return True, "复位请求已接受；正在安全停车并重建实验"

    def _check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise RuntimeError("UI 正在关闭，复位流程已取消")

    def _wait_agent_idle(self) -> None:
        deadline = self.clock() + self.agent_timeout
        while bool(self.agent.status().get("busy")):
            self._check_cancelled()
            if self.clock() >= deadline:
                raise RuntimeError("Agent 未能在限时内结束，拒绝启动新仿真")
            self._cancel.wait(0.1)

    def _wait_simulation_stopped(self) -> None:
        deadline = self.clock() + 10.0
        while True:
            status = self.simulation.status()
            if not any(
                status.get(key)
                for key in ("mcp", "command_center", "process_alive", "starting")
            ):
                return
            self._check_cancelled()
            if self.clock() >= deadline:
                raise RuntimeError("旧仿真端口未能完全释放")
            self._cancel.wait(0.1)

    def _clear_spatial_memory(self) -> bool:
        path = self.spatial_memory_path
        if path.is_symlink():
            raise RuntimeError(f"拒绝清理符号链接形式的空间记忆目录：{path}")
        removed = path.exists()
        if removed:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return removed

    def _simulation_ready(self, reset_started_epoch: float) -> bool:
        runtime_host = getattr(self, "runtime_host", None)
        if runtime_host is not None:
            runtime_status = runtime_host.reconcile()
            if runtime_status.state.value != "READY":
                return False
        simulation = self.simulation.status()
        if not all(
            simulation.get(key)
            for key in ("mcp", "command_center", "process_alive")
        ):
            return False
        if simulation.get("third_person_enabled") and not simulation.get(
            "third_person_available"
        ):
            return False
        pose = self.probe.pose()
        if pose is None:
            return False
        pose_timestamp = getattr(pose, "timestamp", reset_started_epoch)
        if isinstance(pose_timestamp, (int, float)) and pose_timestamp < reset_started_epoch:
            return False
        if self.probe.camera_path() is None:
            return False
        capabilities = simulation.get("capabilities") or {}
        sensor_mapping_required = bool(
            capabilities.get("recovery_costmap")
            or capabilities.get("costmap") is not False
        )
        if not sensor_mapping_required:
            return True
        mapping = self.costmap.status()
        lidar_proximity = getattr(self, "lidar_proximity", None)
        lidar_ready = bool(
            lidar_proximity is None
            or lidar_proximity.status().get("available")
        )
        mapping_source = str(mapping.get("source", ""))
        return bool(
            mapping.get("available")
            and (mapping_source == "live" or mapping_source.endswith("_live"))
            and lidar_ready
        )

    def _wait_ready(self, reset_started_epoch: float) -> None:
        deadline = self.clock() + self.readiness_timeout
        while not self._simulation_ready(reset_started_epoch):
            self._check_cancelled()
            simulation = self.simulation.status()
            if (
                not simulation.get("starting")
                and not simulation.get("process_alive")
                and self.clock() + 1.0 < deadline
            ):
                raise RuntimeError("新仿真进程启动失败")
            if self.clock() >= deadline:
                raise RuntimeError("等待新仿真、相机和实时地图就绪超时")
            self._cancel.wait(0.25)

    def _run(self, generation: int) -> None:
        try:
            if self.cancel_long_tasks is not None:
                self.cancel_long_tasks("experiment_reset")
            self.agent.cancel()
            self.navigation.force_stop(announce=True)
            force_simulation_stop = getattr(self.simulation, "force_stop", None)
            if callable(force_simulation_stop):
                force_simulation_stop()
            if self.safety_recovery is not None:
                self.safety_recovery.reset()
            if self.task_recovery is not None:
                self.task_recovery.reset()
            if self.auxiliary_lifecycle is not None:
                # Let cancellation reach the robot-local Agents before DDS is
                # torn down. This prevents an old durable task from replaying
                # into the replacement MuJoCo runtime.
                self._wait_agent_idle()
                self.auxiliary_lifecycle.stop()
            if self.runtime_host is not None:
                self.runtime_host.stop()
            else:
                stop_for_reset = getattr(self.simulation, "stop_for_reset", None)
                if callable(stop_for_reset):
                    stop_for_reset()
                else:
                    self.simulation.stop_owned()
            self._wait_simulation_stopped()
            if self.auxiliary_lifecycle is None:
                self._wait_agent_idle()
            self._check_cancelled()

            self._set_phase("clearing")
            reset_session = getattr(self.agent, "reset_session", None)
            if callable(reset_session):
                reset_session()
            cleared = list(self.costmap.reset())
            self.monitor.reset()
            self.probe.reset()
            self.navigation.reset()
            if self.third_person.clear():
                frame_path = getattr(self.third_person, "path", None)
                if frame_path is not None:
                    cleared.append(str(frame_path))
            if self.lidar_proximity is not None and self.lidar_proximity.clear():
                cleared.append(str(self.lidar_proximity.path))
            if self._clear_spatial_memory():
                cleared.append(str(self.spatial_memory_path))
            with self._lock:
                self._cleared = cleared

            self._check_cancelled()
            self._set_phase("starting")
            reset_started_epoch = time.time()
            if self.runtime_host is not None:
                self.runtime_host.start()
                if self.runtime_host.reconcile().state.value == "FAULT":
                    raise RuntimeError("机器人 RuntimeHost 拒绝启动仿真")
            elif not self.simulation.start_async():
                raise RuntimeError("仿真启动请求被拒绝")
            if self.auxiliary_lifecycle is not None:
                self.auxiliary_lifecycle.start()
            self._wait_ready(reset_started_epoch)
            self._check_cancelled()
            if self.runtime_host is not None:
                self.runtime_host.confirm_restart_stationary()
            # A critical observation from the old process can race the first
            # reset above.  Clear both holds only after fresh pose, lidar, and
            # mapping evidence from the replacement simulation are ready.
            if self.safety_recovery is not None:
                self.safety_recovery.reset()
            if self.task_recovery is not None:
                self.task_recovery.reset()

            with self._lock:
                self._phase = "ready"
                self._active = False
                self._completed_at = utc_now()
                self._last_error = ""
            self.events.append(
                "system",
                "reset",
                "Experiment reset completed",
                (
                    "Fresh Isaac pose and RGB are ready; navigation and lidar safety remain disabled."
                    if self.simulation.status().get("backend") == "isaac-g1"
                    else "Fresh MuJoCo pose, camera streams, current lidar, live costmap, and spatial memory are ready."
                ),
                data={"generation": generation, "cleared": cleared},
            )
        except Exception as error:  # noqa: BLE001 - surface reset failures in the UI
            message = str(error)[:2_000]
            with self._lock:
                self._phase = "failed"
                self._active = False
                self._completed_at = utc_now()
                self._last_error = message
            self.events.append(
                "system",
                "error",
                "Experiment reset failed",
                message,
                level="danger",
                data={"generation": generation},
            )
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def close(self) -> None:
        self._cancel.set()
        self.wait(timeout=10.0)






# 记录被安全恢复打断的用户目标，等待旧任务取消后，有限次地开启新一轮重规划。
# 继续任务通过新 Agent turn 完成，不直接恢复旧规划器的运动命令。
class RecoveryTaskCoordinator:
    """Resume an interrupted user goal only through a fresh Agent turn."""

    def __init__(
        self,
        events: EventStore,
        *,
        agent: Any,
        recovery_status: Callable[[], dict[str, Any]],
        release_recovery_hold: Callable[[], bool],
        max_resumes: int = 2,
        wait_timeout_seconds: float = 75.0,
        poll_interval_seconds: float = 0.10,
    ) -> None:
        self.events = events
        self.agent = agent
        self.recovery_status = recovery_status
        self.release_recovery_hold = release_recovery_hold
        self.max_resumes = max(0, int(max_resumes))
        self.wait_timeout_seconds = max(0.01, float(wait_timeout_seconds))
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self._lock = threading.RLock()
        self._current_instruction: str | None = None
        self._pending_instruction: str | None = None
        self._pending_mode = "terminal"
        self._generation = 0
        self._resume_attempts = 0
        self._resume_in_progress = False
        self._state = "idle"
        self._interrupted_task_id: str | None = None
        self._reason = ""
        self._detail = ""

    def record_user_instruction(self, instruction: str) -> None:
        """Start a new top-level task and invalidate stale recovery callbacks."""

        with self._lock:
            self._current_instruction = instruction.strip() or None
            self._pending_instruction = None
            self._resume_attempts = 0
            self._generation += 1
            self._state = "task_active"
            self._interrupted_task_id = None
            self._reason = ""
            self._detail = ""

    def reject_recovery(self, reason: str, *, task_id: str | None = None, detail: str = "") -> None:
        with self._lock:
            if task_id is not None:
                self._interrupted_task_id = task_id
            self._pending_instruction = None
            self._state = "continuation_unavailable"
            self._reason = reason
            self._detail = detail[:500]
            self._generation += 1
            data = {"task_id": self._interrupted_task_id, "reason": reason, "detail": self._detail}
        self.events.append("safety", "recovery", "Automatic task continuation unavailable",
                           reason, level="warning", data=data)

    def cancel_active_for_recovery(self, *, rejection_reason: str | None = None) -> dict[str, Any]:
        """The execution owner registers intent and pause atomically by task ID."""
        try:
            intent = (self.agent.request_safety_recovery(rejection_reason=rejection_reason)
                      if rejection_reason else self.agent.request_safety_recovery())
        except Exception:
            self.reject_recovery("recovery_registration_failed")
            self.agent.cancel()
            return {}
        with self._lock:
            self._generation += 1
            self._interrupted_task_id = intent.get("task_id")
            self._pending_mode = intent.get("execution_mode", "terminal")
            self._reason = ""
            self._pending_instruction = intent.get("instruction")
            self._state = "waiting_for_progress_save"
        if intent.get("state") == "unavailable":
            self.reject_recovery(intent.get("reason") or "recovery_registration_failed")
        elif not self._interrupted_task_id or not self._pending_instruction:
            self.reject_recovery("task_identity_unavailable")
        else:
            self.events.append(
                "safety", "recovery", "Interrupted task registered for safe replanning",
                "Waiting for the interrupted task to stop and save its progress.",
                level="warning", data={"task_id": self._interrupted_task_id},
            )
        return intent

    @staticmethod
    def _continuation_instruction(instruction: str) -> str:
        return (
            "继续执行被安全恢复中断的原始用户任务：\n"
            f"{instruction}\n\n"
            "机器人刚刚因 critical 风险停车，并沿实时 costmap 验证过的近期轨迹撤离。"
            "现在已确认实际静止、风险为非 critical，且最新第一人称观测与前向在线地图视野良好。"
            "请先使用本轮自动提供的最新观测重新判断并重新规划；不得恢复旧速度、旧路径或旧规划器目标。"
            "若新观测表明任务仍不安全，应停车并明确报告未完成。"
        )

    def resume_once(self) -> bool:
        """Handoff only the saved task, never a stale instruction or another task."""
        with self._lock:
            if self._resume_in_progress or self._pending_instruction is None:
                return False
            if self._resume_attempts >= self.max_resumes:
                self.reject_recovery("resume_limit_reached")
                self._state = "resume_limit_reached"
                return False
            instruction = self._pending_instruction
            task_id = self._interrupted_task_id
            generation = self._generation
            self._resume_in_progress = True
        try:
            if self.recovery_status().get("state") != "recovered_waiting_replan":
                return False
            snapshot = self.agent.recovery_snapshot(task_id)
            with self._lock:
                if generation != self._generation:
                    return False
                if snapshot.get("state") == "unavailable":
                    self.reject_recovery(snapshot.get("reason") or "progress_save_failed")
                    return False
                if snapshot.get("busy"):
                    return False
                if snapshot.get("state") != "saved":
                    self.reject_recovery("progress_not_saved")
                    return False
                options = {"recovery_task_id": task_id}
                if self._pending_mode == "composed":
                    paused = snapshot.get("paused_goal")
                    if not paused or paused.get("instruction") != instruction:
                        self.reject_recovery("confirmed_goal_unavailable")
                        return False
                    options.update(execution_mode="composed", task_key=paused["task_key"])
                    continuation = instruction
                else:
                    continuation = self._continuation_instruction(instruction)
                accepted, message = self.agent.submit_with_handoff(
                    continuation, self.release_recovery_hold, **options,
                )
                if not accepted:
                    self.reject_recovery("handoff_rejected", detail=str(message))
                    return False
                self._resume_attempts += 1
                self._pending_instruction = None
                self._state = "replanning_original_task"
                self._reason = ""
                self.events.append(
                    "safety", "recovery", "Original task resumed with a new plan",
                    "A new Agent turn owns control and starts from fresh first-person observations.",
                    data={"interrupted_task_id": task_id},
                )
            return True
        except Exception:
            with self._lock:
                if generation == self._generation:
                    self.reject_recovery("recovery_state_unavailable")
            return False
        finally:
            with self._lock:
                self._resume_in_progress = False

    def _wait_and_resume(self, generation: int) -> None:
        deadline = time.monotonic() + self.wait_timeout_seconds
        while time.monotonic() < deadline:
            with self._lock:
                if generation != self._generation or self._pending_instruction is None:
                    return
            if self.recovery_status().get("state") != "recovered_waiting_replan":
                return
            if self.resume_once():
                return
            time.sleep(self.poll_interval_seconds)
        with self._lock:
            if generation != self._generation or self._pending_instruction is None:
                return
            self.reject_recovery("agent_cancel_timeout")

    # 安全恢复就绪后，安排等待旧 Agent turn 退出，再尝试重新规划用户任务。
    def recovery_ready(self) -> None:
        """Schedule continuation after the old Agent turn finishes cancelling."""

        with self._lock:
            if self._pending_instruction is None:
                return
            generation = self._generation
            self._state = "waiting_for_agent"
        threading.Thread(
            target=self._wait_and_resume,
            args=(generation,),
            name="recovery-task-replan",
            daemon=True,
        ).start()

    def reset(self) -> None:
        with self._lock:
            self._current_instruction = None
            self._pending_instruction = None
            self._resume_attempts = 0
            self._generation += 1
            self._state = "idle"
            self._interrupted_task_id = None
            self._reason = ""
            self._detail = ""

    def status(self) -> dict[str, Any]:
        # Surface a failed save even when physical retreat cannot finish.
        with self._lock:
            task_id = self._interrupted_task_id if self._pending_instruction else None
            generation = self._generation
        if task_id is not None:
            try:
                snapshot = self.agent.recovery_snapshot(task_id)
            except Exception:
                snapshot = {"state": "unavailable", "reason": "recovery_state_unavailable"}
            with self._lock:
                if generation == self._generation and snapshot.get("state") == "unavailable":
                    self.reject_recovery(snapshot.get("reason") or "progress_save_failed")
        try:
            busy = bool(self.agent.status().get("busy"))
        except Exception:  # noqa: BLE001 - status must remain observable
            busy = True
        with self._lock:
            if (
                not busy
                and self._pending_instruction is None
                and not self._resume_in_progress
                and self._state in {"task_active", "replanning_original_task"}
            ):
                self._current_instruction = None
                self._state = "idle"
            return {
                "state": self._state,
                "pending": self._pending_instruction is not None,
                "resume_attempts": self._resume_attempts,
                "max_resumes": self.max_resumes,
                "task_id": self._interrupted_task_id,
                "reason": self._reason,
                "detail": self._detail,
            }


# 操作面板的应用组装中心：连接观测、模拟器、RuntimeHost、安全控制和 Agent。
# HTTP 处理器调用本类的方法提交任务、查询状态、复位和急停。
class LuxiApplication:
    # 按后端和环境开关组装应用依赖，并连接 RuntimeHost、控制通道与 Agent。
    # 这里只负责初始化和绑定，持续运行的服务由 start() 统一启动。
    def __init__(
        self,
        asset_root: Path,
        port: int,
        start_sim: bool,
        *,
        backend: str | None = None,
        composition_options: dict[str, Any] | None = None,
    ) -> None:
        self.asset_root = asset_root
        self.port = port
        self.events = EventStore()
        environment = dict(os.environ)
        from harness.runtime.configuration import validate_runtime_environment
        validate_runtime_environment(environment)
        self.backend = normalize_ui_backend(
            backend or environment.get("LUXI_UI_BACKEND", "mujoco")
        )
        if self.backend == "isaac-g1":
            # Keep an older MuJoCo dashboard on the workstation from sharing
            # velocity traffic with this Isaac UI. An explicit operator
            # address still takes precedence, but every Isaac subscriber asks
            # for enough UDP receive space to hold multiple fragmented RGB-D
            # messages instead of inheriting Linux's ~208 KiB default.
            isaac_lcm_url = lcm_url_with_receive_buffer(
                environment.get("DIMOS_LCM_URL") or DEFAULT_ISAAC_UI_LCM_URL
            )
            environment["DIMOS_LCM_URL"] = isaac_lcm_url
            environment["LCM_DEFAULT_URL"] = isaac_lcm_url
            os.environ["DIMOS_LCM_URL"] = isaac_lcm_url
            os.environ["LCM_DEFAULT_URL"] = isaac_lcm_url
        blind_mode = env_enabled_from_mapping(environment, "LUXI_BLIND_MODE")
        if blind_mode and self.backend != "mujoco":
            raise RuntimeError("Blind evaluation currently supports only MuJoCo")
        if blind_mode:
            # This check deliberately precedes simulator/agent construction.
            # The model has only sanitized observations and allow-listed tools.
            require_blind_harness(environment)
        self.blind_mode = blind_mode
        blind_run = None
        if blind_mode:
            runtime_root = Path(
                environment.get(
                    "DIMOS_RUNTIME_DIR",
                    str(asset_root / "runtime"),
                )
            ).expanduser().resolve()
            blind_run = open_prepared_blind_run(
                runtime_root,
                environment.get("LUXI_BLIND_RUN_TOKEN", ""),
            )
        blind_runtime_paths = blind_run.runtime_paths() if blind_run is not None else None
        self.isaac_paths = IsaacRuntimePaths.configured(environment)
        self.go2_paths = Go2RuntimePaths.configured(
            environment,
            asset_root=asset_root,
        )
        self.probe = (
            IsaacFileProbe(self.isaac_paths)
            if self.backend == "isaac-g1"
            else Go2FileProbe(self.go2_paths)
            if self.backend == "mujoco-go2"
            else SharedMemoryProbe(
                manifest_path=(
                    blind_runtime_paths.shm_manifest_path
                    if blind_runtime_paths is not None
                    else Path(environment["LUXI_COMPOSED_SHM_MANIFEST"]) if environment.get("LUXI_COMPOSED_SHM_MANIFEST") else None
                )
            )
        )
        third_person_enabled = bool(
            not blind_mode
            and (
                (
                    self.backend == "mujoco"
                    and third_person_enabled_for_environment(environment)
                )
                or (
                    self.backend == "isaac-g1"
                    and env_enabled_from_mapping(
                        environment,
                        "LUXI_ISAAC_OBSERVER_ENABLED",
                        True,
                    )
                )
                or (
                    self.backend == "mujoco-go2"
                    and env_enabled_from_mapping(
                        environment,
                        "LUXI_GO2_OBSERVER_ENABLED",
                        True,
                    )
                )
            )
        )
        if self.backend == "isaac-g1":
            self.third_person = FreshIsaacObserverFrame(
                self.isaac_paths,
                enabled=third_person_enabled,
            )
        elif self.backend == "mujoco-go2":
            self.third_person = FreshJpegFrame(
                self.go2_paths.observer,
                enabled=third_person_enabled,
            )
        else:
            observer_path = Path(
                environment.get(
                    "LUXI_THIRD_PERSON_FRAME",
                    f"/dev/shm/luxi-dimos-third-person-{port}.jpg",
                )
            ).expanduser()
            self.third_person = FreshJpegFrame(
                observer_path,
                enabled=third_person_enabled,
            )
        # Scene geometry is never an observation source.  Proximity comes from
        # the live sensor-built costmap in every mode, including legacy office1.
        self.obstacles: list[RectangleObstacle] = []
        runtime_dir = (
            blind_runtime_paths.run_directory / "agent"
            if blind_runtime_paths is not None
            else asset_root
            / (
                "runtime/luxi-isaac-ui"
                if self.backend == "isaac-g1"
                else "runtime/luxi-go2-ui"
                if self.backend == "mujoco-go2"
                else "runtime/luxi-ui"
            )
        )
        self.session_id = self.events.attach_session_store(
            runtime_dir / "luxi-sessions.sqlite3",
            resume_latest=not self.blind_mode,
            metadata={
                "projection": "harness-ui",
                "backend": self.backend,
                "blind_mode": self.blind_mode,
            },
        )
        self.lidar_proximity = FreshLidarProximity(
            (
                self.isaac_paths.lidar_proximity
                if self.backend == "isaac-g1"
                else self.go2_paths.lidar_proximity
                if self.backend == "mujoco-go2"
                else runtime_dir / "lidar-proximity.json"
            ),
            max_age_seconds=bounded_env_float(
                "LUXI_LIDAR_MAX_AGE", 1.25, 0.6, 3.0
            ),
        )
        self.costmap = CostmapMonitor(
            self.events,
            (
                blind_runtime_paths.costmap_path
                if blind_runtime_paths is not None
                else runtime_dir / "maps/latest-costmap.json.gz"
            ),
            persist_interval_seconds=bounded_env_float(
                "LUXI_COSTMAP_SAVE_INTERVAL", 2.0, 0.25, 60.0
            ),
        )
        self.display_costmap = (
            IsaacLidarCostmap(
                self.events,
                self.isaac_paths,
                resolution=bounded_env_float(
                    "LUXI_ISAAC_COSTMAP_RESOLUTION", 0.10, 0.05, 0.25
                ),
                width=bounded_env_int(
                    "LUXI_ISAAC_COSTMAP_WIDTH", 160, 80, 320
                ),
                height=bounded_env_int(
                    "LUXI_ISAAC_COSTMAP_HEIGHT", 160, 80, 320
                ),
                inflation_radius_m=bounded_env_float(
                    "LUXI_ISAAC_COSTMAP_INFLATION", 0.30, 0.0, 0.60
                ),
                max_range_m=bounded_env_float(
                    "LUXI_ISAAC_COSTMAP_RANGE", 8.0, 2.0, 12.0
                ),
                max_age_seconds=bounded_env_float(
                    "LUXI_LIDAR_MAX_AGE", 1.25, 0.6, 3.0
                ),
            )
            if self.backend == "isaac-g1"
            else Go2LidarCostmap(
                self.go2_paths,
                max_age_seconds=bounded_env_float(
                    "LUXI_LIDAR_MAX_AGE", 1.25, 0.6, 3.0
                ),
            )
            if self.backend == "mujoco-go2"
            else self.costmap
        )
        self.isaac_command_writer = (
            IsaacCommandWriter(self.isaac_paths)
            if self.backend == "isaac-g1"
            else None
        )
        self.go2_command_writer = (
            Go2CommandWriter(self.go2_paths)
            if self.backend == "mujoco-go2"
            else None
        )
        recovery_speed = (
            bounded_env_float(
                "LUXI_ISAAC_RECOVERY_GAIT_SPEED", 0.40, 0.25, 0.40
            )
            if self.backend == "isaac-g1"
            else bounded_env_float(
                "LUXI_RECOVERY_MAX_PLANAR_SPEED", 0.06, 0.02, 0.10
            )
        )
        self.isaac_recovery_channel = (
            IsaacSafetyCommandChannel(
                self.isaac_command_writer,
                max_planar_speed_mps=recovery_speed,
            )
            if self.isaac_command_writer is not None
            else None
        )
        self.navigation_bridge = NavigationVelocityBridge(
            self.events,
            max_planar_speed=bounded_env_float(
                "LUXI_NAV_MAX_PLANAR_SPEED", 0.22, 0.05, 0.5
            ),
            max_yaw_rate=bounded_env_float("LUXI_NAV_MAX_YAW_RATE", 0.55, 0.1, 1.0),
            warning_planar_speed=bounded_env_float(
                "LUXI_NAV_WARNING_PLANAR_SPEED", 0.10, 0.03, 0.2
            ),
            warning_yaw_rate=bounded_env_float(
                "LUXI_NAV_WARNING_YAW_RATE", 0.25, 0.05, 0.5
            ),
            recovery_planar_speed=recovery_speed,
            watchdog_seconds=bounded_env_float(
                "LUXI_NAV_WATCHDOG_SECONDS", 0.65, 0.2, 2.0
            ),
        )
        self.navigation_bridge_enabled = bool(
            self.backend == "mujoco" and env_enabled("LUXI_NAV_BRIDGE", True)
        )
        self.costmap_monitor_enabled = bool(
            self.backend == "mujoco" and env_enabled("LUXI_COSTMAP_MONITOR", True)
        )
        recovery_costmap = (
            self.display_costmap
            if self.backend in {"isaac-g1", "mujoco-go2"}
            else self.costmap
        )
        self.monitor = ObservationMonitor(
            self.probe,
            self.events,
            self.obstacles,
            recovery_costmap.status,
            self.navigation_bridge.status,
            recovery_costmap.payload,
            self.lidar_proximity.payload,
        )
        self.navigation_bridge.set_risk_provider(self.monitor.risk_state)
        self.simulation = SimulationSupervisor(
            self.events,
            asset_root,
            self.third_person,
            backend=self.backend,
            isaac_paths=self.isaac_paths,
            go2_paths=self.go2_paths,
            third_person_enabled=third_person_enabled,
            blind_runtime_paths=blind_runtime_paths,
            lidar_proximity=self.lidar_proximity,
        )
        self.robot_runtime = None
        self.robot_runtime_projection = None
        self.stop_service = None
        self.native_tool_broker = None
        self.operator_motion_gateway = None
        self.native_stop_service = None
        self.native_motion_service = None
        self.native_navigation_service = None
        self.stop_authority_projection = None
        isaac_stop_cutover_enabled = False
        isaac_relative_move_cutover_enabled = False
        isaac_move_distance_cutover_enabled = False
        isaac_turn_around_cutover_enabled = False
        isaac_move_robot_cutover_enabled = False
        isaac_navigate_to_pose_cutover_enabled = False
        isaac_navigate_to_tag_cutover_enabled = False
        isaac_explore_frontiers_cutover_enabled = False
        isaac_object_search_cutover_enabled = False
        isaac_follow_person_cutover_enabled = False
        isaac_approach_person_cutover_enabled = False
        isaac_navigate_with_text_cutover_enabled = False
        if self.backend == "isaac-g1":
            from harness.runtime.capability_policy import (
                configured_cutover_tools,
            )

            isaac_cutover_tools = configured_cutover_tools(
                PROJECT_ROOT,
                self.backend,
                environment.get("LUXI_PHYSICAL_PIPELINE_TOOLS"),
                setting="LUXI_PHYSICAL_PIPELINE_TOOLS",
            )
            isaac_stop_cutover_enabled = "stop_robot" in isaac_cutover_tools
            isaac_relative_move_cutover_enabled = (
                "relative_move" in isaac_cutover_tools
            )
            isaac_move_distance_cutover_enabled = (
                "move_distance" in isaac_cutover_tools
            )
            isaac_turn_around_cutover_enabled = (
                "turn_around" in isaac_cutover_tools
            )
            isaac_move_robot_cutover_enabled = "move_robot" in isaac_cutover_tools
            isaac_navigate_to_pose_cutover_enabled = (
                "navigate_to_pose" in isaac_cutover_tools
            )
            isaac_navigate_to_tag_cutover_enabled = (
                "navigate_to_tag" in isaac_cutover_tools
            )
            isaac_explore_frontiers_cutover_enabled = (
                "explore_frontiers" in isaac_cutover_tools
            )
            isaac_object_search_cutover_enabled = (
                "object_search" in isaac_cutover_tools
            )
            isaac_follow_person_cutover_enabled = (
                "follow_person" in isaac_cutover_tools
            )
            isaac_approach_person_cutover_enabled = (
                "approach_person" in isaac_cutover_tools
            )
            isaac_navigate_with_text_cutover_enabled = (
                "navigate_with_text" in isaac_cutover_tools
            )
        if self.backend in {"mujoco", "isaac-g1"}:
            runtime_adapter = (
                build_isaac_g1_readonly_adapter(
                    paths=self.isaac_paths,
                    map_payload_provider=self.display_costmap.payload,
                    artifact_root=runtime_dir / "artifacts/runtime-maps",
                    start_runtime=lambda: (
                        self.simulation.start_async() if self.start_sim else None
                    ),
                    stop_runtime=self.simulation.stop_owned,
                    status_provider=self.simulation.status,
                )
                if self.backend == "isaac-g1"
                else build_mujoco_g1_readonly_adapter(
                    probe=self.probe,
                    map_path_provider=lambda: (
                        self.costmap.snapshot_path
                        if self.costmap.status().get("available") is True
                        else None
                    ),
                    start_runtime=lambda: (
                        self.simulation.start_async() if self.start_sim else None
                    ),
                    stop_runtime=self.simulation.stop_owned,
                    status_provider=self.simulation.status,
                )
            )
            self.robot_runtime = LuxiRobotRuntimeHost(
                robot_id="g1-01",
                adapter=runtime_adapter,
                events=self.events,
                runtime_revision=(
                    "isaac-g1-navigation-v7"
                    if isaac_navigate_with_text_cutover_enabled
                    else "isaac-g1-navigation-v6"
                    if isaac_approach_person_cutover_enabled
                    else "isaac-g1-navigation-v5"
                    if isaac_follow_person_cutover_enabled
                    else "isaac-g1-navigation-v4"
                    if isaac_object_search_cutover_enabled
                    else "isaac-g1-navigation-v3"
                    if isaac_explore_frontiers_cutover_enabled
                    else "isaac-g1-navigation-v2"
                    if isaac_navigate_to_tag_cutover_enabled
                    else "isaac-g1-navigation-v1"
                    if isaac_navigate_to_pose_cutover_enabled
                    else "isaac-g1-motion-v4"
                    if isaac_move_robot_cutover_enabled
                    else "isaac-g1-motion-v3"
                    if isaac_turn_around_cutover_enabled
                    else "isaac-g1-motion-v2"
                    if isaac_move_distance_cutover_enabled
                    else "isaac-g1-relative-move-v1"
                    if isaac_relative_move_cutover_enabled
                    else "isaac-g1-stop-v1"
                    if isaac_stop_cutover_enabled
                    else f"{self.backend}-readonly-v1"
                ),
            )
            self.robot_runtime_projection = RuntimeStatusProjection(
                self.events,
                robot_id="g1-01",
            )
        elif self.backend == "mujoco-go2":
            runtime_adapter = build_go2_readonly_adapter(
                paths=self.go2_paths,
                robot_id="go2-01",
                start_runtime=lambda: (
                    self.simulation.start_async() if self.start_sim else None
                ),
                stop_runtime=self.simulation.stop_owned,
                status_provider=self.simulation.status,
                owns_lifecycle=True,
            )
            self.robot_runtime = LuxiRobotRuntimeHost(
                robot_id="go2-01",
                adapter=runtime_adapter,
                events=self.events,
                runtime_revision="mujoco-go2-readonly-v1",
            )
            self.robot_runtime_projection = RuntimeStatusProjection(
                self.events,
                robot_id="go2-01",
            )
        if isaac_stop_cutover_enabled and self.robot_runtime is not None:
            from harness.robots.g1.isaac.stop_service import IsaacStopControlChannel, IsaacStopRequestService
            from harness.robots.g1.isaac.motion_service import IsaacMotionControlChannel, IsaacMotionRequestService
            from harness.robots.g1.isaac.navigation_service import IsaacNavigationControlChannel, IsaacNavigationRequestService
            from harness.robots.g1.isaac.tool_broker import IsaacToolBroker
            from harness.robots.g1.isaac.motion_results import _parse_native_result
            from harness.robots.g1.isaac.navigation_results import parse_navigation_result
            from harness.runtime.isaac_stop import bind_isaac_stop_gateway
            from harness.runtime.stop_service import RobotStopService, StopAuthorityProjection
            from harness.runtime.safety_kernel import SafetyPolicy

            stop_gateway = bind_isaac_stop_gateway(self.robot_runtime, bind_port=False)
            self.stop_service = RobotStopService(
                self.events, stop_gateway, backend=self.backend, robot_id=self.robot_runtime.robot_id,
                policy=SafetyPolicy(stop_confirmation_timeout_s=6.0),
            )
            self.stop_authority_projection = StopAuthorityProjection(self.events)
            self.native_stop_service = IsaacStopRequestService(
                self.stop_service, IsaacStopControlChannel(self.isaac_paths.root / "agent-stop"))
            motion_ids = frozenset({"relative_move", "move_distance", "turn_around", "move_robot"})
            native_ids = isaac_cutover_tools - {"stop_robot", "stop_navigation"}

            def parse_native_result(raw, capability_id):
                return (_parse_native_result(raw, capability_id) if capability_id in motion_ids
                        else parse_navigation_result(raw, capability_id))

            self.native_tool_broker = IsaacToolBroker(self.robot_runtime, native_ids, parse_result=parse_native_result)
            self.native_motion_service = IsaacMotionRequestService(
                self.native_tool_broker, IsaacMotionControlChannel(self.isaac_paths.root / "agent-motion"))
            self.native_navigation_service = IsaacNavigationRequestService(
                self.native_tool_broker, IsaacNavigationControlChannel(self.isaac_paths.root / "agent-navigation"))
            self.robot_runtime.bind_emergency_stop(self._request_isaac_stop)
            self.simulation.bind_stop_callback(self._request_isaac_stop)
            if self.isaac_recovery_channel is not None:
                self.isaac_recovery_channel.bind_stop_callback(
                    self._request_isaac_stop
                )
            from harness.runtime.operator_motion import IsaacOperatorMotionGateway


            def operator_safety_observation(_robot_id: str) -> dict[str, Any]:
                observation = stop_gateway.observation.snapshot("g1-01")
                fresh = bool(
                    observation.values.get("pose_available") is True
                    and observation.freshness_s
                    <= self.stop_service.safety.policy.max_observation_age_s
                )
                return {
                    "timestamp_monotonic": time.monotonic() if fresh else 0.0,
                    "risk": self.monitor.risk_state() if fresh else "fault",
                }

            self.operator_motion_gateway = IsaacOperatorMotionGateway(
                runtime_host=self.robot_runtime,
                motion_port=stop_gateway.port,
                safety=self.stop_service.safety,
                safety_observation=operator_safety_observation,
                stop_callback=self._request_isaac_stop,
                events=self.events,
            )
        self.go2_ros_lifecycle = Go2RosComposeLifecycle(
            enabled=bool(
                self.backend == "mujoco-go2"
                and env_enabled_from_mapping(
                    environment, "LUXI_GO2_EXTERNAL_ROS_AGENT", False
                )
            )
        )
        long_task_control_root = (
            blind_runtime_paths.control_directory
            if blind_runtime_paths is not None
            else self.isaac_paths.root
            if self.backend == "isaac-g1"
            else None
        )
        self.long_task_control = LongTaskControlChannel(long_task_control_root)

        def build_long_task_command(
            tool: str,
            arguments: dict[str, Any],
            job_id: str,
        ) -> list[str]:
            payload = {**arguments, "job_id": job_id}
            return [
                str(PROJECT_ROOT / "scripts/dimos.sh"),
                "mcp",
                "call",
                tool,
                "--json-args",
                json.dumps(payload, ensure_ascii=False),
            ]

        self.mcp_jobs = MCPJobManager(
            runtime_dir / "mcp-jobs",
            command_builder=build_long_task_command,
            channel=self.long_task_control,
            events=self.events,
            enabled=get_backend_profile(self.backend).capabilities.object_fetch,
            idle_provider=self._long_task_idle_status,
            stop_callback=lambda: self.navigation_bridge.force_stop(announce=True),
        )
        from harness.runtime.composition import create_agent_runtime_service
        self.agent = create_agent_runtime_service(
            self.events, self.monitor, project_root=PROJECT_ROOT, backend=self.backend,
            long_task_runner=self.mcp_jobs.run_and_wait, runtime_host=self.robot_runtime,
            scene_id_provider=lambda: self.simulation.scene_status().get("selected_id"),
            navigation_stop_callback=(lambda: self.navigation_bridge.force_stop(preserve_recovery=True)) if self.backend == "mujoco" else None,
            safety=self.stop_service.safety if self.stop_service else None,
            native_broker=self.native_tool_broker,
            native_motion_port=stop_gateway.port if self.backend == "isaac-g1" and self.stop_service else None,
            **(composition_options or {}),
        )
        if self.operator_motion_gateway is not None:
            self.operator_motion_gateway.motion_port = self.robot_runtime.adapter.ports.motion
        self.task_recovery = RecoveryTaskCoordinator(
            self.events,
            agent=self.agent,
            recovery_status=lambda: self.safety_recovery.status(),
            release_recovery_hold=lambda: self.safety_recovery.acknowledge_replan(),
            max_resumes=bounded_env_int(
                "LUXI_RECOVERY_MAX_AUTO_RESUMES", 2, 0, 5
            ),
            wait_timeout_seconds=bounded_env_float(
                "LUXI_RECOVERY_AGENT_WAIT_TIMEOUT", 75.0, 5.0, 180.0
            ),
        )
        self.safety_recovery = CriticalRecoveryController(
            self.events,
            observation_provider=self.monitor.recovery_observation,
            costmap_provider=recovery_costmap.payload,
            force_stop=(
                self.isaac_recovery_channel.force_stop
                if self.isaac_recovery_channel is not None
                else self.navigation_bridge.force_stop
            ),
            cancel_agent=self._cancel_for_safety_recovery,
            begin_hold=(
                self.isaac_recovery_channel.begin_hold
                if self.isaac_recovery_channel is not None
                else self.navigation_bridge.begin_safety_recovery
            ),
            publish_recovery=(
                self.isaac_recovery_channel.publish
                if self.isaac_recovery_channel is not None
                else self.navigation_bridge.publish_safety_recovery
            ),
            end_hold=(
                self.isaac_recovery_channel.end_hold
                if self.isaac_recovery_channel is not None
                else self.navigation_bridge.end_safety_recovery
            ),
            on_recovery_ready=self.task_recovery.recovery_ready,
            motion_handoff_ready=lambda: not self.agent.status().get("busy", True),
            enabled=bool(
                get_backend_profile(self.backend).capabilities.critical_recovery
                and
                env_enabled(
                    (
                        "LUXI_ISAAC_CRITICAL_RECOVERY"
                        if self.backend == "isaac-g1"
                        else "LUXI_CRITICAL_RECOVERY"
                    ),
                    True,
                )
            ),
            config=RecoveryConfig(
                recovery_speed_mps=recovery_speed,
                max_breadcrumb_age_seconds=bounded_env_float(
                    "LUXI_RECOVERY_BREADCRUMB_MAX_AGE",
                    120.0 if self.backend == "isaac-g1" else 30.0,
                    30.0,
                    180.0,
                ),
                safe_clearance_m=bounded_env_float(
                    "LUXI_RECOVERY_SAFE_CLEARANCE", 0.55, 0.50, 1.50
                ),
                max_costmap_age_seconds=bounded_env_float(
                    "LUXI_RECOVERY_MAX_COSTMAP_AGE", 1.25, 0.25, 3.0
                ),
                costmap_wait_timeout_seconds=bounded_env_float(
                    "LUXI_RECOVERY_COSTMAP_WAIT_TIMEOUT", 2.0, 0.25, 5.0
                ),
                max_retreat_distance_m=bounded_env_float(
                    "LUXI_RECOVERY_MAX_DISTANCE",
                    1.20,
                    0.30,
                    2.00,
                ),
                max_recovery_seconds=bounded_env_float(
                    "LUXI_RECOVERY_TIMEOUT",
                    15.0 if self.backend == "isaac-g1" else 8.0,
                    2.0,
                    20.0,
                ),
                stop_confirmation_timeout_seconds=bounded_env_float(
                    "LUXI_RECOVERY_STOP_TIMEOUT",
                    6.0 if self.backend == "isaac-g1" else 2.0,
                    1.0,
                    8.0,
                ),
                progress_timeout_seconds=bounded_env_float(
                    "LUXI_RECOVERY_PROGRESS_TIMEOUT",
                    3.0 if self.backend == "isaac-g1" else 1.5,
                    0.5,
                    5.0,
                ),
                warning_approach_guard=self.backend == "isaac-g1",
            ),
        )
        self.monitor.recovery_status = self.safety_recovery.status
        self.reset_controller = ExperimentResetController(
            self.events,
            simulation=self.simulation,
            agent=self.agent,
            navigation=self.navigation_bridge,
            costmap=self.display_costmap,
            monitor=self.monitor,
            safety_recovery=self.safety_recovery,
            task_recovery=self.task_recovery,
            auxiliary_lifecycle=(
                self.go2_ros_lifecycle
                if self.go2_ros_lifecycle.enabled
                else None
            ),
            runtime_host=self.robot_runtime,
            cancel_long_tasks=self.mcp_jobs.cancel_active,
            probe=self.probe,
            third_person=self.third_person,
            lidar_proximity=self.lidar_proximity,
            spatial_memory_path=(
                blind_runtime_paths.memory_directory
                if blind_runtime_paths is not None
                else (
                    runtime_dir / "spatial-memory"
                    if self.backend == "isaac-g1"
                    else asset_root / "upstream/assets/output/memory/spatial_memory"
                )
            ),
            readiness_timeout=bounded_env_float(
                "LUXI_RESET_TIMEOUT", 120.0, 30.0, 300.0
            ),
        )
        self.start_sim = start_sim
        self.manual_control = (
            IsaacManualControl(
                self.events,
                self.operator_motion_gateway,
                simulation_status=self.simulation.status,
                agent_status=self.agent.status,
                recovery_status=self.safety_recovery.status,
                proximity_payload=self.lidar_proximity.payload,
                person_writer=IsaacPersonControlWriter(
                    self.isaac_paths.person_command
                ),
                scene_status=lambda: (
                    self.simulation.selected_scene()
                    if isinstance(self.simulation.selected_scene(), str)
                    else None
                ),
                person_status=lambda: (
                    (read_fresh_state(self.isaac_paths) or {}).get("person_control")
                ),
                stop_callback=self._request_isaac_stop,
                control_owner_status=(
                    self.robot_runtime.status
                    if self.robot_runtime is not None
                    else None
                ),
            )
            if self.operator_motion_gateway is not None
            else Go2PersonManualControl(
                self.events,
                Go2PersonControlWriter(self.go2_paths),
                simulation_status=self.simulation.status,
                person_status=lambda: (
                    (read_fresh_go2_state(self.go2_paths) or {}).get("person_control")
                ),
            )
            if self.backend == "mujoco-go2"
            else None
        )

    def _long_task_idle_status(self) -> dict[str, Any]:
        agent_idle = not bool(getattr(self, "agent", None) and self.agent.status().get("busy"))
        navigation = self.navigation_bridge.status()
        navigation_idle = not bool(
            navigation.get("active") or navigation.get("recovery_active")
        )
        arm_state = EntityControlChannel().state() if self.backend == "mujoco" else None
        arm_idle = bool(
            self.backend != "mujoco"
            or (
                isinstance(arm_state, dict)
                and arm_state.get("available") is True
                and arm_state.get("busy") is False
            )
        )
        return {
            "idle": agent_idle and navigation_idle and arm_idle,
            "agent_idle": agent_idle,
            "navigation_idle": navigation_idle,
            "arm_idle": arm_idle,
        }

    # 通过统一停车执行器请求 Isaac 停车；执行器不可用时返回明确的失败证据。
    def _request_isaac_stop(self, source: str) -> dict[str, Any]:
        executor = getattr(self, "stop_service", None)
        if self.backend != "isaac-g1" or executor is None:
            return {
                "ok": False,
                "completed": False,
                "task_status": "runtime_error",
                "error": "provider stop coordinator is unavailable",
                "safety_evidence": {
                    "stop_command_completed": False,
                    "stationary_confirmed": False,
                },
            }
        return executor.request_stop(source)

    # 安全恢复前取消当前任务，并区分长任务取消与可重新规划的 Agent 任务。
    def _cancel_for_safety_recovery(self) -> bool:
        # Register with the execution owner before slow dashboard projections or diagnostics.
        long_task_cancelled = self.mcp_jobs.cancel_active("safety_recovery")
        intent = self.task_recovery.cancel_active_for_recovery(
            rejection_reason="independent_long_task" if long_task_cancelled else None,
        )
        try:
            monitor = getattr(self, "monitor", None)
            observation = monitor.recovery_observation() if monitor is not None else {}
            metrics = observation.get("metrics", {})
            self._last_safety_interruption = {
                "task_id": intent.get("task_id"), "source": "critical_recovery",
                "risk": metrics.get("risk"),
                "robot_surface_clearance_m": metrics.get("robot_surface_clearance_m"),
                "lidar_sequence": metrics.get("lidar_sequence"),
            }
            if hasattr(self, "events"):
                self.events.append("safety", "interruption", "Task interrupted by proximity protection",
                                   "The current task is being stopped before local recovery.",
                                   data=dict(self._last_safety_interruption))
        except Exception:
            self._last_safety_interruption = None
        return long_task_cancelled

    # 启动地图、观测、安全和机器人运行服务，并按配置启动仿真。
    def start(self) -> None:
        if self.isaac_recovery_channel is not None:
            self.isaac_recovery_channel.reset(clear_stale=True)
        if self.costmap_monitor_enabled:
            try:
                self.costmap.start()
            except Exception as error:  # noqa: BLE001 - keep the operator UI available
                self.events.append(
                    "mapping",
                    "error",
                    "Costmap monitor failed to start",
                    str(error),
                    level="danger",
                )
        if self.display_costmap is not self.costmap:
            try:
                self.display_costmap.start()
            except Exception as error:  # noqa: BLE001 - retain RGB/odom controls
                self.events.append(
                    "mapping",
                    "error",
                    "Isaac display costmap failed to start",
                    str(error),
                    level="danger",
                )
        if self.navigation_bridge_enabled:
            try:
                self.navigation_bridge.start()
            except Exception as error:  # noqa: BLE001 - manual bounded motion still works
                self.events.append(
                    "navigation",
                    "error",
                    "Navigation velocity bridge failed to start",
                    str(error),
                    level="danger",
                )
        self.monitor.start()
        self.safety_recovery.start()
        if self.robot_runtime is not None:
            self.robot_runtime.start()
        elif self.start_sim:
            self.simulation.start_async()
        if self.native_stop_service is not None:
            self.native_stop_service.start()
        if self.native_motion_service is not None:
            self.native_motion_service.start()
        if self.native_navigation_service is not None:
            self.native_navigation_service.start()
        self.events.append(
            "ui",
            "lifecycle",
            "Luxi operator UI ready",
            (
                "World state uses fresh Isaac RGB/odometry and verified explored-area navigation."
                if self.backend == "isaac-g1"
                else "World state is sourced from live sensors and the online costmap."
            ),
        )

    # 取消任务并关闭应用管理的后台服务、运行时和 Agent 资源。
    def close(self) -> None:
        self.reset_controller.close()
        self.task_recovery.reset()
        self.mcp_jobs.cancel_active("ui_shutdown")
        self.agent.cancel()
        self.mcp_jobs.close()
        if self.manual_control is not None:
            self.manual_control.stop()
        if self.operator_motion_gateway is not None:
            self.operator_motion_gateway.close()
        self.safety_recovery.stop()
        self.navigation_bridge.stop()
        self.costmap.stop()
        if self.display_costmap is not self.costmap:
            self.display_costmap.stop()
        self.monitor.stop()
        if self.native_motion_service is not None:
            self.native_motion_service.stop()
        if self.native_navigation_service is not None:
            self.native_navigation_service.stop()
        if self.robot_runtime is not None:
            self.robot_runtime.stop()
        else:
            self.simulation.stop_owned()
        if self.native_stop_service is not None:
            self.native_stop_service.stop()
        close_agent = getattr(self.agent, "close", None)
        if callable(close_agent):
            close_agent()

    # 协调 RuntimeHost 状态并生成面板投影；未接入时显式返回不可用状态。
    def _robot_runtime_status(self) -> dict[str, Any]:
        robot_runtime = getattr(self, "robot_runtime", None)
        projection = getattr(self, "robot_runtime_projection", None)
        if robot_runtime is None or projection is None:
            return {
                "available": False,
                "backend": getattr(self, "backend", None),
                "reason": "runtime_host_cutover_pending",
            }
        robot_runtime.reconcile()
        return dict(projection.snapshot())

    # 汇总当前可用的停车控制入口，供面板展示停车控制权的接入情况。
    def _stop_authority_status(self) -> dict[str, Any]:
        projection = getattr(self, "stop_authority_projection", None)
        if projection is None:
            return {
                "available": False,
                "reason": "provider_stop_cutover_pending",
            }
        snapshot = dict(projection.snapshot())
        snapshot["software_bindings"] = {
            "runtime_host": bool(
                self.robot_runtime is not None
                and getattr(self.robot_runtime, "_emergency_stop", None)
                is not None
            ),
            "simulation_supervisor": bool(
                getattr(self.simulation, "_stop_callback", None) is not None
            ),
            "critical_recovery": bool(
                self.isaac_recovery_channel is not None
                and getattr(
                    self.isaac_recovery_channel,
                    "_stop_callback",
                    None,
                )
                is not None
            ),
            "manual_control": bool(
                self.manual_control is not None
                and getattr(self.manual_control, "stop_callback", None)
                is not None
            ),
            "native_tool_service": self.native_stop_service is not None,
        }
        snapshot["bypass_detected"] = not all(
            snapshot["software_bindings"].values()
        )
        return snapshot

    # 汇总机器人、Agent、地图、安全恢复等状态，作为 /api/state 的返回内容。
    def state(self) -> dict[str, Any]:
        world = self.monitor.snapshot()
        robot_runtime = self._robot_runtime_status()
        simulation = self.simulation.status()
        simulation["lifecycle_authority"] = (
            "robot_runtime_projection"
            if robot_runtime.get("available") is True
            else "simulation_supervisor_compatibility"
        )
        simulation["runtime_state"] = robot_runtime.get("state")
        state = {
            "evaluation": {
                "backend": self.backend,
                "blind_mode": self.blind_mode,
                "information_isolation": (
                    "harness_sanitized" if self.blind_mode else "not_blind"
                ),
            },
            "simulation": simulation,
            "robot_runtime": robot_runtime,
            "stop_authority": self._stop_authority_status(),
            "agent": self.agent.status(),
            "navigation_bridge": self.navigation_bridge.status(),
            "safety_recovery": self.safety_recovery.status(),
            "task_recovery": self.task_recovery.status(),
            "long_task": self.mcp_jobs.snapshot(),
            "costmap": self.display_costmap.status(),
            "manual_control": (
                self.manual_control.status()
                if self.manual_control is not None
                else {
                    "supported": False,
                    "enabled": False,
                    "active": False,
                }
            ),
            "world": world,
            "reset": self.reset_controller.status(),
            "event_cursor": self.events.latest_id,
            "server_time": utc_now(),
        }
        interruption = getattr(self, "_last_safety_interruption", None)
        task = state["agent"].get("task_progress", {})
        if (interruption and interruption.get("task_id")
                and interruption["task_id"] == task.get("task_id")):
            detail = dict(interruption)
            detail["recovery_reason"] = state["safety_recovery"].get("reason")
            detail["route_diagnostic"] = state["safety_recovery"].get("route_diagnostic", {})
            state["agent"]["safety_interruption"] = detail
            result = state["agent"].get("last_task_result", {})
            if result.get("task_status") == "cancelled":
                result["safety_interruption"] = detail
                descriptions = {
                    "no_fresh_verified_retreat_path": "未能验证安全后退路线，保持停车",
                    "recovery_channel_unavailable": "恢复控制通道不可用，保持停车",
                    "waiting_for_task_stop_barrier": "等待上一任务完成停车",
                    "task_stop_barrier_timeout": "等待上一任务停车超时，保持停车",
                    "retracing_verified_path": "正在沿已验证路线后退",
                    "awaiting_replan": "已安全撤离，等待重新规划",
                }
                reason = descriptions.get(detail["recovery_reason"], "请查看安全恢复状态")
                clearance = detail.get("robot_surface_clearance_m")
                distance_text = f"（机器人表面间距约 {clearance:.2f} 米）" if isinstance(clearance, (int, float)) else ""
                state["agent"]["last_response"] = f"任务因障碍过近而停车{distance_text}；{reason}。"
        return state

    # 构造 Agent 可读取的环境上下文，过滤第三人称和展示专用能力。
    def agent_context(self) -> dict[str, Any]:
        world = self.monitor.agent_snapshot()
        simulation = self.simulation.status()
        robot_runtime = self._robot_runtime_status()
        advertised_capabilities = simulation.get("capabilities")
        agent_capabilities = (
            {
                key: value
                for key, value in advertised_capabilities.items()
                if key not in {"third_person", "costmap_display"}
            }
            if isinstance(advertised_capabilities, dict)
            else advertised_capabilities
        )
        return {
            "simulation": {
                key: simulation.get(key)
                for key in (
                    "mcp",
                    "command_center",
                    "process_alive",
                    "owned_by_ui",
                    "starting",
                )
            } | {
                "backend": simulation.get("backend"),
                "capabilities": agent_capabilities,
            },
            "robot_runtime": {
                "available": robot_runtime.get("available"),
                "state": robot_runtime.get("state"),
                "runtime_revision": robot_runtime.get("runtime_revision"),
                "health": {
                    key: (robot_runtime.get("health") or {}).get(key)
                    for key in (
                        "ready",
                        "bridge_ready",
                        "observation_ready",
                        "mcp_ready",
                        "command_center_ready",
                    )
                },
            },
            "world": world,
            "server_time": utc_now(),
        }

    # 提交用户任务前检查运行时就绪、手动控制和安全恢复状态。
    # 通过检查后交给 Agent bridge；需要重新规划时先完成安全控制权交接。
    # HTTP 指令进入 Agent 前先检查运行状态和控制权，再转交 AgentRuntimeService。
    def submit_instruction(
        self,
        instruction: str,
        task_mode: str | None = None,
        execution_mode: str = "terminal",
        task_key: str | None = None,
    ) -> tuple[bool, str]:
        if execution_mode not in ("terminal", "composed"):
            return False, "执行模式必须为 terminal 或 composed"
        mode_options = ({"execution_mode": execution_mode, "task_key": task_key}
                        if execution_mode != "terminal" or task_key is not None else {})
        if task_mode not in {None, "single"}:
            return False, "当前仅支持单机器人任务，双机模式已删除"
        robot_runtime = getattr(self, "robot_runtime", None)
        if robot_runtime is not None:
            runtime_status = self._robot_runtime_status()
            if (
                runtime_status.get("available") is not True
                or runtime_status.get("state") != "READY"
            ):
                return False, "机器人 RuntimeHost 尚未 READY，拒绝启动新的 Agent Turn"
        mcp_jobs = getattr(self, "mcp_jobs", None)
        if mcp_jobs is not None and mcp_jobs.snapshot().get("active"):
            return False, "已有长运动任务在执行；请等待结束或先取消"
        manual_control = getattr(self, "manual_control", None)
        if (
            manual_control is not None
            and manual_control.status().get("enabled")
            and manual_control.status().get("target") == "robot"
        ):
            return False, "请先退出机器人键盘控制模式，再把控制权交给 Agent"
        recovery = self.safety_recovery.status()
        if recovery.get("active"):
            return False, "机器人正在执行安全回撤；完成停车确认前不接受新任务"
        if recovery.get("safety_hold") and not recovery.get(
            "reconsideration_required"
        ):
            return False, "安全回撤缺少可信路径，机器人保持锁定；请急停检查或复位实验"
        if recovery.get("reconsideration_required"):
            submit_with_handoff = getattr(self.agent, "submit_with_handoff", None)
            if not callable(submit_with_handoff):
                return False, "当前 Agent 不支持安全控制权交接；机器人继续保持停车"
            accepted, message = submit_with_handoff(instruction, self.safety_recovery.acknowledge_replan, **mode_options)
        else:
            accepted, message = self.agent.submit(instruction, **mode_options)
        task_recovery = getattr(self, "task_recovery", None)
        if accepted and task_recovery is not None:
            task_recovery.record_user_instruction(instruction)
        return accepted, message

    # 检查 Agent 和机器人状态后，通过 MCP 任务管理器启动指定长任务。
    def start_long_task(self, tool: str, arguments: Any) -> dict[str, Any]:
        if self.reset_controller.status().get("active"):
            raise RuntimeError("实验正在复位")
        if self.agent.status().get("busy"):
            raise RuntimeError("Agent 正在执行指令")
        if self.manual_control is not None and self.manual_control.status().get("enabled"):
            raise RuntimeError("请先退出键盘控制模式")
        recovery = self.safety_recovery.status()
        if recovery.get("active") or recovery.get("safety_hold"):
            raise RuntimeError("安全恢复或安全锁定期间不能启动长任务")
        risk = (self.monitor.snapshot().get("metrics") or {}).get("risk")
        if risk != "clear":
            raise RuntimeError(f"长运动任务只允许在 clear 风险下启动，当前为 {risk}")
        if self.robot_runtime is not None:
            runtime_status = self._robot_runtime_status()
            if (
                runtime_status.get("available") is not True
                or runtime_status.get("state") != "READY"
            ):
                raise RuntimeError("机器人 RuntimeHost 尚未 READY")
        else:
            simulation = self.simulation.status()
            if not all(
                simulation.get(key)
                for key in ("mcp", "command_center", "process_alive")
            ):
                raise RuntimeError("DimOS 仿真与 MCP 尚未就绪")
        return self.mcp_jobs.start(tool, arguments, source="ui")

    # 校验场景选择并协调切换与实验复位，避免运行中直接替换场景状态。
    def select_scene(
        self,
        scene_id: str,
        seed: int,
        *,
        include_person: bool,
    ) -> tuple[bool, str]:
        if self.blind_mode:
            return False, "正式盲测场景已锁定，不能从 UI 切换"
        if self.reset_controller.status()["active"]:
            return False, "实验正在复位，请等待当前流程完成"
        simulation_status = self.simulation.status()
        externally_managed = bool(
            (
                simulation_status.get("mcp")
                or simulation_status.get("command_center")
                or simulation_status.get("process_alive")
            )
            and not simulation_status.get("owned_by_ui")
        )
        if externally_managed:
            return False, "当前 DimOS 不是由此 UI 托管；为避免干扰其他项目，拒绝切换场景"
        manual_control = getattr(self, "manual_control", None)
        if manual_control is not None:
            manual_control.set_enabled(False)
        mcp_jobs = getattr(self, "mcp_jobs", None)
        if mcp_jobs is not None:
            mcp_jobs.cancel_active("scene_switch")

        previous = self.simulation.selected_scene()
        try:
            selected = self.simulation.configure_scene(
                scene_id,
                seed,
                include_person=include_person,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            return False, str(error)

        accepted, message = self.reset_controller.start_async()
        if not accepted:
            self.simulation.restore_scene(previous)
            return False, message
        if getattr(self, "backend", "mujoco") == "isaac-g1":
            assert isinstance(selected, str)
            descriptor = next(
                item
                for item in ISAAC_SCENE_CATALOG
                if item["scene_id"] == selected
            )
            self.events.append(
                "simulation",
                "scene",
                "Isaac scene selected",
                f"{descriptor['label']} · navigation unavailable",
                level="warning",
            )
            return True, f"正在切换到“{descriptor['label']}”并重启 Isaac"
        assert isinstance(selected, OperatorScene)
        descriptor = operator_scene_descriptor(selected.scene_id)
        self.events.append(
            "simulation",
            "scene",
            "Operator scene selected",
            f"{descriptor.label} · seed {selected.seed} · person "
            f"{'enabled' if selected.person is not None else 'disabled'}",
            level="warning",
        )
        return True, f"正在切换到“{descriptor.label}”并清空旧地图与空间记忆"

    # 取消长任务和 Agent，关闭机器人手动控制，并按后端触发停车。
    # 停车命令完成与物理静止确认是不同证据，不能仅凭请求已提交就判定停稳。
    def emergency_stop_async(self) -> None:
        self.mcp_jobs.cancel_active("emergency_stop")
        self.agent.cancel()
        self.navigation_bridge.force_stop(announce=True)
        if self.manual_control is not None:
            self.manual_control.set_enabled(False)
        isaac_writer = getattr(self, "isaac_command_writer", None)
        if isaac_writer is not None:
            result = self._request_isaac_stop("ui_emergency_stop")
            evidence = result.get("safety_evidence", {})
            self.events.append(
                "safety",
                "command" if evidence.get("stop_command_completed") else "error",
                (
                    "Isaac emergency stop verified"
                    if evidence.get("stationary_confirmed")
                    else "Isaac emergency stop incomplete"
                ),
                f"source=ui_emergency_stop · status={result.get('task_status')}",
                level=(
                    "warning"
                    if evidence.get("stop_command_completed")
                    else "danger"
                ),
            )
            return
        go2_writer = getattr(self, "go2_command_writer", None)
        if go2_writer is not None:
            try:
                sequence = go2_writer.request_stop("emergency_stop")
                self.events.append(
                    "safety",
                    "command",
                    "Go2 emergency stop written",
                    f"stop sequence={sequence}",
                    level="warning",
                )
            except OSError as error:
                self.events.append(
                    "safety",
                    "error",
                    "Go2 emergency stop failed",
                    str(error),
                    level="danger",
                )
            return

        def stop_robot() -> None:
            self.events.append(
                "safety", "command", "Emergency stop requested", "Cancelling navigation and zeroing velocity.", level="warning"
            )
            zero_command = [
                    str(PROJECT_ROOT / "scripts/dimos.sh"),
                    "move",
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--yaw",
                    "0",
                    "--duration",
                    "0.2",
                ]
            commands = (
                [zero_command]
                if getattr(self, "backend", "mujoco") == "isaac-g1"
                else [
                    [
                        str(PROJECT_ROOT / "scripts/dimos.sh"),
                        "mcp",
                        "call",
                        "stop_navigation",
                    ],
                    zero_command,
                ]
            )
            outputs: list[str] = []
            for command in commands:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=8,
                        check=False,
                    )
                    outputs.append((completed.stdout + completed.stderr).strip()[-2_000:])
                except (OSError, subprocess.TimeoutExpired) as error:
                    outputs.append(str(error))
            self.events.append(
                "safety",
                "result",
                "Emergency stop completed",
                "\n".join(part for part in outputs if part),
            )

        threading.Thread(target=stop_robot, name="emergency-stop", daemon=True).start()

    # 切换手动控制模式，并把具体控制逻辑交给对应后端的控制器。
    def set_manual_control(self, enabled: Any) -> tuple[bool, str]:
        if self.manual_control is None:
            return False, "当前后端不支持键盘控制模式"
        if not isinstance(enabled, bool):
            return False, "enabled 必须是布尔值"
        if self.reset_controller.status()["active"]:
            return False, "实验正在复位，不能切换键盘控制"
        return self.manual_control.set_enabled(enabled)

    # 将面板的前进和转向输入交给手动控制器校验、执行。
    def command_manual_control(
        self,
        *,
        forward: Any,
        turn: Any,
    ) -> tuple[bool, str]:
        if self.manual_control is None:
            return False, "当前后端不支持键盘控制模式"
        if self.reset_controller.status()["active"]:
            return False, "实验正在复位，键盘命令已拒绝"
        return self.manual_control.command(forward=forward, turn=turn)

    # 切换手动操作目标，例如机器人或场景中的人物。
    def set_manual_control_target(self, target: Any) -> tuple[bool, str]:
        if self.manual_control is None:
            return False, "当前后端不支持键盘控制模式"
        if self.reset_controller.status()["active"]:
            return False, "实验正在复位，不能切换控制对象"
        return self.manual_control.set_target(target)

    # 把人物操作请求交给当前后端的人物控制接口。
    def command_person_control(self, action: Any) -> tuple[bool, str]:
        if self.manual_control is None:
            return False, "当前后端不支持人物控制"
        if self.reset_controller.status()["active"]:
            return False, "实验正在复位，人物命令已拒绝"
        return self.manual_control.person_action(action)


# 持有 LuxiApplication 的多线程 HTTP 服务，每个请求交给 LuxiRequestHandler。
class LuxiHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: LuxiApplication) -> None:
        self.app = app
        super().__init__(address, LuxiRequestHandler)


# 面板 HTTP 接口层：提供静态页面、观测查询和控制请求，业务操作交给应用对象。
class LuxiRequestHandler(BaseHTTPRequestHandler):
    server: LuxiHTTPServer

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            # Camera polling deliberately abandons stale requests while a new
            # frame is loading.  A disconnected localhost browser is not a
            # server error and should not fill the operator terminal.
            return

    def log_message(self, format_string: str, *args: Any) -> None:
        return

    # 序列化 JSON 响应并设置编码、长度和禁用缓存等响应头。
    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    # 读取大小受限的 JSON 对象请求体，格式或长度不合法时返回 None。
    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0 or length > 32_768:
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    # 若请求携带 Origin，要求其主机为本机回环地址且端口与当前服务一致。
    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        return is_loopback_host(parsed.hostname or "") and parsed.port == self.server.server_port

    # 处理页面、状态、事件、任务进度和相机图像等读取请求。
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._static("index.html", "text/html; charset=utf-8")
        elif parsed.path == "/app.js":
            self._static("app.js", "text/javascript; charset=utf-8")
        elif parsed.path == "/styles.css":
            self._static("styles.css", "text/css; charset=utf-8")
        elif parsed.path == "/api/health":
            self._json({"ok": True, "time": utc_now()})
        elif parsed.path == "/api/state":
            self._json(self.server.app.state())
        elif parsed.path == "/api/agent-context":
            self._json(self.server.app.agent_context())
        elif parsed.path == "/api/events":
            query = parse_qs(parsed.query)
            try:
                cursor = max(0, int(query.get("after", ["0"])[0]))
            except ValueError:
                cursor = 0
            events = self.server.app.events.since(cursor)
            self._json(
                {
                    "events": events,
                    "cursor": events[-1]["id"] if events else cursor,
                }
            )
        elif parsed.path.startswith("/api/mcp-jobs/"):
            job_id = parsed.path.removeprefix("/api/mcp-jobs/")
            if not job_id or "/" in job_id:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                self._json(self.server.app.mcp_jobs.status(job_id))
            except KeyError as error:
                self._json(
                    {"ok": False, "error": str(error)},
                    HTTPStatus.NOT_FOUND,
                )
        elif parsed.path == "/api/costmap":
            self._json(self.server.app.display_costmap.payload())
        elif parsed.path == "/api/camera.jpg":
            image = self.server.app.probe.camera_jpeg()
            if image is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "camera unavailable")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(image)
        elif parsed.path == "/api/third-person.jpg":
            image = self.server.app.third_person.jpeg()
            if image is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "third-person camera unavailable")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(image)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    # 提供支持资源的响应头，供客户端探测资源而不下载正文。
    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        static_routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        if parsed.path in static_routes:
            name, content_type = static_routes[parsed.path]
            self._static(name, content_type, include_body=False)
        elif parsed.path in {
            "/api/health",
            "/api/state",
            "/api/agent-context",
            "/api/costmap",
        }:
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    # 处理任务提交、急停、手动控制、复位等操作，并返回受理或拒绝结果。
    # /api/commands → app.submit_instruction() → Agent bridge → active 模式的 loop.run()。
    def do_POST(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._json({"ok": False, "error": "origin not allowed"}, HTTPStatus.FORBIDDEN)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/commands":
            payload = self._read_json()
            if payload is None:
                self._json({"ok": False, "error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
                return
            if self.server.app.reset_controller.status()["active"]:
                self._json(
                    {"ok": False, "error": "实验正在复位，暂不接受运动指令"},
                    HTTPStatus.CONFLICT,
                )
                return
            accepted, message = self.server.app.submit_instruction(
                str(payload.get("instruction", "")),
                execution_mode=str(payload.get("execution_mode", "terminal")),
                task_key=payload.get("task_key"),
                task_mode=(
                    str(payload["task_mode"])
                    if payload.get("task_mode") is not None
                    else None
                ),
            )
            status = HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT
            self._json({"ok": accepted, "message": message}, status)
        elif parsed.path == "/api/composed/pause":
            accepted, message = self.server.app.agent.pause()
            self._json({"ok": accepted, "message": message}, HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT)
        elif parsed.path == "/api/mcp-jobs/start":
            payload = self._read_json()
            if payload is None:
                self._json(
                    {"ok": False, "error": "invalid JSON"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                result = self.server.app.start_long_task(
                    str(payload.get("tool", "")),
                    payload.get("arguments"),
                )
            except ValueError as error:
                self._json(
                    {"ok": False, "error": str(error)},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            except RuntimeError as error:
                self._json(
                    {"ok": False, "error": str(error)},
                    HTTPStatus.CONFLICT,
                )
                return
            self._json(result, HTTPStatus.ACCEPTED)
        elif parsed.path.startswith("/api/mcp-jobs/") and parsed.path.endswith("/cancel"):
            job_id = parsed.path.removeprefix("/api/mcp-jobs/").removesuffix(
                "/cancel"
            )
            if not job_id or "/" in job_id:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                result = self.server.app.mcp_jobs.cancel(
                    job_id,
                    reason="operator_cancelled",
                )
            except KeyError as error:
                self._json(
                    {"ok": False, "error": str(error)},
                    HTTPStatus.NOT_FOUND,
                )
                return
            self._json(result, HTTPStatus.ACCEPTED)
        elif parsed.path == "/api/stop":
            self.server.app.emergency_stop_async()
            self._json({"ok": True, "message": "急停已触发"}, HTTPStatus.ACCEPTED)
        elif parsed.path == "/api/manual-control/mode":
            payload = self._read_json()
            if payload is None:
                self._json(
                    {"ok": False, "error": "invalid JSON"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            accepted, message = self.server.app.set_manual_control(
                payload.get("enabled")
            )
            self._json(
                {"ok": accepted, "message": message},
                HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT,
            )
        elif parsed.path == "/api/manual-control/command":
            payload = self._read_json()
            if payload is None:
                self._json(
                    {"ok": False, "error": "invalid JSON"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            accepted, message = self.server.app.command_manual_control(
                forward=payload.get("forward"),
                turn=payload.get("turn"),
            )
            self._json(
                {"ok": accepted, "message": message},
                HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT,
            )
        elif parsed.path == "/api/manual-control/target":
            payload = self._read_json()
            if payload is None:
                self._json(
                    {"ok": False, "error": "invalid JSON"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            accepted, message = self.server.app.set_manual_control_target(
                payload.get("target")
            )
            self._json(
                {"ok": accepted, "message": message},
                HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT,
            )
        elif parsed.path == "/api/person-control/action":
            payload = self._read_json()
            if payload is None:
                self._json(
                    {"ok": False, "error": "invalid JSON"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            accepted, message = self.server.app.command_person_control(
                payload.get("action")
            )
            self._json(
                {"ok": accepted, "message": message},
                HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT,
            )
        elif parsed.path == "/api/reset":
            if self.server.app.manual_control is not None:
                self.server.app.manual_control.set_enabled(False)
            accepted, message = self.server.app.reset_controller.start_async()
            self._json(
                {"ok": accepted, "message": message},
                HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT,
            )
        elif parsed.path == "/api/scene/select":
            payload = self._read_json()
            if payload is None:
                self._json({"ok": False, "error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                seed = int(payload.get("seed", 0))
            except (TypeError, ValueError):
                self._json(
                    {"ok": False, "error": "seed 必须是整数"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            include_person = payload.get("include_person", False)
            if not isinstance(include_person, bool):
                self._json(
                    {"ok": False, "error": "include_person 必须是布尔值"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            accepted, message = self.server.app.select_scene(
                str(payload.get("scene_id", "")),
                seed,
                include_person=include_person,
            )
            self._json(
                {"ok": accepted, "message": message, "error": None if accepted else message},
                HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT,
            )
        elif parsed.path == "/api/simulation/start":
            if self.server.app.reset_controller.status()["active"]:
                self._json(
                    {"ok": False, "error": "实验正在复位"},
                    HTTPStatus.CONFLICT,
                )
                return
            started = self.server.app.simulation.start_async()
            self._json({"ok": started}, HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT)
        elif parsed.path == "/api/costmap/save":
            if self.server.app.reset_controller.status()["active"]:
                self._json(
                    {"ok": False, "error": "实验正在复位，地图尚未就绪"},
                    HTTPStatus.CONFLICT,
                )
                return
            saved, message = self.server.app.costmap.save_now()
            self._json(
                {"ok": saved, "message": message},
                HTTPStatus.OK if saved else HTTPStatus.CONFLICT,
            )
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    # 读取固定静态资源并发送正确的内容类型；HEAD 请求只发送响应头。
    def _static(self, name: str, content_type: str, *, include_body: bool = True) -> None:
        path = STATIC_ROOT / name
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'")
        self.end_headers()
        if include_body:
            self.wfile.write(body)


# 在后台延迟打开本地面板页面，避免阻塞 HTTP 服务启动。
def open_browser_later(url: str) -> None:
    def launch() -> None:
        time.sleep(0.8)
        browser = os.environ.get("LUXI_UI_BROWSER")
        if browser:
            command = [browser, url]
        elif shutil.which("xdg-open"):
            command = ["xdg-open", url]
        elif shutil.which("firefox"):
            command = ["firefox", "--new-tab", url]
        else:
            return
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return

    threading.Thread(target=launch, name="browser-launcher", daemon=True).start()


# 定义启动参数：监听地址、端口、资产目录、仿真后端及自动启动选项。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LuxiAgent local robot operator dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP listen host (loopback only)")
    parser.add_argument("--port", type=int, default=8787, help="HTTP listen port")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--backend",
        choices=sorted(UI_BACKENDS),
        default=os.environ.get("LUXI_UI_BACKEND", "mujoco"),
        help="simulation backend (default: LUXI_UI_BACKEND or mujoco)",
    )
    parser.add_argument("--no-start-sim", action="store_true", help="Do not start DimOS automatically")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the dashboard")
    parser.add_argument("--task", type=Path, action="append", default=[], help="Add a fixed MuJoCo composed task")
    parser.add_argument("--locations", type=Path, help="Trusted name -> [x,y,yaw] references for composed tasks")
    return parser


# 程序入口：校验参数，创建应用和 HTTP 服务，注册退出信号并进入请求循环。
# 退出时关闭 HTTP 服务，再由应用清理任务、监控和它管理的运行资源。
def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not is_loopback_host(args.host):
        print("Refusing a non-loopback listen address; robot controls must stay local.", file=sys.stderr)
        return 2
    if not 1 <= args.port <= 65_535:
        print("Port must be between 1 and 65535.", file=sys.stderr)
        return 2

    from harness.runtime.composition import dashboard_composition
    with dashboard_composition(backend=args.backend, project_root=PROJECT_ROOT,
                               task_paths=args.task, locations_path=args.locations) as options:
        return serve_dashboard(args, options)


def serve_dashboard(args, composition_options) -> int:
    app = LuxiApplication(
        args.asset_root.resolve(),
        args.port,
        not args.no_start_sim,
        backend=args.backend,
        composition_options=composition_options,
    )
    try:
        server = LuxiHTTPServer((args.host, args.port), app)
    except OSError as error:
        print(f"Could not listen on {args.host}:{args.port}: {error}", file=sys.stderr)
        app.close()
        return 1

    stop_once = threading.Event()

    def request_shutdown(signum: int, _frame: Any) -> None:
        if stop_once.is_set():
            return
        stop_once.set()
        app.events.append("ui", "lifecycle", "Dashboard shutdown requested", f"signal={signum}")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    app.start()
    url = f"http://{args.host}:{args.port}/"
    print(f"Luxi operator UI: {url}")
    print("Press Ctrl-C to stop the UI and any simulation instance it started.")
    if not args.no_browser:
        open_browser_later(url)

    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
