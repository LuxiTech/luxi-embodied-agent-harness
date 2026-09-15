"""DimOS connection for the isolated Isaac Sim Unitree G1 runtime."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
from threading import Thread
import time
from typing import Any, Mapping

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.unitree.g1.connection import G1ConnectionBase
from dimos.utils.logging_config import setup_logger

from harness.robots.g1.isaac.isaac_protocol import (
    IsaacCameraFrame,
    IsaacLidarFrame,
    IsaacCommandWriter,
    IsaacRuntimePaths,
    configured_runtime_dir,
    read_fresh_camera_frame,
    read_fresh_lidar_frame,
    read_fresh_state,
)


logger = setup_logger()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LAUNCHER = PROJECT_ROOT / "scripts/isaac_g1.sh"


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def pose_from_isaac_state(state: Mapping[str, Any]) -> PoseStamped:
    pose = state["pose"]
    position = pose["position"]
    w, x, y, z = pose["quaternion_wxyz"]
    return PoseStamped(
        ts=float(state["written_at"]),
        frame_id="world",
        position=Vector3(float(position[0]), float(position[1]), float(position[2])),
        orientation=Quaternion(float(x), float(y), float(z), float(w)),
    )


def camera_info_from_frame(frame: IsaacCameraFrame) -> CameraInfo:
    height, width = frame.rgb.shape[:2]
    intrinsics = frame.intrinsics
    info = CameraInfo.from_intrinsics(
        fx=float(intrinsics[0, 0]),
        fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]),
        cy=float(intrinsics[1, 2]),
        width=int(width),
        height=int(height),
        frame_id="camera_optical",
    )
    info.ts = frame.timestamp
    return info


def depth_image_from_frame(frame: IsaacCameraFrame) -> Image:
    """Build the DimOS depth image paired exactly with the Isaac RGB frame."""

    return Image(
        data=frame.depth,
        format=ImageFormat.DEPTH,
        frame_id="camera_optical",
        ts=frame.timestamp,
    )


def pointcloud_from_lidar_frame(frame: IsaacLidarFrame) -> PointCloud2:
    return PointCloud2.from_numpy(
        frame.points,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
    )


class G1IsaacConfig(ModuleConfig):
    runtime_dir: str = Field(default_factory=lambda: str(configured_runtime_dir()))
    launcher: str = Field(default_factory=lambda: os.getenv("LUXI_ISAAC_LAUNCHER", str(DEFAULT_LAUNCHER)))
    scene: str = Field(default_factory=lambda: os.getenv("LUXI_ISAAC_SCENE", "grid"))
    autostart: bool = Field(
        default_factory=lambda: _environment_bool("LUXI_ISAAC_AUTOSTART", False)
    )
    startup_timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("LUXI_ISAAC_STARTUP_TIMEOUT_S", "600")),
        ge=10.0,
        le=1200.0,
    )
    poll_interval_s: float = Field(default=0.02, ge=0.01, le=0.25)
    sensor_hz: float = Field(
        default_factory=lambda: float(os.getenv("LUXI_ISAAC_SENSOR_HZ", "10")),
        ge=1.0,
        le=30.0,
    )
    sensor_width: int = Field(
        default_factory=lambda: int(os.getenv("LUXI_ISAAC_SENSOR_WIDTH", "640")),
        ge=1,
        le=1920,
    )
    sensor_height: int = Field(
        default_factory=lambda: int(os.getenv("LUXI_ISAAC_SENSOR_HEIGHT", "360")),
        ge=1,
        le=1080,
    )
    lidar_enabled: bool = Field(
        default_factory=lambda: _environment_bool("LUXI_ISAAC_LIDAR_ENABLED", True)
    )
    observer_enabled: bool = Field(
        default_factory=lambda: _environment_bool("LUXI_ISAAC_OBSERVER_ENABLED", True)
    )
    observer_hz: float = Field(
        default_factory=lambda: float(os.getenv("LUXI_ISAAC_OBSERVER_HZ", "1")),
        ge=0.2,
        le=10.0,
    )
    observer_width: int = Field(
        default_factory=lambda: int(os.getenv("LUXI_ISAAC_OBSERVER_WIDTH", "320")),
        ge=1,
        le=960,
    )
    observer_height: int = Field(
        default_factory=lambda: int(os.getenv("LUXI_ISAAC_OBSERVER_HEIGHT", "180")),
        ge=1,
        le=540,
    )


class G1IsaacConnection(G1ConnectionBase):
    camera_info_static: CameraInfo = CameraInfo.from_fov(
        fov_deg=90.0,
        width=640,
        height=360,
        axis="horizontal",
        frame_id="camera_optical",
    )

    """Publish fresh Isaac observations and forward bounded, expiring commands.

    RTX lidar is published only when the filesystem protocol proves that every
    return was resolved through StableIdMap and the exact /World/G1 subtree was
    removed.  Missing, stale, or unverified identity data produces no cloud.
    """

    config: G1IsaacConfig
    cmd_vel: In[Twist]
    lidar: Out[PointCloud2]
    odom: Out[PoseStamped]
    color_image: Out[Image]
    depth_image: Out[Image]
    camera_info: Out[CameraInfo]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._paths = IsaacRuntimePaths(Path(self.config.runtime_dir).expanduser().resolve())
        self._command_writer = IsaacCommandWriter(self._paths)
        self._stop_event = threading.Event()
        self._poll_thread: Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._simulator_log: Any | None = None
        self._last_state_sequence = 0
        self._last_camera_sequence = 0
        self._last_lidar_sequence = 0
        self._stale_reported = False
        self._stopped = False

    @rpc
    def start(self) -> None:
        super().start()
        self._paths.root.mkdir(parents=True, exist_ok=True)
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self.move)))
        try:
            if not self._runtime_is_ready():
                if not self.config.autostart:
                    raise RuntimeError(
                        "Isaac G1 runtime is not ready; start scripts/isaac_g1.sh run "
                        "or set LUXI_ISAAC_AUTOSTART=1"
                    )
                self._launch_runtime()
                self._wait_until_ready()
            logger.info("Isaac G1 bridge is ready at %s", self._paths.root)
            self._poll_thread = Thread(
                target=self._poll_loop,
                name="luxi-isaac-g1-observations",
                daemon=True,
            )
            self._poll_thread.start()
        except BaseException:
            self._stop_owned_runtime()
            raise

    def _runtime_is_ready(self) -> bool:
        return (
            read_fresh_state(self._paths, max_age_s=1.5) is not None
            and read_fresh_camera_frame(self._paths, max_age_s=1.5) is not None
        )

    def _launch_runtime(self) -> None:
        launcher = Path(self.config.launcher).expanduser().resolve()
        if not launcher.is_file() or not os.access(launcher, os.X_OK):
            raise RuntimeError(f"Isaac G1 launcher is unavailable: {launcher}")
        environment = os.environ.copy()
        environment["LUXI_ISAAC_RUNTIME_DIR"] = str(self._paths.root)
        environment["LUXI_ISAAC_SCENE"] = self.config.scene
        self._simulator_log = self._paths.simulator_log.open("ab", buffering=0)
        command = [
            str(launcher),
            "run",
            "--sensor-hz",
            f"{self.config.sensor_hz:g}",
            "--sensor-width",
            str(self.config.sensor_width),
            "--sensor-height",
            str(self.config.sensor_height),
            "--observer-hz",
            f"{self.config.observer_hz:g}",
            "--observer-width",
            str(self.config.observer_width),
            "--observer-height",
            str(self.config.observer_height),
        ]
        if not self.config.lidar_enabled:
            command.append("--disable-lidar")
        if not self.config.observer_enabled:
            command.append("--disable-observer")
        self._process = subprocess.Popen(
            command,
            stdout=self._simulator_log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + float(self.config.startup_timeout_s)
        while time.monotonic() < deadline:
            if self._runtime_is_ready():
                return
            if self._process is not None and self._process.poll() is not None:
                code = self._process.returncode
                raise RuntimeError(
                    f"Isaac G1 runtime exited before readiness (exit code {code}); "
                    f"see {self._paths.simulator_log}"
                )
            time.sleep(0.1)
        raise RuntimeError(
            f"Isaac G1 runtime did not become ready in {self.config.startup_timeout_s:.0f}s; "
            f"see {self._paths.simulator_log}"
        )

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            state = read_fresh_state(self._paths, max_age_s=1.5)
            if state is None:
                if not self._stale_reported:
                    logger.error("Isaac G1 state is stale; issuing fail-closed zero velocity")
                    try:
                        self._command_writer.stop()
                    except OSError:
                        logger.exception("Failed to write Isaac G1 zero velocity")
                    self._stale_reported = True
                self._stop_event.wait(float(self.config.poll_interval_s))
                continue

            self._stale_reported = False
            state_sequence = int(state["sequence"])
            if state_sequence > self._last_state_sequence:
                self._publish_pose(pose_from_isaac_state(state))
                self._last_state_sequence = state_sequence

            camera_sequence = int(state.get("camera_sequence", 0))
            if camera_sequence > self._last_camera_sequence:
                frame = read_fresh_camera_frame(self._paths, max_age_s=1.5)
                if frame is not None and frame.sequence > self._last_camera_sequence:
                    self._publish_camera(frame)
                    self._last_camera_sequence = frame.sequence

            lidar_sequence = int(state.get("lidar_sequence", 0))
            if self.config.lidar_enabled and lidar_sequence > self._last_lidar_sequence:
                lidar_frame = read_fresh_lidar_frame(self._paths, max_age_s=1.5)
                if (
                    lidar_frame is not None
                    and lidar_frame.sequence > self._last_lidar_sequence
                ):
                    self.lidar.publish(pointcloud_from_lidar_frame(lidar_frame))
                    self._last_lidar_sequence = lidar_frame.sequence

            if self._process is not None and self._process.poll() is not None:
                logger.error(
                    "Owned Isaac G1 runtime exited with code %s; observations stopped",
                    self._process.returncode,
                )
                try:
                    self._command_writer.stop()
                except OSError:
                    pass
                return
            self._stop_event.wait(float(self.config.poll_interval_s))

    def _publish_pose(self, msg: PoseStamped) -> None:
        self.odom.publish(msg)
        self.tf.publish(Transform.from_pose("base_link", msg))
        timestamp = msg.ts
        self.tf.publish(
            Transform(
                translation=Vector3(0.12, 0.0, 0.55),
                rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
                child_frame_id="camera_link",
                ts=timestamp,
            ),
            Transform(
                translation=Vector3(0.0, 0.0, 0.0),
                rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
                frame_id="camera_link",
                child_frame_id="camera_optical",
                ts=timestamp,
            ),
            Transform(
                translation=Vector3(0.0, 0.0, 0.0),
                rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                frame_id="map",
                child_frame_id="world",
                ts=timestamp,
            ),
        )

    def _publish_camera(self, frame: IsaacCameraFrame) -> None:
        self.color_image.publish(
            Image.from_numpy(
                frame.rgb,
                format=ImageFormat.RGB,
                frame_id="camera_optical",
                ts=frame.timestamp,
            )
        )
        self.depth_image.publish(depth_image_from_frame(frame))
        self.camera_info.publish(camera_info_from_frame(frame))

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> None:
        self._command_writer.write_velocity(
            (
                float(twist.linear.x),
                float(twist.linear.y),
                float(twist.linear.z),
            ),
            (
                float(twist.angular.x),
                float(twist.angular.y),
                float(twist.angular.z),
            ),
            duration_s=float(duration),
        )

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        logger.warning("Isaac G1 runtime does not implement request topic %s", topic)
        return {"ok": False, "topic": topic, "error": "unsupported_by_isaac_g1_adapter"}

    def _stop_owned_runtime(self) -> None:
        if self._process is not None:
            launcher = Path(self.config.launcher).expanduser().resolve()
            try:
                subprocess.run(
                    [str(launcher), "stop"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30.0,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                logger.exception("Failed to stop the owned Isaac G1 container cleanly")
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2.0)
            self._process = None
        if self._simulator_log is not None:
            self._simulator_log.close()
            self._simulator_log = None

    @rpc
    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        try:
            self._command_writer.stop()
        except OSError:
            logger.exception("Failed to write final Isaac G1 zero velocity")
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._stop_owned_runtime()
        super().stop()
