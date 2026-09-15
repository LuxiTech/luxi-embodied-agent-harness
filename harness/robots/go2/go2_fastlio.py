"""Actual DimOS FAST-LIO2 process adapter for simulated Livox MID-360 data."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import deque
import threading
import select
import time
import os
import shutil
import struct
import subprocess
from typing import Any
import uuid

import numpy as np

from harness.robots.go2.go2_navigation import SlamPose, wrap_angle


FASTLIO2_IMAGE = "luxi-fastlio2-sim:a32c9f5"
_REQUEST = struct.Struct("<IIdII")
_IMU = struct.Struct("<7d")
_POINT = struct.Struct("<3fI")
_RESPONSE = struct.Struct("<IIII7d")
_REGISTERED_POINT = struct.Struct("<4f")
_REQUEST_MAGIC = 0x324F494C
_RESPONSE_MAGIC = 0x32534F50


@dataclass(frozen=True)
class FastLioResult:
    ready: bool
    pose: SlamPose | None
    registered_points: np.ndarray[Any, np.dtype[np.float32]]


class FastLio2Process:
    """One isolated FAST-LIO2 estimator, transported over binary stdio."""

    def __init__(self, *, robot_id: str, anchor: SlamPose) -> None:
        self.robot_id = robot_id
        self.anchor = anchor
        local_requested = os.environ.get("LUXI_FASTLIO2_MODE", "").strip().lower() in {
            "local",
            "executable",
        }
        configured_executable = os.environ.get(
            "LUXI_FASTLIO2_EXECUTABLE", "luxi_fastlio2_sim"
        )
        local_executable = shutil.which(configured_executable)
        if local_requested:
            if local_executable is None:
                raise RuntimeError(
                    f"FAST-LIO2 executable {configured_executable!r} is missing"
                )
            self.container_name = None
            self.transport = "local_pinned_executable_stdio"
            command = (local_executable,)
        else:
            docker = shutil.which("docker")
            if docker is None:
                raise RuntimeError("MID-360 + FAST-LIO2 requires Docker")
            image = os.environ.get("LUXI_FASTLIO2_IMAGE", FASTLIO2_IMAGE)
            inspect = subprocess.run(
                (docker, "image", "inspect", image),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if inspect.returncode:
                raise RuntimeError(
                    f"FAST-LIO2 image {image!r} is missing; run scripts/build_go2_fastlio2.sh"
                )
            self.container_name = f"luxi-fastlio2-{robot_id}-{uuid.uuid4().hex[:8]}"
            self.transport = "simulated_livox_custommsg_docker_stdio"
            command = (
                docker,
                "run",
                "--rm",
                "--name",
                self.container_name,
                "-i",
                image,
            )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._diagnostics = deque(maxlen=8)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, name="fastlio-stderr", daemon=True)
        self._stderr_thread.start()
        self.frames = 0
        self.ready_frames = 0
        self._last_pose = None
        self.last_error: str | None = None

    def _drain_stderr(self):
        # Native FAST-LIO logs continuously. An unread stderr pipe fills and
        # blocks its binary response, freezing the simulator sensor loop.
        while True:
            chunk = self._process.stderr.read(1024)
            if not chunk:
                return
            self._diagnostics.append(chunk.decode("utf-8", errors="replace"))

    @staticmethod
    def _read_exact(stream: Any, size: int) -> bytes:
        data = bytearray()
        deadline = time.monotonic() + 2.0
        while len(data) < size:
            if not select.select([stream], [], [], max(0.0, deadline-time.monotonic()))[0]:
                raise TimeoutError("FAST-LIO2 response timed out; motion must remain stopped")
            chunk = stream.read(size - len(data))
            if not chunk:
                raise EOFError("FAST-LIO2 process closed its output")
            data.extend(chunk)
        return bytes(data)

    def observe(
        self,
        points: np.ndarray[Any, Any],
        *,
        scan_start_s: float,
        gyro_xyz: tuple[float, float, float],
        acceleration_xyz: tuple[float, float, float],
        imu_samples: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ] | None = None,
        scan_period_s: float = 0.10,
    ) -> FastLioResult:
        if self._process.poll() is not None:
            error = "".join(self._diagnostics)[-1000:]
            raise RuntimeError(f"FAST-LIO2 process exited ({self._process.returncode}): {error}")
        cloud = np.asarray(points, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] < 3:
            raise ValueError("MID-360 point cloud must have shape (N, >=3)")
        # 50 Hz IMU samples cover the whole 10 Hz non-repetitive scan and one
        # sample beyond its final point, which FAST-LIO uses to close a package.
        samples = imu_samples or [(gyro_xyz, acceleration_xyz)] * 6
        if len(samples) < 2:
            samples = [*samples, (gyro_xyz, acceleration_xyz)]
        imu_times = np.linspace(scan_start_s, scan_start_s + scan_period_s, len(samples))
        # The simulated non-repetitive pattern is returned in acquisition
        # order. Preserve its scan interval so FAST-LIO can perform the same
        # IMU deskew stage used by the physical MID-360.
        offsets = np.linspace(
            0, int(scan_period_s * 0.9e9), len(cloud), dtype=np.uint32
        )
        payload = bytearray(
            _REQUEST.pack(_REQUEST_MAGIC, 1, scan_start_s, len(imu_times), len(cloud))
        )
        for timestamp, (sample_gyro, sample_acceleration) in zip(
            imu_times, samples, strict=True
        ):
            payload.extend(
                _IMU.pack(float(timestamp), *sample_gyro, *sample_acceleration)
            )
        for point, offset in zip(cloud, offsets, strict=True):
            payload.extend(_POINT.pack(float(point[0]), float(point[1]), float(point[2]), int(offset)))
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(payload)
        self._process.stdin.flush()
        header = _RESPONSE.unpack(self._read_exact(self._process.stdout, _RESPONSE.size))
        magic, version, ready, point_count, *pose_values = header
        if magic != _RESPONSE_MAGIC or version != 1 or point_count > 200000:
            raise RuntimeError("FAST-LIO2 returned an invalid response")
        raw = self._read_exact(self._process.stdout, point_count * _REGISTERED_POINT.size)
        registered = np.frombuffer(raw, dtype="<f4").reshape((-1, 4)).copy()
        self.frames += 1
        if not ready:
            return FastLioResult(False, None, registered)
        self.ready_frames += 1
        x, y, _z, qx, qy, qz, qw = pose_values
        local_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cosine, sine = math.cos(self.anchor.yaw), math.sin(self.anchor.yaw)
        world_pose = SlamPose(
            self.anchor.x + cosine * x - sine * y,
            self.anchor.y + sine * x + cosine * y,
            wrap_angle(self.anchor.yaw + local_yaw),
        )
        previous = self._last_pose
        if not all(math.isfinite(v) for v in (world_pose.x, world_pose.y, world_pose.yaw)) or (
            previous is not None and math.dist((previous.x, previous.y), (world_pose.x, world_pose.y)) > 3.0 * scan_period_s
        ):
            self.last_error = "FAST-LIO2 pose jump exceeds bounded Go2 motion; navigation evidence rejected"
            raise RuntimeError(self.last_error)
        self._last_pose = world_pose
        return FastLioResult(True, world_pose, registered)

    def status(self) -> dict[str, Any]:
        return {
            "backend": "dimos_fastlio2_native",
            "transport": self.transport,
            "core_commit": "a32c9f599940a94595aa72868e2e4ab436a44b75",
            "frames": self.frames,
            "ready_frames": self.ready_frames,
            "ready": self.ready_frames > 0,
            "process_alive": self._process.poll() is None,
            "container": self.container_name,
            "anchor": {"x": self.anchor.x, "y": self.anchor.y, "yaw": self.anchor.yaw},
            "last_error": self.last_error,
        }

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=3.0)
