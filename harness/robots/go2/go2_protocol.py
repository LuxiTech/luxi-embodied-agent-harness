"""Atomic local file protocol for the standalone Go2 MuJoCo runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping


@dataclass(frozen=True)
class Go2RuntimePaths:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def camera(self) -> Path:
        return self.root / "hikrobot-mv-cu013-a0uc-color.jpg"

    @property
    def observer(self) -> Path:
        return self.root / "operator-third-person.jpg"

    @property
    def command(self) -> Path:
        return self.root / "command.json"

    @property
    def stop(self) -> Path:
        return self.root / "stop.json"

    @property
    def person_control(self) -> Path:
        return self.root / "person-control.json"

    @property
    def lidar_proximity(self) -> Path:
        return self.root / "lidar-proximity.json"

    @property
    def mid360_frame(self) -> Path:
        """Latest raw simulated MID-360 point/IMU frame for the ROS adapter."""

        return self.root / "mid360-frame.npz"

    @property
    def ros_task(self) -> Path:
        return self.root / "ros-task-request.json"

    @property
    def ros_cmd_vel(self) -> Path:
        """Simulation-adapter inbox for one robot-scoped ROS Twist."""

        return self.root / "ros-cmd-vel.json"

    @property
    def ros_navigation_status(self) -> Path:
        """Robot-local planner status projected through the ROS adapter."""

        return self.root / "ros-navigation-status.json"

    @property
    def costmap(self) -> Path:
        return self.root / "mid360-costmap.json"

    @classmethod
    def configured(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        asset_root: Path | None = None,
    ) -> "Go2RuntimePaths":
        values = os.environ if environment is None else environment
        configured = str(values.get("LUXI_GO2_RUNTIME_DIR", "")).strip()
        if configured:
            root = Path(configured).expanduser().resolve()
        else:
            base = asset_root or Path(
                values.get(
                    "DIMOS_ASSET_ROOT",
                    str(Path.home() / "work/Asset/dimos"),
                )
            )
            root = base.expanduser().resolve() / "runtime/luxi-go2-ui"
        return cls(root)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def clear_observations(self) -> None:
        for path in (
            self.state,
            self.camera,
            self.root / "d435i-color.jpg",
            self.observer,
            self.lidar_proximity,
            self.mid360_frame,
            self.costmap,
            self.ros_navigation_status,
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass






def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_fresh_go2_state(
    paths: Go2RuntimePaths,
    *,
    max_age_s: float = 1.5,
) -> dict[str, Any] | None:
    payload = read_json(paths.state)
    if payload is None or payload.get("schema_version") != 1:
        return None
    if payload.get("backend") != "mujoco-go2":
        return None
    try:
        written_at = float(payload["written_at"])
        pose = payload["pose"]
        position = [float(value) for value in pose["position"]]
        quaternion = [float(value) for value in pose["quaternion_wxyz"]]
        command = [float(value) for value in payload["command"]]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        len(position) != 3
        or len(quaternion) != 4
        or len(command) != 6
        or not all(math.isfinite(value) for value in (*position, *quaternion, *command))
        or time.time() - written_at < -0.25
        or time.time() - written_at > max(0.1, max_age_s)
    ):
        return None
    return payload


class Go2CommandWriter:
    def __init__(self, paths: Go2RuntimePaths) -> None:
        self.paths = paths
        self.paths.ensure()

    def submit(self, instruction: str, *, via_ros: bool = False) -> int:
        text = instruction.strip()
        if not text:
            raise ValueError("指令不能为空")
        if len(text) > 4_000:
            raise ValueError("单条指令不能超过 4000 个字符")
        from harness.runtime.task_policy import reject_multi_robot_instruction
        rejection = reject_multi_robot_instruction(text)
        if rejection:
            raise ValueError(rejection)
        state = read_fresh_go2_state(self.paths) or {}
        sequence = time.time_ns()
        atomic_write_json(
            self.paths.ros_task if via_ros else self.paths.command,
            {
                "schema_version": 1,
                "robot_id": "go2-01",
                "sequence": sequence,
                "instruction": text,
                "runtime_boot_epoch": state.get("boot_epoch"),
                "written_at": time.time(),
            },
        )
        return sequence

    def request_stop(self, reason: str = "operator_stop") -> int:
        sequence = time.time_ns()
        atomic_write_json(
            self.paths.stop,
            {
                "schema_version": 1,
                "sequence": sequence,
                "reason": str(reason)[:200],
                "written_at": time.time(),
            },
        )
        return sequence
