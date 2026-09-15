"""Small side channel for simulator-only controls not present upstream.

The fixed DimOS shared-memory schema stays untouched.  A request and an
acknowledgement are atomic JSON files under the isolated runtime directory.
Only the active simulator worker applies the reset because it alone owns
physics state.  The MuJoCo and Isaac adapters intentionally share this small
request/acknowledgement contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid


CONTROL_PATH_ENV = "LUXI_SIM_CONTROL_PATH"


def configured_control_path() -> Path:
    value = os.environ.get(CONTROL_PATH_ENV, "").strip()
    if value:
        return Path(value).expanduser().resolve()
    runtime = Path(
        os.environ.get(
            "DIMOS_RUNTIME_DIR",
            str(Path.home() / "work/Asset/dimos/runtime"),
        )
    )
    return (runtime / "luxi-sim-control").resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


class SimulationControlChannel:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or configured_control_path()).expanduser().resolve()
        self.request_path = self.root / "request.json"
        self.ack_path = self.root / "ack.json"

    def request_reset(self) -> str:
        token = uuid.uuid4().hex
        _atomic_json(
            self.request_path,
            {"action": "reset_pose", "token": token, "requested_at": time.time()},
        )
        return token

    def acknowledged_token(self) -> str | None:
        payload = self.acknowledgement()
        token = payload.get("token") if payload else None
        return token if isinstance(token, str) and token else None

    def acknowledgement(self) -> dict[str, Any] | None:
        return self._read(self.ack_path)

    def wait_for_ack(self, token: str, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self.acknowledged_token() == token:
                return True
            time.sleep(0.03)
        return self.acknowledged_token() == token

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if len(raw) > 8_192:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None


class SimulationResetController:
    """Worker-local reset applicator polled from the MuJoCo viewer loop."""

    def __init__(
        self,
        model: Any,
        data: Any,
        channel: SimulationControlChannel | None = None,
        *,
        forward: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.channel = channel or SimulationControlChannel()
        self._initial_qpos = data.qpos.copy()
        self._forward = forward
        self._last_token: str | None = None

    def poll(self) -> bool:
        payload = self.channel._read(self.channel.request_path)
        if payload is None or payload.get("action") != "reset_pose":
            return False
        token = payload.get("token")
        if not isinstance(token, str) or not token or token == self._last_token:
            return False
        self._last_token = token
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        actuator_count = len(self.data.ctrl)
        self.data.ctrl[:] = self._initial_qpos[7 : 7 + actuator_count]
        try:
            from harness.robots.g1.mujoco.g1_idle_stabilizer import reset_active_idle_stabilizers

            reset_active_idle_stabilizers()
        except ImportError:
            pass
        if self._forward is None:
            import mujoco

            mujoco.mj_forward(self.model, self.data)
        else:
            self._forward(self.model, self.data)
        _atomic_json(
            self.channel.ack_path,
            {
                "action": "reset_pose",
                "token": token,
                "applied_at": time.time(),
                "position": [float(value) for value in self._initial_qpos[:3]],
                "quaternion_wxyz": [
                    float(value) for value in self._initial_qpos[3:7]
                ],
            },
        )
        return True
