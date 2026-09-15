"""Process-safe cancellation edge for terminal MuJoCo navigation Skills."""

from __future__ import annotations

import os
from pathlib import Path
import time
import uuid

from harness.robots.g1.isaac.isaac_protocol import atomic_write_json, read_json
from harness.control.simulation_control import configured_control_path


TERMINAL_CANCEL_PATH_ENV = "LUXI_TERMINAL_CANCEL_PATH"


def configured_terminal_cancel_path() -> Path:
    configured = os.environ.get(TERMINAL_CANCEL_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return configured_control_path() / "terminal-navigation-cancel.json"


class TerminalCancellationChannel:
    """A monotonic edge, not persistent cancelled state.

    A Skill snapshots the current token before starting.  Only a later token
    cancels that invocation, so stale files cannot cancel work after restart.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or configured_terminal_cancel_path()

    def revision(self) -> str | None:
        payload = read_json(self.path)
        token = payload.get("token") if payload else None
        return token if isinstance(token, str) and token else None

    def signal(self, reason: str) -> str:
        token = uuid.uuid4().hex
        atomic_write_json(
            self.path,
            {
                "schema_version": 1,
                "token": token,
                "reason": str(reason)[:200],
                "requested_at": time.time(),
            },
        )
        return token

    def changed_since(self, revision: str | None) -> bool:
        current = self.revision()
        return current is not None and current != revision
