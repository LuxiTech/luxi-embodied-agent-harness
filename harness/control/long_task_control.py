"""Cross-process lifecycle channel for one long robot task.

The operator UI and the DimOS skill worker are separate processes.  Killing an
MCP client does not cancel a skill that is already executing in the worker, so
long tasks use one atomic JSON control record for progress and cooperative
cancellation.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
import uuid

from harness.control.simulation_control import _atomic_json, configured_control_path


MAX_CONTROL_BYTES = 32_768
ACTIVE_STATES = frozenset({"starting", "running", "cancelling", "settling"})
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "interrupted"})


class LongTaskControlChannel:
    """Atomic single-task record shared by the UI and the skill worker."""

    def __init__(self, root: Path | None = None) -> None:
        control_root = (root or configured_control_path()).expanduser().resolve()
        self.path = control_root / "long-task.json"

    def read(self) -> dict[str, Any] | None:
        try:
            raw = self.path.read_bytes()
        except OSError:
            return None
        if len(raw) > MAX_CONTROL_BYTES:
            return None
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        job_id = payload.get("job_id")
        try:
            payload["job_id"] = str(uuid.UUID(str(job_id)))
        except (ValueError, TypeError, AttributeError):
            return None
        return payload

    def begin(
        self,
        job_id: str,
        *,
        tool: str,
        owner: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = str(uuid.UUID(str(job_id)))
        existing = self.read()
        if (
            existing is not None
            and existing.get("job_id") == normalized
            and existing.get("state") == "cancelling"
        ):
            return existing
        now = time.time()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "job_id": normalized,
            "tool": str(tool),
            "owner": str(owner),
            "state": "running",
            "stage": "starting",
            "created_at": (
                existing.get("created_at")
                if existing is not None and existing.get("job_id") == normalized
                else now
            ),
            "updated_at": now,
        }
        if arguments is not None:
            payload["arguments"] = dict(arguments)
        _atomic_json(self.path, payload)
        return payload

    def update(
        self,
        job_id: str,
        *,
        state: str | None = None,
        stage: str | None = None,
        result: dict[str, Any] | None = None,
        cancel_reason: str | None = None,
    ) -> bool:
        normalized = str(uuid.UUID(str(job_id)))
        payload = self.read()
        if payload is None or payload.get("job_id") != normalized:
            return False
        # Once cancellation is requested, the worker may add a result but may
        # not accidentally promote the task back to running.
        if payload.get("state") == "cancelling" and state not in {
            None,
            "cancelling",
            "cancelled",
        }:
            return False
        if state is not None:
            payload["state"] = str(state)
        if stage is not None:
            payload["stage"] = str(stage)
        if result is not None:
            payload["last_structured_result"] = dict(result)
        if cancel_reason is not None:
            payload["cancel_reason"] = str(cancel_reason)[:500]
        payload["updated_at"] = time.time()
        if payload.get("state") in TERMINAL_STATES:
            payload["completed_at"] = time.time()
        _atomic_json(self.path, payload)
        return True

    def cancel(self, job_id: str, reason: str) -> bool:
        normalized = str(uuid.UUID(str(job_id)))
        payload = self.read()
        if payload is None or payload.get("job_id") != normalized:
            return False
        if payload.get("state") in TERMINAL_STATES:
            return False
        payload.update(
            state="cancelling",
            stage="stopping",
            cancel_reason=str(reason)[:500],
            cancel_requested_at=time.time(),
            updated_at=time.time(),
        )
        _atomic_json(self.path, payload)
        return True

    def cancellation(self, job_id: str | None) -> str | None:
        if not job_id:
            return None
        try:
            normalized = str(uuid.UUID(str(job_id)))
        except (ValueError, TypeError, AttributeError):
            return "invalid_job_id"
        payload = self.read()
        if (
            payload is not None
            and payload.get("job_id") == normalized
            and payload.get("state") == "cancelling"
        ):
            return str(payload.get("cancel_reason") or "cancel_requested")
        return None
