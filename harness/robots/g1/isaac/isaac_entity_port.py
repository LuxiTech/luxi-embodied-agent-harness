"""Host-side EntityManipulationPort for the physics-owning Isaac runtime."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Sequence
import uuid

from harness.robots.entity_port import (
    EntityOperationResult,
    entity_result,
    normalize_entity_id,
)
from harness.robots.g1.isaac.isaac_protocol import (
    BACKEND_NAME,
    IsaacRuntimePaths,
    atomic_write_json,
    configured_runtime_dir,
    read_json,
)


class IsaacEntityControlChannel:
    """Bounded atomic request channel; Isaac remains the only physics owner."""

    backend = BACKEND_NAME

    def __init__(self, root: Path | None = None) -> None:
        runtime = (root or configured_runtime_dir()).expanduser().resolve()
        self.paths = IsaacRuntimePaths(runtime)

    def request(
        self,
        action: str,
        arguments: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        token = uuid.uuid4().hex
        atomic_write_json(
            self.paths.entity_request,
            {
                "schema_version": 1,
                "backend": BACKEND_NAME,
                "token": token,
                "action": action,
                "arguments": arguments,
                "requested_at": time.time(),
            },
        )
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            response = read_json(self.paths.entity_ack)
            if response and response.get("token") == token:
                return response
            time.sleep(0.02)
        response = read_json(self.paths.entity_ack)
        if response and response.get("token") == token:
            return response
        return {
            "ok": False,
            "token": token,
            "action": action,
            "error": "entity control timed out",
        }

    def _operation(
        self,
        action: str,
        entity_id: str,
        arguments: dict[str, Any],
        *,
        success_status: str,
        timeout: float,
    ) -> EntityOperationResult:
        canonical = normalize_entity_id(entity_id)
        response = self.request(
            action,
            {"entity_id": canonical, **arguments},
            timeout=timeout,
        )
        operation_ok = bool(response.get("ok"))
        error = response.get("error")
        timed_out = error == "entity control timed out"
        result = entity_result(
            backend=self.backend,
            entity_id=canonical,
            task_status=(
                success_status
                if operation_ok
                else "entity_backend_timeout"
                if timed_out
                else "contact_required"
                if error == "fresh PhysX hand/entity contact is required"
                else "entity_operation_failed"
            ),
            operation_ok=operation_ok,
            completed=operation_ok,
            tool_ok=not timed_out,
            error=str(error) if error else None,
            details=dict(response),
        )
        applied_at = response.get("applied_at")
        if (
            isinstance(applied_at, (int, float))
            and not isinstance(applied_at, bool)
            and math.isfinite(float(applied_at))
        ):
            result["evidence_timestamp"] = float(applied_at)
        for field in ("hand", "pose", "contact", "attached"):
            if field in response:
                result[field] = response[field]
        return result

    def entity_state(self, entity_id: str, *, timeout: float = 2.0) -> EntityOperationResult:
        return self._operation(
            "status", entity_id, {}, success_status="entity_state_available", timeout=timeout
        )

    def contact_state(
        self, entity_id: str, hand: str, *, timeout: float = 2.0
    ) -> EntityOperationResult:
        return self._operation(
            "contact",
            entity_id,
            {"hand": hand},
            success_status="contact_observed",
            timeout=timeout,
        )

    def approach(
        self, entity_id: str, hand: str, *, timeout: float = 8.0
    ) -> EntityOperationResult:
        return self._operation(
            "approach",
            entity_id,
            {"hand": hand},
            success_status="approach_completed",
            timeout=timeout,
        )

    def grasp(
        self,
        entity_id: str,
        hand: str,
        *,
        evidence_timestamp: float | None = None,
        timeout: float = 8.0,
    ) -> EntityOperationResult:
        arguments: dict[str, Any] = {"hand": hand}
        if evidence_timestamp is not None:
            arguments["evidence_timestamp"] = float(evidence_timestamp)
        return self._operation(
            "grasp",
            entity_id,
            arguments,
            success_status="grasp_completed",
            timeout=timeout,
        )

    def carry(
        self,
        entity_id: str,
        hand: str,
        target_pose: Sequence[float] | None = None,
        *,
        timeout: float = 8.0,
    ) -> EntityOperationResult:
        arguments: dict[str, Any] = {"hand": hand}
        if target_pose is not None:
            arguments["target_pose"] = list(target_pose)
        return self._operation(
            "carry",
            entity_id,
            arguments,
            success_status="carry_pose_completed",
            timeout=timeout,
        )

    def place(
        self,
        entity_id: str,
        hand: str,
        target_position: Sequence[float],
        *,
        timeout: float = 8.0,
    ) -> EntityOperationResult:
        return self._operation(
            "place",
            entity_id,
            {"hand": hand, "target_position": list(target_position)},
            success_status="place_completed",
            timeout=timeout,
        )

    def release(
        self, entity_id: str, hand: str, *, timeout: float = 2.0
    ) -> EntityOperationResult:
        return self._operation(
            "release",
            entity_id,
            {"hand": hand},
            success_status="release_completed",
            timeout=timeout,
        )

    def reset_entity(
        self, entity_id: str, *, timeout: float = 2.0
    ) -> EntityOperationResult:
        return self._operation(
            "reset", entity_id, {}, success_status="entity_reset", timeout=timeout
        )
