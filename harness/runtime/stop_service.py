"""Local stop service; independent of model, task admission and event availability."""
from __future__ import annotations
import re
from copy import deepcopy
import threading
import uuid
from typing import Any, Mapping
from .contracts import ToolResult
from .safety_kernel import LuxiSafetyKernel

class RobotStopService:
    def __init__(self, events, gateway, *, backend, robot_id, policy=None):
        self.events = events.session_store
        self.session_id = events.session_id
        self.gateway = gateway
        self.backend = backend
        self.robot_id = robot_id
        self.safety = LuxiSafetyKernel(gateway, policy=policy)

    def execute(self, provider, arguments, turn_id, task_id, step_id, tool_call_id,
                cancelled=False, stop_action="stop_robot"):
        if arguments:
            return {"ok": False, "completed": False, "task_status": "invalid_input",
                    "error": "stop_robot accepts no arguments"}
        source = self._stop_source(provider)
        ids = dict(turn_id=turn_id, task_id=task_id, step_id=step_id, tool_call_id=tool_call_id)
        self._emit_provenance("safety/stop_requested", source, **ids, payload={"action": stop_action})
        evidence = self.safety.stop(self.robot_id, source)
        details = {"stop_command_completed": evidence.stop_command_completed,
                   "stationary_confirmed": evidence.stationary_confirmed,
                   "stop_command_completed_at": evidence.stop_command_completed_at,
                   "stationary_confirmed_at": evidence.stationary_confirmed_at,
                   "event_store_independent_stop": True}
        result = ToolResult("stopped" if evidence.stationary_confirmed else "stop_unconfirmed",
                            evidence.stop_command_completed, False, evidence=details)
        value = self._legacy_result(result)
        value['stop_source'] = source
        self._emit_provenance("safety/stop_outcome", source, **ids, payload={"action": stop_action, **value})
        return value

    def request_stop(
        self,
        source: str,
        *,
        action: str = "stop_robot",
        turn_id: str | None = None,
        task_id: str | None = None,
        step_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit a software stop source through the same physical barrier."""

        if action not in {"stop_robot", "stop_navigation"}:
            raise ValueError(f"unsupported stop action: {action}")
        token = uuid.uuid4().hex
        normalized = self._stop_source(source)
        result = self.execute(
            normalized,
            {},
            turn_id or f"stop-turn-{token}",
            task_id or f"stop-task-{token}",
            step_id or f"stop-step-{token}",
            tool_call_id or f"stop-call-{token}",
            False,
            action,
        )
        result["stop_action"] = action
        return result

    @staticmethod
    def _stop_source(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9_.:-]+", "-", str(value).strip().lower())
        normalized = normalized.strip("-.:_")[:80]
        return normalized or "unknown"

    def _emit_provenance(
        self,
        event_type: str,
        stop_source: str,
        *,
        turn_id: str,
        task_id: str,
        step_id: str,
        tool_call_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        try:
            self.events.emit(
                event_type,
                session_id=self.session_id,
                turn_id=turn_id,
                task_id=task_id,
                step_id=step_id,
                tool_call_id=tool_call_id,
                source="provider-stop-coordinator",
                payload={"stop_source": stop_source, **dict(payload)},
                idempotency_key=f"{tool_call_id}:{event_type}:provenance",
            )
        except Exception:
            # Provenance is best-effort; local emergency zero is independent.
            pass

    @staticmethod
    def _legacy_result(result: ToolResult) -> dict[str, Any]:
        value = dict(result.raw or result.payload)
        value.update(
            {
                "ok": result.tool_ok,
                "completed": result.completed,
                "task_status": result.status,
                "safety_evidence": dict(result.evidence),
            }
        )
        if result.error:
            value["error"] = result.error
        return value


class StopAuthorityProjection:
    """Rebuild the latest stop provenance from the durable event stream."""

    def __init__(self, events: Any) -> None:
        self.events = events
        self._after_sequence = 0
        self._latest: Any | None = None
        self._lock = threading.Lock()
        self._store = None
        self._session_id = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            try:
                return self._snapshot()
            except Exception as exc:
                return {"available": False, "reason": "session_store_unavailable", "error": str(exc)[:500]}

    def _snapshot(self) -> dict[str, Any]:
        store = self.events.session_store
        session_id = self.events.session_id
        if store is None or session_id is None:
            return {"available": False, "reason": "session_store_unavailable"}
        if store is not self._store or session_id != self._session_id:
            self._store, self._session_id = store, session_id
            self._after_sequence = 0
            self._latest = None
        while True:
            batch = store.events(
                session_id,
                after_sequence=self._after_sequence,
                limit=1_000,
            )
            if not batch:
                break
            for event in batch:
                if event.event_type in {
                    "safety/stop_requested",
                    "safety/stop_outcome",
                }:
                    self._latest = event
            self._after_sequence = batch[-1].sequence
            if len(batch) < 1_000:
                break
        latest = self._latest
        return {
            "available": True,
            "owner": "RobotStopService",
            "safety_kernel": "LuxiSafetyKernel",
            "motion_port": "IsaacG1StopOnlyMotionPort",
            "latest": (
                {
                    "event_type": latest.event_type,
                    "sequence": latest.sequence,
                    "tool_call_id": latest.tool_call_id,
                    **deepcopy(dict(latest.payload)),
                }
                if latest is not None
                else None
            ),
        }
