"""Correlated AgentOS physical-Skill handoff to the unified runtime owner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping
import uuid

from harness.robots.g1.isaac.isaac_protocol import (
    BACKEND_NAME,
    SCHEMA_VERSION,
    IsaacRuntimePaths,
    atomic_write_json,
    read_json,
)


AGENT_MOTION_CONTROL_PATH_ENV = "LUXI_AGENT_MOTION_CONTROL_PATH"


def configured_agent_motion_root() -> Path:
    configured = os.environ.get(AGENT_MOTION_CONTROL_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return IsaacRuntimePaths.configured().root / "agent-motion"


class IsaacMotionControlChannel:
    """One-active-command atomic channel; RuntimeHost remains the owner."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or configured_agent_motion_root()).expanduser().resolve()
        self.request_path = self.root / "request.json"
        self.grant_path = self.root / "grant.json"
        self.result_path = self.root / "result.json"
        self.ack_path = self.root / "ack.json"
        self.cancel_path = self.root / "cancel.json"
        self.owner_path = self.root / "owner.json"

    def owner_available(
        self,
        capability_id: str = "relative_move",
        *,
        max_age_s: float = 1.0,
    ) -> bool:
        owner = read_json(self.owner_path)
        updated_at = owner.get("updated_at") if owner else None
        capabilities = owner.get("capability_ids") if owner else None
        advertised = (
            capability_id in capabilities
            if isinstance(capabilities, list)
            else owner.get("capability_id") == capability_id
            if owner
            else False
        )
        return bool(
            owner
            and owner.get("schema_version") == SCHEMA_VERSION
            and owner.get("backend") == BACKEND_NAME
            and advertised
            and owner.get("execution_enabled") is True
            and isinstance(updated_at, (int, float))
            and not isinstance(updated_at, bool)
            and 0.0 <= time.time() - float(updated_at) <= max_age_s
        )

    def owner_advertised(self, capability_id: str) -> bool:
        """Return whether this runtime has claimed the capability at all."""

        owner = read_json(self.owner_path)
        capabilities = owner.get("capability_ids") if owner else None
        advertised = (
            capability_id in capabilities
            if isinstance(capabilities, list)
            else owner.get("capability_id") == capability_id
            if owner
            else False
        )
        return bool(
            owner
            and owner.get("schema_version") == SCHEMA_VERSION
            and owner.get("backend") == BACKEND_NAME
            and owner.get("released") is not True
            and advertised
        )

    def request_relative_move(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float = 70.0,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "relative_move",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )

    def request_move_distance(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float = 90.0,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "move_distance",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )

    def request_turn_around(
        self,
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float = 80.0,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "turn_around",
            {},
            execute_local,
            timeout_s=timeout_s,
        )

    def request_move_robot(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float = 8.0,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "move_robot",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )

    def request_motion(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        """Return ``None`` only when this capability has no unified owner.

        Once an owner advertises itself, loss of that owner fails closed and
        never falls back to legacy non-zero execution.
        """

        if not self.owner_available(capability_id):
            if not self.owner_advertised(capability_id):
                return None
            return {
                "ok": False,
                "tool_ok": False,
                "completed": False,
                "task_status": "runtime_unavailable",
                "error": (
                    f"unified {capability_id} owner is unavailable; "
                    "legacy physical fallback is forbidden"
                ),
            }
        token = uuid.uuid4().hex
        correlation = {
            "turn_id": f"native-turn-{token}",
            "task_id": f"native-task-{token}",
            "step_id": f"native-step-{token}",
            "tool_call_id": f"native-{capability_id}-{token}",
        }
        atomic_write_json(
            self.request_path,
            {
                "schema_version": SCHEMA_VERSION,
                "backend": BACKEND_NAME,
                "action": capability_id,
                "token": token,
                "requested_at": time.time(),
                "arguments": dict(arguments),
                **correlation,
            },
        )
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        granted = False
        while time.monotonic() < deadline:
            acknowledgement = read_json(self.ack_path)
            if acknowledgement and acknowledgement.get("token") == token:
                result = acknowledgement.get("result")
                return dict(result) if isinstance(result, Mapping) else {
                    "ok": False,
                    "completed": False,
                    "task_status": "invalid_tool_result",
                    "error": f"{capability_id} acknowledgement omitted result",
                }
            grant = read_json(self.grant_path)
            if grant and grant.get("token") == token:
                granted = grant.get("admitted") is True
                break
            time.sleep(0.02)
        if not granted:
            return {
                "ok": False,
                "completed": False,
                "task_status": "runtime_error",
                "error": f"unified {capability_id} owner did not admit the request",
            }

        def cancelled() -> bool:
            payload = read_json(self.cancel_path)
            return bool(
                (payload and payload.get("token") == token)
                or not self.owner_available(capability_id)
            )

        try:
            raw = execute_local(cancelled)
        except Exception as exc:  # noqa: BLE001 - report through Pipeline
            raw = f"{capability_id}_failed: runtime_error={exc}"
        atomic_write_json(
            self.result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "backend": BACKEND_NAME,
                "action": capability_id,
                "token": token,
                "finished_at": time.time(),
                "raw_result": str(raw),
            },
        )
        while time.monotonic() < deadline + 10.0:
            acknowledgement = read_json(self.ack_path)
            if acknowledgement and acknowledgement.get("token") == token:
                result = acknowledgement.get("result")
                return dict(result) if isinstance(result, Mapping) else {
                    "ok": False,
                    "completed": False,
                    "task_status": "invalid_tool_result",
                    "error": f"{capability_id} acknowledgement omitted result",
                }
            time.sleep(0.02)
        return {
            "ok": False,
            "completed": False,
            "task_status": "side_effect_unknown",
            "error": (
                f"{capability_id} produced physical progress but final "
                "acknowledgement was lost"
            ),
            "automatic_tool_replay": False,
        }


class IsaacMotionRequestService:
    """Consume native Skill requests without becoming a model loop."""

    def __init__(
        self,
        executor: Any,
        channel: IsaacMotionControlChannel,
        *,
        max_request_age_s: float = 10.0,
    ) -> None:
        self.executor = executor
        self.channel = channel
        self.max_request_age_s = max(1.0, float(max_request_age_s))
        self._last_token: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._active_token: str | None = None
        self._lock = threading.Lock()
        self._service_instance_id = uuid.uuid4().hex

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.channel.root.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="luxi-native-motion-service",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self.cancel_active("service_shutdown")
        self._stop.set()
        threads = tuple(
            thread
            for thread in (self._thread, self._worker)
            if thread is not None and thread is not threading.current_thread()
        )
        for thread in threads:
            thread.join(timeout=2.0)
        if all(not thread.is_alive() for thread in threads):
            self._release_owner()
        self._thread = None
        self._worker = None

    def _release_owner(self) -> None:
        """Release only this service instance's gracefully stopped lease.

        A stale descriptor left by a crash intentionally remains advertised so
        callers fail closed.  The instance check prevents an older service from
        releasing a newer service's ownership during a handover.
        """

        owner = read_json(self.channel.owner_path)
        if not owner or owner.get("service_instance_id") != self._service_instance_id:
            return
        released_at = time.time()
        atomic_write_json(
            self.channel.owner_path,
            {
                **owner,
                "execution_enabled": False,
                "released": True,
                "released_at": released_at,
                "updated_at": released_at,
            },
        )

    def cancel_active(self, reason: str) -> None:
        with self._lock:
            token = self._active_token
        if token is not None:
            self.executor.cancel_active(reason)
            atomic_write_json(
                self.channel.cancel_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "backend": BACKEND_NAME,
                    "token": token,
                    "reason": str(reason)[:200],
                    "cancelled_at": time.time(),
                },
            )

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            try:
                execution_enabled = bool(self.executor.prepare_owner())
            except Exception:
                execution_enabled = False
            atomic_write_json(
                self.channel.owner_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "backend": BACKEND_NAME,
                    "capability_id": "relative_move",
                    "capability_ids": sorted(self.executor.capability_ids),
                    "execution_enabled": execution_enabled,
                    "released": False,
                    "service_instance_id": self._service_instance_id,
                    "updated_at": time.time(),
                },
            )
            if not execution_enabled:
                continue
            request = read_json(self.channel.request_path)
            if request is None:
                continue
            token = request.get("token")
            with self._lock:
                busy = self._active_token is not None
            if (
                not isinstance(token, str)
                or not token
                or token == self._last_token
                or busy
            ):
                continue
            self._last_token = token
            if not self._valid(request):
                self._ack(
                    token,
                    {
                        "ok": False,
                        "completed": False,
                        "task_status": "tool_denied",
                        "error": "AgentOS motion request is stale or malformed",
                    },
                )
                continue
            with self._lock:
                self._active_token = token
            self._worker = threading.Thread(
                target=self._execute,
                args=(dict(request),),
                name=f"luxi-native-{request['action']}",
                daemon=True,
            )
            self._worker.start()

    def _valid(self, request: Mapping[str, Any]) -> bool:
        requested_at = request.get("requested_at")
        age = (
            time.time() - float(requested_at)
            if isinstance(requested_at, (int, float))
            and not isinstance(requested_at, bool)
            else math.inf
        )
        arguments = request.get("arguments")
        return bool(
            request.get("schema_version") == SCHEMA_VERSION
            and request.get("backend") == BACKEND_NAME
            and request.get("action") in self.executor.capability_ids
            and math.isfinite(age)
            and -0.25 <= age <= self.max_request_age_s
            and isinstance(arguments, Mapping)
            and all(
                isinstance(request.get(name), str) and request.get(name)
                for name in ("turn_id", "task_id", "step_id", "tool_call_id")
            )
        )

    def _execute(self, request: Mapping[str, Any]) -> None:
        token = str(request["token"])
        try:
            result = self.executor.execute_request(request, self.channel)
        except Exception as exc:  # noqa: BLE001 - durable incomplete result
            result = {
                "ok": False,
                "completed": False,
                "task_status": "runtime_error",
                "error": str(exc)[:1_000],
            }
        self._ack(token, result)
        with self._lock:
            if self._active_token == token:
                self._active_token = None

    def _ack(self, token: str, result: Mapping[str, Any]) -> None:
        request = read_json(self.channel.request_path)
        action = (
            str(request.get("action"))
            if request and request.get("token") == token
            else "motion"
        )
        atomic_write_json(
            self.channel.ack_path,
            {
                "schema_version": SCHEMA_VERSION,
                "backend": BACKEND_NAME,
                "action": action,
                "token": token,
                "acknowledged_at": time.time(),
                "result": dict(result),
            },
        )


def encode_motion_result(result: Mapping[str, Any]) -> str:
    return json.dumps(dict(result), ensure_ascii=False, separators=(",", ":"))
