"""The only admission, execution and verification path for Agent tools."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from .capabilities import LuxiCapabilityRegistry
from .contracts import (
    AgentScope,
    CancellationToken,
    CancelledError,
    CapabilityDescriptor,
    CapabilitySnapshot,
    SafetyRequest,
    SideEffect,
    ToolRequest,
    ToolResult,
)


class ToolAdapter(Protocol):
    def execute(self, request: ToolRequest, cancel: CancellationToken) -> ToolResult | Mapping[str, Any]: ...


class ToolExecutionFence(Protocol):
    """Optional robot-runtime fence around an admitted Tool execution."""

    def applies(self, request: ToolRequest) -> bool: ...

    def admit(
        self, request: ToolRequest, cancel: CancellationToken
    ) -> CancellationToken: ...

    def finish(self, request: ToolRequest, result: ToolResult) -> None: ...


class EvidenceOutcomePolicy:
    """Keep protocol success separate from verified task completion."""

    def reduce(self, descriptor: CapabilityDescriptor, result: ToolResult) -> ToolResult:
        if not result.completed:
            return result
        evidence = result.evidence
        valid = True
        error = result.error
        if descriptor.side_effect is SideEffect.PHYSICAL:
            valid = bool(evidence.get("stationary_confirmed") is True)
            if not valid:
                error = "physical completion lacks stationary_confirmed evidence"
        if valid and descriptor.verifier:
            valid = evidence.get(descriptor.verifier) is True
            if not valid:
                error = f"completion lacks {descriptor.verifier} evidence"
        if valid:
            return result
        return ToolResult(
            status="verification_failed",
            tool_ok=result.tool_ok,
            completed=False,
            error=error,
            evidence=evidence,
            artifact_refs=result.artifact_refs,
            timing=result.timing,
            payload=result.payload,
            raw=result.raw,
        )


class LuxiToolRegistry:
    """Bind exactly one implementation to each declared capability."""

    def __init__(self, capabilities: LuxiCapabilityRegistry) -> None:
        self.capabilities = capabilities
        self._adapters: dict[str, ToolAdapter] = {}
        self._lock = threading.RLock()

    def register(self, descriptor: CapabilityDescriptor, adapter: ToolAdapter) -> None:
        with self._lock:
            existing = self._adapters.get(descriptor.capability_id)
            if existing is not None and existing is not adapter:
                raise ValueError(f"tool implementation already registered: {descriptor.capability_id}")
            self.capabilities.register(descriptor)
            self._adapters[descriptor.capability_id] = adapter

    def adapter(self, capability_id: str) -> ToolAdapter | None:
        with self._lock:
            return self._adapters.get(capability_id)


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "null":
        return value is None
    return True


def _numeric_bounds_ok(value: float, schema: Mapping[str, Any]) -> bool:
    return not (
        ("minimum" in schema and value < float(schema["minimum"]))
        or ("maximum" in schema and value > float(schema["maximum"]))
        or ("exclusiveMinimum" in schema and value <= float(schema["exclusiveMinimum"]))
        or ("exclusiveMaximum" in schema and value >= float(schema["exclusiveMaximum"]))
    )


def validate_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> str | None:
    if schema.get("type") == "object" and not isinstance(arguments, Mapping):
        return "arguments must be an object"
    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [name for name in required if name not in arguments]
        if missing:
            return f"missing required arguments: {sorted(missing)}"
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            return f"unexpected arguments: {unexpected}"
    for name, value in arguments.items():
        field = properties.get(name)
        if not isinstance(field, Mapping):
            continue
        expected = field.get("type")
        allowed_types = [expected] if isinstance(expected, str) else expected
        if isinstance(allowed_types, list) and not any(
            _type_ok(value, item) for item in allowed_types if isinstance(item, str)
        ):
            return f"argument {name!r} has invalid type"
        if "enum" in field and value not in field["enum"]:
            return f"argument {name!r} is outside enum"
        if isinstance(value, str):
            if "minLength" in field and len(value) < int(field["minLength"]):
                return f"argument {name!r} is shorter than allowed"
            if "maxLength" in field and len(value) > int(field["maxLength"]):
                return f"argument {name!r} is longer than allowed"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            alternatives = field.get("anyOf")
            if isinstance(alternatives, list) and alternatives and not any(
                isinstance(candidate, Mapping)
                and _numeric_bounds_ok(numeric, candidate)
                for candidate in alternatives
            ):
                return f"argument {name!r} is outside allowed ranges"
            if not _numeric_bounds_ok(numeric, field):
                return f"argument {name!r} is outside allowed range"
    return None


def normalize_legacy_result(value: ToolResult | Mapping[str, Any]) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    payload = dict(value)
    tool_ok = bool(payload.pop("tool_ok", payload.get("ok", False)))
    payload.pop("ok", None)
    completed = payload.pop("completed", False) is True
    status = str(payload.pop("task_status", payload.pop("status", "tool_succeeded" if tool_ok else "runtime_error")))
    if payload.get("cancelled") is True and not tool_ok:
        status = "cancelled"
    error = payload.pop("error", None)
    evidence_value = payload.pop("evidence", {})
    evidence = dict(evidence_value) if isinstance(evidence_value, Mapping) else {}
    explicit_status_evidence = {
        "turn_verified": "turn_verified",
        "relative_move_verified": "relative_move_verified",
        "room_loop_verified": "room_loop_verified",
        "distance_verified": "distance_verified",
        "navigation_verified": "navigation_verified",
        "arrived_verified": "verification_observed",
        "follow_verified": "follow_verified",
        "object_fetched": "attachment_retained",
    }.get(status)
    if explicit_status_evidence is not None:
        # Provider adapters may have performed stricter physical verification
        # than can be inferred from a legacy status string.  Preserve an
        # explicit false claim so a forged/stale success cannot be upgraded by
        # normalization alone.
        evidence.setdefault(explicit_status_evidence, True)
    for key in (
        "planner_goal_reached",
        "stationary_confirmed",
        "stop_command_completed",
        "verification_observed",
        "frontier_exploration_verified",
        "text_navigation_verified",
        "attachment_retained",
    ):
        if key in payload:
            evidence[key] = payload[key]
    return ToolResult(
        status=status,
        tool_ok=tool_ok,
        completed=completed,
        error=str(error) if error else None,
        evidence=evidence,
        payload=payload,
        raw=dict(value),
    )


class LuxiToolPipeline:
    def __init__(
        self,
        registry: LuxiToolRegistry,
        *,
        events: Any,
        safety: Any,
        outcome_policy: Any | None = None,
        observation: Callable[[str], Mapping[str, Any]] | None = None,
        execution_fence: ToolExecutionFence | None = None,
    ) -> None:
        self.registry = registry
        self.events = events
        self.safety = safety
        self.outcome_policy = outcome_policy or EvidenceOutcomePolicy()
        self.observation = observation or (lambda _robot_id: {})
        self.execution_fence = execution_fence
        self._resource_locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def _denied(self, status: str, error: str) -> ToolResult:
        return ToolResult(status=status, tool_ok=False, completed=False, error=error, evidence={"execution_started": False})

    def _emit(self, event_type: str, request: ToolRequest, payload: Mapping[str, Any]) -> Any:
        return self.events.emit(
            event_type,
            session_id=request.session_id,
            turn_id=request.turn_id,
            step_id=request.step_id,
            task_id=request.task_id,
            tool_call_id=request.tool_call_id,
            source="tool-pipeline",
            payload=payload,
            idempotency_key=f"{request.tool_call_id}:{event_type}",
        )

    def _prior_state(self, request: ToolRequest) -> str | None:
        lookup = getattr(self.events, "tool_call_state", None)
        if callable(lookup):
            return lookup(request.session_id, request.tool_call_id)
        try:
            events = self.events.iter_events(request.session_id)
        except AttributeError:
            return None
        relevant = [event.event_type for event in events if event.tool_call_id == request.tool_call_id]
        if "tool/result" in relevant:
            return "finished"
        if "tool/started" in relevant:
            return "side_effect_unknown"
        return None

    def _execution_blocked(self, request: ToolRequest, cancel: CancellationToken) -> ToolResult | None:
        if cancel.cancelled:
            return self._denied("cancelled", "turn was cancelled before tool execution")
        if request.deadline_monotonic is not None and time.monotonic() >= request.deadline_monotonic:
            return self._denied("tool_timeout", "tool deadline expired before execution")
        return None

    def _execute_stop(self, request):
        """Emergency zero is independent of turn cancellation and durable storage."""
        audit_error = None
        try:
            self._emit("tool/started", request, {"capability_id": request.capability_id, "robot_id": request.robot_id})
        except Exception as exc:
            audit_error = str(exc)[:500]
        evidence = self.safety.stop(request.robot_id, "agent_stop")
        details = {"stop_command_completed": evidence.stop_command_completed,
                   "stationary_confirmed": evidence.stationary_confirmed,
                   "stop_command_completed_at": evidence.stop_command_completed_at,
                   "stationary_confirmed_at": evidence.stationary_confirmed_at,
                   "event_store_independent_stop": True}
        result = ToolResult("runtime_error" if audit_error else "stopped" if evidence.stationary_confirmed else "verification_failed",
                            evidence.stop_command_completed, False, error=audit_error, evidence=details)
        try:
            self._emit("tool/result", request, result.for_model())
        except Exception:
            pass
        return result

    # 统一工具执行边界：检查能力快照、权限、参数、期限、取消和重复调用，再做安全准入。
    # 负责执行适配器并记录结果；tool_ok 表示工具协议成功，不等于用户任务完成。
    def execute(
        self,
        request: ToolRequest,
        *,
        scope: AgentScope,
        snapshot: CapabilitySnapshot,
        cancel: CancellationToken,
    ) -> ToolResult:
        descriptor = snapshot.capabilities.get(request.capability_id)
        if descriptor is None or request.snapshot_revision != snapshot.snapshot_revision:
            result = self._denied("tool_denied", "capability is absent from the immutable Step snapshot")
            self._emit("tool/denied", request, result.for_model())
            return result
        current = self.registry.capabilities.get(request.capability_id)
        adapter = self.registry.adapter(request.capability_id)
        if current != descriptor or adapter is None:
            result = self._denied("tool_denied", "capability registration changed or implementation is unavailable")
            self._emit("tool/denied", request, result.for_model())
            return result
        if not scope.permits(request.capability_id, request.robot_id):
            result = self._denied("tool_denied", "Agent scope denies this capability or robot")
            self._emit("tool/denied", request, result.for_model())
            return result
        error = validate_arguments(request.arguments, descriptor.input_schema)
        if error:
            result = self._denied("invalid_input", error)
            self._emit("tool/denied", request, result.for_model())
            return result
        if request.capability_id in {"stop_robot", "stop_navigation"} and request.robot_id is not None:
            return self._execute_stop(request)
        if cancel.cancelled:
            result = self._denied("cancelled", "turn was cancelled before tool admission")
            self._emit("tool/denied", request, result.for_model())
            return result
        now = time.monotonic()
        deadline = request.deadline_monotonic
        if deadline is not None and deadline <= now:
            result = self._denied("tool_timeout", "tool deadline expired before admission")
            self._emit("tool/denied", request, result.for_model())
            return result
        if self._prior_state(request) is not None:
            result = self._denied(
                "side_effect_unknown",
                "tool_call_id was already started or finished; automatic replay is forbidden",
            )
            self._emit("tool/denied", request, result.for_model())
            return result

        blocked = self._execution_blocked(request, cancel)
        if blocked is not None:
            self._emit("tool/denied", request, blocked.for_model())
            return blocked

        decision = None
        if descriptor.side_effect is SideEffect.PHYSICAL:
            if request.robot_id is None:
                result = self._denied("tool_denied", "physical capability requires robot_id")
                self._emit("tool/denied", request, result.for_model())
                return result
            decision = self.safety.admit(
                SafetyRequest(
                    request.robot_id,
                    request.capability_id,
                    descriptor.side_effect,
                    deadline,
                    self.observation(request.robot_id),
                )
            )
            self._emit(
                "safety/decision",
                request,
                {"admitted": decision.admitted, "reason": decision.reason, "policy_revision": decision.policy_revision},
            )
            if not decision.admitted:
                result = self._denied("risk_blocked", decision.reason)
                self._emit("tool/denied", request, result.for_model())
                return result

        resources = sorted(descriptor.resources | descriptor.conflicts)
        with self._locks_lock:
            locks = [self._resource_locks.setdefault(name, threading.Lock()) for name in resources]
        acquired: list[threading.Lock] = []
        for lock in locks:
            if not lock.acquire(blocking=False):
                for held in reversed(acquired):
                    held.release()
                result = self._denied("resource_conflict", "capability resource is already owned")
                self._emit("tool/denied", request, result.for_model())
                return result
            acquired.append(lock)

        started = time.monotonic()
        self._emit(
            "tool/admitted",
            request,
            {"capability_id": request.capability_id, "capability_version": descriptor.version},
        )
        execution_cancel = cancel
        fenced = bool(
            self.execution_fence is not None
            and self.execution_fence.applies(request)
        )
        if fenced:
            try:
                execution_cancel = self.execution_fence.admit(request, cancel)
            except Exception as exc:
                for lock in reversed(acquired):
                    lock.release()
                if descriptor.side_effect is SideEffect.PHYSICAL:
                    self.safety.release(request.robot_id)
                status = (
                    "side_effect_unknown"
                    if "replay" in str(exc).casefold()
                    else "tool_denied"
                )
                result = self._denied(status, str(exc)[:1_000])
                self._emit("tool/denied", request, result.for_model())
                return result
        adapter_executed = False
        try:
            result = self._execution_blocked(request, execution_cancel)
            if result is None:
                # This durable boundary can itself block on storage. Check the
                # deadline again afterwards, before any adapter side effects.
                self._emit("tool/started", request, {"arguments": dict(request.arguments),
                           "capability_id": request.capability_id, "robot_id": request.robot_id,
                           "side_effect": descriptor.side_effect.value})
                result = self._execution_blocked(request, execution_cancel)
            if result is None:
                if descriptor.side_effect is SideEffect.PHYSICAL:
                    self.safety.mark_active(request.robot_id)
                adapter_executed = True
                raw = adapter.execute(request, execution_cancel)
                result = normalize_legacy_result(raw)
            if (
                deadline is not None
                and time.monotonic() > deadline
                and result.status != "side_effect_unknown"
                and adapter_executed
            ):
                result = ToolResult(
                    "tool_timeout",
                    False,
                    False,
                    "tool returned after its execution deadline",
                    evidence=result.evidence,
                    payload=result.payload,
                    raw=result.raw,
                )
            elif (
                execution_cancel.cancelled
                and result.status != "side_effect_unknown"
                and adapter_executed
            ):
                result = ToolResult(
                    "cancelled",
                    False,
                    False,
                    "tool returned after cancellation",
                    evidence=result.evidence,
                    payload=result.payload,
                    raw=result.raw,
                )
        except CancelledError as exc:
            result = ToolResult("tool_timeout" if deadline is not None and time.monotonic() >= deadline else "cancelled", False, False, str(exc))
        except Exception as exc:
            result = ToolResult("runtime_error", False, False, str(exc)[:1_000])
        finally:
            for lock in reversed(acquired):
                lock.release()

        if descriptor.side_effect is SideEffect.PHYSICAL:
            # A synchronous Skill result never owns final motion truth. Always
            # cross the robot-local stop barrier and collect independent fresh
            # odometry, including after a nominally successful terminal tool.
            evidence = self.safety.stop(request.robot_id, result.status)
            merged = dict(result.evidence)
            merged.update(
                {
                    "stop_command_completed": evidence.stop_command_completed,
                    "stationary_confirmed": evidence.stationary_confirmed,
                    "stop_command_completed_at": evidence.stop_command_completed_at,
                    "stationary_confirmed_at": evidence.stationary_confirmed_at,
                }
            )
            stop_verified = bool(
                evidence.stop_command_completed
                and evidence.stationary_confirmed
            )
            result = ToolResult(
                result.status if stop_verified else "side_effect_unknown",
                result.tool_ok and stop_verified,
                result.completed and stop_verified,
                (
                    result.error
                    if stop_verified
                    else (
                        "physical side effect is unknown because Safety Kernel "
                        "could not verify the stationary stop barrier"
                    )
                ),
                merged,
                result.artifact_refs,
                result.timing,
                result.payload,
                result.raw,
            )
        # Final task reduction must see the independent stop evidence. Running
        # it before the physical barrier would permanently downgrade a valid
        # terminal result that correctly leaves stationarity to the Kernel.
        result = self.outcome_policy.reduce(descriptor, result)
        if fenced:
            try:
                self.execution_fence.finish(request, result)
            except Exception as exc:
                result = ToolResult(
                    "runtime_error",
                    False,
                    False,
                    f"RuntimeHost fence completion failed: {str(exc)[:800]}",
                    result.evidence,
                    result.artifact_refs,
                    result.timing,
                    result.payload,
                    result.raw,
                )
        result = ToolResult(
            result.status,
            result.tool_ok,
            result.completed,
            result.error,
            result.evidence,
            result.artifact_refs,
            {**dict(result.timing), "elapsed_s": max(0.0, time.monotonic() - started)},
            result.payload,
            result.raw,
        )
        self._emit("tool/result", request, result.for_model())
        return result
