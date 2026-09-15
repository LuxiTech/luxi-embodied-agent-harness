"""Versioned values and dependency-inversion ports for the unified runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4


SCHEMA_VERSION = 1


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class SideEffect(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    PHYSICAL = "physical"


class RetryPolicy(str, Enum):
    NEVER = "never"
    IDEMPOTENT = "idempotent"


class TaskOutcome(str, Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    sha256: str
    media_type: str
    captured_at: float


@dataclass(frozen=True)
class ToolDecision:
    name: str
    arguments: Mapping[str, Any]
    call_id: str = field(default_factory=lambda: new_id("tool"))


@dataclass(frozen=True)
class ModelReply:
    content: str = ""
    tool_calls: tuple[ToolDecision, ...] = ()
    finish_reason: str = "stop"
    usage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    session_id: str
    turn_id: str
    step_id: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    capability_revision: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityDescriptor:
    capability_id: str
    version: str
    provider: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    robot_types: frozenset[str] = frozenset()
    backends: frozenset[str] = frozenset()
    resources: frozenset[str] = frozenset()
    conflicts: frozenset[str] = frozenset()
    side_effect: SideEffect = SideEffect.NONE
    exclusive: bool = False
    timeout_s: float = 30.0
    retry: RetryPolicy = RetryPolicy.NEVER
    terminal: bool = False
    safety_policy: str | None = None
    verifier: str | None = None
    execution_modes: frozenset[str] = frozenset({"terminal"})
    composition_contract: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.execution_modes or not self.execution_modes <= {"terminal", "composed"}:
            raise ValueError("invalid capability execution_modes")
        if not self.capability_id.strip() or not self.version.strip():
            raise ValueError("capability_id and version are required")
        if self.timeout_s <= 0:
            raise ValueError("capability timeout must be positive")
        if self.side_effect is SideEffect.PHYSICAL and self.retry is not RetryPolicy.NEVER:
            raise ValueError("physical capabilities must use retry=never")

    def model_tool(self) -> Mapping[str, Any]:
        import json
        description = self.description or f"{self.provider} capability {self.capability_id}"
        if self.composition_contract:
            description += "\n组合契约：" + json.dumps(dict(self.composition_contract), ensure_ascii=False)
        return {
            "type": "function",
            "function": {
                "name": self.capability_id,
                "description": description,
                "parameters": dict(self.input_schema),
            },
        }


@dataclass(frozen=True)
class AgentScope:
    agent_id: str
    session_id: str
    robot_ids: frozenset[str]
    allowed_capabilities: frozenset[str]
    denied_capabilities: frozenset[str] = frozenset()
    visible_data: frozenset[str] = frozenset()
    budget_steps: int = 8
    budget_tools: int = 8
    deadline_monotonic: float | None = None
    safety_policy: str = "default"
    scope_revision: str = field(default_factory=lambda: new_id("scope"))

    def permits(self, capability_id: str, robot_id: str | None = None) -> bool:
        if capability_id in self.denied_capabilities:
            return False
        if capability_id not in self.allowed_capabilities:
            return False
        return robot_id is None or robot_id in self.robot_ids


@dataclass(frozen=True)
class CapabilitySnapshot:
    scope_revision: str
    snapshot_revision: str
    capabilities: Mapping[str, CapabilityDescriptor]
    created_at_monotonic: float = field(default_factory=time.monotonic)

    def model_tools(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(value.model_tool() for value in self.capabilities.values())


@dataclass(frozen=True)
class ToolRequest:
    session_id: str
    turn_id: str
    step_id: str
    task_id: str
    tool_call_id: str
    agent_id: str
    capability_id: str
    arguments: Mapping[str, Any]
    snapshot_revision: str
    robot_id: str | None = None
    boot_epoch: str | None = None
    deadline_monotonic: float | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True)
class ToolResult:
    status: str
    tool_ok: bool
    completed: bool
    error: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[ArtifactRef, ...] = ()
    timing: Mapping[str, float] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.completed and not self.tool_ok:
            raise ValueError("completed result must also be tool_ok")
        if self.completed and self.status in {
            "cancelled",
            "runtime_error",
            "navigation_timeout",
            "incomplete_budget_exhausted",
            "interrupted",
        }:
            raise ValueError("incomplete status cannot be completed")

    def for_model(self) -> Mapping[str, Any]:
        return {
            "status": self.status,
            "tool_ok": self.tool_ok,
            "completed": self.completed,
            "error": self.error,
            "evidence": dict(self.evidence),
            "artifact_refs": [ref.__dict__ for ref in self.artifact_refs],
            **dict(self.payload),
        }


@dataclass(frozen=True)
class SafetyRequest:
    robot_id: str
    capability_id: str
    side_effect: SideEffect
    deadline_monotonic: float | None
    observation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyDecision:
    admitted: bool
    reason: str
    policy_revision: str
    constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyEvidence:
    robot_id: str
    stop_command_completed: bool
    stationary_confirmed: bool
    stop_command_completed_at: float | None = None
    stationary_confirmed_at: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopLimits:
    max_steps: int = 8
    max_tool_calls: int = 8


@dataclass(frozen=True)
class LoopResult:
    session_id: str
    turn_id: str
    task_id: str
    outcome: TaskOutcome
    task_status: str
    completed: bool
    response: str
    planning_steps: int
    tool_calls: int
    last_tool_result: ToolResult | None = None


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError("operation cancelled")


class CancelledError(RuntimeError):
    pass


class ModelProvider(Protocol):
    def complete(self, request: ModelRequest, cancel: CancellationToken) -> ModelReply: ...


class ContextProvider(Protocol):
    def build_context(
        self, *, session_id: str, turn_id: str, instruction: str
    ) -> Sequence[Mapping[str, Any]]: ...


class ToolExecutor(Protocol):
    def execute(
        self,
        request: ToolRequest,
        *,
        scope: AgentScope,
        snapshot: CapabilitySnapshot,
        cancel: CancellationToken,
    ) -> ToolResult: ...


class OutcomePolicy(Protocol):
    def reduce(self, descriptor: CapabilityDescriptor, result: ToolResult) -> ToolResult: ...


class EventSink(Protocol):
    def emit(
        self,
        event_type: str,
        *,
        session_id: str,
        turn_id: str | None = None,
        step_id: str | None = None,
        task_id: str | None = None,
        tool_call_id: str | None = None,
        source: str,
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any: ...


class SafetyController(Protocol):
    def admit(self, request: SafetyRequest) -> SafetyDecision: ...

    def stop(self, robot_id: str, reason: str) -> SafetyEvidence: ...


HealthCheck = Callable[[], bool]
