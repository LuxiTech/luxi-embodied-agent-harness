"""Privileged development skills composed by the existing LuxiAgentLoop.

This is deliberately NOT a challenge policy. Backend implementations expose
oracle conditions to diagnose planning independently of learned perception.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any, Iterator, Mapping, Protocol

from harness.runtime.agent_loop import LuxiAgentLoop
from harness.runtime.capabilities import AgentScopeResolver, LuxiCapabilityRegistry
from harness.runtime.contracts import (
    AgentScope, CancellationToken, CancelledError, CapabilityDescriptor, LoopLimits,
    SideEffect, ToolRequest, ToolResult,
)
from harness.runtime.safety_kernel import LuxiSafetyKernel
from harness.runtime.tool_pipeline import LuxiToolPipeline, LuxiToolRegistry


PRIMITIVES = {
    "behavior_navigate": "NAVIGATE_TO",
    "behavior_pick": "GRASP",
    "behavior_place_inside": "PLACE_INSIDE",
    "behavior_place_on_top": "PLACE_ON_TOP",
    "behavior_open": "OPEN",
    "behavior_close": "CLOSE",
    "behavior_toggle_on": "TOGGLE_ON",
    "behavior_toggle_off": "TOGGLE_OFF",
}
ARITY = {"holding": 1, "open": 1, "toggled_on": 1, "inside": 2, "ontop": 2}
PROVENANCE = {"profile": "oracle-dev", "privileged_information": True, "challenge_eligible": False}


@dataclass(frozen=True)
class Condition:
    predicate: str
    subject: str
    target: str | None = None
    expected: bool = True

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> Condition:
        if not isinstance(value, Mapping) or set(value) - {"predicate", "subject", "target", "expected"}:
            raise ValueError("condition must contain only predicate, subject, target, expected")
        condition = cls(**value)
        if condition.predicate not in ARITY:
            raise ValueError("unsupported predicate")
        if not isinstance(condition.subject, str) or not condition.subject.strip():
            raise ValueError("condition subject is required")
        if type(condition.expected) is not bool:
            raise ValueError("condition expected must be boolean")
        if ARITY[condition.predicate] == 2:
            if not isinstance(condition.target, str) or not condition.target.strip():
                raise ValueError("binary predicate requires target")
        elif condition.target is not None:
            raise ValueError("unary predicate cannot have target")
        return condition


def parse_conditions(values: Any) -> tuple[Condition, ...]:
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise ValueError("conditions must be a nonempty list of at most 32 predicates")
    return tuple(Condition.parse(value) for value in values)


@dataclass(frozen=True)
class DevelopmentTask:
    instruction: str
    goals: tuple[Condition, ...]

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> DevelopmentTask:
        if set(value) != {"instruction", "goals"}:
            raise ValueError("task requires instruction and goals; no executable plan is accepted")
        instruction = value["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("task instruction is required")
        return cls(instruction, parse_conditions(value["goals"]))


class PrimitiveFailure(RuntimeError):
    """A bounded primitive attempt failed; the physical state may have changed."""


class EpisodeEnded(RuntimeError):
    pass


class DevelopmentBackend(Protocol):
    robot_id: str
    revision: int
    kind: str

    def observe(self) -> Mapping[str, Any]: ...
    def condition(self, condition: Condition) -> bool | None: ...
    def primitive(self, name: str, object_ref: str) -> Iterator[Any]: ...
    def step(self, action: Any) -> None: ...
    def hold_step(self) -> None: ...
    def motion_sample(self) -> Mapping[str, Any]: ...


class DevelopmentGateway:
    """Local stop barrier; hold joint targets instead of zeroing joint positions."""

    def __init__(self, backend: DevelopmentBackend) -> None:
        self.backend = backend
        self.samples: list[Mapping[str, Any]] = []

    def publish_zero(self, robot_id: str) -> float:
        if robot_id != self.backend.robot_id:
            raise ValueError("robot mismatch")
        self.samples.clear()
        self.backend.hold_step()
        return time.monotonic()

    def stationary_samples(self, robot_id: str, after_monotonic: float) -> list[Mapping[str, Any]]:
        if robot_id != self.backend.robot_id:
            raise ValueError("robot mismatch")
        self.backend.hold_step()
        sample = dict(self.backend.motion_sample())
        speed, yaw = sample.get("planar_speed_mps"), sample.get("yaw_rate_rps")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (speed, yaw)):
            return []
        self.samples.append(sample)
        return [s for s in self.samples if s["timestamp_monotonic"] > after_monotonic]


def schema(properties: Mapping[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": dict(properties), "required": required, "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1, "maxLength": 300}
CONDITIONS_SCHEMA = {"type": "array", "items": {"type": "object"}}


class DevelopmentSkills:
    """One episode's fixed goals, model-created subgoals and bounded attempts."""

    def __init__(self, backend: DevelopmentBackend, task: DevelopmentTask, *,
                 mode: str, max_action_steps: int = 2000, max_episode_steps: int = 10000,
                 max_attempts: int = 3) -> None:
        if mode != "oracle-dev":
            raise ValueError("privileged development skills cannot run in challenge mode")
        if min(max_action_steps, max_episode_steps, max_attempts) < 1:
            raise ValueError("budgets must be positive")
        if not task.goals:
            raise ValueError("task must have fixed nonempty goals")
        self.backend, self.task = backend, task
        self.max_action_steps, self.max_episode_steps = max_action_steps, max_episode_steps
        self.max_attempts = max_attempts
        self.action_steps = 0
        self.subgoals: dict[str, tuple[Condition, ...]] = {}
        self.attempts: dict[tuple[str, str], int] = {}
        self.checked_revision: dict[str, int] = {}
        self.checked_verdict: dict[str, str] = {}
        self.observed_revision: int | None = None
        self.last_stop: Mapping[str, Any] = {}
        self.owner: tuple[str, str, str] | None = None

    def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        specs = {
            "behavior_observe": (schema({}, []), "Observe exact development object references and revision."),
            "behavior_set_subgoal": (schema({"subgoal_id": TEXT, "conditions": CONDITIONS_SCHEMA},
                                              ["subgoal_id", "conditions"]),
                                     "Declare immutable subgoal conditions. Does not change final task goals."),
            "behavior_check_subgoal": (schema({"subgoal_id": TEXT}, ["subgoal_id"]),
                                       "Fresh subgoal verification: passed, failed or unknown. Required before each action."),
            "behavior_check_task": (schema({}, []), "Check ALL fixed task goals freshly. Only this tool can complete the task."),
            "behavior_stop": (schema({}, []), "Stop the current development task without claiming completion."),
        }
        for name, primitive in PRIMITIVES.items():
            specs[name] = (schema({"object_ref": TEXT, "subgoal_id": TEXT, "recovery_reason": TEXT},
                                  ["object_ref", "subgoal_id"]),
                           f"One {primitive} attempt. object_ref is target object/container. "
                           "Requires fresh subgoal check. Repeated attempt requires recovery_reason. "
                           "Primitive return is not subgoal or task completion.")
        return tuple(CapabilityDescriptor(
            name, "dev-v1", "behavior-oracle-dev", input_schema,
            description=f"DEVELOPMENT ONLY / PRIVILEGED. {description}",
            backends=frozenset({"behavior-oracle-dev"}),
            resources=frozenset({f"behavior:{self.backend.robot_id}"}),
            side_effect=SideEffect.PHYSICAL if name in PRIMITIVES or name == "behavior_stop" else SideEffect.READ_ONLY,
            exclusive=True, timeout_s=120.0,
            terminal=name == "behavior_stop",
            verifier="task_goals_verified" if name == "behavior_check_task" else None,
        ) for name, (input_schema, description) in specs.items())

    def result(self, status: str, *, tool_ok: bool = True, completed: bool = False,
               error: str | None = None, evidence: Mapping[str, Any] | None = None,
               **payload: Any) -> ToolResult:
        return ToolResult(status, tool_ok, completed, error, evidence=evidence or {},
                          payload={**PROVENANCE, "backend_kind": self.backend.kind,
                                   "revision": self.backend.revision, **payload})

    def check(self, conditions: tuple[Condition, ...]) -> tuple[str, list[dict[str, Any]]]:
        values = []
        for condition in conditions:
            value = self.backend.condition(condition)
            if value is not None and type(value) is not bool:
                raise ValueError("condition backend must return bool or None")
            verdict = "unknown" if value is None else "passed" if value == condition.expected else "failed"
            values.append({"condition": asdict(condition), "verdict": verdict})
        verdicts = {v["verdict"] for v in values}
        return ("failed" if "failed" in verdicts else "unknown" if "unknown" in verdicts else "passed"), values

    def execute(self, request: ToolRequest, cancel: CancellationToken) -> ToolResult:
        cancel.raise_if_cancelled()
        owner = (request.session_id, request.turn_id, request.task_id)
        if self.owner is None:
            self.owner = owner
        if owner != self.owner:
            return self.result("tool_denied", tool_ok=False, error="create a fresh development runtime for each episode/turn")
        try:
            return self._execute(request, cancel)
        except (ValueError, KeyError, TypeError) as exc:
            return self.result("invalid_input", tool_ok=False, error=str(exc))

    def _execute(self, request: ToolRequest, cancel: CancellationToken) -> ToolResult:
        name, args = request.capability_id, request.arguments
        if name == "behavior_observe":
            observation = dict(self.backend.observe())
            self.observed_revision = self.backend.revision
            return self.result("observation_available", observation=observation,
                               goals=[asdict(c) for c in self.task.goals])
        if name == "behavior_set_subgoal":
            subgoal_id = args["subgoal_id"]
            conditions = parse_conditions(args["conditions"])
            if subgoal_id in self.subgoals and self.subgoals[subgoal_id] != conditions:
                raise ValueError("existing subgoal conditions are immutable; use a new subgoal id")
            if subgoal_id not in self.subgoals and len(self.subgoals) >= 32:
                return self.result("incomplete_budget_exhausted", tool_ok=False)
            self.subgoals[subgoal_id] = conditions
            return self.result("subgoal_declared", subgoal_id=subgoal_id,
                               conditions=[asdict(c) for c in conditions])
        if name in {"behavior_check_subgoal", "behavior_check_task"}:
            final = name == "behavior_check_task"
            conditions = self.task.goals if final else self.subgoals[args["subgoal_id"]]
            verdict, checks = self.check(conditions)
            self.observed_revision = self.backend.revision
            if not final:
                self.checked_revision[args["subgoal_id"]] = self.backend.revision
                self.checked_verdict[args["subgoal_id"]] = verdict
            # A fresh check is required AFTER the independent stop barrier.
            stationary = self.last_stop.get("stationary_confirmed") is True
            completed = final and verdict == "passed" and stationary
            return self.result("task_verified" if completed else "condition_checked",
                               completed=completed, verdict=verdict, checks=checks,
                               subgoal_id=None if final else args["subgoal_id"],
                               evidence={**self.last_stop, "task_goals_verified": completed,
                                         "verification_timestamp_monotonic": time.monotonic()})
        if name == "behavior_stop":
            return self.result("cancelled")  # Pipeline owns the stop barrier.
        if name not in PRIMITIVES:
            raise ValueError("unsupported development skill")
        subgoal_id, object_ref = args["subgoal_id"], args["object_ref"]
        if subgoal_id not in self.subgoals:
            raise ValueError("declare subgoal first")
        if self.checked_revision.get(subgoal_id) != self.backend.revision:
            return self.result("verification_required", tool_ok=False, error="check this subgoal using fresh observations before acting")
        if self.checked_verdict.get(subgoal_id) == "unknown":
            return self.result("verification_required", tool_ok=False, error="subgoal state is unknown; obtain evidence before acting")
        key = (name, object_ref)
        attempts = self.attempts.get(key, 0)
        if attempts >= self.max_attempts:
            return self.result("incomplete_budget_exhausted", tool_ok=False, error="primitive attempt budget exhausted")
        if attempts and not str(args.get("recovery_reason", "")).strip():
            raise ValueError("new attempt requires recovery_reason after fresh verification")
        self.attempts[key] = attempts + 1
        self.last_stop = {}
        self.checked_revision.clear()
        generated = self.backend.primitive(PRIMITIVES[name], object_ref)
        steps = 0
        try:
            while True:
                cancel.raise_if_cancelled()
                if request.deadline_monotonic is not None and time.monotonic() >= request.deadline_monotonic:
                    return self.result("tool_timeout", tool_ok=False)
                if steps >= self.max_action_steps or self.action_steps >= self.max_episode_steps:
                    return self.result("incomplete_budget_exhausted", tool_ok=False)
                try:
                    action = next(generated)
                except StopIteration:
                    break
                cancel.raise_if_cancelled()
                if request.deadline_monotonic is not None and time.monotonic() >= request.deadline_monotonic:
                    return self.result("tool_timeout", tool_ok=False)
                # Count even a None yield to bound a malformed generator.
                steps += 1
                self.action_steps += 1
                if action is not None:
                    self.backend.step(action)
        except PrimitiveFailure as exc:
            return self.result("primitive_failed", error=str(exc), subgoal_id=subgoal_id,
                               attempt=attempts + 1, action_steps=steps)
        except EpisodeEnded as exc:
            return self.result("incomplete_budget_exhausted", tool_ok=False, error=str(exc))
        except CancelledError:
            return self.result("cancelled", tool_ok=False)
        except Exception as exc:
            return self.result("side_effect_unknown", tool_ok=False, error=str(exc))
        finally:
            close = getattr(generated, "close", None)
            if close:
                close()
        return self.result("primitive_executed", subgoal_id=subgoal_id,
                           attempt=attempts + 1, action_steps=steps)


class DevelopmentSafety(LuxiSafetyKernel):
    def __init__(self, skills: DevelopmentSkills) -> None:
        super().__init__(DevelopmentGateway(skills.backend))
        self.skills = skills

    def stop(self, robot_id: str, reason: str):
        evidence = super().stop(robot_id, reason)
        self.skills.last_stop = {
            "stationary_confirmed": evidence.stationary_confirmed,
            "stop_command_completed": evidence.stop_command_completed,
            "stationary_confirmed_at": evidence.stationary_confirmed_at,
        }
        return evidence


SYSTEM_PROMPT = """You compose BEHAVIOR DEVELOPMENT skills, using privileged observations.
This run is not a challenge submission. The fixed final goals cannot be changed.
Call one tool at a time and use its actual result to select the next action.
Observe exact object references. Declare a small subgoal with conditions, check it,
then act if necessary. After each action check again; tool_ok is not goal success.
For placement object_ref is the destination container/surface, not the held object.
For a failed action inspect the current state and provide recovery_reason for any
new attempt. Unknown evidence requires more observation, not claimed success.
Finally call behavior_check_task to check all fixed goals together. If they already
hold at startup, call behavior_stop only if aborting; otherwise a local stop barrier
is established at runtime startup. Never invent tools, object references or evidence.
"""


class DevelopmentContext:
    def __init__(self, task: DevelopmentTask) -> None:
        self.task = task

    def build_context(self, *, session_id: str, turn_id: str, instruction: str):
        return ({"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"instruction": instruction,
                 "fixed_goals": [asdict(c) for c in self.task.goals], **PROVENANCE}, ensure_ascii=False)})


def build_development_runtime(*, backend: DevelopmentBackend, task: DevelopmentTask,
                              model: Any, events: Any, mode: str,
                              max_steps: int = 64, max_tools: int = 64,
                              deadline_s: float = 600.0, **skill_options: Any):
    """Fresh isolated composition per episode; no changes to existing catalogs."""
    skills = DevelopmentSkills(backend, task, mode=mode, **skill_options)
    registry = LuxiCapabilityRegistry()
    tools = LuxiToolRegistry(registry)
    for descriptor in skills.descriptors():
        tools.register(descriptor, skills)
    names = frozenset(registry.all())
    resolver = AgentScopeResolver(registry, robot_capabilities=lambda _: names,
                                  robot_backends=lambda _: "behavior-oracle-dev")
    safety = DevelopmentSafety(skills)
    pipeline = LuxiToolPipeline(tools, events=events, safety=safety,
                               observation=lambda _: {"timestamp_monotonic": time.monotonic()})
    session_id = events.create_session(metadata={**PROVENANCE, "backend_kind": backend.kind})
    scope = AgentScope("behavior-development", session_id, frozenset({backend.robot_id}), names,
                       budget_steps=max_steps, budget_tools=max_tools,
                       deadline_monotonic=time.monotonic() + deadline_s)
    loop = LuxiAgentLoop(model=model, tools=pipeline, capabilities=resolver, events=events,
                         safety=safety, context=DevelopmentContext(task),
                         limits=LoopLimits(max_steps, max_tools),
                         completion_capability="behavior_check_task")
    return loop, scope, skills
