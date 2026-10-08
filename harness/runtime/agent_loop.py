"""Backend-independent, bounded and auditable Luxi Agent loop."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

from .contracts import (
    AgentScope,
    CancellationToken,
    CancelledError,
    ContextProvider,
    LoopLimits,
    LoopResult,
    ModelProvider,
    ModelRequest,
    TaskOutcome,
    ToolRequest,
    ToolResult,
    new_id,
)


class EmptyContextProvider:
    def build_context(
        self, *, session_id: str, turn_id: str, instruction: str
    ) -> Sequence[Mapping[str, Any]]:
        return ({"role": "user", "content": instruction},)


class LuxiAgentLoop:
    """One state machine; providers and tools supply only boundary behavior."""

    def __init__(
        self,
        *,
        model: ModelProvider,
        tools: Any,
        capabilities: Any,
        events: Any,
        safety: Any,
        context: ContextProvider | None = None,
        limits: LoopLimits | None = None,
        completion_capability: str | None = None,
        boot_epoch_provider=None,
        argument_policy=None,
        composed_observe=None,
    ) -> None:
        self.composed_observe = composed_observe
        self.boot_epoch_provider = boot_epoch_provider
        self.argument_policy = argument_policy
        self.model = model
        self.tools = tools
        self.capabilities = capabilities
        self.events = events
        self.safety = safety
        self.context = context or EmptyContextProvider()
        self.limits = limits or LoopLimits()
        # Opt-in composition mode: only this verifier may finish the task.
        # Existing terminal capabilities keep their stop-after-one-call semantics.
        if completion_capability is not None and not completion_capability.strip():
            raise ValueError("completion_capability must be nonempty when configured")
        self.completion_capability = completion_capability

    @staticmethod
    def _persistent_messages(messages):
        from .context import persistent_messages
        return persistent_messages(messages)

    def _emit(
        self,
        event_type: str,
        *,
        session_id: str,
        turn_id: str,
        task_id: str,
        step_id: str | None = None,
        tool_call_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        return self.events.emit(
            event_type,
            session_id=session_id,
            turn_id=turn_id,
            step_id=step_id,
            task_id=task_id,
            tool_call_id=tool_call_id,
            source="agent-loop",
            payload=payload or {},
            idempotency_key=(
                f"{turn_id}:{step_id or '-'}:{tool_call_id or '-'}:{event_type}"
            ),
        )

    def _stop_all(
        self,
        scope: AgentScope,
        *,
        session_id: str,
        turn_id: str,
        task_id: str,
        reason: str,
    ) -> None:
        for robot_id in sorted(scope.robot_ids):
            try:
                evidence = self.safety.stop(robot_id, reason)
                payload = {
                    "robot_id": robot_id,
                    "reason": reason,
                    "stop_command_completed": evidence.stop_command_completed,
                    "stationary_confirmed": evidence.stationary_confirmed,
                    "stop_command_completed_at": evidence.stop_command_completed_at,
                    "stationary_confirmed_at": evidence.stationary_confirmed_at,
                }
            except Exception as exc:
                payload = {
                    "robot_id": robot_id,
                    "reason": reason,
                    "stop_command_completed": False,
                    "stationary_confirmed": False,
                    "error": str(exc)[:1_000],
                }
            try:
                self._emit(
                    "safety/stopped",
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    payload=payload,
                )
            except Exception:
                # Emergency behavior is independent from event-store health.
                pass

    # 唯一模型执行循环：生成工具快照、读取当前观测、调用模型，再经 Pipeline 执行技能。
    # composed 每轮最多执行一个模型工具调用，将反馈带入下一轮；计划不是自动播放的脚本。
    # 全程执行预算/取消约束，模型文本或普通技能成功不能替代 compose_verify 的整体验收。
    def run(
        self,
        instruction: str,
        *,
        scope: AgentScope,
        cancel: CancellationToken | None = None,
        turn_id: str | None = None,
        task_id: str | None = None,
        execution_policy=None,
        task_spec=None,
    ) -> LoopResult:
        from .execution_policy import TerminalPolicy
        from .task_state import context_summary

        # Legacy isolated BEHAVIOR keeps its existing completion contract.
        policy = execution_policy or TerminalPolicy()
        completion_capability = (policy.completion_capability if execution_policy
                                 else self.completion_capability)
        composed = policy.mode == "composed"
        if composed and task_spec is None:
            raise ValueError("composed task requires immutable final conditions")
        text = instruction.strip()
        if not text:
            raise ValueError("instruction must not be empty")
        cancel = cancel or CancellationToken()
        session_id = scope.session_id
        turn_id = turn_id or new_id("turn")
        task_id = task_id or new_id("task")
        if composed and text != task_spec.instruction:
            raise ValueError("指令与已固定的组合任务不一致")
        started = time.monotonic()
        max_steps = min(self.limits.max_steps, scope.budget_steps)
        max_tools = min(self.limits.max_tool_calls, scope.budget_tools)
        planning_steps = 0
        tool_calls = 0
        last_result: ToolResult | None = None
        last_capability: str | None = None
        response = ""
        reply_error = ""
        finished_by_response = False
        messages = list(
            self.context.build_context(
                session_id=session_id,
                turn_id=turn_id,
                instruction=text,
            )
        )
        if composed:
            # Mode isolation: old terminal tool histories and camera frames are not model instructions.
            messages = [{"role": "user", "content": text}]
        no_progress = 0
        argument_repairs = 0
        progress_fingerprint = None
        self._emit(
            "turn/started",
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            payload={"instruction": text, "agent_id": scope.agent_id},
        )
        self._emit(
            "task/accepted", session_id=session_id, turn_id=turn_id, task_id=task_id,
            payload={"execution_mode": "composed" if completion_capability else "terminal",
                     "task_schema_version": getattr(task_spec, "schema_version", 1),
                     "goal": task_spec.payload() if task_spec else {"instruction": text},
                     "completion_capability": completion_capability},
        )
        try:
            for _ in range(max_steps):
                cancel.raise_if_cancelled()
                if scope.deadline_monotonic is not None and time.monotonic() >= scope.deadline_monotonic:
                    raise TimeoutError("turn deadline expired")
                step_id = new_id("step")
                snapshot = self.capabilities.snapshot(scope)
                if execution_policy is not None or self.completion_capability is None:
                    snapshot = policy.snapshot(snapshot)
                if composed and completion_capability not in snapshot.capabilities:
                    raise RuntimeError("final verifier is unavailable in current capability scope")
                planning_capabilities = []
                plan_required = False
                if composed:
                    from dataclasses import replace
                    dynamic = getattr(task_spec, "kind", "") == "dynamic"
                    if dynamic:
                        from harness.skills.composition.contracts import dynamic_model_tools
                        # Phase gating hides executable tools, not their planning semantics.
                        # Only the already scope/backend-filtered snapshot may enter this catalog.
                        planning_capabilities = [
                            {"name": tool["function"]["name"], "description": tool["function"]["description"]}
                            for tool in dynamic_model_tools(snapshot.model_tools(), task_spec)]
                    allowed = set(snapshot.capabilities)
                    # 未确认时只开放目标提议/阻塞工具；确认但无计划时仍不开放运动工具。
                    if dynamic and not task_spec.confirmed:
                        allowed &= {"compose_propose_goal", "compose_blocked"}
                    else:
                        allowed -= {"compose_propose_goal"}
                        if not dynamic:
                            allowed -= {"compose_blocked", "compose_locate", "compose_place"}
                    if dynamic and task_spec.confirmed:
                        from .task_state import task_projection
                        plan_required = not task_projection(self.events, session_id, task_id)["plan"]
                        if plan_required:
                            allowed -= {"compose_navigate", "compose_face", "compose_attach", "compose_release", "compose_place", "compose_locate"}
                    snapshot = replace(snapshot, capabilities={k: v for k, v in snapshot.capabilities.items() if k in allowed})
                model_messages = messages
                if composed:
                    # Rebuild durable task state for every decision, independent of chat history.
                    summary = context_summary(self.events, session_id, task_id)
                    if dynamic:
                        planning_context = json.loads(summary["content"])
                        planning_context["planning_state"] = {
                            "phase": "proposal" if not task_spec.confirmed else "planning" if plan_required else "execution",
                            "next_required_tool": "compose_plan" if plan_required else None,
                        }
                        catalog = []
                        for entry in planning_capabilities:
                            name = entry["name"]
                            callable_now = name in snapshot.capabilities
                            item = {**({"name": name} if callable_now else entry),
                                    "callable_now": callable_now}
                            if not callable_now:
                                if not task_spec.confirmed:
                                    reason, available_after = "goal_confirmation_required", "user_confirmation"
                                elif name == "compose_propose_goal":
                                    reason, available_after = "goal_already_confirmed", None
                                elif plan_required:
                                    reason, available_after = "plan_required", "compose_plan"
                                else:
                                    reason, available_after = "not_available_in_phase", None
                                item.update(unavailable_reason=reason, available_after=available_after)
                            catalog.append(item)
                        planning_context["planning_capabilities"] = catalog
                        summary = {"role": "user", "content": json.dumps(planning_context, ensure_ascii=False)}
                    if self.composed_observe is not None:
                        observation = self.composed_observe()
                        from harness.skills.composed_tasks import _fresh
                        if not _fresh(observation):
                            last_result = ToolResult("observation_unavailable", False, False)
                            self._stop_all(scope, session_id=session_id, turn_id=turn_id, task_id=task_id,
                                           reason=last_result.status)
                            break
                        if getattr(task_spec, "world_revision", "") and observation.get("world_revision") != task_spec.world_revision:
                            last_result = ToolResult("side_effect_unknown", False, False, "world revision changed")
                            self._stop_all(scope, session_id=session_id, turn_id=turn_id, task_id=task_id, reason=last_result.status)
                            break
                        from .task_state import task_projection
                        state = task_projection(self.events, session_id, task_id)
                        if state.get("held_entity") and observation.get("attached_entity") != state["held_entity"]:
                            last_result = ToolResult("side_effect_unknown", False, False, "attachment lost")
                            self._stop_all(scope, session_id=session_id, turn_id=turn_id, task_id=task_id, reason=last_result.status)
                            break
                        summary = {"role": "user", "content": json.dumps({
                            "task_state": json.loads(summary["content"]), "current_observation": observation,
                            "remaining_steps": max_steps-planning_steps, "remaining_tools": max_tools-tool_calls}, ensure_ascii=False)}
                    # Retain complete recent assistant/tool exchanges, not an unbounded transcript.
                    starts = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
                    recent = messages[starts[-8]:] if len(starts) > 8 else messages[1:]
                    model_messages = [messages[0], *recent, summary]
                self._emit(
                    "step/started",
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    step_id=step_id,
                    payload={
                        "scope_revision": scope.scope_revision,
                        "capability_revision": snapshot.snapshot_revision,
                        "capabilities": list(snapshot.capabilities),
                    },
                )
                from harness.skills.composition.prompt import PROMPT_VERSION
                prompt_version = PROMPT_VERSION if composed and getattr(task_spec, "kind", "") == "dynamic" else "fixed-v1"
                goal_phase = "proposal" if composed and getattr(task_spec, "kind", "") == "dynamic" and not task_spec.confirmed else "execution"
                self._emit(
                    "model/context",
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    step_id=step_id,
                    payload={"messages": self._persistent_messages(model_messages), "prompt_version": prompt_version, "goal_phase": goal_phase},
                )
                # 按任务类型细化模型看到的参数 schema，要求动态动作关联目标和步骤。
                model_tools = snapshot.model_tools()
                if composed and getattr(task_spec, "kind", "") == "dynamic":
                    from harness.skills.composition.contracts import dynamic_model_tools
                    model_tools = dynamic_model_tools(model_tools, task_spec)
                if composed:
                    planning_steps += 1
                reply = self.model.complete(
                    ModelRequest(
                        session_id,
                        turn_id,
                        step_id,
                        tuple(model_messages),
                        model_tools,
                        snapshot.snapshot_revision,
                        {"instruction": text, "planning_step": planning_steps if composed else planning_steps + 1,
                         "execution_mode": policy.mode,
                         "goal_kind": getattr(task_spec, "kind", "fixed"),
                         "goal_phase": goal_phase,
                         "prompt_version": prompt_version},
                    ),
                    cancel,
                )
                if not composed:
                    planning_steps += 1
                if reply.tool_argument_errors:
                    argument_repairs += 1
                    retry = argument_repairs <= 2 and planning_steps < max_steps
                    feedback = (
                        "上一轮工具调用的 arguments 不是合法 JSON 对象，整批调用均未执行。"
                        "请依据当前工具 schema 重新生成完整、合法的 JSON 参数；不要用 Markdown 包裹，"
                        "不要猜测已执行成功。解析错误："
                        + json.dumps(reply.tool_argument_errors, ensure_ascii=False)
                    )
                    self._emit(
                        "model/replied", session_id=session_id, turn_id=turn_id,
                        task_id=task_id, step_id=step_id,
                        payload={"content": "模型工具参数格式错误，正在重新生成。" if retry else "模型工具参数格式错误，纠错预算已耗尽。",
                                 "prompt_version": prompt_version, "goal_phase": goal_phase, "finish_reason": reply.finish_reason,
                                 "usage": dict(reply.usage), "tool_calls": [],
                                 "tool_argument_errors": list(reply.tool_argument_errors),
                                 "repair_attempt": argument_repairs, "retry": retry,
                                 "retry_feedback": feedback},
                    )
                    messages.append({"role": "user", "content": feedback})
                    self._emit("step/completed", session_id=session_id, turn_id=turn_id,
                               task_id=task_id, step_id=step_id,
                               payload={"next": "continue" if retry else "finish"})
                    if retry:
                        continue
                    reply_error = "模型返回的工具参数不是合法 JSON 对象；已耗尽本轮纠错预算。"
                    last_result = ToolResult("runtime_error", False, False, reply_error)
                    last_capability = None
                    response = "模型生成的工具参数格式仍不合法，已停止本轮任务；未执行这批工具调用。"
                    self._stop_all(scope, session_id=session_id, turn_id=turn_id,
                                   task_id=task_id, reason=last_result.status)
                    break
                assistant: dict[str, Any] = {"role": "assistant", "content": reply.content}
                if reply.tool_calls:
                    assistant["tool_calls"] = [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in reply.tool_calls
                    ]
                messages.append(assistant)
                self._emit(
                    "model/replied",
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    step_id=step_id,
                    payload={
                        "content": reply.content,
                        "prompt_version": prompt_version,
                        "goal_phase": goal_phase,
                        "finish_reason": reply.finish_reason,
                        "usage": dict(reply.usage),
                        "tool_calls": [
                            {"name": call.name, "arguments": dict(call.arguments), "call_id": call.call_id}
                            for call in reply.tool_calls
                        ],
                    },
                )
                if not reply.tool_calls:
                    response = reply.content.strip()
                    if not response:
                        raise RuntimeError("model returned neither content nor tools")
                    finished_by_response = True
                    self._emit(
                        "step/completed",
                        session_id=session_id,
                        turn_id=turn_id,
                        task_id=task_id,
                        step_id=step_id,
                        payload={"next": "finish"},
                    )
                    break

                if completion_capability and len(reply.tool_calls) > 1:
                    # Composition decisions must consume each action's feedback.
                    # Supply a result for every rejected call to preserve model protocol.
                    last_result = ToolResult(
                        "tool_denied", False, False,
                        "composition mode requires exactly one tool per model reply",
                    )
                    last_capability = None
                    for decision in reply.tool_calls:
                        messages.append({"role": "tool", "tool_call_id": decision.call_id,
                                         "content": json.dumps(last_result.for_model())})
                        self._emit(
                            "tool/denied", session_id=session_id, turn_id=turn_id,
                            task_id=task_id, step_id=step_id, tool_call_id=decision.call_id,
                            payload=last_result.for_model(),
                        )
                    self._emit(
                        "step/completed", session_id=session_id, turn_id=turn_id,
                        task_id=task_id, step_id=step_id, payload={"next": "continue"},
                    )
                    continue

                terminal = False
                for decision in reply.tool_calls:
                    last_capability = decision.name
                    descriptor = snapshot.capabilities.get(decision.name)
                    if tool_calls >= max_tools:
                        last_result = ToolResult(
                            "incomplete_budget_exhausted",
                            False,
                            False,
                            "tool budget exhausted",
                        )
                    else:
                        deadline = scope.deadline_monotonic
                        if descriptor is not None:
                            capability_deadline = time.monotonic() + descriptor.timeout_s
                            deadline = min(deadline, capability_deadline) if deadline else capability_deadline
                        request = ToolRequest(
                            session_id=session_id,
                            turn_id=turn_id,
                            step_id=step_id,
                            task_id=task_id,
                            tool_call_id=decision.call_id,
                            agent_id=scope.agent_id,
                            capability_id=decision.name,
                            arguments=(self.argument_policy(instruction, decision.name, decision.arguments)
                                       if self.argument_policy else decision.arguments),
                            snapshot_revision=snapshot.snapshot_revision,
                            robot_id=(next(iter(scope.robot_ids)) if len(scope.robot_ids) == 1 else None),
                            boot_epoch=self.boot_epoch_provider() if self.boot_epoch_provider else None,
                            deadline_monotonic=deadline,
                            idempotency_key=f"{turn_id}:{decision.call_id}",
                        )
                        last_result = self.tools.execute(
                            request,
                            scope=scope,
                            snapshot=snapshot,
                            cancel=cancel,
                        )
                        if last_result.status != "tool_denied" and last_result.evidence.get("execution_started", True):
                            tool_calls += 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": decision.call_id,
                            "content": json.dumps(last_result.for_model(), ensure_ascii=False),
                        }
                    )
                    terminal = bool(descriptor and descriptor.terminal)
                    if completion_capability:
                        terminal = terminal or bool(
                            decision.name == completion_capability
                            and last_result.completed
                        ) or last_result.status == "side_effect_unknown"
                    if composed and last_result.status == "goal_proposed":
                        terminal = True
                    if composed and decision.name == "compose_blocked":
                        terminal = True
                        self._stop_all(scope, session_id=session_id, turn_id=turn_id, task_id=task_id, reason="incomplete")
                    if composed and getattr(task_spec, "kind", "") == "dynamic":
                        fingerprint = (last_result.status, json.dumps(last_result.payload.get("observation", {}).get("pose")),
                                       last_result.payload.get("acquired_goal"))
                        if last_result.status in {"subgoal_verified", "action_verified"} and fingerprint != progress_fingerprint:
                            no_progress = 0
                            progress_fingerprint = fingerprint
                        else:
                            no_progress += 1
                        if no_progress >= 8 and not terminal:
                            last_result = ToolResult("incomplete_budget_exhausted", False, False, "no progress budget exhausted")
                    if composed and last_result.status in {
                        "side_effect_unknown", "risk_blocked", "cancelled", "tool_timeout",
                        "incomplete_budget_exhausted", "runtime_error",
                    }:
                        terminal = True
                        if last_result.status != "incomplete_budget_exhausted":
                            self._stop_all(scope, session_id=session_id, turn_id=turn_id,
                                           task_id=task_id, reason=last_result.status)
                    if terminal or last_result.status == "incomplete_budget_exhausted":
                        break
                self._emit(
                    "step/completed",
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    step_id=step_id,
                    payload={"next": "finish" if terminal else "continue"},
                )
                if terminal:
                    if completion_capability and not composed and last_result.status == "side_effect_unknown":
                        self._stop_all(
                            scope, session_id=session_id, turn_id=turn_id,
                            task_id=task_id, reason="side_effect_unknown",
                        )
                    response = (
                        "任务已完成并通过物理证据验证。"
                        if last_result and last_result.completed and (
                            completion_capability is None
                            or last_capability == completion_capability
                        )
                        else ("目标已生成，请核对并确认后执行。" if last_result and last_result.status == "goal_proposed"
                              else str(last_result.payload.get("blocked_reason") or f"任务未完成（{last_result.status}）。") if last_result
                              else "任务未完成。")
                    )
                    break
                if last_result and last_result.status == "incomplete_budget_exhausted":
                    break
            else:
                last_result = ToolResult(
                    "incomplete_budget_exhausted", False, False, "planning budget exhausted"
                )

            if last_result and last_result.status == "incomplete_budget_exhausted":
                self._stop_all(
                    scope,
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    reason=last_result.status,
                )
                response = str(last_result.payload.get("blocked_reason") or "预算已耗尽，任务未完成；机器人已进入安全停车流程。")

            completed = bool(
                finished_by_response
                and (last_result is None or last_result.tool_ok)
                or (
                    not finished_by_response
                    and last_result is not None
                    and last_result.completed
                )
            )
            task_status = (
                "agent_finished"
                if finished_by_response
                and (last_result is None or last_result.tool_ok)
                else last_result.status if last_result is not None else "incomplete"
            )
            if completion_capability:
                completed = bool(
                    last_capability == completion_capability
                    and last_result is not None
                    and last_result.completed
                )
                if finished_by_response or (last_result and last_result.completed and not completed):
                    task_status = "verification_failed"
                    response = "任务未完成：缺少指定终态验收工具的完成证据。" + (f" {response}" if response else "")
                    self._stop_all(
                        scope, session_id=session_id, turn_id=turn_id,
                        task_id=task_id, reason=task_status,
                    )
            outcome = TaskOutcome.COMPLETED if completed else TaskOutcome.INCOMPLETE
            self._emit(
                "turn/completed" if completed else "turn/incomplete",
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                payload={
                    "task_status": task_status,
                    "completed": completed,
                    "planning_steps": planning_steps,
                    "tool_calls": tool_calls,
                    "elapsed_s": max(0.0, time.monotonic() - started),
                    **({"error": reply_error} if reply_error else {}),
                },
            )
            return LoopResult(
                session_id,
                turn_id,
                task_id,
                outcome,
                task_status,
                completed,
                response,
                planning_steps,
                tool_calls,
                last_result,
                error=reply_error,
            )
        except (CancelledError, TimeoutError) as exc:
            status = "cancelled" if isinstance(exc, CancelledError) else "navigation_timeout"
            self._stop_all(
                scope,
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                reason=status,
            )
            self._emit(
                "turn/incomplete",
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                payload={"task_status": status, "completed": False, "error": str(exc)},
            )
            return LoopResult(
                session_id,
                turn_id,
                task_id,
                TaskOutcome.INCOMPLETE,
                status,
                False,
                "任务已取消并进入安全停车流程。" if status == "cancelled" else "任务超时并进入安全停车流程。",
                planning_steps,
                tool_calls,
                last_result,
            )
        except Exception as exc:
            self._stop_all(
                scope,
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                reason="runtime_error",
            )
            try:
                self._emit(
                    "turn/incomplete",
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    payload={"task_status": "runtime_error", "completed": False, "error": str(exc)[:1_000]},
                )
            except Exception:
                pass
            return LoopResult(
                session_id,
                turn_id,
                task_id,
                TaskOutcome.INCOMPLETE,
                "runtime_error",
                False,
                "任务因运行时错误未完成；已执行安全停车。",
                planning_steps,
                tool_calls,
                last_result,
                error=str(exc)[:1_000],
            )
