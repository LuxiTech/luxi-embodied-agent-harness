"""The single operator-facing lifecycle for the shared Agent Loop."""
from __future__ import annotations

import threading
from typing import Any, Callable

from .contracts import CancellationToken, new_id
from .execution_policy import ComposedPolicy, validate_execution_mode
from .task_state import task_projection
from .task_policy import reject_multi_robot_instruction


class AgentRuntimeService:
    """Submit, cancel and reset one session without owning planning or tools."""

    def __init__(self, loop, scope, store, *, model="", model_provider="",
                 configuration_error: Callable[[], str | None] = lambda: None,
                 on_cancel: Callable[[], None] | None = None,
                 on_close: Callable[[], None] | None = None,
                 composed_tasks=None, composed_scope=None, composed_reference_provider=None):
        self.composed_reference_provider = composed_reference_provider
        self.composed_tasks = dict(composed_tasks or {})
        self.composed_scope = composed_scope
        self._pause_requested = False
        self._execution_mode = "terminal"
        self._active_task_id = None
        self.loop = loop
        self.scope = scope
        self.store = store
        self.model = model
        self.model_provider = model_provider
        self.configuration_error = configuration_error
        self._on_cancel = on_cancel
        self._on_close = on_close
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._cancel: CancellationToken | None = None
        self._result: Any = None
        self._error = ""
        self._closed = False
        self._recovered = False
        for event in store.iter_events(scope.session_id):
            if event.event_type == "session/context_reset":
                self._active_task_id = None
                self._execution_mode = "terminal"
            elif event.event_type == "task/accepted":
                self._active_task_id = event.task_id
                self._execution_mode = event.payload.get("execution_mode", "terminal")

    # 指令入口：交给带控制权交接的提交路径；此处不会直接执行机器人技能。
    def submit(self, instruction: str, *, execution_mode="terminal", task_key=None) -> tuple[bool, str]:
        return self.submit_with_handoff(instruction, lambda: True,
                                        execution_mode=execution_mode, task_key=task_key)

    # 区分首次提交、目标确认和暂停恢复；校验原指令、单次令牌及控制权。
    # 校验通过后启动 _run 线程，HTTP 请求无需等待整个物理任务完成。
    def submit_with_handoff(self, instruction: str, handoff: Callable[[], bool], *,
                            execution_mode="terminal", task_key=None):
        try:
            validate_execution_mode(execution_mode)
        except ValueError as exc:
            return False, str(exc)
        task = None
        if execution_mode == "composed":
            if self.composed_scope is None or not (self.composed_tasks or self.composed_reference_provider):
                return False, "当前后端尚未开放已验收的组合能力"
            if isinstance(task_key, str) and task_key in self.composed_tasks:
                task = self.composed_tasks[task_key]
            elif task_key is not None:
                from .composition_goals import DynamicTask
                proposal = self.status().get("proposed_goal") or self.status().get("paused_goal")
                if not proposal or proposal["task_key"] != task_key:
                    return False, "目标提议已失效，请重新输入任务"
                task = DynamicTask.from_payload(proposal)
            elif self.composed_reference_provider is None:
                return False, "请选择已配置的组合验收任务"
            if task is not None and instruction.strip() != task.instruction:
                return False, "指令与所选组合任务目标不一致，请重新选择任务"
        elif task_key is not None:
            return False, "terminal 模式不能指定组合任务"
        text = instruction.strip()
        if not text or len(text) > 4000:
            return False, "指令不能为空且不能超过 4000 个字符"
        rejection = reject_multi_robot_instruction(text)
        if rejection:
            return False, rejection
        with self._lock:
            if self._closed:
                return False, "Agent 服务已关闭"
            if self._thread is not None and self._thread.is_alive():
                return False, "agent 正在执行上一条指令"
            error = self.configuration_error()
            if error:
                return False, error
            try:
                self.recover_interrupted()
                if not handoff():
                    return False, "安全控制权交接失败"
            except Exception as exc:
                return False, f"安全控制权交接失败：{exc}"
            if task is not None and getattr(task, "kind", "") == "dynamic":
                consumed = any(e.event_type == "task/goal_confirmed" and e.payload.get("task_key") == task.task_key
                               for e in self.store.iter_events(self.scope.session_id))
                if consumed:
                    return False, "该目标已经执行过，请提交新任务"
                self.store.emit("task/goal_confirmed", session_id=self.scope.session_id, source="agent-service",
                                payload={"task_key": task.task_key, "instruction": text})
            self._pause_requested = False
            self._execution_mode = execution_mode
            self._active_task_id = new_id("task") if execution_mode == "composed" else None
            self._cancel = CancellationToken()
            self._result = None
            self._error = ""
            self._thread = threading.Thread(target=self._run, args=(text, self._cancel, execution_mode, task),
                                            name="luxi-agent-loop", daemon=True)
            self._thread.start()
        return True, "指令已交给 LuxiAgent Harness"

    def recover_interrupted(self):
        """Stop unresolved work before accepting a new task, never replay it."""
        with self._lock:
            if self._recovered:
                return []
            recovered = []
            for session_id in self.store.session_ids():
                opened = {}
                for event in self.store.iter_events(session_id):
                    if event.event_type == "tool/started":
                        opened[event.tool_call_id] = event
                    elif event.event_type in {"tool/result", "tool/reconciled"}:
                        opened.pop(event.tool_call_id, None)
                for call_id, event in opened.items():
                    for robot_id in self.scope.robot_ids:
                        evidence = self.loop.safety.stop(robot_id, "interrupted_tool_recovery")
                        if not evidence.stationary_confirmed:
                            raise RuntimeError("中断任务停止状态未确认，禁止恢复运动")
                    self.store.emit("tool/reconciled", session_id=session_id,
                        turn_id=event.turn_id, task_id=event.task_id, tool_call_id=call_id,
                        source="agent-service", payload={"completed": False,
                        "task_status": "side_effect_unknown", "stationary_confirmed": True,
                        "automatic_tool_replay": False}, idempotency_key=f"reconciled:{call_id}")
                    recovered.append(call_id)
                self.store.mark_interrupted_turns(session_id)
            self._recovered = True
            return recovered

    # 首次提交：将配置位置、start 和指令中的显式坐标合并为 DynamicTask。
    # 确认/恢复：沿用已确认目标，扣除已耗预算；恢复还需核对停车、epoch 和持物状态。
    # 最后进入同一个 Loop，指定 compose_verify 为组合任务的整体完成入口。
    def _run(self, text, cancel, execution_mode="terminal", task=None):
        try:
            if execution_mode == "composed":
                from dataclasses import replace
                import time
                scope = replace(self.composed_scope, deadline_monotonic=time.monotonic() + 600)
                if task is None:
                    from .composition_goals import DynamicTask, instruction_references
                    catalog = self.composed_reference_provider()
                    refs = dict(catalog["references"])
                    for key, pose in instruction_references(text).items():
                        if key in refs and tuple(refs[key]) != pose:
                            raise ValueError(f"位置 {key} 与已有引用冲突，请使用不同名称")
                        refs[key] = pose
                    task = DynamicTask(self._active_task_id, text, references=refs,
                                       world_revision=catalog["world_revision"],
                                       visual_regions=catalog.get("visual_regions", {}),
                                       placement_surfaces=catalog.get("placement_surfaces", {}),
                                       supported_entities=catalog.get("supported_entities", ()),
                                       entity_catalog=catalog.get("entity_catalog", {}),
                                       reference_metadata=catalog.get("reference_metadata", {}),
                                       schema_version=5 if catalog.get("placement_surfaces") else 4 if catalog.get("visual_regions") else 3)
                if getattr(task, "kind", "") == "dynamic" and task.confirmed:
                    scope = replace(scope, budget_steps=max(0, scope.budget_steps-task.preparation_steps),
                                    budget_tools=max(0, scope.budget_tools-task.preparation_tools),
                                    deadline_monotonic=time.monotonic()+max(0, 600-task.preparation_seconds))
                if getattr(task, "resume_state", None):
                    from harness.skills.composed_tasks import _fresh
                    for robot_id in scope.robot_ids:
                        if not self.loop.safety.stop(robot_id, "composition_resume").stationary_confirmed:
                            raise RuntimeError("恢复前未确认停车")
                    observation = self.loop.composed_observe()
                    if (not _fresh(observation) or observation.get("world_revision", "") != task.world_revision
                            or observation.get("attached_entity") != task.resume_state["projection"].get("held_entity")):
                        raise RuntimeError("暂停后的物理状态已变化，禁止自动恢复；请核对并提交新任务")
                    scope = replace(scope, budget_steps=task.resume_state["remaining_steps"],
                                    budget_tools=task.resume_state["remaining_tools"],
                                    deadline_monotonic=time.monotonic()+task.resume_state["remaining_seconds"])
                result = self.loop.run(text, scope=scope, cancel=cancel,
                                       execution_policy=ComposedPolicy("compose_verify"), task_spec=task,
                                       task_id=self._active_task_id)
            else:
                result = self.loop.run(text, scope=self.scope, cancel=cancel)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)[:1000]
        else:
            with self._lock:
                self._result = result
                self._error = getattr(result, "error", "")
                if (self._pause_requested and execution_mode == "composed" and getattr(task, "confirmed", False)
                        and result.task_status == "cancelled"):
                    import time
                    from dataclasses import replace
                    state = task_projection(self.store, self.scope.session_id, result.task_id)
                    remaining_steps = scope.budget_steps-result.planning_steps
                    remaining_tools = scope.budget_tools-result.tool_calls
                    remaining_seconds = max(0, scope.deadline_monotonic-time.monotonic())
                    if remaining_steps > 0 and remaining_tools > 0 and remaining_seconds > 0:
                        projection = {k: state.get(k, {} if k != "held_entity" else None)
                                      for k in ("attempts", "goal_evidence", "held_entity")}
                        # Keep acquisition evidence, not executable old plans or unfinished actions.
                        projection["steps"] = [step for step in state["steps"] if step["result"].get("acquired_goal")]
                        paused = replace(task, task_key=new_id("resume"), resume_state={
                            "projection": projection, "remaining_steps": remaining_steps,
                            "remaining_tools": remaining_tools, "remaining_seconds": remaining_seconds,
                            "consumed_steps": task.resume_state.get("consumed_steps", 0)+result.planning_steps,
                            "consumed_tools": task.resume_state.get("consumed_tools", 0)+result.tool_calls})
                        self.store.emit("task/paused", session_id=self.scope.session_id, task_id=result.task_id,
                                        source="agent-service", payload={"paused_goal": paused.payload()})

    def pause(self):
        with self._lock:
            if self._execution_mode != "composed" or self._thread is None or not self._thread.is_alive():
                return False, "当前没有可暂停的组合任务"
            self._pause_requested = True
        self.cancel()
        return True, "正在停车并保存进度；恢复时将重新观察和规划"

    def cancel(self):
        with self._lock:
            cancel = self._cancel
        if cancel is not None:
            cancel.cancel()
        if self._on_cancel is not None:
            self._on_cancel()

    def reset_session(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("agent 仍在结束上一轮，无法清空会话")
            self.store.emit("session/context_reset", session_id=self.scope.session_id,
                            source="agent-service", payload={"automatic_tool_replay": False})
            self._result = None
            self._active_task_id = None
            self._error = ""

    def status(self):
        with self._lock:
            busy = self._thread is not None and self._thread.is_alive()
            result = self._result
            error = self._error
            closed = self._closed
        configuration_error = self.configuration_error()
        progress_task_id = self._active_task_id or getattr(result, "task_id", None)
        progress = task_projection(self.store, self.scope.session_id, progress_task_id) if progress_task_id else {}
        if progress:
            progress["steps"] = progress["steps"][-4:]
        goal = progress.get("goal", {})
        resumed = goal.get("resume_state", {})
        totals = {"planning_steps": goal.get("preparation_steps", 0)+resumed.get("consumed_steps", 0)+(result.planning_steps if result else 0),
                  "tool_calls": goal.get("preparation_tools", 0)+resumed.get("consumed_tools", 0)+(result.tool_calls if result else 0)}
        return {
            "task_totals": totals,
            "runtime": "luxi-agent-harness", "provider": "harness",
            "provider_label": "LuxiAgent Harness", "model_provider": self.model_provider,
            "model": self.model, "session_id": self.scope.session_id,
            "robot_id": next(iter(self.scope.robot_ids)) if len(self.scope.robot_ids) == 1 else None,
            "available": not closed and configuration_error is None, "busy": busy,
            "last_response": result.response if result else "", "last_error": error or configuration_error or "",
            "last_task_result": {"task_status": result.task_status, "completed": result.completed,
                                 "planning_steps": result.planning_steps, "tool_calls": result.tool_calls} if result else {},
            "execution_mode": self._execution_mode,
            "execution_modes": ["terminal", "composed"] if self.composed_scope is not None and (self.composed_tasks or self.composed_reference_provider) else ["terminal"],
            "dynamic_composition": self.composed_reference_provider is not None,
            "paused_goal": progress.get("paused_goal"),
            "proposed_goal": progress.get("proposed_goal") if progress.get("task_status") == "goal_proposed" else None,
            "composed_tasks": [{"task_key": key, "instruction": task.instruction}
                               for key, task in self.composed_tasks.items()],
            "task_progress": progress,
            "agent_loop": {"execution_owner": "LuxiAgentLoop"},
        }

    def close(self):
        with self._lock:
            self._closed = True
            thread = self._thread
        self.cancel()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise RuntimeError("Agent tool is still stopping; resources have not been closed")
        if self._on_close is not None:
            self._on_close()
