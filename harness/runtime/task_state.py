"""Durable task projection. Progress is evidence, never a second scheduler."""
from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class ComposedTask:
    task_key: str
    instruction: str
    destination: tuple[float, float, float]
    pickup: tuple[float, float, float] | None = None
    entity_id: str | None = None
    release: bool = False
    position_tolerance_m: float = 0.15
    yaw_tolerance_rad: float = 0.15

    def __post_init__(self):
        for name in ("destination", "pickup"):
            pose = getattr(self, name)
            if pose is None and name == "pickup":
                continue
            if (not isinstance(pose, (tuple, list)) or len(pose) != 3
                    or any(isinstance(x, bool) or not isinstance(x, (float, int))
                           or not math.isfinite(x) for x in pose)):
                raise ValueError(f"{name} must be finite x/y/yaw")
            object.__setattr__(self, name, tuple(float(x) for x in pose))
        if not isinstance(self.task_key, str) or not self.task_key.strip():
            raise ValueError("task_key is required")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("fixed task instruction is required")
        if self.entity_id is not None and (not isinstance(self.entity_id, str) or not self.entity_id.strip()):
            raise ValueError("entity_id must be a nonempty string")
        if bool(self.entity_id) != (self.pickup is not None):
            raise ValueError("transport requires both entity_id and pickup")
        if not isinstance(self.release, bool) or (self.release and not self.entity_id):
            raise ValueError("release requires a transport task")
        for value in (self.position_tolerance_m, self.yaw_tolerance_rad):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 0.3:
                raise ValueError("task tolerances must be in (0, 0.3]")

    def payload(self):
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_payload(cls, payload):
        return cls(**payload)


# 按任务重放持久事件，恢复计划、步骤回执、尝试次数和用户目标证据。
# 这是状态投影，不会执行动作；重规划清空旧步骤回执，防止下标相同而误用许可。
def task_projection(store, session_id, task_id):
    """Rebuild from events, including after history compaction or process restart."""
    state = {"task_id": task_id, "execution_mode": "terminal", "plan": [],
             "plan_revision": 0, "step_evidence": {}, "object_bindings": {}, "attempts": {}, "steps": [], "subgoal_progress": {}, "completed": False}
    active_actions = {}
    stream = store.iter_task_events(session_id, task_id) if hasattr(store, "iter_task_events") else store.iter_events(session_id)
    for event in stream:
        if event.task_id != task_id:
            continue
        payload = dict(event.payload)
        if event.event_type == "task/accepted":
            state.update(payload)
            if payload.get("goal", {}).get("resume_state"):
                state.update(payload["goal"]["resume_state"].get("projection", {}))
        elif event.event_type == "task/paused":
            state["paused_goal"] = payload["paused_goal"]
            state["paused"] = True
        elif event.event_type == "task/goal_proposed":
            state["proposed_goal"] = payload["proposed_goal"]
        elif event.event_type == "task/blocked":
            state["blocked_reason"] = payload["reason"]
        elif event.event_type == "task/facts":
            state.update({k: payload[k] for k in ("goal_evidence", "goal_status") if k in payload})
        elif event.event_type == "task/object_localized":
            state['object_bindings'][payload['goal_id']] = payload
        elif event.event_type == "task/object_invalidated":
            if payload['goal_id'] in state['object_bindings']:
                state['object_bindings'][payload['goal_id']]['invalidated'] = True
        elif event.event_type == "task/plan":
            state["plan"] = payload["subgoals"]
            state["plan_revision"] += 1
            state["subgoal_progress"] = {}
            state["step_evidence"] = {}
            state["plan_schema_version"] = payload.get("plan_schema_version", 1)
        elif event.event_type == "task/step_verified":
            if payload.get("plan_revision") == state["plan_revision"]:
                state["step_evidence"][str(payload["subgoal_index"])] = payload
        elif event.event_type == "task/attempt":
            key = payload["action_key"]
            state["attempts"][key] = state["attempts"].get(key, 0) + 1
            active_actions[event.tool_call_id] = payload
            state["subgoal_progress"][str(payload["subgoal_index"])] = {
                "status": "executing", "action_key": key, "plan_revision": payload["plan_revision"]}
        elif event.event_type == "tool/result":
            if "held_entity" in payload:
                state["held_entity"] = payload["held_entity"]
            action = active_actions.get(event.tool_call_id)
            if action and action["plan_revision"] == state["plan_revision"]:
                state["subgoal_progress"][str(action["subgoal_index"])] = {
                    "status": "step_verified" if payload.get("step_status", {}).get(str(action["subgoal_index"])) is True else "action_verified" if payload.get("action_verified", payload.get("subgoal_verified")) is True else "action_unverified",
                    "action_key": action["action_key"], "evidence_event_id": event.event_id,
                    "plan_revision": action["plan_revision"],
                }
            state["steps"].append({"tool_call_id": event.tool_call_id,
                                   "event_id": event.event_id, "result": payload})
        elif event.event_type in {"turn/completed", "turn/incomplete", "turn/interrupted"}:
            state["completed"] = payload.get("completed") is True and event.event_type == "turn/completed"
            state["task_status"] = payload.get("task_status", "interrupted")
        elif event.event_type == "tool/reconciled":
            state["completed"] = False
            state["task_status"] = "side_effect_unknown"
    return state


# 为下一次模型推理生成有界摘要；完整证据留在事件库，摘要不等于当前物理观测。
def context_summary(store, session_id, task_id):
    state = task_projection(store, session_id, task_id)
    # Full evidence stays durable; do not put an unbounded event stream in prompts.
    state["steps"] = state["steps"][-4:]
    return {"role": "user", "content": json.dumps(
        {"composed_task_state": state, "note": "计划不等于执行证据；历史观察不代表当前物理状态。"},
        ensure_ascii=False)}
