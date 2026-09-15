"""Evidence-driven composition skills on one registered robot backend.

Goals come from the accepted task, never from model-supplied success conditions.
The backend owns motion and attachment; these adapters own task-level checks.
"""
from dataclasses import dataclass
import math
import time
from typing import Protocol

from harness.runtime.contracts import CapabilityDescriptor, SideEffect, ToolResult
from harness.runtime.task_state import ComposedTask, task_projection


COMPOSED_PROMPT = """你在唯一 LuxiAgentLoop 中执行组合任务，只使用当前提供的工具。
任务状态中的 goal 是执行前固定的用户目标，不得修改、降低或增加最终条件。
先 compose_plan 声明子目标；动作的 subgoal_index 对应当前计划下标（从 0 开始）。
每次只调用一个工具，根据反馈决定下一步；局部动作验证不代替整个语义子目标或最终条件验证。
compose_navigate 只到位，compose_face 满足指定朝向；附着前需要两者条件满足。
第一版取物为 sim_attachment 到位吸附，不是真实抓取。携带时必须保持同一对象。
失败后先读当前状态；新尝试必须说明恢复理由且次数有限。未知副作用、风险和取消必须结束。
计划文字和子技能成功不能完成任务；最后调用 compose_verify 检查整体终态。
"""


class CompositionBackend(Protocol):
    backend: str
    interaction_model: str

    def observe(self) -> dict: ...
    def navigate(self, pose, cancel, deadline) -> dict:
        """Navigate once; pose yaw=None leaves terminal heading unconstrained."""
        ...
    def attach(self, entity_id, pickup, cancel, deadline) -> dict: ...
    def release(self, entity_id, cancel, deadline) -> dict: ...
    def place(self, entity_id, surface, cancel, deadline) -> dict: ...


def _schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {},
            "required": list(required), "additionalProperties": False}


def _fresh(observation, after=None):
    stamp = observation.get("timestamp_monotonic")
    pose = observation.get("pose")
    return (isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
            and math.isfinite(stamp) and 0 <= time.monotonic() - stamp <= 1.0
            and (after is None or stamp > after)
            and isinstance(pose, (list, tuple)) and len(pose) == 3
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in pose))


def _at(observation, pose, task, *, yaw=True):
    actual = observation["pose"]
    return (math.hypot(actual[0] - pose[0], actual[1] - pose[1]) <= task.position_tolerance_m
            and (not yaw or abs(math.atan2(math.sin(actual[2] - pose[2]),
                                           math.cos(actual[2] - pose[2]))) <= task.yaw_tolerance_rad))


@dataclass
class ComposedSkills:
    backend: CompositionBackend
    events: object
    safety: object
    max_attempts: int = 2

    # 注册组合技能名称、参数 schema、执行模式和超时，供工具准入及模型工具列表使用。
    def descriptors(self):
        from .composition.plan_steps import COMPLETION_SCHEMA
        reason = {"type": "string", "maxLength": 500}
        tools = {
            "compose_plan": ("记录或修订子目标计划；动态任务记录计划后开放运动工具，不会执行或证明子目标成功。", _schema({
                "subgoals": {"type": "array", "minItems": 1, "maxItems": 12,
                             "items": {"type": "string", "minLength": 1, "maxLength": 200}},
                "reason": reason}, ("subgoals",))),
            "compose_locate": ("在已标注地点用新鲜 RGB-D 定位 water_bottle，生成取物位姿；不附着、不完成持物目标。",
                               _schema({"recovery_reason": reason})),
            "compose_observe": ("读取当前机器人位置和附着状态。", _schema()),
            "compose_navigate": ("到任务给定的位置，保持当前朝向；携物时检查附着。", _schema({
                "target": {"type": "string", "enum": ["pickup", "destination"]},
                "recovery_reason": reason}, ("target",))),
            "compose_face": ("在已到达任务位置后转向该位置要求的朝向。", _schema({
                "target": {"type": "string", "enum": ["pickup", "destination"]},
                "recovery_reason": reason}, ("target",))),
            "compose_attach": ("到指定取物位置并停稳后附着对象；这是仿真吸附替代，不是真实抓取。",
                               _schema({"recovery_reason": reason})),
            "compose_place": ("到已标注桌前并停稳后，将持有对象仿真放置到固定桌面点、解除附着并验证；只引用目标，不输入坐标。",
                              _schema({"recovery_reason": reason})),
            "compose_release": ("在交付位置解除附着，仅适用于明确要求放下的任务。",
                                _schema({"recovery_reason": reason})),
            "compose_verify": ("停车后用新鲜证据检查固定任务的全部最终条件；唯一整体完成入口。", _schema()),
        }
        tools["compose_propose_goal"] = ("提出自然语言目标的完整验收条件，等待用户确认；不执行运动。", _schema({
            "conditions": {"type": "array", "minItems": 1, "maxItems": 24, "items": {
                "type": "object", "properties": {
                    "id": {"type": "string"}, "predicate": {"type": "string", "enum": ["visited", "at", "holding", "released", "acquired", "placed_on"]},
                    "require_heading": {"type": "boolean", "description": "位置目标仅在用户要求指定朝向时为 true；取放操作必须为 true。"},
                    "target": {"type": "string"}, "entity_id": {"type": "string", "enum": list(getattr(self.backend, "supported_entities", ("water_bottle",)))},
                    "target_source": {"type": "string", "enum": ["reference", "visual"], "description": "已标注厨房中的取物 holding 必须 visual；厨房是搜索地点，water_bottle 是对象。"},
                    "depends_on": {"type": "array", "items": {"type": "string"}}},
                "required": ["id", "predicate", "target", "depends_on"], "additionalProperties": False}}}, ("conditions",)))
        tools["compose_blocked"] = ("缺能力、关键信息或无法推进时结束任务，说明具体阻塞。",
                                     _schema({"reason": reason}, ("reason",)))
        # Fixed fixtures retain string plans; dynamic plans are validated structurally at execution.
        tools["compose_plan"][1]["properties"]["subgoals"]["items"] = {"anyOf": [
            {"type": "string"}, {"type": "object", "properties": {
                "completion": COMPLETION_SCHEMA,
                "label": {"type": "string"}, "goal_ids": {"type": "array", "items": {"type": "string"}},
                "depends_on": {"type": "array", "items": {"type": "integer"}}},
                "required": ["label", "goal_ids", "depends_on"], "additionalProperties": False}]}
        tools["compose_plan"][1]["properties"]["subgoals"]["maxItems"] = 24
        for name, (description, schema) in tools.items():
            physical = name in {"compose_navigate", "compose_face", "compose_attach", "compose_release", "compose_place", "compose_locate"}
            if physical:
                schema["properties"]["subgoal_index"] = {"type": "integer", "minimum": 0, "maximum": 23}
                schema["properties"]["goal_id"] = {"type": "string"}
                if "target" in schema["properties"]:
                    schema["properties"]["target"] = {"type": "string"}
            yield CapabilityDescriptor(
                name, "v1", "composed-skills", schema, description=description,
                backends=frozenset({self.backend.backend}),
                side_effect=SideEffect.PHYSICAL if physical else SideEffect.READ_ONLY,
                resources=frozenset({f"robot-motion:{self.backend.backend}"}),
                exclusive=physical, timeout_s=120 if physical else 45 if name == "compose_verify" else 10,
                verifier="task_verified" if name == "compose_verify" else None,
                execution_modes=frozenset({"composed"}),
                composition_contract={"version": "1", "world_frame": "backend map; epoch scoped",
                    "preconditions": (["confirmed goal", "fresh observation", "plan dependencies", "runtime safety admission"] if physical else []),
                    "invariants": ["same attached entity during motion"] if name in {"compose_navigate", "compose_face"} else [],
                    "effects": {"compose_navigate": "robot at target xy", "compose_face": "robot at target yaw",
                                "compose_place": "simulated placement on annotated surface verified",
                                "compose_attach": "simulated attachment acquired", "compose_release": "attachment removed",
                                "compose_verify": "all confirmed predicates verified"}.get(name, "no physical motion"),
                    "retry": "never automatically; bounded new attempt requires reason",
                    "verification": "fresh backend evidence; tool_ok is not task completion"},
            )

    def _emit(self, kind, request, payload):
        self.events.emit(kind, session_id=request.session_id, turn_id=request.turn_id,
                         task_id=request.task_id, step_id=request.step_id,
                         tool_call_id=request.tool_call_id, source="composed-skills",
                         payload=payload)

    def _result(self, status, *, ok=True, completed=False, evidence=None, payload=None):
        return ToolResult(status, ok, completed, evidence=evidence or {}, payload={
            "interaction_model": self.backend.interaction_model, **(payload or {})})

    # 从事件恢复任务状态；kind=dynamic 转入 composition/dynamic.py，固定样例走下方旧逻辑。
    def execute(self, request, cancel):
        cancel.raise_if_cancelled()
        state = task_projection(self.events, request.session_id, request.task_id)
        if state.get("execution_mode") != "composed":
            return self._result("tool_denied", ok=False)
        if state["goal"].get("kind") == "dynamic":
            from harness.skills.composition.dynamic import execute
            return execute(self, request, cancel, state)
        if request.capability_id in {"compose_propose_goal", "compose_blocked", "compose_locate", "compose_place"}:
            return self._result("tool_denied", ok=False)
        task = ComposedTask.from_payload(state["goal"])
        name, args = request.capability_id, request.arguments
        if name == "compose_plan":
            if state["plan_revision"] and not args.get("reason", "").strip():
                return self._result("invalid_input", ok=False)
            nodes = args.get("subgoals")
            if not isinstance(nodes, list) or not 1 <= len(nodes) <= 12 or any(not isinstance(n, str) or not 1 <= len(n.strip()) <= 200 for n in nodes):
                return self._result("invalid_input", ok=False)
            self._emit("task/plan", request, dict(args))
            return self._result("plan_recorded", payload={"plan_revision": state["plan_revision"] + 1})
        observation = self.backend.observe()
        if not _fresh(observation):
            return self._result("observation_unavailable", ok=False)
        if name == "compose_observe":
            self._emit("task/observed", request, {"timestamp_monotonic": observation["timestamp_monotonic"]})
            return self._result("observation_available", payload={"observation": observation})
        if name == "compose_verify":
            stop = self.safety.stop(request.robot_id, "composition_verification")
            if not stop.stationary_confirmed or stop.stationary_confirmed_at is None:
                return self._result("verification_failed", ok=False)
            cancel.raise_if_cancelled()
            observation = self.backend.observe()
            valid = _fresh(observation, stop.stationary_confirmed_at)
            valid = valid and _at(observation, task.destination, task)
            if task.entity_id:
                if task.release:
                    # Backend verifies object location/settling; robot arrival alone is insufficient.
                    valid = valid and observation.get("attached_entity") is None
                    valid = valid and observation.get("released_entity") == task.entity_id
                    object_xy = observation.get("object_xy")
                    valid = valid and observation.get("object_stationary") is True
                    valid = valid and isinstance(object_xy, (list, tuple)) and len(object_xy) == 2
                    if valid:
                        valid = all(isinstance(x, (float, int)) and not isinstance(x, bool)
                                    and math.isfinite(x) for x in object_xy)
                        valid = valid and math.dist(object_xy, task.destination[:2]) <= 1.0
                else:
                    valid = valid and observation.get("attached_entity") == task.entity_id
            return self._result("task_verified" if valid else "verification_failed", completed=bool(valid),
                                evidence={"task_verified": bool(valid), "stationary_confirmed": True,
                                          "stationary_confirmed_at": stop.stationary_confirmed_at,
                                          "verification_frame_timestamp": observation.get("timestamp_monotonic")},
                                payload={"observation": observation, "goal": task.payload()})
        if not state["plan"]:
            return self._result("tool_denied", ok=False, payload={"reason": "先记录子目标计划"})
        subgoal_index = args.get("subgoal_index", 0)
        if subgoal_index >= len(state["plan"]):
            return self._result("invalid_input", ok=False)
        target_name = args.get("target", "pickup" if name == "compose_attach" else "destination")
        if target_name not in {"pickup", "destination"}:
            return self._result("invalid_input", ok=False)
        target = task.pickup if target_name == "pickup" else task.destination
        if target is None or (name in {"compose_attach", "compose_release"} and not task.entity_id):
            return self._result("invalid_input", ok=False)
        attached = observation.get("attached_entity")
        if attached is not None and attached != task.entity_id:
            return self._result("verification_failed", ok=False)
        # Once this task has attached, subsequent motion must retain that identity.
        had_attachment = any(step["result"].get("attachment_acquired") is True
                             for step in state["steps"])
        released = any(step["result"].get("attachment_released") is True
                       for step in state["steps"])
        if had_attachment and not released and attached != task.entity_id:
            return self._result("side_effect_unknown", ok=False)
        key = f"{name}:{target_name}:{task.entity_id or '-'}"
        attempts = state["attempts"].get(key, 0)
        if attempts >= self.max_attempts:
            return self._result("incomplete_budget_exhausted", ok=False)
        if attempts and not args.get("recovery_reason", "").strip():
            return self._result("tool_denied", ok=False, payload={"reason": "新尝试需恢复理由和当前观察"})
        if name == "compose_face" and not _at(observation, target, task, yaw=False):
            return self._result("verification_failed", ok=False)
        if name == "compose_attach" and not any(
            step["result"].get("navigation_target") == "pickup"
            and step["result"].get("subgoal_verified") is True for step in state["steps"]
        ):
            return self._result("tool_denied", ok=False, payload={"reason": "需要先完成取物位置导航并取得到达证据"})
        if name == "compose_attach" and observation.get("entity_available") is False:
            return self._result("target_not_found", ok=False)
        if name in {"compose_attach", "compose_release"}:
            stop = self.safety.stop(request.robot_id, "composition_handoff")
            observation = self.backend.observe()
            if (not stop.stationary_confirmed or stop.stationary_confirmed_at is None
                    or not _fresh(observation, stop.stationary_confirmed_at)
                    or not _at(observation, target, task)):
                return self._result("verification_failed", ok=False)
            attached = observation.get("attached_entity")
            if name == "compose_attach" and attached is not None:
                return self._result("tool_denied", ok=False)
            if name == "compose_release" and (not task.release or attached != task.entity_id):
                return self._result("tool_denied", ok=False)
        cancel.raise_if_cancelled()
        self._emit("task/attempt", request, {"action_key": key,
                   "recovery_reason": args.get("recovery_reason", ""),
                   "observation_timestamp": observation["timestamp_monotonic"],
                   "plan_revision": state["plan_revision"], "subgoal_index": subgoal_index})
        if name in {"compose_navigate", "compose_face"}:
            pose = (target[0], target[1], observation["pose"][2]) if name == "compose_navigate" else target
            result = self.backend.navigate(pose, cancel, request.deadline_monotonic)
        elif name == "compose_attach":
            result = self.backend.attach(task.entity_id, task.pickup, cancel, request.deadline_monotonic)
        elif name == "compose_release":
            result = self.backend.release(task.entity_id, cancel, request.deadline_monotonic)
        else:
            return self._result("tool_denied", ok=False)
        cancel.raise_if_cancelled()
        if not result.get("operation_ok"):
            return self._result(result.get("task_status", "verification_failed"), ok=False)
        after = self.backend.observe()
        valid = _fresh(after) and after["timestamp_monotonic"] > observation["timestamp_monotonic"]
        if name in {"compose_navigate", "compose_face"}:
            valid = valid and result.get("planner_goal_reached") is True and _at(after, pose, task)
            valid = valid and (attached is None or after.get("attached_entity") == attached)
        elif name == "compose_attach":
            valid = valid and after.get("attached_entity") == task.entity_id
        else:
            valid = valid and after.get("attached_entity") is None and after.get("released_entity") == task.entity_id
        if attached is not None and (not _fresh(after) or after.get("attached_entity") != attached) and name in {"compose_navigate", "compose_face"}:
            return self._result("side_effect_unknown", ok=False)
        if not valid and name in {"compose_attach", "compose_release"}:
            return self._result("side_effect_unknown", ok=False)
        return self._result("subgoal_verified" if valid else "verification_failed", payload={
            "subgoal_verified": bool(valid), "observation": after,
            "navigation_target": target_name if name == "compose_navigate" and valid else None,
            "attachment_acquired": bool(valid and name == "compose_attach"),
            "attachment_released": bool(valid and name == "compose_release"),
        })
