"""Extracted shared implementation; independent of Agent entry points."""

from __future__ import annotations
import json
from typing import Any

from harness.skills.tool_results import (
    APPROACH_PERSON_TIMEOUT,
    APPROACH_VISUAL_TARGET_TIMEOUT,
    FETCH_OBJECT_PROCESS_TIMEOUT,
    FOLLOW_PERSON_TIMEOUT,
    NATIVE_VLM_TIMEOUT,
    OBJECT_SEARCH_TIMEOUT,
    _closed_loop_mcp_result,
    _fetch_object_mcp_result,
    _isaac_navigation_mcp_result,
    _structured_mcp_payload,
    _validate_visual_approach_standoff,
)


class G1TaskSkills:
    """Shared terminal task adapters, preserving native skill contracts."""

    def _navigate_with_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.backend in {"isaac-g1", "mujoco"}:
            guard = self._motion_guard()
            if guard:
                return {
                    "ok": False,
                    "task_status": "precondition_failed",
                    "completed": False,
                    "error": guard,
                }
            if set(arguments) - {"query", "standoff_distance", "timeout"}:
                return {
                    "ok": False,
                    "task_status": "invalid_input",
                    "completed": False,
                    "error": "navigate_with_text 收到未知参数",
                }
            query = arguments.get("query")
            if (
                not isinstance(query, str)
                or not query.strip()
                or len(query.strip()) > 200
            ):
                return {
                    "ok": False,
                    "task_status": "invalid_input",
                    "completed": False,
                    "error": "query 必须是 1 到 200 字符",
                }
            try:
                distance = self._number(
                    {
                        "standoff_distance": arguments.get(
                            "standoff_distance", 0.9
                        )
                    },
                    "standoff_distance",
                    0.5,
                    3.0,
                )
                timeout = self._number(
                    {"timeout": arguments.get("timeout", 120.0)},
                    "timeout",
                    20.0,
                    180.0,
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "task_status": "invalid_input",
                    "completed": False,
                    "error": str(exc),
                }
            transport = self._mcp_call(
                "navigate_with_text",
                {
                    "query": query.strip(),
                    "standoff_distance": distance,
                    "timeout": timeout,
                },
                timeout=min(195.0, timeout + 10.0),
            )
            payload = _structured_mcp_payload(transport)
            if not isinstance(payload, dict):
                outcome = {
                    "ok": False,
                    "task_status": "invalid_tool_result",
                    "completed": False,
                    "error": "navigate_with_text 未返回结构化任务状态",
                }
            else:
                status = str(payload.get("task_status") or "invalid_tool_result")
                position_error = payload.get("position_error_m")
                navigation_ok = bool(
                    status == "navigation_verified"
                    and payload.get("planner_goal_reached") is True
                    and payload.get("stationary_confirmed") is True
                    and isinstance(position_error, (int, float))
                    and not isinstance(position_error, bool)
                    and 0.0 <= float(position_error) <= 0.5
                )
                verification_frame_timestamp = payload.get(
                    "verification_frame_timestamp"
                )
                stationary_confirmed_at = payload.get("stationary_confirmed_at")
                visual_ok = bool(
                    status == "arrived_verified"
                    and payload.get("planner_goal_reached") is True
                    and payload.get("stationary_confirmed") is True
                    and isinstance(payload.get("verification"), dict)
                    and isinstance(verification_frame_timestamp, (int, float))
                    and not isinstance(verification_frame_timestamp, bool)
                    and isinstance(stationary_confirmed_at, (int, float))
                    and not isinstance(stationary_confirmed_at, bool)
                    and float(verification_frame_timestamp)
                    > float(stationary_confirmed_at)
                )
                evidence_ok = bool(
                    payload.get("completed") is True
                    and (navigation_ok or visual_ok)
                    and payload.get("used_scene_truth") is False
                    and payload.get("navigate_with_text_route")
                    in {
                        "exact_tag",
                        "current_rgbd",
                        "persistent_clip_memory",
                        "frontier_fallback",
                    }
                )
                outcome = {
                    **payload,
                    "ok": bool(
                        transport.get("ok")
                        and payload.get("tool_ok") is True
                        and evidence_ok
                    ),
                    "completed": evidence_ok,
                    "task_status": status,
                    "text_navigation_verified": evidence_ok,
                }
                if payload.get("completed") is True and not evidence_ok:
                    outcome.update(
                        ok=False,
                        completed=False,
                        task_status="invalid_tool_result",
                        error="自然语言导航成功声明缺少到达/停车/传感器证据",
                    )
            if not outcome.get("completed"):
                outcome["stop"] = self._stop_robot({})
            outcome["observation_after"] = self._observation()
            self._observed_this_turn = True
            return outcome

    def _tag_location(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("location_name")
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
            return {"ok": False, "error": "location_name 必须是 1 到 80 字符的字符串"}
        return self._mcp_call("tag_location", {"location_name": name.strip()})

    def _isaac_terminal_navigation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if self.backend != "isaac-g1":
            return {"ok": False, "error": f"{tool_name} 仅支持 Isaac G1"}
        guard = self._motion_guard()
        if guard:
            return {"ok": False, "error": guard}
        timeout = float(arguments.get("timeout_seconds", 60.0))
        transport = self._mcp_call(
            tool_name,
            arguments,
            timeout=min(118.0, timeout + 8.0),
        )
        outcome = _isaac_navigation_mcp_result(transport)
        if not outcome.get("completed"):
            outcome["stop"] = self._stop_robot({})
        outcome["observation_after"] = self._observation()
        self._observed_this_turn = True
        return outcome

    def _navigate_to_pose(self, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = {"x", "y", "yaw_degrees", "timeout_seconds"}
        if not set(arguments).issubset(allowed) or not {"x", "y"}.issubset(arguments):
            return {"ok": False, "error": "navigate_to_pose 必须提供 x、y"}
        try:
            normalized = {
                "x": self._number(arguments, "x", -100.0, 100.0),
                "y": self._number(arguments, "y", -100.0, 100.0),
                "yaw_degrees": self._number(
                    {"yaw_degrees": arguments.get("yaw_degrees", 0.0)},
                    "yaw_degrees",
                    -360.0,
                    360.0,
                ),
                "timeout_seconds": self._number(
                    {"timeout_seconds": arguments.get("timeout_seconds", 60.0)},
                    "timeout_seconds",
                    1.0,
                    110.0,
                ),
            }
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return self._isaac_terminal_navigation("navigate_to_pose", normalized)

    def _navigate_to_tag(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not set(arguments).issubset({"location_name", "timeout_seconds"}):
            return {"ok": False, "error": "navigate_to_tag 参数无效"}
        name = arguments.get("location_name")
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
            return {"ok": False, "error": "location_name 必须是 1 到 80 字符"}
        try:
            timeout = self._number(
                {"timeout_seconds": arguments.get("timeout_seconds", 60.0)},
                "timeout_seconds",
                1.0,
                110.0,
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return self._isaac_terminal_navigation(
            "navigate_to_tag",
            {"location_name": name.strip(), "timeout_seconds": timeout},
        )

    def _stop_navigation(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return {"ok": False, "error": "stop_navigation 不接受参数"}
        result = self._mcp_call("stop_navigation")
        return {
            **result,
            "observation_after": self._observation(),
        }

    def _native_visual_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        argument_name: str,
        max_length: int,
    ) -> dict[str, Any]:
        if set(arguments) != {argument_name}:
            return {"ok": False, "error": f"{tool_name} 只接受 {argument_name}"}
        value = arguments.get(argument_name)
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > max_length:
            return {
                "ok": False,
                "error": f"{argument_name} 必须是 1 到 {max_length} 字符的字符串",
            }
        transport = self._mcp_call(
            tool_name,
            {argument_name: value.strip()},
            timeout=NATIVE_VLM_TIMEOUT,
        )
        payload = transport.get("result")
        if not isinstance(payload, dict):
            output = transport.get("output")
            if isinstance(output, str):
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    payload = parsed
        if not isinstance(payload, dict):
            return transport
        envelope = {
            key: item
            for key, item in transport.items()
            if key not in {"result", "output", "ok"}
        }
        return {
            **envelope,
            **payload,
            "ok": bool(transport.get("ok") and payload.get("ok", True)),
        }

    def _analyze_scene(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._native_visual_tool("analyze_scene", arguments, "question", 500)

    def _find_visual_target(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._native_visual_tool("find_visual_target", arguments, "target", 300)

    def _verify_visual_condition(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._native_visual_tool(
            "verify_visual_condition",
            arguments,
            "condition",
            500,
        )

    def _object_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        guard = self._motion_guard()
        if guard:
            return {
                "ok": False,
                "task_status": "precondition_failed",
                "completed": False,
                "error": guard,
            }
        if set(arguments) - {"query", "standoff_distance", "timeout"}:
            return {
                "ok": False,
                "task_status": "invalid_input",
                "completed": False,
                "error": "object_search 收到未知参数",
            }
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 200:
            return {
                "ok": False,
                "task_status": "invalid_input",
                "completed": False,
                "error": "query 必须是 1 到 200 字符",
            }
        try:
            distance = self._number(
                {"standoff_distance": arguments.get("standoff_distance", 0.9)},
                "standoff_distance",
                0.5,
                3.0,
            )
            timeout = self._number(
                {"timeout": arguments.get("timeout", 120.0)},
                "timeout",
                20.0,
                180.0,
            )
        except ValueError as exc:
            return {
                "ok": False,
                "task_status": "invalid_input",
                "completed": False,
                "error": str(exc),
            }
        result = self._mcp_call(
            "object_search",
            {
                "query": query.strip(),
                "standoff_distance": distance,
                "timeout": timeout,
            },
            timeout=OBJECT_SEARCH_TIMEOUT,
        )
        outcome = _closed_loop_mcp_result(result, "object_search")
        outcome = _validate_visual_approach_standoff(
            outcome,
            query=query.strip(),
            requested_distance=distance,
        )
        outcome["observation_after"] = self._observation()
        self._observed_this_turn = True
        return outcome

    def _explore_frontiers(self, arguments: dict[str, Any]) -> dict[str, Any]:
        guard = self._motion_guard()
        if guard:
            return {
                "ok": False,
                "task_status": "precondition_failed",
                "completed": False,
                "error": guard,
            }
        if set(arguments) - {"timeout", "max_frontiers"}:
            return {
                "ok": False,
                "task_status": "invalid_input",
                "completed": False,
                "error": "explore_frontiers 收到未知参数",
            }
        try:
            timeout = self._number(
                {"timeout": arguments.get("timeout", 90.0)},
                "timeout",
                20.0,
                180.0,
            )
            max_frontiers = int(arguments.get("max_frontiers", 8))
            if not 1 <= max_frontiers <= 20:
                raise ValueError("max_frontiers 必须在 1 到 20 之间")
        except (TypeError, ValueError, OverflowError) as exc:
            return {
                "ok": False,
                "task_status": "invalid_input",
                "completed": False,
                "error": str(exc),
            }
        result = self._mcp_call(
            "explore_frontiers",
            {
                "timeout": timeout,
                "max_frontiers": max_frontiers,
            },
            timeout=min(195.0, timeout + 10.0),
        )
        payload = _structured_mcp_payload(result)
        if not isinstance(payload, dict):
            outcome = {
                "ok": False,
                "task_status": "invalid_tool_result",
                "completed": False,
                "error": "explore_frontiers 未返回结构化任务状态",
            }
        else:
            status = str(payload.get("task_status") or "invalid_tool_result")
            goals = payload.get("goals")
            planner_goal_reached = bool(
                isinstance(goals, list)
                and any(
                    isinstance(goal, dict)
                    and goal.get("goal_space") == "known_free"
                    and goal.get("navigation_status") == "frontier_reached"
                    for goal in goals
                )
            )
            evidence_ok = bool(
                payload.get("completed") is True
                and status
                in {"exploration_complete", "exploration_budget_complete"}
                and payload.get("planner_goal_space") == "known_free_only"
                and int(payload.get("frontiers_reached", 0)) > 0
                and planner_goal_reached
            )
            outcome = {
                **payload,
                "ok": bool(result.get("ok") and payload.get("tool_ok") is True),
                "completed": evidence_ok,
                "task_status": status,
                "frontier_exploration_verified": evidence_ok,
            }
            if payload.get("completed") is True and not evidence_ok:
                outcome.update(
                    ok=False,
                    completed=False,
                    task_status="invalid_tool_result",
                    error="探索成功声明缺少已知自由侧 frontier 到达证据",
                )
        if not outcome.get("completed"):
            outcome["stop"] = self._stop_robot({})
        outcome["observation_after"] = self._observation()
        self._observed_this_turn = True
        return outcome

    def _follow_person(self, arguments: dict[str, Any]) -> dict[str, Any]:
        guard = self._motion_guard()
        if guard:
            return {
                "ok": False,
                "task_status": "precondition_failed",
                "completed": False,
                "error": guard,
            }
        if set(arguments) - {"query", "follow_distance", "duration", "timeout"}:
            return {
                "ok": False,
                "task_status": "invalid_input",
                "completed": False,
                "error": "follow_person 收到未知参数",
            }
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 200:
            return {
                "ok": False,
                "task_status": "invalid_input",
                "completed": False,
                "error": "query 必须是 1 到 200 字符",
            }
        try:
            follow_distance = self._number(
                {"follow_distance": arguments.get("follow_distance", 1.5)},
                "follow_distance",
                1.2,
                2.0,
            )
            duration = self._number(
                {"duration": arguments.get("duration", 30.0)},
                "duration",
                5.0,
                60.0,
            )
            timeout = self._number(
                {"timeout": arguments.get("timeout", 150.0)},
                "timeout",
                65.0,
                180.0,
            )
        except ValueError as exc:
            return {
                "ok": False,
                "task_status": "invalid_input",
                "completed": False,
                "error": str(exc),
            }
        if timeout < duration + 55.0:
            return {
                "ok": False,
                "task_status": "invalid_input",
                "completed": False,
                "error": "timeout 必须至少比 duration 多 55 秒",
            }
        result = self._mcp_call(
            "follow_person",
            {
                "query": query.strip(),
                "follow_distance": follow_distance,
                "duration": duration,
                "timeout": timeout,
            },
            timeout=FOLLOW_PERSON_TIMEOUT,
        )
        payload = _structured_mcp_payload(result)
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "task_status": "invalid_tool_result",
                "completed": False,
                "error": "follow_person 未返回结构化任务状态",
            }
        evidence_ok = bool(
            result.get("ok")
            and payload.get("tool_ok") is True
            and payload.get("completed") is True
            and payload.get("task_status") == "follow_verified"
            and payload.get("planner_goal_reached") is False
            and isinstance(payload.get("stationary_confirmed_at"), (int, float))
            and isinstance(payload.get("verification_frame_timestamp"), (int, float))
            and payload["verification_frame_timestamp"]
            > payload["stationary_confirmed_at"]
        )
        outcome = {
            **payload,
            "ok": bool(result.get("ok") and payload.get("tool_ok") is True),
            "completed": evidence_ok,
        }
        if payload.get("completed") is True and not evidence_ok:
            outcome.update(
                ok=False,
                completed=False,
                reported_task_status=payload.get("task_status"),
                task_status="invalid_tool_result",
                error="follow_person 成功声明缺少停车后 RGB-D 证据",
            )
        outcome["observation_after"] = self._observation()
        self._observed_this_turn = True
        return outcome

    def _approach_visual_target(self, arguments: dict[str, Any]) -> dict[str, Any]:
        guard = self._motion_guard()
        if guard:
            return {
                "ok": False,
                "error": guard,
                "task_status": (
                    "precondition_failed"
                    if not self._observed_this_turn
                    else "risk_blocked"
                ),
                "completed": False,
            }
        if set(arguments) - {"query", "standoff_distance", "timeout"}:
            return {
                "ok": False,
                "error": "approach_visual_target 收到未知参数",
                "task_status": "invalid_input",
                "completed": False,
            }
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 200:
            return {
                "ok": False,
                "error": "query 必须是 1 到 200 字符",
                "task_status": "invalid_input",
                "completed": False,
            }
        risk = (self.monitor.agent_snapshot().get("metrics") or {}).get("risk")
        if risk != "clear":
            return {
                "ok": False,
                "error": f"视觉接近只允许在 clear 风险下启动，当前为 {risk}",
                "task_status": "risk_blocked",
                "completed": False,
            }
        try:
            distance = self._number(
                {"standoff_distance": arguments.get("standoff_distance", 0.9)},
                "standoff_distance",
                0.5,
                3.0,
            )
            timeout = self._number(
                {"timeout": arguments.get("timeout", 50.0)},
                "timeout",
                3.0,
                60.0,
            )
        except ValueError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "task_status": "invalid_input",
                "completed": False,
            }

        result = self._mcp_call(
            "approach_visual_target",
            {
                "query": query.strip(),
                "standoff_distance": distance,
                "timeout": timeout,
            },
            timeout=APPROACH_VISUAL_TARGET_TIMEOUT,
        )
        outcome = _closed_loop_mcp_result(result, "approach_visual_target")
        outcome = _validate_visual_approach_standoff(
            outcome,
            query=query.strip(),
            requested_distance=distance,
        )
        outcome["observation_after"] = self._observation()
        self._observed_this_turn = True
        return outcome

    def _approach_person(self, arguments: dict[str, Any]) -> dict[str, Any]:
        guard = self._motion_guard()
        if guard:
            return {
                "ok": False,
                "error": guard,
                "task_status": (
                    "precondition_failed"
                    if not self._observed_this_turn
                    else "risk_blocked"
                ),
                "completed": False,
            }
        if set(arguments) - {"query", "standoff_distance", "timeout"}:
            return {
                "ok": False,
                "error": "approach_person 收到未知参数",
                "task_status": "invalid_input",
                "completed": False,
            }
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 200:
            return {
                "ok": False,
                "error": "query 必须是 1 到 200 字符的字符串",
                "task_status": "invalid_input",
                "completed": False,
            }
        risk = (self.monitor.agent_snapshot().get("metrics") or {}).get("risk")
        if risk != "clear":
            return {
                "ok": False,
                "error": f"人物接近只允许在 clear 风险下启动，当前为 {risk}",
                "task_status": "risk_blocked",
                "completed": False,
            }
        try:
            distance = self._number(
                {"standoff_distance": arguments.get("standoff_distance", 1.2)},
                "standoff_distance",
                1.0,
                2.0,
            )
            timeout = self._number(
                {"timeout": arguments.get("timeout", 20.0)},
                "timeout",
                3.0,
                20.0,
            )
        except ValueError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "task_status": "invalid_input",
                "completed": False,
            }
        result = self._mcp_call(
            "approach_person",
            {
                "query": query.strip(),
                "standoff_distance": distance,
                "timeout": timeout,
            },
            timeout=APPROACH_PERSON_TIMEOUT,
        )
        outcome = _closed_loop_mcp_result(result, "approach_person")
        outcome = _validate_visual_approach_standoff(
            outcome,
            query=query.strip(),
            requested_distance=distance,
        )
        if not outcome.get("completed"):
            outcome["stop"] = self._stop_robot({})
        outcome["observation_after"] = self._observation()
        self._observed_this_turn = True
        return outcome

    def _fetch_object(self, arguments: dict[str, Any]) -> dict[str, Any]:
        guard = self._motion_guard()
        if guard:
            return {
                "ok": False,
                "error": guard,
                "task_status": (
                    "precondition_failed"
                    if not self._observed_this_turn
                    else "risk_blocked"
                ),
                "completed": False,
            }
        if set(arguments) - {
            "object_id",
            "pickup_pose",
            "destination",
            "hand",
        }:
            return {
                "ok": False,
                "error": "fetch_object 收到未知参数",
                "task_status": "invalid_input",
                "completed": False,
            }
        object_id = arguments.get("object_id")
        pickup_pose = arguments.get("pickup_pose")
        destination = arguments.get("destination")
        hand = arguments.get("hand", "right")
        if object_id != "water_bottle":
            return {
                "ok": False,
                "error": "当前只验收 water_bottle",
                "task_status": "invalid_input",
                "completed": False,
            }
        if pickup_pose != "kitchen":
            return {
                "ok": False,
                "error": "当前取物地点只支持 kitchen",
                "task_status": "invalid_input",
                "completed": False,
            }
        if destination not in {"start", "living_room"}:
            return {
                "ok": False,
                "error": "destination 只支持 start 或 living_room",
                "task_status": "invalid_input",
                "completed": False,
            }
        if hand != "right":
            return {
                "ok": False,
                "error": "当前取物闭环只验收 right hand",
                "task_status": "invalid_input",
                "completed": False,
            }
        risk = (self.monitor.agent_snapshot().get("metrics") or {}).get("risk")
        if risk != "clear":
            return {
                "ok": False,
                "error": f"完整取物只允许在 clear 风险下启动，当前为 {risk}",
                "task_status": "risk_blocked",
                "completed": False,
            }

        fetch_arguments = {
            "object_id": object_id,
            "pickup_pose": pickup_pose,
            "destination": destination,
            "hand": hand,
        }
        result = (
            self._long_task_runner(
                "fetch_object",
                fetch_arguments,
                FETCH_OBJECT_PROCESS_TIMEOUT,
            )
            if self._long_task_runner is not None
            else self._mcp_call(
                "fetch_object",
                fetch_arguments,
                timeout=FETCH_OBJECT_PROCESS_TIMEOUT,
            )
        )
        outcome = _fetch_object_mcp_result(
            result,
            object_id=object_id,
            destination=destination,
        )
        if not outcome.get("completed"):
            outcome["stop"] = self._stop_robot({})
        outcome["observation_after"] = self._observation()
        self._observed_this_turn = True
        return outcome
