"""Extracted shared implementation; independent of Agent entry points."""

from __future__ import annotations
import re


MAX_PLANNING_STEPS = 8

MAX_TOOL_CALLS = 8

VISUAL_BRANCH_TOOLS = frozenset(
    {
        "analyze_scene",
        "find_visual_target",
        "verify_visual_condition",
        "approach_visual_target",
        "approach_person",
        "explore_frontiers",
        "object_search",
        "follow_person",
        "navigate_with_text",
    }
)

COMPLETED_TASK_STATUSES = frozenset(
    {
        "arrived_verified",
        "turn_verified",
        "room_loop_verified",
        "distance_verified",
        "object_fetched",
        "navigation_verified",
        "follow_verified",
        "exploration_complete",
        "exploration_budget_complete",
    }
)

TERMINAL_CLOSED_LOOP_TOOLS = frozenset(
    {
        "approach_visual_target",
        "approach_person",
        "object_search",
        "follow_person",
        "explore_frontiers",
        "walk_room_loop",
        "move_distance",
        "fetch_object",
        "navigate_to_pose",
        "navigate_to_tag",
        "navigate_with_text",
    }
)

EXPLICIT_DISTANCE_PATTERN = re.compile(
    r"^(?:请)?(?:让(?:机器人|它))?"
    r"(?P<direction>向前|往前|前进|直走|走|向后|往后|后退)"
    r"(?:走)?(?P<distance>\d+(?:\.\d+)?|[一二两三四五六七八九十])(?:米|m)$"
)

def _explicit_distance_for_instruction(instruction: str) -> float | None:
    normalized = "".join(instruction.casefold().split()).rstrip("。.!！")
    match = EXPLICIT_DISTANCE_PATTERN.fullmatch(normalized)
    if match is None:
        return None
    raw = match.group("distance")
    distance = float({"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}.get(raw, raw))
    if match.group("direction") in {"向后", "往后", "后退"}:
        distance = -distance
    return distance

def _forced_tool_for_instruction(instruction: str) -> str | None:
    """Route explicit approach-and-stop intents to a verified closed loop."""

    normalized = "".join(instruction.casefold().split()).rstrip("。.!！")
    if _explicit_distance_for_instruction(normalized) is not None:
        return "move_distance"
    if not normalized or any(
        marker in normalized
        for marker in (
            "绕着房间",
            "绕房间",
            "走一圈",
            "转身",
            "转180",
            "探索",
        )
    ):
        return None
    if any(marker in normalized for marker in ("跟随", "跟着", "follow")) and any(
        marker in normalized
        for marker in ("人物", "行人", "person", "human", "人")
    ):
        return "follow_person"
    search_requested = any(
        marker in normalized for marker in ("去找", "寻找", "搜寻", "查找")
    )
    report_only = any(
        marker in normalized for marker in ("告诉我", "在哪里", "是否", "有没有")
    )
    if search_requested and not report_only:
        return (
            "approach_person"
            if any(
                marker in normalized
                for marker in ("人物", "行人", "person", "human", "人")
            )
            else "object_search"
        )
    approach_requested = any(
        marker in normalized
        for marker in ("走到", "前往", "靠近", "接近")
    )
    stopping_near_target = any(
        marker in normalized
        for marker in ("停下", "停在", "附近", "旁边", "面前", "前停")
    )
    if not (approach_requested and stopping_near_target):
        return None
    if any(
        marker in normalized
        for marker in ("人物", "行人", "person", "human", "一个人", "那个人")
    ):
        return "approach_person"
    return "approach_visual_target"



def reject_multi_robot_instruction(instruction: str) -> str | None:
    """Reject removed fleet intents without blocking sequential single-robot work."""
    patterns = (
        r"go2[-_ ]?0?[2-9]\b", r"(?:两|二|双|多|[2-9])(?:台|只|个)?(?:机器狗|机器人|go2)",
        r"(?:两|二|[2-9])台", r"双机|多机|另一台|第二台|分别巡检(?:南北|北南)",
        r"一台.{0,100}(?:另一台|一台)",
        r"\b(?:two|multiple|both)\s+(?:robots|dogs|go2s)\b", r"\bfleet\b",
    )
    if any(re.search(pattern, instruction, re.IGNORECASE) for pattern in patterns):
        return "当前仅支持单机器人任务，已取消双机协同和第二台机器人。"
    return None


def trusted_tool_arguments(instruction, name, arguments):
    """Use exact operator distances, never a model-invented movement amount."""
    result = dict(arguments)
    distance = _explicit_distance_for_instruction(instruction)
    if name == "move_distance" and distance is not None:
        result["distance_m"] = distance
    return result
