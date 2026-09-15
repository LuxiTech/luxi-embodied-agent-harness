"""Deterministic single-robot skill arguments, with no simulator dependency."""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class ParsedInstruction:
    action: str
    target: str
    duration_s: float = 12.0
    standoff_m: float = 0.9


_TARGET_ALIASES = {
    "red_cube": ("红色方块", "红方块", "红色箱子", "red cube", "cube"),
    "blue_ball": ("蓝色球", "蓝球", "blue ball", "ball"),
    "bottle": ("水瓶", "瓶子", "矿泉水", "bottle"),
}


def parse_instruction(text: str) -> ParsedInstruction:
    from harness.runtime.task_policy import reject_multi_robot_instruction
    rejection = reject_multi_robot_instruction(text)
    if rejection:
        raise ValueError(rejection)
    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("instruction must not be empty")
    if "巡检" in normalized or "inspection" in normalized:
        region = "south" if any(
            token in normalized for token in ("南区", "南侧", "south")
        ) else "north"
        return ParsedInstruction("inspection", region)
    if "跟随" in normalized or "follow" in normalized:
        duration_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s(?:ec(?:onds?)?)?)", normalized)
        duration = float(duration_match.group(1)) if duration_match else 12.0
        return ParsedInstruction("follow", "person", min(max(duration, 1.0), 60.0), 1.2)
    if any(token in normalized for token in ("寻找", "寻物", "搜索", "找", "find", "search")):
        for target, aliases in _TARGET_ALIASES.items():
            if any(alias in normalized for alias in aliases):
                return ParsedInstruction("search", target)
        raise ValueError("请指定红色方块、蓝色球或水瓶")
    raise ValueError("仅支持仓库巡检、寻物和跟随指令")
