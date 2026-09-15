"""Correlated AgentOS navigation-Skill handoff to the unified runtime owner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping

from harness.robots.g1.isaac.motion_service import (
    IsaacMotionControlChannel,
    IsaacMotionRequestService,
)
from harness.robots.g1.isaac.isaac_protocol import IsaacRuntimePaths


AGENT_NAVIGATION_CONTROL_PATH_ENV = "LUXI_AGENT_NAVIGATION_CONTROL_PATH"


def configured_agent_navigation_root() -> Path:
    configured = os.environ.get(AGENT_NAVIGATION_CONTROL_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return IsaacRuntimePaths.configured().root / "agent-navigation"


class IsaacNavigationControlChannel(IsaacMotionControlChannel):
    """Use the proven atomic request/grant/result protocol on a separate port."""

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(root or configured_agent_navigation_root())

    def request_navigate_to_pose(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "navigate_to_pose",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )

    def request_navigate_to_tag(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "navigate_to_tag",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )

    def request_explore_frontiers(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "explore_frontiers",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )

    def request_object_search(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "object_search",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )

    def request_follow_person(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "follow_person",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )

    def request_approach_person(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "approach_person",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )

    def request_navigate_with_text(
        self,
        arguments: Mapping[str, Any],
        execute_local: Callable[[Callable[[], bool]], str],
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        return self.request_motion(
            "navigate_with_text",
            arguments,
            execute_local,
            timeout_s=timeout_s,
        )


class IsaacNavigationRequestService(IsaacMotionRequestService):
    """One active terminal navigation request; contains no model loop."""
