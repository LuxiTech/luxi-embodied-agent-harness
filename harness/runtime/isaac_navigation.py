"""Runtime-fenced Isaac NavigationPort over the shared native planner Skill."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

from .contracts import CancellationToken
from .ports import CommandEvidence, RuntimeCommand


class IsaacG1NavigationPort:
    """Fence correlated native navigation; this class publishes no commands."""

    supported_capabilities = frozenset(
        {
            "navigate_to_pose",
            "navigate_to_tag",
            "explore_frontiers",
            "object_search",
            "follow_person",
            "approach_person",
            "navigate_with_text",
        }
    )

    def __init__(
        self,
        *,
        robot_id: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.robot_id = robot_id
        self.clock = clock
        self._boot_epoch = "unbound"
        self._seen_calls: set[tuple[str, str]] = set()
        self._executor: (
            Callable[[RuntimeCommand, CancellationToken], Mapping[str, Any]] | None
        ) = None
        self._active_tokens: dict[str, CancellationToken] = {}
        self._lock = threading.RLock()

    def bind_navigation_executor(
        self,
        executor: Callable[[RuntimeCommand, CancellationToken], Mapping[str, Any]],
    ) -> None:
        with self._lock:
            if self._executor is not None and self._executor is not executor:
                raise RuntimeError("Isaac NavigationPort already has an owner")
            self._executor = executor

    def bind_boot_epoch(self, boot_epoch: str) -> None:
        with self._lock:
            if boot_epoch != self._boot_epoch:
                self._seen_calls.clear()
            self._boot_epoch = boot_epoch

    def execute(
        self,
        command: RuntimeCommand,
        cancel: CancellationToken,
    ) -> Mapping[str, Any]:
        with self._lock:
            if command.cancel_token_id in self._active_tokens:
                raise PermissionError("navigation cancel token replay rejected")
            self._active_tokens[command.cancel_token_id] = cancel
        try:
            evidence = self.navigate(command)
            raw = evidence.evidence.get("raw")
            if not isinstance(raw, Mapping):
                raise RuntimeError("Isaac NavigationPort returned no structured result")
            return dict(raw)
        finally:
            with self._lock:
                self._active_tokens.pop(command.cancel_token_id, None)

    def navigate(self, command: RuntimeCommand) -> CommandEvidence:
        now = self.clock()
        with self._lock:
            if command.robot_id != self.robot_id:
                raise PermissionError("Isaac navigation command robot mismatch")
            if command.boot_epoch != self._boot_epoch:
                raise PermissionError("Isaac navigation command boot epoch mismatch")
            if command.deadline_monotonic <= now:
                raise TimeoutError("Isaac navigation command deadline expired")
            capability_id = command.payload.get("capability_id")
            if capability_id not in self.supported_capabilities:
                return CommandEvidence(False, "unsupported", now)
            if not isinstance(command.payload.get("arguments"), Mapping):
                raise ValueError(f"{capability_id} arguments are missing")
            cancel = self._active_tokens.get(command.cancel_token_id)
            if cancel is None:
                raise PermissionError("navigation command has no cancellation owner")
            executor = self._executor
            if executor is None:
                return CommandEvidence(False, "unsupported", now)
            key = (command.boot_epoch, command.tool_call_id)
            if key in self._seen_calls:
                raise PermissionError("Isaac navigation command replay rejected")
            self._seen_calls.add(key)
        raw = executor(command, cancel)
        return CommandEvidence(
            True,
            str(raw.get("task_status", "navigation_failed")),
            self.clock(),
            {
                "raw": dict(raw),
                "publisher_implementation": "existing-native-navigation-skill",
            },
        )

    def cancel_navigation(self, command: RuntimeCommand) -> CommandEvidence:
        with self._lock:
            cancel = self._active_tokens.get(command.cancel_token_id)
        if cancel is None:
            return CommandEvidence(False, "navigation_not_active", self.clock())
        cancel.cancel()
        return CommandEvidence(True, "navigation_cancel_requested", self.clock())
