"""Single capability registry and least-privilege Agent scope resolution."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import threading
from typing import Callable, Iterable, Mapping

from .contracts import (
    AgentScope,
    CapabilityDescriptor,
    CapabilitySnapshot,
)


class LuxiCapabilityRegistry:
    def __init__(self) -> None:
        self._items: dict[str, CapabilityDescriptor] = {}
        self._lock = threading.RLock()
        self._revision = 0

    def register(self, descriptor: CapabilityDescriptor) -> None:
        with self._lock:
            existing = self._items.get(descriptor.capability_id)
            if existing is not None and existing != descriptor:
                raise ValueError(
                    f"capability already registered with different semantics: "
                    f"{descriptor.capability_id}"
                )
            if existing is None:
                self._items[descriptor.capability_id] = descriptor
                self._revision += 1

    def get(self, capability_id: str) -> CapabilityDescriptor | None:
        with self._lock:
            return self._items.get(capability_id)

    def all(self) -> Mapping[str, CapabilityDescriptor]:
        with self._lock:
            return dict(self._items)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision


class AgentScopeResolver:
    """Intersect declarations, grants, robot reality and provider health."""

    def __init__(
        self,
        registry: LuxiCapabilityRegistry,
        *,
        provider_health: Mapping[str, Callable[[], bool]] | None = None,
        robot_capabilities: Callable[[str], Iterable[str]] | None = None,
        robot_types: Callable[[str], str | None] | None = None,
        robot_backends: Callable[[str], str | None] | None = None,
    ) -> None:
        self.registry = registry
        self.provider_health = dict(provider_health or {})
        self.robot_capabilities = robot_capabilities or (lambda _robot_id: ())
        self.robot_types = robot_types or (lambda _robot_id: None)
        self.robot_backends = robot_backends or (lambda _robot_id: None)

    def snapshot(self, scope: AgentScope) -> CapabilitySnapshot:
        actual_by_robot = {
            robot_id: frozenset(self.robot_capabilities(robot_id))
            for robot_id in scope.robot_ids
        }
        admitted: dict[str, CapabilityDescriptor] = {}
        for capability_id, descriptor in sorted(self.registry.all().items()):
            if not scope.permits(capability_id):
                continue
            health = self.provider_health.get(descriptor.provider)
            if health is not None:
                try:
                    if not health():
                        continue
                except Exception:
                    continue
            if scope.robot_ids:
                supported = False
                for robot_id in scope.robot_ids:
                    if capability_id not in actual_by_robot[robot_id]:
                        continue
                    robot_type = self.robot_types(robot_id)
                    backend = self.robot_backends(robot_id)
                    if descriptor.robot_types and robot_type not in descriptor.robot_types:
                        continue
                    if descriptor.backends and backend not in descriptor.backends:
                        continue
                    supported = True
                    break
                if not supported:
                    continue
            admitted[capability_id] = descriptor
        revision_material = {
            "registry_revision": self.registry.revision,
            "scope_revision": scope.scope_revision,
            "capabilities": [
                [item.capability_id, item.version] for item in admitted.values()
            ],
        }
        digest = hashlib.sha256(
            json.dumps(revision_material, sort_keys=True).encode()
        ).hexdigest()[:24]
        return CapabilitySnapshot(
            scope_revision=scope.scope_revision,
            snapshot_revision=f"caps_{digest}",
            capabilities=admitted,
        )

    def refresh_scope(self, scope: AgentScope, **changes: object) -> AgentScope:
        """Create a new revision; an existing Step snapshot never mutates."""

        from .contracts import new_id

        return replace(scope, scope_revision=new_id("scope"), **changes)
