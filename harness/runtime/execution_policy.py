"""Task-local execution strategies; neither strategy executes tools."""
from dataclasses import dataclass, replace
import hashlib

from .contracts import CapabilitySnapshot


@dataclass(frozen=True)
class TerminalPolicy:
    mode: str = "terminal"
    completion_capability: str | None = None
    sequential: bool = False

    def snapshot(self, snapshot):
        return _filter(snapshot, self.mode)


@dataclass(frozen=True)
class ComposedPolicy:
    completion_capability: str
    mode: str = "composed"
    sequential: bool = True

    def __post_init__(self):
        if not self.completion_capability.strip():
            raise ValueError("composed execution requires a final verifier")

    def snapshot(self, snapshot):
        return _filter(snapshot, self.mode)


def _filter(snapshot: CapabilitySnapshot, mode: str):
    capabilities = {
        name: descriptor for name, descriptor in snapshot.capabilities.items()
        if mode in descriptor.execution_modes
        and (mode != "composed" or not descriptor.terminal)
    }
    revision = hashlib.sha256(
        f"{snapshot.snapshot_revision}:{mode}:{sorted(capabilities)}".encode()
    ).hexdigest()[:24]
    return replace(snapshot, capabilities=capabilities, snapshot_revision=f"caps_{revision}")


def validate_execution_mode(mode):
    if mode not in ("terminal", "composed"):
        raise ValueError("execution_mode 必须为 terminal 或 composed；自动路由尚未开放")
    return mode
