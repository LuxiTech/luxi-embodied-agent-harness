"""Small versioned ports hiding simulator and robot implementation details."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .contracts import ArtifactRef


PORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ObservationSnapshot:
    robot_id: str
    boot_epoch: str
    frame_id: str
    timestamp_monotonic: float
    freshness_s: float
    source: str
    calibration_revision: str
    values: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float
    frame_id: str
    timestamp_monotonic: float


@dataclass(frozen=True)
class RuntimeCommand:
    robot_id: str
    boot_epoch: str
    task_id: str
    tool_call_id: str
    deadline_monotonic: float
    cancel_token_id: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class CommandEvidence:
    accepted: bool
    status: str
    timestamp_monotonic: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


class ObservationPort(Protocol):
    def snapshot(self, robot_id: str) -> ObservationSnapshot: ...


class MappingLocalizationPort(Protocol):
    def pose(self, robot_id: str) -> Pose2D: ...

    def map_artifact(self, robot_id: str) -> ArtifactRef | None: ...


class NavigationPort(Protocol):
    def navigate(self, command: RuntimeCommand) -> CommandEvidence: ...

    def cancel_navigation(self, command: RuntimeCommand) -> CommandEvidence: ...


class MotionPort(Protocol):
    """Only the Safety Kernel's SafeCommandGateway may own this port."""

    def command_motion(self, command: RuntimeCommand) -> CommandEvidence: ...

    def command_zero(self, command: RuntimeCommand) -> CommandEvidence: ...


class ManipulationEntityPort(Protocol):
    def manipulate(self, command: RuntimeCommand) -> CommandEvidence: ...

    def cancel_manipulation(self, command: RuntimeCommand) -> CommandEvidence: ...


class LifecycleHealthPort(Protocol):
    def start(self, robot_id: str, boot_epoch: str) -> None: ...

    def stop(self, robot_id: str, boot_epoch: str) -> None: ...

    def health(self, robot_id: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RobotPorts:
    observation: ObservationPort
    mapping_localization: MappingLocalizationPort | None
    navigation: NavigationPort | None
    motion: MotionPort | None
    manipulation_entity: ManipulationEntityPort | None
    lifecycle_health: LifecycleHealthPort


class RobotWorldAdapter:
    """Thin contract composition; it contains no planning or safety policy."""

    def __init__(
        self,
        *,
        adapter_id: str,
        robot_type: str,
        backend: str,
        ports: RobotPorts,
        capabilities: frozenset[str],
    ) -> None:
        self.adapter_id = adapter_id
        self.robot_type = robot_type
        self.backend = backend
        self.ports = ports
        self.capabilities = capabilities

    def require(self, capability_id: str) -> None:
        if capability_id not in self.capabilities:
            raise NotImplementedError(
                f"{self.adapter_id} explicitly reports unsupported capability: {capability_id}"
            )
