"""Robot-neutral orchestration core for Luxi embodied agents.

The package deliberately contains no MuJoCo, Isaac, Go2, ROS, DimOS or model
imports.  Concrete deployments connect those systems in the composition root.
"""

from .agent_loop import LuxiAgentLoop
from .capabilities import AgentScopeResolver, LuxiCapabilityRegistry
from .contracts import (
    AgentScope,
    CapabilityDescriptor,
    CapabilitySnapshot,
    LoopLimits,
    SideEffect,
    TaskOutcome,
    ToolRequest,
    ToolResult,
)
from .session_store import LuxiSessionStore, SessionEvent
from .tool_pipeline import LuxiToolPipeline, LuxiToolRegistry

__all__ = [
    "AgentScope",
    "AgentScopeResolver",
    "CapabilityDescriptor",
    "CapabilitySnapshot",
    "LoopLimits",
    "LuxiAgentLoop",
    "LuxiCapabilityRegistry",
    "LuxiSessionStore",
    "LuxiToolPipeline",
    "LuxiToolRegistry",
    "SessionEvent",
    "SideEffect",
    "TaskOutcome",
    "ToolRequest",
    "ToolResult",
]
