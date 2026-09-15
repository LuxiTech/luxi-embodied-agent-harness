"""DimOS tool protocol wrappers over robot-owned services."""
import json
from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.core.module import Module
from harness.robots.g1.isaac.stop_service import IsaacStopControlChannel

class IsaacStopSkillContainer(Module):
    """Expose the unchanged no-argument ``stop_robot`` capability to MCP."""

    @skill(uses=[CAP_MOVEMENT])
    def stop_robot(self) -> str:
        """Immediately stop the robot through the unified safety owner."""

        result = IsaacStopControlChannel().request_stop()
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
