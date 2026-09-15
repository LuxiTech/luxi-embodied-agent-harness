"""DimOS tools-only compositions; Agent planning belongs to Harness."""

from __future__ import annotations







from typing import Any








from dimos.agents.mcp.mcp_server import McpServer


from dimos.core.coordination.blueprints import autoconnect



from dimos.hardware.sensors.camera.module import CameraModule

from dimos.mapping.costmapper import CostMapper

from dimos.mapping.voxels import VoxelGridMapper

from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner

from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
)

from dimos.robot.unitree.g1.blueprints.basic.unitree_g1_basic_sim import (
    unitree_g1_basic_sim,
)

from dimos.robot.unitree.g1.blueprints.primitive.unitree_g1_primitive_no_nav import (
    unitree_g1_primitive_no_nav,
)

from dimos.robot.unitree.g1.blueprints.perceptive.unitree_g1_sim import unitree_g1_sim

from harness.integrations.dimos.tool_skills import IsaacStopSkillContainer

from harness.robots.g1.isaac.isaac_costmap import IsaacGlobalCostmapModule

from harness.robots.g1.isaac.isaac_spatial_memory import IsaacPersistentSpatialMemory

from harness.robots.g1.isaac.location_tagging import IsaacLocationTagSkillContainer

from harness.integrations.dimos.local_blueprints import LOCAL_BLUEPRINTS as LOCAL_BLUEPRINTS

from harness.integrations.dimos.local_blueprints import register_luxi_blueprints as register_luxi_blueprints

from harness.evaluation.navigation_benchmark import NavigationBenchmark

from harness.control.navigation_relay import BoundedNavigationRelay

from harness.robots.g1.mujoco.mujoco_navigation_ports import (
    MujocoExplorationCostmapPort,
    MujocoLocationTagSkillContainer,
)

from harness.skills.object_skills import ObjectTaskSkillContainer


from harness.skills.rgbd_skills import ApproachPersonSkillContainer, HeadDepthSource

from harness.skills.vlm_observation import VlmObservationSkill

from harness.robots.world_adapter import get_world_adapter

MUJOCO_WORLD = get_world_adapter("mujoco")

ISAAC_WORLD = get_world_adapter("isaac-g1")

def _shared_visual_task_modules(world: Any) -> tuple[Any, ...]:
    """The single RGB-D skill layer used by every World Adapter."""

    overrides: dict[str, Any] = {}
    if world.adapter_id == "mujoco":
        # A cold 180-degree sensor acquisition produces a safe first route at
        # about 0.95 m in home_complex.  The former 0.70 m cap rejected that
        # route even though its complete 0.75 m clearance disk was known free.
        # This widens only the bounded candidate distance; it does not weaken
        # clearance or bypass the live dynamic-costmap intersection.
        overrides["frontier_max_goal_path_distance_m"] = 1.0
        # MuJoCo's replanning state publication can trail entry into the
        # odometry checkpoint. Keep the planner active for its full bounded
        # evidence window instead of converting proximity into arrival.
        overrides["frontier_arrival_grace_s"] = 2.0
    return (
        VlmObservationSkill.blueprint(),
        ApproachPersonSkillContainer.blueprint(
            camera_info=world.camera_info,
            **overrides,
        ),
    )

def _mcp_server_blueprint() -> Any:
    """Align MCP's internal RPC deadline with the public object-search budget."""

    return McpServer.blueprint(
        rpc_timeouts={
            "object_search": 240.0,
            "explore_frontiers": 195.0,
            "navigate_with_text": 195.0,
        }
    )

luxi_g1_tools_sim = autoconnect(
    unitree_g1_sim,
    HeadDepthSource.blueprint(),
    *_shared_visual_task_modules(MUJOCO_WORLD),
    ObjectTaskSkillContainer.blueprint(
        camera_info=MUJOCO_WORLD.camera_info,
        world_adapter=MUJOCO_WORLD.adapter_id,
        global_semantic_map=True,
        allow_backend_seeded_depth_verification=True,
    ),
    MujocoExplorationCostmapPort.blueprint(),
    _mcp_server_blueprint(),
    MujocoLocationTagSkillContainer.blueprint(),
    MUJOCO_WORLD.motion_module.blueprint(),
).disabled_modules(CameraModule).global_config(n_workers=12)

luxi_g1_isaac_tools_sim = autoconnect(
    unitree_g1_primitive_no_nav,
    ISAAC_WORLD.connection_module.blueprint(),
    IsaacGlobalCostmapModule.blueprint(),
    ReplanningAStarPlanner.blueprint(
        robot_width=0.30,
        robot_rotation_diameter=0.60,
    ),
    BoundedNavigationRelay.blueprint(
        max_planar_speed=0.40,
        max_yaw_rate=0.80,
        isaac_gait=True,
    ),
    IsaacPersistentSpatialMemory.blueprint(),
    *_shared_visual_task_modules(ISAAC_WORLD),
    _mcp_server_blueprint(),
    IsaacStopSkillContainer.blueprint(),
    ISAAC_WORLD.motion_module.blueprint(),
    IsaacLocationTagSkillContainer.blueprint(),
).disabled_modules(
    CameraModule,
    VoxelGridMapper,
    CostMapper,
    WavefrontFrontierExplorer,
).global_config(n_workers=10)

luxi_g1_navigation_benchmark = autoconnect(
    unitree_g1_basic_sim,
    BoundedNavigationRelay.blueprint(),
    NavigationBenchmark.blueprint(),
).disabled_modules(CameraModule).global_config(n_workers=11)
