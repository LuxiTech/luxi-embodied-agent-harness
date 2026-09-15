"""Robot-specific declarations kept outside the reusable AgentOS core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformCapabilities:
    """Backend evidence contract shared by UI, agents, and skill discovery."""

    rgb: bool
    aligned_depth: bool
    odometry: bool
    bounded_motion: bool
    pose_reset: bool
    third_person: bool
    navigation: bool
    costmap: bool
    costmap_display: bool
    recovery_costmap: bool
    critical_recovery: bool
    lidar_safety: bool
    scene_seed: bool
    scene_person: bool
    manipulation: bool
    object_fetch: bool
    multi_robot: bool
    shared_mapping: bool
    shared_localization: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            name: bool(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class RobotProfile:
    name: str
    backend: str
    robot_model: str
    base_blueprint: str
    tools_blueprint: str
    benchmark_blueprint: str | None
    capabilities: PlatformCapabilities
    generic_skills: tuple[str, ...]
    robot_skills: tuple[str, ...]
    agent_tools: tuple[str, ...]

    @property
    def navigation_verified(self) -> bool:
        return self.capabilities.navigation

    @property
    def skill_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.generic_skills, *self.robot_skills)))


G1_SIMULATION = RobotProfile(
    name="g1",
    backend="mujoco",
    robot_model="unitree_g1",
    base_blueprint="unitree-g1-sim",
    tools_blueprint="luxi-g1-tools-sim",
    benchmark_blueprint="luxi-g1-navigation-benchmark",
    capabilities=PlatformCapabilities(
        rgb=True,
        aligned_depth=True,
        odometry=True,
        bounded_motion=True,
        pose_reset=True,
        third_person=True,
        navigation=True,
        costmap=True,
        costmap_display=True,
        recovery_costmap=True,
        critical_recovery=True,
        lidar_safety=True,
        scene_seed=True,
        scene_person=True,
        manipulation=True,
        object_fetch=True,
        multi_robot=False,
        shared_mapping=False,
        shared_localization=False,
    ),
    generic_skills=(
        "analyze_scene",
        "find_visual_target",
        "verify_visual_condition",
        "navigate_with_text",
        "explore_frontiers",
        "tag_location",
        "approach_person",
        "navigate_to",
        "locate_entity",
        "approach_entity",
        "grasp_entity",
        "prepare_carry_entity",
        "carry_entity",
        "place_entity",
        "fetch_object",
    ),
    robot_skills=("move", "relative_move", "reset_pose"),
    agent_tools=(
        "observe_environment",
        "get_dimos_status",
        "move_robot",
        "stop_robot",
        "turn_around",
        "walk_room_loop",
        "navigate_with_text",
        "explore_frontiers",
        "tag_location",
        "analyze_scene",
        "find_visual_target",
        "verify_visual_condition",
        "approach_visual_target",
        "approach_person",
        "fetch_object",
    ),
)


G1_ISAAC_SIMULATION = RobotProfile(
    name="g1-isaac",
    backend="isaac-g1",
    robot_model="unitree_g1",
    base_blueprint="unitree-g1-primitive-no-nav",
    tools_blueprint="luxi-g1-isaac-tools-sim",
    benchmark_blueprint=None,
    capabilities=PlatformCapabilities(
        rgb=True,
        aligned_depth=True,
        odometry=True,
        bounded_motion=True,
        pose_reset=True,
        third_person=True,
        navigation=True,
        costmap=True,
        costmap_display=True,
        recovery_costmap=True,
        critical_recovery=True,
        lidar_safety=True,
        scene_seed=False,
        scene_person=True,
        manipulation=False,
        object_fetch=False,
        multi_robot=False,
        shared_mapping=False,
        shared_localization=False,
    ),
    generic_skills=(
        "analyze_scene",
        "find_visual_target",
        "verify_visual_condition",
        "navigate_with_text",
        "explore_frontiers",
        "object_search",
        "approach_person",
        "follow_person",
        "stop_following",
        "tag_location",
        "navigate_to_pose",
        "navigate_to_tag",
        "stop_navigation",
    ),
    robot_skills=(
        "move",
        "relative_move",
        "move_distance",
        "turn_around",
        "reset_pose",
    ),
    agent_tools=(
        "observe_environment",
        "get_dimos_status",
        "move_robot",
        "move_distance",
        "stop_robot",
        "turn_around",
        "tag_location",
        "navigate_to_pose",
        "navigate_to_tag",
        "stop_navigation",
        "analyze_scene",
        "find_visual_target",
        "verify_visual_condition",
        "navigate_with_text",
        "explore_frontiers",
        "object_search",
        "approach_person",
        "follow_person",
    ),
)


GO2_MUJOCO_SIMULATION = RobotProfile(
    name="go2",
    backend="mujoco-go2",
    robot_model="unitree_go2",
    base_blueprint="luxi-go2-hikrobot-mid360",
    tools_blueprint="luxi-go2-hikrobot-mid360",
    benchmark_blueprint=None,
    capabilities=PlatformCapabilities(
        rgb=True,
        aligned_depth=False,
        odometry=True,
        bounded_motion=True,
        pose_reset=False,
        third_person=True,
        navigation=False,
        costmap=False,
        costmap_display=True,
        recovery_costmap=False,
        critical_recovery=False,
        lidar_safety=True,
        scene_seed=False,
        scene_person=True,
        manipulation=False,
        object_fetch=False,
        multi_robot=False,
        shared_mapping=False,
        shared_localization=False,
    ),
    generic_skills=("inspect_warehouse", "object_search", "follow_person"),
    robot_skills=(),
    agent_tools=("inspect_warehouse", "object_search", "follow_person"),
)


_PROFILES = {
    G1_SIMULATION.name: G1_SIMULATION,
    G1_ISAAC_SIMULATION.name: G1_ISAAC_SIMULATION,
    GO2_MUJOCO_SIMULATION.name: GO2_MUJOCO_SIMULATION,
}
_BACKEND_PROFILES = {profile.backend: profile for profile in _PROFILES.values()}


def get_robot_profile(name: str) -> RobotProfile:
    """Resolve explicitly; never make a different robot inherit G1 controls."""

    key = name.strip().lower()
    try:
        return _PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"Unsupported robot profile {name!r}; supported profiles: {supported}"
        ) from exc


def get_backend_profile(backend: str) -> RobotProfile:
    key = backend.strip().lower()
    try:
        return _BACKEND_PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_BACKEND_PROFILES))
        raise ValueError(
            f"Unsupported simulator backend {backend!r}; supported backends: "
            f"{supported}"
        ) from exc
