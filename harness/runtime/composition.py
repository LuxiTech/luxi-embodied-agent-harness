"""Production composition; concrete services are wired only here."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .capabilities import AgentScopeResolver, LuxiCapabilityRegistry
from .contracts import (
    CapabilityDescriptor,
    RetryPolicy,
    SideEffect,
)
from .tool_pipeline import LuxiToolRegistry


@contextmanager
def dashboard_composition(*, backend, project_root, task_paths=(), locations_path=None):
    """Own dashboard composition configuration and its per-run sensor manifest.

    MuJoCo G1 uses the existing candidate contract; other backends and blind
    evaluation retain their accepted tools. No production gate is promoted.
    """
    import json
    import os
    import tempfile
    blind = os.environ.get("LUXI_BLIND_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    if backend != "mujoco" or blind:
        if task_paths or locations_path:
            raise ValueError("组合任务和位置配置仅支持非盲测 MuJoCo G1")
        yield {}
        return
    from .composition_goals import pose_value
    from .task_state import ComposedTask
    from .composed_backend import mujoco_candidate_backend

    locations = ({k: pose_value(v) for k, v in json.loads(Path(locations_path).read_text()).items()}
                 if locations_path else {})
    tasks = {}
    for path in task_paths:
        task = ComposedTask.from_payload(json.loads(Path(path).read_text()))
        if task.release or task.entity_id not in (None, "water_bottle"):
            raise ValueError("固定组合任务只支持位置任务和 water_bottle 附着运输")
        if task.task_key in tasks:
            raise ValueError("task_key 必须唯一")
        tasks[task.task_key] = task

    def make_backend(skills):
        value = mujoco_candidate_backend(skills)
        value.references = locations
        value.location_catalog_path = Path(project_root) / "config/composed/home_complex-locations.json"
        return value

    keys = ("LUXI_COMPOSED_DEV", "LUXI_SIM_ATTACHMENT_DEV", "LUXI_COMPOSED_SHM_MANIFEST")
    previous = {key: os.environ.get(key) for key in keys}
    with tempfile.TemporaryDirectory(prefix="luxi-composed-") as directory:
        try:
            os.environ.update(LUXI_COMPOSED_DEV="1", LUXI_SIM_ATTACHMENT_DEV="1",
                              LUXI_COMPOSED_SHM_MANIFEST=str(Path(directory) / "shm.json"))
            yield {"composed_backend": make_backend, "composed_tasks": tasks}
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


READ_ONLY_TOOLS = frozenset(
    {
        "observe_environment",
        "get_dimos_status",
        "analyze_scene",
        "find_visual_target",
        "verify_visual_condition",
    }
)

TERMINAL_TOOLS = frozenset(
    {
        "move_distance",
        "turn_around",
        "walk_room_loop",
        "navigate_with_text",
        "navigate_to_pose",
        "navigate_to_tag",
        "explore_frontiers",
        "object_search",
        "follow_person",
        "approach_visual_target",
        "approach_person",
        "fetch_object",
        "inspect_warehouse",
    }
)

VERIFIERS = {
    "turn_around": "turn_verified",
    "walk_room_loop": "room_loop_verified",
    "navigate_with_text": "text_navigation_verified",
    "navigate_to_pose": "planner_goal_reached",
    "navigate_to_tag": "planner_goal_reached",
    "explore_frontiers": "frontier_exploration_verified",
    "object_search": "verification_observed",
    "follow_person": "follow_verified",
    "approach_visual_target": "verification_observed",
    "approach_person": "verification_observed",
    "fetch_object": "attachment_retained",
}




def descriptor_from_tool_spec(
    spec: Mapping[str, Any], *, backend: str
) -> CapabilityDescriptor:
    function = spec.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("robot tool is missing function descriptor")
    name = str(function.get("name", "")).strip()
    schema = function.get("parameters", {"type": "object"})
    if not name or not isinstance(schema, Mapping):
        raise ValueError("robot tool name/schema is invalid")
    read_only = name in READ_ONLY_TOOLS
    return CapabilityDescriptor(
        capability_id=name,
        version="v1",
        provider="robot-skills",
        input_schema=dict(schema),
        description=str(function.get("description", "")),
        backends=frozenset({backend}),
        side_effect=SideEffect.READ_ONLY if read_only else SideEffect.PHYSICAL,
        exclusive=not read_only,
        resources=frozenset({f"robot-motion:{backend}"}) if not read_only else frozenset(),
        timeout_s=(
            240.0
            if name in {"fetch_object", "inspect_warehouse"} or backend == "mujoco-go2"
            else 198.0
            if name in {"explore_frontiers", "navigate_with_text", "object_search", "follow_person", "navigate_to_pose", "navigate_to_tag"}
            else 210.0
            if name == "walk_room_loop"
            else 75.0
        ),
        retry=RetryPolicy.NEVER,
        terminal=name in TERMINAL_TOOLS,
        safety_policy="safety-v1" if not read_only else None,
        verifier=VERIFIERS.get(name),
    )




@dataclass(frozen=True)
class RuntimeComposition:
    capabilities: LuxiCapabilityRegistry
    tools: LuxiToolRegistry
    scopes: AgentScopeResolver


def empty_composition(**resolver_kwargs: Any) -> RuntimeComposition:
    capabilities = LuxiCapabilityRegistry()
    tools = LuxiToolRegistry(capabilities)
    return RuntimeComposition(
        capabilities,
        tools,
        AgentScopeResolver(capabilities, **resolver_kwargs),
    )


def create_agent_runtime_service(events, monitor, *, project_root, backend,
                                 runtime_host=None, long_task_runner=None,
                                 navigation_stop_callback=None, model_provider=None,
                                 model_config=None, skills=None, safety=None,
                                 enabled_tools=None, native_broker=None, command_gateway=None, native_motion_port=None,
                                 composed_backend=None, composed_tasks=None, scene_id_provider=None):
    """Compose one loop and one registry/pipeline without an old Agent bridge."""
    import os
    from .agent_loop import LuxiAgentLoop
    from .agent_service import AgentRuntimeService
    from .capability_policy import configured_cutover_tools, runtime_fenced_cutover_tools
    from .command_gateway import (MonitorStopGateway, ExistingSafeCommandMotionPort,
                                  RuntimeCommandFence, RuntimeFencedPhysicalToolAdapter,
                                  SafetyStopIntentAdapter, GatewayPhysicalToolAdapter)
    from .context import RobotContextProvider
    from .contracts import AgentScope, LoopLimits
    from .safety_kernel import LuxiSafetyKernel
    from .task_policy import MAX_PLANNING_STEPS, MAX_TOOL_CALLS, _forced_tool_for_instruction, trusted_tool_arguments
    from .tool_catalog import G1_AGENT_CONTRACT, ISAAC_AGENT_CONTRACT
    from .tool_adapters import SkillToolAdapter, safety_observation
    from .tool_pipeline import LuxiToolPipeline
    from harness.integrations.qwen.config import QwenModelConfig
    from harness.skills.robot_tools import G1RobotTools

    store, session_id = events.session_store, events.session_id
    if store is None or session_id is None:
        raise RuntimeError('Agent Harness requires an attached SessionStore')
    from .configuration import validate_runtime_environment
    validate_runtime_environment(os.environ)
    config = model_config or QwenModelConfig.from_environment()
    if skills is None:
        if backend == 'mujoco-go2':
            from harness.robots.go2.go2_tools import Go2RobotTools
            skills = Go2RobotTools(events, monitor, project_root, paths=getattr(getattr(monitor, "probe", None), "paths", None))
        else:
            skills = G1RobotTools(events, monitor, project_root, backend=backend,
                                  long_task_runner=long_task_runner)
    skills.native_broker = native_broker
    physical = (frozenset(enabled_tools) if enabled_tools is not None else
                configured_cutover_tools(project_root, backend,
                                         os.environ.get('LUXI_PHYSICAL_PIPELINE_TOOLS'),
                                         setting='LUXI_PHYSICAL_PIPELINE_TOOLS'))
    if callable(composed_backend):
        composed_backend = composed_backend(skills)
    if composed_backend is not None:
        if composed_backend.backend != backend or skills.blind_mode:
            raise ValueError("组合候选后端不匹配或处于盲测模式")
        if runtime_host is None:
            raise ValueError("组合物理候选需要 RuntimeHost")
    names = frozenset(spec['function']['name'] for spec in skills._tools)
    enabled = names & (READ_ONLY_TOOLS | physical)
    if 'tag_location' in names:
        enabled |= {'tag_location'}
    robot_id = runtime_host.robot_id if runtime_host else {'mujoco': 'g1-01', 'isaac-g1': 'g1-01', 'mujoco-go2': 'go2-01'}[backend]
    composed_names = set()
    composition = empty_composition(robot_capabilities=lambda _: enabled | composed_names,
                                    robot_backends=lambda _: backend)
    gateway = command_gateway or MonitorStopGateway(skills, emergency_stop_callback=navigation_stop_callback)
    safety = safety or LuxiSafetyKernel(gateway)
    if composed_backend is not None:
        composed_backend.scene_id_provider = scene_id_provider or (lambda: None)
        composed_backend.world_revision = lambda: runtime_host.boot_epoch
        composed_backend.emergency_stop = lambda reason: safety.stop(robot_id, reason)
    if runtime_host is not None and hasattr(runtime_host, 'bind_emergency_stop') and not runtime_host.emergency_stop_bound:
        runtime_host.bind_emergency_stop(lambda reason: safety.stop(robot_id, reason))
    fenced = enabled & runtime_fenced_cutover_tools(project_root, backend)
    fence = None
    if fenced or native_motion_port is not None:
        if runtime_host is None:
            raise RuntimeError('Physical command ownership requires RobotRuntimeHost')
        motion_port = ExistingSafeCommandMotionPort(gateway, fenced, native_port=native_motion_port)
        runtime_host.bind_motion_port(motion_port, capability_ids=fenced | ({"operator_manual_velocity", "stop_robot", "stop_navigation"} if native_motion_port is not None else set()))
        navigation_port = None
        if backend == 'isaac-g1':
            from .isaac_navigation import IsaacG1NavigationPort
            navigation_tools = fenced & IsaacG1NavigationPort.supported_capabilities
            if navigation_tools:
                navigation_port = IsaacG1NavigationPort(robot_id=robot_id)
                navigation_port.bind_navigation_executor(motion_port.execute)
                runtime_host.bind_navigation_port(navigation_port, capability_ids=navigation_tools)
        fence = RuntimeCommandFence(runtime_host, motion_port, fenced_tools=fenced, navigation_port=navigation_port)
    for spec in skills._tools:
        name = spec['function']['name']
        if name not in enabled:
            continue
        descriptor = descriptor_from_tool_spec(spec, backend=backend)
        if name == 'tag_location':
            from dataclasses import replace
            descriptor = replace(descriptor, side_effect=SideEffect.NONE, exclusive=False,
                                 resources=frozenset(), safety_policy=None)
        delegate = (SafetyStopIntentAdapter() if name in {'stop_robot', 'stop_navigation'} else
                    RuntimeFencedPhysicalToolAdapter(fence) if name in fenced else
                    GatewayPhysicalToolAdapter(gateway, name) if descriptor.side_effect is SideEffect.PHYSICAL else None)
        composition.tools.register(descriptor, SkillToolAdapter(skills, delegate))
    if composed_backend is not None:
        from harness.skills.composed_tasks import ComposedSkills
        composition_skills = ComposedSkills(composed_backend, store, safety)
        for descriptor in composition_skills.descriptors():
            composition.tools.register(descriptor, composition_skills)
            composed_names.add(descriptor.capability_id)
    execution_fence = fence
    if composed_names:
        from .composed_backend import MultiplexExecutionFence
        composition_fence = RuntimeCommandFence(runtime_host, None, fenced_tools=frozenset({
            "compose_navigate", "compose_face", "compose_attach", "compose_release", "compose_place", "compose_locate"}))
        execution_fence = MultiplexExecutionFence(fence, composition_fence)
    pipeline = LuxiToolPipeline(composition.tools, events=store, safety=safety,
                               observation=lambda _: safety_observation(skills._observation),
                               execution_fence=execution_fence)

    def prepare_turn():
        skills._cancel.clear()
        skills._observed_this_turn = True
        skills._visual_branch_failed = False

    probe = getattr(monitor, 'probe', None)
    context = RobotContextProvider(store, observe=skills._observation, prepare_turn=prepare_turn,
                                   blind_mode=skills.blind_mode,
                                   camera=getattr(probe, 'camera_jpeg', None), vision_enabled=config.vision_enabled)
    system_prompt = ISAAC_AGENT_CONTRACT if backend == 'isaac-g1' else G1_AGENT_CONTRACT
    if backend == 'mujoco-go2':
        from harness.robots.go2.go2_tools import GO2_AGENT_CONTRACT
        system_prompt = GO2_AGENT_CONTRACT
    injected = model_provider is not None
    if model_provider is None:
        def tool_choice(request):
            if request.metadata.get('execution_mode') == 'composed':
                return 'auto'
            forced = _forced_tool_for_instruction(str(request.metadata.get('instruction', '')))
            if request.metadata.get('planning_step') == 1 and forced in enabled:
                return {'type': 'function', 'function': {'name': forced}}
            return 'auto'
        from harness.integrations.qwen.model import create_qwen_model_provider
        from harness.skills.composed_tasks import COMPOSED_PROMPT
        from harness.skills.composition.prompt import dynamic_prompt
        prompt_for_request = lambda request: ((dynamic_prompt(request) if request.metadata.get("goal_kind") == "dynamic" else COMPOSED_PROMPT) if request.metadata.get("execution_mode") == "composed" else system_prompt)
        model_provider = create_qwen_model_provider(config, system_prompt=prompt_for_request, tool_choice=tool_choice)
    scope = AgentScope(agent_id='luxi-agent', session_id=session_id, robot_ids=frozenset({robot_id}),
                       allowed_capabilities=enabled, budget_steps=MAX_PLANNING_STEPS,
                       budget_tools=MAX_TOOL_CALLS, safety_policy=safety.policy.revision)
    loop = LuxiAgentLoop(model=model_provider, tools=pipeline, capabilities=composition.scopes,
                        events=store, safety=safety, context=context,
                        limits=LoopLimits(64 if composed_names else MAX_PLANNING_STEPS,
                                          64 if composed_names else MAX_TOOL_CALLS),
                        argument_policy=trusted_tool_arguments,
                        composed_observe=getattr(composed_backend, "observe", None),
                        boot_epoch_provider=(lambda: runtime_host.boot_epoch) if runtime_host else None)
    composed_scope = None
    if composed_names:
        from dataclasses import replace
        composed_scope = replace(scope, allowed_capabilities=frozenset(composed_names),
                                 budget_steps=64, budget_tools=64)
    return AgentRuntimeService(loop, scope, store, model=config.model, model_provider="qwen",
                               configuration_error=(lambda: None) if injected else config.configuration_error,
                               composed_scope=composed_scope, composed_tasks=composed_tasks,
                               composed_reference_provider=getattr(composed_backend, "reference_catalog", None),
                               on_cancel=skills._cancel.set,
                               on_close=getattr(model_provider, 'close', None))
