"""Independent dynamic skill adapters and evidence-based goal verification."""
import json
import math
import time
from harness.robots.composed_pose import POSITION_TOLERANCE_M, YAW_TOLERANCE_RAD

from harness.runtime.composition_goals import DynamicTask, validate_conditions, proposal_digest
from harness.runtime.task_state import task_projection
from harness.skills.composed_tasks import _fresh, _schema
from .plan_steps import validate_completion, step_feedback
from .grounding import is_visual, target_pose, binding_for, validate_visual


def at(observation, pose, yaw=True):
    actual = observation['pose']
    return (math.dist(actual[:2], pose[:2]) <= POSITION_TOLERANCE_M and
            (not yaw or abs(math.atan2(math.sin(actual[2]-pose[2]), math.cos(actual[2]-pose[2]))) <= YAW_TOLERANCE_RAD))


# 从真实观测计算用户目标：visited 保存历史到访，at/holding 检查当前状态。
# holding 还要求本任务曾在声明的取物点验证取得该对象，不能只凭对象名称判定持物。
def goal_evidence(task, observation, state):
    """Only actual observations establish facts. Dependencies gate historical visits."""
    evidence = dict(state.get('goal_evidence', {}))
    current = {}
    for condition in task.conditions:
        key, predicate = condition['id'], condition['predicate']
        pose = task.references[condition['target']]
        deps = all(current.get(dep, False) for dep in condition['depends_on'])
        if predicate == 'visited':
            valid = key in evidence or (deps and at(observation, pose, yaw=condition['require_heading']))
            if valid and key not in evidence:
                evidence[key] = {'timestamp_monotonic': observation['timestamp_monotonic'],
                                 'world_revision': task.world_revision, 'predicate': predicate}
        elif predicate == 'at':
            valid = deps and at(observation, pose, yaw=condition['require_heading'])
        elif predicate == 'acquired':
            valid = any(step['result'].get('acquired_goal') == key for step in state['steps'])
        elif predicate == 'placed_on':
            from harness.robots.composed_placement import placement_matches
            valid = (deps and observation.get('attached_entity') is None
                     and placement_matches(observation.get('placement'), task.placement_surfaces[condition['target']], condition['entity_id'])
                     and any(step['result'].get('placed_goal') == key for step in state['steps']))
        elif predicate == 'holding':
            valid = deps and observation.get('attached_entity') == condition['entity_id']
            # Acquisition must have been checked at its declared pickup, not inferred from a name.
            valid = valid and any(step['result'].get('acquired_goal') == key for step in state['steps'])
        else:
            xy = observation.get('object_xy')
            valid = (deps and observation.get('released_entity') == condition['entity_id']
                     and observation.get('attached_entity') is None and observation.get('object_stationary') is True
                     and isinstance(xy, (list, tuple)) and len(xy) == 2
                     and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in xy)
                     and math.dist(xy, pose[:2]) <= 1.0)
        current[key] = bool(valid)
    return current, evidence


# 向模型返回目标状态与位置/朝向残差；这是诊断信息，不是新的执行许可。
def goal_feedback(task, observation, current, state=None):
    details = {}
    for condition in task.conditions:
        pose = target_pose(task, condition, state or {})
        if pose is None:
            details[condition['id']] = {'satisfied': current[condition['id']],
                'target': condition['target'], 'target_source': 'visual',
                'unmet_conditions': [] if current[condition['id']] else ['visual_localization', 'object_evidence']}
            continue
        actual = observation['pose']
        position_error = math.dist(actual[:2], pose[:2])
        heading_error = abs(math.atan2(math.sin(pose[2]-actual[2]), math.cos(pose[2]-actual[2])))
        missing = []
        if not current[condition['id']]:
            if any(not current.get(d, False) for d in condition['depends_on']):
                missing.append('dependencies')
            if position_error > POSITION_TOLERANCE_M:
                missing.append('position')
            if condition['require_heading'] and heading_error > YAW_TOLERANCE_RAD:
                missing.append('heading')
            if condition['predicate'] in {'holding', 'released', 'acquired', 'placed_on'}:
                missing.append('object_evidence')
        details[condition['id']] = {
            'satisfied': current[condition['id']], 'target': condition['target'],
            'require_heading': condition['require_heading'],
            'position_error_m': position_error, 'heading_error_rad': heading_error,
            'position_tolerance_m': POSITION_TOLERANCE_M, 'heading_tolerance_rad': YAW_TOLERANCE_RAD if condition['require_heading'] else None,
            'unmet_conditions': missing,
        }
    return {'goal_status': current, 'goal_details': details}


# 校验可调整的步骤计划：目标引用、前向依赖及 pose/goal 完成条件。
# 允许“导航到位→附着”分步，但不允许用到位步骤省略持物/放置的最终要求。
def validated_plan(args, task):
    nodes = args.get('subgoals')
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 24:
        raise ValueError('计划需要 1–24 个结构化子目标')
    ids = {c['id'] for c in task.conditions}
    covered = set()
    terminal_coverage = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or (set(node) - {'label', 'goal_ids', 'depends_on', 'completion'} or not {'label', 'goal_ids', 'depends_on'} <= set(node)):
            raise ValueError('计划项需要 label、goal_ids、depends_on')
        if not isinstance(node['label'], str) or not 1 <= len(node['label'].strip()) <= 200:
            raise ValueError('子目标描述无效')
        goals, deps = node['goal_ids'], node['depends_on']
        if not isinstance(goals, list) or not goals or any(not isinstance(g, str) or g not in ids for g in goals):
            raise ValueError('计划必须引用已确认目标')
        if not isinstance(deps, list) or any(type(d) is not int or not 0 <= d < index for d in deps):
            raise ValueError('计划依赖必须是前面的下标')
        terminal_coverage.update(validate_completion(node, task.conditions))
        for dep in deps:
            prior = nodes[dep]
            if not prior.get('completion') and set(prior['goal_ids']) & set(goals):
                raise ValueError('同一目标的准备/操作步骤须明确 completion：导航用 pose，附着用 goal；不能要求先持物才附着')
        covered.update(goals)
    if covered != ids:
        raise ValueError('计划不能删除用户目标')
    if terminal_coverage != ids:
        raise ValueError('持物/放置目标需要 goal 完成步骤，只有到位不能替代取放物体：' + ', '.join(sorted(ids-terminal_coverage)))
    return nodes


# 动态技能主分发：先处理目标提议/计划，再用新鲜观测检查世界版本、持物和依赖。
# 物理动作调用后端；结果分别更新动作、步骤及用户目标，只有最终验证可 completed=true。
def execute(owner, request, cancel, state):
    result, emit = owner._result, lambda kind, payload: owner._emit(kind, request, payload)
    task = DynamicTask.from_payload(state['goal'])
    name, args = request.capability_id, request.arguments
    if name == 'compose_blocked':
        reason = args.get('reason', '').strip()
        if not reason:
            return result('invalid_input', ok=False)
        emit('task/blocked', {'reason': reason})
        return result('incomplete', ok=False, payload={'blocked_reason': reason})
    # 提议仅生成待确认的 GoalSpec 并结束当前轮；此分支不执行运动。
    if name == 'compose_propose_goal':
        if task.confirmed or state.get('proposed_goal'):
            return result('tool_denied', ok=False, payload={'reason': '目标已固定，不能改写'})
        try:
            conditions = validate_conditions(args.get('conditions'), task.references,
                release_supported=getattr(owner.backend, 'release_supported', False), schema_version=task.schema_version, visual_regions=task.visual_regions,
                placement_surfaces=task.placement_surfaces, place_supported=getattr(owner.backend, 'place_supported', False),
                entity_ids=tuple(e for e in task.supported_entities if e in getattr(owner.backend, "supported_entities", ("water_bottle",))))
        except (ValueError, TypeError) as exc:
            return result('invalid_input', ok=False, payload={'reason': str(exc)})
        proposal = {**task.payload(), 'conditions': conditions, 'confirmed': True}
        records = list(owner.events.iter_task_events(request.session_id, request.task_id))
        proposal['preparation_steps'] = sum(e.event_type == 'model/replied' for e in records)
        proposal['preparation_tools'] = sum(e.event_type == 'tool/started' for e in records)
        proposal['preparation_seconds'] = max(0, time.monotonic()-records[0].monotonic_ns/1e9) if records else 0
        proposal['task_key'] = proposal_digest(proposal)
        emit('task/goal_proposed', {'proposed_goal': proposal})
        return result('goal_proposed', payload={'proposed_goal': proposal,
                      'message': '请核对目标条件；确认前不会执行物理动作。'})
    if not task.confirmed:
        return result('tool_denied', ok=False, payload={'reason': '先提出并确认完整目标'})
    if name == 'compose_plan':
        try:
            nodes = validated_plan(args, task)
            if state['plan_revision'] and not args.get('reason', '').strip():
                raise ValueError('重新规划需要说明原因')
        except (ValueError, TypeError) as exc:
            return result('invalid_input', ok=False, payload={'reason': str(exc)})
        emit('task/plan', {'subgoals': nodes, 'reason': args.get('reason', ''), 'plan_schema_version': 2})
        return result('plan_recorded')

    observation = owner.backend.observe()
    if not _fresh(observation):
        return result('observation_unavailable', ok=False)
    if task.world_revision and observation.get('world_revision') != task.world_revision:
        return result('side_effect_unknown', ok=False, payload={'reason': '世界或运行版本已变化，请重新确认目标'})
    held = state.get('held_entity')
    if held and observation.get('attached_entity') != held:
        return result('side_effect_unknown', ok=False, payload={'reason': '持物关系已丢失'})
    current, evidence = goal_evidence(task, observation, state)
    emit('task/facts', {'goal_evidence': evidence, 'goal_status': current,
                        'observation_timestamp': observation['timestamp_monotonic']})
    steps = step_feedback(task, observation, state, current, at)
    if name == 'compose_observe':
        return result('observation_available', payload={'observation': observation,
                      **goal_feedback(task, observation, current, state), **steps})
    # 整体验收：先停车，再用晚于停车确认的新观测重算全部用户目标。
    if name == 'compose_verify':
        stop = owner.safety.stop(request.robot_id, 'composition_verification')
        cancel.raise_if_cancelled()
        after = owner.backend.observe()
        if (not stop.stationary_confirmed or stop.stationary_confirmed_at is None
                or not _fresh(after, stop.stationary_confirmed_at)):
            return result('verification_failed', ok=False)
        if task.world_revision and after.get('world_revision') != task.world_revision:
            return result('side_effect_unknown', ok=False)
        if held and after.get('attached_entity') != held:
            return result('side_effect_unknown', ok=False)
        current, evidence = goal_evidence(task, after, {**state, 'goal_evidence': evidence})
        emit('task/facts', {'goal_evidence': evidence, 'goal_status': current})
        valid = bool(current) and all(current.values())
        return result('task_verified' if valid else 'verification_failed', completed=valid,
            evidence={'task_verified': valid, 'stationary_confirmed': True,
                      'stationary_confirmed_at': stop.stationary_confirmed_at,
                      'verification_frame_timestamp': after['timestamp_monotonic']},
            payload={**goal_feedback(task, after, current, state), 'observation': after})

    index = args.get('subgoal_index')
    if type(index) is not int or not 0 <= index < len(state['plan']):
        return result('tool_denied', ok=False, payload={'reason': '需要当前计划的 subgoal_index'})
    node = state['plan'][index]
    # 步骤依赖按各自 completion 判断；取物前只需导航到位，不要求 holding 已成立。
    if any(not steps['step_ready'][str(d)] for d in node['depends_on']):
        return result('tool_denied', ok=False, payload={'reason': '前置步骤的完成条件未满足或物理状态已失效，请检查 step_details',
                      **goal_feedback(task, observation, current, state), **steps})
    goal_id = args.get('goal_id')
    condition = next((c for c in task.conditions if c['id'] == goal_id), None)
    if condition is None or goal_id not in node['goal_ids']:
        return result('tool_denied', ok=False, payload={'reason': '动作必须关联当前子目标'})
    # 用户目标依赖仍独立生效：运输目标要求的 holding 不能被步骤回执绕过。
    if not all(current.get(d, False) for d in condition['depends_on']):
        return result('tool_denied', ok=False, payload={'reason': '目标前置条件尚未满足', **goal_feedback(task, observation, current, state)})
    target_name = condition['target']
    target = target_pose(task, condition, state)
    if name == 'compose_locate':
        return locate(owner, request, cancel, state, task, condition, observation, index)
    if target is None:
        return result('tool_denied', ok=False, payload={'reason': '取物位姿尚未定位或已过期；先 compose_locate', **steps})
    if args.get('target', target_name) != target_name:
        return result('invalid_input', ok=False)
    if name == 'compose_face' and not at(observation, target, yaw=False):
        return result('tool_denied', ok=False, payload={'reason': '转向前先到位'})
    if condition['predicate'] == 'placed_on' and name in {'compose_navigate', 'compose_face', 'compose_place'}:
        if observation.get('attached_entity') != condition['entity_id']:
            return result('tool_denied', ok=False, payload={'reason': '运输及放置前必须当前持有目标对象'})
    if name in {'compose_attach', 'compose_release', 'compose_place'}:
        predicates = {'compose_attach': {'holding', 'acquired'}, 'compose_release': {'released'}, 'compose_place': {'placed_on'}}
        if condition['predicate'] not in predicates[name] or (name == 'compose_release' and not getattr(owner.backend, 'release_supported', False)):
            return result('tool_denied', ok=False)
        if name == 'compose_place' and not getattr(owner.backend, 'place_supported', False):
            return result('tool_denied', ok=False)
        stop = owner.safety.stop(request.robot_id, 'composition_handoff')
        observation = owner.backend.observe()
        if not stop.stationary_confirmed or stop.stationary_confirmed_at is None or not _fresh(observation, stop.stationary_confirmed_at) or not at(observation, target):
            return result('verification_failed', ok=False)
        if task.world_revision and observation.get('world_revision') != task.world_revision:
            return result('side_effect_unknown', ok=False)
        attached = observation.get('attached_entity')
        if (name == 'compose_attach' and attached is not None) or (name in {'compose_release', 'compose_place'} and attached != condition['entity_id']):
            return result('tool_denied', ok=False)
    key = f'{name}:{target_name}:{condition.get("entity_id", "-")}'
    count = state['attempts'].get(key, 0)
    if count >= owner.max_attempts:
        return result('incomplete_budget_exhausted', ok=False)
    if count and not args.get('recovery_reason', '').strip():
        return result('tool_denied', ok=False, payload={'reason': '重复动作需要新观察和恢复理由'})
    emit('task/attempt', {'action_key': key, 'subgoal_index': index, 'plan_revision': state['plan_revision'],
                         'recovery_reason': args.get('recovery_reason', ''), 'goal_id': goal_id})
    cancel.raise_if_cancelled()
    if name in {'compose_navigate', 'compose_face'}:
        pose = target if condition['require_heading'] or name == 'compose_face' else (*target[:2], None)
        raw = owner.backend.navigate(pose, cancel, request.deadline_monotonic)
    elif name == 'compose_attach':
        raw = owner.backend.attach(condition['entity_id'], target, cancel, request.deadline_monotonic)
    elif name == 'compose_place':
        raw = owner.backend.place(condition['entity_id'], task.placement_surfaces[target_name], cancel, request.deadline_monotonic)
    elif name == 'compose_release':
        raw = owner.backend.release(condition['entity_id'], cancel, request.deadline_monotonic)
    else:
        return result('tool_denied', ok=False)
    cancel.raise_if_cancelled()
    after = owner.backend.observe()
    if held and name not in {'compose_release', 'compose_place'} and (not _fresh(after) or after.get('attached_entity') != held):
        return result('side_effect_unknown', ok=False)
    if task.world_revision and after.get('world_revision') != task.world_revision:
        return result('side_effect_unknown', ok=False)
    if not raw.get('operation_ok'):
        return result(raw.get('task_status', 'verification_failed'), ok=False, payload={
            'backend_error': raw.get('error'), **goal_feedback(task, after, goal_evidence(task, after, state)[0], state)} if _fresh(after) else {'reason': '缺少新鲜结果观测'})
    completion = node.get('completion', {})
    milestone_stop = None
    if completion.get('kind') == 'pose' and name in {'compose_navigate', 'compose_face'}:
        milestone_stop = owner.safety.stop(request.robot_id, 'composition_step_verification')
        cancel.raise_if_cancelled()
        after = owner.backend.observe()
        if (not milestone_stop.stationary_confirmed or milestone_stop.stationary_confirmed_at is None
                or not _fresh(after, milestone_stop.stationary_confirmed_at)):
            return result('verification_failed', ok=False, payload={'reason': '步骤缺少停车后的新鲜位姿证据'})
        if task.world_revision and after.get('world_revision') != task.world_revision:
            return result('side_effect_unknown', ok=False)
        if held and after.get('attached_entity') != held:
            return result('side_effect_unknown', ok=False)
    valid = _fresh(after, observation['timestamp_monotonic'])
    extra = {}
    if name in {'compose_navigate', 'compose_face'}:
        valid = valid and raw.get('planner_goal_reached') is True and at(after, target, yaw=condition['require_heading'] or name == 'compose_face')
    elif name == 'compose_attach':
        valid = valid and after.get('attached_entity') == condition['entity_id']
        if valid:
            extra = {'acquired_goal': goal_id, 'held_entity': condition['entity_id']}
    elif name == 'compose_place':
        from harness.robots.composed_placement import placement_matches
        valid = (valid and after.get('attached_entity') is None
                 and placement_matches(after.get('placement'), task.placement_surfaces[target_name], condition['entity_id']))
        if valid:
            extra = {'held_entity': None, 'attachment_released': True, 'placed_goal': goal_id}
    else:
        valid = valid and after.get('attached_entity') is None and after.get('released_entity') == condition['entity_id']
        if valid:
            extra = {'held_entity': None, 'attachment_released': True}
    if not valid and name in {'compose_attach', 'compose_release', 'compose_place'}:
        return result('side_effect_unknown', ok=False)
    if not valid:
        return result('verification_failed', payload={'action_verified': False, 'subgoal_verified': False,
                      'observation': after, **(goal_feedback(task, after, goal_evidence(task, after, state)[0], state) if _fresh(after) else {})})
    updated = {**state, 'goal_evidence': evidence,
               'steps': [*state['steps'], {'result': extra}]}
    current, evidence = goal_evidence(task, after, updated)
    emit('task/facts', {'goal_status': current, 'goal_evidence': evidence})
    # 到位回执只在动作验证通过后写入，绑定本计划修订、目标和停车后观测时间。
    if milestone_stop is not None:
        receipt = {'subgoal_index': index, 'plan_revision': state['plan_revision'],
                   'world_revision': task.world_revision, 'goal_id': goal_id,
                   'capability_id': name, 'stationary_confirmed': True,
                   'stationary_confirmed_at': milestone_stop.stationary_confirmed_at,
                   'observation_timestamp': after['timestamp_monotonic']}
        if is_visual(condition):
            receipt['binding_id'] = state['object_bindings'][goal_id]['binding_id']
        emit('task/step_verified', receipt)
        updated['step_evidence'] = {**state.get('step_evidence', {}), str(index): receipt}
    steps = step_feedback(task, after, updated, current, at)
    satisfied = steps['step_status'][str(index)]
    return result('subgoal_verified' if satisfied else 'action_verified',
                  payload={'action_verified': True, 'subgoal_verified': satisfied, 'observation': after,
                           **goal_feedback(task, after, current, state), **steps, **extra})


def locate(owner, request, cancel, state, task, condition, observation, index):
    """一次受预算约束的定位；搜索点为标注，取物 XY 只能来自本轮 RGB-D。"""
    emit = lambda kind, payload: owner._emit(kind, request, payload)
    if not is_visual(condition) or observation.get('attached_entity') is not None:
        return owner._result('tool_denied', ok=False)
    region_pose = task.references[condition['target']]
    previous = state.get('object_bindings', {}).get(condition['id'])
    # First observation is made at the annotated viewpoint; subsequent recovery may
    # reobserve from the measured approach pose in the same bounded search region.
    if not at(observation, region_pose, yaw=False) and not (previous and
            math.dist(observation['pose'][:2], region_pose[:2]) <= task.visual_regions[condition['target']]['search_radius_m']):
        return owner._result('tool_denied', ok=False, payload={'reason': '先导航至已标注厨房观察点'})
    key = f"compose_locate:{condition['target']}:{condition['entity_id']}"
    count = state['attempts'].get(key, 0)
    if count >= owner.max_attempts:
        return owner._result('incomplete_budget_exhausted', ok=False)
    if count and not request.arguments.get('recovery_reason', '').strip():
        return owner._result('tool_denied', ok=False, payload={'reason': '重复定位需要恢复理由'})
    emit('task/attempt', {'action_key': key, 'subgoal_index': index,
        'plan_revision': state['plan_revision'], 'goal_id': condition['id']})
    # The catalog yaw is a camera viewing direction, not a stricter user arrival goal.
    if at(observation, region_pose, yaw=False) and not at(observation, region_pose):
        raw = owner.backend.navigate((*observation['pose'][:2], region_pose[2]), cancel, request.deadline_monotonic)
        if not raw.get('operation_ok') or raw.get('planner_goal_reached') is not True:
            return owner._result(raw.get('task_status', 'verification_failed'), ok=False)
    stop = owner.safety.stop(request.robot_id, 'composition_visual_localization')
    cancel.raise_if_cancelled()
    before = owner.backend.observe()
    if (not stop.stationary_confirmed or stop.stationary_confirmed_at is None
            or not _fresh(before, stop.stationary_confirmed_at)
            or before.get('attached_entity') is not None
            or (at(observation, region_pose, yaw=False) and not at(before, region_pose))
            or (task.world_revision and before.get('world_revision') != task.world_revision)):
        return owner._result('observation_unavailable', ok=False)
    requested_at = time.time()
    raw = owner.backend.locate(condition['entity_id'], cancel, request.deadline_monotonic)
    cancel.raise_if_cancelled()
    after = owner.backend.observe()
    try:
        binding = validate_visual(task, condition, raw, after_wall=requested_at)
        if (not _fresh(after) or after.get('attached_entity') is not None or not at(after, before['pose'])
                or (task.world_revision and after.get('world_revision') != task.world_revision)):
            raise ValueError('定位期间机器人位姿或世界版本变化')
    except (ValueError, TypeError) as exc:
        emit('task/object_invalidated', {'goal_id': condition['id']})
        return owner._result(raw.get('task_status', 'verification_failed') if not raw.get('operation_ok') else 'verification_failed',
                             ok=False, payload={'reason': str(exc)})
    emit('task/object_localized', binding)
    receipt = {'subgoal_index': index, 'plan_revision': state['plan_revision'],
        'world_revision': task.world_revision, 'goal_id': condition['id'],
        'binding_id': binding['binding_id'], 'capability_id': 'compose_locate',
        'stationary_confirmed': True, 'stationary_confirmed_at': stop.stationary_confirmed_at,
        'observation_timestamp': after['timestamp_monotonic']}
    if state['plan'][index].get('completion', {}).get('kind') == 'localized':
        emit('task/step_verified', receipt)
    updated = task_projection(owner.events, request.session_id, request.task_id)
    goals, _ = goal_evidence(task, after, updated)
    steps = step_feedback(task, after, updated, goals, at)
    satisfied = steps['step_status'][str(index)]
    return owner._result('subgoal_verified' if satisfied else 'action_verified', payload={
        'action_verified': True, 'subgoal_verified': satisfied, 'visual_binding': binding,
        'observation': after, **steps, **goal_feedback(task, after, goals, updated)})
