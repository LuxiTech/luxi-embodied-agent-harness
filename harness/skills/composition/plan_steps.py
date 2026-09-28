"""Verified execution milestones, separate from immutable user goals."""

COMPLETION_SCHEMA = {
    'type': 'object',
    'properties': {
        'kind': {'type': 'string', 'enum': ['pose', 'goal', 'localized'],
                 'description': 'pose: 到该目标地点并停稳；goal: 满足该用户目标；localized: 当前有效的视觉对象绑定，不代表到位或持物。'},
        'goal_id': {'type': 'string'},
    },
    'required': ['kind', 'goal_id'], 'additionalProperties': False,
}


# 校验步骤声明的完成条件，并返回它能覆盖的用户目标 ID。
# pose 可覆盖位置目标，但到取物点本身不能覆盖 holding/released。
def validate_completion(node, conditions):
    completion = node.get('completion')
    if 'completion' not in node:
        return set(node['goal_ids'])  # Historical plans retain whole-goal semantics.
    if (not isinstance(completion, dict) or set(completion) != {'kind', 'goal_id'}
            or completion['kind'] not in {'pose', 'goal', 'localized'}
            or not isinstance(completion['goal_id'], str)
            or completion['goal_id'] not in node['goal_ids']):
        raise ValueError('步骤 completion 需要 kind=pose/goal/localized 与本步骤关联的 goal_id')
    if len(node['goal_ids']) != 1:
        raise ValueError('显式步骤只关联一个目标；多个目标请拆为多个步骤')
    condition = next(c for c in conditions if c['id'] == completion['goal_id'])
    if completion['kind'] == 'localized':
        if condition.get('target_source') != 'visual':
            raise ValueError('localized 只用于视觉取物目标')
        return set()
    # A verified pose fully covers a position goal, but never covers holding/release.
    covers_goal = completion['kind'] == 'goal' or condition['predicate'] in {'at', 'visited'}
    return {completion['goal_id']} if covers_goal else set()


# 将步骤完成与用户目标分开：pose 使用本计划的到位回执，goal 使用目标验证结果。
# step_status 保留到位历史；step_ready 还检查当前位姿，离开取物点后不能直接附着。
def step_feedback(task, observation, state, goals, at):
    """A pose milestone requires a current-plan receipt AND current physical readiness."""
    from .grounding import target_pose, binding_for, is_visual
    statuses, ready, details = {}, {}, {}
    conditions = {c['id']: c for c in task.conditions}
    for index, node in enumerate(state['plan']):
        key = str(index)
        completion = node.get('completion')
        if completion and completion['kind'] == 'localized':
            condition = conditions[completion['goal_id']]
            receipt = state.get('step_evidence', {}).get(key, {})
            binding = binding_for(task, condition, state)
            verified = (receipt.get('plan_revision') == state['plan_revision']
                        and receipt.get('world_revision') == task.world_revision
                        and receipt.get('goal_id') == condition['id'])
            available = bool(verified and binding and receipt.get('binding_id') == binding['binding_id'])
            missing = [] if available else ['fresh_visual_binding']
        elif completion and completion['kind'] == 'pose':
            condition = conditions[completion['goal_id']]
            receipt = state.get('step_evidence', {}).get(key, {})
            verified = (receipt.get('plan_revision') == state['plan_revision']
                        and receipt.get('world_revision') == task.world_revision
                        and receipt.get('goal_id') == condition['id'])
            pose = target_pose(task, condition, state)
            binding = binding_for(task, condition, state) if is_visual(condition) else None
            if is_visual(condition):
                verified = bool(verified and binding and receipt.get('binding_id') == binding['binding_id'])
            available = verified and pose is not None and at(observation, pose, yaw=condition['require_heading'])
            # A visited goal is a historical arrival. Leaving its observation
            # point is expected during search/approach, not loss of permission.
            # Holding preparation poses still require current physical readiness.
            if condition['predicate'] == 'visited':
                available = verified and goals.get(condition['id'], False)
            missing = [] if available else ['pose_evidence' if not verified else 'current_pose']
        else:
            ids = [completion['goal_id']] if completion else node['goal_ids']
            available = verified = all(goals.get(g, False) for g in ids)
            missing = [g for g in ids if not goals.get(g, False)]
        statuses[key], ready[key] = bool(verified), bool(available)
        details[key] = {'label': node['label'], 'completion': completion or {'kind': 'legacy_goals'},
                        'completed': bool(verified), 'ready_for_dependents': bool(available),
                        'unmet_conditions': missing}
    return {'step_status': statuses, 'step_ready': ready, 'step_details': details}
