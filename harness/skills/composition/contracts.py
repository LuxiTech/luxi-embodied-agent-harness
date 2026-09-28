"""Declarative skill semantics, not task recipes or an automatic planner."""
from copy import deepcopy
import json

PREDICATE_DESCRIPTIONS = {
    'visited': '机器人依赖满足后已到访引用位置的历史事实；离开后仍成立；禁止 entity_id。',
    'at': '机器人最终位置，不表示先到达某处；多个不同最终位置不能同时成立；禁止 entity_id。仅用户要求朝向时 require_heading=true。',
    'holding': '在指定取物目标取得对象且当前仍持有；视觉 target 是搜索区域。',
    'acquired': '本任务在指定取物目标取得对象的历史事实；释放后仍成立。',
    'released': '在指定位置释放对象且独立落置位置、静止验证通过；需后端支持。',
    'placed_on': '对象当前在可信表面上且支撑、位置、朝向、无附着和稳定证据通过；'
                 'depends_on 必须直接包含同一对象的 acquired 条件 ID，间接依赖不足；不能与该对象最终 holding 并存。',
}

# These describe individual skills. No instruction-to-plan mapping belongs here.
DYNAMIC_CONTRACTS = {
    'compose_propose_goal': {
        'preconditions': ['unconfirmed goal'],
        'effects': ['proposal only; user confirmation required; no physical motion'],
        'predicates': PREDICATE_DESCRIPTIONS,
        'references': 'target 引用可信目录；visual_regions 是观察/搜索区域，placement_surfaces 是放置表面。',
    },
    'compose_plan': {
        'preconditions': ['confirmed goal'],
        'effects': ['record new plan revision; invalidate old step receipts; no physical motion'],
        'steps': '每项只关联一个 goal_id；depends_on 引用前面的计划下标。completion 可为 '
                 'localized（有效视觉绑定）、pose（目标位姿及停稳证据）、goal（用户目标成立）。'
                 '持物/取得/释放/放置目标必须由 goal 步骤覆盖；位置目标也可由 pose 覆盖。'
                 '一个步骤可调用多个技能；步骤完成与用户目标完成分别验证。',
    },
    'compose_observe': {
        'preconditions': ['fresh observation'],
        'effects': ['current observation and goal/step feedback; does not create pose completion receipts'],
    },
    'compose_locate': {
        'preconditions': ['visual holding/acquired goal', 'no attached entity',
                          'at annotated observation position, or reobservation within the region after prior binding',
                          'fresh synchronized RGB-D after stationary confirmation'],
        'effects': ['fresh goal/entity/epoch-bound visual binding and pickup pose; no attachment'],
        'invalidated_by': ['expiry', 'relocalization', 'world change', 'pause/resume'],
    },
    'compose_navigate': {
        'preconditions': ['valid goal target pose; visual pickup requires fresh binding',
                          'current matching held entity for placed_on goals', 'runtime path/safety admission'],
        'effects': ['requested position and required heading verified; pose receipt when completion.kind=pose'],
        'target': 'target 必须等于所关联 goal_id 的 target；视觉取物目标由运行时解析到绑定的取物位姿，'
                  '其他目标使用可信引用。一次完成所需位置和朝向。',
        'invariants': ['same attached entity during motion'],
    },
    'compose_face': {
        'preconditions': ['at goal target position', 'valid goal target pose'],
        'effects': ['target heading verified; pose receipt when completion.kind=pose'],
        'invariants': ['same attached entity during motion'],
    },
    'compose_attach': {
        'preconditions': ['holding/acquired goal for backend-supported entity',
                          'valid pickup pose; fresh visual binding when target_source=visual',
                          'current pickup position and heading satisfied', 'stationary confirmed with fresh observation',
                          'no attached entity', 'backend entity available and within operation range'],
        'effects': ['historical acquisition evidence and current attachment for this goal/entity'],
        'interaction_model': 'sim_attachment',
    },
    'compose_place': {
        'preconditions': ['placed_on goal and backend placement support', 'trusted annotated surface',
                          'current matching held entity', 'current surface robot_pose satisfied',
                          'stationary confirmed with fresh observation', 'backend support/space/range checks'],
        'effects': ['simulated placement verified on annotated surface; attachment removed; acquisition history retained'],
        'verification': 'fresh consecutive support/position/upright/detachment/stability evidence; release ack is insufficient',
        'interaction_model': 'sim_placement',
    },
    'compose_release': {
        'preconditions': ['released goal and independent release verification support',
                          'current matching held entity', 'current target pose satisfied', 'stationary confirmed'],
        'effects': ['attachment removed; release alone does not prove released goal'],
    },
    'compose_verify': {
        'preconditions': ['confirmed goal', 'stationary confirmation followed by fresh observation'],
        'effects': ['completed=true only if all confirmed predicates independently verified'],
    },
    'compose_blocked': {
        'preconditions': ['specific missing information/capability/verifier or inability to progress'],
        'effects': ['incomplete with explicit reason; never task completion'],
    },
}

DYNAMIC_DESCRIPTIONS = {
    'compose_navigate': '到关联目标的位置及要求的朝向，内部完成转向、行走和调整；携物时检查附着。',
    'compose_verify': '停车后用新鲜证据检查已确认目标的全部条件；唯一整体完成入口。',
}


def dynamic_model_tools(tools, task):
    """Build task-scoped schemas without mutating registered/fixed descriptors."""
    result = deepcopy(tuple(tools))
    physical = {'compose_navigate', 'compose_face', 'compose_attach', 'compose_release', 'compose_place', 'compose_locate'}
    for tool in result:
        function = tool['function']
        name = function['name']
        parameters = function['parameters']
        if name in DYNAMIC_CONTRACTS:
            contract = {'version': '2', **deepcopy(DYNAMIC_CONTRACTS[name])}
            if name in physical:
                contract['preconditions'] = ['confirmed goal', 'current plan and goal dependencies satisfied',
                    'fresh observation in the same world epoch', *contract.get('preconditions', [])]
                contract['retry'] = 'bounded new attempt with recovery_reason; never automatic replay'
            description = DYNAMIC_DESCRIPTIONS.get(name, function['description'].split('\n组合契约：', 1)[0])
            function['description'] = description + '\n组合契约：' + json.dumps(contract, ensure_ascii=False)
        if name in physical:
            parameters['required'] = list(dict.fromkeys([*parameters.get('required', []), 'goal_id', 'subgoal_index']))
        if name == 'compose_propose_goal':
            items = parameters['properties']['conditions']['items']
            # Operation headings are fixed by the proposal boundary; only
            # navigation goals expose a meaningful model choice.
            items['required'] = [key for key in items['required'] if key != 'require_heading']
            items['properties']['target']['enum'] = list(task.references)
            predicates = items['properties']['predicate']['enum']
            if task.schema_version < 5:
                predicates[:] = [p for p in predicates if p not in {'acquired', 'placed_on'}]
            elif not task.placement_surfaces:
                predicates[:] = [p for p in predicates if p != 'placed_on']
            entity_schema = items['properties'].get('entity_id')
            entities = [e for e in task.supported_entities if e in (entity_schema or {}).get('enum', [])]
            if entities:
                entity_schema['enum'] = entities
            else:
                # JSON Schema enum must not be empty, even on an optional property.
                items['properties'].pop('entity_id', None)
                predicates[:] = [p for p in predicates if p in {'visited', 'at'}]
            if task.schema_version < 4:
                items['properties'].pop('target_source', None)
            elif not task.visual_regions:
                items['properties']['target_source']['enum'] = ['reference']
                items['properties']['target_source']['description'] = '当前目录仅提供固定参考位姿，使用 reference，无需视觉绑定。'
            branches = []
            for predicate in predicates:
                branch = deepcopy(items)
                branch['description'] = PREDICATE_DESCRIPTIONS[predicate]
                branch['properties']['predicate'] = {'type': 'string', 'enum': [predicate]}
                if predicate in {'at', 'visited'}:
                    branch['properties'].pop('entity_id', None)
                else:
                    branch['required'] = [*branch['required'], 'entity_id']
                    # Runtime derives the operation heading before confirmation.
                    branch['properties'].pop('require_heading', None)
                if predicate == 'placed_on':
                    branch['properties']['target']['enum'] = [key for key in task.placement_surfaces if key in task.references]
                    branch['properties']['depends_on']['description'] = (
                        '必须直接列出前面同一 entity_id 的 acquired 条件 ID；只依赖中间位置条件不够。')
                if predicate not in {'holding', 'acquired'} and 'target_source' in branch['properties']:
                    branch['properties']['target_source']['enum'] = ['reference']
                branches.append(branch)
            parameters['properties']['conditions']['items'] = {'anyOf': branches}
        if name == 'compose_plan':
            items = parameters['properties']['subgoals']['items']['anyOf'][1]
            items['required'] = list(dict.fromkeys([*items['required'], 'completion']))
            parameters['properties']['subgoals']['items'] = items
    return result
