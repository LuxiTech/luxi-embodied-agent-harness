"""Versioned, user-confirmed goals. Models select references, never invent poses."""
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import re

PREDICATES = frozenset({'visited', 'at', 'holding', 'released', 'acquired', 'placed_on'})


def proposal_conditions(conditions):
    """Fix operation headings before confirmation; never rewrite stored goals."""
    if not isinstance(conditions, (list, tuple)):
        return conditions
    return [
        {**item, 'require_heading': True}
        if isinstance(item, dict)
        and isinstance(item.get('predicate'), str)
        and item.get('predicate') in {'holding', 'released', 'acquired', 'placed_on'}
        and ('require_heading' not in item or type(item['require_heading']) is bool)
        else item
        for item in conditions
    ]


def pose_value(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise ValueError('位置必须是有限的 x/y/yaw，单位为米和弧度')
    return tuple(float(v) for v in value)


# 只解析用户明确写出的 name=(x,y,yaw)，不会将 kitchen 等自然地名猜成坐标。
def instruction_references(instruction):
    """Only explicitly named metric poses in user text; no scene/oracle discovery."""
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
    pattern = rf'([\w\-]+)\s*[=＝]\s*[（(]\s*({number})\s*[,，]\s*({number})\s*[,，]\s*({number})\s*[)）]'
    result = {}
    for match in re.finditer(pattern, instruction):
        key, *values = match.groups()
        pose = pose_value([float(v) for v in values])
        if key in result and result[key] != pose:
            raise ValueError(f'位置 {key} 的定义冲突')
        result[key] = pose
    return result


class GoalValidationError(ValueError):
    """All independently detectable goal errors, without rewriting conditions."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__('；'.join(
            f"{', '.join(e['condition_ids']) or 'conditions'}: {e['message']}" for e in errors))


def invalid_proposal_fingerprint(conditions, errors, schema_version):
    """Compare rejected proposals without rewriting or granting goal validity."""
    normalized = []
    for raw in conditions if isinstance(conditions, (list, tuple)) else [conditions]:
        if not isinstance(raw, dict):
            normalized.append(raw)
            continue
        item = dict(raw)
        item.setdefault('depends_on', [])
        if isinstance(item['depends_on'], list) and all(isinstance(d, str) for d in item['depends_on']):
            item['depends_on'] = sorted(item['depends_on'])
        item.setdefault('target_source', 'reference')
        item.setdefault('require_heading', schema_version == 2 or item.get('predicate') in ('holding', 'acquired', 'released', 'placed_on'))
        normalized.append(item)
    encoded = json.dumps({'conditions': normalized, 'errors': errors}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


# Validate the entire proposal so a model can correct all detected conflicts in
# one decision. Structurally unusable fields do not prevent unrelated checks.
def validate_conditions(conditions, references, *, release_supported=False, entity_ids=(), schema_version=3, visual_regions=None, placement_surfaces=None, place_supported=False):
    errors = []

    def error(code, message, ids=(), field='', indices=()):
        errors.append({'code': code, 'condition_ids': list(ids), 'field': field,
                       'indices': list(indices), 'message': message})

    if not isinstance(conditions, (list, tuple)) or not 1 <= len(conditions) <= 24:
        error('condition_count', '需要 1–24 个可验证的目标条件')
        raise GoalValidationError(errors)
    known, clean = set(), []
    operations = {'holding', 'released', 'acquired', 'placed_on'}
    for index, raw in enumerate(conditions):
        if not isinstance(raw, dict):
            error('condition_fields', '目标条件必须是对象', indices=(index,))
            continue
        item = dict(raw)
        key = item.get('id')
        ids = (key,) if isinstance(key, str) else ()

        def add(code, message, field):
            error(code, message, ids, field, (index,))

        if set(raw) - {'id', 'predicate', 'target', 'entity_id', 'depends_on', 'require_heading', 'target_source'}:
            add('condition_fields', '目标条件字段无效', '')
        valid_id = isinstance(key, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}', key) and key not in known
        if not valid_id:
            add('condition_id', '目标 ID 必须唯一且为 ASCII 标识符', 'id')
        predicate = item.get('predicate')
        valid_predicate = isinstance(predicate, str) and predicate in PREDICATES
        if not valid_predicate:
            add('predicate', '缺少该目标的独立验证器', 'predicate')
        dependencies = item.get('depends_on', [])
        valid_dependencies = isinstance(dependencies, list) and all(isinstance(d, str) for d in dependencies)
        if (not valid_dependencies or any(d not in known for d in dependencies)
                or len(dependencies) != len(set(dependencies))):
            add('dependencies', '依赖必须引用前面的目标，不能循环或重复', 'depends_on')
        target = item.get('target')
        valid_target = isinstance(target, str) and target in references
        if not valid_target:
            add('target', '目标位置未解析：请提供位置引用或 name=(x,y,yaw)', 'target')
        entity = item.get('entity_id')
        if valid_predicate and predicate in operations:
            if entity not in entity_ids:
                add('entity', '缺少该对象的操作能力', 'entity_id')
            if predicate == 'released' and not release_supported:
                add('release_unsupported', '缺少独立的落置位置和静止验证，当前不能放置', 'predicate')
            if predicate in {'acquired', 'placed_on'} and schema_version < 5:
                add('schema_version', 'acquired/placed_on 需要 GoalSpec v5', 'predicate')
            if predicate == 'placed_on':
                if not place_supported or not valid_target or target not in (placement_surfaces or {}):
                    add('placement_surface', '缺少可信放置表面或放置能力', 'target')
                if valid_dependencies and not any(
                    c['id'] in dependencies and c.get('predicate') == 'acquired' and c.get('entity_id') == entity
                    for c in clean
                ):
                    add('acquired_dependency', 'placed_on 必须依赖同一对象的 acquired 历史目标（必须直接列入 depends_on，间接依赖不满足协议）', 'depends_on')
        elif valid_predicate and entity is not None:
            add('position_entity', '位置条件不能包含对象；at/visited 只表示机器人位置', 'entity_id')
        source = item.get('target_source', 'reference')
        if source not in ('reference', 'visual'):
            add('target_source', 'target_source 必须为 reference 或 visual', 'target_source')
        if source == 'visual' and (schema_version < 4 or predicate not in ('holding', 'acquired')
                                  or not valid_target or target not in (visual_regions or {})):
            add('visual_target', '视觉目标仅支持已标注搜索地点中的 holding/acquired 对象', 'target_source')
        if predicate in ('holding', 'acquired') and valid_target and target in (visual_regions or {}) and source != 'visual':
            add('visual_required', '已标注区域是搜索地点，取物必须 target_source=visual；观察入口不是对象取物位姿', 'target_source')
        heading = item.get('require_heading', schema_version == 2 or (valid_predicate and predicate in operations))
        if type(heading) is not bool:
            add('heading_type', 'require_heading 必须是布尔值', 'require_heading')
        elif schema_version == 2 and not heading:
            add('legacy_heading', '旧版目标必须保留朝向要求', 'require_heading')
        elif valid_predicate and predicate in operations and not heading:
            add('operation_heading', '操作交接仍要求完整取放位姿', 'require_heading')
        item['require_heading'] = heading
        item['depends_on'] = list(dependencies) if valid_dependencies else []
        # Retain identifiable conditions for cross-condition diagnostics even
        # when another field is invalid. Any error prevents returning a goal.
        if valid_id:
            clean.append(item)
            known.add(key)

    placed = [c for c in clean if c.get('predicate') == 'placed_on' and isinstance(c.get('entity_id'), str)]
    for entity in sorted({c['entity_id'] for c in placed}):
        destinations = [c for c in placed if c['entity_id'] == entity]
        held = [c for c in clean if c.get('predicate') == 'holding' and c.get('entity_id') == entity]
        if held:
            error('holding_placement_conflict', '同一对象不能同时最终持有和放置；取物历史使用 acquired',
                  [c['id'] for c in held + destinations], 'predicate')
        if len(destinations) > 1:
            error('placement_conflict', '同一对象不能同时放到多个表面', [c['id'] for c in destinations], 'target')
    finals = [c for c in clean if c.get('predicate') == 'at' and isinstance(c.get('target'), str)]
    if len({c['target'] for c in finals}) > 1:
        error('final_position_conflict', '不能同时位于多个终点；中途到访请使用 visited，准备动作应放入确认后的计划；不得擅自删除用户要求',
              [c['id'] for c in finals], 'predicate')
    if errors:
        raise GoalValidationError(errors)
    return clean


@dataclass(frozen=True)
# 已确认的用户目标及其位置引用；与执行中可调整的步骤计划分开保存。
class DynamicTask:
    task_key: str
    instruction: str
    references: dict = field(default_factory=dict)
    conditions: tuple = ()
    world_revision: str = ''
    confirmed: bool = False
    schema_version: int = 3
    kind: str = 'dynamic'
    resume_state: dict = field(default_factory=dict)
    preparation_steps: int = 0
    preparation_tools: int = 0
    preparation_seconds: float = 0.0
    visual_regions: dict = field(default_factory=dict)
    placement_surfaces: dict = field(default_factory=dict)
    supported_entities: tuple = ()
    entity_catalog: dict = field(default_factory=dict)
    reference_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.instruction, str) or not self.instruction.strip() or len(self.instruction) > 4000:
            raise ValueError('指令不能为空且不能超过 4000 个字符')
        if not isinstance(self.references, dict) or len(self.references) > 256 or any(not isinstance(k, str) or not 1 <= len(k) <= 80 for k in self.references):
            raise ValueError('位置引用必须使用 1–80 字符名称，最多 256 个')
        if type(self.confirmed) is not bool or self.kind != 'dynamic' or self.schema_version not in {2, 3, 4, 5}:
            raise ValueError('动态目标版本或确认状态无效')
        if any(type(v) is not int or v < 0 for v in (self.preparation_steps, self.preparation_tools)) or not math.isfinite(self.preparation_seconds) or self.preparation_seconds < 0:
            raise ValueError('目标准备预算无效')
        if not isinstance(self.world_revision, str):
            raise ValueError('运行版本必须为字符串')
        # Round-trip copies keep caller-owned dictionaries out of the accepted goal.
        object.__setattr__(self, 'references', {k: pose_value(v) for k, v in self.references.items()})
        regions = json.loads(json.dumps(self.visual_regions, allow_nan=False))
        if not isinstance(regions, dict) or (regions and self.schema_version < 4):
            raise ValueError('视觉区域需要 GoalSpec v4')
        for name, region in regions.items():
            if (name not in self.references or not isinstance(region, dict)
                    or region.get('source') != 'operator_annotation'
                    or type(region.get('search_radius_m')) not in (int, float)
                    or not 0 < region['search_radius_m'] <= 3
                    or type(region.get('approach_yaw')) not in (int, float)
                    or not math.isfinite(region['approach_yaw'])):
                raise ValueError('视觉搜索区域标注无效')
        object.__setattr__(self, 'visual_regions', regions)
        if (not isinstance(self.supported_entities, (tuple, list)) or len(self.supported_entities) > 256
                or any(not isinstance(v, str) or not 1 <= len(v) <= 80 for v in self.supported_entities)
                or len(set(self.supported_entities)) != len(self.supported_entities)):
            raise ValueError('对象能力目录无效')
        object.__setattr__(self, 'supported_entities', tuple(self.supported_entities))
        for field_name, allowed_ids, fields in (
                ('entity_catalog', self.supported_entities, {'name', 'aliases', 'description', 'operations'}),
                ('reference_metadata', self.references, {'kind', 'aliases', 'description'})):
            catalog = json.loads(json.dumps(getattr(self, field_name), allow_nan=False))
            if not isinstance(catalog, dict) or any(key not in allowed_ids for key in catalog):
                raise ValueError('目录元数据只能描述已声明的对象或位置')
            for entry in catalog.values():
                if not isinstance(entry, dict) or set(entry) - fields:
                    raise ValueError('目录元数据字段无效')
                for key, value in entry.items():
                    if key in {'aliases', 'operations'}:
                        if (not isinstance(value, list) or len(value) > 64
                                or any(not isinstance(v, str) or not 1 <= len(v) <= 80 for v in value)):
                            raise ValueError('目录别名或操作列表无效')
                    elif not isinstance(value, str) or not 1 <= len(value) <= 1000:
                        raise ValueError('目录说明无效')
                if 'kind' in entry and entry['kind'] not in {'pose', 'visual_region', 'placement_surface'}:
                    raise ValueError('位置目录类型无效')
                if set(entry.get('operations', [])) - {'locate', 'attach', 'release', 'place'}:
                    raise ValueError('对象目录操作无效')
            object.__setattr__(self, field_name, catalog)
        from harness.robots.composed_placement import validate_surface
        surfaces = json.loads(json.dumps(self.placement_surfaces, allow_nan=False))
        if not isinstance(surfaces, dict) or (surfaces and self.schema_version < 5):
            raise ValueError('放置表面需要 GoalSpec v5')
        for name, surface in surfaces.items():
            validate_surface(surface)
            if name not in self.references or tuple(surface['robot_pose']) != self.references[name]:
                raise ValueError('桌前导航引用与放置位姿不一致')
        object.__setattr__(self, 'placement_surfaces', surfaces)
        conditions = validate_conditions(self.conditions, self.references, release_supported=True, schema_version=self.schema_version, visual_regions=regions, entity_ids=self.supported_entities, placement_surfaces=surfaces, place_supported=True) if self.conditions else []
        object.__setattr__(self, 'conditions', tuple(json.loads(json.dumps(conditions))))
        if self.confirmed and not self.conditions:
            raise ValueError('已确认目标不能为空')

    def payload(self):
        return asdict(self)

    @classmethod
    def from_payload(cls, payload):
        # Preserve historical payloads that predate explicit entity declarations.
        return cls(**{'schema_version': 2, 'supported_entities': ('water_bottle',), **payload})


def proposal_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
