"""Versioned, user-confirmed goals. Models select references, never invent poses."""
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import re

PREDICATES = frozenset({'visited', 'at', 'holding', 'released', 'acquired', 'placed_on'})


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


# 校验模型提出的用户目标：可信地点、受支持对象/谓词、朝向要求和目标依赖。
# 这里只检查可表达性与结构；语义是否忠实于原指令仍由目标确认步骤核对。
def validate_conditions(conditions, references, *, release_supported=False, entity_ids=('water_bottle',), schema_version=3, visual_regions=None, placement_surfaces=None, place_supported=False):
    if not isinstance(conditions, (list, tuple)) or not 1 <= len(conditions) <= 24:
        raise ValueError('需要 1–24 个可验证的目标条件')
    known = set()
    clean = []
    for raw in conditions:
        if not isinstance(raw, dict) or set(raw) - {'id', 'predicate', 'target', 'entity_id', 'depends_on', 'require_heading', 'target_source'}:
            raise ValueError('目标条件字段无效')
        item = dict(raw)
        key = item.get('id')
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}', key) or key in known:
            raise ValueError('目标 ID 必须唯一且为 ASCII 标识符')
        predicate = item.get('predicate')
        if predicate not in PREDICATES:
            raise ValueError('缺少该目标的独立验证器')
        dependencies = item.get('depends_on', [])
        if (not isinstance(dependencies, list) or any(not isinstance(d, str) or d not in known for d in dependencies)
                or len(dependencies) != len(set(dependencies))):
            raise ValueError('依赖必须引用前面的目标，不能循环或重复')
        if item.get('target') not in references:
            raise ValueError('目标位置未解析：请提供位置引用或 name=(x,y,yaw)')
        if predicate in {'holding', 'released', 'acquired', 'placed_on'}:
            if item.get('entity_id') not in entity_ids:
                raise ValueError('缺少该对象的操作能力')
            if predicate == 'released' and not release_supported:
                raise ValueError('缺少独立的落置位置和静止验证，当前不能放置')
            if predicate in {'acquired', 'placed_on'} and schema_version < 5:
                raise ValueError('acquired/placed_on 需要 GoalSpec v5')
            if predicate == 'placed_on':
                if not place_supported or item['target'] not in (placement_surfaces or {}):
                    raise ValueError('缺少可信放置表面或放置能力')
                if not any(c['id'] in dependencies and c['predicate'] == 'acquired' and c['entity_id'] == item['entity_id'] for c in clean):
                    raise ValueError('placed_on 必须依赖同一对象的 acquired 历史目标')
        elif item.get('entity_id') is not None:
            raise ValueError('位置条件不能包含对象')
        source = item.get('target_source', 'reference')
        if source not in {'reference', 'visual'}:
            raise ValueError('target_source 必须为 reference 或 visual')
        if source == 'visual' and (schema_version < 4 or predicate not in {'holding', 'acquired'}
                                  or item['target'] not in (visual_regions or {})):
            raise ValueError('视觉目标仅支持已标注搜索地点中的 holding 对象')
        if predicate in {'holding', 'acquired'} and item['target'] in (visual_regions or {}) and source != 'visual':
            raise ValueError('厨房是搜索地点，取物必须 target_source=visual；不能将房间入口当成水瓶位置')
        heading = item.get('require_heading', schema_version == 2 or predicate in {'holding', 'released', 'acquired', 'placed_on'})
        if type(heading) is not bool:
            raise ValueError('require_heading 必须是布尔值')
        if schema_version == 2 and not heading:
            raise ValueError('旧版目标必须保留朝向要求')
        if predicate in {'holding', 'released', 'acquired', 'placed_on'} and not heading:
            raise ValueError('操作交接仍要求完整取放位姿')
        item['require_heading'] = heading
        item['depends_on'] = list(dependencies)
        clean.append(item)
        known.add(key)
    placed = {c['entity_id'] for c in clean if c['predicate'] == 'placed_on'}
    if any(c['predicate'] == 'holding' and c['entity_id'] in placed for c in clean):
        raise ValueError('同一对象不能同时最终持有和放置；取物历史使用 acquired')
    destinations = [(c['entity_id'], c['target']) for c in clean if c['predicate'] == 'placed_on']
    if len({e for e, _ in destinations}) != len(destinations):
        raise ValueError('同一对象不能同时放到多个表面')
    # Multiple distinct final robot locations cannot all hold simultaneously.
    finals = {c['target'] for c in clean if c['predicate'] == 'at'}
    if len(finals) > 1:
        raise ValueError('不能同时位于多个终点；中途到访请使用 visited')
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
    supported_entities: tuple = ("water_bottle",)

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
        return cls(**{'schema_version': 2, **payload})


def proposal_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
