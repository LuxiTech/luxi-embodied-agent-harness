"""Extracted shared implementation; independent of Agent entry points."""

from __future__ import annotations
import copy
from typing import Any
from harness.robots.robot_profiles import get_backend_profile


G1_AGENT_CONTRACT = """
你是 LuxiAgent 本地操作台中的机器人操作代理。你控制的对象仅限当前机器上隔离运行的
DimOS Unitree G1 MuJoCo 仿真，不是真实机器人。

必须遵守：
1. Agent 运行时每一轮都会先自动调用 observe_environment，并把结果放入当前请求。运动只能通过
   已提供的白名单工具完成；不要索要或猜测 Shell、文件、Git、凭据、网络或源码访问能力。
2. 只在用户明确要求运动、导航、探索或停止时调用运动类工具。只观察、解释或查询时不得移动。
3. 工具返回的是事实来源。运动后检查工具附带的最新观测，不把“命令已发送”说成“目标已到达”。
4. 风险 unknown/critical 时不得运动。critical 会触发本地安全回撤状态机；它只沿在线地图仍确认
   安全的近期轨迹低速回撤，直到稳定非 critical、第一人称相机可用且前向在线地图视野良好。旧导航
   永远不会恢复；若原用户任务仍在执行，运行时会在安全锁释放后创建一个携带原目标的新 Agent turn，
   由最新观测重新规划。safety_recovery.safety_hold 为真时不得调用任何运动工具。warning 时只允许
   短时低速试探。任何异常先 stop_robot。
5. move_robot 的 x 是机器人前向速度，y 是机器人左向速度，yaw 为逆时针角速度；均为有限时脉冲，
   结束后会自动归零。
   用户要求“身后/向后转/转身/180 度”时优先调用一次 turn_around；不要把转身拆成多轮
   move_robot。turn_around 默认左转、根据实际航向闭环，并持续服从实时风险停车。
   用户要求“绕着房间走一圈/巡逻一圈”时必须调用一次 walk_room_loop；它会产生真实平移、逐段读取
   当前雷达和 live costmap、规划一条完整足迹已知安全的闭环，并以里程计和全程物理净空审计验证。
   冷地图可能先原地采集传感器；不得把这段采集转身冒充绕房间行走，也不要自行拼接低层脉冲。
6. 自然语言导航只调用一次终端 navigate_with_text；探索/建图只调用一次终端 explore_frontiers。
   两者均在内部使用实时地图、规划器到达和停车证据，失败后不得用低层运动补做或自动重试。观测中的
   navigation_bridge.running 表示限速/watchdog 桥是否在线，mapping 表示真实 /global_costmap 状态；
   若桥未运行或规划器有输出但里程计没有变化，立即停止并如实说明。
7. 请求可能附带标注为 first_person 的本轮 JPEG，也可能没有。第三人称仅供本地操作员 UI，永不
   发送给 Agent。只有实际附带第一人称图像时才可描述可见内容；结构化距离和风险仍是运动安全的
   事实来源，不要根据像素猜测精确距离。
8. 调用运动工具前再次核对方向映射：后退 x<0、前进 x>0、左移 y>0、右移 y<0、左转 yaw>0、
   右转 yaw<0。参数方向与用户指令冲突时禁止调用。
9. 用户明确要求“记住/标记这里”时调用 tag_location，并使用简短、唯一的地点名。用户要求返回已
   标记地点或前往某个可见/记忆目标时调用 navigate_with_text；query 每次只写一个目标名称。
10. analyze_scene、find_visual_target 和 verify_visual_condition 是 DimOS 内部原生视觉技能，会读取
    最新仿真相机帧。用户要求走到门或其他可见物体前时，直接调用一次 approach_visual_target，
    不要先 find_visual_target 或 navigate_with_text；保持默认 50 秒导航截止，除非用户明确要求更短；
    只有其 task_status=arrived_verified 且
    completed=true 才能报告到达。只有用户明确要求接近人物时才可调用 approach_person；它必须保留
    至少 1 米停距，并以 RGB-D、规划器和里程计的最终校验结果为准。
11. 用简洁中文汇报实际调用、观测变化和是否停止，不泄露或复述任何密钥。
12. 用户要求从厨房取矿泉水瓶并带到起点或客厅时，必须只调用一次 fetch_object，完整传入
    object_id、pickup_pose 和 destination；不得拆成原子工具、不得重复调用，失败后也不得用低层
    运动补做。只有 object_fetched、completed=true、attachment_retained=true 且
    stationary_confirmed=true 才能声称完成。
""".strip()

ISAAC_AGENT_CONTRACT = """
你是 LuxiAgent 本地操作台中的机器人操作代理。你控制的对象仅限当前机器上隔离运行的
DimOS Unitree G1 Isaac Sim 5.1 仿真，不是真实机器人。

必须遵守：
1. 每轮会先自动读取新鲜 RGB/里程计观测。只使用本请求列出的白名单工具，不访问 Shell、文件、
   Git、凭据、网络或源码。
2. 当前 Isaac 具备身份过滤并校验过的 RTX lidar、双 costmap、同楼层 A* 导航、
   已知自由侧 frontier 探索、第一人称寻物和人物跟随，以及 safety_hold/沿已验证轨迹回撤。
   探索地图保留 unknown 只用于选边界，A* 地图继续阻塞 unknown；frontier 目标绝不落入未知栅格。
   地图或 lidar 不新鲜时工具会拒绝，禁止改用低层脉冲绕过。
3. 只有用户明确要求运动时，才可在新鲜位姿和相机可用、且 safety_hold=false 时发送一次
   |x|<=0.30 m/s、|y|<=0.18 m/s、|yaw|<=0.30 rad/s、最长 1 秒的脉冲。
   用户明确给出不超过 3 米的直行距离时必须使用 move_distance；它只做里程计闭环和停车验收，
   不具备避障能力，不得把完成距离描述成路径安全。
4. “转身/身后/180 度”只调用一次 turn_around；底层允许为 Isaac 步态加入小幅交替前后踏步，并
   回到原位置。只有 turn_verified 才能宣称转身完成，失败后必须停止，不得改用低层脉冲补转。
5. tag_location 把当前新鲜 world 位姿记录为精确名称；返回该地点使用一次 navigate_to_tag，
   已知 world 坐标使用一次 navigate_to_pose。不得把路线拆成 move_robot。只有
   navigation_verified、completed=true、planner_goal_reached=true 且 stationary_confirmed=true
   才能声称到达。
6. 工具返回是事实来源。motion_evidence 将命令成功与实测位移分开；运动后读取新里程计并确保速度
   归零，不把命令已接受描述成动作已完成。
7. 请求可能附带第一人称 JPEG。只有真正附带时才描述可见内容，不根据像素猜测精确距离。
8. 用户明确要求探索/建图时只调用一次 explore_frontiers；寻物只调用一次 object_search；跟随只
   调用一次 follow_person。它们会自行完成 frontier 选择或视觉锁定、风险监控和停车，失败后不得
   改用低层运动补做。只有 exploration_complete/exploration_budget_complete、
   arrived_verified 或 follow_verified 且 completed=true 才能声称完成。
   自然语言地点/目标导航只调用一次 navigate_with_text；它依次使用精确标签、当前 RGB-D、
   持久化 CLIP 视点记忆和已知自由侧 frontier，并在内部完成到达、停车和新鲜 RGB-D 复核。
9. 任何异常先 stop_robot。用简洁中文汇报真实结果，不泄露凭据。
""".strip()

G1_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "observe_environment",
            "description": "读取当前 G1 位姿、速度、人物/桌面距离、风险和相机可用性；不移动机器人。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dimos_status",
            "description": "查询 DimOS MCP 服务、模块和技能状态；不移动机器人。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_robot",
            "description": (
                "发送一次低速、有限时的 G1 速度脉冲，结束后自动归零并返回新观测。"
                "调用前必须先观察。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "minimum": -0.18,
                        "maximum": 0.18,
                        "description": "机器人前向速度，m/s；负数为后退。",
                    },
                    "y": {
                        "type": "number",
                        "minimum": -0.18,
                        "maximum": 0.18,
                        "description": "机器人左向速度，m/s；负数为右移。",
                    },
                    "yaw": {
                        "type": "number",
                        "minimum": -0.45,
                        "maximum": 0.45,
                        "description": "逆时针角速度，rad/s；负数为顺时针。",
                    },
                    "duration": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 2.5,
                        "description": "脉冲持续时间，秒。",
                    },
                },
                "required": ["x", "y", "yaw", "duration"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_robot",
            "description": "停止探索和导航，并向 G1 重复发送零速度；可随时调用。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_around",
            "description": (
                "安全转身约 180 度的高层动作。根据实际航向闭环执行多个限时脉冲，"
                "默认左转、逐步检查实时风险，最终自动归零。"
                "身后、向后转、掉头或 180 度任务必须优先使用本工具，不要拆成多轮 move_robot。"
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "walk_room_loop",
            "description": (
                "让 G1 根据新鲜在线 costmap 在当前房间可用空间规划并真实行走四段闭环。冷地图会"
                "先做受保护的原地传感器采集；每个脉冲重查地图、物理雷达和里程计，短暂地图延迟"
                "保持零速等待。最后验证全程最小物理距离、路径长度、起点闭合、航向和实际静止。"
                "绕房间/巡逻一圈任务必须用本工具。"
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_with_text",
            "description": (
                "调用 DimOS navigate_with_text：依次查询已标记地点、当前视觉目标和 CLIP 空间记忆，"
                "并导航到匹配位置。Luxi 安全层会在限定时间、风险升级或取消时自动停车。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 300,
                        "description": "简短导航目标。",
                    },
                    "duration": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 12.0,
                        "default": 8.0,
                        "description": "允许导航运行的最长秒数。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tag_location",
            "description": "使用 DimOS 给当前位置添加一个短名称；不移动机器人。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 80,
                    }
                },
                "required": ["location_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to_pose",
            "description": (
                "Isaac 专用终态导航：在同楼层、雷达已观测自由空间内规划到 world 坐标，"
                "等待 A* 到达和最终静止验证；不要用 move_robot 拼接替代。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "minimum": -100.0, "maximum": 100.0},
                    "y": {"type": "number", "minimum": -100.0, "maximum": 100.0},
                    "yaw_degrees": {
                        "type": "number",
                        "minimum": -360.0,
                        "maximum": 360.0,
                        "default": 0.0,
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 110.0,
                        "default": 60.0,
                    },
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to_tag",
            "description": (
                "Isaac 专用终态导航：返回 tag_location 保存的精确地点，"
                "只在已观测自由空间内规划并等待最终静止验证。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 80,
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 110.0,
                        "default": 60.0,
                    },
                },
                "required": ["location_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_navigation",
            "description": "立即取消 Isaac A* 导航并停车。",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_scene",
            "description": "调用 DimOS 原生 VLM，基于最新第一人称相机帧回答一个视觉问题；不移动机器人。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_visual_target",
            "description": "调用 DimOS 原生 VLM，在最新第一人称画面中定位一个目标并返回像素坐标框；不移动机器人。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 300,
                    }
                },
                "required": ["target"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_visual_condition",
            "description": "调用 DimOS 原生 VLM，仅依据最新第一人称画面验证一个视觉条件；不移动机器人。",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    }
                },
                "required": ["condition"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_frontiers",
            "description": (
                "在 Isaac Sim 中遍走遍建图。独立探索图保留 unknown 用于寻找边界，"
                "每个 A* 目标只落在机器人可达且周围已知自由的栅格。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "minimum": 20.0,
                        "maximum": 180.0,
                        "default": 90.0,
                    },
                    "max_frontiers": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 8,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "object_search",
            "description": (
                "在 Isaac Sim 中用新鲜第一人称 RGB-D、RTX lidar 和已知自由侧 "
                "frontier 做寻物；发现后在同一工具内规划接近并验证停车。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 200},
                    "standoff_distance": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 3.0,
                        "default": 0.9,
                    },
                    "timeout": {
                        "type": "number",
                        "minimum": 20.0,
                        "maximum": 180.0,
                        "default": 120.0,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "follow_person",
            "description": (
                "在 Isaac Sim 中单次 VLM 锁定人物，再用 CSRT、对齐 RGB-D 和 RTX lidar "
                "进行有界跟随；遮挡或证据过期立即停车，不会换人。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 200},
                    "follow_distance": {
                        "type": "number",
                        "minimum": 1.2,
                        "maximum": 2.0,
                        "default": 1.5,
                    },
                    "duration": {
                        "type": "number",
                        "minimum": 5.0,
                        "maximum": 60.0,
                        "default": 30.0,
                    },
                    "timeout": {
                        "type": "number",
                        "minimum": 65.0,
                        "maximum": 180.0,
                        "default": 150.0,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approach_visual_target",
            "description": (
                "使用最新对齐 RGB-D 定位一个门或其他可见目标，在 DimOS 内部完成规划、限时等待、"
                "直接停车和运动后 RGB-D 距离/朝向复核。不要与 find_visual_target 或 "
                "navigate_with_text 组合调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                        "description": "一个可见目标，例如 door。",
                    },
                    "standoff_distance": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 3.0,
                        "default": 0.9,
                    },
                    "timeout": {
                        "type": "number",
                        "minimum": 3.0,
                        "maximum": 60.0,
                        "default": 50.0,
                        "description": "导航截止秒数；本机固定场景保持默认 50 秒。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approach_person",
            "description": (
                "使用对齐 RGB-D 定位一个可见人物，经全局规划接近，并在安全停距处通过新 RGB-D 与"
                "里程计复核。仅在用户明确要求接近人物时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                        "description": "人物的可见描述。",
                    },
                    "standoff_distance": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 2.0,
                        "default": 1.2,
                    },
                    "timeout": {
                        "type": "number",
                        "minimum": 3.0,
                        "maximum": 20.0,
                        "default": 20.0,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_object",
            "description": (
                "一次性执行完整取物闭环：导航到厨房、RGB-D 定位矿泉水瓶、接近并以真实接触抓取、"
                "切换携带姿态、导航到目的地并确认最终静止。调用一次后无论成功失败都不得重试或"
                "改用低层工具补做。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {
                        "type": "string",
                        "enum": ["water_bottle"],
                        "description": "当前验收的可操作实体。",
                    },
                    "pickup_pose": {
                        "type": "string",
                        "enum": ["kitchen"],
                        "description": "物体所在语义地点。",
                    },
                    "destination": {
                        "type": "string",
                        "enum": ["start", "living_room"],
                        "description": "携带目的地；start 为本次任务起点。",
                    },
                    "hand": {
                        "type": "string",
                        "enum": ["right"],
                        "default": "right",
                    },
                },
                "required": ["object_id", "pickup_pose", "destination"],
                "additionalProperties": False,
            },
        },
    },
]

ISAAC_MOVE_DISTANCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "move_distance",
        "description": (
            "执行用户明确要求的直线相对位移。工具在本地根据里程计连续管理多个最长 1 秒的"
            "低速子脉冲，检查偏航/横向漂移，最后发送零速并确认物理静止；不提供避障保证。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "distance_m": {
                    "type": "number",
                    "minimum": -3.0,
                    "maximum": 3.0,
                    "description": "相对直行距离（米）；正数向前，负数向后。",
                }
            },
            "required": ["distance_m"],
            "additionalProperties": False,
        },
    },
}

def _tools_for_backend(backend: str) -> list[dict[str, Any]]:
    profile = get_backend_profile(backend)
    allowed = frozenset(profile.agent_tools)
    tools = copy.deepcopy(
        [
            tool
            for tool in G1_TOOLS
            if tool["function"]["name"] in allowed
        ]
    )
    text_navigation = next(
        (
            tool
            for tool in tools
            if tool["function"]["name"] == "navigate_with_text"
        ),
        None,
    )
    if text_navigation is not None:
        text_navigation["function"]["description"] = (
            "终端自然语言导航：精确标签 → 当前 RGB-D → 持久化 CLIP "
            "第一人称视点记忆 → 已知自由侧 frontier。CLIP 视点到达后必须用新鲜 "
            "RGB-D 重新定位；一次调用内完成停车和验收。"
        )
        text_navigation["function"]["parameters"] = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "standoff_distance": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 3.0,
                    "default": 0.9,
                },
                "timeout": {
                    "type": "number",
                    "minimum": 20.0,
                    "maximum": 180.0,
                    "default": 120.0,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }
    if backend != "isaac-g1":
        return tools
    tools.append(copy.deepcopy(ISAAC_MOVE_DISTANCE_TOOL))
    move_tool = next(
        tool for tool in tools if tool["function"]["name"] == "move_robot"
    )
    move_properties = move_tool["function"]["parameters"]["properties"]
    x_schema = move_properties["x"]
    x_schema["minimum"] = -0.30
    x_schema["maximum"] = 0.30
    y_schema = move_properties["y"]
    y_schema["minimum"] = -0.18
    y_schema["maximum"] = 0.18
    yaw_schema = move_properties["yaw"]
    yaw_schema["minimum"] = -0.30
    yaw_schema["maximum"] = 0.30
    duration_schema = move_properties["duration"]
    duration_schema["minimum"] = 0.1
    duration_schema["maximum"] = 1.0
    return tools
