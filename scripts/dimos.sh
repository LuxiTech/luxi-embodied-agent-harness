#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
用法：scripts/dimos.sh <命令> [参数]

  bootstrap       安装固定版本、隔离环境和 G1 MuJoCo 资产
  preflight       安装前检查系统、磁盘、显存、图形会话和 LCM 配置
  doctor          检查版本、依赖、资产和图形会话
  isaac-doctor    只读检查 Isaac Sim 5.1、Genie Sim 元数据和 Unitree G1 资产
  isaac-status    查看本项目的 Isaac G1 容器与 bridge 状态
  isaac-stop      停止本项目的 Isaac G1 容器
  g1-isaac        只运行 Isaac G1 bridge（不启动 DimOS）
  list            列出官方和 Luxi 本地 blueprints
  g1-basic        官方 G1 基础 MuJoCo 仿真（推荐首跑）
  g1-perception   官方 G1 感知与空间记忆仿真
  g1-tools        共享模拟器与 MCP 工具服务；由 Harness 调用
  go2-demo        Go2 + 海康 MV-CU013-A0UC RGB + MID-360，仅提供寻物和人物跟随
  go2-ros-demo    单 Go2 仿真 + ROS 2/DDS 传感器与执行器
  go2-fastlio-build  构建 MuJoCo MID-360 使用的固定版 FAST-LIO2 原生镜像
  g1-isaac-tools    Isaac G1 共享 MCP 工具服务
  blind-eval      按 --scene/--seed 启动信息隔离的未知场景 UI
  ui              启动 Luxi 本地操作台，并按需托管 g1-tools 仿真
  record ...      录制同步的轨迹地图回放和 G1 第一人称 MP4
  move ...        给运行中的 G1 仿真发送限速、定时速度脉冲
  mcp ...         透传 dimos mcp 子命令
  mcp-job ...     通过操作 UI 启动、查询或取消长时间 MCP 任务
  status          查看当前 DimOS 实例
  log ...         查看当前 DimOS 日志
  stop            停止当前 DimOS 实例
  shell           打开已激活的隔离 shell

可在命令末尾追加其他官方参数。当前固定版的 --daemon 会丢失 worker，
本项目的 G1 命令会拒绝该参数；请在一个终端中保持前台运行。
EOF
}

require_foreground() {
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "--daemon" ]]; then
            printf '当前固定版 DimOS 的 --daemon 会在启动命令退出后丢失仿真/MCP worker。\n' >&2
            printf '请去掉 --daemon，在一个终端中保持前台运行；从另一终端调用工具。\n' >&2
            exit 2
        fi
    done
}

command_name="${1:-}"
if [[ -z "${command_name}" ]]; then
    usage
    exit 1
fi
shift

case "${command_name}" in
    bootstrap)
        exec "${ROOT}/scripts/bootstrap_dimos.sh" "$@"
        ;;
    preflight)
        exec "${ROOT}/scripts/preflight_dimos.sh" "$@"
        ;;
    isaac-doctor)
        exec "${ROOT}/scripts/isaac_g1.sh" doctor "$@"
        ;;
    isaac-status)
        exec "${ROOT}/scripts/isaac_g1.sh" status "$@"
        ;;
    isaac-stop)
        exec "${ROOT}/scripts/isaac_g1.sh" stop "$@"
        ;;
    g1-isaac)
        require_foreground "$@"
        exec "${ROOT}/scripts/isaac_g1.sh" run "$@"
        ;;
esac

# shellcheck disable=SC1091
_luxi_requested_isaac_scene="${LUXI_ISAAC_SCENE:-}"
source "${ROOT}/scripts/lib/dimos_env.sh"
# Machine-local dotenv values are shell variables by design.  The Go2 RGB
# detector runs in a subprocess, so explicitly pass only the configured key
# file path (never the key contents) and VLM endpoint/model settings.
for _luxi_vlm_name in \
    LUXI_QWEN_API_KEY_FILE LUXI_DIMOS_VLM_API_KEY_FILE \
    LUXI_DIMOS_VLM_BASE_URL LUXI_DIMOS_VLM_MODEL; do
    if [[ -n "${!_luxi_vlm_name:-}" ]]; then
        export "${_luxi_vlm_name}"
    fi
done
unset _luxi_vlm_name
if [[ -n "${_luxi_requested_isaac_scene}" ]]; then
    export LUXI_ISAAC_SCENE="${_luxi_requested_isaac_scene}"
fi
unset _luxi_requested_isaac_scene
# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/isaac_env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/runtime_guard.sh"

if [[ ! -x "${VIRTUAL_ENV}/bin/dimos" ]]; then
    printf 'DimOS 尚未安装。先运行：scripts/dimos.sh bootstrap\n' >&2
    exit 1
fi

DIMOS_BIN="${VIRTUAL_ENV}/bin/dimos"
VIEWER="${DIMOS_VIEWER:-none}"
RERUN_OPEN="${DIMOS_RERUN_OPEN:-none}"
DIMOS_VIS_ARGS=(--viewer "${VIEWER}" --rerun-open "${RERUN_OPEN}")
# Always use the project-local worker wrapper for G1 commands. It keeps the
# pinned DimOS checkout untouched while providing a true zero-velocity idle
# state; the third-person observer remains optional.
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
DIMOS_G1_RUNNER=("${VIRTUAL_ENV}/bin/python" "${ROOT}/scripts/dimos_with_observer.py")

case "${command_name}" in
    doctor)
        exec "${ROOT}/scripts/doctor_dimos.sh" "$@"
        ;;
    list)
        exec "${DIMOS_G1_RUNNER[@]}" list "$@"
        ;;
    g1-basic)
        require_foreground "$@"
        luxi_prepare_sim_launch 4096 "G1 基础仿真"
        exec "${DIMOS_G1_RUNNER[@]}" --simulation mujoco "${DIMOS_VIS_ARGS[@]}" \
            run unitree-g1-basic-sim "$@"
        ;;
    g1-perception)
        require_foreground "$@"
        luxi_prepare_sim_launch 6144 "G1 感知仿真"
        exec "${DIMOS_G1_RUNNER[@]}" --simulation mujoco "${DIMOS_VIS_ARGS[@]}" \
            run unitree-g1-sim "$@"
        ;;
    g1-tools)
        require_foreground "$@"
        luxi_prepare_sim_launch 6144 "G1 外部 agent 仿真"
        export LUXI_HEAD_DEPTH_PATH="${LUXI_HEAD_DEPTH_PATH:-${DIMOS_RUNTIME_DIR}/luxi-sim-control/head-depth.npz}"
        # External-agent runs still need the same fresh, self-filtered lidar
        # safety input as the UI-managed simulator.  Keep the path overridable
        # for isolated acceptance runs and blind-evaluation sandboxes.
        export LUXI_LIDAR_PROXIMITY_PATH="${LUXI_LIDAR_PROXIMITY_PATH:-${DIMOS_RUNTIME_DIR}/luxi-ui/lidar-proximity.json}"
        export LUXI_TAGGED_LOCATIONS_PATH="${LUXI_TAGGED_LOCATIONS_PATH:-${DIMOS_RUNTIME_DIR}/luxi-sim-control/tagged-locations.json}"
        exec "${DIMOS_G1_RUNNER[@]}" --simulation mujoco "${DIMOS_VIS_ARGS[@]}" \
            run luxi-g1-tools-sim "$@"
        ;;
    go2-demo)
        require_foreground "$@"
        if ! command -v docker >/dev/null 2>&1; then
            printf 'Go2 MID-360 + FAST-LIO2 仿真需要 Docker。\n' >&2
            exit 2
        fi
        if ! docker image inspect "${LUXI_FASTLIO2_IMAGE:-luxi-fastlio2-sim:a32c9f5}" >/dev/null 2>&1; then
            printf 'FAST-LIO2 仿真镜像尚未构建，请先运行：scripts/dimos.sh go2-fastlio-build\n' >&2
            exit 2
        fi
        if [[ -z "${LUXI_QWEN_API_KEY_FILE:-}${LUXI_DIMOS_VLM_API_KEY_FILE:-}" ]]; then
            printf 'Go2 海康 RGB 寻物需要配置 Qwen VLM Key 文件。\n' >&2
            exit 2
        fi
        luxi_prepare_sim_launch 2048 "Go2 海康 RGB/MID-360 寻物跟随仿真"
        export LUXI_SIM_BACKEND=mujoco-go2
        exec "${VIRTUAL_ENV}/bin/python" -m harness.robots.go2.go2_mujoco "$@"
        ;;
    go2-ros-demo)
        require_foreground "$@"
        if ! command -v docker >/dev/null 2>&1; then
            printf 'Go2 MID-360 + FAST-LIO2 仿真需要 Docker。\n' >&2
            exit 2
        fi
        if ! command -v ros2 >/dev/null 2>&1; then
            printf 'Go2 ROS 通信仿真需要先 source ROS 2 环境，使 ros2/rclpy/std_msgs 可用。\n' >&2
            exit 2
        fi
        if ! docker image inspect "${LUXI_FASTLIO2_IMAGE:-luxi-fastlio2-sim:a32c9f5}" >/dev/null 2>&1; then
            printf 'FAST-LIO2 仿真镜像尚未构建，请先运行：scripts/dimos.sh go2-fastlio-build\n' >&2
            exit 2
        fi
        if [[ -z "${LUXI_QWEN_API_KEY_FILE:-}${LUXI_DIMOS_VLM_API_KEY_FILE:-}" ]]; then
            printf 'Go2 海康 RGB 寻物需要配置 Qwen VLM Key 文件。\n' >&2
            exit 2
        fi
        luxi_prepare_sim_launch 2048 "单 Go2 ROS 2/DDS 仿真"
        export LUXI_SIM_BACKEND=mujoco-go2
        exec "${VIRTUAL_ENV}/bin/python" -m harness.robots.go2.go2_mujoco "$@"
        ;;
    go2-fastlio-build)
        exec "${ROOT}/scripts/build_go2_fastlio2.sh" "$@"
        ;;
    g1-isaac-tools)
        require_foreground "$@"
        export LUXI_GPU_INDEX="${LUXI_ISAAC_GPU_INDEX}"
        luxi_prepare_sim_launch 12288 "Isaac Sim 5.1 Unitree G1 外部 agent 仿真"
        export LUXI_ISAAC_AUTOSTART=1
        export LUXI_SIM_BACKEND=isaac-g1
        export LUXI_SIM_CONTROL_PATH="${LUXI_ISAAC_RUNTIME_DIR}"
        export LUXI_LIDAR_PROXIMITY_PATH="${LUXI_ISAAC_RUNTIME_DIR}/lidar-proximity.json"
        exec "${DIMOS_G1_RUNNER[@]}" "${DIMOS_VIS_ARGS[@]}" \
            run luxi-g1-isaac-tools-sim "$@"
        ;;
    blind-eval)
        require_foreground "$@"
        export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        exec "${VIRTUAL_ENV}/bin/python" -m harness.evaluation.blind_evaluation "$@"
        ;;
    ui)
        export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        exec "${VIRTUAL_ENV}/bin/python" -m harness.app.server "$@"
        ;;
    record)
        export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        exec "${VIRTUAL_ENV}/bin/python" -m harness.evaluation.recorder "$@"
        ;;
    move)
        exec "${VIRTUAL_ENV}/bin/python" "${ROOT}/scripts/send_g1_velocity.py" "$@"
        ;;
    mcp)
        exec "${DIMOS_G1_RUNNER[@]}" mcp "$@"
        ;;
    mcp-job)
        exec "${VIRTUAL_ENV}/bin/python" "${ROOT}/scripts/mcp_job.py" "$@"
        ;;
    status)
        exec "${DIMOS_BIN}" status "$@"
        ;;
    log)
        exec "${DIMOS_BIN}" log "$@"
        ;;
    stop)
        exec "${DIMOS_BIN}" stop "$@"
        ;;
    shell)
        exec "${SHELL:-/bin/bash}" --noprofile --norc
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        printf '未知命令：%s\n\n' "${command_name}" >&2
        usage >&2
        exit 1
        ;;
esac
