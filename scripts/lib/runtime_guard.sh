#!/usr/bin/env bash

# Fail-closed launch checks for GPU-heavy simulation commands.  This file is
# sourced after dimos_env.sh; it does not start, stop, or inspect robot state.
# A process check complements the port check because pinned DimOS can finish
# shutting down its workers yet leave the top-level process waiting on a child.

luxi_require_single_sim() {
    if [[ "${LUXI_ALLOW_PARALLEL_SIM:-0}" == "1" ]]; then
        printf '警告：LUXI_ALLOW_PARALLEL_SIM=1，已显式关闭单实例保护。\n' >&2
        return 0
    fi
    if ! command -v ss >/dev/null 2>&1; then
        printf '安全检查失败：找不到 ss，无法确认是否已有 DimOS 实例。\n' >&2
        return 2
    fi
    if ! command -v pgrep >/dev/null 2>&1; then
        printf '安全检查失败：找不到 pgrep，无法确认是否有残留仿真进程。\n' >&2
        return 2
    fi

    local command_port="${COMMAND_CENTER_PORT:-7779}"
    local mcp_port="${MCP_PORT:-9990}"
    local port listeners
    for port in "${command_port}" "${mcp_port}"; do
        listeners="$(ss -H -ltn "sport = :${port}" 2>/dev/null || true)"
        if [[ -n "${listeners}" ]]; then
            printf '拒绝启动第二套仿真：本机端口 %s 已被监听。\n' "${port}" >&2
            printf '请复用现有实例或先运行 scripts/dimos.sh stop。\n' >&2
            return 2
        fi
    done

    local stale_sim
    stale_sim="$(
        pgrep -af -u "$(id -u)" -- \
            '(/scripts/dimos_with_observer\.py|/harness/robots/g1/mujoco/mujoco_observer_launcher\.py)' \
            2>/dev/null || true
    )"
    if [[ -n "${stale_sim}" ]]; then
        printf '拒绝启动第二套仿真：发现仍存活的 Luxi/DimOS 仿真或残留进程。\n' >&2
        printf '  %s\n' "${stale_sim%%$'\n'*}" >&2
        printf '请先确认并停止该 PID；安全守卫不会自动杀死进程。\n' >&2
        return 2
    fi
}

luxi_require_gpu_headroom() {
    local default_required_mb="$1"
    local workload="$2"
    if [[ "${LUXI_GPU_GUARD:-1}" == "0" ]]; then
        printf '警告：LUXI_GPU_GUARD=0，已显式关闭显存启动保护。\n' >&2
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '安全检查失败：找不到 nvidia-smi，无法确认 %s 的显存余量。\n' "${workload}" >&2
        return 2
    fi

    local required_mb="${LUXI_MIN_GPU_FREE_MB:-${default_required_mb}}"
    if [[ ! "${required_mb}" =~ ^[0-9]+$ ]] || (( required_mb < 1024 )); then
        printf '安全检查失败：LUXI_MIN_GPU_FREE_MB 必须是至少 1024 的整数。\n' >&2
        return 2
    fi

    local gpu="${LUXI_GPU_INDEX:-}"
    if [[ -z "${gpu}" ]]; then
        gpu="${CUDA_VISIBLE_DEVICES:-0}"
        gpu="${gpu%%,*}"
    fi
    if [[ -z "${gpu}" || "${gpu}" == "all" ]]; then
        gpu="0"
    fi
    local memory_line free_mb total_mb
    memory_line="$(
        nvidia-smi --id="${gpu}" \
            --query-gpu=memory.free,memory.total \
            --format=csv,noheader,nounits 2>/dev/null || true
    )"
    memory_line="${memory_line%%$'\n'*}"
    free_mb="${memory_line%%,*}"
    total_mb="${memory_line#*,}"
    free_mb="${free_mb//[[:space:]]/}"
    total_mb="${total_mb//[[:space:]]/}"
    if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || [[ ! "${total_mb}" =~ ^[0-9]+$ ]]; then
        printf '安全检查失败：无法读取 GPU %s 的显存。\n' "${gpu}" >&2
        return 2
    fi
    if (( free_mb < required_mb )); then
        printf '拒绝启动 %s：GPU %s 仅剩 %s MiB，最低要求 %s MiB。\n' \
            "${workload}" "${gpu}" "${free_mb}" "${required_mb}" >&2
        printf '请先停止其他 GPU 任务；不要通过并行启动来重试。\n' >&2
        return 2
    fi
    printf '显存检查通过：%s，GPU %s 空闲 %s/%s MiB（要求 >= %s MiB）。\n' \
        "${workload}" "${gpu}" "${free_mb}" "${total_mb}" "${required_mb}"
}

luxi_prepare_sim_launch() {
    local required_mb="$1"
    local workload="$2"
    luxi_require_single_sim
    luxi_require_gpu_headroom "${required_mb}" "${workload}"
}
