#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/dimos_env.sh"

errors=0
warnings=0

ok() {
    printf 'OK   %-24s %s\n' "$1" "$2"
}

warn() {
    printf 'WARN %-24s %s\n' "$1" "$2" >&2
    warnings=$((warnings + 1))
}

fail() {
    printf 'FAIL %-24s %s\n' "$1" "$2" >&2
    errors=$((errors + 1))
}

architecture="$(uname -m)"
if [[ "${architecture}" == "x86_64" ]]; then
    ok "architecture" "${architecture}"
else
    fail "architecture" "${architecture}; 当前原生库路径只支持 x86_64"
fi

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" == "ubuntu" || "${ID:-}" == "debian" || "${ID_LIKE:-}" == *"debian"* ]]; then
        ok "operating system" "${PRETTY_NAME:-${ID:-unknown}}"
    else
        fail "operating system" "${PRETTY_NAME:-unknown}; bootstrap 需要 Debian/Ubuntu apt"
    fi
else
    fail "operating system" "无法读取 /etc/os-release"
fi

missing_commands=()
for command_name in git git-lfs curl apt-get dpkg-deb ip sysctl; do
    if command -v "${command_name}" >/dev/null 2>&1; then
        ok "command ${command_name}" "$(command -v "${command_name}")"
    else
        fail "command ${command_name}" "未安装"
        missing_commands+=("${command_name}")
    fi
done

if ((${#missing_commands[@]} > 0)); then
    printf '\n安装基础依赖：\n' >&2
    printf '  sudo apt-get update\n' >&2
    printf '  sudo apt-get install -y git git-lfs curl ca-certificates build-essential pkg-config dpkg iproute2 procps wmctrl libgl1 libegl1 libglib2.0-0 libportaudio2 libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6\n' >&2
    printf '  git lfs install\n' >&2
fi

free_kib="$(df -Pk "${DIMOS_ASSET_ROOT}" 2>/dev/null | awk 'NR == 2 {print $4}')"
recommended_kib=$((30 * 1024 * 1024))
if [[ "${free_kib}" =~ ^[0-9]+$ ]] && ((free_kib >= recommended_kib)); then
    ok "free disk" "$(awk -v kib="${free_kib}" 'BEGIN {printf "%.1f GiB", kib / 1024 / 1024}')"
else
    warn "free disk" "建议至少 30 GiB 可用；Asset 完整安装约 17 GiB，升级和缓存还需余量"
fi

memory_kib="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null)"
if [[ "${memory_kib}" =~ ^[0-9]+$ ]] && ((memory_kib >= 16 * 1024 * 1024)); then
    ok "system memory" "$(awk -v kib="${memory_kib}" 'BEGIN {printf "%.1f GiB", kib / 1024 / 1024}')"
else
    warn "system memory" "建议至少 16 GiB"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_line="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
    gpu_name="${gpu_line%%,*}"
    gpu_memory="${gpu_line##*,}"
    gpu_memory="${gpu_memory//[[:space:]]/}"
    if [[ "${gpu_memory}" =~ ^[0-9]+$ ]] && ((gpu_memory >= 8192)); then
        ok "NVIDIA GPU" "${gpu_name}, ${gpu_memory} MiB"
    else
        warn "NVIDIA GPU" "${gpu_line:-检测失败}; 低于 8 GiB 时先只运行 g1-basic"
    fi
else
    warn "NVIDIA GPU" "未找到 nvidia-smi；MuJoCo 仍可能运行，但未验证该配置"
fi

if [[ -n "${DISPLAY:-}" ]]; then
    ok "desktop display" "DISPLAY=${DISPLAY}"
else
    warn "desktop display" "未设置 DISPLAY；从纯 SSH 会话无法打开 MuJoCo 窗口"
fi

if command -v ip >/dev/null 2>&1; then
    if ip -o link show dev lo 2>/dev/null | grep -q 'MULTICAST'; then
        ok "LCM loopback multicast" "enabled"
    else
        warn "LCM loopback multicast" "未启用；运行 sudo ip link set lo multicast on，否则上游启动会请求 sudo"
    fi
    if ip route show 224.0.0.0/4 2>/dev/null | grep -Eq 'dev[[:space:]]+lo([[:space:]]|$)'; then
        ok "LCM multicast route" "224.0.0.0/4 dev lo"
    else
        warn "LCM multicast route" "缺少 loopback 路由；运行 sudo ip route replace 224.0.0.0/4 dev lo"
    fi
else
    warn "LCM multicast route" "缺少 ip 命令（安装 iproute2）"
fi

target_rmem=67108864
for key in net.core.rmem_max net.core.rmem_default; do
    value="$(sysctl -n "${key}" 2>/dev/null || true)"
    if [[ "${value}" =~ ^[0-9]+$ ]] && ((value >= target_rmem)); then
        ok "${key}" "${value}"
    else
        warn "${key}" "${value:-unknown}; 运行 sudo sysctl -w ${key}=${target_rmem}"
    fi
done

printf '\nAsset root: %s\n' "${DIMOS_ASSET_ROOT}"
printf 'Local config: %s%s\n' "${DIMOS_LOCAL_ENV_FILE}" "$([[ -f "${DIMOS_LOCAL_ENV_FILE}" ]] && printf ' (loaded)' || printf ' (using defaults)')"

if ((errors > 0)); then
    printf '\n预检失败：%d 项错误，%d 项警告。修复错误后再运行 bootstrap。\n' "${errors}" "${warnings}" >&2
    exit 1
fi

printf '\n预检通过：%d 项警告。警告不阻止安装，但可能影响大数据流或图形界面。\n' "${warnings}"
