#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_PORT=8787
UI_URL="http://127.0.0.1:${UI_PORT}"

ui_pid() {
    ss -ltnp 2>/dev/null \
        | sed -n "s/.*127\\.0\\.0\\.1:${UI_PORT}.*pid=\\([0-9][0-9]*\\).*/\\1/p" \
        | head -n 1
}

is_luxi_isaac_ui() {
    local pid="$1"
    local command_line

    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    [[ "${command_line}" == *"-m harness.app.server"* ]] \
        && [[ "${command_line}" == *"--backend isaac-g1"* ]]
}

stop_existing_stack() {
    local pid=""
    pid="$(ui_pid || true)"

    if [[ -n "${pid}" ]] && ! is_luxi_isaac_ui "${pid}"; then
        printf '端口 %s 已被非 Luxi Isaac UI 进程占用（PID %s），拒绝停止它。\n' \
            "${UI_PORT}" "${pid}" >&2
        exit 2
    fi

    if [[ -n "${pid}" ]]; then
        printf '正在向旧操作台发送急停请求（PID %s）...\n' "${pid}"
        curl --silent --show-error --max-time 2 \
            --request POST \
            --header 'Content-Type: application/json' \
            --data '{}' \
            "${UI_URL}/api/stop" >/dev/null 2>&1 || true
    fi

    "${ROOT}/scripts/dimos.sh" stop || true
    "${ROOT}/scripts/dimos.sh" isaac-stop || true

    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        printf '正在停止旧操作台（PID %s）...\n' "${pid}"
        kill -TERM "${pid}"
        for _ in $(seq 1 20); do
            if ! kill -0 "${pid}" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "${pid}" 2>/dev/null; then
            printf '旧操作台未在 10 秒内退出，终止其已核验的 Luxi UI 进程。\n' >&2
            kill -KILL "${pid}"
        fi
    fi
}

stop_existing_stack

# Automatically export machine-local values so the DimOS worker processes can
# read the configured Qwen key-file path. The key contents remain outside the
# environment and are read directly from the chmod-600 file.
export DIMOS_LOCAL_ENV_FILE="${DIMOS_LOCAL_ENV_FILE:-${ROOT}/config/dimos.local.env}"
if [[ -f "${DIMOS_LOCAL_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${DIMOS_LOCAL_ENV_FILE}"
    set +a
fi

export PYTEST_VERSION=1
export LUXI_ISAAC_SCENE=task_apartment
export LUXI_MODEL_PROVIDER=qwen

printf '\n正在启动 Isaac 跟随测试环境：\n'
printf '  场景：task_apartment\n'
printf '  操作台：%s/\n' "${UI_URL}"
printf '  退出：在本终端按 Ctrl-C\n\n'

exec "${ROOT}/scripts/luxi-ui.sh" --backend isaac-g1 "$@"
