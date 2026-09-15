#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/dimos_env.sh"

failures=0

printf 'Host preflight\n'
if ! "${ROOT}/scripts/preflight_dimos.sh"; then
    failures=$((failures + 1))
fi
printf '\nInstalled environment\n'

check_file() {
    local label="$1"
    local path="$2"
    if [[ -e "${path}" ]]; then
        printf 'OK   %-22s %s\n' "${label}" "${path}"
    else
        printf 'FAIL %-22s %s\n' "${label}" "${path}" >&2
        failures=$((failures + 1))
    fi
}

check_file "upstream repository" "${DIMOS_UPSTREAM_DIR}/.git"
check_file "isolated python" "${VIRTUAL_ENV}/bin/python"
check_file "dimos CLI" "${VIRTUAL_ENV}/bin/dimos"
check_file "PortAudio header" "${DIMOS_NATIVE_ROOT}/usr/include/portaudio.h"
check_file "TurboJPEG runtime" "${DIMOS_NATIVE_ROOT}/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0"
check_file "MuJoCo assets" "${DIMOS_UPSTREAM_DIR}/data/mujoco_sim"
check_file "person asset" "${DIMOS_UPSTREAM_DIR}/data/person"
check_file "CLIP spatial model" "${DIMOS_UPSTREAM_DIR}/data/models_clip"
MENAGERIE_DIR="${VIRTUAL_ENV}/lib/python3.12/site-packages/mujoco_playground/external_deps/mujoco_menagerie"
check_file "MuJoCo Menagerie" "${MENAGERIE_DIR}/.git"

if [[ -e "${DIMOS_UPSTREAM_DIR}/.git" ]]; then
    actual_commit="$(git -C "${DIMOS_UPSTREAM_DIR}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${actual_commit}" == "${DIMOS_UPSTREAM_COMMIT}" ]]; then
        printf 'OK   %-22s %s\n' "upstream commit" "${actual_commit}"
    else
        printf 'FAIL %-22s expected=%s actual=%s\n' \
            "upstream commit" "${DIMOS_UPSTREAM_COMMIT}" "${actual_commit}" >&2
        failures=$((failures + 1))
    fi
fi

if [[ -e "${MENAGERIE_DIR}/.git" ]]; then
    actual_menagerie_commit="$(git -C "${MENAGERIE_DIR}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${actual_menagerie_commit}" == "${DIMOS_MENAGERIE_COMMIT}" ]]; then
        printf 'OK   %-22s %s\n' "menagerie commit" "${actual_menagerie_commit}"
    else
        printf 'FAIL %-22s expected=%s actual=%s\n' \
            "menagerie commit" "${DIMOS_MENAGERIE_COMMIT}" "${actual_menagerie_commit}" >&2
        failures=$((failures + 1))
    fi
fi

if [[ -n "${DISPLAY:-}" ]]; then
    printf 'OK   %-22s %s\n' "XWayland DISPLAY" "${DISPLAY}"
else
    printf 'WARN %-22s MuJoCo UI 不能在无图形会话中打开\n' "XWayland DISPLAY"
fi

if [[ "${LCM_DEFAULT_URL}" == *"ttl=0"* ]]; then
    printf 'OK   %-22s %s\n' "isolated LCM bus" "${LCM_DEFAULT_URL}"
else
    printf 'WARN %-22s %s 可离开本机；仅连接真机时才应覆盖\n' \
        "isolated LCM bus" "${LCM_DEFAULT_URL}" >&2
fi

if [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
    if "${VIRTUAL_ENV}/bin/python" - <<'PY'
import dimos
import cv2
import glfw
import mujoco
import onnxruntime
import pyaudio
from turbojpeg import TurboJPEG

assert glfw.platform_supported(glfw.PLATFORM_X11), glfw.get_version_string()
assert hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"), (
    "OpenCV CSRT tracker missing; rerun scripts/dimos.sh bootstrap"
)
tracker = cv2.legacy.TrackerCSRT_create()
assert tracker is not None
TurboJPEG()
print(
    "OK   Python imports         "
    f"mujoco={mujoco.__version__} onnxruntime={onnxruntime.__version__} "
    f"opencv={cv2.__version__} glfw={glfw.get_version_string().decode()}"
)
PY
    then
        :
    else
        printf 'FAIL Python 依赖导入失败\n' >&2
        failures=$((failures + 1))
    fi
fi

if [[ -x "${VIRTUAL_ENV}/bin/dimos" ]]; then
    if "${VIRTUAL_ENV}/bin/dimos" list 2>/dev/null | grep -Eq '^unitree-g1-(basic-sim|sim|agentic-sim)'; then
        printf 'OK   %-22s G1 blueprints registered\n' "blueprint registry"
    else
        printf 'FAIL G1 blueprints 未注册\n' >&2
        failures=$((failures + 1))
    fi
fi

printf '\nAsset root: %s\n' "${DIMOS_ASSET_ROOT}"
printf 'Runtime:    %s\n' "${DIMOS_RUNTIME_DIR}"
printf 'Cache:      %s\n' "${DIMOS_CACHE_ROOT}"

if ((failures > 0)); then
    printf '\n诊断失败：%d 项。先运行 scripts/dimos.sh bootstrap。\n' "${failures}" >&2
    exit 1
fi

printf '\n诊断通过。\n'
