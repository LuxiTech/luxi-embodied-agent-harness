#!/usr/bin/env bash

# This file is sourced by the DimOS helper scripts. It intentionally keeps all
# mutable third-party state under DIMOS_ASSET_ROOT.

DIMOS_LUXI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DIMOS_LOCAL_ENV_FILE="${DIMOS_LOCAL_ENV_FILE:-${DIMOS_LUXI_ROOT}/config/dimos.local.env}"

if [[ "${LUXI_LOCAL_ENV_LOADED_PID:-}" != "$$" && -f "${DIMOS_LOCAL_ENV_FILE}" ]]; then
    # shellcheck disable=SC1091
    source "${DIMOS_LOCAL_ENV_FILE}"
    LUXI_LOCAL_ENV_LOADED_PID="$$"
fi
# Older revisions exported this marker, which made fresh systemd processes
# incorrectly skip their machine-local configuration.
unset LUXI_LOCAL_ENV_LOADED

# shellcheck disable=SC1091
source "${DIMOS_LUXI_ROOT}/config/dimos-version.env"

export DIMOS_ASSET_ROOT="${DIMOS_ASSET_ROOT:-${HOME}/work/Asset/dimos}"
export DIMOS_UPSTREAM_DIR="${DIMOS_ASSET_ROOT}/upstream"
export DIMOS_RUNTIME_DIR="${DIMOS_ASSET_ROOT}/runtime"
export DIMOS_CACHE_ROOT="${DIMOS_ASSET_ROOT}/cache"
export DIMOS_NATIVE_ROOT="${DIMOS_ASSET_ROOT}/native"
export DIMOS_TOOLS_DIR="${DIMOS_ASSET_ROOT}/tools"
export DIMOS_UV_BIN="${DIMOS_TOOLS_DIR}/uv"

export UV_PROJECT_ENVIRONMENT="${DIMOS_RUNTIME_DIR}/.venv"
export UV_CACHE_DIR="${DIMOS_CACHE_ROOT}/uv"
export UV_PYTHON_INSTALL_DIR="${DIMOS_ASSET_ROOT}/python"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"

export XDG_CACHE_HOME="${DIMOS_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${DIMOS_RUNTIME_DIR}/config"
export XDG_DATA_HOME="${DIMOS_RUNTIME_DIR}/data"
export XDG_STATE_HOME="${DIMOS_RUNTIME_DIR}/state"
export HF_HOME="${DIMOS_CACHE_ROOT}/huggingface"
export TORCH_HOME="${DIMOS_CACHE_ROOT}/torch"
export NUMBA_CACHE_DIR="${DIMOS_CACHE_ROOT}/numba"
export MPLCONFIGDIR="${DIMOS_CACHE_ROOT}/matplotlib"

# Keep this reproduction on its own host-local LCM bus. DimOS defaults to
# port 7667; using a distinct port prevents commands and sensor topics from
# crossing into other robotics projects on the same workstation. ttl=0 keeps
# multicast packets on this host.
_luxi_lcm_url="${DIMOS_LCM_URL:-udpm://239.255.76.67:17667?ttl=0}"
if [[ "${_luxi_lcm_url}" != *"recv_buf_size="* ]]; then
    if [[ "${_luxi_lcm_url}" == *"?"* ]]; then
        _luxi_lcm_url="${_luxi_lcm_url}&recv_buf_size=4194304"
    else
        _luxi_lcm_url="${_luxi_lcm_url}?recv_buf_size=4194304"
    fi
fi
export DIMOS_LCM_URL="${_luxi_lcm_url}"
export LCM_DEFAULT_URL="${_luxi_lcm_url}"
unset _luxi_lcm_url

DIMOS_NATIVE_INCLUDE="${DIMOS_NATIVE_ROOT}/usr/include"
DIMOS_NATIVE_LIB="${DIMOS_NATIVE_ROOT}/usr/lib/x86_64-linux-gnu"

export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${DIMOS_TOOLS_DIR}:${PATH}"
export CPPFLAGS="-I${DIMOS_NATIVE_INCLUDE}${CPPFLAGS:+ ${CPPFLAGS}}"
export CFLAGS="-I${DIMOS_NATIVE_INCLUDE}${CFLAGS:+ ${CFLAGS}}"
export LDFLAGS="-L${DIMOS_NATIVE_LIB} -Wl,-rpath,${DIMOS_NATIVE_LIB}${LDFLAGS:+ ${LDFLAGS}}"
export LD_LIBRARY_PATH="${DIMOS_NATIVE_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PKG_CONFIG_PATH="${DIMOS_NATIVE_LIB}/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"

# MuJoCo's native viewer uses GLFW. On the current Wayland desktop this goes
# through XWayland via DISPLAY=:0, which is more reliable than forcing EGL.
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
# The pyGLFW wheel ships separate X11 and Wayland libraries. Its automatic
# Wayland choice cannot create a working MuJoCo context on this desktop, while
# the X11 library works through XWayland.
export PYGLFW_LIBRARY_VARIANT="${PYGLFW_LIBRARY_VARIANT:-x11}"
export PYGAME_HIDE_SUPPORT_PROMPT=1

mkdir -p \
    "${DIMOS_RUNTIME_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${XDG_DATA_HOME}" \
    "${XDG_STATE_HOME}" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${NUMBA_CACHE_DIR}" \
    "${MPLCONFIGDIR}"
