#!/usr/bin/env bash

# Shared paths for the isolated Isaac Sim G1 backend.  The simulator image,
# robot assets, controller reference and mutable caches all stay outside the
# DimOS virtual environment.

if [[ -n "${LUXI_ISAAC_ENV_LOADED:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi
LUXI_ISAAC_ENV_LOADED=1

LUXI_ISAAC_PROJECT_ROOT="${LUXI_ISAAC_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DIMOS_LOCAL_ENV_FILE="${DIMOS_LOCAL_ENV_FILE:-${LUXI_ISAAC_PROJECT_ROOT}/config/dimos.local.env}"
_luxi_requested_scene="${LUXI_ISAAC_SCENE:-}"
if [[ "${LUXI_LOCAL_ENV_LOADED_PID:-}" != "$$" && -f "${DIMOS_LOCAL_ENV_FILE}" ]]; then
    # shellcheck disable=SC1091
    source "${DIMOS_LOCAL_ENV_FILE}"
    LUXI_LOCAL_ENV_LOADED_PID="$$"
fi
unset LUXI_LOCAL_ENV_LOADED
if [[ -n "${_luxi_requested_scene}" ]]; then
    LUXI_ISAAC_SCENE="${_luxi_requested_scene}"
fi
unset _luxi_requested_scene

LUXI_ISAAC_REFERENCE_ROOT="${LUXI_ISAAC_REFERENCE_ROOT:-${HOME}/work/SandBox/genie_sim_g1_inspire}"
LUXI_GENIE_SIM_ROOT="${LUXI_GENIE_SIM_ROOT:-${HOME}/work/SandBox/genie_sim}"
LUXI_ISAAC_ASSET_ROOT="${LUXI_ISAAC_ASSET_ROOT:-${HOME}/work/Asset/unitree_sim_isaaclab_usds}"
LUXI_ISAAC_IMAGE="${LUXI_ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
LUXI_ISAAC_CACHE_ROOT="${LUXI_ISAAC_CACHE_ROOT:-${HOME}/work/Asset/genie_sim_isaac_5_1_cache}"
LUXI_ISAAC_RUNTIME_DIR="${LUXI_ISAAC_RUNTIME_DIR:-${HOME}/work/Asset/dimos/runtime/luxi-isaac-g1}"
LUXI_ISAAC_CONTAINER="${LUXI_ISAAC_CONTAINER:-luxi-isaac-g1-$(id -u)}"
LUXI_ISAAC_GPU_INDEX="${LUXI_ISAAC_GPU_INDEX:-${LUXI_GPU_INDEX:-0}}"
LUXI_ISAAC_SCENE="${LUXI_ISAAC_SCENE:-grid}"
LUXI_ISAAC_HEADLESS="${LUXI_ISAAC_HEADLESS:-1}"
LUXI_ISAAC_PERSON_ROOT="${LUXI_ISAAC_PERSON_ROOT:-${LUXI_ISAAC_PROJECT_ROOT}/reference/assets/person/quaternius}"
LUXI_ISAAC_ENTITIES="${LUXI_ISAAC_ENTITIES:-}"
LUXI_ISAAC_MANIPULATION_ACCEPTANCE="${LUXI_ISAAC_MANIPULATION_ACCEPTANCE:-0}"
LUXI_ISAAC_BROWNSTONE_ROOT="${LUXI_ISAAC_BROWNSTONE_ROOT:-${HOME}/work/Asset/isaac_residential_brownstone/content}"
# Set after inspecting the NVIDIA archive. The value is always relative to
# LUXI_ISAAC_BROWNSTONE_ROOT so the launcher never mounts an arbitrary parent.
LUXI_ISAAC_BROWNSTONE_ASSET="${LUXI_ISAAC_BROWNSTONE_ASSET:-Demos/AEC/BrownstoneDemo/Assets/Brownstone01.usd}"

export LUXI_ISAAC_PROJECT_ROOT LUXI_ISAAC_REFERENCE_ROOT LUXI_GENIE_SIM_ROOT
export LUXI_ISAAC_ASSET_ROOT LUXI_ISAAC_IMAGE LUXI_ISAAC_CACHE_ROOT
export LUXI_ISAAC_RUNTIME_DIR LUXI_ISAAC_CONTAINER LUXI_ISAAC_GPU_INDEX
export LUXI_ISAAC_SCENE LUXI_ISAAC_HEADLESS
export LUXI_ISAAC_PERSON_ROOT
export LUXI_ISAAC_ENTITIES
export LUXI_ISAAC_MANIPULATION_ACCEPTANCE
export LUXI_ISAAC_BROWNSTONE_ROOT LUXI_ISAAC_BROWNSTONE_ASSET

_luxi_isaac_init_docker() {
    if [[ -n "${LUXI_ISAAC_DOCKER_INITIALIZED:-}" ]]; then
        return
    fi
    LUXI_ISAAC_DOCKER_INITIALIZED=1
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        LUXI_ISAAC_DOCKER_CMD=(docker)
    elif command -v sudo >/dev/null 2>&1; then
        LUXI_ISAAC_DOCKER_CMD=(sudo docker)
    else
        LUXI_ISAAC_DOCKER_CMD=(docker)
    fi
}

luxi_isaac_docker() {
    _luxi_isaac_init_docker
    "${LUXI_ISAAC_DOCKER_CMD[@]}" "$@"
}
