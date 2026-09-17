#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_luxi_requested_isaac_scene="${LUXI_ISAAC_SCENE:-}"
source "${ROOT}/scripts/lib/dimos_env.sh"
for _luxi_vlm_name in LUXI_QWEN_API_KEY_FILE LUXI_DIMOS_VLM_API_KEY_FILE LUXI_DIMOS_VLM_BASE_URL LUXI_DIMOS_VLM_MODEL; do
    if [[ -n "${!_luxi_vlm_name:-}" ]]; then
        export "${_luxi_vlm_name}"
    fi
done
if [[ -n "${_luxi_requested_isaac_scene}" ]]; then
    export LUXI_ISAAC_SCENE="${_luxi_requested_isaac_scene}"
fi
source "${ROOT}/scripts/lib/isaac_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${VIRTUAL_ENV}/bin/python" -m harness.app.server "$@"
