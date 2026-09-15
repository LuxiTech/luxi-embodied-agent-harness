#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/isaac_env.sh"

usage() {
    cat <<'EOF'
用法：scripts/isaac_g1.sh <命令> [参数]

  doctor              只读检查镜像、G1 USD、策略、控制器、GPU 和目录
  run [运行时参数]    前台运行 Isaac Sim 5.1 G1 bridge
  status              查看容器和最新 bridge 状态
  stop                停止本项目的 Isaac G1 容器

常用环境变量：
  LUXI_ISAAC_GPU_INDEX=1       指定物理 GPU（默认继承 LUXI_GPU_INDEX）
  LUXI_ISAAC_SCENE=grid        grid / skill_demo / task_apartment / office / warehouse / brownstone
  LUXI_ISAAC_HEADLESS=1        1 为无窗口；0 使用当前 X11 DISPLAY
  LUXI_ISAAC_PERSON_ROOT=...   task_apartment 使用的本地 PBR 人物模型目录
  LUXI_ISAAC_RUNTIME_DIR=...   宿主与容器共享的原子文件协议目录
EOF
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        printf '缺少 %s：%s\n' "${label}" "${path}" >&2
        return 1
    fi
}

doctor() {
    local failed=0
    local scene="${LUXI_ISAAC_SCENE:-grid}"
    require_file \
        "${LUXI_ISAAC_REFERENCE_ROOT}/g1_locomotion_controller.py" \
        "已验证的 G1 控制器" || failed=1
    require_file \
        "${LUXI_ISAAC_ASSET_ROOT}/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd" \
        "Unitree G1 whole-body USD" || failed=1
    require_file \
        "${LUXI_ISAAC_ASSET_ROOT}/assets/model/policy1.onnx" \
        "Unitree G1 locomotion policy" || failed=1
    require_file \
        "${LUXI_ISAAC_ASSET_ROOT}/policy_runtime/onnxruntime/__init__.py" \
        "ONNX Runtime Python 包" || failed=1
    require_file "${LUXI_GENIE_SIM_ROOT}/VERSION" "Genie Sim 版本元数据" || failed=1
    require_file "${ROOT}/harness/robots/g1/isaac/isaac_g1_runtime.py" "Luxi Isaac runtime" || failed=1
    case "${scene}" in
        grid|skill_demo|office|warehouse) ;;
        task_apartment)
            require_file \
                "${LUXI_ISAAC_PERSON_ROOT}/Superhero_Male_FullBody.gltf" \
                "Isaac 公寓 PBR 人物 glTF" || failed=1
            require_file \
                "${LUXI_ISAAC_PERSON_ROOT}/Superhero_Male_FullBody.bin" \
                "Isaac 公寓 PBR 人物网格" || failed=1
            ;;
        brownstone)
            if [[ -z "${LUXI_ISAAC_BROWNSTONE_ASSET}" ]]; then
                printf 'Brownstone 主 USD 尚未配置：LUXI_ISAAC_BROWNSTONE_ASSET 为空。\n' >&2
                failed=1
            elif [[ ! -d "${LUXI_ISAAC_BROWNSTONE_ROOT}" ]]; then
                printf '缺少 Brownstone 场景目录：%s\n' \
                    "${LUXI_ISAAC_BROWNSTONE_ROOT}" >&2
                failed=1
            else
                require_file \
                    "${LUXI_ISAAC_BROWNSTONE_ROOT}/PACKAGE-INFO.yaml" \
                    "Brownstone 包元数据" || failed=1
                require_file \
                    "${LUXI_ISAAC_BROWNSTONE_ROOT}/Demos/AEC/BrownstoneDemo/Assets/Revit_Brownstone01/Revit_Brownstone01_Interior.usd" \
                    "Brownstone01 室内 USD" || failed=1
                local brownstone_root brownstone_asset
                brownstone_root="$(realpath -e "${LUXI_ISAAC_BROWNSTONE_ROOT}")"
                brownstone_asset="$(
                    realpath -e \
                        "${LUXI_ISAAC_BROWNSTONE_ROOT}/${LUXI_ISAAC_BROWNSTONE_ASSET}" \
                        2>/dev/null || true
                )"
                if [[ -z "${brownstone_asset}" \
                    || "${brownstone_asset}" != "${brownstone_root}/"* \
                    || ! -f "${brownstone_asset}" ]]; then
                    printf 'Brownstone 主 USD 无效或越出只读场景目录：%s\n' \
                        "${LUXI_ISAAC_BROWNSTONE_ASSET}" >&2
                    failed=1
                fi
            fi
            ;;
        *)
            printf '不支持的 Isaac 场景：%s\n' "${scene}" >&2
            failed=1
            ;;
    esac

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '找不到 nvidia-smi。\n' >&2
        failed=1
    elif ! nvidia-smi --id="${LUXI_ISAAC_GPU_INDEX}" \
        --query-gpu=name,memory.free,memory.total \
        --format=csv,noheader; then
        printf '无法读取 GPU %s。\n' "${LUXI_ISAAC_GPU_INDEX}" >&2
        failed=1
    fi

    if ! luxi_isaac_docker image inspect "${LUXI_ISAAC_IMAGE}" >/dev/null 2>&1; then
        printf '缺少 Isaac Sim 镜像：%s\n' "${LUXI_ISAAC_IMAGE}" >&2
        failed=1
    fi

    if (( failed != 0 )); then
        return 1
    fi
    printf 'Isaac G1 检查通过。\n'
    printf '  image:     %s\n' "${LUXI_ISAAC_IMAGE}"
    printf '  gpu:       %s\n' "${LUXI_ISAAC_GPU_INDEX}"
    printf '  assets:    %s\n' "${LUXI_ISAAC_ASSET_ROOT}"
    printf '  reference: %s\n' "${LUXI_ISAAC_REFERENCE_ROOT}"
    printf '  runtime:   %s\n' "${LUXI_ISAAC_RUNTIME_DIR}"
    printf '  scene:     %s\n' "${scene}"
    if [[ "${scene}" == "brownstone" ]]; then
        printf '  scene usd: %s/%s\n' \
            "${LUXI_ISAAC_BROWNSTONE_ROOT}" "${LUXI_ISAAC_BROWNSTONE_ASSET}"
    fi
}

require_safe_gpu_headroom() {
    local max_temp="${LUXI_ISAAC_MAX_GPU_TEMP_C:-75}"
    local max_util="${LUXI_ISAAC_MAX_GPU_UTILIZATION:-20}"
    local min_free="${LUXI_ISAAC_MIN_GPU_FREE_MB:-12288}"
    local value
    for value in "${max_temp}" "${max_util}" "${min_free}"; do
        if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
            printf 'Isaac GPU 安全阈值必须是非负整数。\n' >&2
            return 2
        fi
    done
    if (( max_temp < 40 || max_temp > 90 || max_util > 100 || min_free < 4096 )); then
        printf 'Isaac GPU 安全阈值超出允许范围。\n' >&2
        return 2
    fi

    local metrics temp util free_mb
    metrics="$(
        nvidia-smi --id="${LUXI_ISAAC_GPU_INDEX}" \
            --query-gpu=temperature.gpu,utilization.gpu,memory.free \
            --format=csv,noheader,nounits 2>/dev/null || true
    )"
    metrics="${metrics%%$'\n'*}"
    IFS=',' read -r temp util free_mb <<<"${metrics}"
    temp="${temp//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"
    if [[ ! "${temp}" =~ ^[0-9]+$ || ! "${util}" =~ ^[0-9]+$ || ! "${free_mb}" =~ ^[0-9]+$ ]]; then
        printf '无法读取 GPU %s 的启动安全指标。\n' "${LUXI_ISAAC_GPU_INDEX}" >&2
        return 2
    fi
    if (( temp > max_temp || util > max_util || free_mb < min_free )); then
        printf '拒绝启动 Isaac：GPU %s 当前 %s°C、利用率 %s%%、空闲 %s MiB；' \
            "${LUXI_ISAAC_GPU_INDEX}" "${temp}" "${util}" "${free_mb}" >&2
        printf '要求 <=%s°C、<=%s%%、>=%s MiB。\n' \
            "${max_temp}" "${max_util}" "${min_free}" >&2
        return 2
    fi
    printf 'GPU 启动保护通过：GPU %s，%s°C，利用率 %s%%，空闲 %s MiB。\n' \
        "${LUXI_ISAAC_GPU_INDEX}" "${temp}" "${util}" "${free_mb}"
}

watch_runtime_gpu() {
    local docker_pid="$1"
    local max_temp="${LUXI_ISAAC_RUNTIME_MAX_GPU_TEMP_C:-82}"
    local max_temp_samples="${LUXI_ISAAC_RUNTIME_MAX_GPU_TEMP_SAMPLES:-5}"
    local hard_temp="${LUXI_ISAAC_RUNTIME_HARD_GPU_TEMP_C:-88}"
    local min_free="${LUXI_ISAAC_RUNTIME_MIN_GPU_FREE_MB:-8192}"
    local metrics temp free_mb stop_reason
    local over_temp_samples=0

    if [[ ! "${max_temp}" =~ ^[0-9]+$ \
        || ! "${max_temp_samples}" =~ ^[0-9]+$ \
        || ! "${hard_temp}" =~ ^[0-9]+$ \
        || ! "${min_free}" =~ ^[0-9]+$ \
        || ${max_temp} -lt 40 \
        || ${max_temp} -gt 90 \
        || ${max_temp_samples} -lt 1 \
        || ${max_temp_samples} -gt 60 \
        || ${hard_temp} -lt ${max_temp} \
        || ${hard_temp} -gt 98 \
        || ${min_free} -lt 4096 ]]; then
        printf 'Isaac GPU 运行保护阈值无效；停止容器 %s。\n' \
            "${LUXI_ISAAC_CONTAINER}" >&2
        luxi_isaac_docker stop --time 10 "${LUXI_ISAAC_CONTAINER}" >/dev/null 2>&1 || true
        return 2
    fi

    while kill -0 "${docker_pid}" >/dev/null 2>&1; do
        if luxi_isaac_docker container inspect "${LUXI_ISAAC_CONTAINER}" >/dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
    while kill -0 "${docker_pid}" >/dev/null 2>&1; do
        metrics="$(
            nvidia-smi --id="${LUXI_ISAAC_GPU_INDEX}" \
                --query-gpu=temperature.gpu,memory.free \
                --format=csv,noheader,nounits 2>/dev/null || true
        )"
        metrics="${metrics%%$'\n'*}"
        IFS=',' read -r temp free_mb <<<"${metrics}"
        temp="${temp//[[:space:]]/}"
        free_mb="${free_mb//[[:space:]]/}"
        stop_reason=""
        if [[ "${temp}" =~ ^[0-9]+$ && "${free_mb}" =~ ^[0-9]+$ ]]; then
            if (( free_mb < min_free )); then
                stop_reason="空闲显存低于 ${min_free} MiB"
            elif (( temp >= hard_temp )); then
                stop_reason="温度达到立即停机线 ${hard_temp}°C"
            elif (( temp >= max_temp )); then
                over_temp_samples=$((over_temp_samples + 1))
                if (( over_temp_samples == 1 )); then
                    printf 'Isaac GPU 温度告警：GPU %s 为 %s°C；需连续 %s 次达到 %s°C 才停止。\n' \
                        "${LUXI_ISAAC_GPU_INDEX}" "${temp}" \
                        "${max_temp_samples}" "${max_temp}" >&2
                fi
                if (( over_temp_samples >= max_temp_samples )); then
                    stop_reason="连续 ${over_temp_samples} 次达到 ${max_temp}°C"
                fi
            else
                over_temp_samples=0
            fi
        fi
        if [[ -n "${stop_reason}" ]]; then
            printf 'Isaac GPU 运行保护触发：GPU %s 为 %s°C、空闲 %s MiB（%s）；停止容器 %s。\n' \
                "${LUXI_ISAAC_GPU_INDEX}" "${temp}" "${free_mb}" \
                "${stop_reason}" "${LUXI_ISAAC_CONTAINER}" >&2
            luxi_isaac_docker stop --time 10 "${LUXI_ISAAC_CONTAINER}" >/dev/null 2>&1 || true
            return
        fi
        sleep 1
    done
}

run_bridge() {
    doctor
    require_safe_gpu_headroom

    if luxi_isaac_docker container inspect "${LUXI_ISAAC_CONTAINER}" >/dev/null 2>&1; then
        printf '容器已存在：%s；请先运行 scripts/isaac_g1.sh stop。\n' \
            "${LUXI_ISAAC_CONTAINER}" >&2
        return 2
    fi

    mkdir -p \
        "${LUXI_ISAAC_RUNTIME_DIR}" \
        "${LUXI_ISAAC_CACHE_ROOT}/cache/kit" \
        "${LUXI_ISAAC_CACHE_ROOT}/cache/ov" \
        "${LUXI_ISAAC_CACHE_ROOT}/cache/pip" \
        "${LUXI_ISAAC_CACHE_ROOT}/cache/glcache" \
        "${LUXI_ISAAC_CACHE_ROOT}/cache/computecache" \
        "${LUXI_ISAAC_CACHE_ROOT}/logs" \
        "${LUXI_ISAAC_CACHE_ROOT}/data" \
        "${LUXI_ISAAC_CACHE_ROOT}/documents"

    local scene="${LUXI_ISAAC_SCENE:-grid}"
    local headless="${LUXI_ISAAC_HEADLESS:-1}"
    local -a display_args=()
    local -a scene_mount_args=()
    local -a runtime_args=(
        --runtime-dir /workspace/runtime
        --controller-root /workspace/g1ref
        --scene "${scene}"
    )
    if [[ -n "${LUXI_ISAAC_ENTITIES:-}" ]]; then
        local entity
        local -a configured_entities=()
        IFS=',' read -r -a configured_entities <<<"${LUXI_ISAAC_ENTITIES}"
        for entity in "${configured_entities[@]}"; do
            entity="${entity//[[:space:]]/}"
            if [[ -n "${entity}" ]]; then
                runtime_args+=(--entity "${entity}")
            fi
        done
    fi
    if [[ "${LUXI_ISAAC_MANIPULATION_ACCEPTANCE:-0}" == "1" ]]; then
        runtime_args+=(--manipulation-acceptance-test)
    elif [[ "${LUXI_ISAAC_MANIPULATION_ACCEPTANCE:-0}" != "0" ]]; then
        printf 'LUXI_ISAAC_MANIPULATION_ACCEPTANCE 必须是 0 或 1。\n' >&2
        return 2
    fi
    if [[ "${scene}" == "brownstone" ]]; then
        scene_mount_args+=(
            -v "${LUXI_ISAAC_BROWNSTONE_ROOT}:/workspace/scenes/brownstone:ro"
        )
        runtime_args+=(
            --scene-asset "/workspace/scenes/brownstone/${LUXI_ISAAC_BROWNSTONE_ASSET}"
        )
    fi
    if [[ "${scene}" == "task_apartment" ]]; then
        scene_mount_args+=(
            -v "${LUXI_ISAAC_PERSON_ROOT}:/workspace/person:ro"
        )
        runtime_args+=(
            --person-asset /workspace/person/Superhero_Male_FullBody.gltf
        )
    fi
    if [[ "${headless}" == "1" ]]; then
        runtime_args+=(--headless)
    elif [[ "${headless}" == "0" ]]; then
        display_args+=(
            -e "DISPLAY=${DISPLAY:-:0}"
            -e QT_X11_NO_MITSHM=1
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw
        )
        if [[ -n "${XAUTHORITY:-}" && -r "${XAUTHORITY}" ]]; then
            display_args+=(
                -e XAUTHORITY=/tmp/.luxi-xauthority
                -v "${XAUTHORITY}:/tmp/.luxi-xauthority:ro"
            )
        else
            printf '无法读取当前 X11 授权文件 XAUTHORITY；拒绝启动无显示权限的 Isaac 窗口。\n' >&2
            return 2
        fi
    else
        printf 'LUXI_ISAAC_HEADLESS 必须是 0 或 1。\n' >&2
        return 2
    fi
    runtime_args+=("$@")

    printf '启动 Isaac G1 bridge：scene=%s gpu=%s runtime=%s\n' \
        "${scene}" "${LUXI_ISAAC_GPU_INDEX}" "${LUXI_ISAAC_RUNTIME_DIR}"

    set +e
    luxi_isaac_docker run --rm \
        --name "${LUXI_ISAAC_CONTAINER}" \
        --label com.luxi.backend=isaac-g1 \
        --gpus "device=${LUXI_ISAAC_GPU_INDEX}" \
        --network host \
        --shm-size=2g \
        --user 0:0 \
        --entrypoint /isaac-sim/python.sh \
        -e ACCEPT_EULA=Y \
        -e PRIVACY_CONSENT=Y \
        -e HOME=/root \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e PYTHONPATH=/workspace/luxi \
        -e HTTP_PROXY= \
        -e HTTPS_PROXY= \
        -e ALL_PROXY= \
        -e http_proxy= \
        -e https_proxy= \
        -e all_proxy= \
        -e 'NO_PROXY=*' \
        -e 'no_proxy=*' \
        -v "${ROOT}:/workspace/luxi:ro" \
        -v "${LUXI_ISAAC_REFERENCE_ROOT}:/workspace/g1ref:ro" \
        -v "${LUXI_GENIE_SIM_ROOT}:/workspace/genie_sim:ro" \
        -v "${LUXI_ISAAC_ASSET_ROOT}:/workspace/assets:ro" \
        -v "${LUXI_ISAAC_RUNTIME_DIR}:/workspace/runtime:rw" \
        -v "${LUXI_ISAAC_CACHE_ROOT}/cache/kit:/isaac-sim/kit/cache:rw" \
        -v "${LUXI_ISAAC_CACHE_ROOT}/cache/ov:/root/.cache/ov:rw" \
        -v "${LUXI_ISAAC_CACHE_ROOT}/cache/pip:/root/.cache/pip:rw" \
        -v "${LUXI_ISAAC_CACHE_ROOT}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
        -v "${LUXI_ISAAC_CACHE_ROOT}/cache/computecache:/root/.nv/ComputeCache:rw" \
        -v "${LUXI_ISAAC_CACHE_ROOT}/logs:/root/.nvidia-omniverse/logs:rw" \
        -v "${LUXI_ISAAC_CACHE_ROOT}/data:/root/.local/share/ov/data:rw" \
        -v "${LUXI_ISAAC_CACHE_ROOT}/documents:/root/Documents:rw" \
        "${scene_mount_args[@]}" \
        "${display_args[@]}" \
        "${LUXI_ISAAC_IMAGE}" \
        /workspace/luxi/harness/robots/g1/isaac/isaac_g1_runtime.py \
        "${runtime_args[@]}" &
    local docker_pid=$!
    watch_runtime_gpu "${docker_pid}" &
    local watchdog_pid=$!
    wait "${docker_pid}"
    local status=$?
    kill "${watchdog_pid}" >/dev/null 2>&1 || true
    wait "${watchdog_pid}" >/dev/null 2>&1 || true
    set -e
    return "${status}"
}

status_bridge() {
    luxi_isaac_docker ps -a \
        --filter "name=^/${LUXI_ISAAC_CONTAINER}$" \
        --format 'container={{.Names}} status={{.Status}} image={{.Image}}'
    if [[ -f "${LUXI_ISAAC_RUNTIME_DIR}/state.json" ]]; then
        printf 'state=%s\n' "$(<"${LUXI_ISAAC_RUNTIME_DIR}/state.json")"
    else
        printf 'state=unavailable\n'
    fi
}

stop_bridge() {
    if ! luxi_isaac_docker container inspect "${LUXI_ISAAC_CONTAINER}" >/dev/null 2>&1; then
        printf 'Isaac G1 容器未运行：%s\n' "${LUXI_ISAAC_CONTAINER}"
        return 0
    fi
    luxi_isaac_docker stop --time 15 "${LUXI_ISAAC_CONTAINER}"
}

main() {
    local command_name="${1:-}"
    if [[ -z "${command_name}" ]]; then
        usage
        return 1
    fi
    shift

    case "${command_name}" in
        doctor) doctor "$@" ;;
        run) run_bridge "$@" ;;
        status) status_bridge "$@" ;;
        stop) stop_bridge "$@" ;;
        help|-h|--help) usage ;;
        *)
            printf '未知命令：%s\n\n' "${command_name}" >&2
            usage >&2
            return 2
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
