#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/dimos_env.sh"

ASSET_ROOT="${DIMOS_ASSET_ROOT}"
UPSTREAM_DIR="${DIMOS_UPSTREAM_DIR}"
TOOLS_DIR="${DIMOS_TOOLS_DIR}"
UV_BIN="${DIMOS_UV_BIN}"
NATIVE_ROOT="${DIMOS_NATIVE_ROOT}"
DEB_CACHE="${ASSET_ROOT}/cache/debs"

for command in git git-lfs curl apt-get dpkg-deb; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        printf '缺少命令：%s\n' "${command}" >&2
        exit 1
    fi
done

mkdir -p "${ASSET_ROOT}" "${TOOLS_DIR}" "${DEB_CACHE}" "${NATIVE_ROOT}"

if [[ ! -x "${UV_BIN}" ]] || [[ "$("${UV_BIN}" --version 2>/dev/null || true)" != "uv ${DIMOS_UV_VERSION}"* ]]; then
    printf '安装 uv %s 到 %s\n' "${DIMOS_UV_VERSION}" "${TOOLS_DIR}"
    curl -LsSf "https://astral.sh/uv/${DIMOS_UV_VERSION}/install.sh" \
        | env UV_INSTALL_DIR="${TOOLS_DIR}" UV_NO_MODIFY_PATH=1 sh
fi

if [[ ! -e "${UPSTREAM_DIR}/.git" ]]; then
    if [[ -d "${UPSTREAM_DIR}" ]] && [[ -n "$(find "${UPSTREAM_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf '目录已存在但不是 Git 仓库：%s\n' "${UPSTREAM_DIR}" >&2
        exit 1
    fi

    printf '克隆 DimOS %s 到 %s\n' "${DIMOS_UPSTREAM_TAG}" "${UPSTREAM_DIR}"
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        --branch "${DIMOS_UPSTREAM_TAG}" \
        --depth 1 \
        --filter=blob:none \
        "${DIMOS_UPSTREAM_URL}" \
        "${UPSTREAM_DIR}"
fi

actual_commit="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${DIMOS_UPSTREAM_COMMIT}" ]]; then
    printf '上游版本不匹配。期望 %s，实际 %s。为保护本地改动，脚本不会自动 reset。\n' \
        "${DIMOS_UPSTREAM_COMMIT}" "${actual_commit}" >&2
    exit 1
fi

if [[ ! -f "${NATIVE_ROOT}/usr/include/portaudio.h" ]] \
    || [[ ! -e "${NATIVE_ROOT}/usr/lib/x86_64-linux-gnu/libportaudio.so" ]]; then
    printf '下载并解压私有 PortAudio 开发包（不使用 sudo）\n'
    (
        cd "${DEB_CACHE}"
        apt-get download portaudio19-dev libportaudio2 libportaudiocpp0
        for package in ./*.deb; do
            dpkg-deb -x "${package}" "${NATIVE_ROOT}"
        done
    )
fi

if [[ ! -e "${NATIVE_ROOT}/usr/lib/x86_64-linux-gnu/libjack.so.0" ]]; then
    printf '下载并解压 PortAudio 所需的 JACK 运行库（不使用 sudo）\n'
    (
        cd "${DEB_CACHE}"
        apt-get download libjack-jackd2-0
        for package in ./*libjack-jackd2-0*.deb; do
            dpkg-deb -x "${package}" "${NATIVE_ROOT}"
        done
    )
fi

if [[ ! -e "${NATIVE_ROOT}/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0" ]]; then
    printf '下载并解压私有 TurboJPEG 运行库（用于 replay 图像解码，不使用 sudo）\n'
    (
        cd "${DEB_CACHE}"
        apt-get download libturbojpeg0
        for package in ./*libturbojpeg0*.deb; do
            dpkg-deb -x "${package}" "${NATIVE_ROOT}"
        done
    )
fi

printf '同步隔离环境（Python 3.12；unitree + sim + CPU ONNX）\n'
"${DIMOS_UV_BIN}" sync \
    --project "${DIMOS_UPSTREAM_DIR}" \
    --locked \
    --no-dev \
    --extra unitree \
    --extra sim \
    --extra cpu

# The pinned upstream lock currently resolves both opencv-python and
# opencv-contrib-python.  They install the same cv2 package, and the newer base
# wheel can overwrite contrib's CSRT tracker files depending on install order.
# Reinstall the upstream-pinned contrib wheel last so object navigation has the
# backend that DimOS calls at runtime.  This stays inside the isolated venv.
printf '恢复 DimOS 目标跟踪所需的 OpenCV contrib %s\n' "${DIMOS_OPENCV_CONTRIB_VERSION}"
"${DIMOS_UV_BIN}" pip install \
    --python "${VIRTUAL_ENV}/bin/python" \
    --reinstall \
    --no-deps \
    "opencv-contrib-python==${DIMOS_OPENCV_CONTRIB_VERSION}"

printf '准备 MuJoCo Menagerie（由锁定版 mujoco-playground 固定提交）\n'
"${VIRTUAL_ENV}/bin/python" - <<'PY'
from mujoco_playground._src import mjx_env

mjx_env.ensure_menagerie_exists()
print(f"已准备 MuJoCo Menagerie: {mjx_env.MENAGERIE_PATH}")
PY

MENAGERIE_DIR="${VIRTUAL_ENV}/lib/python3.12/site-packages/mujoco_playground/external_deps/mujoco_menagerie"
actual_menagerie_commit="$(git -C "${MENAGERIE_DIR}" rev-parse HEAD)"
if [[ "${actual_menagerie_commit}" != "${DIMOS_MENAGERIE_COMMIT}" ]]; then
    printf 'MuJoCo Menagerie 版本不匹配。期望 %s，实际 %s。\n' \
        "${DIMOS_MENAGERIE_COMMIT}" "${actual_menagerie_commit}" >&2
    exit 1
fi

if [[ "${DIMOS_FETCH_SIM_ASSETS:-1}" == "1" ]]; then
    printf '只下载 G1 MuJoCo 与空间记忆所需的三项 LFS 资产\n'
    git -C "${DIMOS_UPSTREAM_DIR}" lfs pull \
        --include="data/.lfs/mujoco_sim.tar.gz,data/.lfs/person.tar.gz,data/.lfs/models_clip.tar.gz" \
        --exclude=""

    "${VIRTUAL_ENV}/bin/python" - <<'PY'
from dimos.utils.data import get_data

for name in ("mujoco_sim", "person", "models_clip"):
    path = get_data(name)
    print(f"已准备 {name}: {path}")
PY
fi

"${VIRTUAL_ENV}/bin/python" - <<'PY'
import dimos
import mujoco
import onnxruntime
import cv2

print("DimOS 导入成功")
print(f"MuJoCo: {mujoco.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
assert hasattr(cv2.legacy, "TrackerCSRT_create")
print(f"OpenCV contrib tracker: {cv2.__version__}")
PY

printf '\n安装完成。下一步：%s/scripts/dimos.sh doctor\n' "${ROOT}"
