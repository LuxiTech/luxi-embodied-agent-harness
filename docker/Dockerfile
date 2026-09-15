ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE} AS builder

ARG DIMOS_UPSTREAM_URL=https://github.com/dimensionalOS/dimos.git
ARG DIMOS_UPSTREAM_TAG=v0.0.13.post1
ARG DIMOS_UPSTREAM_COMMIT=e022d808002a86fad45905cf3efb31426abba0fa
ARG DIMOS_UV_VERSION=0.11.28

ENV DEBIAN_FRONTEND=noninteractive \
    DIMOS_ASSET_ROOT=/opt/dimos \
    DIMOS_UPSTREAM_DIR=/opt/dimos/upstream \
    DIMOS_RUNTIME_DIR=/opt/dimos/runtime \
    UV_PROJECT_ENVIRONMENT=/opt/dimos/runtime/.venv \
    UV_CACHE_DIR=/opt/dimos/cache/uv \
    UV_PYTHON_INSTALL_DIR=/opt/dimos/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libturbojpeg0-dev \
        pkg-config \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

RUN install -d /opt/dimos/cache /opt/dimos/runtime /opt/dimos/python /opt/dimos/tools \
    && curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_VERSION="${DIMOS_UV_VERSION}" UV_INSTALL_DIR=/opt/dimos/tools sh \
    && GIT_LFS_SKIP_SMUDGE=1 git clone \
        --branch "${DIMOS_UPSTREAM_TAG}" \
        --depth 1 \
        --filter=blob:none \
        "${DIMOS_UPSTREAM_URL}" \
        /opt/dimos/upstream \
    && test "$(git -C /opt/dimos/upstream rev-parse HEAD)" = "${DIMOS_UPSTREAM_COMMIT}"

RUN --mount=type=cache,id=luxi-dimos-uv,target=/opt/dimos/cache/uv \
    /opt/dimos/tools/uv sync \
        --project /opt/dimos/upstream \
        --locked \
        --no-dev \
        --no-editable \
        --extra unitree \
        --extra sim \
        --extra cpu \
    && /opt/dimos/tools/uv pip install \
        --python /opt/dimos/runtime/.venv/bin/python \
        --reinstall \
        --no-deps \
        opencv-contrib-python==4.10.0.84

RUN git -C /opt/dimos/upstream lfs pull \
        --include="data/.lfs/mujoco_sim.tar.gz,data/.lfs/person.tar.gz,data/.lfs/models_clip.tar.gz" \
        --exclude="" \
    && /opt/dimos/runtime/.venv/bin/python - <<'PY'
from dimos.utils.data import get_data
from mujoco_playground._src import mjx_env

mjx_env.ensure_menagerie_exists()
for name in ("mujoco_sim", "person", "models_clip"):
    get_data(name)
PY


FROM ${BASE_IMAGE} AS runtime

ARG DIMOS_UID=1000
ARG DIMOS_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    DIMOS_ASSET_ROOT=/opt/dimos \
    DIMOS_LOCAL_ENV_FILE=/dev/null \
    LUXI_MUJOCO_HEADLESS=1 \
    MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/luxi

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libegl1 \
        libgl1 \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libgomp1 \
        iproute2 \
        libportaudio2 \
        libturbojpeg \
        libx11-6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${DIMOS_GID}" luxi \
    && useradd --uid "${DIMOS_UID}" --gid "${DIMOS_GID}" \
        --home-dir /home/luxi --create-home --shell /usr/sbin/nologin luxi

COPY --from=builder /opt/dimos/upstream /opt/dimos/upstream
COPY --from=builder /opt/dimos/runtime /opt/dimos/runtime
COPY --from=builder /opt/dimos/python /opt/dimos/python
COPY --from=builder /opt/dimos/tools /opt/dimos/tools
COPY . /opt/luxi

RUN install -d -o luxi -g luxi \
        /opt/dimos/cache/xdg \
        /opt/dimos/cache/huggingface \
        /opt/dimos/cache/torch \
        /opt/dimos/cache/numba \
        /opt/dimos/cache/matplotlib \
        /opt/dimos/runtime/config \
        /opt/dimos/runtime/data \
        /opt/dimos/runtime/state \
        /opt/dimos/runtime/luxi-sim-control \
        /opt/dimos/upstream/logs \
        /opt/dimos/upstream/assets/output \
    && chown -R luxi:luxi \
        /opt/dimos/cache \
        /opt/dimos/runtime/config \
        /opt/dimos/runtime/data \
        /opt/dimos/runtime/state \
        /opt/dimos/runtime/luxi-sim-control \
        /opt/dimos/upstream/logs \
        /opt/dimos/upstream/assets/output

WORKDIR /opt/luxi
USER luxi:luxi

ENTRYPOINT ["/opt/luxi/scripts/dimos.sh"]
CMD ["g1-tools"]
