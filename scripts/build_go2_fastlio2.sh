#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUXI_FASTLIO2_IMAGE:-luxi-fastlio2-sim:a32c9f5}"

docker build \
  --file "${ROOT}/docker/Dockerfile.fastlio2-sim" \
  --tag "${IMAGE}" \
  "${ROOT}"
docker image inspect "${IMAGE}" --format 'FAST-LIO2 仿真镜像已就绪：{{.RepoTags}} {{.Id}}'
