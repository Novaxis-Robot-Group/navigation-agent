#!/usr/bin/env bash
# 复用镜像自带 CUDA 库；不改变模型、提示词和动作解码。
set -euo pipefail
for library in "${VIRTUAL_ENV:-/opt/lightnav}"/lib/python*/site-packages/nvidia/*/lib; do
    [[ ! -d "$library" ]] || export LD_LIBRARY_PATH="$library${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
mkdir -p "$HOME" "${TMPDIR:-/tmp}"
exec "$@"
