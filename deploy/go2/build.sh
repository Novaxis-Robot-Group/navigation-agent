#!/usr/bin/env bash
# 在目标 ARM64 主机构建，不启动本体客户端。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec docker build --platform linux/arm64 \
    -f "$project_root/deploy/go2/Dockerfile" \
    -t navigation-agent-lightnav-go2:humble "$@" "$project_root"
