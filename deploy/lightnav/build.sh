#!/usr/bin/env bash
# 固定上游源码版本；Thor 使用仓库内的 CUDA 13 / SM110 构建配方。
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
cd "$project_root"
source_dir="${LIGHTNAV_SOURCE:-$project_root/.runtime/lightnav/upstream}"
image="${LIGHTNAV_IMAGE:-navigation-lightnav:local}"
platform="${LIGHTNAV_PLATFORM:-}"
if [[ -z "$platform" ]]; then
    case "$(uname -m)" in
        x86_64) platform=upstream ;;
        aarch64)
            if [[ -r /proc/device-tree/model ]] && grep -qi Thor /proc/device-tree/model; then
                platform=thor
            else
                fail '此 ARM 主机未识别为 Thor；请显式选择已验证的 LIGHTNAV_PLATFORM 或兼容镜像。'
            fi
            ;;
        *) fail '不支持自动选择镜像；请设置 LIGHTNAV_PLATFORM 或使用已有兼容镜像。' ;;
    esac
fi
[[ -f "$source_dir/pyproject.toml" ]] || fail '缺少 LightNav 源码；请先运行 make models-setup。'
if [[ "$platform" == thor ]]; then
    build_jobs="${VLLM_BUILD_JOBS:-2}"
    [[ "$build_jobs" =~ ^[1-9][0-9]*$ ]] || fail 'VLLM_BUILD_JOBS 必须是正整数。'
    docker build --build-context "deployment=$project_root/deploy/lightnav/thor" \
        --build-arg "VLLM_BUILD_JOBS=$build_jobs" \
        --build-arg "LIGHTNAV_UID=$(id -u)" --build-arg "LIGHTNAV_GID=$(id -g)" \
        -f "$project_root/deploy/lightnav/thor/Dockerfile" -t "$image" "$source_dir"
elif [[ "$platform" == upstream ]]; then
    docker build -t "$image" "$source_dir"
else
    fail "未知 LIGHTNAV_PLATFORM=$platform；可选 thor / upstream。"
fi
