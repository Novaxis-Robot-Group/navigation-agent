#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd "$project_root"
action="${1:-help}"
case "$action" in
    configure)
        [[ -f deploy/local.env ]] || cp deploy/local.env.example deploy/local.env
        [[ -f deploy/.env ]] || (umask 077; cp deploy/.env.example deploy/.env)
        echo '配置文件已就绪；默认 MuJoCo 只需在 deploy/.env 填写模型密钥。本机选项见 deploy/local.env，已有文件已保留。'
        ;;
    mujoco-setup) exec bash deploy/sim/bootstrap.sh ;;
    agent-setup|sim-setup) bash "deploy/${action%-setup}/setup.sh" ;;
    go2-build) bash deploy/go2/build.sh ;;
    models-setup) bash deploy/lightnav/setup.sh ;;
    lightnav) bash deploy/lightnav/run.sh ;;
    yolo) bash deploy/detector/run.sh "${DETECTOR_MODELS:-runs/target-detector/models}" ;;
    models)
        bash deploy/lightnav/run.sh
        bash deploy/detector/run.sh "${DETECTOR_MODELS:-runs/target-detector/models}"
        ;;
    mujoco-agent|go2-agent|mujoco-check|go2-check)
        backend="${action%%-*}"
        config="${MUJOCO_CONFIG:-configs/mujoco-lightnav.toml}"
        [[ "$backend" != go2 ]] || config="${GO2_CONFIG:-configs/go2-lightnav.toml}"
        args=()
        [[ "$action" != *-check ]] || args+=(--check)
        exec bash deploy/agent/run.sh --backend-config "$config" "${args[@]}"
        ;;
    mujoco-sim) exec bash deploy/sim/run.sh --backend-config "${MUJOCO_CONFIG:-configs/mujoco-lightnav.toml}" ;;
    mujoco|go2) exec bash deploy/agent/run.sh --session "$action" ;;
    go2-client) exec "${PYTHON:-python3}" deploy/session.py "$action" ;;
    *) fail "未知入口 $action；运行 make 查看帮助。" ;;
esac
