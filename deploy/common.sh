#!/usr/bin/env bash
# 所有部署入口读取同一份本机选项；不改写 TOML 或密钥。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$project_root/deploy/local.env" ]]; then
    source "$project_root/deploy/local.env"
fi
fail() { echo "$*" >&2; exit 1; }

# 仅复用带归属标签的容器；旧部署可由用户显式选择复用。
reuse_container() {
    local container="$1" service="$2" labels running owner known_service
    docker info >/dev/null 2>&1 || fail '无法连接 Docker；请检查 Docker 服务和当前用户权限。'
    docker container inspect "$container" >/dev/null 2>&1 || return 1
    labels="$(docker inspect --format '{{index .Config.Labels "navigation-agent.service"}}|{{index .Config.Labels "navigation-agent.owner"}}' "$container")"
    owner="${labels#*|}"
    known_service="${labels%%|*}"
    if [[ "${REUSE_EXISTING:-0}" != 1 ]] && {
        [[ "$owner" != "$(id -un)" ]] || [[ "$known_service" != "$service" ]];
    }; then
        fail "容器 $container 归属或服务标签不匹配。确认属于自己且服务用途正确后设置 REUSE_EXISTING=1；不会自动替换。"
    fi
    running="$(docker inspect --format '{{.State.Running}}' "$container")"
    if [[ "$running" != true ]]; then docker start "$container"; fi
    echo "复用 ${container}；保留原镜像、端口和挂载。"
}
