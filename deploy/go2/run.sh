#!/usr/bin/env bash
# 本体容器常驻定义可复用；本次启动的容器在退出会话时停止。
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
cd "$project_root"
container="${CONTAINER_NAME:-navigation-agent-lightnav-go2}"
image="${IMAGE_TAG:-navigation-agent-lightnav-go2:humble}"
started=0
logs_pid=
cleanup() {
    trap - EXIT INT TERM
    if [[ -n "$logs_pid" ]]; then kill "$logs_pid" 2>/dev/null || true; fi
    if [[ "$started" == 1 ]]; then docker stop --time 20 "$container" >/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker info >/dev/null 2>&1 || fail '无法连接 Docker；请检查 Docker 服务和当前用户权限。'
if docker container inspect "$container" >/dev/null 2>&1; then
    running="$(docker inspect --format '{{.State.Running}}' "$container")"
    reuse_container "$container" go2
    [[ "$running" == true ]] || started=1
else
    iface="${NETWORK_INTERFACE:-}"
    [[ -n "$iface" ]] || fail '请在 deploy/local.env 设置 NETWORK_INTERFACE 为连接机器人 SDK 的实际网卡。'
    iface_ip="$(ip -4 addr show dev "$iface" | awk '/inet /{print $2; exit}' | cut -d/ -f1)"
    [[ -n "$iface_ip" ]] || fail "网卡 $iface 没有 IPv4 地址；请检查网络。"
    docker image inspect "$image" >/dev/null 2>&1 || fail "缺少 $image；请先运行 make go2-build。"
    # Fast DDS 与 Unitree 自带 CycloneDDS 分开，保留现有相机与会话行为。
    docker run -d --init --pull=never --network host --name "$container" \
        --label navigation-agent.service=go2 --label "navigation-agent.owner=$(id -un)" \
        -v /dev/bus/usb:/dev/bus/usb \
        -v "$project_root/deploy/go2/entrypoint.sh:/usr/local/bin/entrypoint.sh:ro" \
        -e NETWORK_INTERFACE="$iface" -e NETWORK_INTERFACE_IP="$iface_ip" \
        -e ENUMERATE_NET_DEVICE="${ENUMERATE_NET_DEVICE:-true}" \
        -e DEVICE_ACCESS_MODE="${DEVICE_ACCESS_MODE:-EA}" \
        -e RMW_IMPLEMENTATION="${CONTAINER_RMW:-rmw_fastrtps_cpp}" \
        -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
        "$image" "$@"
    started=1
fi
if [[ "$started" == 0 ]]; then
    echo '客户端原本已运行；退出本会话时保留该容器。'
fi
docker logs --follow --since 1s "$container" &
logs_pid=$!
wait "$logs_pid"
