#!/usr/bin/env bash
# Container entrypoint for the on-robot (Go2 + Orin) stack.
#
# Sources the three overlay environments (ROS 2, Orbbec driver, this workspace)
# and launches vln_bringup. NETWORK_INTERFACE / ENUMERATE_NET_DEVICE /
# DEVICE_ACCESS_MODE / ROS_DOMAIN_ID are set through `docker run -e ...` (or the
# defaults baked into the image) so one image serves USB and ethernet cameras
# without a rebuild. DEVICE_ACCESS_MODE=EA is the ethernet-access mode the
# Gemini 335Le requires.
#
# INTERACTIVE=1 skips the launch and drops into a shell instead -- for
# `ros2 run orbbec_camera list_devices_node`, `ros2 topic hz ...` and the rest
# of the staged verification in docker/README.md.
#
# Any extra arguments passed to `docker run` are appended to the launch.
#
# The ROS setup scripts read variables that are unset on a fresh shell (e.g.
# AMENT_TRACE_SETUP_FILES), so `-u` from the usual `set -euo pipefail` is not
# viable here: the very first `source` would die with "unbound variable".
# `-e` and `-o pipefail` stay.
set -eo pipefail

# Pin Fast DDS (the fastrtps RMW) to the robot interface. By default it binds
# its multicast discovery to the DEFAULT-ROUTE interface -- on this host that
# is the dock/router NIC, whose low-end router chokes on the multicast flood
# (and `ip route add 224.0.0.0/4 dev eth0` does not help: the socket-level
# interface choice ignores the multicast route table). Generate a whitelist
# profile from NETWORK_INTERFACE's IP and export it via
# FASTRTPS_DEFAULT_PROFILES_FILE; every fastrtps participant in the container
# (launch nodes AND `ros2` CLI tools) picks it up. The Unitree SDK's own
# cyclonedds is already pinned via ChannelFactoryInitialize and is unaffected.
#
# The IP comes from NETWORK_INTERFACE_IP, resolved by run_go2.sh on the HOST
# (the container may not ship iproute2 -- `set -e` would kill startup on the
# missing `ip` command). The in-container fallback is guarded by command -v
# and `|| true`; no IP resolved -> warn and leave Fast DDS unpinned.
iface="${NETWORK_INTERFACE:-enP8p1s0}"
iface_ip="${NETWORK_INTERFACE_IP:-}"
if [ -z "${iface_ip}" ] && command -v ip >/dev/null 2>&1; then
    iface_ip="$(ip -4 addr show dev "${iface}" 2>/dev/null \
        | awk '/inet /{print $2; exit}' | cut -d/ -f1 || true)"
fi
if [ -n "${iface_ip}" ]; then
    profile="/tmp/fastdds_pinned_${iface}.xml"
    cat > "${profile}" <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>udp_pinned</transport_id>
      <type>UDPv4</type>
      <interfaceWhiteList>
        <address>127.0.0.1</address>
        <address>${iface_ip}</address>
      </interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="pinned_interface" is_default_profile="true">
    <rtps>
      <userTransports>
        <transport_id>udp_pinned</transport_id>
      </userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
EOF
    export FASTRTPS_DEFAULT_PROFILES_FILE="${profile}"
    export FASTDDS_DEFAULT_PROFILES_FILE="${profile}"
    echo "[entrypoint] Fast DDS pinned to ${iface} (${iface_ip})"
else
    echo "[entrypoint] WARNING: no IPv4 resolved for ${iface}; Fast DDS left unpinned" >&2
fi

. /opt/ros/humble/setup.bash
. /opt/orbbec_ws/install/setup.bash
. /opt/vln_ws/install/setup.bash

if [ "${INTERACTIVE:-0}" = "1" ]; then
    exec "${SHELL:-/bin/bash}" "$@"
fi

args=(
    ros2 launch vln_bringup go2.launch.py
    "network_interface:=${NETWORK_INTERFACE:-enP8p1s0}"
    "enumerate_net_device:=${ENUMERATE_NET_DEVICE:-false}"
    "device_access_mode:=${DEVICE_ACCESS_MODE:-Default}"
)

exec "${args[@]}" "$@"
