# 从仓库根目录构建 Go2 客户端，包含 ROS、相机驱动、Unitree SDK 和 Agent 协议扩展。
# 入口：make go2-build
# 默认直接使用公开依赖；网络受限时可通过 build-arg 指定镜像源。

FROM ros:humble-ros-base-jammy@sha256:1813d3c85d7f96ff7d3012d865204583255740182db5d0065f8f8cd029a83138

# APT_MIRROR rewrites the Ubuntu (archive/security/ports) and ROS apt hosts in
# sources.list (path layout is identical on the mirrors, so only the host
# changes). PIP_INDEX_URL is read by pip itself. ROSDEP_INDEX_URL redirects
# BOTH of rosdep's network paths: the sources.list yaml entries (sed rewrite
# of /etc/ros/rosdep/sources.list.d/20-default.list) AND the rosdistro index
# query, which is controlled by the ROSDISTRO_INDEX_URL environment variable,
# not the sources.list -- missing that export leaves "Query rosdistro index"
# pointed at raw.githubusercontent.com and the build times out there. The
# mirror root is derived from the index URL by dropping `index-v4.yaml`. The
# unconditional fuerte-line deletion below drops the one EOL-2012 gbpdistro
# source: its release data lives on long-dead hosts (code.ros.org snapshots),
# so processing it times out and fails `rosdep update` even though every
# source that matters succeeded. ORBBEC_REPO / UNITREE_REPO swap the two git
# clones; Orbbec maintains an official Gitee mirror.
ARG MAX_JOBS=4
ENV CMAKE_BUILD_PARALLEL_LEVEL=4 MAKEFLAGS="-j4"
ARG APT_MIRROR=
ARG PIP_INDEX_URL=
ARG ROSDEP_INDEX_URL=
ARG ORBBEC_REPO=https://github.com/orbbec/OrbbecSDK_ROS2.git
ARG UNITREE_REPO=https://github.com/unitreerobotics/unitree_sdk2_python.git

# Toolchain layer: colcon + rosdep for the workspace builds below, build
# tools for the two sdist builds (cyclonedds, and OrbbecSDK_ROS2's C++).
# The mirror rewrite runs before the first apt-get update; `|| true` keeps
# hosts that do not exist in this image from failing the build (a bad rewrite
# surfaces at apt-get update anyway).
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i -e "s@//archive.ubuntu.com@//${APT_MIRROR}@g" \
               -e "s@//security.ubuntu.com@//${APT_MIRROR}@g" \
               -e "s@//ports.ubuntu.com@//${APT_MIRROR}@g" \
               -e "s@//packages.ros.org@//${APT_MIRROR}@g" \
               /etc/apt/sources.list /etc/apt/sources.list.d/*.list || true; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libdw-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libssl-dev \
        libusb-1.0-0-dev \
        mesa-utils \
        nlohmann-json3-dev \
        python3-colcon-common-extensions \
        python3-pip \
        python3-rosdep \
        ros-humble-compressed-image-transport \
        ros-humble-theora-image-transport \
    && rm -rf /var/lib/apt/lists/*

# ── Orbbec driver (source build) ────────────────────────────────────────────
# The apt ros-humble-orbbec-camera package follows the v1 line; Gemini 33xLe
# (ethernet) cameras are only supported on the v2 line, so build v2 from
# source. The 335Le uses gemini_330_series.launch.py -- the same launch file
# go2.launch.py includes. Pin the tag for reproducibility. The apt-get update
# before rosdep install is required: the toolchain layer cleared the apt
# indexes, and rosdep install only runs `apt-get install` -- without a refresh
# every package resolves to "Unable to locate".
ARG ORBBEC_REF=v2.9.3
RUN mkdir -p /opt/orbbec_ws/src \
    && git clone --depth 1 --branch "${ORBBEC_REF}" \
        "${ORBBEC_REPO}" /opt/orbbec_ws/src/OrbbecSDK_ROS2 \
    && if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
           rosdep init; \
       fi \
    && if [ -n "${ROSDEP_INDEX_URL}" ]; then \
           export ROSDISTRO_INDEX_URL="${ROSDEP_INDEX_URL}"; \
           rosdep_root="${ROSDEP_INDEX_URL%index-v4.yaml}"; \
           sed -i "s@https://raw.githubusercontent.com/ros/rosdistro/master@${rosdep_root%/}@g" \
               /etc/ros/rosdep/sources.list.d/20-default.list; \
       fi \
    && sed -i '/releases\/fuerte.yaml/d' /etc/ros/rosdep/sources.list.d/20-default.list \
    && rosdep update \
    && apt-get update \
    && rosdep install --from-paths /opt/orbbec_ws/src --ignore-src -y \
    && cd /opt/orbbec_ws \
    && . /opt/ros/humble/setup.sh \
    && colcon build --parallel-workers 2 --event-handlers console_direct+ --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
    && rm -rf /opt/orbbec_ws/src/OrbbecSDK_ROS2/.git /opt/orbbec_ws/build /opt/orbbec_ws/log \
    && rm -rf /var/lib/apt/lists/*

# udev rules are for USB cameras; the 335Le is an ethernet camera, so they are
# not needed in the container and installing them (host /etc/udev) is skipped.

# ── Unitree SDK + MPC solver (system python, not a venv) ────────────────────
# unitree_sdk2_python is a plain DDS library, not a ROS package; the adapter
# imports it from the interpreter that runs the nodes (the image's python3).
#
# The pip `cyclonedds` sdist is the Python bindings ONLY: building it needs a
# preinstalled CycloneDDS C library (libddsc, found via CYCLONEDDS_HOME) --
# without it pip dies with "Could not locate cyclonedds", the first entry in
# unitree_sdk2_python's own FAQ. So build the C library first (the
# fixed releases/0.10.x commit, ABI-compatible with the 0.10.2 bindings) into /opt/cyclonedds, then pip-install
# the bindings and the SDK against it. CasADi last. All pip installs pick up
# PIP_INDEX_URL from the build environment when it is set. The `cd /` after
# the rm is required: the shell sits INSIDE /tmp/unitree_sdk2_python when it
# deletes it, and any later command in this RUN inherits a deleted CWD --
# pip3 dies at import time with FileNotFoundError from os.getcwd().
ARG UNITREE_REF=65691c8
# Declared here, next to the layer that uses it (like ORBBEC_REF above): an ARG
# inserted into the top block breaks the parent chain of every later layer and
# re-invalidates the expensive Orbbec build.
ARG CYCLONEDDS_REPO=https://github.com/eclipse-cyclonedds/cyclonedds.git
ARG CYCLONEDDS_REF=5041f3560c088c99e5088b2b8520b69169621196
RUN git init /tmp/cyclonedds \
    && git -C /tmp/cyclonedds remote add origin "${CYCLONEDDS_REPO}" \
    && git -C /tmp/cyclonedds fetch --depth 1 origin "${CYCLONEDDS_REF}" \
    && git -C /tmp/cyclonedds checkout --detach FETCH_HEAD \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
        -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_DDSPERF=OFF \
    && cmake --build /tmp/cyclonedds/build --target install --parallel "${MAX_JOBS}" \
    && rm -rf /tmp/cyclonedds \
    && CYCLONEDDS_HOME=/opt/cyclonedds pip3 install --no-cache-dir cyclonedds==0.10.2 \
    && git clone "${UNITREE_REPO}" /tmp/unitree_sdk2_python \
    && cd /tmp/unitree_sdk2_python \
    && git checkout "${UNITREE_REF}" \
    && CYCLONEDDS_HOME=/opt/cyclonedds pip3 install --no-cache-dir . \
    && rm -rf /tmp/unitree_sdk2_python \
    && cd /

# CasADi as its own layer (it does not depend on the SDK): the 3.7.x wheels
# are built with a GCC-13 toolchain and import-fail on jammy's GCC-12
# libstdc++ ("GLIBCXX_3.4.32 not found"). 3.6.5 is built on the older
# manylinux toolchain and covers every API vln_mpc uses (Opti/horzcat/diag/
# mtimes). The import smoke check surfaces ABI problems at BUILD time instead
# of when the first node starts.
RUN pip3 install --no-cache-dir "casadi==3.6.5" \
    && cd / && python3 -c "import casadi; print(casadi.__version__)"

# ── This workspace ──────────────────────────────────────────────────────────
# rosdep keys orbbec_camera / orbbec_description resolve against the apt v1
# package; they are provided by /opt/orbbec_ws instead, so skip them here.
# The .bashrc lines make `docker exec -it ... bash` shells usable too: exec
# shells bypass the ENTRYPOINT, so without them ros2/colcon are not on PATH.
# (Bash sources .bashrc for interactive shells only; non-interactive exec of
# commands is unaffected.)
COPY deploy/go2/scripts /opt/vln_ws/scripts
COPY src/navigation_agent/robots/go2/go2_adapter /opt/vln_ws/src/robot_adapters/go2_adapter
COPY src/navigation_agent/robots/go2/vln_mpc /opt/vln_ws/src/vln_mpc
COPY src/navigation_agent/vln/lightnav/vln_client /opt/vln_ws/src/vln_client
COPY src/navigation_agent/app/go2/vln_web /opt/vln_ws/src/vln_web
COPY src/navigation_agent/app/go2/vln_bringup /opt/vln_ws/src/vln_bringup
RUN apt-get update \
    && rosdep install --from-paths /opt/vln_ws/src --ignore-src -y \
        --skip-keys "orbbec_camera orbbec_description" \
    && apt-get install -y --no-install-recommends ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/* \
    && cd /opt/vln_ws && bash scripts/build.sh \
    && python3 -m pip install "pytest>=8,<9" "pytest-asyncio>=0.23,<1" \
    && bash scripts/test-agent.sh \
    && rm -rf /opt/vln_ws/build /opt/vln_ws/log \
    && echo ". /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo ". /opt/orbbec_ws/install/setup.bash" >> /root/.bashrc \
    && echo ". /opt/vln_ws/install/setup.bash" >> /root/.bashrc

# ── Runtime ─────────────────────────────────────────────────────────────────
# host networking (see docker/run_go2.sh) means no ports to expose. RMW stays
# at the Humble default (fastrtps): go2_adapter loads TWO DDS stacks in one
# process -- rclpy's RMW plus unitree_sdk2's own cyclonedds 0.10.2 -- and with
# rmw_cyclonedds_cpp both are Cyclone; the second domain creation segfaults
# (verified: same process, dual libddsc). fastrtps keeps them disjoint. The
# rmw-cyclonedds-cpp package stays installed for explicit opt-in use; root
# keeps the real-robot debugging phase simple (ethernet camera + host network
# = no device nodes); tighten with a non-root user once proven on your robot.
ENV ROS_DOMAIN_ID=0 \
    CYCLONEDDS_HOME=/opt/cyclonedds \
    LD_LIBRARY_PATH=/opt/cyclonedds/lib \
    NETWORK_INTERFACE=enP8p1s0 \
    ENUMERATE_NET_DEVICE=false

COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /usr/share/doc/navigation-agent/
COPY docs/lightnav-source.json /usr/share/doc/navigation-agent/docs/lightnav-source.json

COPY deploy/go2/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

LABEL org.lightnav.agent-protocol="1"
