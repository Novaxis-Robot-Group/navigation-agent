#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "${script_dir}/.." && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble was not found at /opt/ros/humble" >&2
    exit 1
fi

source /opt/ros/humble/setup.bash
cd "${workspace_dir}"
# No --symlink-install: the Docker image rm -rf's build/ afterwards to stay slim,
# and a symlinked install space sources its hooks from build/ -- deleting it orphans
# every package. Pass --symlink-install via "$@" for live-edit dev builds.
colcon build "$@"
