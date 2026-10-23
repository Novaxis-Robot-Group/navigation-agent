#!/usr/bin/env bash
# 无硬件测试，使用容器内 ROS 消息定义，不启动 ROS 节点。
set -eo pipefail
source /opt/ros/humble/setup.bash
workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$workspace_dir/install/setup.bash"
cd "$workspace_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$PWD/src/vln_web:$PWD/src/vln_mpc:$PWD/src/vln_client:$PWD/src/robot_adapters/go2_adapter:${PYTHONPATH:-}"
python3 -m pytest --import-mode=importlib -p no:cacheprovider src/vln_web/test src/vln_mpc/test src/vln_client/test src/robot_adapters/go2_adapter/test -q
