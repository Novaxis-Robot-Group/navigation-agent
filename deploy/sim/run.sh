#!/usr/bin/env bash
# 独立 Conda 环境；不修改 base 或 shell 启动配置。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_exe="${CONDA_EXE:-$(type -P conda || true)}"
conda_exe="${conda_exe:-$HOME/miniconda3/bin/conda}"
[[ -x "$conda_exe" ]] || { echo '找不到 Conda；请安装对应 CPU 架构的版本，或设置 CONDA_EXE。' >&2; exit 1; }
source "$("$conda_exe" info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1

conda activate nav-sim || { echo '请先运行 bash deploy/sim/setup.sh' >&2; exit 1; }
export PYTHONPATH="$project_root/src"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec python -m navigation_agent.app.mujoco.main \
    --backend-config "$project_root/configs/mujoco-lightnav.toml" "$@"
