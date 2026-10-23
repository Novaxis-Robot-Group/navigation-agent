#!/usr/bin/env bash
# 独立 Conda 环境；不修改 base 或 shell 启动配置。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_exe="${CONDA_EXE:-$(type -P conda || true)}"
conda_exe="${conda_exe:-$HOME/miniconda3/bin/conda}"
[[ -x "$conda_exe" ]] || { echo '找不到 Conda；请安装对应 CPU 架构的版本，或设置 CONDA_EXE。' >&2; exit 1; }
source "$("$conda_exe" info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1

[[ $# == 0 ]] || { echo '用法：bash deploy/sim/setup.sh' >&2; exit 1; }
if ! conda activate nav-sim 2>/dev/null; then
    "$conda_exe" create -y -n nav-sim --override-channels -c conda-forge python=3.11 pip
    conda activate nav-sim
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "需要 Python 3.11；请检查已有 nav-sim 环境"'
python -m pip install -r "$project_root/deploy/sim/requirements.txt"
python -m pip check
python -c 'import mujoco, casadi, numpy, PIL, aiohttp, pydantic, dotenv; from vln_mujoco.model import scene_path; assert scene_path().is_file(), "缺少仿真场景资产"'
printf 'nav-sim 环境已准备；没有启动服务。\n'
