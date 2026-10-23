#!/usr/bin/env bash
# 独立 Conda 环境；不修改 base 或 shell 启动配置。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_exe="${CONDA_EXE:-$(type -P conda || true)}"
conda_exe="${conda_exe:-$HOME/miniconda3/bin/conda}"
[[ -x "$conda_exe" ]] || { echo '找不到 Conda；请安装对应 CPU 架构的版本，或设置 CONDA_EXE。' >&2; exit 1; }
source "$("$conda_exe" info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1

[[ $# == 0 ]] || { echo '用法：bash deploy/agent/setup.sh' >&2; exit 1; }
if ! conda activate nav-agent 2>/dev/null; then
    "$conda_exe" create -y -n nav-agent --override-channels -c conda-forge python=3.11 pip
    conda activate nav-agent
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "需要 Python 3.11；请检查已有 nav-agent 环境"'
python -m pip install -r "$project_root/deploy/agent/requirements.txt"
python -m pip check
python -c 'import fastapi, uvicorn, watchfiles, pydantic, dotenv, aiohttp'
if [[ ! -f "$project_root/deploy/.env" ]]; then
    (umask 077; cp "$project_root/deploy/.env.example" "$project_root/deploy/.env")
fi
printf 'nav-agent 环境已准备；没有启动服务。\n'
