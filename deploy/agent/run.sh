#!/usr/bin/env bash
# 独立 Conda 环境；不修改 base 或 shell 启动配置。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_exe="${CONDA_EXE:-$(type -P conda || true)}"
conda_exe="${conda_exe:-$HOME/miniconda3/bin/conda}"
[[ -x "$conda_exe" ]] || { echo '找不到 Conda；请安装对应 CPU 架构的版本，或设置 CONDA_EXE。' >&2; exit 1; }
source "$("$conda_exe" info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1

conda activate nav-agent || { echo '请先运行 bash deploy/agent/setup.sh' >&2; exit 1; }
export PYTHONPATH="$project_root/src"
# 组合启动复用 nav-agent 的 Python，用户无需手动激活环境。
if [[ "${1:-}" == --session ]]; then
    [[ $# == 2 && ( "$2" == mujoco || "$2" == go2 ) ]] || { echo '用法：make mujoco 或 make go2' >&2; exit 1; }
    exec python "$project_root/deploy/session.py" "$2"
fi
backend_args=(--backend-config "$project_root/configs/mujoco-lightnav.toml")
for argument in "$@"; do
    if [[ "$argument" == --backend-config || "$argument" == --backend-config=* ]]; then
        backend_args=()
        break
    fi
done
exec python -m navigation_agent.app.main \
    --agent-config "$project_root/configs/agent.toml" \
    "${backend_args[@]}" "$@"
