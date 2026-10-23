#!/usr/bin/env bash
# MuJoCo 首次准备入口；复用安装脚本，不启动业务服务、不覆盖已有配置。
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
cd "$project_root"

for command in git python3; do
    command -v "$command" >/dev/null || fail "缺少 $command；请先按 docs/zh/mujoco.md 准备主机。"
done
conda_exe="${CONDA_EXE:-$(type -P conda || true)}"
conda_exe="${conda_exe:-$HOME/miniconda3/bin/conda}"
[[ -x "$conda_exe" ]] || fail '找不到 Conda；请安装 Miniconda，或设置 CONDA_EXE，见 docs/zh/mujoco.md。'
if [[ "${START_MODELS:-1}" == 1 ]]; then
    command -v docker >/dev/null || fail '缺少 Docker；请先按 docs/zh/mujoco.md 准备主机。'
    docker info >/dev/null 2>&1 || fail '无法连接 Docker；请检查 Docker 服务和当前用户权限。'
fi

recording_ready() {
    command -v gst-launch-1.0 >/dev/null || return 1
    command -v gst-inspect-1.0 >/dev/null || return 1
    local plugin
    for plugin in fdsrc jpegparse jpegdec videoconvert x264enc h264parse mp4mux filesink; do
        gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || return 1
    done
}

# 准备录像插件与模型源码补丁工具；已有依赖直接复用。
if ! recording_ready || { [[ "${START_MODELS:-1}" == 1 ]] && ! command -v patch >/dev/null; }; then
    command -v apt-get >/dev/null || fail '缺少 GStreamer 或 patch；请按 docs/zh/mujoco.md 准备系统依赖。'
    privilege=()
    if [[ "$(id -u)" != 0 ]]; then
        command -v sudo >/dev/null || fail '安装系统依赖需要 sudo；请联系管理员按 docs/zh/mujoco.md 安装。'
        privilege=(sudo)
    fi
    echo '准备 GStreamer 和 patch 系统依赖；系统可能要求输入 sudo 密码。'
    "${privilege[@]}" apt-get update
    "${privilege[@]}" apt-get install -y --no-install-recommends \
        gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly patch
    recording_ready || fail 'GStreamer 插件仍不可用；请查看安装错误，见 docs/zh/replay.md。'
fi

bash deploy/manage.sh configure
bash deploy/agent/setup.sh
bash deploy/sim/setup.sh
if [[ "${START_MODELS:-1}" == 1 ]]; then
    bash deploy/lightnav/setup.sh
fi
printf '\nMuJoCo 准备完成；尚未启动服务。\n'
printf '在 deploy/.env 填写 DASHSCOPE_API_KEY，然后运行 make mujoco。\n'
