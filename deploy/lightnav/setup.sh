#!/usr/bin/env bash
# 首次准备会下载源码、镜像依赖和权重；日常启动不会调用本脚本。
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
cd "$project_root"
source_dir="${LIGHTNAV_SOURCE:-$project_root/.runtime/lightnav/upstream}"
patch_dir="${LIGHTNAV_PATCH:-$project_root/.runtime/lightnav/capture-time}"
weights="${LIGHTNAV_WEIGHTS:-$source_dir/checkpoints/LightNav-0}"
image="${LIGHTNAV_IMAGE:-navigation-lightnav:local}"
revision="$(python3 -c 'import json; print(json.load(open("deploy/lightnav/sources.json"))["revision"])')"

command -v patch >/dev/null || fail '缺少 patch 命令；请先安装 patch，或运行 make mujoco-setup。'
if [[ ! -e "$source_dir" ]]; then
    # 下载成功后再放入目标目录；失败仅删除本次临时目录，允许直接重试。
    mkdir -p "$(dirname -- "$source_dir")"
    download_dir="$(mktemp -d "${source_dir}.download.XXXXXX")"
    trap 'rm -rf -- "$download_dir"' EXIT
    git -C "$download_dir" init
    git -C "$download_dir" remote add origin https://github.com/lightorigins/LightNav-0.git
    git -C "$download_dir" fetch --depth 1 origin "$revision"
    git -C "$download_dir" checkout --detach FETCH_HEAD
    mv -- "$download_dir" "$source_dir"
    trap - EXIT
fi
[[ -f "$source_dir/pyproject.toml" ]] || fail "源码目录不完整：$source_dir；请检查后重新准备，不自动覆盖。"
# 校验并生成补丁；已有目录只核对，不覆盖。
python3 deploy/lightnav/prepare.py "$source_dir" --output "$patch_dir" --reuse
if ! docker image inspect "$image" >/dev/null 2>&1; then
    bash deploy/lightnav/build.sh
fi
read -r model_repo model_revision < <(python3 -c 'import json; m=json.load(open("deploy/lightnav/sources.json")); print(m["model_repository"], m["model_revision"])')
if ! python3 deploy/lightnav/check_weights.py "$weights" >/dev/null 2>&1; then
    mkdir -p "$weights"
    weights="$(cd "$weights" && pwd)"
    docker run --rm --pull=never --user "$(id -u):$(id -g)" --entrypoint python \
        -e "MODEL_REPO=$model_repo" -e "MODEL_REVISION=$model_revision" \
        -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e HF_HOME=/tmp/.cache/huggingface \
        -e HF_HUB_OFFLINE=0 -e "HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}" \
        --mount "type=bind,source=$weights,target=/models" "$image" -c \
        'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["MODEL_REPO"], revision=os.environ["MODEL_REVISION"], local_dir="/models")'
fi

python3 deploy/lightnav/check_weights.py "$weights"

detector_image="${DETECTOR_IMAGE:-navigation-target-detector:local}"
if ! docker image inspect "$detector_image" >/dev/null 2>&1; then
    bash deploy/detector/build.sh "$image"
fi
models="${DETECTOR_MODELS:-$project_root/runs/target-detector/models}"
mkdir -p "$models/clip"
models="$(cd "$models" && pwd)"
docker run --rm --pull=never --user "$(id -u):$(id -g)" --entrypoint python \
    -e HOME=/tmp --mount "type=bind,source=$models,target=/models" \
    "$detector_image" -c '
from pathlib import Path
from urllib.request import urlretrieve
import clip
path = Path("/models/yolov8l-worldv2.pt")
if not path.exists():
    temporary = path.with_suffix(".download")
    urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt", temporary)
    temporary.replace(path)
clip.clip._download(clip.clip._MODELS["ViT-B/32"], "/models/clip")
'
echo 'LightNav 和 YOLO 已准备；没有启动业务服务。使用 make mujoco 或 make go2。'
