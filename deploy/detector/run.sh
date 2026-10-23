#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
container="${DETECTOR_CONTAINER:-navigation-target-detector}"
if reuse_container "$container" target-detector; then exit 0; fi
if [[ $# != 1 ]]; then
  echo "用法：deploy/detector/run.sh 模型目录（包含 yolov8l-worldv2.pt 和 clip/ViT-B-32.pt）" >&2
  exit 2
fi
docker image inspect "${DETECTOR_IMAGE:-navigation-target-detector:local}" >/dev/null 2>&1 || fail "缺少检测镜像；请先运行 make models-setup。"
models_dir="$(cd "$1" && pwd)"
weights="${DETECTOR_WEIGHTS:-yolov8l-worldv2.pt}"
if [[ "$weights" == */* || ! -f "$models_dir/$weights" || ! -f "$models_dir/clip/ViT-B-32.pt" ]]; then
  echo "缺少检测权重或 CLIP 文本编码器；请先准备模型目录。" >&2
  exit 2
fi
container="${DETECTOR_CONTAINER:-navigation-target-detector}"
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "容器 $container 已存在。请先检查归属和任务状态；已有服务用 docker start/restart，不自动覆盖。" >&2
  exit 2
fi
exec docker run -d --init --pull=never --name "$container" \
  --label navigation-agent.service=target-detector --label "navigation-agent.owner=$(id -un)" \
  --restart unless-stopped --runtime "${DETECTOR_RUNTIME:-nvidia}" --gpus all \
  --shm-size=256m --stop-timeout=40 \
  --log-driver=json-file --log-opt max-size=10m --log-opt max-file=3 \
  -p "127.0.0.1:${DETECTOR_PORT:-18060}:8060" \
  --mount "type=bind,source=$models_dir,target=/models,readonly" \
  "${DETECTOR_IMAGE:-navigation-target-detector:local}" \
  --weights "/models/$weights" --device "${DETECTOR_DEVICE:-0}" \
  --host 0.0.0.0 --port 8060 --inference-timeout "${DETECTOR_INFERENCE_TIMEOUT:-30}"
