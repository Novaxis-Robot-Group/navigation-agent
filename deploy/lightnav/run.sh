#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
cd "$project_root"
container="${LIGHTNAV_CONTAINER:-navigation-lightnav}"
if reuse_container "$container" lightnav; then exit 0; fi

image="${LIGHTNAV_IMAGE:-navigation-lightnav:local}"
source_dir="${LIGHTNAV_SOURCE:-$project_root/.runtime/lightnav/upstream}"
weights="${LIGHTNAV_WEIGHTS:-$source_dir/checkpoints/LightNav-0}"
patch_dir="${LIGHTNAV_PATCH:-$project_root/.runtime/lightnav/capture-time}"
docker image inspect "$image" >/dev/null 2>&1 || fail "缺少镜像 $image；请先运行 make models-setup。"
[[ -f "$weights/config.json" && -f "$source_dir/src/lightnav/serving/ws_server.py" ]] || fail '缺少 LightNav 源码或权重；请先运行 make models-setup。'
[[ -f "$patch_dir/src/lightnav/serving/ws_server.py" ]] || fail '缺少采图时间补丁；请先运行 make models-setup。'
python3 deploy/lightnav/check_weights.py "$weights"
python3 deploy/lightnav/prepare.py "$source_dir" --output "$patch_dir" --reuse
source_dir="$(cd "$source_dir" && pwd)"
weights="$(cd "$weights" && pwd)"
patch_dir="$(cd "$patch_dir" && pwd)"
cache="$project_root/.runtime/lightnav/cache"
output="$project_root/.runtime/lightnav/output"
# 只读源码挂载下的嵌套挂载点必须预先存在；全新源码尚无这些目录。
mkdir -p "$cache" "$output" "$source_dir/.cache" "$source_dir/output"
mounts=(--mount "type=bind,source=$source_dir,target=/workspace/lightnav-0,readonly"
        --mount "type=bind,source=$weights,target=/models,readonly"
        --mount "type=bind,source=$cache,target=/workspace/lightnav-0/.cache"
        --mount "type=bind,source=$output,target=/workspace/lightnav-0/output")
for name in inference/policies.py serving/ws_server.py serving/tracking_service.py data_processor.py processing.py; do
    [[ -f "$patch_dir/src/lightnav/$name" ]] || fail "缺少补丁文件 $name"
    mounts+=(--mount "type=bind,source=$patch_dir/src/lightnav/$name,target=/workspace/lightnav-0/src/lightnav/$name,readonly")
done
# 两类镜像共用启动参数；只设置依赖库搜索路径，不使用镜像内另一套推理默认值。
exec docker run -d --init --pull=never --name "$container" \
    --label navigation-agent.service=lightnav --label "navigation-agent.owner=$(id -un)" \
    --restart no --runtime "${LIGHTNAV_RUNTIME:-nvidia}" --gpus all --shm-size=4g \
    --user "$(id -u):$(id -g)" \
    --log-driver=json-file --log-opt max-size=20m --log-opt max-file=3 \
    -p "127.0.0.1:${LIGHTNAV_PORT:-8053}:8053" "${mounts[@]}" \
    --mount "type=bind,source=$project_root/deploy/lightnav/entrypoint.sh,target=/navigation-entrypoint.sh,readonly" \
    --entrypoint bash -w /workspace/lightnav-0 \
    -e "LOGNAME=$(id -un)" -e "USER=$(id -un)" \
    -e PYTHONPATH=/workspace/lightnav-0/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e HOME=/workspace/lightnav-0/.cache/home -e XDG_CACHE_HOME=/workspace/lightnav-0/.cache \
    -e HF_HOME=/workspace/lightnav-0/.cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_CACHE_PATH=/workspace/lightnav-0/.cache/cuda -e TMPDIR=/workspace/lightnav-0/.cache/tmp \
    -e TRITON_CACHE_DIR=/workspace/lightnav-0/.cache/triton -e TORCHINDUCTOR_CACHE_DIR=/workspace/lightnav-0/.cache/torchinductor \
    -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e LIGHTNAV_ATTN=sdpa \
    "$image" /navigation-entrypoint.sh python -m lightnav.serving.ws_server \
    --backend vllm_local --task vln --device cuda --model_path /models --host 0.0.0.0 --port 8053 \
    --max_batch_size 1 --gpu_memory_utilization "${LIGHTNAV_GPU_MEMORY:-0.35}" \
    --record_dir /workspace/lightnav-0/output/live-episodes --cam_hfov_deg 90 --cam_height 1.5
