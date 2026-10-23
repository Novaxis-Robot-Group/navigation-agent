#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "用法：deploy/detector/build.sh 已安装的兼容GPU基础镜像" >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker image inspect "$1" >/dev/null
exec docker build --pull=false --build-arg "BASE_IMAGE=$1" \
  -t "${DETECTOR_IMAGE:-navigation-target-detector:local}" \
  -f "$repo_dir/deploy/detector/Dockerfile" "$repo_dir"
