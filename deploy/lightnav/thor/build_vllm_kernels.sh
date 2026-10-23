#!/usr/bin/env bash
# Rebuild only vLLM's core extensions; retain the release's Python/API version.
set -euo pipefail
mkdir -p /opt/vllm-build
cd /opt/vllm-build
curl -fL --retry 3 https://codeload.github.com/vllm-project/vllm/tar.gz/refs/tags/v0.19.1 -o vllm.tar.gz
echo '49ee6f462817e2e9a0dab47c7924c6b7716be712b7dcf54aed9fc144bee2e2cc  vllm.tar.gz' | sha256sum -c -
curl -fL --retry 3 https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.2.1 -o cutlass.tar.gz
echo 'a4513ba33ae82fd754843c6d8437bee1ac71a6ef1c74df886de2338e3917d4df  cutlass.tar.gz' | sha256sum -c -
tar -xzf vllm.tar.gz
tar -xzf cutlass.tar.gz
# These external projects are already supplied by the pinned wheel. They are
# unrelated to the core targets and otherwise fetch several more repositories.
python - <<'PY'
from pathlib import Path
p = Path('vllm-0.19.1/CMakeLists.txt')
text = p.read_text()
for name in ('triton_kernels', 'flashmla', 'qutlass', 'vllm_flash_attn'):
    line = f'include(cmake/external_projects/{name}.cmake)'
    assert text.count(line) == 1, f'Unexpected vLLM build file: {name}'
    text = text.replace(line, f'# Thor core-only build: {line}')
p.write_text(text)
PY
cmake -S vllm-0.19.1 -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DVLLM_PYTHON_EXECUTABLE=/opt/lightnav/bin/python \
    -DVLLM_CUTLASS_SRC_DIR=/opt/vllm-build/cutlass-4.2.1 \
    -DFETCHCONTENT_BASE_DIR=/opt/vllm-build/deps \
    -DNVCC_THREADS=1
cmake --build build --target _C _C_stable_libtorch cumem_allocator --parallel "${VLLM_BUILD_JOBS:-2}"
mkdir -p /opt/thor-kernels
cp build/_C.abi3.so build/_C_stable_libtorch.abi3.so build/cumem_allocator.abi3.so /opt/thor-kernels/
# Verify the artifact really contains native Thor code before shipping it.
cuobjdump --list-elf /opt/thor-kernels/_C.abi3.so > /opt/thor-kernels/core-cubins.txt
grep -q 'sm_110' /opt/thor-kernels/core-cubins.txt
