# syntax=docker/dockerfile:1.4
# Jetson Thor / ARM64 / CUDA 13.0. Weights are mounted at runtime.
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-base-ubuntu24.04
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS dependencies

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/lightnav PATH=/opt/lightnav/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates build-essential ffmpeg \
    && python3 -m pip install --break-system-packages --no-cache-dir uv==0.11.21 \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.11 && uv venv --python 3.11 /opt/lightnav

WORKDIR /opt/lightnav-src
COPY pyproject.toml README.md LICENSE ./
COPY --from=deployment constraints.txt /opt/thor-constraints.txt
# Use the release's CUDA 13 ARM wheel; PyPI's default vLLM wheel is CUDA 12.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/lightnav/bin/python \
    --constraint /opt/thor-constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    'torch==2.10.0+cu130' 'torchvision==0.25.0+cu130' 'torchaudio==2.10.0+cu130' \
    'https://github.com/vllm-project/vllm/releases/download/v0.19.1/vllm-0.19.1%2Bcu130-cp38-abi3-manylinux_2_35_aarch64.whl' \
    -r pyproject.toml --extra vllm --extra video --extra test ruff
COPY src ./src
# NVIDIA's cuSPARSELt 0.8.0 ARM wheel has a nonstandard `sbsa` WHEEL tag.
# Verify the actual ELF before correcting that tag; keep dependency checks on.
COPY --from=deployment fix_cusparselt_metadata.py /opt/fix_cusparselt_metadata.py
RUN uv pip install --python /opt/lightnav/bin/python --no-deps . pip \
    && python /opt/fix_cusparselt_metadata.py \
    && uv pip check --python /opt/lightnav/bin/python \
    && python -m pip check

# Triton 3.6 bundles a CUDA 12 ptxas-blackwell, which rejects Thor sm_110a.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/lightnav/bin/python 'nvidia-cuda-nvcc==13.0.88' \
    'nvidia-cuda-crt==13.0.88' 'nvidia-nvvm==13.0.88' \
    && /opt/lightnav/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas --version

# The release ARM wheel lacks runnable Thor kernels in its core extension.
# Rebuild the same version against the same torch ABI for SM 11.0. Keep nvcc,
# headers and source out of the runtime image.
FROM ${CUDA_DEVEL_IMAGE} AS vllm-builder
ENV PATH=/opt/lightnav/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=11.0
COPY --from=dependencies /opt/python /opt/python
COPY --from=dependencies /opt/lightnav /opt/lightnav
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deployment build_vllm_kernels.sh /opt/build_vllm_kernels.sh
ARG VLLM_BUILD_JOBS=2
RUN bash /opt/build_vllm_kernels.sh

FROM dependencies AS runtime
COPY --from=vllm-builder /opt/thor-kernels/ /opt/lightnav/lib/python3.11/site-packages/vllm/
COPY --from=deployment env.sh /etc/profile.d/lightnav-thor.sh
COPY --from=deployment entrypoint.sh /usr/local/bin/lightnav-thor-entrypoint
RUN chmod +x /usr/local/bin/lightnav-thor-entrypoint \
    && echo 'source /etc/profile.d/lightnav-thor.sh' >> /etc/bash.bashrc
# A passwd entry is needed by Python/getpass and compiler cache code. Build for
# the host UID/GID so bind-mounted output stays owned by the developer.
ARG LIGHTNAV_UID=1000
ARG LIGHTNAV_GID=1000
RUN (getent group "${LIGHTNAV_GID}" || groupadd --gid "${LIGHTNAV_GID}" lightnav) \
    && (getent passwd "${LIGHTNAV_UID}" || useradd --create-home \
        --uid "${LIGHTNAV_UID}" --gid "${LIGHTNAV_GID}" --shell /bin/bash lightnav)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    VLN_KV_CACHE_GIB=2 VLN_VLLM_ENFORCE_EAGER=0 \
    TRITON_PTXAS_PATH=/opt/lightnav/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas \
    TRITON_PTXAS_BLACKWELL_PATH=/opt/lightnav/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas \
    VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel,FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel \
    XDG_CACHE_HOME=/workspace/lightnav-0/.cache \
    CUDA_CACHE_PATH=/workspace/lightnav-0/.cache/cuda CUDA_CACHE_MAXSIZE=1073741824 \
    HF_HOME=/workspace/lightnav-0/.cache/huggingface \
    PYTHONPATH=/workspace/lightnav-0/src
WORKDIR /workspace/lightnav-0
USER ${LIGHTNAV_UID}:${LIGHTNAV_GID}
ENTRYPOINT ["/usr/local/bin/lightnav-thor-entrypoint"]
CMD ["sleep", "infinity"]
