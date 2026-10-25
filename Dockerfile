ARG BASE_IMAGE=ghcr.io/local-inference-lab/vllm@sha256:376745af9ebffe0ecb3a9b7ee3d2040027834e58e3248898e52487255f33125a
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.10.10

FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE}

ENV HF_HUB_OFFLINE=1 \
    SAFETENSORS_FAST_GPU=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_DEVICE_MAX_CONNECTIONS=32 \
    OMP_NUM_THREADS=8 \
    CUTE_DSL_ARCH=sm_120a \
    TORCH_CUDA_ARCH_LIST=12.0a \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_LEVEL=SYS \
    NCCL_P2P_READ_ENABLE=1 \
    NCCL_PROTO=LL,LL128,Simple \
    NCCL_MIN_NCHANNELS=16 \
    NCCL_MAX_NCHANNELS=16 \
    NCCL_BUFFSIZE=2097152 \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_USE_AOT_COMPILE=1 \
    VLLM_USE_STANDALONE_COMPILE=1 \
    VLLM_USE_MEGA_AOT_ARTIFACT=1 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    VLLM_USE_FLASHINFER_SAMPLER=1 \
    VLLM_USE_FASTOKENS=1 \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024 \
    VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    VLLM_PCIE_DMA_MIN_BYTES=6MB \
    VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=96KB \
    VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=96KB \
    VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=off \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    VLLM_DS41_DRAFT_NVFP4_HEAD=1 \
    VLLM_DSPARK_SHARD_MARKOV_HEAD=1 \
    VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
    TORCH_EXTENSIONS_DIR=/opt/brrr/torch_extensions \
    VLLM_CACHE_ROOT=/cache/jit \
    TORCHINDUCTOR_CACHE_DIR=/cache/jit \
    TRITON_CACHE_DIR=/cache/jit \
    FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
    B12X_COMPILE_CACHE_DIR=/cache/jit/b12x \
    CUTE_DSL_CACHE_DIR=/cache/jit/cute_dsl \
    TILELANG_CACHE_DIR=/cache/jit/tilelang \
    TVM_FFI_CACHE_DIR=/cache/jit/tvm_ffi \
    CUDA_CACHE_PATH=/cache/jit/cuda

RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,target=/src \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps /src

RUN --mount=type=bind,source=vllm.patch,target=/tmp/vllm.patch \
    patch -p1 --fuzz=0 -d /opt/venv/lib/python3.12/site-packages/vllm -i /tmp/vllm.patch

RUN ["/opt/venv/bin/lil-runtime-bootstrap", "/opt/venv/bin/python", "-m", "brrr.prebuild"]

COPY --chmod=755 entrypoint.sh /usr/local/bin/brrr-serve

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/lil-entrypoint", "/usr/local/bin/brrr-serve"]
