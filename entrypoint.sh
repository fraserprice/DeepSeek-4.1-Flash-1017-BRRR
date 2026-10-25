#!/usr/bin/env bash
set -euo pipefail

exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
    --served-model-name DeepSeek-4.1-Flash-1017-BRRR \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.96 \
    --max-model-len 1048576 \
    --max-num-seqs 16 \
    --reasoning-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v41 \
    --max-num-batched-tokens 4096 \
    --attention-backend B12X \
    --moe-backend b12x \
    --kv-cache-dtype fp8 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":'"${ADAPTIVE:-false}"',"attention_backend":"B12X","moe_backend":"b12x"}' \
    --override-generation-config '{"temperature":1.0,"top_p":1.0}' \
    --tokenizer-mode deepseek_v41 \
    --mm-processor-cache-gb 1 \
    --linear-backend b12x \
    --engram-config '{"projection_tp":true}' \
    --block-size 256 \
    --swa-block-size 128 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --max-cudagraph-capture-size 128 \
    --async-scheduling \
    --enable-scale-out \
    --enable-prompt-tokens-details \
    "$@"
