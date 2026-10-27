#!/usr/bin/env bash
set -euo pipefail

name=deepseek-4.1-flash-1017-brrr
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<EOF
Usage: bash serve.sh MODEL_DIR [options] [-- vllm serve args...]

Builds the image if needed, then serves DeepSeek-4.1-Flash-1017-BRRR as an
OpenAI-compatible API. The first run takes a while; later runs start straight away.

Options:
  --gpus LIST      four GPU indices (default 0,1,2,3)
  --port N         host port (default 8000)
  --context N      max tokens per request (default 1048576)
  --api-key KEY    require this key as a bearer token
  --adaptive       enable DSpark adaptive verification (faster at high context and concurrency)
  --background     keep serving in the background, restarting on crash and reboot
  -h, --help       show this help
EOF
}

die() {
    echo "serve.sh: $*" >&2
    exit 1
}

model=
gpus=0,1,2,3
port=8000
adaptive=false
mode=(--rm --sig-proxy=true)
vllm=()

while (($#)); do
    case $1 in
        -h | --help) usage; exit ;;
        --gpus) gpus=${2:?--gpus needs a value}; shift 2 ;;
        --port) port=${2:?--port needs a value}; shift 2 ;;
        --context) vllm+=(--max-model-len "${2:?--context needs a value}"); shift 2 ;;
        --api-key) vllm+=(--api-key "${2:?--api-key needs a value}"); shift 2 ;;
        --adaptive) adaptive=true; shift ;;
        --background) mode=(-d --restart unless-stopped); shift ;;
        --) vllm+=("${@:2}"); break ;;
        -*) die "unknown option $1 (see --help)" ;;
        *) [[ -z $model ]] || die "unexpected argument $1"; model=$1; shift ;;
    esac
done

[[ -n $model ]] || { usage >&2; exit 1; }
[[ -f $model/config.json ]] || die "no config.json in $model; pass the DeepSeek-V4.1-Flash-1017 weights directory"
[[ -z $(find "$model" -maxdepth 1 -type l -print -quit) ]] || die "$model holds symlinks (a Hugging Face cache snapshot?); download it with --local-dir instead"
model=$(realpath "$model")
case $(findmnt -no FSTYPE -T "$model") in
    nfs* | cifs | smb* | tmpfs | fuse*) die "$model is on a network or in-memory filesystem; the weights need a local NVMe drive (ext4 or xfs)" ;;
esac
IFS=, read -ra ids <<< "$gpus"
((${#ids[@]} == 4)) || die "--gpus needs exactly four GPUs, got '$gpus'"
command -v docker >/dev/null || die "docker not found; install Docker and the NVIDIA Container Toolkit"
bash "$here/p2p.sh" "$gpus"

docker build -t "$name" "$here"
docker rm -f "$name" >/dev/null 2>&1 || true

run=(
    docker run "${mode[@]}"
    --name "$name"
    --init
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --security-opt seccomp=unconfined
    --gpus all
    -e CUDA_VISIBLE_DEVICES="$gpus"
    -e PORT="$port"
    -e ADAPTIVE="$adaptive"
    -p "$port:$port"
    -v "$model:/model:ro"
    -v "$name-jit:/cache/jit"
    "$name" "${vllm[@]}"
)

[[ ${mode[0]} == -d ]] || exec "${run[@]}"
"${run[@]}" >/dev/null
cat <<EOF
Serving in the background on port $port; it restarts on crash and after reboots.
The API answers once the model has loaded; watch the logs for it.
  logs: docker logs -f $name
  stop: docker rm -f $name
EOF
