#!/usr/bin/env bash
set -euo pipefail
umask 077

# Launch the pinned vLLM 3df4 K3 four-layer NIXL PD target comparison on 114/115.
# The checkpoint is read directly from 3FS; only the task artifact path is local.
role=${1:?producer or consumer}
case "$role" in
  producer) host_ip=11.163.39.114; port=26100; kv_role=kv_producer ;;
  consumer) host_ip=11.163.39.115; port=26200; kv_role=kv_consumer ;;
  *) echo "invalid role: $role" >&2; exit 2 ;;
esac

image='mirrors-ssl.aliyuncs.com/vllm/vllm-openai@sha256:dfaab3570be5b1f66c21e60c60f1616ad3a0143f9899b8738257004f289979fd'
model='/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers'
name="lhc_k3_vllm_3df4_${role}_114115_20260929"
task_dir="/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/vllm-pd-3df4-114115-${role}-20260929"

test "$(id -u)" = 19357313
hostname -I | tr ' ' '\n' | grep -qx "$host_ip"
test "$(findmnt -T "$model" -n -o FSTYPE)" = fuse.hf3fs
test -f "$model/model.safetensors.index.json"
test -d /dev/infiniband
docker image inspect "$image" >/dev/null
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "task container already exists: $name" >&2
  exit 2
fi
if test -e "$task_dir"; then
  echo "task directory already exists: $task_dir" >&2
  exit 2
fi
mkdir -p "$task_dir"/{cache,hf,triton,profiles}

kv_config="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$kv_role\",\"kv_load_failure_policy\":\"fail\"}"
docker run -d \
  --name "$name" --init \
  --user 19357313:100 --group-add 19051 \
  --gpus all --network host --ipc host \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v /dev/infiniband:/dev/infiniband \
  -v /mnt/hf3fs/3fs:/mnt/hf3fs/3fs:ro \
  -v "$task_dir:/task" \
  --workdir /task \
  -e HOME=/task -e XDG_CACHE_HOME=/task/cache \
  -e HF_HOME=/task/hf -e HF_HUB_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/task/triton \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_KIMI_K3_GEMM_AR=0 -e VLLM_KIMI_K3_GEMM_RS=0 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=0 -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST="$host_ip" \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=5610 \
  -e UCX_NET_DEVICES=all \
  "$image" "$model" \
  --host 0.0.0.0 --port "$port" \
  --served-model-name kimi-k3 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --disable-custom-all-reduce \
  --no-enable-flashinfer-autotune \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' \
  --kv-cache-dtype fp8_e4m3 \
  --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"TOKENSPEED_MLA"}' \
  --dtype bfloat16 \
  --max-model-len 69632 --max-num-batched-tokens 65536 \
  --max-num-seqs 1 --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --load-format safetensors --trust-remote-code \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/task/profiles","torch_profiler_with_stack":false,"torch_profiler_use_gzip":false}' \
  --kv-transfer-config "$kv_config"
