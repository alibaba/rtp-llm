#!/usr/bin/env bash
set -euo pipefail

# Launch one side of the pinned vLLM 3df4 K3 four-layer NIXL PD online FP8 E4M3 comparison.
# All paths are task-owned; the checkpoint is read directly from 3FS.
role=${1:?producer or consumer}
case "$role" in
  producer) host_ip=11.163.39.114; port=26300; kv_role=kv_producer; task_dir=/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/vllm-pd-3df4-fp8-3fs-r6-producer ;;
  consumer) host_ip=11.163.39.115; port=26400; kv_role=kv_consumer; task_dir=/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/vllm-pd-3df4-fp8-3fs-r6-consumer ;;
  *) echo "invalid role: $role" >&2; exit 2 ;;
esac

image='mirrors-ssl.aliyuncs.com/vllm/vllm-openai@sha256:dfaab3570be5b1f66c21e60c60f1616ad3a0143f9899b8738257004f289979fd'
model='/data0/luohaocheng.lhc/models/kimi-k3-4layers-ms-3fs-view-20260929'
name="lhc_k3_vllm_3df4_fp8_3fs_r6_${role}_20260929"

test "$(id -u)" = 19357313
test -f "$model/model.safetensors.index.json"
test "$(findmnt -T "$model" -n -o FSTYPE)" = xfs
test "$(findmnt -T /mnt/hf3fs/3fs/models/kimi/kimi-k3 -n -o FSTYPE)" = fuse.hf3fs
test "$(readlink -f "$model/model-00001-of-00007.safetensors")" = /mnt/hf3fs/3fs/models/kimi/kimi-k3/model-00001-of-000096.safetensors
test "$(sha256sum "$model/config.json" | cut -d' ' -f1)" = 72e146f1be7061dc281ab86ad9a1544c8f5dcdbfbe510c92b85a0d943b58fe54
test "$(sha256sum "$model/model.safetensors.index.json" | cut -d' ' -f1)" = dc08028ec4f45b41dfe51d5edff3168cc088404a92cf3b72dd9daaa10e29b358
test -d /dev/infiniband
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "task container already exists: $name" >&2
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
  -v "$model:$model:ro" \
  -v /mnt/hf3fs/3fs/models/kimi/kimi-k3:/mnt/hf3fs/3fs/models/kimi/kimi-k3:ro \
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
  --dtype bfloat16 --quantization fp8 --moe-backend deep_gemm \
  --max-model-len 69632 --max-num-batched-tokens 65536 \
  --max-num-seqs 1 --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --load-format safetensors --trust-remote-code \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/task/profiles","torch_profiler_with_stack":false,"torch_profiler_use_gzip":false}' \
  --kv-transfer-config "$kv_config"
