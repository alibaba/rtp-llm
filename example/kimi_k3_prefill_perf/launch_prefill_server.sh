#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

: "${RUN_ROOT:?RUN_ROOT must point to this run artifact directory}"
: "${OPS_OVERLAY:?OPS_OVERLAY must contain the bundled FlashKDA/DeepGEMM wheels}"

checkpoint="${CHECKPOINT_PATH:-/data0/luohaocheng.lhc/Kimi-K3-4layers-preflight}"
start_port="${START_PORT:-27188}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
server_runfiles="${repo_root}/bazel-bin/example/kimi_k3_prefill_perf/kimi_k3_prefill_server.runfiles"
server_binary="${repo_root}/bazel-bin/example/kimi_k3_prefill_perf/kimi_k3_prefill_server"

if [[ ! -x "${server_binary}" || ! -d "${server_runfiles}" ]]; then
  echo "missing Bazel server binary/runfiles: ${server_binary}" >&2
  echo "build //example/kimi_k3_prefill_perf:kimi_k3_prefill_server first" >&2
  exit 2
fi
if [[ ! -f "${checkpoint}/config.json" ]]; then
  echo "checkpoint config not found: ${checkpoint}/config.json" >&2
  exit 2
fi

export TMPDIR="${K3_PERF_TMPDIR:-/tmp/k3p-${start_port}-$$}"
export CUDA_VISIBLE_DEVICES="${cuda_devices}"
export PYTHONPATH="${OPS_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RTP_LLM_STARTUP_TIMEOUT_S="${RTP_LLM_STARTUP_TIMEOUT_S:-14400}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export LOG_PATH="${RUN_ROOT}/logs/service"
export START_PORT="${start_port}"
export FRONTEND_SERVER_COUNT=1
export MODEL_TYPE=kimi_k3
export CHECKPOINT_PATH="${checkpoint}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${checkpoint}}"
export LOAD_METHOD=fastsafetensors

export KIMI_K3_EXECUTION_MODE=optimized
export KIMI_K3_PERF_MODE=1
export KIMI_K3_PERF_FUSIONS=1
export KIMI_K3_USE_HOST_METADATA=1
export KIMI_K3_SP_MOE=1
export KIMI_K3_KDA_BACKEND=flash_kda
export KIMI_K3_MOE_BACKEND=deep_gemm_mega
export KIMI_K3_MLA_BACKEND=kernel
export KIMI_K3_DEEPGEMM_EXPECTED_PATH="${OPS_OVERLAY}"
export KIMI_K3_MEGA_MAX_TOKENS_PER_RANK=8192
export KIMI_K3_ACCURACY_ALLOW_TOKEN_IDS=1
export KIMI_K3_ACCURACY_CANONICAL_TP=0
export KIMI_K3_ACCURACY_CANONICAL_EP=0
export KIMI_K3_ACCURACY_CANONICAL_MLA=0
export KIMI_K3_ACCURACY_LOCAL_EAGER_MLA=0

export DSV4_MEGA_MOE_INPUT_PACKER=fused
export DG_JIT_CACHE_DIR="${K3_PERF_DG_JIT_CACHE_DIR:-${HOME}/.cache/kimi_k3_prefill_perf/deep_gemm_sm103}"
export DG_JIT_USE_NVRTC=0
export DG_JIT_WITH_LINEINFO=0
export DG_PRINT_CONFIGS="${DG_PRINT_CONFIGS:-1}"
export TORCH_CUDA_PROFILER_DIR="${RUN_ROOT}/traces"
export GEN_TIMELINE_SYNC=1

unset REMOTE_RPC_SERVER_IP
unset MODEL_SERVICE_CONFIG
unset KIMI_K3_CPU_OFFLOAD_EXPERT_LAYER_START
unset KIMI_K3_FULL_DECODE_CPU_OFFLOAD_START
unset KIMI_K3_ACCURACY_TRACE_DIR
unset KIMI_K3_ACCURACY_TRACE_MODE
unset KIMI_K3_ACCURACY_TRACE_ENABLE_FILE
unset KIMI_K3_ACCURACY_TRACE_FULL_ROUTER

mkdir -p \
  "${TMPDIR}" \
  "${LOG_PATH}" \
  "${TORCH_CUDA_PROFILER_DIR}" \
  "${DG_JIT_CACHE_DIR}" \
  "${RUN_ROOT}/work"
cd "${RUN_ROOT}/work"

exec "${server_binary}" \
  --role_type PDFUSION \
  --tp_size 8 \
  --dp_size 1 \
  --ep_size 8 \
  --world_size 8 \
  --local_world_size 8 \
  --max_seq_len 524289 \
  --max_context_batch_size 1 \
  --max_batch_tokens_size 524288 \
  --seq_size_per_block 4096 \
  --kernel_seq_size_per_block 128 \
  --kv_cache_mem_mb 8192 \
  --ssm_state_dtype fp32 \
  --warm_up 0 \
  --reuse_cache 0 \
  --enable_device_cache 1 \
  --concurrency_limit 1 \
  --use_deepep_moe 1 \
  --use_deepep_internode 0 \
  --use_deepep_low_latency 0 \
  --deep_ep_num_sm 24 \
  --use_all_gather 0 \
  --enable_cuda_graph 0 \
  --load_method fastsafetensors \
  --ft_core_dump_on_exception 0 \
  --shutdown_timeout 5
