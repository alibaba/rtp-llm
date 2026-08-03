#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

: "${RUN_ROOT:?RUN_ROOT must point to this run artifact directory}"
: "${OPS_OVERLAY:?OPS_OVERLAY must contain cuLA/FLA/FlashKDA/DeepGEMM}"

checkpoint="${CHECKPOINT_PATH:-/data0/luohaocheng.lhc/Kimi-K3-4layers-preflight}"
start_port="${START_PORT:-27188}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"
kda_comm_backend="${KIMI_K3_KDA_COMM_BACKEND:-rs_ag}"
enable_cuda_graph="${ENABLE_CUDA_GRAPH:-0}"
enable_cuda_graph_debug_mode="${ENABLE_CUDA_GRAPH_DEBUG_MODE:-0}"
decode_capture_config="${DECODE_CAPTURE_CONFIG:-}"
prefill_capture_config="${PREFILL_CAPTURE_CONFIG:-}"

for flag_name in enable_cuda_graph enable_cuda_graph_debug_mode; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "${flag_name} must resolve to 0 or 1, got ${flag_value}" >&2
    exit 2
  fi
done
if [[ "${enable_cuda_graph_debug_mode}" == "1" && "${enable_cuda_graph}" != "1" ]]; then
  echo "ENABLE_CUDA_GRAPH_DEBUG_MODE=1 requires ENABLE_CUDA_GRAPH=1" >&2
  exit 2
fi
server_runfiles="${repo_root}/bazel-bin/example/kimi_k3_prefill_perf/kimi_k3_prefill_server.runfiles"
server_binary="${repo_root}/bazel-bin/example/kimi_k3_prefill_perf/kimi_k3_prefill_server"
flashinfer_site_packages="${server_runfiles}/pip_gpu_cuda13_torch_flashinfer_python/site-packages"
tvm_ffi_site_packages="${server_runfiles}/pip_gpu_cuda13_torch_apache_tvm_ffi/site-packages"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python binary is not executable: ${python_bin}" >&2
  exit 2
fi
if [[ ! -x "${server_binary}" || ! -d "${server_runfiles}" ]]; then
  echo "missing Bazel server binary/runfiles: ${server_binary}" >&2
  echo "build //example/kimi_k3_prefill_perf:kimi_k3_prefill_server first" >&2
  exit 2
fi
if [[ ! -f "${checkpoint}/config.json" ]]; then
  echo "checkpoint config not found: ${checkpoint}/config.json" >&2
  exit 2
fi
if [[ "${kda_comm_backend}" != "rs_ag" && "${kda_comm_backend}" != "a2a" ]]; then
  echo "KIMI_K3_KDA_COMM_BACKEND must be rs_ag or a2a" >&2
  exit 2
fi

# CpuTpBroadcaster uses a Unix-domain socket below TMPDIR, whose full path
# must stay below 108 bytes.  /dev/shm avoids both that limit and host /tmp
# inode pressure while keeping the large JIT caches under RUN_ROOT.
export TMPDIR="${K3_PERF_TMPDIR:-/dev/shm/k3p-${start_port}-$$}"
export PATH="$(dirname -- "${python_bin}"):${PATH}"
export CUDA_VISIBLE_DEVICES="${cuda_devices}"
# The lhc_GPU image may carry a PYTHONPATH from a different Bazel output base.
# Put this target's runfiles packages first so FlashInfer JIT resolves headers
# from the same build that launches the server.
export PYTHONPATH="${OPS_OVERLAY}:${flashinfer_site_packages}:${tvm_ffi_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLA_TILELANG=0
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
export KIMI_K3_PERF_MODE="${KIMI_K3_PERF_MODE:-1}"
export KIMI_K3_PERF_FUSIONS=1
export KIMI_K3_USE_HOST_METADATA=1
export KIMI_K3_SP_MOE=1
export KIMI_K3_KDA_BACKEND="${KIMI_K3_KDA_BACKEND:-cula}"
export KIMI_K3_KDA_COMM_BACKEND="${kda_comm_backend}"
export KIMI_K3_KDA_A2A_SAFETY_GIB="${KIMI_K3_KDA_A2A_SAFETY_GIB:-8}"
export KIMI_K3_MOE_BACKEND=deep_gemm_mega
export KIMI_K3_MLA_BACKEND="${KIMI_K3_MLA_BACKEND:-flashmla}"
export KIMI_K3_DEEPGEMM_EXPECTED_PATH="${OPS_OVERLAY}"
export KIMI_K3_MEGA_MAX_TOKENS_PER_RANK=8192
export KIMI_K3_ACCURACY_ALLOW_TOKEN_IDS=1
export KIMI_K3_ACCURACY_CANONICAL_TP=0
export KIMI_K3_ACCURACY_CANONICAL_EP=0
export KIMI_K3_ACCURACY_CANONICAL_MLA=0
export KIMI_K3_ACCURACY_LOCAL_EAGER_MLA=0

export DSV4_MEGA_MOE_INPUT_PACKER=fused
export DG_JIT_CACHE_DIR="${K3_PERF_DG_JIT_CACHE_DIR:-${RUN_ROOT}/runtime/deep_gemm_cache}"
export TRITON_CACHE_DIR="${K3_PERF_TRITON_CACHE_DIR:-${RUN_ROOT}/runtime/triton_cache}"
export FLASHINFER_WORKSPACE_BASE="${K3_PERF_FLASHINFER_WORKSPACE_BASE:-${RUN_ROOT}/runtime/flashinfer_workspace}"
export DG_JIT_USE_NVRTC=0
export DG_JIT_WITH_LINEINFO=0
export DG_PRINT_CONFIGS="${DG_PRINT_CONFIGS:-1}"
export TORCH_CUDA_PROFILER_DIR="${RUN_ROOT}/traces"
export GEN_TIMELINE_SYNC=1

unset REMOTE_RPC_SERVER_IP
unset MODEL_SERVICE_CONFIG
unset KIMI_K3_CPU_OFFLOAD_EXPERT_LAYER_START
unset KIMI_K3_FULL_DECODE_CPU_OFFLOAD_START
if [[ "${KIMI_K3_PERF_MODE}" == "1" ]]; then
  unset KIMI_K3_ACCURACY_TRACE_DIR
  unset KIMI_K3_ACCURACY_TRACE_MODE
  unset KIMI_K3_ACCURACY_TRACE_ENABLE_FILE
  unset KIMI_K3_ACCURACY_TRACE_FULL_ROUTER
fi

mkdir -p \
  "${TMPDIR}" \
  "${LOG_PATH}" \
  "${TORCH_CUDA_PROFILER_DIR}" \
  "${DG_JIT_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${RUN_ROOT}/work"
echo \
  "[K3_PERF_CONFIG] kda_comm=${KIMI_K3_KDA_COMM_BACKEND} " \
  "kda=${KIMI_K3_KDA_BACKEND} moe=${KIMI_K3_MOE_BACKEND} " \
  "mla=${KIMI_K3_MLA_BACKEND} graph=${enable_cuda_graph}" \
  >&2
cd "${RUN_ROOT}/work"

server_args=(
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
  --enable_cuda_graph "${enable_cuda_graph}" \
  --enable_cuda_graph_debug_mode "${enable_cuda_graph_debug_mode}" \
  --load_method fastsafetensors \
  --ft_core_dump_on_exception 0 \
  --shutdown_timeout 5
)
if [[ -n "${decode_capture_config}" ]]; then
  server_args+=(--decode_capture_config "${decode_capture_config}")
fi
if [[ -n "${prefill_capture_config}" ]]; then
  server_args+=(--prefill_capture_config "${prefill_capture_config}")
fi

exec "${server_binary}" "${server_args[@]}"
