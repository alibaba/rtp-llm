#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda310/bin/python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"

GPU="${GPU:-3}"
PORT="${PORT:-39080}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the model checkpoint}"
MODEL_TYPE="${MODEL_TYPE:-deepseek_v4}"
LOAD_METHOD="${LOAD_METHOD:-fastsafetensors}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
RESERVER_RUNTIME_MEM_MB="${RESERVER_RUNTIME_MEM_MB:-65536}"
ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-0}"
WARM_UP="${WARM_UP:-0}"
HACK_LAYER_NUM="${HACK_LAYER_NUM:-4}"
ACT_TYPE="${ACT_TYPE:-BF16}"
SEQ_SIZE_PER_BLOCK="${SEQ_SIZE_PER_BLOCK:-128}"
KERNEL_SEQ_SIZE_PER_BLOCK="${KERNEL_SEQ_SIZE_PER_BLOCK:-128}"
FP8_KV_CACHE="${FP8_KV_CACHE:-1}"
TEST_BLOCK_NUM="${TEST_BLOCK_NUM:-64}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
WORLD_SIZE="${WORLD_SIZE:-${TP_SIZE}}"
ROLE_TYPE="${ROLE_TYPE:-}"
CP_ROTATE_METHOD="${CP_ROTATE_METHOD:-}"
USE_LOCAL="${USE_LOCAL:-0}"
REUSE_CACHE="${REUSE_CACHE:-0}"
ENABLE_DEVICE_CACHE="${ENABLE_DEVICE_CACHE:-0}"
ENABLE_MEMORY_CACHE="${ENABLE_MEMORY_CACHE:-0}"
MEMORY_CACHE_SIZE_MB="${MEMORY_CACHE_SIZE_MB:-256}"
WRITE_CACHE_SYNC="${WRITE_CACHE_SYNC:-0}"
USE_DEEPEP_MOE="${USE_DEEPEP_MOE:-0}"
USE_DEEPEP_LOW_LATENCY="${USE_DEEPEP_LOW_LATENCY:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
BAZEL_BIN="$(command -v bazelisk || command -v bazel)"
BAZEL_OUTPUT_BASE="$("${BAZEL_BIN}" info output_base 2>/dev/null)"
if [[ -z "${BAZEL_OUTPUT_BASE}" ]]; then
  echo "failed to resolve Bazel output_base" >&2
  exit 2
fi

PIP_REPOS=("${BAZEL_OUTPUT_BASE}"/external/pip_gpu_cuda13_torch_*/site-packages)
PIP_PATH="$(IFS=:; echo "${PIP_REPOS[*]}")"
TORCH_LIB="${BAZEL_OUTPUT_BASE}/external/pip_gpu_cuda13_torch_torch/site-packages/torch/lib"
export PYTHONPATH="${REPO_ROOT}/bazel-bin:${PIP_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${REPO_ROOT}/bazel-bin:${TORCH_LIB}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SHIM="$("${PYTHON_BIN}" -c "
from torch_memory_saver.utils import get_binary_path_from_package
print(get_binary_path_from_package('torch_memory_saver_hook_mode_preload'))")"
if [[ ! -f "${SHIM}" ]]; then
  echo "torch_memory_saver preload shim does not exist: ${SHIM}" >&2
  exit 2
fi

export LD_PRELOAD="${SHIM}${LD_PRELOAD:+:${LD_PRELOAD}}"
export RTP_LLM_FREEZE_WEIGHTS_SAVER=1
export ENABLE_SLEEP_MODE=1
export SLEEP_MODE_LEVEL=3
export RTP_LLM_SLEEP_FREE_MEGA_SYMM="${RTP_LLM_SLEEP_FREE_MEGA_SYMM:-1}"
export USE_RPC_MODEL=1
export LOAD_PYTHON_MODEL=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export START_PORT="${PORT}"
export WARM_UP
export HACK_LAYER_NUM
export TOKENIZER_PATH="${MODEL_PATH}"
export CHECKPOINT_PATH="${MODEL_PATH}"
export MODEL_TYPE
export ACT_TYPE
export SEQ_SIZE_PER_BLOCK
export KERNEL_SEQ_SIZE_PER_BLOCK
export PATH="/opt/rh/gcc-toolset-12/root/usr/bin:${PYTHON_BIN%/*}:${CUDA_HOME}/bin:${PATH}"
export CC="${CC:-/opt/rh/gcc-toolset-12/root/usr/bin/gcc}"
export CXX="${CXX:-/opt/rh/gcc-toolset-12/root/usr/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin=${CXX}}"
export DG_JIT_CPP_STANDARD="${DG_JIT_CPP_STANDARD:-20}"
export DG_JIT_NVCC_PATH="${DG_JIT_NVCC_PATH:-${CUDA_HOME}/bin/nvcc}"

echo "Starting RTP-LLM sleep mode level-3 service"
echo "  repo:        ${REPO_ROOT}"
echo "  gpu:         ${GPU}"
echo "  port:        ${PORT}"
echo "  model_path:  ${MODEL_PATH}"
echo "  model_type:  ${MODEL_TYPE}"
echo "  max_seq_len: ${MAX_SEQ_LEN}"
echo "  hack_layers: ${HACK_LAYER_NUM}"
echo "  parallelism: tp=${TP_SIZE} dp=${DP_SIZE} ep=${EP_SIZE} world=${WORLD_SIZE}"
echo "  role/cp:     role=${ROLE_TYPE:-default} cp_rotate=${CP_ROTATE_METHOD:-default}"
echo "  TMS shim:    ${SHIM}"

ROLE_ARGS=()
if [[ -n "${ROLE_TYPE}" ]]; then
  ROLE_ARGS+=(
    --role_type "${ROLE_TYPE}"
    --use_local "${USE_LOCAL}"
    --reuse_cache "${REUSE_CACHE}"
    --enable_device_cache "${ENABLE_DEVICE_CACHE}"
    --enable_memory_cache "${ENABLE_MEMORY_CACHE}"
    --memory_cache_size_mb "${MEMORY_CACHE_SIZE_MB}"
    --write_cache_sync "${WRITE_CACHE_SYNC}"
    --use_deepep_moe "${USE_DEEPEP_MOE}"
    --use_deepep_low_latency "${USE_DEEPEP_LOW_LATENCY}"
  )
fi
if [[ -n "${CP_ROTATE_METHOD}" ]]; then
  ROLE_ARGS+=(--cp_rotate_method "${CP_ROTATE_METHOD}")
fi

exec "${PYTHON_BIN}" -m rtp_llm.start_server \
  --checkpoint_path "${MODEL_PATH}" \
  --tokenizer_path "${MODEL_PATH}" \
  --model_type "${MODEL_TYPE}" \
  --load_method "${LOAD_METHOD}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --hack_layer_num "${HACK_LAYER_NUM}" \
  --enable_sleep_mode 1 \
  --sleep_mode_level 3 \
  --enable_cuda_graph "${ENABLE_CUDA_GRAPH}" \
  --act_type "${ACT_TYPE}" \
  --tp_size "${TP_SIZE}" \
  --dp_size "${DP_SIZE}" \
  --ep_size "${EP_SIZE}" \
  --world_size "${WORLD_SIZE}" \
  --seq_size_per_block "${SEQ_SIZE_PER_BLOCK}" \
  --kernel_seq_size_per_block "${KERNEL_SEQ_SIZE_PER_BLOCK}" \
  --fp8_kv_cache "${FP8_KV_CACHE}" \
  --test_block_num "${TEST_BLOCK_NUM}" \
  --frontend_server_count 1 \
  --concurrency_limit 1 \
  --max_context_batch_size 1 \
  --reserver_runtime_mem_mb "${RESERVER_RUNTIME_MEM_MB}" \
  "${ROLE_ARGS[@]}"
