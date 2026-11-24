#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=${QWEN36_RL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LEGACY=${RTP_LEGACY_ROOT:?Set RTP_LEGACY_ROOT to the native Python runtime directory}
export RTP_SOURCE=${RTP_SOURCE:-$(cd "$ROOT/../../.." && pwd -P)}
export RTP_PYTHON=${RTP_PYTHON:-$LEGACY/.rtp/bin/python}
export RTP_NATIVE_LIBRARY_DIR=${RTP_NATIVE_LIBRARY_DIR:-$ROOT/build/rtp-native/rawlp-v1}
export RTP_KERNEL_PYTHON_PATH=${RTP_KERNEL_PYTHON_PATH:-$ROOT/build/rtp-native/python-deps}
export RTP_MODEL_PATH=${RTP_MODEL_PATH:?RTP_MODEL_PATH must be the pinned local/OSS checkpoint}
export RTP_START_PORT=${RTP_START_PORT:-24000}
export RTP_TP_SIZE=${RTP_TP_SIZE:-8}
export RTP_EP_SIZE=${RTP_EP_SIZE:-1}
export RTP_MOE_PURE_TP_PRESHARD=${RTP_MOE_PURE_TP_PRESHARD:-false}
# The pod's rank must not override the explicit rank within this TP replica.
unset WORLD_RANK
export LOCAL_WORLD_SIZE=${RTP_LOCAL_WORLD_SIZE:-$RTP_TP_SIZE}
export RTP_LLM_RL_MODE=1
export RTP_LLM_RL_RUN_ID=${RTP_LLM_RL_RUN_ID:-$("$RTP_PYTHON" -c 'import uuid; print(uuid.uuid4().hex)')}
export RTP_BATCH_INVARIANT=${RTP_BATCH_INVARIANT:-0}
export RTP_BI_HF_EAGER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${RTP_OMP_NUM_THREADS:-4}
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
export PATH="$(dirname "$RTP_PYTHON"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT/rtp:$ROOT/adapter:$RTP_SOURCE:$RTP_SOURCE/bazel-bin:$RTP_NATIVE_LIBRARY_DIR${PYTHONPATH:+:$PYTHONPATH}"
RTP_NVIDIA_LIB_DIRS="$("$RTP_PYTHON" -c 'import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia"; print(":".join(str(x) for x in sorted(p.glob("*/lib"))))')"
RTP_TORCH_LIB_DIR="$("$RTP_PYTHON" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "lib")' 2>/dev/null || echo "$LEGACY/.rtp/lib/python3.10/site-packages/torch/lib")"
export LD_LIBRARY_PATH="$RTP_NATIVE_LIBRARY_DIR:$RTP_SOURCE/rtp_llm/libs:$RTP_SOURCE/bazel-bin:$RTP_SOURCE/bazel-bin/external/flashinfer_cpp_cu13:$RTP_TORCH_LIB_DIR:$RTP_NVIDIA_LIB_DIRS:/opt/conda310/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME="$ROOT/cache/rtp"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions/rtp-py310"
export LOG_PATH="${RLLM_RTP_ACCEPTANCE_DIR:-$ROOT/logs}/native-replica-${RTP_REPLICA_RANK:-standalone}"
export MAX_CONTEXT_BATCH_SIZE=${RTP_MAX_CONTEXT_BATCH_SIZE:-8}
export CONCURRENCY_LIMIT=${RTP_CONCURRENCY_LIMIT:-8}
export MAX_BATCH_TOKENS_SIZE=${RTP_MAX_BATCH_TOKENS_SIZE:-16384}
export RLLM_RTP_RANK_PORT_STRIDE=${RLLM_RTP_RANK_PORT_STRIDE:-9}
if [[ "$RTP_EP_SIZE" != 1 ]]; then
  echo 'The native RL engine currently requires EP=1; use TP tensor sharding for this acceptance.' >&2
  exit 2
fi
if [[ "$RLLM_RTP_RANK_PORT_STRIDE" != 9 ]]; then
  echo 'The pinned native RTP configuration uses a nine-port rank stride.' >&2
  exit 2
fi
mkdir -p "$LOG_PATH" "$TORCH_EXTENSIONS_DIR"
if [[ "$(uname -m)" != "aarch64" && "${RTP_SKIP_ARM_NCCL_SCOPE:-0}" == "1" ]]; then
  echo "Skipping ARM64 pinned NCCL scope on $(uname -m) (RTP_SKIP_ARM_NCCL_SCOPE=1)." >&2
else
  # Select one pinned project-local NCCL, including Torch's absolute-path lookup.
  # scope_config validates the official ARM64 bytes before any native GPU launch.
  RTP_NCCL_EXPORTS=$("$RTP_PYTHON" - "$ROOT" <<'PY_NCCL_SCOPE'
import shlex, sys
from q397_nccl_scope import scope_config
for key, value in scope_config(sys.argv[1])["environment"].items():
    print("export " + key + "=" + shlex.quote(value))
PY_NCCL_SCOPE
)
  eval "$RTP_NCCL_EXPORTS"
  unset RTP_NCCL_EXPORTS
fi
cd "$RTP_SOURCE"
EXTRA_ARGS=()
if [[ "${RTP_GEN_TIMELINE_SYNC:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    --gen_timeline_sync true
    --gen_timeline_start_step "${RTP_GEN_TIMELINE_START_STEP:-0}"
    --gen_timeline_num_steps "${RTP_GEN_TIMELINE_NUM_STEPS:-2}"
    --gen_timeline_trace_name "${RTP_GEN_TIMELINE_TRACE_NAME:-profiler}"
  )
fi
exec "$RTP_PYTHON" "${RTP_ENTRYPOINT:-$ROOT/rtp/entrypoint.py}" \
  --model_type "${RTP_MODEL_TYPE:-qwen35_moe_text}" --checkpoint_path "$RTP_MODEL_PATH" \
  --start_port "$RTP_START_PORT" --act_type BF16 \
  --world_size "$RTP_TP_SIZE" --local_world_size "$LOCAL_WORLD_SIZE" --world_rank "${RTP_WORLD_RANK:-0}" \
  --tp_size "$RTP_TP_SIZE" --dp_size 1 --ep_size "$RTP_EP_SIZE" --ffn_sp_size 1 \
  --moe_pure_tp_preshard "$RTP_MOE_PURE_TP_PRESHARD" \
  --frontend_server_count 0 --frontend_pre_stop_drain_seconds 0 \
  --dash_sc_grpc_pre_stop_drain_seconds 0 --backend_post_frontend_drain_seconds 0 \
  --batch_invariant "${RTP_ENABLE_BI:-false}" --enable_fp32_lm_head false \
  --max_seq_len "${RTP_MAX_SEQ_LEN:-8192}" --kv_cache_mem_mb "${RTP_KV_CACHE_MEM_MB:-4096}" \
  --ssm_state_dtype fp32 --seq_size_per_block 8192 --kernel_seq_size_per_block 64 \
  --max_context_batch_size "$MAX_CONTEXT_BATCH_SIZE" --max_batch_tokens_size "$MAX_BATCH_TOKENS_SIZE" \
  --concurrency_limit "$CONCURRENCY_LIMIT" \
  --reuse_cache false --enable_device_cache false --enable_remote_cache false \
  --warm_up false --enable_cuda_graph "${RTP_ENABLE_CUDA_GRAPH:-false}" --enable_native_cuda_graph false \
  "${EXTRA_ARGS[@]}" \
  "$@"
