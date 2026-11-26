#!/usr/bin/env bash
# Launch one native RTP-LLM RL replica (CUDA 13, wheel-installed Py3.10 venv).
# Avoids mixing source tree and bazel-bin so librtp_compute_ops / libth_transformer load once.
set -euo pipefail
ulimit -c 0

export RTP_PYTHON=${RTP_PYTHON:?Set RTP_PYTHON to the Py3.10 RTP-LLM venv python with rtp_llm wheel installed}
export RTP_MODEL_PATH=${RTP_MODEL_PATH:?Set RTP_MODEL_PATH to the pinned checkpoint directory}
export RTP_MODEL_TYPE=${RTP_MODEL_TYPE:-qwen4_exp}
export RTP_START_PORT=${RTP_START_PORT:-24000}
export RTP_TP_SIZE=${RTP_TP_SIZE:-8}
export RTP_DP_SIZE=${RTP_DP_SIZE:-1}
export RTP_EP_SIZE=${RTP_EP_SIZE:-$RTP_TP_SIZE}
export RTP_WORLD_SIZE=${RTP_WORLD_SIZE:-$((RTP_TP_SIZE * RTP_DP_SIZE))}
export RTP_WORLD_RANK=${RTP_WORLD_RANK:-0}
export LOCAL_WORLD_SIZE=${RTP_LOCAL_WORLD_SIZE:-$RTP_WORLD_SIZE}
export RLLM_RTP_RANK_PORT_STRIDE=${RLLM_RTP_RANK_PORT_STRIDE:-9}
unset WORLD_RANK

export RTP_LLM_RL_MODE=1
export RTP_LLM_RL_RUN_ID=${RTP_LLM_RL_RUN_ID:-$("$RTP_PYTHON" -c 'import uuid; print(uuid.uuid4().hex)')}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${RTP_OMP_NUM_THREADS:-4}
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$(dirname "$RTP_PYTHON"):$CUDA_HOME/bin:$PATH"

# The training Ray worker may set CUDNN_HOME/CUDNN_PATH to its Py3.12 wheel.
# Resolve these paths again for the separately installed Py3.10 serving wheel.
RTP_CUDNN_ROOT="$("$RTP_PYTHON" -c '
import pathlib, sysconfig
root = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cudnn"
print(root if (root / "lib" / "libcudnn.so.9").is_file() else "")
')"
if [[ -n "$RTP_CUDNN_ROOT" ]]; then
  export CUDNN_HOME="$RTP_CUDNN_ROOT"
  export CUDNN_PATH="$RTP_CUDNN_ROOT"
else
  unset CUDNN_HOME CUDNN_PATH
fi

# Strip any source-tree RTP_SOURCE from PYTHONPATH when running the wheel-installed runtime
# so Python never loads duplicate copies of libth_transformer.so / librtp_compute_ops.so.
unset PYTHONPATH

RTP_SITE_LIBS="$("$RTP_PYTHON" -c '
import pathlib, sysconfig
p = pathlib.Path(sysconfig.get_paths()["purelib"])
dirs = [str(p / "rtp_llm" / "libs"), str(p / "torch" / "lib")]
dirs.extend(str(x) for x in sorted((p / "nvidia").glob("*/lib")))
print(":".join(d for d in dirs if pathlib.Path(d).is_dir()))
')"
export LD_LIBRARY_PATH="$RTP_SITE_LIBS:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CACHE_BASE=${RTP_CACHE_BASE:-${XDG_CACHE_HOME:-$HOME/.cache}/qwen4_rl_infer_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CACHE_BASE/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_BASE/triton}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$CACHE_BASE/torch_extensions}
export LOG_PATH="${RLLM_RTP_ACCEPTANCE_DIR:-$CACHE_BASE/logs}/native-replica-${RTP_REPLICA_RANK:-standalone}"
mkdir -p "$LOG_PATH" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# Official Qwen4-Exp CUDA13 FlashInfer/QSA/GDN runtime defaults.
export QWEN4_GDN_BACKEND=${QWEN4_GDN_BACKEND:-flashinfer}
export QWEN4_QSA_INDEXER_BACKEND=${QWEN4_QSA_INDEXER_BACKEND:-flashinfer}
export QWEN4_QSA_FUSED_INDEXER_PREP=${QWEN4_QSA_FUSED_INDEXER_PREP:-1}

EXTRA_MODEL_ARGS=()
if [[ "$RTP_MODEL_TYPE" == "qwen4_exp" ]]; then
  EXTRA_MODEL_ARGS+=(
    --use_deepep_moe 0
    --vit_separation 2
    --seq_size_per_block "${RTP_SEQ_SIZE_PER_BLOCK:-128}"
    --kernel_seq_size_per_block "${RTP_KERNEL_SEQ_SIZE_PER_BLOCK:-128}"
    --linear_pool_blocks "${RTP_LINEAR_POOL_BLOCKS:-8}"
    --decode_capture_config "${RTP_DECODE_CAPTURE_CONFIG:-1}"
    --use_batch_decode_scheduler 0
  )
fi

exec "$RTP_PYTHON" -m rtp_llm.start_server \
  --model_type "$RTP_MODEL_TYPE" \
  --checkpoint_path "$RTP_MODEL_PATH" \
  --start_port "$RTP_START_PORT" \
  --act_type BF16 \
  --world_size "$RTP_WORLD_SIZE" \
  --local_world_size "$LOCAL_WORLD_SIZE" \
  --world_rank "$RTP_WORLD_RANK" \
  --tp_size "$RTP_TP_SIZE" \
  --dp_size "$RTP_DP_SIZE" \
  --ep_size "$RTP_EP_SIZE" \
  --ffn_sp_size 1 \
  --frontend_server_count 0 \
  --frontend_pre_stop_drain_seconds 0 \
  --dash_sc_grpc_pre_stop_drain_seconds 0 \
  --backend_post_frontend_drain_seconds 0 \
  --enable_fp32_lm_head 0 \
  --max_seq_len "${RTP_MAX_SEQ_LEN:-8192}" \
  --kv_cache_mem_mb "${RTP_KV_CACHE_MEM_MB:-8192}" \
  --max_context_batch_size "${RTP_MAX_CONTEXT_BATCH_SIZE:-16}" \
  --max_batch_tokens_size "${RTP_MAX_BATCH_TOKENS_SIZE:-16384}" \
  --concurrency_limit "${RTP_CONCURRENCY_LIMIT:-16}" \
  --reuse_cache "${RTP_REUSE_CACHE:-1}" \
  --enable_device_cache "${RTP_ENABLE_DEVICE_CACHE:-1}" \
  --enable_remote_cache 0 \
  --warm_up 0 \
  --enable_cuda_graph "${RTP_ENABLE_CUDA_GRAPH:-1}" \
  "${EXTRA_MODEL_ARGS[@]}" \
  "$@"
