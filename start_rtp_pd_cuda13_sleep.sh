#!/usr/bin/env bash
# cuda13 PD-separated e2e launch for DeepSeek-V4-Flash, with sleep enabled.
#   - prefill: tp2/ep2/world2 on GPUs 6,7  (START_PORT=22000, cuda_graph=0)
#   - decode : tp1/dp2/ep2/world2 on GPUs 4,5 (START_PORT=21000, cuda_graph=1)
#   - PD KV transfer over RDMA (--cache_store_rdma_mode 1, Barex/ACCL backend)
#   - full smoke config: MTP spec-decode + framework KV + disk/memory cache
#     (from smoke_v4_flash_pd_cp2ep2_dp2ep2_sm100_fp8kv1_mtp_batch4, BUILD:1683)
#   - sleep via torch_memory_saver preload; SLEEP_MODE_LEVEL={1,2,3}
# Toolchain / TMS header adapted verbatim from start_rtp_cp4_cuda13_sleep.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
BAZEL_BIN_DIR="$REPO_ROOT/bazel-bin"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda310/bin/python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"

BAZEL_BIN="$(command -v bazelisk || command -v bazel)"
BAZEL_OUTPUT_BASE="$(USE_BAZEL_VERSION=6.4.0 "$BAZEL_BIN" info output_base 2>/dev/null)"
if [[ -z "$BAZEL_OUTPUT_BASE" ]]; then
    echo "[start_pd] failed to resolve bazel output_base" >&2
    exit 1
fi

# 1. cuda13 pip site-packages
PIP_REPOS=( "$BAZEL_OUTPUT_BASE"/external/pip_gpu_cuda13_torch_*/site-packages )
PIP_PATH="$(IFS=:; echo "${PIP_REPOS[*]}")"
TORCH_LIB="$BAZEL_OUTPUT_BASE/external/pip_gpu_cuda13_torch_torch/site-packages/torch/lib"
export PYTHONPATH="$BAZEL_BIN_DIR:$PIP_PATH${PYTHONPATH:+:$PYTHONPATH}"

# 2. RDMA needs pinned memory. Raise memlock before installing any LD_PRELOAD
# shim so sudo/prlimit runs in a clean process environment. Both role trees
# inherit the updated limit from this shell.
if ! ulimit -l unlimited 2>/dev/null; then
    if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
        sudo prlimit --memlock=unlimited:unlimited --pid $$ \
            && echo "[start_pd] raised memlock via sudo prlimit -> $(ulimit -l)" \
            || echo "[start_pd] WARN: sudo prlimit failed; RDMA may hit ibv_create_cq ENOMEM" >&2
    else
        echo "[start_pd] WARN: cannot raise memlock (hard cap $(ulimit -H -l) KB, no sudo); RDMA may fail" >&2
    fi
fi
echo "[start_pd] memlock now: $(ulimit -l)"

# 3. torch_memory_saver preload shim (sleep). cu13 variant.
ENABLE_TMS="${ENABLE_TMS:-1}"
if [[ "$ENABLE_TMS" != "1" ]]; then
    echo "[start_pd] ENABLE_TMS=0: skipping TMS preload"
    export RTP_LLM_FREEZE_WEIGHTS_SAVER=0
else
    SHIM="$("$PYTHON_BIN" -c "
from torch_memory_saver.utils import get_binary_path_from_package
print(get_binary_path_from_package('torch_memory_saver_hook_mode_preload'))" 2>/dev/null || true)"
    if [[ -z "$SHIM" || ! -f "$SHIM" ]]; then
        echo "[start_pd] torch_memory_saver preload shim not found; disabling TMS" >&2
        export RTP_LLM_FREEZE_WEIGHTS_SAVER=0
    else
        export LD_PRELOAD="$SHIM${LD_PRELOAD:+:$LD_PRELOAD}"
        export RTP_LLM_FREEZE_WEIGHTS_SAVER=1
        echo "[start_pd] TMS preload shim: $SHIM"
    fi
fi

# 4. Toolchain / libs
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/bin/triton_ptxas_cuda_wrapper.sh}"
export DG_JIT_CPP_STANDARD=20
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export NVCC_PREPEND_FLAGS="-ccbin=/opt/rh/gcc-toolset-12/root/usr/bin/g++"
export PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/opt/conda310/bin:${CUDA_HOME}/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:$BAZEL_BIN_DIR:$TORCH_LIB:${CUDA_HOME}/lib64:/opt/conda310/lib:${CUDA_HOME}/extras/CUPTI/lib64:/usr/local/nvidia/lib64:/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DETERMINISTIC_GEMM=1
export ENABLE_STABLE_SCATTER_ADD=ON

# 5. Common model / runtime config (shared by both roles)
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the DeepSeek-V4-Flash checkpoint}"
SLEEP_MODE_LEVEL="${SLEEP_MODE_LEVEL:-1}"
# MTP spec-decode toggle. L2 sleep is now COMPATIBLE with a checkpoint-backed
# draft model: the MTP model's own GPU weights are reloaded on wake via the
# chained WeightManager reload (ModelFactory.from_model_configs). Default ON so
# PD sleep/wake is exercised with the production-shape spec-decode config; MTP
# flags are appended to both roles when USE_MTP=1.
USE_MTP="${USE_MTP:-1}"
if [[ "$USE_MTP" == "1" ]]; then
    MTP_ARGS="--sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path $MODEL_DIR --sp_act_type bf16"
else
    MTP_ARGS=""
fi
DECODE_PORT="${DECODE_PORT:-21000}"
PREFILL_PORT="${PREFILL_PORT:-22000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
CACHE_STORE_RDMA_MODE="${CACHE_STORE_RDMA_MODE:-1}"
DECODE_RESERVER_RUNTIME_MEM_MB="${DECODE_RESERVER_RUNTIME_MEM_MB:-49152}"
PREFILL_RESERVER_RUNTIME_MEM_MB="${PREFILL_RESERVER_RUNTIME_MEM_MB:-65536}"
LOG_DIR="${LOG_DIR:-/tmp/diag_pd}"
mkdir -p "$LOG_DIR" "$LOG_DIR/disk_kv_prefill0" "$LOG_DIR/disk_kv_prefill1"

# Level 3 destroys and rebuilds NCCL/symmetric-memory resources. Keep the
# multicast fabric handles in CUDA-free role-local holders so NVLS and torch
# symmetric-memory multicast remain enabled across rank checkpoint/restore.
ENABLE_MULTICAST_KEEPER="${RTP_LLM_ENABLE_MULTICAST_KEEPER:-1}"
MULTICAST_KEEPER="$REPO_ROOT/rtp_llm/cpp/cuda_checkpoint/multicast_keeper/multicast_keeper.sh"
MULTICAST_KEEPER_BASE_DIR="${RTP_LLM_MULTICAST_KEEPER_BASE_DIR:-$LOG_DIR/multicast_keeper}"
DECODE_KEEPER_DIR="$MULTICAST_KEEPER_BASE_DIR/decode"
PREFILL_KEEPER_DIR="$MULTICAST_KEEPER_BASE_DIR/prefill"
DECODE_KEEPER_ENV=""
PREFILL_KEEPER_ENV=""
DECODE_PID=""
PREFILL_PID=""

cleanup() {
    trap - EXIT INT TERM
    for pid in "$DECODE_PID" "$PREFILL_PID"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    if [[ -n "$DECODE_KEEPER_ENV" ]]; then
        "$MULTICAST_KEEPER" stop --keeper-dir "$DECODE_KEEPER_DIR" >/dev/null 2>&1 || true
    fi
    if [[ -n "$PREFILL_KEEPER_ENV" ]]; then
        "$MULTICAST_KEEPER" stop --keeper-dir "$PREFILL_KEEPER_DIR" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ServiceRoute for prefill (routes to decode). Fields per host_service.py EndPoint/GroupEndPoint/ServiceRoute.
MODEL_SERVICE_CONFIG_JSON="$(cat <<JSON
{"service_id":"pd_local","use_local":true,"master_endpoint":null,"role_endpoints":[{"group":"default","vit_endpoint":null,"pd_fusion_endpoint":null,"prefill_endpoint":{"type":"Vipserver","address":"127.0.0.1:${PREFILL_PORT}","protocol":"http","path":"/"},"decode_endpoint":{"type":"Vipserver","address":"127.0.0.1:${DECODE_PORT}","protocol":"http","path":"/"}}]}
JSON
)"

common_env() {
    export MODEL_TYPE=deepseek_v4
    export USE_RPC_MODEL=1
    export LOAD_PYTHON_MODEL=1
    export ACT_TYPE=BF16
    export LOG_LEVEL=INFO
    export DSV4_USE_MEGA_MOE=1
    export ENABLE_SLEEP_MODE=1
    export SLEEP_MODE_LEVEL="$SLEEP_MODE_LEVEL"
    export TOKENIZER_PATH="$MODEL_DIR"
    export CHECKPOINT_PATH="$MODEL_DIR"
    export WARM_UP="${WARM_UP:-1}"
    export REMOTE_RPC_SERVER_IP=localhost
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NVLS}"
    export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-$LOG_DIR/nccl.%h.%p.log}"
}

DECODE_GPUS="${DECODE_GPUS:-4,5}"
PREFILL_GPUS="${PREFILL_GPUS:-6,7}"

if [[ "$SLEEP_MODE_LEVEL" == "3" && "$ENABLE_MULTICAST_KEEPER" == "1" ]]; then
    [[ -x "$MULTICAST_KEEPER" ]] || {
        echo "[start_pd] multicast keeper launcher is unavailable: $MULTICAST_KEEPER" >&2
        exit 1
    }
    "$MULTICAST_KEEPER" start --gpus "$DECODE_GPUS" --keeper-dir "$DECODE_KEEPER_DIR"
    DECODE_KEEPER_ENV="$DECODE_KEEPER_DIR/keeper.env"
    if [[ "${DECODE_ONLY:-0}" != "1" ]]; then
        "$MULTICAST_KEEPER" start --gpus "$PREFILL_GPUS" --keeper-dir "$PREFILL_KEEPER_DIR"
        PREFILL_KEEPER_ENV="$PREFILL_KEEPER_DIR/keeper.env"
    fi
    echo "[start_pd] multicast keeper enabled for Level 3 (NVLS=1, torch multicast=1)"
fi

echo "[start_pd] python=$PYTHON_BIN  decode_gpus=$DECODE_GPUS:$DECODE_PORT  prefill_gpus=$PREFILL_GPUS:$PREFILL_PORT  sleep_level=$SLEEP_MODE_LEVEL"
"$PYTHON_BIN" - <<'PY'
import torch
print(f"[start_pd] torch={torch.__version__} cuda={torch.version.cuda} ndev={torch.cuda.device_count()}")
PY

# ---- DECODE role (GPUs 4,5) ----
(
    common_env
    if [[ -n "$DECODE_KEEPER_ENV" ]]; then
        # shellcheck disable=SC1090
        source "$DECODE_KEEPER_ENV"
    fi
    export CUDA_VISIBLE_DEVICES="$DECODE_GPUS"
    export START_PORT="$DECODE_PORT"
    export REMOTE_SERVER_PORT="$PREFILL_PORT"
    export DSV4_USE_FRAMEWORK_KV=1
    # Async (stream-overlap) scheduling for the decode role. RTP_LLM_STREAM_ASYNC
    # is read by NormalExecutor/MtpExecutor::useStreamAsync() and only kicks in on
    # decode-only batches (NormalExecutor.cc:355 is_decode_only) — the next decode
    # iteration overlaps forward prep with the prior step's worker D2H/update. It is
    # never auto-exported, so MTP async decode requires setting it here explicitly.
    export RTP_LLM_STREAM_ASYNC="${RTP_LLM_STREAM_ASYNC:-1}"
    exec "$PYTHON_BIN" -m rtp_llm.start_server \
        --load_method fastsafetensors --max_seq_len "$MAX_SEQ_LEN" \
        --enable_cuda_graph 1 --act_type BF16 \
        --tp_size 1 --dp_size 2 --ep_size 2 --world_size 2 \
        --seq_size_per_block 256 --kernel_seq_size_per_block 128 \
        --role_type DECODE --cache_store_rdma_mode "$CACHE_STORE_RDMA_MODE" --use_local 1 \
        --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 \
        --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 120000 \
        --reserver_runtime_mem_mb "$DECODE_RESERVER_RUNTIME_MEM_MB" --fp8_kv_cache 1 \
        $MTP_ARGS \
        --cp_rotate_method PREFILL_CP
) > "$LOG_DIR/decode.log" 2>&1 &
DECODE_PID=$!
echo "[start_pd] decode launched pid=$DECODE_PID -> $LOG_DIR/decode.log"

# ---- PREFILL role (GPUs 6,7) ----
if [[ "${DECODE_ONLY:-0}" == "1" ]]; then
    echo "[start_pd] DECODE_ONLY=1: skipping prefill launch (decode uses existing peer at :$PREFILL_PORT)"
    PREFILL_PID=""
else
(
    common_env
    if [[ -n "$PREFILL_KEEPER_ENV" ]]; then
        # shellcheck disable=SC1090
        source "$PREFILL_KEEPER_ENV"
    fi
    export CUDA_VISIBLE_DEVICES="$PREFILL_GPUS"
    export START_PORT="$PREFILL_PORT"
    export REMOTE_SERVER_PORT="$DECODE_PORT"
    export MODEL_SERVICE_CONFIG="$MODEL_SERVICE_CONFIG_JSON"
    export DSV4_USE_FRAMEWORK_KV=1
    # Sleep-time free + wake-time rebuild of the Mega MoE symm-mem buffer
    # (~4.4 GiB/rank): destroy on sleep, collective re-rendezvous on the first
    # post-wake forward. Without this the symm buffer stays resident and
    # inflates the sleeping residual. See sleep_gpu_reclaim.py.
    export RTP_LLM_SLEEP_FREE_MEGA_SYMM="${RTP_LLM_SLEEP_FREE_MEGA_SYMM:-1}"
    export ENABLE_MEMORY_CACHE_DISK=1
    export MEMORY_CACHE_DISK_PATHS="$LOG_DIR/disk_kv_prefill0,$LOG_DIR/disk_kv_prefill1"
    export MEMORY_CACHE_DISK_SIZE_MB=8192
    export MEMORY_CACHE_DISK_BUFFERED_IO=1
    export MEMORY_CACHE_DISK_SYNC_TIMEOUT_MS=120000
    exec "$PYTHON_BIN" -m rtp_llm.start_server \
        --load_method fastsafetensors --max_seq_len "$MAX_SEQ_LEN" \
        --enable_cuda_graph 0 --act_type BF16 \
        --tp_size 2 --ep_size 2 --world_size 2 \
        --seq_size_per_block 256 --kernel_seq_size_per_block 128 \
        --role_type PREFILL --cache_store_rdma_mode "$CACHE_STORE_RDMA_MODE" --use_local 1 \
        --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 \
        --memory_cache_size_mb 256 --write_cache_sync 1 \
        --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER \
        --reserver_runtime_mem_mb "$PREFILL_RESERVER_RUNTIME_MEM_MB" --max_context_batch_size 1 --fp8_kv_cache 1 \
        $MTP_ARGS
) > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!
echo "[start_pd] prefill launched pid=$PREFILL_PID -> $LOG_DIR/prefill.log"
fi

echo "[start_pd] decode_pid=$DECODE_PID prefill_pid=$PREFILL_PID"
echo "[start_pd] waiting on both roles (Ctrl-C to stop)"
wait
