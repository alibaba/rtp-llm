#!/usr/bin/env bash
#
# Start one side of a Kimi K3 PD topology. Prefill remains TP8 / DP1 / EP8.
# Decode defaults to the validated TP8 / DP1 / EP8 baseline and can opt into
# TP1 / DP8 / EP8 / KTP8 or TP1 / DP16 / EP16 / KTP16 topology.
#
# The script incrementally builds its Bazel launcher with CUDA13/SM10x.  It
# does not install or replace a system rtp-llm wheel.
#
# Example (run the matching command on each host):
#   CHECKPOINT_PATH=/local/path/to/Kimi-K3 \
#   PREFILL_ENDPOINT=${PREFILL_HOST}:27188 \
#   DECODE_ENDPOINT=${DECODE_HOST}:28188 \
#   ./example/k3/start_kimi_k3_pd.sh prefill
#
#   CHECKPOINT_PATH=/local/path/to/Kimi-K3 \
#   PREFILL_ENDPOINT=${PREFILL_HOST}:27188 \
#   DECODE_ENDPOINT=${DECODE_HOST}:28188 \
#   ./example/k3/start_kimi_k3_pd.sh decode
#
# The process stays in the foreground. Set RTP_LLM_DRY_RUN=1 to inspect the
# resolved configuration without starting the model.

set -euo pipefail
ulimit -c 0

# `docker exec -u` may preserve root's HOME even though the effective user is
# not root. Bazelisk and runtime JITs need writable per-user caches.
if [[ -z "${HOME:-}" || ! -d "${HOME}" || ! -w "${HOME}" ]]; then
    resolved_home="$(
        getent passwd "$(id -u)" 2>/dev/null | awk -F: '{print $6}'
    )"
    [[ -n "${resolved_home}" && -d "${resolved_home}" && -w "${resolved_home}" ]] \
        || {
            echo "error: HOME is not writable and the account home could not be resolved" >&2
            exit 2
        }
    export HOME="${resolved_home}"
fi

usage() {
    cat >&2 <<'EOF'
Usage:
  CHECKPOINT_PATH=/path/to/Kimi-K3 \
  PREFILL_ENDPOINT=host:port \
  DECODE_ENDPOINT=host:port \
  example/k3/start_kimi_k3_pd.sh prefill|decode

Required on both hosts:
  CHECKPOINT_PATH                        local-data-disk checkpoint
  PREFILL_ENDPOINT                       externally reachable host:port
  DECODE_ENDPOINT                        externally reachable host:port

Model and cache (normally change these together on both roles):
  TOKENIZER_PATH                         defaults to CHECKPOINT_PATH
  MAX_SEQ_LEN                            defaults to 16384
  MAX_BATCH_TOKENS_SIZE                  optional token admission limit
  KV_CACHE_MEM_MB                        defaults: Prefill 43000, Decode 46000;
                                         BF16 only
  SEQ_SIZE_PER_BLOCK                     defaults to 4096
  KERNEL_SEQ_SIZE_PER_BLOCK              defaults to 128
  REUSE_CACHE                            defaults to 0
  LINEAR_STEP                            defaults to 1
  CONCURRENCY_LIMIT                      defaults to 2
  MAX_CONTEXT_BATCH_SIZE                 defaults to 1
  CACHE_STORE_RDMA_MODE                  defaults to 0; set to 1 on both
                                         Prefill and Decode for Barex RDMA
  CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS    defaults to 2000; bounded wait for
                                         the asynchronous RDMA QP pool

Role-specific high-performance paths:
  KIMI_K3_PREFILL_CHUNK_TOKENS           Prefill only; defaults to 65536
  KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD     Prefill only; defaults to 1
  MEGA_MOE_MAX_TOKENS_PER_RANK           defaults: Prefill 8192, Decode 1;
                                         raise Decode for concurrent batches
  RTP_MLA_DECODE_KERNEL                  Decode is fixed to tokenspeed_mla
  ENABLE_CUDA_GRAPH                      fixed to 0 for Prefill; defaults to 1
                                         for Decode
  DECODE_CAPTURE_CONFIG                  Decode only; defaults to 1
  ENABLE_CUDA_GRAPH_DEBUG_MODE           defaults to 0
  KIMI_K3_DECODE_TOPOLOGY                Decode only; tp8_ep8 (default),
                                         dp8_ep8_tp1_ktp8, or
                                         dp16_ep16_tp1_ktp16

Runtime, build and diagnostics:
  RUN_ROOT                              defaults below TMPDIR
  RTP_LLM_TMPDIR                        defaults to a short role-specific
                                         path below /tmp
  FLASHINFER_WORKSPACE_BASE             defaults below RUN_ROOT;
                                         isolates JIT artifacts from stale
                                         Bazel output-base absolute paths
  FLASHINFER_CUDA_ARCH_LIST             defaults to 10.3a for the validated
                                         B300/SM103a K3 deployment; required
                                         because the virtual GPU product can
                                         otherwise inject or detect SM89
  BAZEL_OUTPUT_BASE                     optional existing Bazel output base;
                                         useful on inode-constrained hosts
  RTP_LLM_SERVER_BINARY                 optional prebuilt Bazel launcher path
  RTP_LLM_SKIP_BUILD=1                  use RTP_LLM_SERVER_BINARY without
                                         running a nested Bazel build
  DEEPGEMM_JIT_COMPILER                 auto|nvcc|nvrtc; auto repairs a stale
                                         NVCC -ccbin path when necessary
  RTP_LLM_SERVICE_ID                    defaults to kimi-k3-pd
  OPS_OVERLAY                           optional prebuilt operator overlay
  RTP_LLM_DRY_RUN=1                     print configuration and exit

The operator implementations and versions are Bazel/runtime dependencies, not
launcher knobs. cuLA KDA, FlashMLA Prefill, fused AG-GEMM and fused router are
selected by K3 source code. Legacy KIMI_K3_* backend toggles are not used.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

[[ $# -eq 1 ]] || {
    usage
    exit 2
}

role="${1^^}"
case "${role}" in
    PREFILL | DECODE) ;;
    *)
        usage
        die "role must be prefill or decode"
        ;;
esac

case "${KIMI_K3_MLA_CACHE_TP:-0}" in
    0) unset KIMI_K3_MLA_CACHE_TP ;;
    1) die "KIMI_K3_MLA_CACHE_TP=1 selects the retired 576/TP ABI; MLA is replicated at width 576" ;;
    *) die "KIMI_K3_MLA_CACHE_TP must be unset or 0" ;;
esac

if [[ "${role}" == "PREFILL" ]]; then
    export KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD="${KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD:-1}"
    [[ "${KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD}" == "0" \
        || "${KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD}" == "1" ]] \
        || die "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD must be 0 or 1"
else
    unset KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD
fi

: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
# The generic same-host PD Smoke runner assigns START_PORT and
# REMOTE_SERVER_PORT dynamically.  Derive the two explicit endpoints when the
# launcher is invoked by that runner, while keeping explicit host:port values
# mandatory for normal cross-host use.
if [[ -z "${PREFILL_ENDPOINT:-}" \
    && -z "${DECODE_ENDPOINT:-}" \
    && -n "${START_PORT:-}" \
    && -n "${REMOTE_SERVER_PORT:-}" ]]; then
    if [[ "${role}" == "PREFILL" ]]; then
        PREFILL_ENDPOINT="127.0.0.1:${START_PORT}"
        DECODE_ENDPOINT="127.0.0.1:${REMOTE_SERVER_PORT}"
    else
        PREFILL_ENDPOINT="127.0.0.1:${REMOTE_SERVER_PORT}"
        DECODE_ENDPOINT="127.0.0.1:${START_PORT}"
    fi
    export PREFILL_ENDPOINT DECODE_ENDPOINT
fi
: "${PREFILL_ENDPOINT:?PREFILL_ENDPOINT is required}"
: "${DECODE_ENDPOINT:?DECODE_ENDPOINT is required}"

endpoint_port() {
    local endpoint="$1"
    [[ "${endpoint}" =~ ^[^:]+:[0-9]+$ ]] \
        || die "endpoint must have host:port form, got ${endpoint}"
    printf '%s\n' "${endpoint##*:}"
}

prefill_port="$(endpoint_port "${PREFILL_ENDPOINT}")"
decode_port="$(endpoint_port "${DECODE_ENDPOINT}")"
prefill_host="${PREFILL_ENDPOINT%:*}"
decode_host="${DECODE_ENDPOINT%:*}"
decode_topology="${KIMI_K3_DECODE_TOPOLOGY:-tp8_ep8}"
case "${decode_topology}" in
    tp8_ep8 | dp8_ep8_tp1_ktp8 | dp16_ep16_tp1_ktp16) ;;
    *)
        die "unsupported KIMI_K3_DECODE_TOPOLOGY=${decode_topology}"
        ;;
esac
cache_store_rdma_mode="${CACHE_STORE_RDMA_MODE:-0}"
[[ "${cache_store_rdma_mode}" == "0" || "${cache_store_rdma_mode}" == "1" ]] \
    || die "CACHE_STORE_RDMA_MODE must be 0 or 1"
cache_store_rdma_connect_timeout_ms="${CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS:-2000}"
[[ "${cache_store_rdma_connect_timeout_ms}" =~ ^[1-9][0-9]*$ ]] \
    || die "CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS must be a positive integer"
# gRPC honors the standard HTTP proxy environment.  The shared CUDA image
# enables a localhost proxy by default, so same-host P/D traffic would
# otherwise be redirected through that proxy instead of reaching the peer
# ModelRpc server.  Preserve the caller's exclusions and add both PD hosts.
pd_no_proxy_hosts="${prefill_host},${decode_host}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${pd_no_proxy_hosts}"
export no_proxy="${no_proxy:+${no_proxy},}${pd_no_proxy_hosts}"

[[ -f "${CHECKPOINT_PATH}/config.json" ]] \
    || die "missing ${CHECKPOINT_PATH}/config.json"
[[ -f "${CHECKPOINT_PATH}/model.safetensors.index.json" ]] \
    || die "missing ${CHECKPOINT_PATH}/model.safetensors.index.json"

tokenizer_path="${TOKENIZER_PATH:-${CHECKPOINT_PATH}}"
[[ -d "${tokenizer_path}" ]] || die "TOKENIZER_PATH is not a directory"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
server_target="//rtp_llm:rtp_llm_server"
server_binary="${RTP_LLM_SERVER_BINARY:-${repo_root}/bazel-bin/rtp_llm/rtp_llm_server}"
skip_build="${RTP_LLM_SKIP_BUILD:-0}"
[[ "${skip_build}" == "0" || "${skip_build}" == "1" ]] \
    || die "RTP_LLM_SKIP_BUILD must be 0 or 1, got ${skip_build}"

service_id="${RTP_LLM_SERVICE_ID:-kimi-k3-pd}"
run_root="${RUN_ROOT:-${SMOKE_ROLE_RUNTIME_DIR:-${TMPDIR:-/tmp}/${service_id}}}"
# CpuTpBroadcaster appends a long per-rank UDS name.  Keep the default runtime
# path short even when RUN_ROOT is an archival path.
runtime_tmpdir="${RTP_LLM_TMPDIR:-/tmp/${service_id}-${role,,}}"
# Export the short/writable runtime directory before any runtime JIT starts.
mkdir -p "${run_root}" "${runtime_tmpdir}"
export TMPDIR="${runtime_tmpdir}"
# FlashInfer's generated build.ninja embeds absolute include paths from the
# active Bazel runfiles tree.  The account-wide default cache can therefore
# become invalid after switching worktrees/output bases.  Keep the JIT cache
# inside this controlled run unless the caller explicitly supplies another
# validated location.
flashinfer_workspace_base="${FLASHINFER_WORKSPACE_BASE:-${run_root}/flashinfer-workspace}"
export FLASHINFER_WORKSPACE_BASE="${flashinfer_workspace_base}"
# A virtualized environment can report a generic architecture even though K3
# runs on the B300/SM103a device, and its base environment can already export
# the wrong value. FlashInfer uses this value to choose the JIT directory and
# compile flags. Force the K3-specific default instead of preserving an
# inherited value unless the caller explicitly provides the generic setting.
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.3a}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

# ---------------------------------------------------------------------------
# Model/cache settings. Cache precision is fixed to BF16 in server_args.
# ---------------------------------------------------------------------------

max_seq_len="${MAX_SEQ_LEN:-16384}"
# Keep frontend ModelArgs and backend --max_seq_len on the same limit.
export MAX_SEQ_LEN="${max_seq_len}"
max_batch_tokens_size="${MAX_BATCH_TOKENS_SIZE:-}"
seq_size_per_block="${SEQ_SIZE_PER_BLOCK:-4096}"
kernel_seq_size_per_block="${KERNEL_SEQ_SIZE_PER_BLOCK:-128}"
concurrency_limit="${CONCURRENCY_LIMIT:-2}"
max_context_batch_size="${MAX_CONTEXT_BATCH_SIZE:-1}"
reuse_cache="${REUSE_CACHE:-0}"
linear_step="${LINEAR_STEP:-1}"
kimi_k3_kda_pool_blocks="${KIMI_K3_KDA_POOL_BLOCKS:-0}"
if [[ "${role}" == "PREFILL" ]]; then
    default_kv_cache_mem_mb=43000
else
    default_kv_cache_mem_mb=46000
fi

kv_cache_mem_mb="${KV_CACHE_MEM_MB:-${default_kv_cache_mem_mb}}"
enable_cuda_graph_debug_mode="${ENABLE_CUDA_GRAPH_DEBUG_MODE:-0}"

# ---------------------------------------------------------------------------
# Role-specific performance settings.
#
# Prefill: cuLA KDA, FlashMLA, fused router and long-sequence fused AG-GEMM
# are selected by code. Shared-expert TP sharding is Prefill-only.
# Decode: full shared-expert weights, TokenSpeed MLA and CUDA Graph batch 1.
# ---------------------------------------------------------------------------
if [[ "${role}" == "PREFILL" ]]; then
    [[ "${ENABLE_CUDA_GRAPH:-0}" == "0" ]] \
        || die "Prefill CUDA Graph is unsupported; set ENABLE_CUDA_GRAPH=0"
    enable_cuda_graph=0
    decode_capture_config=
    prefill_capture_config="${PREFILL_CAPTURE_CONFIG:-}"
    export KIMI_K3_PREFILL_CHUNK_TOKENS="${KIMI_K3_PREFILL_CHUNK_TOKENS:-65536}"
    unset RTP_MLA_DECODE_KERNEL
    default_mega_moe_tokens=8192
else
    enable_cuda_graph="${ENABLE_CUDA_GRAPH:-1}"
    if [[ "${enable_cuda_graph}" == "1" ]]; then
        decode_capture_config="${DECODE_CAPTURE_CONFIG:-1}"
    else
        # Decode KTP phase one forbids CUDA Graph.  Do not leak the generic
        # batch-1 capture default into an explicitly graph-disabled launch.
        decode_capture_config=
    fi
    prefill_capture_config=
    export RTP_MLA_DECODE_KERNEL=tokenspeed_mla
    unset KIMI_K3_PREFILL_CHUNK_TOKENS
    default_mega_moe_tokens=1
fi

# ---------------------------------------------------------------------------
# Operator runtime. Exact cuLA/FlashMLA/DeepGEMM/TokenSpeed versions belong
# to Bazel dependency declarations and are shipped in the server runfiles.
# OPS_OVERLAY is an explicit debugging override; normal launches do not pip
# install or shadow the Bazel dependency set.
# ---------------------------------------------------------------------------
operator_overlay="${OPS_OVERLAY:-}"
if [[ -n "${operator_overlay}" ]]; then
    [[ -d "${operator_overlay}" ]] \
        || die "OPS_OVERLAY is not a directory: ${operator_overlay}"
    export PYTHONPATH="${operator_overlay}${PYTHONPATH:+:${PYTHONPATH}}"
fi

for flag_name in \
    reuse_cache \
    enable_cuda_graph \
    enable_cuda_graph_debug_mode; do
    flag_value="${!flag_name}"
    [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] \
        || die "${flag_name} must resolve to 0 or 1, got ${flag_value}"
done
[[ "${linear_step}" =~ ^[1-9][0-9]*$ ]] \
    || die "LINEAR_STEP must resolve to a positive integer, got ${linear_step}"
[[ "${kimi_k3_kda_pool_blocks}" =~ ^[0-9]+$ ]] \
    || die "KIMI_K3_KDA_POOL_BLOCKS must resolve to a non-negative integer, got ${kimi_k3_kda_pool_blocks}"
if [[ "${enable_cuda_graph_debug_mode}" == "1" && "${enable_cuda_graph}" != "1" ]]; then
    die "ENABLE_CUDA_GRAPH_DEBUG_MODE=1 requires ENABLE_CUDA_GRAPH=1"
fi
for integer_name in \
    max_seq_len \
    kv_cache_mem_mb \
    seq_size_per_block \
    kernel_seq_size_per_block \
    concurrency_limit \
    max_context_batch_size; do
    integer_value="${!integer_name}"
    [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]] \
        || die "${integer_name} must resolve to a positive integer, got ${integer_value}"
done



if [[ "${role}" == "PREFILL" ]]; then
    local_endpoint="${PREFILL_ENDPOINT}"
    remote_endpoint="${DECODE_ENDPOINT}"
    start_port="${prefill_port}"
    remote_port="${decode_port}"
    tp_size=8
    dp_size=1
    ep_size=8
    ktp_size=1
    world_size=8
    local_world_size=8
    export KIMI_K3_DECODE_KTP=0
else
    local_endpoint="${DECODE_ENDPOINT}"
    remote_endpoint="${PREFILL_ENDPOINT}"
    start_port="${decode_port}"
    remote_port="${prefill_port}"
    case "${decode_topology}" in
        tp8_ep8)
            tp_size=8
            dp_size=1
            ep_size=8
            ktp_size=1
            world_size=8
            local_world_size=8
            export KIMI_K3_DECODE_KTP=0
            ;;
        dp8_ep8_tp1_ktp8)
            tp_size=1
            dp_size=8
            ep_size=8
            ktp_size=8
            world_size=8
            local_world_size=8
            export KIMI_K3_DECODE_KTP=1
            ;;
        dp16_ep16_tp1_ktp16)
            tp_size=1
            dp_size=16
            ep_size=16
            ktp_size=16
            world_size=16
            local_world_size="${LOCAL_WORLD_SIZE:-8}"
            [[ "${local_world_size}" == "8" ]] || die "DP16 Decode requires LOCAL_WORLD_SIZE=8"
            world_rank="${WORLD_RANK:-}"
            [[ "${world_rank}" == "0" || "${world_rank}" == "8" ]] \
                || die "DP16 Decode requires WORLD_RANK=0 on node 0 or WORLD_RANK=8 on node 1"
            [[ -n "${GANG_CONFIG_STRING:-}" ]] \
                || die "DP16 Decode requires GANG_CONFIG_STRING for the two Decode nodes"
            export KIMI_K3_DECODE_KTP=1
            ;;
    esac
fi

world_rank="${world_rank:-${WORLD_RANK:-0}}"

model_service_config="$(
    printf '{"service_id":"%s","role_endpoints":[{"group":"default",' \
        "${service_id}"
    printf '"prefill_endpoint":{"type":"PREFILL","address":"%s","protocol":"HTTP","path":""},' \
        "${PREFILL_ENDPOINT}"
    printf '"decode_endpoint":{"type":"DECODE","address":"%s","protocol":"HTTP","path":""}}],"use_local":true}' \
        "${DECODE_ENDPOINT}"
)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TMPDIR="${runtime_tmpdir}"
export RTP_LLM_STARTUP_TIMEOUT_S="${RTP_LLM_STARTUP_TIMEOUT_S:-14400}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export LOG_PATH="${LOG_PATH:-${run_root}/logs/${role,,}}"
export START_PORT="${start_port}"
export FRONTEND_SERVER_COUNT="${FRONTEND_SERVER_COUNT:-1}"
export MODEL_TYPE=kimi_k3
export CHECKPOINT_PATH
export TOKENIZER_PATH="${tokenizer_path}"
export LOAD_METHOD="${LOAD_METHOD:-fastsafetensors}"
[[ "${LOAD_METHOD}" == "fastsafetensors" ]] \
    || die "Kimi K3 PD requires LOAD_METHOD=fastsafetensors"
export ENABLE_CUDA_GRAPH="${enable_cuda_graph}"
export ENABLE_CUDA_GRAPH_DEBUG_MODE="${enable_cuda_graph_debug_mode}"
export REMOTE_RPC_SERVER_IP="${remote_endpoint}"
export MODEL_SERVICE_CONFIG="${model_service_config}"
# MegaMoE 自己做 dispatch/combine,框架侧的 DeepEP MoE 恒不启用。
use_deepep_moe=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
# Bazel's CUDA toolchain may export absolute compiler paths that are valid
# only inside an action sandbox.  FlashInfer reads CC/CXX directly when it
# materializes an MLA kernel at runtime, so normalize stale paths before any
# JIT starts.
if [[ ! -x "${CC:-}" ]]; then
    CC="$(command -v gcc || true)"
    [[ -x "${CC}" ]] || die "FlashInfer JIT requires a host C compiler"
    export CC
fi
if [[ ! -x "${CXX:-}" ]]; then
    CXX="$(command -v g++ || true)"
    [[ -x "${CXX}" ]] || die "FlashInfer/DeepGEMM JIT requires a host C++ compiler"
    export CXX
fi
if [[ ! -x "${CUDAHOSTCXX:-}" ]]; then
    CUDAHOSTCXX="${CXX}"
    export CUDAHOSTCXX
fi
deepgemm_jit_compiler="${DEEPGEMM_JIT_COMPILER:-auto}"
case "${deepgemm_jit_compiler}" in
        auto)
            deepgemm_host_cxx=
            if [[ "${NVCC_PREPEND_FLAGS:-}" =~ -ccbin=([^[:space:]]+) ]]; then
                deepgemm_host_cxx="${BASH_REMATCH[1]}"
            fi
            if [[ ! -x "${deepgemm_host_cxx}" ]] \
                && [[ -x "${CXX:-}" ]]; then
                deepgemm_host_cxx="${CXX}"
            fi
            if [[ ! -x "${deepgemm_host_cxx}" ]] \
                && [[ -x /opt/rh/gcc-toolset-12/root/usr/bin/g++ ]]; then
                deepgemm_host_cxx=/opt/rh/gcc-toolset-12/root/usr/bin/g++
            fi
            if [[ ! -x "${deepgemm_host_cxx}" ]]; then
                deepgemm_host_cxx="$(command -v g++ || true)"
            fi
            [[ -x "${deepgemm_host_cxx}" ]] \
                || die "DeepGEMM NVCC JIT requires a host C++ compiler"
            export NVCC_PREPEND_FLAGS="-ccbin=${deepgemm_host_cxx}"
            deepgemm_jit_compiler=nvcc
            ;;
        nvcc | nvrtc) ;;
        *)
            die "DEEPGEMM_JIT_COMPILER must be auto, nvcc or nvrtc"
            ;;
esac
if [[ "${deepgemm_jit_compiler}" == "nvrtc" ]]; then
    export DG_JIT_USE_NVRTC=1
else
    export DG_JIT_USE_NVRTC=0
fi
export OPS_OVERLAY="${operator_overlay}"
export MEGA_MOE_MAX_TOKENS_PER_RANK="${MEGA_MOE_MAX_TOKENS_PER_RANK:-${default_mega_moe_tokens}}"
export DSV4_MEGA_MOE_INPUT_PACKER=fused
export DSV4_MEGA_MOE_INPUT_PACKER_IMPL=optimized
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-${run_root}/deep-gemm-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${run_root}/triton-cache}"


mkdir -p \
    "${LOG_PATH}" \
    "${run_root}/work/${role,,}" \
    "${TMPDIR}" \
    "${DG_JIT_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}"

echo "Kimi K3 ${role} configuration:"
echo "  local endpoint:  ${local_endpoint}"
echo "  remote endpoint: ${remote_endpoint}"
echo "  PD no-proxy:      ${pd_no_proxy_hosts}"
echo "  checkpoint:      ${CHECKPOINT_PATH}"
echo "  topology:        TP${tp_size}/DP${dp_size}/EP${ep_size}/KTP${ktp_size} world=${world_size}/local=${local_world_size}"
echo "  Prefill MLA:     width=576 replicated on every TP rank"
if [[ "${role}" == "DECODE" ]]; then
    echo "  decode topology: ${decode_topology}"
    echo "  decode MLA:      ${RTP_MLA_DECODE_KERNEL}"
fi
echo "  load method:     ${LOAD_METHOD}"
echo "  DeepGEMM JIT:    ${deepgemm_jit_compiler}"
echo "  concurrency:     generate=${concurrency_limit}, context=${max_context_batch_size}"
if [[ -n "${max_batch_tokens_size}" ]]; then
    echo "  batch tokens:    ${max_batch_tokens_size}"
fi
echo "  reuse cache:     ${reuse_cache}"
if [[ "${role}" == "PREFILL" ]]; then
    echo "  shared weights:  shard=${KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD}"
    echo "  Prefill chunks:  ${KIMI_K3_PREFILL_CHUNK_TOKENS}"
else
    echo "  shared weights:  full (no weight AllGather)"
fi
echo "  MegaMoE packer:  ${DSV4_MEGA_MOE_INPUT_PACKER}/${DSV4_MEGA_MOE_INPUT_PACKER_IMPL}"
echo "  MegaMoE tokens:  ${MEGA_MOE_MAX_TOKENS_PER_RANK}/rank"
echo "  cache blocks:    seq=${seq_size_per_block}, kernel=${kernel_seq_size_per_block}"
echo "  cache precision: bf16 (int8=0, fp8=0), linear_step=${linear_step}, kda_pool_blocks=${kimi_k3_kda_pool_blocks}"
echo "  CUDA Graph:      enabled=${enable_cuda_graph}, debug=${enable_cuda_graph_debug_mode}"
if [[ -n "${decode_capture_config}" ]]; then
    echo "  Decode captures: ${decode_capture_config}"
fi
if [[ -n "${prefill_capture_config}" ]]; then
    echo "  Prefill captures:${prefill_capture_config}"
fi
echo "  FlashInfer JIT:  ${FLASHINFER_WORKSPACE_BASE} (arch=${FLASHINFER_CUDA_ARCH_LIST})"
echo "  runtime tmp:     ${TMPDIR}"
echo "  logs:            ${LOG_PATH}"

server_args=(
    --role_type "${role}"
    --tp_size "${tp_size}"
    --ktp_size "${ktp_size}"
    --dp_size "${dp_size}"
    --ep_size "${ep_size}"
    --world_size "${world_size}"
    --local_world_size "${local_world_size}"
    --world_rank "${world_rank}"
    --remote_server_port "${remote_port}"
    --max_seq_len "${max_seq_len}"
    --max_context_batch_size "${max_context_batch_size}"
    --seq_size_per_block "${seq_size_per_block}"
    --kernel_seq_size_per_block "${kernel_seq_size_per_block}"
    --kv_cache_mem_mb "${kv_cache_mem_mb}"
    --int8_kv_cache 0
    --fp8_kv_cache 0
    --linear_step "${linear_step}"
    --kimi_k3_kda_pool_blocks "${kimi_k3_kda_pool_blocks}"
    --ssm_state_dtype fp32
    --warm_up 0
    --reuse_cache "${reuse_cache}"
    --enable_device_cache 1
    --concurrency_limit "${concurrency_limit}"
    --use_deepep_moe "${use_deepep_moe}"
    --use_deepep_internode 0
    --use_deepep_low_latency 0
    --deep_ep_num_sm 24
    --use_all_gather 0
    --enable_cuda_graph "${enable_cuda_graph}"
    --enable_cuda_graph_debug_mode "${enable_cuda_graph_debug_mode}"
    --cache_store_rdma_mode "${cache_store_rdma_mode}"
    --cache_store_rdma_connect_timeout_ms "${cache_store_rdma_connect_timeout_ms}"
    --load_cache_timeout_ms 7200000
    --load_method "${LOAD_METHOD}"
    --ft_core_dump_on_exception "${FT_CORE_DUMP_ON_EXCEPTION:-0}"
    --shutdown_timeout 5
)

if [[ -n "${GANG_CONFIG_STRING:-}" ]]; then
    server_args+=(--gang_config_string "${GANG_CONFIG_STRING}")
fi

if [[ -n "${max_batch_tokens_size}" ]]; then
    [[ "${max_batch_tokens_size}" =~ ^[1-9][0-9]*$ ]] \
        || die "MAX_BATCH_TOKENS_SIZE must be a positive integer"
    server_args+=(--max_batch_tokens_size "${max_batch_tokens_size}")
fi

if [[ -n "${decode_capture_config}" ]]; then
    server_args+=(--decode_capture_config "${decode_capture_config}")
fi
if [[ -n "${prefill_capture_config}" ]]; then
    server_args+=(--prefill_capture_config "${prefill_capture_config}")
fi

if [[ "${RTP_LLM_DRY_RUN:-0}" == "1" ]]; then
    printf 'command:'
    printf ' %q' "${server_binary}" "${server_args[@]}"
    printf '\n'
    exit 0
fi

if [[ "${skip_build}" == "0" ]]; then
    bazel_startup_args=()
    if [[ -n "${BAZEL_OUTPUT_BASE:-}" ]]; then
        mkdir -p "${BAZEL_OUTPUT_BASE}"
        bazel_startup_args+=("--output_base=${BAZEL_OUTPUT_BASE}")
    fi
    (
        cd "${repo_root}"
        bazelisk "${bazel_startup_args[@]}" \
            build --config=cuda13 --config=sm10x "${server_target}"
    ) || die "failed to build ${server_target}"
fi
[[ -x "${server_binary}" ]] || die "missing Bazel launcher ${server_binary}"

# Keep runtime-only compiler headers out of Bazel action environments.
if [[ -n "${RTP_LLM_RUNTIME_CPATH:-}" ]]; then
    export CPATH="${RTP_LLM_RUNTIME_CPATH}"
fi
cd "${run_root}/work/${role,,}"
exec "${server_binary}" "${server_args[@]}"
