#!/usr/bin/env bash
#
# Start one side of the validated Kimi K3 PD topology:
#   Prefill: TP8 / DP1 / EP8
#   Decode:  TP8 / DP1 / EP8 (default), or TP1 / DP8 / EP8 (legacy)
#
# The script incrementally builds its Bazel launcher with CUDA13/SM10x.  It
# does not install or replace a system rtp-llm wheel.
#
# Example (run the matching command on each host):
#   CHECKPOINT_PATH=/models/Kimi-K3 \
#   PREFILL_ENDPOINT=10.0.0.1:27188 \
#   DECODE_ENDPOINT=10.0.0.2:28188 \
#   ./example/start_kimi_k3_pd.sh prefill
#
#   CHECKPOINT_PATH=/models/Kimi-K3 \
#   PREFILL_ENDPOINT=10.0.0.1:27188 \
#   DECODE_ENDPOINT=10.0.0.2:28188 \
#   ./example/start_kimi_k3_pd.sh decode
#
# The process stays in the foreground. Set KIMI_K3_DRY_RUN=1 to inspect the
# resolved configuration without starting the model.

set -euo pipefail
ulimit -c 0

# `docker exec -u` may preserve root's HOME even though the effective user is
# not root. Bazelisk and pip need a writable home for their per-user caches.
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
  start_kimi_k3_pd.sh prefill|decode

Optional environment variables:
  TOKENIZER_PATH                         defaults to CHECKPOINT_PATH
  RTP_LLM_PYTHON                         defaults to python3 in PATH
  KIMI_K3_RUN_ROOT                      defaults below TMPDIR
  KIMI_K3_TMPDIR                        defaults to a short role-specific
                                         path below /tmp
  KIMI_K3_FLASHINFER_WORKSPACE_BASE     defaults below KIMI_K3_RUN_ROOT;
                                         isolates JIT artifacts from stale
                                         Bazel output-base absolute paths
  KIMI_K3_FLASHINFER_CUDA_ARCH_LIST     defaults to 10.3a for the validated
                                         B300/SM103a K3 deployment; required
                                         because the virtual GPU product can
                                         otherwise inject or detect SM89
  KIMI_K3_BAZEL_OUTPUT_BASE             optional existing Bazel output base;
                                         useful on inode-constrained hosts
  KIMI_K3_SERVER_BINARY                 optional prebuilt Bazel launcher path
  KIMI_K3_SKIP_BUILD=1                  use KIMI_K3_SERVER_BINARY without
                                         running a nested Bazel build
  KIMI_K3_DEEPGEMM_JIT_COMPILER        auto|nvcc|nvrtc; auto repairs a stale
                                         NVCC -ccbin path when necessary
  KIMI_K3_SERVICE_ID                    defaults to kimi-k3-pd
  KIMI_K3_MAX_SEQ_LEN                   defaults to 16384
  KIMI_K3_KV_CACHE_MEM_MB               defaults to 8192
  SEQ_SIZE_PER_BLOCK                    defaults to 4096
  KERNEL_SEQ_SIZE_PER_BLOCK             defaults to 128
  CONCURRENCY_LIMIT                     defaults to 2; set this to the GPQA
                                         GENERATION_WORKERS value
  MAX_CONTEXT_BATCH_SIZE                defaults to 1
  KIMI_K3_REUSE_CACHE                   defaults to 0; set both PD roles to 1
                                         to validate prefix-cache reuse
  KIMI_K3_DECODE_TOPOLOGY               defaults to tp8_ep8; one of:
                                         tp8_ep8, dp8_ep8
  OPS_OVERLAY                           optional KDA/DeepGEMM operator overlay;
                                         optimized Prefill otherwise installs
                                         the bundled fixed wheels automatically
  KIMI_K3_MEGA_MAX_TOKENS_PER_RANK     DeepGEMM MegaMoE buffer capacity per
                                         rank; defaults to 8192
  ENABLE_CUDA_GRAPH                     defaults to 0; set Decode to 1 for
                                         CUDA Graph capture/replay validation
  ENABLE_CUDA_GRAPH_DEBUG_MODE          defaults to 0; emits CUDA Graph DOT
                                         diagnostics when graph is enabled
  DECODE_CAPTURE_CONFIG                 optional Decode batch sizes, for
                                         example 1,2
  PREFILL_CAPTURE_CONFIG                optional Prefill sequence lengths
  KIMI_K3_DRY_RUN=1                     print configuration and exit
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
    tp8_ep8 | dp8_ep8) ;;
    *) die "unsupported KIMI_K3_DECODE_TOPOLOGY=${decode_topology}" ;;
esac
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

python_bin="${RTP_LLM_PYTHON:-$(command -v python3 || true)}"
[[ -n "${python_bin}" && -x "${python_bin}" ]] \
    || die "set RTP_LLM_PYTHON to an executable Python"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
server_target="//example/kimi_k3_prefill_perf:kimi_k3_prefill_server"
server_binary="${KIMI_K3_SERVER_BINARY:-${repo_root}/bazel-bin/example/kimi_k3_prefill_perf/kimi_k3_prefill_server}"
skip_build="${KIMI_K3_SKIP_BUILD:-0}"
[[ "${skip_build}" == "0" || "${skip_build}" == "1" ]] \
    || die "KIMI_K3_SKIP_BUILD must be 0 or 1, got ${skip_build}"

service_id="${KIMI_K3_SERVICE_ID:-kimi-k3-pd}"
run_root="${KIMI_K3_RUN_ROOT:-${SMOKE_ROLE_RUNTIME_DIR:-${TMPDIR:-/tmp}/${service_id}}}"
# CpuTpBroadcaster appends a long per-rank UDS name.  Keep the default runtime
# path short even when RUN_ROOT is an archival path.
runtime_tmpdir="${KIMI_K3_TMPDIR:-/tmp/${service_id}-${role,,}}"
# Wheel overlays are prepared before the final service environment below.
# Export the short/writable runtime directory now so pip does not spill its
# temporary extraction tree into a possibly inode-constrained account home.
mkdir -p "${run_root}" "${runtime_tmpdir}"
export TMPDIR="${runtime_tmpdir}"
# FlashInfer's generated build.ninja embeds absolute include paths from the
# active Bazel runfiles tree.  The account-wide default cache can therefore
# become invalid after switching worktrees/output bases.  Keep the JIT cache
# inside this controlled run unless the caller explicitly supplies another
# validated location.
flashinfer_workspace_base="${KIMI_K3_FLASHINFER_WORKSPACE_BASE:-${run_root}/flashinfer-workspace}"
export FLASHINFER_WORKSPACE_BASE="${flashinfer_workspace_base}"
# A virtualized environment can report a generic architecture even though K3
# runs on the B300/SM103a device, and its base environment can already export
# the wrong value. FlashInfer uses this value to choose the JIT directory and
# compile flags. Force the K3-specific default instead of preserving an
# inherited value; only the namespaced override may change it.
export FLASHINFER_CUDA_ARCH_LIST="${KIMI_K3_FLASHINFER_CUDA_ARCH_LIST:-10.3a}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

# K3 的 MoE 只有 DeepGEMM MegaMoE 一条实现,它的 dispatch/combine 融在
# symmetric-memory kernel 里。DeepEP 那条路径(以及解析 DeepEP wheel 的整段)
# 已删除的 Torch 专家循环会把选中的专家反量化成
# BF16,93 层 Decode 首次使用即耗尽显存。

max_seq_len="${KIMI_K3_MAX_SEQ_LEN:-16384}"
# The frontend builds its own ModelArgs from MAX_SEQ_LEN, while the backend
# also receives --max_seq_len below. Keep both sides of the local service on
# the same limit.
export MAX_SEQ_LEN="${max_seq_len}"
max_batch_tokens_size="${MAX_BATCH_TOKENS_SIZE:-}"
seq_size_per_block="${SEQ_SIZE_PER_BLOCK:-4096}"
kernel_seq_size_per_block="${KERNEL_SEQ_SIZE_PER_BLOCK:-128}"
concurrency_limit="${CONCURRENCY_LIMIT:-2}"
max_context_batch_size="${MAX_CONTEXT_BATCH_SIZE:-1}"
reuse_cache="${KIMI_K3_REUSE_CACHE:-0}"
if [[ "${role}" == "DECODE" ]]; then
    # dp8_ep8 Decode 依赖已经删除的 DeepEP MoE 实现。
    [[ "${decode_topology}" == "tp8_ep8" ]] \
        || die "KIMI_K3_DECODE_TOPOLOGY=${decode_topology} needs the removed DeepEP MoE; use tp8_ep8"
fi
# KDA/MLA 后端与 perf fusions 现在由模型按 PD 角色决定(kimi_k3.py 的
# _is_prefill_role),不再经环境变量传入。
default_kv_cache_mem_mb=8192

kv_cache_mem_mb="${KIMI_K3_KV_CACHE_MEM_MB:-${default_kv_cache_mem_mb}}"
enable_cuda_graph="${ENABLE_CUDA_GRAPH:-0}"
enable_cuda_graph_debug_mode="${ENABLE_CUDA_GRAPH_DEBUG_MODE:-0}"
decode_capture_config="${DECODE_CAPTURE_CONFIG:-}"
prefill_capture_config="${PREFILL_CAPTURE_CONFIG:-}"

operator_overlay="${OPS_OVERLAY:-}"
if [[ -z "${operator_overlay}" ]]; then
        operator_bundle="${repo_root}/example/kimi_k3_prefill_perf/wheels"
        operator_manifest="${operator_bundle}/SHA256SUMS"
        [[ -f "${operator_manifest}" ]] \
            || die "missing bundled operator manifest ${operator_manifest}"
        (
            cd "${operator_bundle}"
            sha256sum --check SHA256SUMS
        ) || die "bundled K3 operator wheel checksum failed"
        operator_manifest_sha="$(sha256sum "${operator_manifest}" | awk '{print $1}')"
        operator_overlay="${run_root}/operator-overlay/${operator_manifest_sha}-${role,,}"
        operator_marker="${operator_overlay}/.kimi-k3-operators-installed"
        if [[ ! -f "${operator_marker}" ]]; then
            mkdir -p "${operator_overlay}"
            operator_wheels=(
                "${operator_bundle}/deep_gemm-2.6.1-cp310-cp310-linux_x86_64.whl"
            )
            [[ "${#operator_wheels[@]}" -gt 0 ]] \
                || die "operator overlay requested without a selected operator wheel"
            "${python_bin}" -m pip install \
                --no-deps --upgrade \
                --target "${operator_overlay}" \
                "${operator_wheels[@]}"
            touch "${operator_marker}"
        fi
fi
# 调用方自带 overlay 时也要进 PYTHONPATH。
export PYTHONPATH="${operator_overlay}${PYTHONPATH:+:${PYTHONPATH}}"

for flag_name in \
    reuse_cache \
    enable_cuda_graph \
    enable_cuda_graph_debug_mode; do
    flag_value="${!flag_name}"
    [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] \
        || die "${flag_name} must resolve to 0 or 1, got ${flag_value}"
done
if [[ "${enable_cuda_graph_debug_mode}" == "1" && "${enable_cuda_graph}" != "1" ]]; then
    die "ENABLE_CUDA_GRAPH_DEBUG_MODE=1 requires ENABLE_CUDA_GRAPH=1"
fi
if [[ "${role}" == "PREFILL" && -n "${decode_capture_config}" ]]; then
    die "DECODE_CAPTURE_CONFIG is only valid for the Decode role"
fi
if [[ "${role}" == "DECODE" && -n "${prefill_capture_config}" ]]; then
    die "PREFILL_CAPTURE_CONFIG is only valid for the Prefill role"
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
else
    local_endpoint="${DECODE_ENDPOINT}"
    remote_endpoint="${PREFILL_ENDPOINT}"
    start_port="${decode_port}"
    remote_port="${prefill_port}"
    if [[ "${decode_topology}" == "tp8_ep8" ]]; then
        tp_size=8
        dp_size=1
    else
        tp_size=1
        dp_size=8
    fi
fi

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
deepgemm_jit_compiler="${KIMI_K3_DEEPGEMM_JIT_COMPILER:-auto}"
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
            die "KIMI_K3_DEEPGEMM_JIT_COMPILER must be auto, nvcc or nvrtc"
            ;;
esac
if [[ "${deepgemm_jit_compiler}" == "nvrtc" ]]; then
    export DG_JIT_USE_NVRTC=1
else
    export DG_JIT_USE_NVRTC=0
fi
export OPS_OVERLAY="${operator_overlay}"
export KIMI_K3_MEGA_MAX_TOKENS_PER_RANK="${KIMI_K3_MEGA_MAX_TOKENS_PER_RANK:-8192}"


mkdir -p "${LOG_PATH}" "${run_root}/work/${role,,}" "${TMPDIR}"

echo "Kimi K3 ${role} configuration:"
echo "  local endpoint:  ${local_endpoint}"
echo "  remote endpoint: ${remote_endpoint}"
echo "  PD no-proxy:      ${pd_no_proxy_hosts}"
echo "  checkpoint:      ${CHECKPOINT_PATH}"
echo "  topology:        TP${tp_size}/DP${dp_size}/EP8"
if [[ "${role}" == "DECODE" ]]; then
    echo "  decode topology: ${decode_topology}"
fi
echo "  load method:     ${LOAD_METHOD}"
echo "  DeepGEMM JIT:    ${deepgemm_jit_compiler}"
echo "  concurrency:     generate=${concurrency_limit}, context=${max_context_batch_size}"
if [[ -n "${max_batch_tokens_size}" ]]; then
    echo "  batch tokens:    ${max_batch_tokens_size}"
fi
echo "  reuse cache:     ${reuse_cache}"
echo "  cache blocks:    seq=${seq_size_per_block}, kernel=${kernel_seq_size_per_block}"
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
    --dp_size "${dp_size}"
    --ep_size 8
    --world_size 8
    --local_world_size 8
    --remote_server_port "${remote_port}"
    --max_seq_len "${max_seq_len}"
    --max_context_batch_size "${max_context_batch_size}"
    --seq_size_per_block "${seq_size_per_block}"
    --kernel_seq_size_per_block "${kernel_seq_size_per_block}"
    --kv_cache_mem_mb "${kv_cache_mem_mb}"
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
    --cache_store_rdma_mode 0
    --load_cache_timeout_ms 7200000
    --load_method "${LOAD_METHOD}"
    --ft_core_dump_on_exception "${FT_CORE_DUMP_ON_EXCEPTION:-0}"
    --shutdown_timeout 5
)

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

if [[ "${KIMI_K3_DRY_RUN:-0}" == "1" ]]; then
    printf 'command:'
    printf ' %q' "${server_binary}" "${server_args[@]}"
    printf '\n'
    exit 0
fi

if [[ "${skip_build}" == "0" ]]; then
    bazel_startup_args=()
    if [[ -n "${KIMI_K3_BAZEL_OUTPUT_BASE:-}" ]]; then
        mkdir -p "${KIMI_K3_BAZEL_OUTPUT_BASE}"
        bazel_startup_args+=("--output_base=${KIMI_K3_BAZEL_OUTPUT_BASE}")
    fi
    (
        cd "${repo_root}"
        bazelisk "${bazel_startup_args[@]}" \
            build --config=cuda13 --config=sm10x "${server_target}"
    ) || die "failed to build ${server_target}"
fi
[[ -x "${server_binary}" ]] || die "missing Bazel launcher ${server_binary}"

cd "${run_root}/work/${role,,}"
exec "${server_binary}" "${server_args[@]}"
