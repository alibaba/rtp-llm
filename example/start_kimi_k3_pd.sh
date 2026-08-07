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
  KIMI_K3_FASTSAFETENSORS_STREAMING     defaults to 1; close each staging
                                         batch before loading the next
  KIMI_K3_FASTSAFETENSORS_FILES_PER_BATCH
                                         defaults to 1 for bounded GPU staging
  KIMI_K3_SERVICE_ID                    defaults to kimi-k3-pd
  KIMI_K3_MAX_SEQ_LEN                   defaults to 16384
  KIMI_K3_KV_CACHE_MEM_MB               defaults to 8192
  SEQ_SIZE_PER_BLOCK                    defaults to 4096
  KERNEL_SEQ_SIZE_PER_BLOCK             defaults to 128
  CONCURRENCY_LIMIT                     defaults to 2; set this to the GPQA
                                         GENERATION_WORKERS value
  MAX_CONTEXT_BATCH_SIZE                defaults to 1
  KIMI_K3_DECODE_CPU_OFFLOAD_START      defaults to auto; integer or none
  KIMI_K3_DECODE_TOPOLOGY               defaults to tp8_ep8; one of:
                                         tp8_ep8, dp8_ep8
  KIMI_K3_EXECUTION_MODE                defaults to optimized; one of:
                                         optimized, accuracy
  KIMI_K3_USE_HOST_METADATA             optimized defaults to 1; accuracy to 0
  KIMI_K3_SP_MOE                        optimized TP8/EP8 defaults to 1;
                                         all other modes/topologies default to 0
  KIMI_K3_KDA_BACKEND                   optimized Prefill defaults to cula;
                                         accuracy Prefill defaults to kernel;
                                         Decode tp8_ep8 defaults to kernel;
                                         Decode dp8_ep8 defaults to
                                         fla37_precompiled
  KIMI_K3_MOE_BACKEND                   optimized TP8/EP8 defaults to
                                         deep_gemm_mega; otherwise deepep
  KIMI_K3_MLA_BACKEND                   optimized Prefill defaults to
                                         flashmla; Decode defaults to kernel
  KIMI_K3_PERF_FUSIONS                  optimized TP8/EP8 defaults to 1;
                                         all other modes/topologies default to 0
  KIMI_K3_FUSED_AG_GEMM                 auto|off|force; defaults to auto;
                                         Prefill global M<64K always uses NCCL
  KIMI_K3_BATCHED_KDA_DECODE            optimized TP8/EP8 Decode defaults to 1;
                                         all other modes/topologies default to 0
  KIMI_K3_PERF_MODE                     strict performance-path validation only
  KIMI_K3_KDA_FLA37_PRECOMPILED_DIR     defaults to the bundled SM103 image
                                         for fla37_precompiled
  KIMI_K3_DEEP_EP_PYTHONPATH            optional DeepEP site-packages overlay;
                                         otherwise installs the pinned CUDA13
                                         wheel from the repo or shared NAS
  KIMI_K3_DEEP_EP_WHEEL                 optional path to the pinned DeepEP
                                         wheel
  KIMI_K3_OPERATOR_PYTHONPATH           optional KDA/DeepGEMM operator overlay;
                                         optimized Prefill otherwise installs
                                         the bundled fixed wheels automatically
  KIMI_K3_ACCURACY_MODE                 defaults to native; one of:
                                         canonical, native_mla, native
  KIMI_K3_ACCURACY_CANONICAL_TP         optional 0/1 diagnostic override
  KIMI_K3_ACCURACY_CANONICAL_EP         optional 0/1 diagnostic override
  KIMI_K3_ACCURACY_CANONICAL_MLA        optional 0/1 diagnostic override
  KIMI_K3_ACCURACY_RETAIN_FULL_TP_WEIGHTS
                                         defaults to 0 in this service launcher
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
execution_mode="${KIMI_K3_EXECUTION_MODE:-optimized}"
case "${execution_mode}" in
    optimized | accuracy) ;;
    *) die "unsupported KIMI_K3_EXECUTION_MODE=${execution_mode}" ;;
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

# Optimized TP8/EP8 uses DeepGEMM MegaMoE, whose dispatch/combine is fused
# inside the symmetric-memory kernel.  Do not require or install DeepEP on
# that path.  Accuracy mode, legacy DP8, or an explicit deepep selection still
# resolves the pinned wheel exactly as before.
requested_moe_backend="${KIMI_K3_MOE_BACKEND:-}"
need_deep_ep=0
if [[ "${requested_moe_backend}" == "deepep" ]] \
    || { [[ -z "${requested_moe_backend}" ]] \
        && { [[ "${execution_mode}" == "accuracy" ]] \
            || { [[ "${role}" == "DECODE" ]] \
                && [[ "${decode_topology}" == "dp8_ep8" ]]; }; }; }; then
    need_deep_ep=1
fi
if [[ "${need_deep_ep}" == "1" && -z "${KIMI_K3_DEEP_EP_PYTHONPATH:-}" ]]; then
    deep_ep_bundle="${repo_root}/example/kimi_k3_pd/wheels"
    deep_ep_manifest="${deep_ep_bundle}/SHA256SUMS"
    deep_ep_wheel_name="deep_ep-1.2.1.12+37fda1c.base-cp310-cp310-linux_x86_64.whl"
    [[ -f "${deep_ep_manifest}" ]] || die "missing DeepEP checksum manifest"
    deep_ep_expected_sha="$(awk -v name="${deep_ep_wheel_name}" '$2 == name {print $1}' "${deep_ep_manifest}")"
    [[ -n "${deep_ep_expected_sha}" ]] || die "DeepEP wheel is absent from checksum manifest"
    deep_ep_wheel=""
    for candidate in \
        "${KIMI_K3_DEEP_EP_WHEEL:-}" \
        "${deep_ep_bundle}/${deep_ep_wheel_name}" \
        "/mnt/nas1/rtp-llm-k3-deps/${deep_ep_wheel_name}"; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            deep_ep_wheel="${candidate}"
            break
        fi
    done
    [[ -n "${deep_ep_wheel}" ]] \
        || die "pinned CUDA13 DeepEP wheel not found; set KIMI_K3_DEEP_EP_WHEEL"
    [[ "$(sha256sum "${deep_ep_wheel}" | awk '{print $1}')" == "${deep_ep_expected_sha}" ]] \
        || die "DeepEP wheel checksum failed: ${deep_ep_wheel}"
    deep_ep_manifest_sha="$(sha256sum "${deep_ep_manifest}" | awk '{print $1}')"
    deep_ep_overlay="${run_root}/deep-ep-overlay/${deep_ep_manifest_sha}"
    deep_ep_marker="${deep_ep_overlay}/.kimi-k3-deep-ep-installed"
    if [[ ! -f "${deep_ep_marker}" ]]; then
        mkdir -p "${deep_ep_overlay}"
        "${python_bin}" -m pip install \
            --no-deps --upgrade \
            --target "${deep_ep_overlay}" \
            "${deep_ep_wheel}"
        touch "${deep_ep_marker}"
    fi
    export KIMI_K3_DEEP_EP_PYTHONPATH="${deep_ep_overlay}"
fi

runtime_pythonpath=""
if [[ -n "${KIMI_K3_DEEP_EP_PYTHONPATH:-}" ]]; then
    [[ -d "${KIMI_K3_DEEP_EP_PYTHONPATH}" ]] \
        || die "KIMI_K3_DEEP_EP_PYTHONPATH is not a directory"
    runtime_pythonpath="${KIMI_K3_DEEP_EP_PYTHONPATH}"
fi
if [[ -n "${KIMI_K3_OPERATOR_PYTHONPATH:-}" ]]; then
    [[ -d "${KIMI_K3_OPERATOR_PYTHONPATH}" ]] \
        || die "KIMI_K3_OPERATOR_PYTHONPATH is not a directory"
    runtime_pythonpath="${KIMI_K3_OPERATOR_PYTHONPATH}${runtime_pythonpath:+:${runtime_pythonpath}}"
fi
if [[ -n "${runtime_pythonpath}" ]]; then
    export PYTHONPATH="${runtime_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"
fi

max_seq_len="${KIMI_K3_MAX_SEQ_LEN:-16384}"
seq_size_per_block="${SEQ_SIZE_PER_BLOCK:-4096}"
kernel_seq_size_per_block="${KERNEL_SEQ_SIZE_PER_BLOCK:-128}"
concurrency_limit="${CONCURRENCY_LIMIT:-2}"
max_context_batch_size="${MAX_CONTEXT_BATCH_SIZE:-1}"
default_batched_kda_decode=0
case "${execution_mode}" in
    optimized)
        default_accuracy_mode=native
        if [[ "${role}" == "DECODE" ]]; then
            if [[ "${decode_topology}" == "tp8_ep8" ]]; then
                # The bundled FLA 3.7 cubin is specialized for the TP1
                # 96-head state.  TP8 owns 12 heads per rank and therefore
                # uses the shape-generic recurrent Triton kernel.
                default_kda_backend=kernel
                # MegaMoE owns the EP dispatch/combine and consumes the
                # checkpoint-native MXFP4 experts directly.  The old
                # DeepEP+Torch expert loop dequantizes selected experts to
                # BF16 and exhausts a full 93-layer Decode on first use.
                default_moe_backend=deep_gemm_mega
                default_sp_moe=1
                default_perf_fusions=1
                default_batched_kda_decode=1
                default_decode_offload_start=none
            else
                default_kda_backend=fla37_precompiled
                default_moe_backend=deepep
                default_sp_moe=0
                default_perf_fusions=0
                default_decode_offload_start=auto
            fi
            default_kv_cache_mem_mb=8192
        else
            default_kda_backend=cula
            default_moe_backend=deep_gemm_mega
            default_mla_backend=flashmla
            default_sp_moe=1
            default_perf_fusions=1
            default_kv_cache_mem_mb=8192
            default_decode_offload_start=auto
        fi
        if [[ "${role}" == "DECODE" ]]; then
            default_mla_backend=kernel
        fi
        default_use_host_metadata=1
        ;;
    accuracy)
        # The validated 93-layer baseline is fully native TP/EP/MLA with the
        # FLA 0.5.1 recurrent image on Decode.  canonical TP/EP is retained as
        # an explicit diagnostic mode, but gathering full 93-layer weights can
        # exceed one B300.
        default_accuracy_mode=native
        if [[ "${role}" == "DECODE" ]]; then
            default_kv_cache_mem_mb=8192
            if [[ "${decode_topology}" == "tp8_ep8" ]]; then
                default_kda_backend=kernel
                default_sp_moe=1
                default_decode_offload_start=none
            else
                default_kda_backend=fla37_precompiled
                default_sp_moe=0
                default_decode_offload_start=auto
            fi
        else
            default_kda_backend=kernel
            default_kv_cache_mem_mb=8192
            default_sp_moe=0
            default_decode_offload_start=auto
        fi
        default_moe_backend=deepep
        default_mla_backend=kernel
        default_perf_fusions=0
        # These are capacity choices only.  Keep the validated mathematical
        # path while retaining enough margin for eight concurrent
        # fastsafetensors loaders on one B300 node.
        default_use_host_metadata=0
        ;;
    *) die "unsupported KIMI_K3_EXECUTION_MODE=${execution_mode}" ;;
esac

use_host_metadata="${KIMI_K3_USE_HOST_METADATA:-${default_use_host_metadata}}"
sp_moe="${KIMI_K3_SP_MOE:-${default_sp_moe}}"
kda_backend="${KIMI_K3_KDA_BACKEND:-${default_kda_backend}}"
moe_backend="${KIMI_K3_MOE_BACKEND:-${default_moe_backend}}"
mla_backend="${KIMI_K3_MLA_BACKEND:-${default_mla_backend}}"
perf_fusions="${KIMI_K3_PERF_FUSIONS:-${default_perf_fusions}}"
fused_ag_gemm="${KIMI_K3_FUSED_AG_GEMM:-auto}"
batched_kda_decode="${KIMI_K3_BATCHED_KDA_DECODE:-${default_batched_kda_decode}}"
perf_mode="${KIMI_K3_PERF_MODE:-0}"
accuracy_mode="${KIMI_K3_ACCURACY_MODE:-${default_accuracy_mode}}"
kv_cache_mem_mb="${KIMI_K3_KV_CACHE_MEM_MB:-${default_kv_cache_mem_mb}}"
kda_fla37_precompiled_dir="${KIMI_K3_KDA_FLA37_PRECOMPILED_DIR:-${repo_root}/example/kimi_k3_pd/fla37-sm103}"
enable_cuda_graph="${ENABLE_CUDA_GRAPH:-0}"
enable_cuda_graph_debug_mode="${ENABLE_CUDA_GRAPH_DEBUG_MODE:-0}"
decode_capture_config="${DECODE_CAPTURE_CONFIG:-}"
prefill_capture_config="${PREFILL_CAPTURE_CONFIG:-}"

if { [[ "${role}" == "PREFILL" ]] \
        && [[ "${kda_backend}" == "flash_kda" ]]; } \
    || [[ "${moe_backend}" == "deep_gemm_mega" ]]; then
    if [[ -z "${KIMI_K3_OPERATOR_PYTHONPATH:-}" ]]; then
        operator_bundle="${repo_root}/example/kimi_k3_prefill_perf/wheels"
        operator_manifest="${operator_bundle}/SHA256SUMS"
        [[ -f "${operator_manifest}" ]] \
            || die "missing bundled operator manifest ${operator_manifest}"
        (
            cd "${operator_bundle}"
            sha256sum --check SHA256SUMS
        ) || die "bundled K3 operator wheel checksum failed"
        operator_manifest_sha="$(sha256sum "${operator_manifest}" | awk '{print $1}')"
        operator_overlay="${run_root}/operator-overlay/${operator_manifest_sha}-${kda_backend}-${moe_backend}"
        operator_marker="${operator_overlay}/.kimi-k3-operators-installed"
        if [[ ! -f "${operator_marker}" ]]; then
            mkdir -p "${operator_overlay}"
            operator_wheels=()
            if [[ "${kda_backend}" == "flash_kda" ]]; then
                operator_wheels+=(
                    "${operator_bundle}/flash_kda-0.0.1-cp310-cp310-linux_x86_64.whl"
                )
            fi
            if [[ "${moe_backend}" == "deep_gemm_mega" ]]; then
                operator_wheels+=(
                    "${operator_bundle}/deep_gemm-2.6.1-cp310-cp310-linux_x86_64.whl"
                )
            fi
            [[ "${#operator_wheels[@]}" -gt 0 ]] \
                || die "operator overlay requested without a selected operator wheel"
            "${python_bin}" -m pip install \
                --no-deps --upgrade \
                --target "${operator_overlay}" \
                "${operator_wheels[@]}"
            touch "${operator_marker}"
        fi
        export KIMI_K3_OPERATOR_PYTHONPATH="${operator_overlay}"
        export PYTHONPATH="${operator_overlay}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
fi

for flag_name in \
    use_host_metadata \
    sp_moe \
    perf_fusions \
    batched_kda_decode \
    perf_mode \
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

case "${kda_backend}" in
    kernel | reference) ;;
    cula | flash_kda)
        [[ "${role}" == "PREFILL" ]] \
            || die "KIMI_K3_KDA_BACKEND=${kda_backend} is Prefill-only"
        ;;
    fla37_precompiled)
        [[ -d "${kda_fla37_precompiled_dir}" ]] \
            || die "KIMI_K3_KDA_FLA37_PRECOMPILED_DIR is not a directory"
        ;;
    *) die "unsupported KIMI_K3_KDA_BACKEND=${kda_backend}" ;;
esac

case "${moe_backend}" in
    deepep) ;;
    deep_gemm_mega)
        if [[ "${role}" == "DECODE" && "${decode_topology}" != "tp8_ep8" ]]; then
            die "KIMI_K3_MOE_BACKEND=deep_gemm_mega Decode requires tp8_ep8"
        fi
        [[ -n "${KIMI_K3_OPERATOR_PYTHONPATH:-}" ]] \
            || die "deep_gemm_mega requires KIMI_K3_OPERATOR_PYTHONPATH"
        ;;
    *) die "unsupported KIMI_K3_MOE_BACKEND=${moe_backend}" ;;
esac

case "${mla_backend}" in
    kernel | reference) ;;
    flashmla)
        [[ "${role}" == "PREFILL" ]] \
            || die "KIMI_K3_MLA_BACKEND=flashmla is Prefill-only"
        ;;
    *) die "unsupported KIMI_K3_MLA_BACKEND=${mla_backend}" ;;
esac

case "${accuracy_mode}" in
    canonical)
        canonical_tp=1
        canonical_ep=1
        canonical_mla=1
        ;;
    native_mla)
        canonical_tp=1
        canonical_ep=1
        canonical_mla=0
        ;;
    native)
        canonical_tp=0
        canonical_ep=0
        canonical_mla=0
        ;;
    *) die "unsupported KIMI_K3_ACCURACY_MODE=${accuracy_mode}" ;;
esac

canonical_tp="${KIMI_K3_ACCURACY_CANONICAL_TP:-${canonical_tp}}"
canonical_ep="${KIMI_K3_ACCURACY_CANONICAL_EP:-${canonical_ep}}"
canonical_mla="${KIMI_K3_ACCURACY_CANONICAL_MLA:-${canonical_mla}}"
for flag_name in canonical_tp canonical_ep canonical_mla; do
    flag_value="${!flag_name}"
    [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] \
        || die "${flag_name} must resolve to 0 or 1, got ${flag_value}"
done

if [[ "${role}" == "DECODE" && "${decode_topology}" == "tp8_ep8" ]]; then
    [[ "${sp_moe}" == "1" ]] \
        || die "Decode TP8/EP8 requires KIMI_K3_SP_MOE=1"
    [[ "${canonical_tp}" == "0" ]] \
        || die "Decode TP8/EP8 token-SP requires native TP projections"
fi

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
export KIMI_K3_KDA_BACKEND="${kda_backend}"
export KIMI_K3_MOE_BACKEND="${moe_backend}"
export KIMI_K3_MLA_BACKEND="${mla_backend}"
export KIMI_K3_USE_HOST_METADATA="${use_host_metadata}"
export KIMI_K3_SP_MOE="${sp_moe}"
export KIMI_K3_PERF_FUSIONS="${perf_fusions}"
export KIMI_K3_FUSED_AG_GEMM="${fused_ag_gemm}"
export KIMI_K3_BATCHED_KDA_DECODE="${batched_kda_decode}"
export KIMI_K3_PERF_MODE="${perf_mode}"
[[ "${KIMI_K3_FUSED_AG_GEMM}" =~ ^(auto|off|force)$ ]] \
    || die "KIMI_K3_FUSED_AG_GEMM must be auto, off, or force"
export KIMI_K3_FASTSAFETENSORS_STREAMING="${KIMI_K3_FASTSAFETENSORS_STREAMING:-1}"
export KIMI_K3_FASTSAFETENSORS_FILES_PER_BATCH="${KIMI_K3_FASTSAFETENSORS_FILES_PER_BATCH:-1}"
[[ "${KIMI_K3_FASTSAFETENSORS_STREAMING}" == "0" \
    || "${KIMI_K3_FASTSAFETENSORS_STREAMING}" == "1" ]] \
    || die "KIMI_K3_FASTSAFETENSORS_STREAMING must be 0 or 1"
[[ "${KIMI_K3_FASTSAFETENSORS_FILES_PER_BATCH}" =~ ^[1-8]$ ]] \
    || die "KIMI_K3_FASTSAFETENSORS_FILES_PER_BATCH must be in [1, 8]"
if [[ "${moe_backend}" == "deep_gemm_mega" ]]; then
    export KIMI_K3_REQUIRE_DEEP_EP=0
    use_deepep_moe=0
else
    export KIMI_K3_REQUIRE_DEEP_EP=1
    use_deepep_moe=1
fi
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
deepgemm_jit_compiler=none
if [[ "${moe_backend}" == "deep_gemm_mega" ]]; then
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
    export KIMI_K3_DEEPGEMM_EXPECTED_PATH="${KIMI_K3_OPERATOR_PYTHONPATH}"
    export KIMI_K3_MEGA_MAX_TOKENS_PER_RANK="${KIMI_K3_MEGA_MAX_TOKENS_PER_RANK:-8192}"
    export OPS_OVERLAY="${KIMI_K3_OPERATOR_PYTHONPATH}"
else
    unset DG_JIT_USE_NVRTC
    unset OPS_OVERLAY
fi

unset KIMI_K3_KDA_FLA37_PRECOMPILED_DIR
unset KIMI_K3_KDA_CHUNK_STATE_BACKEND
if [[ "${kda_backend}" == "fla37_precompiled" ]]; then
    export KIMI_K3_KDA_FLA37_PRECOMPILED_DIR="${kda_fla37_precompiled_dir}"
fi
if [[ "${role}" == "PREFILL" && "${kda_backend}" == "kernel" ]]; then
    export KIMI_K3_KDA_CHUNK_STATE_BACKEND=triton
fi

export KIMI_K3_ACCURACY_CANONICAL_TP="${canonical_tp}"
export KIMI_K3_ACCURACY_CANONICAL_EP="${canonical_ep}"
export KIMI_K3_ACCURACY_CANONICAL_MLA="${canonical_mla}"
# A 93-layer TP8 Prefill otherwise retains tens of GiB of gathered diagnostic
# weights and can OOM on its first request. Re-gathering preserves the canonical
# GEMM result while keeping the one-request accuracy service within memory.
export KIMI_K3_ACCURACY_RETAIN_FULL_TP_WEIGHTS="${KIMI_K3_ACCURACY_RETAIN_FULL_TP_WEIGHTS:-0}"
if [[ -z "${KIMI_K3_ACCURACY_TRACE_DIR:-}" ]]; then
    unset KIMI_K3_ACCURACY_TRACE_MODE
    unset KIMI_K3_ACCURACY_TRACE_ENABLE_FILE
    unset KIMI_K3_ACCURACY_TRACE_FULL_ROUTER
    unset KIMI_K3_ACCURACY_TRACE_FORWARD_INDEX
    unset KIMI_K3_ACCURACY_TRACE_INPUT_TOKEN_ID
    unset KIMI_K3_ACCURACY_TRACE_RANK
fi

unset KIMI_K3_CPU_OFFLOAD_EXPERT_LAYER_START
if [[ "${role}" == "DECODE" ]]; then
    offload_start="${KIMI_K3_DECODE_CPU_OFFLOAD_START:-${default_decode_offload_start}}"
    if [[ "${offload_start}" == "auto" ]]; then
        num_hidden_layers="$(
            "${python_bin}" -c '
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
config = config.get("text_config", config)
print(config["num_hidden_layers"])
' "${CHECKPOINT_PATH}/config.json"
        )"
        # Keep a deterministic safety margin during eight concurrent
        # fastsafetensors loads. With the full 93-layer checkpoint, a cutoff
        # of 60 still reaches the B300 limit during rank-0 staging; 30 is the
        # validated cold-start setting. Small sliced checkpoints need none.
        if (( num_hidden_layers > 60 )); then
            offload_start=30
        else
            offload_start=none
        fi
    fi
    if [[ "${offload_start}" != "none" ]]; then
        [[ "${decode_topology}" == "dp8_ep8" ]] \
            || die "Decode TP8/EP8 does not support CPU expert offload"
        [[ "${offload_start}" =~ ^[0-9]+$ ]] \
            || die "KIMI_K3_DECODE_CPU_OFFLOAD_START must be auto, an integer, or none"
        export KIMI_K3_CPU_OFFLOAD_EXPERT_LAYER_START="${offload_start}"
    fi
    if [[ "${execution_mode}" == "accuracy" ]]; then
        export ACCL_DISPATCH_NUM_WARP_GROUPS="${ACCL_DISPATCH_NUM_WARP_GROUPS:-2}"
        export ACCL_COMBINE_NUM_WARP_GROUPS="${ACCL_COMBINE_NUM_WARP_GROUPS:-2}"
    fi
fi

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
echo "  execution mode:  ${execution_mode}"
echo "  accuracy mode:   ${accuracy_mode}"
echo "  host metadata:   ${use_host_metadata}"
echo "  SP MoE:          ${sp_moe}"
echo "  KDA backend:     ${kda_backend}"
echo "  MoE backend:     ${moe_backend}"
echo "  MLA backend:     ${mla_backend}"
echo "  DeepGEMM JIT:    ${deepgemm_jit_compiler}"
echo "  perf fusions:    ${perf_fusions}"
echo "  fused AG/GEMM:   ${fused_ag_gemm}"
echo "  batched KDA:     ${batched_kda_decode}"
echo "  perf validation: ${perf_mode}"
echo "  concurrency:     generate=${concurrency_limit}, context=${max_context_batch_size}"
echo "  cache blocks:    seq=${seq_size_per_block}, kernel=${kernel_seq_size_per_block}"
echo "  CUDA Graph:      enabled=${enable_cuda_graph}, debug=${enable_cuda_graph_debug_mode}"
if [[ -n "${decode_capture_config}" ]]; then
    echo "  Decode captures: ${decode_capture_config}"
fi
if [[ -n "${prefill_capture_config}" ]]; then
    echo "  Prefill captures:${prefill_capture_config}"
fi
echo "  fastsafetensors: streaming=${KIMI_K3_FASTSAFETENSORS_STREAMING}, files_per_batch=${KIMI_K3_FASTSAFETENSORS_FILES_PER_BATCH}"
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
    --reuse_cache 0
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
    --ft_core_dump_on_exception 0
    --shutdown_timeout 5
)

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
