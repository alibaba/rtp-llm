#!/usr/bin/env bash
#
# Start one side of the validated Kimi K3 PD topology:
#   Prefill: TP8 / DP1 / EP8
#   Decode:  TP1 / DP8 / EP8
#
# Build and install this checkout before launching:
#   bazelisk build --config=cuda13 --config=sm10x //rtp_llm:rtp_llm
#   python3 -m pip install --force-reinstall --no-deps \
#     bazel-bin/rtp_llm/rtp_llm-*.whl
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
  KIMI_K3_BAZEL_EXTERNAL_ROOT            defaults to this checkout's
                                         bazel-github-opensource/external
  KIMI_K3_RUN_ROOT                      defaults below TMPDIR
  KIMI_K3_TMPDIR                        defaults to KIMI_K3_RUN_ROOT/tmp
  KIMI_K3_SERVICE_ID                    defaults to kimi-k3-pd
  KIMI_K3_MAX_SEQ_LEN                   defaults to 16384
  KIMI_K3_KV_CACHE_MEM_MB               defaults to 1024
  KIMI_K3_DECODE_CPU_OFFLOAD_START      defaults to auto; integer or none
  KIMI_K3_KDA_BACKEND                   defaults to kernel
  KIMI_K3_KDA_FLA37_PRECOMPILED_DIR     required for fla37_precompiled
  KIMI_K3_DEEP_EP_PYTHONPATH            optional DeepEP site-packages overlay
  KIMI_K3_ACCURACY_MODE                 defaults to native_mla; one of:
                                         canonical, native_mla, native
  KIMI_K3_ACCURACY_RETAIN_FULL_TP_WEIGHTS
                                         defaults to 0 in this service launcher
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
bazel_external_root="${KIMI_K3_BAZEL_EXTERNAL_ROOT:-${repo_root}/bazel-github-opensource/external}"
bazel_pythonpath=""
if [[ -d "${bazel_external_root}" ]]; then
    while IFS= read -r site_packages; do
        bazel_pythonpath="${bazel_pythonpath:+${bazel_pythonpath}:}${site_packages}"
    done < <(
        find -L "${bazel_external_root}" \
            -mindepth 2 -maxdepth 2 -type d -name site-packages \
            -path '*/pip_gpu_cuda13_torch_*/*' \
            -print | LC_ALL=C sort
    )
fi

rtp_llm_libs="$(
    "${python_bin}" -c '
from importlib.metadata import distribution
print(distribution("rtp-llm").locate_file("rtp_llm/libs"))
'
)" || die "rtp-llm wheel is not installed for ${python_bin}"
[[ -f "${rtp_llm_libs}/librtp_compute_ops.so" ]] \
    || die "missing ${rtp_llm_libs}/librtp_compute_ops.so"

runtime_pythonpath="${rtp_llm_libs}"
if [[ -n "${bazel_pythonpath}" ]]; then
    runtime_pythonpath="${runtime_pythonpath}:${bazel_pythonpath}"
fi
if [[ -n "${KIMI_K3_DEEP_EP_PYTHONPATH:-}" ]]; then
    [[ -d "${KIMI_K3_DEEP_EP_PYTHONPATH}" ]] \
        || die "KIMI_K3_DEEP_EP_PYTHONPATH is not a directory"
    runtime_pythonpath="${KIMI_K3_DEEP_EP_PYTHONPATH}:${runtime_pythonpath}"
fi
export PYTHONPATH="${runtime_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"

torch_libs="$(
    "${python_bin}" -c '
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
'
)"
export LD_LIBRARY_PATH="${rtp_llm_libs}:${torch_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

service_id="${KIMI_K3_SERVICE_ID:-kimi-k3-pd}"
run_root="${KIMI_K3_RUN_ROOT:-${TMPDIR:-/tmp}/${service_id}}"
runtime_tmpdir="${KIMI_K3_TMPDIR:-${run_root}/tmp}"
max_seq_len="${KIMI_K3_MAX_SEQ_LEN:-16384}"
kv_cache_mem_mb="${KIMI_K3_KV_CACHE_MEM_MB:-1024}"
kda_backend="${KIMI_K3_KDA_BACKEND:-kernel}"
accuracy_mode="${KIMI_K3_ACCURACY_MODE:-native_mla}"

case "${kda_backend}" in
    kernel | reference) ;;
    fla37_precompiled)
        : "${KIMI_K3_KDA_FLA37_PRECOMPILED_DIR:?required by fla37_precompiled}"
        [[ -d "${KIMI_K3_KDA_FLA37_PRECOMPILED_DIR}" ]] \
            || die "KIMI_K3_KDA_FLA37_PRECOMPILED_DIR is not a directory"
        ;;
    *) die "unsupported KIMI_K3_KDA_BACKEND=${kda_backend}" ;;
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
    tp_size=1
    dp_size=8
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
export KIMI_K3_MLA_BACKEND=kernel

export KIMI_K3_ACCURACY_CANONICAL_TP="${canonical_tp}"
export KIMI_K3_ACCURACY_CANONICAL_EP="${canonical_ep}"
export KIMI_K3_ACCURACY_CANONICAL_MLA="${canonical_mla}"
# A 93-layer TP8 Prefill otherwise retains tens of GiB of gathered diagnostic
# weights and can OOM on its first request. Re-gathering preserves the canonical
# GEMM result while keeping the one-request accuracy service within memory.
export KIMI_K3_ACCURACY_RETAIN_FULL_TP_WEIGHTS="${KIMI_K3_ACCURACY_RETAIN_FULL_TP_WEIGHTS:-0}"
unset KIMI_K3_ACCURACY_TRACE_DIR
unset KIMI_K3_ACCURACY_TRACE_MODE
unset KIMI_K3_ACCURACY_TRACE_ENABLE_FILE
unset KIMI_K3_ACCURACY_TRACE_FULL_ROUTER
unset KIMI_K3_ACCURACY_TRACE_FORWARD_INDEX
unset KIMI_K3_ACCURACY_TRACE_INPUT_TOKEN_ID
unset KIMI_K3_ACCURACY_TRACE_RANK

unset KIMI_K3_CPU_OFFLOAD_EXPERT_LAYER_START
if [[ "${role}" == "DECODE" ]]; then
    offload_start="${KIMI_K3_DECODE_CPU_OFFLOAD_START:-auto}"
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
        # fastsafetensors loads.  Small sliced checkpoints need no offload.
        if (( num_hidden_layers > 60 )); then
            offload_start=60
        else
            offload_start=none
        fi
    fi
    if [[ "${offload_start}" != "none" ]]; then
        [[ "${offload_start}" =~ ^[0-9]+$ ]] \
            || die "KIMI_K3_DECODE_CPU_OFFLOAD_START must be auto, an integer, or none"
        export KIMI_K3_CPU_OFFLOAD_EXPERT_LAYER_START="${offload_start}"
    fi
fi

mkdir -p "${LOG_PATH}" "${run_root}/work/${role,,}" "${TMPDIR}"

echo "Kimi K3 ${role} configuration:"
echo "  local endpoint:  ${local_endpoint}"
echo "  remote endpoint: ${remote_endpoint}"
echo "  checkpoint:      ${CHECKPOINT_PATH}"
echo "  topology:        TP${tp_size}/DP${dp_size}/EP8"
echo "  load method:     ${LOAD_METHOD}"
echo "  accuracy mode:   ${accuracy_mode}"
echo "  runtime tmp:     ${TMPDIR}"
echo "  logs:            ${LOG_PATH}"

server_args=(
    -m rtp_llm.start_server
    --role_type "${role}"
    --tp_size "${tp_size}"
    --dp_size "${dp_size}"
    --ep_size 8
    --world_size 8
    --local_world_size 8
    --remote_server_port "${remote_port}"
    --max_seq_len "${max_seq_len}"
    --seq_size_per_block 4096
    --kernel_seq_size_per_block 128
    --kv_cache_mem_mb "${kv_cache_mem_mb}"
    --ssm_state_dtype fp32
    --warm_up 0
    --reuse_cache 0
    --enable_device_cache 1
    --concurrency_limit 2
    --use_deepep_moe 1
    --use_deepep_internode 0
    --use_deepep_low_latency 0
    --deep_ep_num_sm 24
    --use_all_gather 0
    --enable_cuda_graph 0
    --cache_store_rdma_mode 0
    --load_cache_timeout_ms 7200000
    --load_method "${LOAD_METHOD}"
    --ft_core_dump_on_exception 0
    --shutdown_timeout 5
)

if [[ "${KIMI_K3_DRY_RUN:-0}" == "1" ]]; then
    printf 'command:'
    printf ' %q' "${python_bin}" "${server_args[@]}"
    printf '\n'
    exit 0
fi

"${python_bin}" -c \
    'import rtp_llm.ops.compute_ops; print("RTP-LLM runtime import: OK")'

"${python_bin}" -c \
    'import deep_ep; print(f"DeepEP runtime: {deep_ep.__file__}")' \
    || die "Kimi K3 EP8 requires a CUDA-compatible deep_ep installation"

cd "${run_root}/work/${role,,}"
exec "${python_bin}" "${server_args[@]}"
