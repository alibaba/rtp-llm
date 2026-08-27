#!/usr/bin/env bash
#
# Recommended one-command startup from a controller with SSH access to both
# hosts (the driver enters lhc_GPU and launches both roles concurrently):
#
# Merge-gate requirement: always use SMOKE_SUITE=all for the final 93-layer
# accuracy/cache acceptance run. The flow suite is only a four-layer RDMA
# connectivity and multi-round preflight; it is not a substitute for all.
#
#    PREFILL_SSH_TARGET=L20-dev-112 \
#    DECODE_SSH_TARGET=L20-dev-113 \
#    PREFILL_REPO_ROOT=/data3/user/RTP-LLM/github-opensource \
#    DECODE_REPO_ROOT=/data0/user/RTP-LLM/github-opensource \
#    PREFILL_CHECKPOINT_PATH=/data3/user/Kimi-K3 \
#    DECODE_CHECKPOINT_PATH=/data0/user/Kimi-K3 \
#    PREFILL_ENDPOINT=xx.xx.xx.xx:27188 \
#    DECODE_ENDPOINT=xx.xx.xx.xx:28188 \
#    SMOKE_RUN_ID=my-run \
#    SMOKE_SUITE=all \
#    python3 ./example/k3/kimi_k3_full_model_two_host_pd_smoke_driver.py
#
# Manual role startup remains available for debugging. Run from the RTP-LLM
# repository root inside lhc_GPU; both roles may now be started concurrently:
#
# 1. Start the Decode role (either role may now be launched first):
#    CHECKPOINT_PATH=/ssd/2/kimi-k3 \
#    PREFILL_ENDPOINT=xx.xx.xx.xx:27188 \
#    DECODE_ENDPOINT=xx.xx.xx.xx:28188 \
#    SMOKE_RUN_ID=my-run \
#    SMOKE_SUITE=all \
#    ./example/k3/kimi_k3_full_model_two_host_pd_smoke.sh decode
#
# 2. Start Prefill with the same endpoints and run ID (it waits for both the
#    Decode model and Decode result listener to become ready):
#    CHECKPOINT_PATH=/ssd/2/kimi-k3 \
#    PREFILL_ENDPOINT=xx.xx.xx.xx:27188 \
#    DECODE_ENDPOINT=xx.xx.xx.xx:28188 \
#    SMOKE_RUN_ID=my-run \
#    SMOKE_SUITE=all \
#    ./example/k3/kimi_k3_full_model_two_host_pd_smoke.sh prefill
#
# Lightweight two-host Kimi K3 full-model (93-layer) PD smoke.
#
# Run this same role script inside lhc_GPU on both hosts. There is no committed
# machine address; the optional driver accepts all SSH/host paths at runtime.
# The validated profile always enables Barex RDMA on both roles; this smoke is
# intentionally not a TCP/cache-store fallback test.
# Merge-gate runs must use SMOKE_SUITE=all; it is the only supported suite.
# Prefill checks both services, runs the complete request suite, validates model
# answers and cache metadata, then reports PASS/FAIL back to Decode. Both
# commands therefore have a meaningful exit status and clean only their own
# process group.

set -Eeuo pipefail
ulimit -c 0

die() {
    echo "error: $*" >&2
    exit 2
}

usage() {
    cat >&2 <<'EOF'
Usage (run inside lhc_GPU as the normal user):
  CHECKPOINT_PATH=/local/path/to/Kimi-K3 \
  PREFILL_ENDPOINT=prefill-host:27188 \
  DECODE_ENDPOINT=decode-host:28188 \
  SMOKE_RUN_ID=my-run \
  example/k3/kimi_k3_full_model_two_host_pd_smoke.sh decode|prefill

The two roles may start concurrently. Prefill waits for both the Decode model
and result channel. The default result channel is DECODE host at DECODE port +
100; override SMOKE_RESULT_ENDPOINT on both hosts when that port is unavailable.

The validated BF16 1M model/runtime profile is fixed by this smoke. Only host,
checkpoint, artifact, timeout and prebuilt-launcher settings are configurable.

Merge-gate accuracy validation must use SMOKE_SUITE=all. SMOKE_SUITE=flow is
only a four-layer RDMA connectivity/multi-round preflight and does not satisfy
the final acceptance requirement.

Important optional variables:
  SMOKE_ARTIFACT_ROOT       defaults to /tmp/kimi-k3-two-host-pd-smoke
  SMOKE_STARTUP_TIMEOUT_S   defaults to 14400
  SMOKE_REQUEST_TIMEOUT_S   defaults to 900
  SMOKE_RESULT_TIMEOUT_S    defaults to 18000
  SMOKE_RESULT_ENDPOINT     defaults to decode-host:(decode-port + 100)
  SMOKE_SUITE               all (default) or flow
                            flow: four-layer-friendly multi-round RDMA flow
                                  check without semantic-answer assertions
                            all: identity, single miss/hit, partial hit,
                                 concurrent all-miss/all-hit, mixed hit+miss
                                 batches, and >64K single/batched chunk cases
  SMOKE_EXPECTED_LAYERS     checkpoint layer count; defaults to 93. Set to 4
                            only for the required four-layer RDMA flow smoke.
  SMOKE_BLOCK_SIZE          physical cache page size; defaults to 4096
  SMOKE_KERNEL_BLOCK_SIZE   attention kernel page size; defaults to 128
  SMOKE_CHUNK_TOKENS        whole-model chunk budget; defaults to 65536
  SMOKE_LINEAR_STEP         KDA materialization step; defaults to 1
  SMOKE_CHUNKWISE_RDMA      1 (default) enables Layer x Chunk publication;
                            0 retains compute-all-then-transfer behavior
  KIMI_K3_DECODE_TOPOLOGY   tp8_ep8 (default) validates the legacy baseline;
                            dp8_ep8_tp1_ktp8 validates Decode TP1/DP8/EP8/KTP8
                            with a replicated full-width 576 MLA cache on every
                            Prefill TP rank; KTP partitions only KDA heads
  RTP_LLM_SERVER_BINARY     use an existing Bazel launcher
  RTP_LLM_SKIP_BUILD=1      skip the CUDA13/SM10x build in the launcher
EOF
}

[[ $# -eq 1 ]] || {
    usage
    exit 2
}
role="${1,,}"
[[ "${role}" == "prefill" || "${role}" == "decode" ]] \
    || die "role must be decode or prefill"

[[ "$(id -u)" != "0" ]] || die "run inside lhc_GPU as a normal user, not root"
[[ -f /.dockerenv || -r /proc/1/cgroup ]] \
    || die "run this smoke inside lhc_GPU"

: "${PREFILL_ENDPOINT:?PREFILL_ENDPOINT is required}"
: "${DECODE_ENDPOINT:?DECODE_ENDPOINT is required}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/ssd/2/kimi-k3}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-manual}"
[[ "${SMOKE_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "SMOKE_RUN_ID may contain only letters, digits, dot, underscore and dash"
decode_topology="${KIMI_K3_DECODE_TOPOLOGY:-tp8_ep8}"
case "${decode_topology}" in
    tp8_ep8 | dp8_ep8_tp1_ktp8) ;;
    *) die "KIMI_K3_DECODE_TOPOLOGY must be tp8_ep8 or dp8_ep8_tp1_ktp8" ;;
esac
smoke_prefill_kv_cache_mem_mb="${SMOKE_PREFILL_KV_CACHE_MEM_MB:-43000}"
if [[ "${decode_topology}" == "dp8_ep8_tp1_ktp8" ]]; then
    default_decode_kv_cache_mem_mb=8192
else
    default_decode_kv_cache_mem_mb=44000
fi
smoke_decode_kv_cache_mem_mb="${SMOKE_DECODE_KV_CACHE_MEM_MB:-${default_decode_kv_cache_mem_mb}}"
for value_name in smoke_prefill_kv_cache_mem_mb smoke_decode_kv_cache_mem_mb; do
    value="${!value_name}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
        || die "${value_name} must be a positive integer"
done

endpoint_port() {
    local endpoint="$1"
    [[ "${endpoint}" =~ ^[^:]+:[0-9]+$ ]] \
        || die "endpoint must have host:port form: ${endpoint}"
    printf '%s\n' "${endpoint##*:}"
}

prefill_port="$(endpoint_port "${PREFILL_ENDPOINT}")"
decode_port="$(endpoint_port "${DECODE_ENDPOINT}")"
decode_host="${DECODE_ENDPOINT%:*}"
default_result_port="$((decode_port + 100))"
((default_result_port <= 65535)) \
    || die "DECODE port is too high to derive the result port; set SMOKE_RESULT_ENDPOINT"
SMOKE_RESULT_ENDPOINT="${SMOKE_RESULT_ENDPOINT:-${decode_host}:${default_result_port}}"
result_port="$(endpoint_port "${SMOKE_RESULT_ENDPOINT}")"
result_host="${SMOKE_RESULT_ENDPOINT%:*}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
launcher="${repo_root}/example/k3/start_kimi_k3_pd.sh"
[[ -x "${launcher}" ]] || die "missing executable launcher ${launcher}"
case_runner="${repo_root}/example/k3/kimi_k3_full_model_pd_cases.py"
[[ -f "${case_runner}" ]] || die "missing smoke case runner ${case_runner}"

checkpoint_real="$(realpath -e "${CHECKPOINT_PATH}")" \
    || die "checkpoint does not exist: ${CHECKPOINT_PATH}"
case "${checkpoint_real}" in
    /data[0-9]*/* | /data/* | /ssd/*) ;;
    *) die "checkpoint must be on a local data disk: ${checkpoint_real}" ;;
esac
checkpoint_fs="$(findmnt -T "${checkpoint_real}" -n -o FSTYPE)"
checkpoint_source="$(findmnt -T "${checkpoint_real}" -n -o SOURCE)"
case "${checkpoint_fs}:${checkpoint_source}" in
    nfs*:* | cifs:* | smb*:* | fuse.*:* | *[Nn][Aa][Ss]*)
        die "network/NAS checkpoint is forbidden: ${checkpoint_fs}:${checkpoint_source}"
        ;;
esac
[[ -f "${checkpoint_real}/config.json" ]] \
    || die "missing checkpoint config.json"
[[ -f "${checkpoint_real}/model.safetensors.index.json" ]] \
    || die "missing checkpoint model.safetensors.index.json"
smoke_expected_layers="${SMOKE_EXPECTED_LAYERS:-93}"
[[ "${smoke_expected_layers}" =~ ^[1-9][0-9]*$ ]] \
    || die "SMOKE_EXPECTED_LAYERS must be a positive integer"
python3 - "${checkpoint_real}/config.json" "${smoke_expected_layers}" <<'PY'
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_layers = int(sys.argv[2])
layer_counts = []

def visit(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "num_hidden_layers" and isinstance(child, int):
                layer_counts.append(child)
            visit(child)
    elif isinstance(value, list):
        for child in value:
            visit(child)

visit(config)
if expected_layers not in layer_counts:
    raise SystemExit(
        f"smoke requires a {expected_layers}-layer checkpoint; "
        f"found {layer_counts or 'none'}"
    )
print(f"checkpoint layers={expected_layers}")
PY

artifact_root="${SMOKE_ARTIFACT_ROOT:-/tmp/kimi-k3-two-host-pd-smoke}"
role_dir="${artifact_root}/${SMOKE_RUN_ID}/${role}"
[[ ! -e "${role_dir}" ]] \
    || die "artifact directory already exists: ${role_dir}"
mkdir -p "${role_dir}"
service_log="${role_dir}/service.log"
accuracy_file="${role_dir}/accuracy.json"
result_file="${role_dir}/peer-result.txt"
summary_file="${role_dir}/summary.txt"

startup_timeout="${SMOKE_STARTUP_TIMEOUT_S:-14400}"
request_timeout="${SMOKE_REQUEST_TIMEOUT_S:-900}"
result_timeout="${SMOKE_RESULT_TIMEOUT_S:-18000}"
for timeout_value in "${startup_timeout}" "${request_timeout}" "${result_timeout}"; do
    [[ "${timeout_value}" =~ ^[1-9][0-9]*$ ]] \
        || die "smoke timeouts must be positive integers"
done

smoke_block_size="${SMOKE_BLOCK_SIZE:-4096}"
smoke_kernel_block_size="${SMOKE_KERNEL_BLOCK_SIZE:-128}"
smoke_chunk_tokens="${SMOKE_CHUNK_TOKENS:-65536}"
smoke_linear_step="${SMOKE_LINEAR_STEP:-1}"
smoke_chunkwise_rdma="${SMOKE_CHUNKWISE_RDMA:-1}"
for size_value in \
    "${smoke_block_size}" \
    "${smoke_kernel_block_size}" \
    "${smoke_chunk_tokens}" \
    "${smoke_linear_step}"; do
    [[ "${size_value}" =~ ^[1-9][0-9]*$ ]] \
        || die "smoke block/chunk/linear settings must be positive integers"
done
((smoke_block_size % 64 == 0)) \
    || die "SMOKE_BLOCK_SIZE must be divisible by the cuLA checkpoint step 64"
[[ "${smoke_chunkwise_rdma}" == "0" || "${smoke_chunkwise_rdma}" == "1" ]] \
    || die "SMOKE_CHUNKWISE_RDMA must be 0 or 1"

service_pid=
listener_pid=
notified=0

stop_owned_process() {
    local pid="${1:-}"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
    if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null || true
        return 0
    fi
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in {1..10}; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            wait "${pid}" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
}

notify_decode() {
    local verdict="$1"
    local detail="${2:-}"
    local body
    body="$(printf '{\"run_id\":\"%s\",\"status\":\"%s\",\"detail\":\"%s\"}' \
        "${SMOKE_RUN_ID}" "${verdict}" "${detail//\"/}")"
    NO_PROXY="${NO_PROXY:-},${result_host}" \
    no_proxy="${no_proxy:-},${result_host}" \
        curl -fsS --max-time 10 \
        -H 'Content-Type: application/json' \
        --data-binary "${body}" \
        "http://${SMOKE_RESULT_ENDPOINT}/result" >/dev/null
    notified=1
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ "${role}" == "prefill" && "${notified}" == "0" ]]; then
        notify_decode FAIL "prefill-exit-${rc}" || true
    fi
    stop_owned_process "${service_pid}"
    stop_owned_process "${listener_pid}"
    printf 'role=%s\nstatus=%s\ncheckpoint=%s\nartifacts=%s\n' \
        "${role}" "${rc}" "${checkpoint_real}" "${role_dir}" >"${summary_file}"
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_health() {
    local host="$1"
    local port="$2"
    local deadline=$((SECONDS + startup_timeout))
    while ((SECONDS < deadline)); do
        if [[ -n "${service_pid}" ]] && ! kill -0 "${service_pid}" 2>/dev/null; then
            tail -200 "${service_log}" >&2 || true
            die "${role} service exited before health was ready"
        fi
        if NO_PROXY="${NO_PROXY:-},${host}" no_proxy="${no_proxy:-},${host}" \
            curl -fsS --max-time 2 "http://${host}:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    die "timed out waiting for health at ${host}:${port}"
}

wait_for_backend_ranks() {
    local backend_log="${role_dir}/runtime/logs/${role}/main_0.log"
    local expected_ranks=8
    local deadline=$((SECONDS + startup_timeout))
    while ((SECONDS < deadline)); do
        if [[ -n "${service_pid}" ]] && ! kill -0 "${service_pid}" 2>/dev/null; then
            tail -200 "${service_log}" >&2 || true
            [[ -f "${backend_log}" ]] && tail -200 "${backend_log}" >&2 || true
            die "${role} service exited before all backend ranks were ready"
        fi
        if [[ -f "${backend_log}" ]] \
            && grep -Fq "All ${expected_ranks} ranks started successfully" "${backend_log}"; then
            return 0
        fi
        sleep 2
    done
    [[ -f "${backend_log}" ]] && tail -200 "${backend_log}" >&2 || true
    die "timed out waiting for all ${expected_ranks} ${role} backend ranks"
}

wait_for_result_listener() {
    local deadline=$((SECONDS + startup_timeout))
    while ((SECONDS < deadline)); do
        if [[ -n "${service_pid}" ]] && ! kill -0 "${service_pid}" 2>/dev/null; then
            tail -200 "${service_log}" >&2 || true
            die "Prefill service exited while waiting for Decode result listener"
        fi
        if NO_PROXY="${NO_PROXY:-},${result_host}" \
            no_proxy="${no_proxy:-},${result_host}" \
            curl -fsS --max-time 2 \
                "http://${SMOKE_RESULT_ENDPOINT}/ready" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    die "timed out waiting for Decode result listener at ${SMOKE_RESULT_ENDPOINT}"
}

verify_fastsafetensors_log() {
    grep -Eqi 'fastsafetensors' "${service_log}" \
        || die "startup log has no positive FastSafetensors evidence"
    if grep -Eqi '(load_method|loader).*(scratch|fallback)|fallback.*loader' "${service_log}"; then
        die "startup log contains loader fallback evidence"
    fi
}

verify_rdma_log() {
    local engine_log="${role_dir}/runtime/work/${role}/logs/engine.log"
    local evidence_file="${role_dir}/rdma-evidence.txt"
    grep -E 'rdma listen port is .*rdma_mode is \[1\]' "${engine_log}" \
        >"${evidence_file}" \
        || die "startup log has no positive Barex RDMA evidence"
    if grep -Eqi 'rdma mode not supported|BarexRdma backend not supported' \
        "${service_log}" "${engine_log}"; then
        die "startup log reports that the RDMA backend is unavailable"
    fi
}

verify_decode_cuda_graph_capture_log() {
    [[ "${role}" == "decode" ]] || return 0
    local evidence_file="${role_dir}/cuda-graph-evidence.txt"
    local search_paths=(
        "${service_log}"
        "${role_dir}/runtime/logs/decode"
        "${role_dir}/runtime/work/decode/logs"
    )
    : >"${evidence_file}"
    grep -RhsF "CUDA graph capture is enabled" "${search_paths[@]}" \
        >>"${evidence_file}" 2>/dev/null \
        || die "Decode log has no positive CUDA Graph enablement evidence"
    grep -RhsF "capture success for batch size: 4" "${search_paths[@]}" \
        >>"${evidence_file}" 2>/dev/null \
        || die "Decode log has no successful batch-size-4 graph capture evidence"
}

verify_decode_cuda_graph_replay_log() {
    [[ "${role}" == "decode" ]] || return 0
    local evidence_file="${role_dir}/cuda-graph-evidence.txt"
    local search_paths=(
        "${service_log}"
        "${role_dir}/runtime/logs/decode"
        "${role_dir}/runtime/work/decode/logs"
    )
    grep -RhsF "replay end check for batch size 4" "${search_paths[@]}" \
        >>"${evidence_file}" 2>/dev/null \
        || die "Decode log has no successful graph replay check evidence"
}

verify_role_environment() {
    local env_file="${role_dir}/service.env"
    # Never dump the full process environment: it can contain unrelated
    # credentials. Read and persist only the K3 smoke allowlist.
    python3 - \
        "${service_pid}" \
        "${role}" \
        "${env_file}" \
        "${smoke_block_size}" \
        "${smoke_kernel_block_size}" \
        "${smoke_chunk_tokens}" \
        "${smoke_linear_step}" \
        "${smoke_chunkwise_rdma}" \
        "${smoke_prefill_kv_cache_mem_mb}" \
        "${smoke_decode_kv_cache_mem_mb}" \
        "${decode_topology}" <<'PY'
import pathlib
import sys

(
    pid,
    role,
    output,
    block_size,
    kernel_block_size,
    chunk_tokens,
    linear_step,
    chunkwise_rdma,
    prefill_kv_cache_mem_mb,
    decode_kv_cache_mem_mb,
    decode_topology,
) = sys.argv[1:]
entries = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
env = {}
for entry in entries:
    if b"=" in entry:
        key, value = entry.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")

expected = {
    "LOAD_METHOD": "fastsafetensors",
    "MAX_SEQ_LEN": "1048577",
    "MAX_BATCH_TOKENS_SIZE": "1048576",
    "SEQ_SIZE_PER_BLOCK": block_size,
    "KERNEL_SEQ_SIZE_PER_BLOCK": kernel_block_size,
    "CONCURRENCY_LIMIT": "4",
    "MAX_CONTEXT_BATCH_SIZE": "4",
    "REUSE_CACHE": "1",
    "LINEAR_STEP": linear_step,
    "CACHE_STORE_RDMA_MODE": "1",
    "CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS": "2000",
    "KIMI_K3_CHUNKWISE_RDMA": chunkwise_rdma,
    "DSV4_MEGA_MOE_INPUT_PACKER": "fused",
    "DSV4_MEGA_MOE_INPUT_PACKER_IMPL": "optimized",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "ENABLE_CUDA_GRAPH_DEBUG_MODE": "0",
    "FLASHINFER_CUDA_ARCH_LIST": "10.3a",
    "DEEPGEMM_JIT_COMPILER": "auto",
}
absent = ["CUDA_LAUNCH_BLOCKING", "SP_TYPE"]
expected["KIMI_K3_DECODE_TOPOLOGY"] = decode_topology
expected["KIMI_K3_DECODE_KTP"] = (
    "1" if role == "decode" and decode_topology == "dp8_ep8_tp1_ktp8" else "0"
)
absent.append("KIMI_K3_MLA_CACHE_TP")
if role == "prefill":
    expected.update({
        "KV_CACHE_MEM_MB": prefill_kv_cache_mem_mb,
        "MEGA_MOE_MAX_TOKENS_PER_RANK": "8192",
        "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD": "1",
        "KIMI_K3_PREFILL_CHUNK_TOKENS": chunk_tokens,
        "ENABLE_CUDA_GRAPH": "0",
    })
    absent.extend(["RTP_MLA_DECODE_KERNEL", "DECODE_CAPTURE_CONFIG"])
else:
    expected.update({
        "KV_CACHE_MEM_MB": decode_kv_cache_mem_mb,
        "MEGA_MOE_MAX_TOKENS_PER_RANK": "1",
        "RTP_MLA_DECODE_KERNEL": "tokenspeed_mla",
    })
    if decode_topology == "dp8_ep8_tp1_ktp8":
        expected.update({
            "ENABLE_CUDA_GRAPH": "1",
            # One capture key is mandatory: every KTP rank must replay the
            # same collective shapes even when its live DP-local BS differs.
            "DECODE_CAPTURE_CONFIG": "4",
            "KIMI_K3_KTP_LOCAL_BS_BUCKET": "4",
        })
    else:
        expected.update({
            "ENABLE_CUDA_GRAPH": "1",
            "DECODE_CAPTURE_CONFIG": "1,2,3,4",
        })
        absent.append("KIMI_K3_KTP_LOCAL_BS_BUCKET")
    absent.extend([
        "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD",
        "KIMI_K3_PREFILL_CHUNK_TOKENS",
    ])

for key, value in expected.items():
    if env.get(key) != value:
        raise SystemExit(f"{key}={env.get(key)!r}; expected {value!r}")
for key in absent:
    if key in env:
        raise SystemExit(f"{key} must be absent for {role}")

cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
cmdline = [value.decode(errors="replace") for value in cmdline if value]

def option(name):
    try:
        return cmdline[cmdline.index(name) + 1]
    except (ValueError, IndexError):
        raise SystemExit(f"missing {name} in service command line: {cmdline}")

if role == "prefill":
    expected_parallel = {"--tp_size": "8", "--dp_size": "1", "--ep_size": "8", "--ktp_size": "1"}
elif decode_topology == "dp8_ep8_tp1_ktp8":
    expected_parallel = {"--tp_size": "1", "--dp_size": "8", "--ep_size": "8", "--ktp_size": "8"}
else:
    expected_parallel = {"--tp_size": "8", "--dp_size": "1", "--ep_size": "8", "--ktp_size": "1"}
for name, value in expected_parallel.items():
    actual = option(name)
    if actual != value:
        raise SystemExit(f"{name}={actual!r}; expected {value!r}")

lines = [f"{key}={env[key]}" for key in sorted(expected)]
lines.extend(f"ARGV_{key[2:].upper()}={value}" for key, value in sorted(expected_parallel.items()))
lines.extend(f"{key}=<unset>" for key in sorted(absent))
pathlib.Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

# Operator versions are supplied by the Bazel server target. Keep the exact
# validated BF16 1M runtime profile here instead of relying on the caller's
# shell or on the generic launcher's conservative defaults.
apply_validated_common_profile() {
    local run_hash
    run_hash="$(printf '%s' "${SMOKE_RUN_ID}" | sha256sum)"
    run_hash="${run_hash%% *}"
    export CHECKPOINT_PATH="${checkpoint_real}"
    export TOKENIZER_PATH="${checkpoint_real}"
    export PREFILL_ENDPOINT DECODE_ENDPOINT
    export KIMI_K3_DECODE_TOPOLOGY="${decode_topology}"
    unset KIMI_K3_MLA_CACHE_TP
    export LOAD_METHOD=fastsafetensors
    export MAX_SEQ_LEN=1048577
    export MAX_BATCH_TOKENS_SIZE=1048576
    export SEQ_SIZE_PER_BLOCK="${smoke_block_size}"
    export KERNEL_SEQ_SIZE_PER_BLOCK="${smoke_kernel_block_size}"
    export CONCURRENCY_LIMIT=4
    export MAX_CONTEXT_BATCH_SIZE=4
    export REUSE_CACHE=1
    export LINEAR_STEP="${smoke_linear_step}"
    export CACHE_STORE_RDMA_MODE=1
    export CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS=2000
    export KIMI_K3_CHUNKWISE_RDMA="${smoke_chunkwise_rdma}"
    export DSV4_MEGA_MOE_INPUT_PACKER=fused
    export DSV4_MEGA_MOE_INPUT_PACKER_IMPL=optimized
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export ENABLE_CUDA_GRAPH_DEBUG_MODE=0
    export FLASHINFER_CUDA_ARCH_LIST=10.3a
    export DEEPGEMM_JIT_COMPILER=auto
    # This acceptance run isolates Decode CUDA Graph. MTP/Eagle3 is a separate
    # follow-up matrix and must not leak in from the caller's environment.
    unset SP_TYPE
    export RTP_LLM_SERVICE_ID="kimi-k3-full-pd-${SMOKE_RUN_ID}"
    # Keep TP Unix-domain sockets below Linux's 107-byte path limit even when
    # the externally visible run ID is descriptive and long.
    export RTP_LLM_TMPDIR="/tmp/k3pd-${run_hash:0:12}-${role}"
    export RUN_ROOT="${role_dir}/runtime"

    # Canonical smoke runs asynchronously and uses the operator versions from
    # the Bazel runfiles rather than an inherited debugging overlay.
    unset CUDA_LAUNCH_BLOCKING OPS_OVERLAY
}

apply_validated_prefill_profile() {
    export KV_CACHE_MEM_MB="${smoke_prefill_kv_cache_mem_mb}"
    export MEGA_MOE_MAX_TOKENS_PER_RANK=8192
    export KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD=1
    export KIMI_K3_PREFILL_CHUNK_TOKENS="${smoke_chunk_tokens}"
    export ENABLE_CUDA_GRAPH=0
    unset RTP_MLA_DECODE_KERNEL DECODE_CAPTURE_CONFIG
}

apply_validated_decode_profile() {
    export KV_CACHE_MEM_MB="${smoke_decode_kv_cache_mem_mb}"
    export MEGA_MOE_MAX_TOKENS_PER_RANK=1
    unset KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD KIMI_K3_PREFILL_CHUNK_TOKENS
    if [[ "${decode_topology}" == "dp8_ep8_tp1_ktp8" ]]; then
        # All DP workers capture the same local-BS=4 KTP collective schedule;
        # smaller live batches replay this graph with framework dummy rows.
        export ENABLE_CUDA_GRAPH=1
        export DECODE_CAPTURE_CONFIG=4
        export KIMI_K3_KTP_LOCAL_BS_BUCKET=4
    else
        # Leave headroom for CUDA graph capture with the validated TP8 baseline.
        export ENABLE_CUDA_GRAPH=1
        export DECODE_CAPTURE_CONFIG=1,2,3,4
        unset KIMI_K3_KTP_LOCAL_BS_BUCKET
    fi
    export RTP_MLA_DECODE_KERNEL=tokenspeed_mla
}

apply_validated_common_profile
if [[ "${role}" == "prefill" ]]; then
    apply_validated_prefill_profile
else
    apply_validated_decode_profile
fi

echo "[${role}] artifacts=${role_dir}"
echo "[${role}] checkpoint=${checkpoint_real} (${checkpoint_fs}:${checkpoint_source})"
echo "[${role}] endpoints prefill=${PREFILL_ENDPOINT} decode=${DECODE_ENDPOINT}"
echo "[${role}] decode_topology=${decode_topology} prefill_mla_width=576 replicated=1"

setsid "${launcher}" "${role}" >"${service_log}" 2>&1 &
service_pid=$!
disown "${service_pid}" 2>/dev/null || true
printf '%s\n' "${service_pid}" >"${role_dir}/service.pid"

local_port="${prefill_port}"
[[ "${role}" == "prefill" ]] || local_port="${decode_port}"
wait_for_health 127.0.0.1 "${local_port}"
wait_for_backend_ranks
verify_fastsafetensors_log
verify_rdma_log
verify_role_environment
verify_decode_cuda_graph_capture_log

if [[ "${role}" == "decode" ]]; then
    # One-shot result endpoint. It accepts only the matching run ID and writes
    # PASS/FAIL to a local file; no remote shell or shared filesystem is used.
    python3 - "${result_port}" "${result_file}" "${SMOKE_RUN_ID}" <<'PY' &
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[1])
result_file = pathlib.Path(sys.argv[2])
expected_run_id = sys.argv[3]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/ready":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ready\n")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            if self.path != "/result" or data.get("run_id") != expected_run_id:
                raise ValueError("unexpected path or run_id")
            status = data.get("status")
            if status not in ("PASS", "FAIL"):
                raise ValueError("invalid status")
            result_file.write_text(status + "\n", encoding="utf-8")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok\n")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        except Exception as exc:
            self.send_response(400)
            self.end_headers()
            self.wfile.write((str(exc) + "\n").encode())

server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
server.serve_forever()
PY
    listener_pid=$!
    disown "${listener_pid}" 2>/dev/null || true
    echo "[decode] READY; waiting for Prefill result at ${SMOKE_RESULT_ENDPOINT}"
    deadline=$((SECONDS + result_timeout))
    while ((SECONDS < deadline)); do
        kill -0 "${service_pid}" 2>/dev/null \
            || die "Decode service exited while waiting for Prefill"
        [[ -f "${result_file}" ]] && break
        sleep 1
    done
    [[ -f "${result_file}" ]] || die "timed out waiting for Prefill result"
    verdict="$(tr -d '[:space:]' <"${result_file}")"
    [[ "${verdict}" == "PASS" ]] || die "Prefill reported ${verdict}"
    verify_decode_cuda_graph_replay_log
    echo "PASS: Decode stayed healthy and Prefill validated the PD response and semantic accuracy"
    exit 0
fi

# Prefill is the request/validation side. Confirm the remote Decode endpoint
# before issuing OpenAI-compatible requests through local Prefill. The runner
# also checks local Prefill health before every sequential/concurrent stage.
wait_for_health "${decode_host}" "${decode_port}"
wait_for_result_listener
max_tokens="${SMOKE_MAX_TOKENS:-32}"
[[ "${max_tokens}" =~ ^[1-9][0-9]*$ ]] \
    || die "SMOKE_MAX_TOKENS must be a positive integer"
smoke_suite="${SMOKE_SUITE:-all}"
case "${smoke_suite}" in
    flow | all) ;;
    *) die "SMOKE_SUITE must be flow or all" ;;
esac

python3 "${case_runner}" \
    --base-url "http://127.0.0.1:${prefill_port}" \
    --decode-health-url "http://${decode_host}:${decode_port}/health" \
    --output "${accuracy_file}" \
    --suite "${smoke_suite}" \
    --namespace "${SMOKE_RUN_ID}" \
    --batch-size 4 \
    --block-size "${SEQ_SIZE_PER_BLOCK}" \
    --chunk-tokens "${smoke_chunk_tokens}" \
    --max-tokens "${max_tokens}" \
    --timeout "${request_timeout}"

notify_decode PASS "smoke-suite-${smoke_suite}-validated"
echo "PASS: Prefill validated suite=${smoke_suite}; artifacts=${role_dir}"
exit 0
