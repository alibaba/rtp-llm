#!/usr/bin/env bash
#
# Recommended one-command startup from a controller with SSH access to both
# hosts (the driver enters lhc_GPU, waits for Decode readiness, then launches
# Prefill):
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
#    PREFILL_SP_CHECKPOINT_PATH=/data3/user/Kimi-K3-Eagle3 \
#    DECODE_SP_CHECKPOINT_PATH=/data0/user/Kimi-K3-Eagle3 \
#    PREFILL_ENDPOINT=xx.xx.xx.xx:27188 \
#    DECODE_ENDPOINT=xx.xx.xx.xx:28188 \
#    SMOKE_RUN_ID=my-run \
#    SMOKE_SUITE=all \
#    python3 ./example/k3/kimi_k3_full_model_two_host_pd_smoke_driver.py
#
# Manual role startup remains available for debugging. Run from the RTP-LLM
# repository root inside lhc_GPU. Start Decode first and wait for its health
# endpoint before starting Prefill, matching the controller driver:
#
# 1. Start the Decode role:
#    CHECKPOINT_PATH=/ssd/2/kimi-k3 \
#    SP_CHECKPOINT_PATH=/ssd/2/kimi-k3-eagle3 \
#    PREFILL_ENDPOINT=xx.xx.xx.xx:27188 \
#    DECODE_ENDPOINT=xx.xx.xx.xx:28188 \
#    SMOKE_RUN_ID=my-run \
#    SMOKE_SUITE=all \
#    ./example/k3/kimi_k3_full_model_two_host_pd_smoke.sh decode
#
# 2. Start Prefill with the same endpoints and run ID (it waits for both the
#    Decode model and Decode result listener to become ready):
#    CHECKPOINT_PATH=/ssd/2/kimi-k3 \
#    SP_CHECKPOINT_PATH=/ssd/2/kimi-k3-eagle3 \
#    PREFILL_ENDPOINT=xx.xx.xx.xx:27188 \
#    DECODE_ENDPOINT=xx.xx.xx.xx:28188 \
#    SMOKE_RUN_ID=my-run \
#    SMOKE_SUITE=all \
#    ./example/k3/kimi_k3_full_model_two_host_pd_smoke.sh prefill
#
# Lightweight two-host Kimi K3 full-model (93-layer) PD smoke.
#
# Run this same role script inside lhc_GPU on both hosts. There is no committed
# machine address; the optional driver accepts all SSH/host paths at runtime
# and enforces Decode-ready-before-Prefill ordering.
# The validated profile always enables Barex RDMA on both roles; this smoke is
# intentionally not a TCP/cache-store fallback test.
# Merge-gate runs must use SMOKE_SUITE=all; it is the only supported suite.
# Prefill checks both services, runs the complete request suite, validates model
# answers and cache metadata, then reports PASS/FAIL back to Decode. Both
# commands therefore have a meaningful exit status and clean only their own
# process group.
# The full-model profile always enables Kimi K3 Eagle3 MTP on both roles. The
# role-local draft checkpoint is mandatory; there is no non-MTP fallback.

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
  SP_CHECKPOINT_PATH=/local/path/to/Kimi-K3-Eagle3 \
  PREFILL_ENDPOINT=prefill-host:27188 \
  DECODE_ENDPOINT=decode-host:28188 \
  SMOKE_RUN_ID=my-run \
  example/k3/kimi_k3_full_model_two_host_pd_smoke.sh decode|prefill

The two roles may start concurrently. Prefill waits for both the Decode model
and result channel. The default result channel is DECODE host at DECODE port +
100; override SMOKE_RESULT_ENDPOINT on both hosts when that port is unavailable.

The validated BF16 1M model/runtime profile is fixed by this smoke and always
uses Eagle3 MTP. Only host, target/draft checkpoint, artifact, timeout and
prebuilt-launcher settings are configurable.

Merge-gate accuracy validation must use SMOKE_SUITE=all. SMOKE_SUITE=flow is
only a four-layer RDMA connectivity/multi-round preflight and does not satisfy
the final acceptance requirement.

Important optional variables:
  SMOKE_ARTIFACT_ROOT       defaults to /tmp/kimi-k3-two-host-pd-smoke
  SMOKE_STARTUP_TIMEOUT_S   defaults to 14400
  SMOKE_REQUEST_TIMEOUT_S   defaults to 900
  SMOKE_RESULT_TIMEOUT_S    defaults to 18000
  SMOKE_RESULT_ENDPOINT     defaults to decode-host:(decode-port + 100)
  SMOKE_MAX_TOKENS          defaults to 256 for ordinary cache cases
  SMOKE_IDENTITY_MAX_TOKENS defaults to 256 for the reasoning identity case
  SMOKE_SINGLE_EXACT_MAX_TOKENS
                            defaults to 128 for exact-cache seed/hit answers
  SMOKE_MTP_CHUNK_MAX_TOKENS
                            defaults to 128 for MTP chunk-Prefill coverage
  SMOKE_RDMA_PREWARM_ATTEMPTS
                            defaults to 3 bounded batch-sized prewarm attempts
  SMOKE_RDMA_PREWARM_BACKOFF_S
                            defaults to 5 seconds between failed attempts
  SMOKE_RDMA_PREWARM_SETTLE_S
                            defaults to 2 seconds after a successful prewarm
  SMOKE_ACCL_USE_NICS       optional comma-separated Barex HCA allowlist.
                            When unset, mlx5_bond_0..7 are used only if all
                            are present and active; otherwise ACCL_USE_NICS is
                            left unset for Barex auto-discovery. An explicit
                            allowlist remains strict. Both roles must use the
                            same explicit order.
  SMOKE_SUITE               all (default) or flow
                            flow: four-layer-friendly multi-round RDMA flow
                                  check without semantic-answer assertions
                            all: identity, single miss/hit, partial hit,
                                 concurrent all-miss/all-hit, mixed hit+miss
                                 batches, MTP acceptance after chunk Prefill,
                                 and >64K single/batched chunk cases
  SMOKE_EXPECTED_LAYERS     checkpoint layer count; defaults to 93. Set to 4
                            only for the required four-layer RDMA flow smoke.
  SMOKE_BLOCK_SIZE          physical cache page size; defaults to 4096
  SMOKE_KERNEL_BLOCK_SIZE   attention kernel page size; defaults to 128
  SMOKE_CHUNK_TOKENS        whole-model chunk budget; defaults to 65536
  SMOKE_LINEAR_STEP         KDA materialization step; defaults to 1
  SMOKE_CHUNKWISE_RDMA      1 (default) enables Layer x Chunk publication;
                            0 retains compute-all-then-transfer behavior
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
sp_checkpoint_real="$(realpath -e "${SP_CHECKPOINT_PATH:?SP_CHECKPOINT_PATH is required}")" \
    || die "Eagle3 checkpoint does not exist: ${SP_CHECKPOINT_PATH}"
case "${sp_checkpoint_real}" in
    /data[0-9]*/* | /data/* | /ssd/*) ;;
    *) die "Eagle3 checkpoint must be on a local data disk: ${sp_checkpoint_real}" ;;
esac
sp_checkpoint_fs="$(findmnt -T "${sp_checkpoint_real}" -n -o FSTYPE)"
sp_checkpoint_source="$(findmnt -T "${sp_checkpoint_real}" -n -o SOURCE)"
case "${sp_checkpoint_fs}:${sp_checkpoint_source}" in
    nfs*:* | cifs:* | smb*:* | fuse.*:* | *[Nn][Aa][Ss]*)
        die "network/NAS Eagle3 checkpoint is forbidden: ${sp_checkpoint_fs}:${sp_checkpoint_source}"
        ;;
esac
[[ -f "${sp_checkpoint_real}/config.json" ]] \
    || die "missing Eagle3 checkpoint config.json"
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
case "${smoke_expected_layers}" in
    93) smoke_eagle3_aux_layer_ids=0,44,88 ;;
    4) smoke_eagle3_aux_layer_ids=0,1,3 ;;
    *)
        die "no validated Eagle3 aux-layer profile for ${smoke_expected_layers} target layers"
        ;;
esac

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
smoke_rdma_prewarm_attempts="${SMOKE_RDMA_PREWARM_ATTEMPTS:-3}"
smoke_rdma_prewarm_backoff_s="${SMOKE_RDMA_PREWARM_BACKOFF_S:-5}"
smoke_rdma_prewarm_settle_s="${SMOKE_RDMA_PREWARM_SETTLE_S:-2}"
smoke_accl_use_nics=""
smoke_accl_use_nics_mode="auto-discovery"
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
[[ "${smoke_rdma_prewarm_attempts}" =~ ^[0-9]+$ ]] \
    || die "SMOKE_RDMA_PREWARM_ATTEMPTS must be a non-negative integer"
for seconds_value in \
    "${smoke_rdma_prewarm_backoff_s}" \
    "${smoke_rdma_prewarm_settle_s}"; do
    [[ "${seconds_value}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || die "RDMA prewarm backoff/settle values must be non-negative numbers"
done

resolve_rdma_nic_allowlist() {
    local selector="${repo_root}/example/k3/kimi_k3_full_model_pd_nic_selection.py"
    [[ -f "${selector}" ]] || die "RDMA HCA selector is missing: ${selector}"
    if [[ "${SMOKE_ACCL_USE_NICS+x}" == "x" ]]; then
        [[ -n "${SMOKE_ACCL_USE_NICS}" ]] \
            || die "SMOKE_ACCL_USE_NICS must not be empty when explicitly set"
        smoke_accl_use_nics="$(python3 "${selector}" --explicit "${SMOKE_ACCL_USE_NICS}")" \
            || die "explicit SMOKE_ACCL_USE_NICS validation failed"
        smoke_accl_use_nics_mode="explicit"
    else
        smoke_accl_use_nics="$(python3 "${selector}")" \
            || die "default RDMA HCA discovery failed"
        if [[ -n "${smoke_accl_use_nics}" ]]; then
            smoke_accl_use_nics_mode="default-bond"
        fi
    fi

    if [[ -n "${smoke_accl_use_nics}" ]]; then
        echo "using ${smoke_accl_use_nics_mode} RDMA HCA allowlist: ${smoke_accl_use_nics}"
    else
        echo "using Barex RDMA HCA auto-discovery (ACCL_USE_NICS unset)"
    fi
}

resolve_rdma_nic_allowlist

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

verify_rdma_selected_devices() {
    local engine_log="${role_dir}/runtime/work/${role}/logs/engine.log"
    local evidence_file="${role_dir}/rdma-selected-devices.txt"
    python3 - "${engine_log}" "${smoke_accl_use_nics}" "${evidence_file}" <<'PY'
import pathlib
import re
import sys

engine_log, allowlist, output = sys.argv[1:]
allowed = set(allowlist.split(",")) if allowlist else None
selected = []
for line in pathlib.Path(engine_log).read_text(
    encoding="utf-8", errors="replace"
).splitlines():
    if "XContextImpl::SpawnChannel" not in line:
        continue
    match = re.search(r"device=\[IbvDevice@.*?\bname=([^\]]+)\]", line)
    if match:
        selected.append(match.group(1))
if not selected:
    raise SystemExit("no Barex SpawnChannel device evidence was recorded")
unexpected = sorted(set(selected) - allowed) if allowed is not None else []
if unexpected:
    raise SystemExit(f"Barex selected HCAs outside ACCL_USE_NICS: {unexpected}")
counts = {nic: selected.count(nic) for nic in sorted(set(selected))}
pathlib.Path(output).write_text(
    "ACCL_USE_NICS=" + (allowlist or "<unset>") + "\n"
    + "selected=" + ",".join(f"{nic}:{count}" for nic, count in counts.items()) + "\n",
    encoding="utf-8",
)
PY
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
        "${sp_checkpoint_real}" \
        "${smoke_eagle3_aux_layer_ids}" \
        "${smoke_accl_use_nics}" <<'PY'
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
    sp_checkpoint_path,
    eagle3_aux_layer_ids,
    accl_use_nics,
) = sys.argv[1:]
entries = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
env = {}
for entry in entries:
    if b"=" in entry:
        key, value = entry.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")

expected = {
    "LOAD_METHOD": "fastsafetensors",
    "SEQ_SIZE_PER_BLOCK": block_size,
    "KERNEL_SEQ_SIZE_PER_BLOCK": kernel_block_size,
    "CONCURRENCY_LIMIT": "32",
    "MAX_CONTEXT_BATCH_SIZE": "1",
    "LINEAR_STEP": linear_step,
    "CACHE_STORE_RDMA_MODE": "1",
    "CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS": "30000",
    "KIMI_K3_CHUNKWISE_RDMA": chunkwise_rdma,
    "DSV4_MEGA_MOE_INPUT_PACKER": "fused",
    "DSV4_MEGA_MOE_INPUT_PACKER_IMPL": "optimized",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "ENABLE_CUDA_GRAPH_DEBUG_MODE": "0",
    "FLASHINFER_CUDA_ARCH_LIST": "10.3a",
    "DEEPGEMM_JIT_COMPILER": "auto",
    "FT_CORE_DUMP_ON_EXCEPTION": "1",
    "SP_TYPE": "eagle3",
    "SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3",
    "SP_CHECKPOINT_PATH": sp_checkpoint_path,
    "SP_ACT_TYPE": "BF16",
    "GEN_NUM_PER_CIRCLE": "3",
    "KIMI_K3_EAGLE3_AUX_LAYER_IDS": eagle3_aux_layer_ids,
}
absent = ["CUDA_LAUNCH_BLOCKING", "large_segment_size_mb"]
if accl_use_nics:
    expected["ACCL_USE_NICS"] = accl_use_nics
else:
    absent.append("ACCL_USE_NICS")
if role == "prefill":
    expected.update({
        "MAX_SEQ_LEN": "1258294",
        "MAX_BATCH_TOKENS_SIZE": "1258291",
        "KV_CACHE_MEM_MB": "42000",
        "REUSE_CACHE": "1",
        "KIMI_K3_KDA_POOL_BLOCKS": "0",
        "RESERVER_RUNTIME_MEM_MB": "15000",
        "MEGA_MOE_MAX_TOKENS_PER_RANK": "8192",
        "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD": "1",
        "KIMI_K3_PREFILL_CHUNK_TOKENS": chunk_tokens,
        "ENABLE_CUDA_GRAPH": "0",
        "ENABLE_MEMORY_CACHE": "1",
        "MEMORY_CACHE_SIZE_MB": "65536",
    })
    absent.extend(["RTP_MLA_DECODE_KERNEL", "DECODE_CAPTURE_CONFIG", "MOE_STRATEGY"])
else:
    expected.update({
        "MAX_SEQ_LEN": "1468006",
        "MAX_BATCH_TOKENS_SIZE": "1468006",
        "KV_CACHE_MEM_MB": "42000",
        "REUSE_CACHE": "0",
        "KIMI_K3_KDA_POOL_BLOCKS": "112",
        "RESERVER_RUNTIME_MEM_MB": "8000",
        "MEGA_MOE_MAX_TOKENS_PER_RANK": "16",
        "ENABLE_CUDA_GRAPH": "1",
        "DECODE_CAPTURE_CONFIG": "1,2,3,4",
        "RTP_MLA_DECODE_KERNEL": "tokenspeed_mla",
        "MOE_STRATEGY": "mega_moe_se",
        "RTP_LLM_DEVICE_INPUT": "1",
        "RTP_LLM_DROP_BROAD_SYNC": "1",
        "RTP_LLM_STREAM_ASYNC": "1",
    })
    absent.extend([
        "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD",
        "KIMI_K3_PREFILL_CHUNK_TOKENS",
        "ENABLE_MEMORY_CACHE",
        "MEMORY_CACHE_SIZE_MB",
    ])

for key, value in expected.items():
    if env.get(key) != value:
        raise SystemExit(f"{key}={env.get(key)!r}; expected {value!r}")
for key in absent:
    if key in env:
        raise SystemExit(f"{key} must be absent for {role}")

lines = [f"{key}={env[key]}" for key in sorted(expected)]
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
    export LOAD_METHOD=fastsafetensors
    export SEQ_SIZE_PER_BLOCK="${smoke_block_size}"
    export KERNEL_SEQ_SIZE_PER_BLOCK="${smoke_kernel_block_size}"
    export CONCURRENCY_LIMIT=32
    export MAX_CONTEXT_BATCH_SIZE=1
    export LINEAR_STEP="${smoke_linear_step}"
    export CACHE_STORE_RDMA_MODE=1
    export CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS=30000
    if [[ -n "${smoke_accl_use_nics}" ]]; then
        export ACCL_USE_NICS="${smoke_accl_use_nics}"
    else
        unset ACCL_USE_NICS
    fi
    export KIMI_K3_CHUNKWISE_RDMA="${smoke_chunkwise_rdma}"
    export DSV4_MEGA_MOE_INPUT_PACKER=fused
    export DSV4_MEGA_MOE_INPUT_PACKER_IMPL=optimized
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export ENABLE_CUDA_GRAPH_DEBUG_MODE=0
    export FLASHINFER_CUDA_ARCH_LIST=10.3a
    export DEEPGEMM_JIT_COMPILER=auto
    export FT_CORE_DUMP_ON_EXCEPTION=1
    export SP_TYPE=eagle3
    export SP_MODEL_TYPE=kimi_k3_mla_swa_eagle3
    export SP_CHECKPOINT_PATH="${sp_checkpoint_real}"
    export SP_ACT_TYPE=BF16
    export GEN_NUM_PER_CIRCLE=3
    export KIMI_K3_EAGLE3_AUX_LAYER_IDS="${smoke_eagle3_aux_layer_ids}"
    export RTP_LLM_SERVICE_ID="kimi-k3-full-pd-${SMOKE_RUN_ID}"
    # Keep TP Unix-domain sockets below Linux's 107-byte path limit even when
    # the externally visible run ID is descriptive and long.
    export RTP_LLM_TMPDIR="/tmp/k3pd-${run_hash:0:12}-${role}"
    export RUN_ROOT="${role_dir}/runtime"

    # Canonical smoke runs asynchronously and uses the operator versions from
    # the Bazel runfiles rather than an inherited debugging overlay.
    unset CUDA_LAUNCH_BLOCKING OPS_OVERLAY
    unset large_segment_size_mb
}

apply_validated_prefill_profile() {
    export MAX_SEQ_LEN=1258294
    export MAX_BATCH_TOKENS_SIZE=1258291
    export KV_CACHE_MEM_MB=42000
    export REUSE_CACHE=1
    export KIMI_K3_KDA_POOL_BLOCKS=0
    export RESERVER_RUNTIME_MEM_MB=15000
    export MEGA_MOE_MAX_TOKENS_PER_RANK=8192
    export KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD=1
    export KIMI_K3_PREFILL_CHUNK_TOKENS="${smoke_chunk_tokens}"
    export ENABLE_CUDA_GRAPH=0
    export ENABLE_MEMORY_CACHE=1
    export MEMORY_CACHE_SIZE_MB=65536
    unset RTP_MLA_DECODE_KERNEL DECODE_CAPTURE_CONFIG MOE_STRATEGY
}

apply_validated_decode_profile() {
    export MAX_SEQ_LEN=1468006
    export MAX_BATCH_TOKENS_SIZE=1468006
    export KV_CACHE_MEM_MB=42000
    export REUSE_CACHE=0
    export KIMI_K3_KDA_POOL_BLOCKS=112
    export RESERVER_RUNTIME_MEM_MB=8000
    export MEGA_MOE_MAX_TOKENS_PER_RANK=16
    unset KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD KIMI_K3_PREFILL_CHUNK_TOKENS
    unset ENABLE_MEMORY_CACHE MEMORY_CACHE_SIZE_MB
    export ENABLE_CUDA_GRAPH=1
    # The smoke issues four concurrent requests. Capture every possible
    # coalesced Decode batch size instead of aborting above batch size one.
    export DECODE_CAPTURE_CONFIG=1,2,3,4
    export RTP_MLA_DECODE_KERNEL=tokenspeed_mla
    export MOE_STRATEGY=mega_moe_se
    export RTP_LLM_DEVICE_INPUT=1
    export RTP_LLM_DROP_BROAD_SYNC=1
    export RTP_LLM_STREAM_ASYNC=1
}

apply_validated_common_profile
if [[ "${role}" == "prefill" ]]; then
    apply_validated_prefill_profile
else
    apply_validated_decode_profile
fi

echo "[${role}] artifacts=${role_dir}"
echo "[${role}] checkpoint=${checkpoint_real} (${checkpoint_fs}:${checkpoint_source})"
echo "[${role}] eagle3_checkpoint=${sp_checkpoint_real} (${sp_checkpoint_fs}:${sp_checkpoint_source})"
echo "[${role}] endpoints prefill=${PREFILL_ENDPOINT} decode=${DECODE_ENDPOINT}"

setsid "${launcher}" "${role}" >"${service_log}" 2>&1 &
service_pid=$!
printf '%s\n' "${service_pid}" >"${role_dir}/service.pid"

local_port="${prefill_port}"
[[ "${role}" == "prefill" ]] || local_port="${decode_port}"
wait_for_health 127.0.0.1 "${local_port}"
verify_fastsafetensors_log
verify_rdma_log
verify_role_environment

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
    verify_rdma_selected_devices
    echo "PASS: Decode stayed healthy and Prefill validated the PD response and semantic accuracy"
    exit 0
fi

# Prefill is the request/validation side. Confirm the remote Decode endpoint
# before issuing OpenAI-compatible requests through local Prefill. The runner
# also checks local Prefill health before every sequential/concurrent stage.
wait_for_health "${decode_host}" "${decode_port}"
wait_for_result_listener
max_tokens="${SMOKE_MAX_TOKENS:-256}"
identity_max_tokens="${SMOKE_IDENTITY_MAX_TOKENS:-256}"
single_exact_max_tokens="${SMOKE_SINGLE_EXACT_MAX_TOKENS:-128}"
mtp_chunk_max_tokens="${SMOKE_MTP_CHUNK_MAX_TOKENS:-128}"
for token_budget in \
    "${max_tokens}" \
    "${identity_max_tokens}" \
    "${single_exact_max_tokens}" \
    "${mtp_chunk_max_tokens}"; do
    [[ "${token_budget}" =~ ^[1-9][0-9]*$ ]] \
        || die "smoke output token budgets must be positive integers"
done
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
    --identity-max-tokens "${identity_max_tokens}" \
    --single-exact-max-tokens "${single_exact_max_tokens}" \
    --mtp-chunk-max-tokens "${mtp_chunk_max_tokens}" \
    --rdma-prewarm-attempts "${smoke_rdma_prewarm_attempts}" \
    --rdma-prewarm-backoff-s "${smoke_rdma_prewarm_backoff_s}" \
    --rdma-prewarm-settle-s "${smoke_rdma_prewarm_settle_s}" \
    --timeout "${request_timeout}"

verify_rdma_selected_devices
notify_decode PASS "smoke-suite-${smoke_suite}-validated"
echo "PASS: Prefill validated suite=${smoke_suite}; artifacts=${role_dir}"
exit 0
