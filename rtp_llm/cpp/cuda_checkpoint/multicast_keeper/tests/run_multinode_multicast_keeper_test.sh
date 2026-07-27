#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -ge 6 ]] || {
    echo "internal usage: $0 KEEPER_LAUNCHER HOLDER CREATOR SHIM WORKER {preflight|run}" >&2
    exit 2
}

KEEPER_LAUNCHER="$1"
HOLDER="$2"
CREATOR="$3"
SHIM="$4"
WORKER="$5"
ACTION="$6"

usage() {
    cat <<'EOF'
Run on every node with the same values except NODE_RANK and LOCAL_GPUS:

  RTP_MC_TEST_JOB_ID=gb300-nvls-001 \
  RTP_MC_TEST_ROLE=prefill \
  RTP_MC_TEST_NODE_RANK=0 \
  RTP_MC_TEST_NNODES=2 \
  RTP_MC_TEST_GLOBAL_TEAM_SIZE=4 \
  RTP_MC_TEST_LOCAL_GPUS=0,1 \
  MASTER_ADDR=10.0.0.10 MASTER_PORT=39420 \
    bazel run //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:multinode_multicast_keeper_test \
      --config=cuda13 -- preflight

Replace preflight with run after preflight passes on both nodes. Node 1 uses
RTP_MC_TEST_NODE_RANK=1. The default failure injection kills node 1's holder
after three successful destroy/rebuild rounds.

Required environment:
  RTP_MC_TEST_JOB_ID             Unique test run identifier
  RTP_MC_TEST_ROLE               Role/team name; use a different port per role
  RTP_MC_TEST_NODE_RANK          Contiguous node rank, starting at zero
  RTP_MC_TEST_NNODES             Number of nodes (this runner requires >= 2)
  RTP_MC_TEST_GLOBAL_TEAM_SIZE   Explicit global multicast/C10d GPU count
  RTP_MC_TEST_LOCAL_GPUS         Ordered physical CUDA ordinals for this node
  MASTER_ADDR, MASTER_PORT       Reachable C10d rendezvous endpoint

Optional environment:
  RTP_MC_TEST_SUCCESS_ROUNDS=3
  RTP_MC_TEST_FAIL_HOLDER_NODE=1
  RTP_MC_TEST_BASE_DIR=/tmp/rtp_mc_multinode
  RTP_MC_TEST_IMEX_CHANNEL=/dev/nvidia-caps-imex-channels/channel0
  RTP_MC_TEST_REQUIRE_IMEX=1
  RTP_MC_TEST_REQUIRE_FABRIC=1
  RTP_MC_TEST_REQUIRE_NCCL_NVLS_LOG=1
  PYTHON_BIN=/opt/conda310/bin/python3
EOF
}

[[ "${ACTION}" == "preflight" || "${ACTION}" == "run" ]] || {
    usage >&2
    exit 2
}

PYTHON_BIN="${PYTHON_BIN:-/opt/conda310/bin/python3}"
JOB_ID="${RTP_MC_TEST_JOB_ID:-}"
ROLE="${RTP_MC_TEST_ROLE:-}"
NODE_RANK="${RTP_MC_TEST_NODE_RANK:-}"
NNODES="${RTP_MC_TEST_NNODES:-}"
GLOBAL_TEAM_SIZE="${RTP_MC_TEST_GLOBAL_TEAM_SIZE:-}"
LOCAL_GPUS="${RTP_MC_TEST_LOCAL_GPUS:-}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-}"
SUCCESS_ROUNDS="${RTP_MC_TEST_SUCCESS_ROUNDS:-3}"
FAILURE_NODE="${RTP_MC_TEST_FAIL_HOLDER_NODE:-1}"
BASE_DIR="${RTP_MC_TEST_BASE_DIR:-/tmp/rtp_mc_multinode}"
IMEX_CHANNEL="${RTP_MC_TEST_IMEX_CHANNEL:-/dev/nvidia-caps-imex-channels/channel0}"
REQUIRE_IMEX="${RTP_MC_TEST_REQUIRE_IMEX:-1}"
REQUIRE_FABRIC="${RTP_MC_TEST_REQUIRE_FABRIC:-1}"
REQUIRE_NCCL_NVLS_LOG="${RTP_MC_TEST_REQUIRE_NCCL_NVLS_LOG:-1}"

die() {
    echo "multinode multicast test: $*" >&2
    exit 2
}

require_uint() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an unsigned integer"
}

[[ "${JOB_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "RTP_MC_TEST_JOB_ID is required and must be path-safe"
[[ "${ROLE}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "RTP_MC_TEST_ROLE is required and must be path-safe"
[[ -n "${NODE_RANK}" ]] || die "RTP_MC_TEST_NODE_RANK is required"
[[ -n "${NNODES}" ]] || die "RTP_MC_TEST_NNODES is required"
[[ -n "${GLOBAL_TEAM_SIZE}" ]] || die "RTP_MC_TEST_GLOBAL_TEAM_SIZE is required"
[[ -n "${LOCAL_GPUS}" ]] || die "RTP_MC_TEST_LOCAL_GPUS is required"
[[ -n "${MASTER_ADDR}" ]] || die "MASTER_ADDR is required"
[[ -n "${MASTER_PORT}" ]] || die "MASTER_PORT is required"
require_uint RTP_MC_TEST_NODE_RANK "${NODE_RANK}"
require_uint RTP_MC_TEST_NNODES "${NNODES}"
require_uint RTP_MC_TEST_GLOBAL_TEAM_SIZE "${GLOBAL_TEAM_SIZE}"
require_uint MASTER_PORT "${MASTER_PORT}"
require_uint RTP_MC_TEST_SUCCESS_ROUNDS "${SUCCESS_ROUNDS}"
require_uint RTP_MC_TEST_FAIL_HOLDER_NODE "${FAILURE_NODE}"
(( NNODES >= 2 )) || die "RTP_MC_TEST_NNODES must be at least 2"
(( NODE_RANK < NNODES )) || die "RTP_MC_TEST_NODE_RANK must be less than RTP_MC_TEST_NNODES"
(( SUCCESS_ROUNDS >= 2 )) || die "RTP_MC_TEST_SUCCESS_ROUNDS must be at least 2"
(( FAILURE_NODE < NNODES )) || die "RTP_MC_TEST_FAIL_HOLDER_NODE must identify a participating node"
(( MASTER_PORT + SUCCESS_ROUNDS <= 65535 )) || die "MASTER_PORT range exceeds 65535"
[[ "${MASTER_ADDR}" != "127.0.0.1" && "${MASTER_ADDR}" != "localhost" ]] \
    || die "MASTER_ADDR must be reachable from peer nodes, not loopback"
[[ "${MASTER_ADDR}" != "::1" ]] \
    || die "MASTER_ADDR must be reachable from peer nodes, not IPv6 loopback"

IFS=',' read -r -a GPU_ARRAY <<< "${LOCAL_GPUS}"
LOCAL_WORLD_SIZE="${#GPU_ARRAY[@]}"
(( LOCAL_WORLD_SIZE > 0 )) || die "RTP_MC_TEST_LOCAL_GPUS is empty"
(( NNODES * LOCAL_WORLD_SIZE == GLOBAL_TEAM_SIZE )) \
    || die "balanced mapping requires NNODES * local GPU count == GLOBAL_TEAM_SIZE"
RANK_BASE=$((NODE_RANK * LOCAL_WORLD_SIZE))

declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
    require_uint GPU "${gpu}"
    [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "duplicate local GPU ordinal ${gpu}"
    SEEN_GPUS[${gpu}]=1
done

for path in "${KEEPER_LAUNCHER}" "${HOLDER}" "${CREATOR}" "${SHIM}" "${WORKER}" "${PYTHON_BIN}"; do
    [[ -e "${path}" ]] || die "required artifact is missing: ${path}"
done
[[ -x "${KEEPER_LAUNCHER}" && -x "${HOLDER}" && -x "${CREATOR}" && -x "${PYTHON_BIN}" ]] \
    || die "keeper launcher, holder, creator, and Python must be executable"
"${HOLDER}" --help 2>&1 | grep -q -- '--fabric-team-size' \
    || die "holder does not implement the explicit FABRIC team-size contract"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
getent ahosts "${MASTER_ADDR}" >/dev/null || die "MASTER_ADDR does not resolve: ${MASTER_ADDR}"
if [[ "${NODE_RANK}" -eq 0 ]] && command -v ss >/dev/null; then
    listeners="$(ss -ltnH | awk '{split($4, fields, ":"); print fields[length(fields)]}')"
    for ((port = MASTER_PORT; port <= MASTER_PORT + SUCCESS_ROUNDS; ++port)); do
        ! grep -qx "${port}" <<< "${listeners}" || die "rendezvous port ${port} is already in use"
    done
fi

GPU_INVENTORY="$(nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader,nounits)"
GPU_UUIDS=()
DRIVER_VERSIONS=()
FABRIC_SUMMARY=()
for gpu in "${GPU_ARRAY[@]}"; do
    row="$(awk -F, -v wanted="${gpu}" '$1 ~ "^[[:space:]]*" wanted "[[:space:]]*$" {print}' <<< "${GPU_INVENTORY}")"
    [[ -n "${row}" ]] || die "GPU ${gpu} is not present"
    uuid="$(awk -F, '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' <<< "${row}")"
    name="$(awk -F, '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); print $3}' <<< "${row}")"
    driver="$(awk -F, '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}' <<< "${row}")"
    GPU_UUIDS+=("${uuid}")
    DRIVER_VERSIONS+=("${driver}")
    query="$(nvidia-smi -q -i "${gpu}")"
    fabric_state="$(
        awk '/Fabric/{inside=1; next} inside && /State[[:space:]]*:/{sub(/^.*:[[:space:]]*/, ""); print; exit}' \
            <<< "${query}"
    )"
    fabric_status="$(
        awk '/Fabric/{inside=1; next} inside && /Status[[:space:]]*:/{sub(/^.*:[[:space:]]*/, ""); print; exit}' \
            <<< "${query}"
    )"
    FABRIC_SUMMARY+=("${fabric_state:-unknown}/${fabric_status:-unknown}")
    echo "GPU_MAP node=${NODE_RANK} local_gpu=${gpu} uuid=${uuid} name=${name}" \
        "fabric=${fabric_state:-unknown}/${fabric_status:-unknown}"
    if [[ "${REQUIRE_FABRIC}" == "1" ]]; then
        [[ "${fabric_state}" == "Completed" && "${fabric_status}" == "Success" ]] \
            || die "GPU ${gpu} FABRIC is not Completed/Success"
    fi
done

UNIQUE_DRIVER_COUNT="$(printf '%s\n' "${DRIVER_VERSIONS[@]}" | sort -u | wc -l)"
[[ "${UNIQUE_DRIVER_COUNT}" -eq 1 ]] || die "selected GPUs report different driver versions"
DRIVER_VERSION="${DRIVER_VERSIONS[0]}"
LOCAL_GPU_UUIDS="$(IFS=,; echo "${GPU_UUIDS[*]}")"
LOCAL_FABRIC_STATUS="$(IFS=,; echo "${FABRIC_SUMMARY[*]}")"

if [[ "${REQUIRE_IMEX}" == "1" ]]; then
    [[ -e "${IMEX_CHANNEL}" ]] || die "IMEX channel is unavailable: ${IMEX_CHANNEL}"
    [[ -r "${IMEX_CHANNEL}" && -w "${IMEX_CHANNEL}" ]] \
        || die "IMEX channel is not readable/writable: ${IMEX_CHANNEL}"
fi

echo "IMEX channel=${IMEX_CHANNEL} required=${REQUIRE_IMEX}"
echo "TOPOLOGY node=${NODE_RANK}/${NNODES} rank_base=${RANK_BASE}" \
    "local_world_size=${LOCAL_WORLD_SIZE} global_team_size=${GLOBAL_TEAM_SIZE}"
nvidia-smi topo -m

CUDA_VISIBLE_DEVICES="${LOCAL_GPUS}" \
RTP_MC_PREFLIGHT_LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

expected = int(os.environ["RTP_MC_PREFLIGHT_LOCAL_WORLD_SIZE"])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch")
if torch.cuda.device_count() != expected:
    raise SystemExit(
        f"visible CUDA devices {torch.cuda.device_count()} != local team {expected}"
    )
if not dist.is_nccl_available():
    raise SystemExit("PyTorch NCCL backend is unavailable")
if not hasattr(symm_mem, "empty") or not hasattr(symm_mem, "rendezvous"):
    raise SystemExit("PyTorch symmetric-memory multicast API is unavailable")
print(
    "SOFTWARE_OK "
    f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
    f"nccl={torch.cuda.nccl.version()} visible_devices={torch.cuda.device_count()}"
)
PY

echo "PREFLIGHT_PASS node=${NODE_RANK} role=${ROLE} job=${JOB_ID}" \
    "driver=${DRIVER_VERSION} fabric=${LOCAL_FABRIC_STATUS}"
[[ "${ACTION}" == "run" ]] || exit 0

TEST_ROOT="${BASE_DIR}/${JOB_ID}/${ROLE}/node-${NODE_RANK}"
[[ ! -e "${TEST_ROOT}" ]] || die "test directory already exists; use a unique job id: ${TEST_ROOT}"
KEEPER_DIR="${TEST_ROOT}/keeper"
HOLDER_LOG="${KEEPER_DIR}/holder.log"
mkdir -p "${KEEPER_DIR}"
chmod 0700 "${TEST_ROOT}" "${KEEPER_DIR}"

WORKER_PIDS=()
HOLDER_PID=""
cleanup() {
    local rc=$?
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    for pid in "${WORKER_PIDS[@]:-}"; do
        wait "${pid}" 2>/dev/null || true
    done
    if [[ -n "${HOLDER_PID}" ]]; then
        env -u LD_PRELOAD -u RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER \
            "${KEEPER_LAUNCHER}" stop --keeper-dir "${KEEPER_DIR}" >/dev/null 2>&1 || true
    fi
    echo "artifacts: ${TEST_ROOT}"
    exit "${rc}"
}
trap cleanup EXIT INT TERM

export RTP_LLM_MC_HOLDER_BIN="${HOLDER}"
export RTP_LLM_MC_CREATOR_BIN="${CREATOR}"
export RTP_LLM_MC_SHIM="${SHIM}"
env -u CUDA_VISIBLE_DEVICES -u LD_PRELOAD "${KEEPER_LAUNCHER}" start \
    --gpus "${LOCAL_GPUS}" \
    --fabric-team-size "${GLOBAL_TEAM_SIZE}" \
    --keeper-dir "${KEEPER_DIR}"
read -r HOLDER_PID HOLDER_START_TIME < "${KEEPER_DIR}/holder.pid"
HOLDER_STATUS="$("${KEEPER_LAUNCHER}" status --keeper-dir "${KEEPER_DIR}")"
HOLDER_INSTANCE="$(sed -n 's/.*instance=\([0-9a-fA-F]*\).*/\1/p' <<< "${HOLDER_STATUS}")"
[[ -n "${HOLDER_INSTANCE}" ]] || die "could not read holder instance"
if grep -Eq 'libcuda|libcudart' "/proc/${HOLDER_PID}/maps"; then
    die "holder unexpectedly loaded a CUDA library"
fi
echo "HOLDER_READY node=${NODE_RANK} pid=${HOLDER_PID} instance=${HOLDER_INSTANCE} dir=${KEEPER_DIR}"

# Exercise the same generated environment contract used by production ranks.
# shellcheck disable=SC1090
source "${KEEPER_DIR}/keeper.env"
[[ "${RTP_LLM_MC_LOCAL_GPUS:-}" == "${LOCAL_GPUS}" ]] \
    || die "keeper.env local GPU contract does not match the requested team"
[[ "${RTP_LLM_MC_FABRIC_TEAM_SIZE:-}" == "${GLOBAL_TEAM_SIZE}" ]] \
    || die "keeper.env global FABRIC team contract is missing or incorrect"
[[ "${NCCL_NVLS_ENABLE:-}" == "1" && "${TORCH_SYMM_MEM_DISABLE_MULTICAST:-}" == "0" ]] \
    || die "keeper.env did not enable NCCL NVLS and torch multicast"
export RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER_DEBUG="${RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER_DEBUG:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NVLS}"
export NCCL_DEBUG_FILE="${TEST_ROOT}/nccl.%h.%p.log"
export CUDA_VISIBLE_DEVICES="${LOCAL_GPUS}"
export WORLD_SIZE="${GLOBAL_TEAM_SIZE}"
export LOCAL_WORLD_SIZE
export MASTER_ADDR MASTER_PORT
export RTP_MC_TEST_JOB_ID="${JOB_ID}"
export RTP_MC_TEST_ROLE="${ROLE}"
export RTP_MC_TEST_NODE_RANK="${NODE_RANK}"
export RTP_MC_TEST_NNODES="${NNODES}"
export RTP_MC_TEST_LOCAL_GPUS="${LOCAL_GPUS}"
export RTP_MC_TEST_LOCAL_GPU_UUIDS="${LOCAL_GPU_UUIDS}"
export RTP_MC_TEST_DRIVER_VERSION="${DRIVER_VERSION}"
export RTP_MC_TEST_FABRIC_STATUS="${LOCAL_FABRIC_STATUS}"
export RTP_MC_TEST_SUCCESS_ROUNDS="${SUCCESS_ROUNDS}"
export RTP_MC_TEST_FAIL_HOLDER_NODE="${FAILURE_NODE}"
export RTP_MC_TEST_HOLDER_PID="${HOLDER_PID}"

for ((local_rank = 0; local_rank < LOCAL_WORLD_SIZE; ++local_rank)); do
    global_rank=$((RANK_BASE + local_rank))
    log="${TEST_ROOT}/rank${global_rank}.log"
    env \
        RANK="${global_rank}" \
        LOCAL_RANK="${local_rank}" \
        RTP_LLM_MC_OWNER_ID="$((global_rank + 1))" \
        "${PYTHON_BIN}" "${WORKER}" > "${log}" 2>&1 &
    WORKER_PIDS+=("$!")
done

deadline=$((SECONDS + ${RTP_MC_TEST_TIMEOUT_SECONDS:-900}))
while :; do
    live=0
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
            [[ "${state}" != Z* ]] && live=$((live + 1))
        fi
    done
    (( live > 0 )) || break
    if (( SECONDS >= deadline )); then
        echo "timed out waiting for local workers" >&2
        for log in "${TEST_ROOT}"/rank*.log; do tail -n 100 "${log}" >&2; done
        exit 1
    fi
    sleep 1
done

worker_failure=0
for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" || worker_failure=1
done
(( worker_failure == 0 )) || {
    for log in "${TEST_ROOT}"/rank*.log; do tail -n 100 "${log}" >&2; done
    exit 1
}

for ((local_rank = 0; local_rank < LOCAL_WORLD_SIZE; ++local_rank)); do
    global_rank=$((RANK_BASE + local_rank))
    log="${TEST_ROOT}/rank${global_rank}.log"
    grep -q "^READY rank=${global_rank} " "${log}"
    [[ "$(grep -c "^ROUND_OK rank=${global_rank} " "${log}")" -eq "${SUCCESS_ROUNDS}" ]]
    grep -q "^FAIL_CLOSED rank=${global_rank} " "${log}"
    grep -q "^TEST_PASS rank=${global_rank} " "${log}"
    ! grep -q '^TEST_FAIL ' "${log}"
done

grep -Eq 'creator_start .*handles=0x(8|9)([^0-9a-fA-F]|$)' "${HOLDER_LOG}" \
    || die "holder never observed a FABRIC-capable multicast request"
if [[ "${NODE_RANK}" -ne "${FAILURE_NODE}" ]]; then
    env -u LD_PRELOAD -u RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER \
        "${KEEPER_LAUNCHER}" status --keeper-dir "${KEEPER_DIR}" >/dev/null
    [[ "$(awk '{print $22}' "/proc/${HOLDER_PID}/stat")" == "${HOLDER_START_TIME}" ]] \
        || die "holder process identity changed"
fi
if [[ "${REQUIRE_NCCL_NVLS_LOG}" == "1" ]]; then
    grep -Eiq 'NVLS' "${TEST_ROOT}"/nccl.*.log \
        || die "NCCL logs contain no NVLS evidence"
fi

echo "MULTINODE_MULTICAST_TEST_PASS node=${NODE_RANK} role=${ROLE} job=${JOB_ID}" \
    "successful_rebuilds=${SUCCESS_ROUNDS} failure_node=${FAILURE_NODE}"
