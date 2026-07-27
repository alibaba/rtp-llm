#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -eq 5 ]] || {
    echo "usage: $0 HOLDER CREATOR SHIM WORKER CHECKPOINT_DRIVER" >&2
    exit 2
}
HOLDER="$1"
CREATOR="$2"
SHIM="$3"
WORKER="$4"
CHECKPOINT_DRIVER="$5"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda310/bin/python3}"
CUDA_CHECKPOINT_BIN="${CUDA_CHECKPOINT_BIN:-$(command -v cuda-checkpoint || true)}"
GPUS="${GPUS:-0,1}"
MASTER_PORT="${MASTER_PORT:-39420}"
TEST_ROOT="${TEST_ROOT:-$(mktemp -d /tmp/rtp_mc_gpu_test.XXXXXX)}"
KEEP_TEST_ROOT="${KEEP_TEST_ROOT:-0}"
NEKYIA_KEEPER_DIR="${NEKYIA_KEEPER_DIR:-${TEST_ROOT}/keeper}"
SOCKET="${NEKYIA_KEEPER_DIR}/mcsk.sock"
READY_FILE="${NEKYIA_KEEPER_DIR}/holder.ready"
HOLDER_LOG="${TEST_ROOT}/holder.log"

[[ -x "${PYTHON_BIN}" ]] || { echo "python is not executable: ${PYTHON_BIN}" >&2; exit 2; }
if [[ -n "${CUDA_CHECKPOINT_BIN}" ]]; then
    [[ -x "${CUDA_CHECKPOINT_BIN}" ]] || {
        echo "CUDA_CHECKPOINT_BIN is not executable: ${CUDA_CHECKPOINT_BIN}" >&2
        exit 2
    }
    CHECKPOINT_BACKEND="cli"
else
    [[ -x "${CHECKPOINT_DRIVER}" ]] || {
        echo "checkpoint driver helper is not executable: ${CHECKPOINT_DRIVER}" >&2
        exit 2
    }
    CHECKPOINT_BACKEND="libcuda-driver"
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
[[ "${WORLD_SIZE}" -ge 2 ]] || { echo "GPUS must contain at least two devices" >&2; exit 2; }

mkdir -p "${NEKYIA_KEEPER_DIR}"
chmod 0700 "${NEKYIA_KEEPER_DIR}"
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
        kill -TERM "${HOLDER_PID}" 2>/dev/null || true
        wait "${HOLDER_PID}" 2>/dev/null || true
    fi
    if [[ "${KEEP_TEST_ROOT}" != "1" ]]; then
        rm -rf "${TEST_ROOT}"
    else
        echo "test artifacts: ${TEST_ROOT}"
    fi
    exit "${rc}"
}
trap cleanup EXIT INT TERM

wait_log() {
    local file="$1" pattern="$2" timeout_seconds="$3" watched_pid="${4:-}"
    local deadline=$((SECONDS + timeout_seconds))
    while ! grep -q "${pattern}" "${file}" 2>/dev/null; do
        for peer_pid in "${WORKER_PIDS[@]:-}"; do
            [[ -n "${peer_pid}" ]] || continue
            local peer_state
            peer_state="$(ps -o stat= -p "${peer_pid}" 2>/dev/null || true)"
            if [[ -z "${peer_state}" || "${peer_state}" == Z* ]]; then
                echo "worker ${peer_pid} died while waiting for '${pattern}' in ${file}" >&2
                for peer_log in "${TEST_ROOT}"/rank*.log; do
                    [[ -f "${peer_log}" ]] && tail -n 100 "${peer_log}" >&2
                done
                return 1
            fi
        done
        if [[ -n "${watched_pid}" ]]; then
            local process_state
            process_state="$(ps -o stat= -p "${watched_pid}" 2>/dev/null || true)"
            if [[ -z "${process_state}" || "${process_state}" == Z* ]]; then
                echo "process ${watched_pid} died waiting for '${pattern}' in ${file}" >&2
                tail -n 100 "${file}" >&2 2>/dev/null || true
                return 1
            fi
        fi
        (( SECONDS < deadline )) || {
            echo "timeout waiting for '${pattern}' in ${file}" >&2
            tail -n 100 "${file}" >&2 2>/dev/null || true
            return 1
        }
        sleep 0.2
    done
}

checkpoint_action() {
    local action="$1" pid="$2"
    if [[ "${CHECKPOINT_BACKEND}" == "cli" ]]; then
        "${CUDA_CHECKPOINT_BIN}" --action "${action}" --pid "${pid}"
    else
        "${CHECKPOINT_DRIVER}" --action "${action}" --pid "${pid}"
    fi
}

# The holder is intentionally launched before CUDA_VISIBLE_DEVICES is set for ranks.
# Its short-lived creators consume the physical ordinals in GPUS.
env -u CUDA_VISIBLE_DEVICES -u LD_PRELOAD "${HOLDER}" \
    --socket "${SOCKET}" \
    --ready-file "${READY_FILE}" \
    --creator "${CREATOR}" \
    --gpus "${GPUS}" \
    --fabric-team-size "${WORLD_SIZE}" \
    > "${HOLDER_LOG}" 2>&1 &
HOLDER_PID=$!
for _ in $(seq 1 100); do
    [[ -f "${READY_FILE}" ]] && "${HOLDER}" --check --socket "${SOCKET}" >/dev/null && break
    kill -0 "${HOLDER_PID}" 2>/dev/null || { cat "${HOLDER_LOG}" >&2; exit 1; }
    sleep 0.1
done
"${HOLDER}" --check --socket "${SOCKET}" >/dev/null
HOLDER_START_TIME="$(awk '{print $22}' "/proc/${HOLDER_PID}/stat")"
if grep -Eq "libcuda|libcudart" "/proc/${HOLDER_PID}/maps"; then
    echo "holder unexpectedly loaded a CUDA library" >&2
    exit 1
fi

export RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER=1
export RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER_DEBUG="${RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER_DEBUG:-1}"
export NEKYIA_KEEPER_DIR
export RTP_LLM_MC_LOCAL_GPUS="${GPUS}"
export RTP_LLM_MC_FABRIC_TEAM_SIZE="${WORLD_SIZE}"
export NCCL_NVLS_ENABLE=1
export TORCH_SYMM_MEM_DISABLE_MULTICAST=0
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NVLS}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT
export WORLD_SIZE
# Keep an existing allocator interposer (notably TMS) before the dlsym shim.
export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${SHIM}"

for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
    log="${TEST_ROOT}/rank${rank}.log"
    env RANK="${rank}" LOCAL_RANK="${rank}" "${PYTHON_BIN}" "${WORKER}" > "${log}" 2>&1 &
    WORKER_PIDS+=("$!")
done
for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
    wait_log "${TEST_ROOT}/rank${rank}.log" "^READY rank=${rank}" 180 "${WORKER_PIDS[rank]}"
    grep -Eq "multicast_ptr=0x[1-9a-fA-F][0-9a-fA-F]*" "${TEST_ROOT}/rank${rank}.log"
done
CREATOR_COUNT_BEFORE="$(grep -c "creator_start" "${HOLDER_LOG}" || true)"

for pid in "${WORKER_PIDS[@]}"; do kill -USR1 "${pid}"; done
for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
    wait_log "${TEST_ROOT}/rank${rank}.log" "^TORNDOWN rank=${rank}" 60 "${WORKER_PIDS[rank]}"
done

echo "checkpoint_backend=${CHECKPOINT_BACKEND}"
for pid in "${WORKER_PIDS[@]}"; do checkpoint_action lock "${pid}"; done
for pid in "${WORKER_PIDS[@]}"; do checkpoint_action checkpoint "${pid}"; done
"${HOLDER}" --check --socket "${SOCKET}"
[[ "$(awk '{print $22}' "/proc/${HOLDER_PID}/stat")" == "${HOLDER_START_TIME}" ]]
CHECKPOINT_COMPUTE_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
for pid in "${WORKER_PIDS[@]}"; do
    if grep -qx "${pid}" <<< "${CHECKPOINT_COMPUTE_PIDS}"; then
        echo "checkpointed rank ${pid} still owns a CUDA compute context" >&2
        exit 1
    fi
done
echo "checkpoint_gpu_memory:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits || true
for pid in "${WORKER_PIDS[@]}"; do checkpoint_action restore "${pid}"; done
for pid in "${WORKER_PIDS[@]}"; do checkpoint_action unlock "${pid}"; done

for pid in "${WORKER_PIDS[@]}"; do kill -USR2 "${pid}"; done
for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
    wait_log "${TEST_ROOT}/rank${rank}.log" "^RESULT rank=${rank}" 180 "${WORKER_PIDS[rank]}"
    grep -q "equal=True" "${TEST_ROOT}/rank${rank}.log"
    grep -Eq "multicast_ptr=0x[1-9a-fA-F][0-9a-fA-F]*" "${TEST_ROOT}/rank${rank}.log"
done

CREATOR_COUNT_AFTER="$(grep -c "creator_start" "${HOLDER_LOG}" || true)"
[[ "${CREATOR_COUNT_AFTER}" == "${CREATOR_COUNT_BEFORE}" ]] || {
    echo "rebuild created a new multicast object: before=${CREATOR_COUNT_BEFORE} after=${CREATOR_COUNT_AFTER}" >&2
    cat "${HOLDER_LOG}" >&2
    exit 1
}
if grep -Eq "libcuda|libcudart" "/proc/${HOLDER_PID}/maps"; then
    echo "holder loaded a CUDA library after serving cache entries" >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -qx "${HOLDER_PID}"; then
    echo "holder owns a CUDA compute context" >&2
    exit 1
fi

echo "MULTICAST_CHECKPOINT_TEST_PASS world_size=${WORLD_SIZE} holder_pid=${HOLDER_PID} creator_count=${CREATOR_COUNT_AFTER} checkpoint_backend=${CHECKPOINT_BACKEND}"
"${HOLDER}" --check --socket "${SOCKET}"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader || true
