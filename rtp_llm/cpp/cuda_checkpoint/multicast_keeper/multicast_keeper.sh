#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
TARGET_DIR="rtp_llm/cpp/cuda_checkpoint/multicast_keeper"
BIN_DIR="${RTP_LLM_MC_KEEPER_BIN_DIR:-${REPO_ROOT}/bazel-bin/${TARGET_DIR}}"
HOLDER_BIN="${RTP_LLM_MC_HOLDER_BIN:-${BIN_DIR}/keeper_lite_holder}"
CREATOR_BIN="${RTP_LLM_MC_CREATOR_BIN:-${BIN_DIR}/keeper_lite_creator}"
SHIM="${RTP_LLM_MC_SHIM:-${BIN_DIR}/mc_shim_unified.so}"

canonicalize_artifact() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        readlink -f "${path}"
    else
        printf '%s\n' "${path}"
    fi
}

HOLDER_BIN="$(canonicalize_artifact "${HOLDER_BIN}")"
CREATOR_BIN="$(canonicalize_artifact "${CREATOR_BIN}")"
SHIM="$(canonicalize_artifact "${SHIM}")"

usage() {
    cat <<'EOF'
Usage:
  multicast_keeper.sh start --gpus LIST [--fabric-team-size N] [--keeper-dir DIR] [--socket PATH]
  multicast_keeper.sh stop  [--keeper-dir DIR] [--socket PATH]
  multicast_keeper.sh status [--keeper-dir DIR] [--socket PATH]
  multicast_keeper.sh env [--keeper-dir DIR]
  multicast_keeper.sh run --gpus LIST [options] -- COMMAND [ARG ...]

Options:
  --gpus LIST                CUDA ordinals used by every short-lived creator
  --fabric-team-size N       Exact global FABRIC team size (required for FABRIC)
  --keeper-dir DIR           State dir and NEKYIA_KEEPER_DIR (default /tmp/rtp_llm-mc-$UID)
  --socket PATH              Override DIR/mcsk.sock
  --client-timeout-ms N      Holder request/reply I/O timeout (default 1000)
  --creator-timeout-ms N     Per-size creator timeout (default 120000)

Build first:
  bazel build //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:all --config=cuda13
EOF
}

die() {
    echo "multicast_keeper: $*" >&2
    exit 2
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
ACTION="$1"
shift

GPUS="${RTP_LLM_MC_KEEPER_GPUS:-}"
FABRIC_TEAM_SIZE="${RTP_LLM_MC_FABRIC_TEAM_SIZE:-}"
KEEPER_DIR="${NEKYIA_KEEPER_DIR:-/tmp/rtp_llm-mc-${UID}}"
SOCKET=""
CREATOR_TIMEOUT_MS="${RTP_LLM_MC_CREATOR_TIMEOUT_MS:-120000}"
CLIENT_TIMEOUT_MS="${RTP_LLM_MC_HOLDER_IO_TIMEOUT_MS:-1000}"
REQUEST_TIMEOUT_MS="${RTP_LLM_MC_REQUEST_TIMEOUT_MS:-5000}"
CREATE_TIMEOUT_MS="${RTP_LLM_MC_CREATE_TIMEOUT_MS:-125000}"
COMMAND=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            [[ $# -ge 2 ]] || die "--gpus requires a value"
            GPUS="$2"
            shift 2
            ;;
        --fabric-team-size)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--fabric-team-size requires a positive integer"
            FABRIC_TEAM_SIZE="$2"
            shift 2
            ;;
        --keeper-dir)
            [[ $# -ge 2 ]] || die "--keeper-dir requires a value"
            KEEPER_DIR="$2"
            shift 2
            ;;
        --socket)
            [[ $# -ge 2 ]] || die "--socket requires a value"
            SOCKET="$2"
            shift 2
            ;;
        --creator-timeout-ms)
            [[ $# -ge 2 ]] || die "--creator-timeout-ms requires a value"
            CREATOR_TIMEOUT_MS="$2"
            shift 2
            ;;
        --client-timeout-ms)
            [[ $# -ge 2 ]] || die "--client-timeout-ms requires a value"
            CLIENT_TIMEOUT_MS="$2"
            shift 2
            ;;
        --)
            shift
            COMMAND=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n "${SOCKET}" ]] || SOCKET="${KEEPER_DIR}/mcsk.sock"
PID_FILE="${KEEPER_DIR}/holder.pid"
READY_FILE="${KEEPER_DIR}/holder.ready"
ENV_FILE="${KEEPER_DIR}/keeper.env"
LOG_FILE="${KEEPER_DIR}/holder.log"

process_start_time() {
    awk '{print $22}' "/proc/$1/stat" 2>/dev/null || true
}

read_live_pid() {
    [[ -r "${PID_FILE}" ]] || return 1
    local pid expected actual
    read -r pid expected < "${PID_FILE}" || return 1
    [[ "${pid}" =~ ^[0-9]+$ && -d "/proc/${pid}" ]] || return 1
    actual="$(process_start_time "${pid}")"
    [[ -n "${actual}" && "${actual}" == "${expected}" ]] || return 1
    printf '%s\n' "${pid}"
}

write_env_file() {
    local socket_dir
    socket_dir="$(dirname "${SOCKET}")"
    {
        printf 'export RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER=1\n'
        printf 'export NEKYIA_KEEPER_DIR=%q\n' "${socket_dir}"
        printf 'export RTP_LLM_MC_LOCAL_GPUS=%q\n' "${GPUS}"
        if [[ -n "${FABRIC_TEAM_SIZE}" ]]; then
            printf 'export RTP_LLM_MC_FABRIC_TEAM_SIZE=%q\n' "${FABRIC_TEAM_SIZE}"
        fi
        if [[ "${SOCKET}" != "${socket_dir}/mcsk.sock" ]]; then
            printf 'export RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET=%q\n' "${SOCKET}"
        fi
        printf 'export NCCL_NVLS_ENABLE=1\n'
        printf 'export TORCH_SYMM_MEM_DISABLE_MULTICAST=0\n'
        printf 'export RTP_LLM_MC_REQUEST_TIMEOUT_MS=%q\n' "${REQUEST_TIMEOUT_MS}"
        printf 'export RTP_LLM_MC_CREATE_TIMEOUT_MS=%q\n' "${CREATE_TIMEOUT_MS}"
        # TMS must precede this dlsym-interposing shim: TMS resolves its real
        # cudaMalloc with RTLD_NEXT, which would otherwise recurse into itself.
        printf 'export LD_PRELOAD=${LD_PRELOAD:+${LD_PRELOAD}:}%q\n' "${SHIM}"
    } > "${ENV_FILE}.tmp"
    chmod 0600 "${ENV_FILE}.tmp"
    mv -f "${ENV_FILE}.tmp" "${ENV_FILE}"
}

start_keeper() {
    [[ -n "${GPUS}" ]] || die "start requires --gpus LIST"
    [[ -x "${HOLDER_BIN}" ]] || die "holder binary is not executable: ${HOLDER_BIN}"
    [[ -x "${CREATOR_BIN}" ]] || die "creator binary is not executable: ${CREATOR_BIN}"
    [[ -f "${SHIM}" ]] || die "unified shim does not exist: ${SHIM}"
    mkdir -p "${KEEPER_DIR}"
    chmod 0700 "${KEEPER_DIR}"
    if live_pid="$(read_live_pid)"; then
        die "holder is already running: pid=${live_pid}"
    fi
    rm -f "${PID_FILE}" "${READY_FILE}" "${SOCKET}" "${ENV_FILE}"

    # The launcher itself can already carry TMS. The holder must remain
    # CUDA-free; creators and ranks establish their own CUDA environments.
    local holder_args=(
        --socket "${SOCKET}"
        --ready-file "${READY_FILE}"
        --creator "${CREATOR_BIN}"
        --client-timeout-ms "${CLIENT_TIMEOUT_MS}"
        --creator-timeout-ms "${CREATOR_TIMEOUT_MS}"
        --gpus "${GPUS}"
    )
    if [[ -n "${FABRIC_TEAM_SIZE}" ]]; then
        holder_args+=(--fabric-team-size "${FABRIC_TEAM_SIZE}")
    fi
    env -u LD_PRELOAD -u CUDA_VISIBLE_DEVICES setsid "${HOLDER_BIN}" "${holder_args[@]}" \
        >> "${LOG_FILE}" 2>&1 < /dev/null &
    local pid=$!
    local start_time
    start_time="$(process_start_time "${pid}")"
    [[ -n "${start_time}" ]] || { kill "${pid}" 2>/dev/null || true; die "failed to inspect holder"; }
    printf '%s %s\n' "${pid}" "${start_time}" > "${PID_FILE}"

    local ready=0
    for _ in $(seq 1 100); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            break
        fi
        if [[ -f "${READY_FILE}" ]] && "${HOLDER_BIN}" --check --socket "${SOCKET}" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 0.1
    done
    if [[ "${ready}" != "1" ]]; then
        kill -TERM "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
        rm -f "${PID_FILE}"
        tail -n 40 "${LOG_FILE}" >&2 2>/dev/null || true
        die "holder did not become ready"
    fi
    write_env_file
    echo "KEEPER_READY pid=${pid} socket=${SOCKET} gpus=${GPUS} fabric_team_size=${FABRIC_TEAM_SIZE:-disabled}"
    echo "source ${ENV_FILE}"
}

stop_keeper() {
    local pid=""
    if pid="$(read_live_pid)"; then
        kill -TERM "${pid}" 2>/dev/null || true
        for _ in $(seq 1 100); do
            [[ ! -d "/proc/${pid}" ]] && break
            sleep 0.1
        done
        if [[ -d "/proc/${pid}" ]]; then
            echo "multicast_keeper: holder did not exit after SIGTERM, sending SIGKILL" >&2
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    fi
    rm -f "${PID_FILE}" "${READY_FILE}" "${ENV_FILE}" "${SOCKET}"
    echo "KEEPER_STOPPED${pid:+ pid=${pid}}"
}

status_keeper() {
    local pid checker
    if ! pid="$(read_live_pid)"; then
        echo "KEEPER_STOPPED"
        return 1
    fi
    checker="${HOLDER_BIN}"
    if [[ ! -x "${checker}" ]]; then
        checker="/proc/${pid}/exe"
    fi
    "${checker}" --check --socket "${SOCKET}"
    echo "KEEPER_RUNNING pid=${pid}"
}

case "${ACTION}" in
    start)
        [[ ${#COMMAND[@]} -eq 0 ]] || die "start does not accept a command"
        start_keeper
        ;;
    stop)
        [[ ${#COMMAND[@]} -eq 0 ]] || die "stop does not accept a command"
        stop_keeper
        ;;
    status)
        [[ ${#COMMAND[@]} -eq 0 ]] || die "status does not accept a command"
        status_keeper
        ;;
    env)
        [[ -r "${ENV_FILE}" ]] || die "keeper env is unavailable: ${ENV_FILE}"
        cat "${ENV_FILE}"
        ;;
    run)
        [[ ${#COMMAND[@]} -gt 0 ]] || die "run requires -- COMMAND"
        start_keeper
        # shellcheck disable=SC1090
        source "${ENV_FILE}"
        trap stop_keeper EXIT INT TERM
        "${COMMAND[@]}"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
