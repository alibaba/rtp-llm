#!/usr/bin/env bash
set -euo pipefail
export MOCK_BIND_HOST="${MOCK_BIND_HOST:-0.0.0.0}"
java_cmd=("${JAVA_HOME:-/opt/taobao/java}/bin/java" -jar /home/admin/mock-engine/mock-engine.jar)

case ${ROLE_TYPE:-} in
    PREFILL | prefill) role=prefill ;;
    DECODE | decode) role=decode ;;
    *) exec "${java_cmd[@]}" ;;
esac

dp_size=${DP_SIZE:-1}
if [[ ! $dp_size =~ ^[1-9][0-9]{0,4}$ ]]; then
    echo "DP_SIZE must be a positive integer below 100000" >&2
    exit 2
fi
if ((dp_size == 1)); then
    exec "${java_cmd[@]}"
fi

base_port=${START_PORT:-8088}
port_stride=10
if [[ ${WORKER_INFO_PORT_NUM:-$port_stride} != "$port_stride" ]]; then
    echo "Multi-DP Mock requires WORKER_INFO_PORT_NUM=$port_stride" >&2
    exit 2
fi
if [[ ! $base_port =~ ^[1-9][0-9]{0,4}$ ]] ||
    ((base_port + (dp_size - 1) * port_stride + 5 > 65535)); then
    echo "START_PORT must leave room for the Mock DP ports" >&2
    exit 2
fi
if [[ ${TP_SIZE:-1} != 1 || ${PP_SIZE:-1} != 1 || ${LOCAL_WORLD_SIZE:-$dp_size} != "$dp_size" ]]; then
    echo "Multi-JVM Mock requires TP1, PP1 and all DP workers in the same Pod" >&2
    exit 2
fi

worker_pids=()
stop_workers() {
    if [[ -n ${worker_pids[0]:-} ]]; then
        kill -TERM "${worker_pids[@]}" 2>/dev/null || true
        wait "${worker_pids[@]}" 2>/dev/null || true
    fi
}
trap stop_workers EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for ((rank = 0; rank < dp_size; rank++)); do
    port=$((base_port + rank * port_stride))
    START_PORT=$port DP_RANK=$rank MOCK_NODE_ID="${MOCK_NODE_ID:-$role-${HOSTNAME:-mock}}-$rank" "${java_cmd[@]}" &
    worker_pids+=("$!")
done

while true; do
    for pid in "${worker_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            worker_status=0
            wait "$pid" || worker_status=$?
            echo "Mock $role JVM exited with status $worker_status; stopping the Pod's workers" >&2
            exit 1
        fi
    done
    sleep 1
done
