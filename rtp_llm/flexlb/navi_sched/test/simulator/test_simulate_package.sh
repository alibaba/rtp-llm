#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAVI_SCHED_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${NAVI_SCHED_PYTHON_BIN:-/opt/conda310/bin/python3}
DEFAULT_ARCHIVE=${NAVI_SCHED_DIR}/release/navi_sched_simulate-0.1.0.tar.gz
ARCHIVE=${NAVI_SCHED_SIMULATE_ARCHIVE:-${DEFAULT_ARCHIVE}}
TEST_DIR=${NAVI_SCHED_SIMULATE_TEST_DIR:-${NAVI_SCHED_DIR}/tmp/simulate_package_test}
TRAFFIC_LOG=${NAVI_SCHED_TRAFFIC_LOG:-}
TRAFFIC_SERVICE_NAME=${NAVI_SCHED_TRAFFIC_SERVICE_NAME:-traffic_service}
KEEP_TEST_DIR=${NAVI_SCHED_KEEP_SIMULATE_TEST_DIR:-0}

if [[ ! -f "${ARCHIVE}" ]]; then
    echo "missing simulate archive: ${ARCHIVE}" >&2
    exit 1
fi

PORT_BASE=$(
    "${PYTHON_BIN}" - <<'PY'
import os
import socket

port_count = (60000 - 40000) // 16
for index in range(port_count):
    base = 40000 + ((os.getpid() + index) % port_count) * 16
    sockets = []
    try:
        for port in range(base, base + 6):
            value = socket.socket()
            value.bind(("127.0.0.1", port))
            sockets.append(value)
        print(base)
        break
    except OSError:
        pass
    finally:
        for value in sockets:
            value.close()
else:
    raise RuntimeError("no contiguous port range for simulate package test")
PY
)
AGENT_GRPC_PORT=${PORT_BASE}
MASTER_GRPC_PORT=$((PORT_BASE + 1))

AGENT_PID=
MASTER_PID=
CLIENT_PID=
CLIENT_1_PID=
CLIENT_2_PID=

stop_pid_group() {
    local timeout_seconds=$1
    shift
    local pid
    for pid in "$@"; do
        [[ -z "${pid}" ]] || kill -TERM "${pid}" 2>/dev/null || true
    done
    for _ in $(seq 1 $((timeout_seconds * 10))); do
        local running=0
        for pid in "$@"; do
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                running=1
                break
            fi
        done
        [[ ${running} -eq 0 ]] && break
        sleep 0.1
    done
    for pid in "$@"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "$@"; do
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    done
}

stop_roles() {
    stop_pid_group 10 "${CLIENT_1_PID}" "${CLIENT_2_PID}" "${CLIENT_PID}"
    stop_pid_group 25 "${AGENT_PID}"
    stop_pid_group 10 "${MASTER_PID}"
    CLIENT_PID=
    CLIENT_1_PID=
    CLIENT_2_PID=
    MASTER_PID=
    AGENT_PID=
}

cleanup() {
    local status=$?
    stop_roles
    if [[ ${status} -ne 0 ]]; then
        while IFS= read -r log_file; do
            printf '%s\n' "--- ${log_file} ---" >&2
            grep -Ei 'ERROR|WARN|failed|abort|invalid|not exist|not found' \
                "${log_file}" | tail -80 >&2 || true
        done < <(find "${TEST_DIR}" -type f -name '*.log' | sort)
    fi
    if [[ "${KEEP_TEST_DIR}" != 1 ]]; then
        rm -rf "${TEST_DIR}"
    fi
    return ${status}
}
trap cleanup EXIT

rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"
tar -xzf "${ARCHIVE}" -C "${TEST_DIR}"
PACKAGE_DIR=${TEST_DIR}/navi_sched_simulate-0.1.0
mkdir -p "${PACKAGE_DIR}/runtime/agent" \
    "${PACKAGE_DIR}/runtime/master" \
    "${PACKAGE_DIR}/runtime/client" \
    "${PACKAGE_DIR}/runtime/client_1"
env LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" PYTHONPATH="${PACKAGE_DIR}/python" \
    "${PYTHON_BIN}" - <<'PY'
import navi_sched_simulate


class Options:
    grpc_port = 12345


service_options = navi_sched_simulate._service_options(
    "agent", Options(), "/absolute/runtime/path")
assert service_options["log_file_name"] == "navi_sched.log"
assert service_options["gig"]["log_prefix"] == "gig"
assert service_options["gig"]["snapshot_log_dir"] == "gig_snapshot"
PY
if [[ -n "${TRAFFIC_LOG}" ]]; then
    cp "${TRAFFIC_LOG}" "${PACKAGE_DIR}/runtime/client/traffic_replay.jsonl"
else
    first_request='{"requestId":"traffic_replay_first","reqTime":"10",'
    first_request+='"service_name":"traffic_service",'
    first_request+='"usage":{"inputTokens":"520"},'
    first_request+='"block_cache_keys":[11,12]}'
    late_request='{"requestId":"traffic_replay_late","reqTime":"30",'
    late_request+='"service_name":"traffic_service",'
    late_request+='"usage":{"inputTokens":"780"},'
    late_request+='"block_cache_keys":[11,12,13]}'
    second_request='{"requestId":"traffic_replay_second","reqTime":"30",'
    second_request+='"service_name":"traffic_service",'
    second_request+='"usage":{"inputTokens":"1040"},'
    second_request+='"block_cache_keys":[11,12,13,14]}'
    printf '%s\n' \
        "${first_request}" "${late_request}" "${second_request}" \
        >"${PACKAGE_DIR}/runtime/client/traffic_replay.jsonl"
fi
TRAFFIC_REQUEST_COUNT=$(grep -cve '^$' \
    "${PACKAGE_DIR}/runtime/client/traffic_replay.jsonl")
CLIENT_0_REQUEST_COUNT=$(((TRAFFIC_REQUEST_COUNT + 1) / 2))
CLIENT_1_REQUEST_COUNT=$((TRAFFIC_REQUEST_COUNT / 2))
master_config='{"biz_name":"master_biz","scheduler_name":"random","time_compression":2,'
master_config+='"local_nodes":[{"biz_name":"agent_biz",'
master_config+='"ip":"127.0.0.1","grpc_port":%d,'
master_config+='"support_heartbeat":true}]}\n'
printf "${master_config}" "${AGENT_GRPC_PORT}" \
    >"${PACKAGE_DIR}/runtime/master/config.json"

(
    cd "${PACKAGE_DIR}/runtime/agent"
    exec env PATH=/usr/bin:/bin LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" \
        "${PACKAGE_DIR}/bin/navi_sched_simulate_agent" \
        --biz agent_biz \
        --grpc-port "${AGENT_GRPC_PORT}" \
        --server-port "${AGENT_GRPC_PORT}" \
        --time-compression 2 \
        --runtime-dir "${PACKAGE_DIR}/runtime/agent"
) >"${TEST_DIR}/agent.log" 2>&1 &
AGENT_PID=$!
(
    cd "${PACKAGE_DIR}/runtime/master"
    exec env PATH=/usr/bin:/bin LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" \
        NAVI_SCHED_MASTER_BIZ_NAME=master_biz \
        NAVI_SCHED_MASTER_CONFIG_FILE=config.json \
        NAVI_SCHED_MASTER_GRPC_PORT="${MASTER_GRPC_PORT}" \
        NAVI_SCHED_MASTER_LOG_FILE=navi_sched.log \
        NAVI_SCHED_MASTER_SNAPSHOT_DIR=navi_sched \
        "${PACKAGE_DIR}/bin/navi_sched_simulate_master"
) >"${TEST_DIR}/master.log" 2>&1 &
MASTER_PID=$!

(
    cd "${PACKAGE_DIR}/runtime/client"
    env LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" PYTHONPATH="${PACKAGE_DIR}/python" \
        "${PYTHON_BIN}" - "${MASTER_GRPC_PORT}" "${AGENT_GRPC_PORT}" <<'PY'
import sys
import time

from navi_sched import NaviSchedClient, NaviSchedRequest

master_port, agent_port = map(int, sys.argv[1:])
runtime_config = {
    "thread_num": 4,
    "local_nodes": [{
        "biz_name": "master_biz",
        "ip": "127.0.0.1",
        "grpc_port": master_port,
    }],
}
with NaviSchedClient(
        "simulate_package_test_client_biz",
        "master_biz",
        name="simulate_package_test_client",
        runtime_config=runtime_config,
        enable_agent_forward=True,
        enable_kmon=False,
        enable_snapshot=False,
        wait_no_query=False) as client:
    client.init()
    deadline = time.monotonic() + 30
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        request = NaviSchedRequest(
            request_id=123456789 + attempt,
            prompt_token_count=32,
            block_hashes=[])
        begin = time.monotonic()
        try:
            decision = client.schedule(
                request,
                "agent_biz",
                timeout_ms=500)
        except RuntimeError:
            time.sleep(0.05)
            continue
        if decision.target:
            if decision.target.address != f"127.0.0.1:{agent_port}":
                raise RuntimeError("simulate package selected an unexpected Agent")
            if decision.request_timing is None:
                raise RuntimeError("simulate package did not return request timing")
            if decision.request_timing.execution_latency_ms <= 0:
                raise RuntimeError("simulate package returned invalid execution latency")
            if decision.request_timing.client_total_latency_ms <= 0:
                raise RuntimeError("simulate package returned invalid total latency")
            if time.monotonic() - begin < 0.008:
                raise RuntimeError("simulate package did not wait for Agent execution")
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("simulate package Agent was not scheduled")
PY
)

# Replay requests are one-shot, so launch them only after the full path is ready.
(
    cd "${PACKAGE_DIR}/runtime/client"
    exec env PATH=/usr/bin:/bin LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" \
        "${PACKAGE_DIR}/bin/navi_sched_simulate_client" \
        --biz client_biz \
        --master-biz master_biz \
        --node "master_biz,127.0.0.1,${MASTER_GRPC_PORT}" \
        --traffic-log "${PACKAGE_DIR}/runtime/client/traffic_replay.jsonl" \
        --traffic-biz agent_biz \
        --traffic-service-name "${TRAFFIC_SERVICE_NAME}" \
        --traffic-client-count 2 \
        --traffic-client-index 0 \
        --traffic-collect-replay-sequence 0 \
        --traffic-timeout-ms 30000 \
        --time-compression 2 \
        --runtime-dir "${PACKAGE_DIR}/runtime/client"
) >"${TEST_DIR}/client.log" 2>&1 &
CLIENT_1_PID=$!
(
    cd "${PACKAGE_DIR}/runtime/client"
    exec env PATH=/usr/bin:/bin LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" \
        "${PACKAGE_DIR}/bin/navi_sched_simulate_client" \
        --biz client_biz \
        --master-biz master_biz \
        --node "master_biz,127.0.0.1,${MASTER_GRPC_PORT}" \
        --traffic-log "${PACKAGE_DIR}/runtime/client/traffic_replay.jsonl" \
        --traffic-biz agent_biz \
        --traffic-service-name "${TRAFFIC_SERVICE_NAME}" \
        --traffic-client-count 2 \
        --traffic-client-index 1 \
        --traffic-timeout-ms 30000 \
        --time-compression 2 \
        --runtime-dir "${PACKAGE_DIR}/runtime/client_1"
) >"${TEST_DIR}/client_1.log" 2>&1 &
CLIENT_2_PID=$!

grep -q 'navi_sched.agent_simulate_execution_k' \
    "${PACKAGE_DIR}/runtime/agent/navi.sched_agent.snapshot.vis"
grep -q 'navi_sched.master_update_k' \
    "${PACKAGE_DIR}/runtime/master/navi.master_snapshot.snapshot.vis"
for _ in $(seq 1 800); do
    if grep -q "traffic replay completed, requestCount \\[${CLIENT_0_REQUEST_COUNT}\\]" \
        "${PACKAGE_DIR}/runtime/client/navi_sched.log" 2>/dev/null &&
        grep -q "traffic replay completed, requestCount \\[${CLIENT_1_REQUEST_COUNT}\\]" \
            "${PACKAGE_DIR}/runtime/client_1/navi_sched.log" 2>/dev/null; then
        break
    fi
    sleep 0.05
done
grep -q "traffic replay completed, requestCount \\[${CLIENT_0_REQUEST_COUNT}\\]" \
    "${PACKAGE_DIR}/runtime/client/navi_sched.log"
grep -q "traffic replay completed, requestCount \\[${CLIENT_1_REQUEST_COUNT}\\]" \
    "${PACKAGE_DIR}/runtime/client_1/navi_sched.log"
test -s "${PACKAGE_DIR}/runtime/client/traffic_schedule_replay.jsonl"
test -s "${PACKAGE_DIR}/runtime/client/traffic_request_latency.jsonl"
test -s "${PACKAGE_DIR}/runtime/client_1/traffic_request_latency.jsonl"
test "$(wc -l <"${PACKAGE_DIR}/runtime/client/traffic_request_latency.jsonl")" \
    -eq "${CLIENT_0_REQUEST_COUNT}"
test "$(wc -l <"${PACKAGE_DIR}/runtime/client_1/traffic_request_latency.jsonl")" \
    -eq "${CLIENT_1_REQUEST_COUNT}"
LATEST_MASTER_SNAPSHOT=$(find \
    "${PACKAGE_DIR}/runtime/master/navi_sched/master_snapshot" \
    -type f -name '*_snapshot' | sort | tail -1)
test -n "${LATEST_MASTER_SNAPSHOT}"
grep -q 'requests: inflight=0, waiting_for_batch=0, queued_for_execution=0, running=0' \
    "${LATEST_MASTER_SNAPSHOT}"
grep -q 'abnormal_batch_finish_count: 0' "${LATEST_MASTER_SNAPSHOT}"
grep -q 'queue_predicted_ms: 0.000' "${LATEST_MASTER_SNAPSHOT}"
grep -Eq 'pending_cache: .*snapshot_ready=1, revision_gaps=0, state_mismatches=0, resync_triggers=0' \
    "${LATEST_MASTER_SNAPSHOT}"
if grep -R -q 'register gig arpc service failed\|register http proto jsonizer failed' \
    "${PACKAGE_DIR}/runtime/agent" \
    "${PACKAGE_DIR}/runtime/master" \
    "${PACKAGE_DIR}/runtime/client" \
    "${PACKAGE_DIR}/runtime/client_1"; then
    printf 'Navi RPC service registration failed\n' >&2
    exit 1
fi
if grep -q 'request pending cache full snapshot' \
    "${PACKAGE_DIR}/runtime/master/navi_sched.log" \
    "${PACKAGE_DIR}/runtime/master/logs/navi_sched_alog.log"; then
    printf 'normal replay unexpectedly requested a pending cache full snapshot\n' >&2
    exit 1
fi
"${PYTHON_BIN}" - \
    "${PACKAGE_DIR}/runtime/client/traffic_request_latency.jsonl" \
    "${PACKAGE_DIR}/runtime/client_1/traffic_request_latency.jsonl" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path) as source:
        for line in source:
            timing = json.loads(line)["request_timing"]
            if timing["client_total_latency_ms"] <= 0:
                raise RuntimeError(f"invalid client total latency in {path}")
PY
stop_roles

"${PYTHON_BIN}" - \
    "${AGENT_GRPC_PORT}" \
    "${MASTER_GRPC_PORT}" <<'PY'
import socket
import sys
import time

ports = list(map(int, sys.argv[1:]))
deadline = time.monotonic() + 10
while True:
    listening = []
    for port in ports:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2)
        except OSError:
            continue
        listening.append(port)
    if not listening:
        break
    if time.monotonic() >= deadline:
        raise RuntimeError(
            f"simulate package roles still listen on {listening}")
    time.sleep(0.1)
PY

printf 'simulate package test passed\n'
